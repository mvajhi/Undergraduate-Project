"""بند 7.17 سند فاز ۷ — خانواده‌ی ۸: سلسله‌مراتبی و بیزی (NumPyro/NUTS روی GPU).

## چرا این خانواده منطبق‌ترین با ساختار مسئله است

سه یافته‌ی فاز ۴ مستقیماً یک مدل سلسله‌مراتبی را توصیف می‌کنند:

- **F10:** ICC(روز)=۰.۲۲۵ و ICC(سلف)=۰.۲۰۵ — دو منبع همبستگی **متقاطع**، نه تودرتو.
- **F59:** ۸۳.۲٪ واریانس سلول شوک مشترک روزانه است ⇒ اثر تصادفی روز با واریانس بزرگ.
- **F07:** بیش‌پراکندگی صعودی (۳.۷× → ۱۵.۶×) ⇒ دوجمله‌ای خالص کافی نیست، Beta-Binomial لازم است.

و مهم‌تر: خروجی بیزی یک **توزیع پسین کامل** است، پس کوانتایل مستقیم از آن درمی‌آید
(مسیر Q4، بند 7.23) — بدون فرض نرمال‌بودن، بدون بوت‌استرپ، و **با احتساب عدم‌قطعیت
خودِ اثر روزِ ندیده**. این آخری نکته‌ی کلیدی است: روز آزمون یک روز جدید است، پس
$u_{\\text{day}}$ آن **معلوم نیست** و باید از پیشین پسین‌آموخته ($\\sigma_{day}$)
قرعه بخورد. مدل‌های نقطه‌ای این عدم‌قطعیت را اصلاً نمی‌بینند — به همین دلیل بند 7.17
انتظار دارد این خانواده حتی اگر pinball را نبرد، **کالیبراسیون** بهتری بدهد.

## سه ساختار پیاده‌شده (بند 7.17.1)

| model_id | عضو WBS | چه چیزی را می‌آزماید |
|---|---|---|
| `bhm_beta_binomial_restaurant` | ۱/۵ | فقط اثر تصادفی سلف — مبنای مقایسه |
| `bhm_beta_binomial_crossed` | ⭐ ۲/۵ | اثر متقاطع **روز + سلف** — منطبق‌ترین با F10 |
| `bhm_varying_dispersion` | ⭐ ۶ | $\\phi$ خودش تابع $\\log Res$ — پاسخ مستقیم به F06/F07 |

## قواعد اجباری این خانواده

- **پارامترسازی غیرمرکزی** (`z ~ N(0,1); u = σ·z`) — بند 7.17.2: برای اثر تصادفی با
  واریانس کوچک تقریباً همیشه لازم است، وگرنه NUTS در قیف نیل واگرا می‌شود.
- **چک‌لیست همگرایی بند 7.17.3 جای قاعده‌ی A6 را می‌گیرد** — `convergence_report`.
- **PPC اجباری**: توزیع شبیه‌سازی‌شده باید چولگی ~۴.۰۶ و تورم صفر ~۴.۹٪ را بازتولید
  کند (F02/F03). «مدلی که میانگین را درست می‌زند ولی چولگی را بازتولید نمی‌کند،
  کوانتایل‌هایش غلط است» — بند 7.17.3.
- **تحلیل حساسیت پیشین** اجباری (`PRIOR_PRESETS`).
"""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from scipy import stats

from src.cv import DATE_COL
from src.models.gpu_runner import FamilyFitter, models_from_fitters
from src.models.registry import ModelSpec, register
from src.models.spaces import register_space

FAMILY = "F08"
LEVEL = "L1"
FEATURE_SET = "FS_F08_bayes_v1"

#: فیچرست عمداً **کوچک و استانداردشده**: NUTS با ۱۵۴ ستون یک‌هات (فیچرست کامل خ۱)
#: هم کند می‌شود و هم به‌خاطر هم‌خطی (یافته‌ی ۹) قیف پسین بدشکل می‌سازد. اثر سلف و
#: روز از راه **اثر تصادفی** وارد می‌شود، نه از راه دامی — که کل ایده‌ی این خانواده است.
BAYES_FEATURES = [
    "log_res", "res_vs_history", "day_shock_lag1", "day_shock_roll_mean_7",
    "cell_expanding_rate", "cell_shrunk_rate", "cell_dow_shrunk_rate",
    "rho_roll_mean_7", "rho_cell_lag7", "food_shrunk_rate", "competitor_food_rate",
    "is_holiday_any", "is_day_before_holiday", "is_exam_period", "is_ramadan",
    "is_lunch", "dow_sin1", "dow_cos1", "dow_sin2", "dow_cos2",
]

#: بند 7.17.2 ردیف «پیشین» — تحلیل حساسیت با این سه مجموعه اجباری است
PRIOR_PRESETS: dict[str, dict] = {
    "tight":   {"prior_sigma": 0.5, "prior_beta": 1.0},
    "default": {"prior_sigma": 1.0, "prior_beta": 2.5},
    "wide":    {"prior_sigma": 2.0, "prior_beta": 5.0},
}

#: F02/F03 — هدف PPC: مدل باید این دو عدد را بازتولید کند
PPC_TARGET_SKEW = 4.06
PPC_TARGET_ZERO_SHARE = 0.049


@lru_cache(maxsize=1)
def _features() -> list[str]:
    return list(BAYES_FEATURES)


class BayesPrep:
    """استانداردسازی عددی + واژگان سلف/روز. فقط از train برازش می‌شود."""

    def __init__(self):
        self.cols: list[str] = []
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.rest_vocab: dict[str, int] = {}
        self.medians: dict = {}

    def fit(self, train: pd.DataFrame) -> "BayesPrep":
        self.cols = [c for c in _features() if c in train.columns]
        X = train[self.cols].astype(float)
        self.medians = X.median().to_dict()
        X = X.fillna(pd.Series(self.medians))
        self.mean = X.mean().to_numpy()
        self.std = np.where(X.std().to_numpy() > 1e-9, X.std().to_numpy(), 1.0)
        self.rest_vocab = {r: i for i, r in enumerate(sorted(map(str, pd.unique(train["RestaurantName"]))))}
        return self

    def design(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.cols].astype(float).fillna(pd.Series(self.medians))
        return ((X.to_numpy() - self.mean) / self.std).astype(np.float32)

    def restaurant_index(self, df: pd.DataFrame) -> np.ndarray:
        """سلف دیده‌نشده ⇒ ``-1`` که پایین‌دست یعنی «از پیشین قرعه بخور»، نه «صفر بگیر»."""
        return df["RestaurantName"].astype(str).map(
            lambda r: self.rest_vocab.get(r, -1)).to_numpy(dtype=np.int64)

    @property
    def log_res_col(self) -> int:
        return self.cols.index("log_res") if "log_res" in self.cols else 0


def _numpyro_model():
    import jax
    import jax.numpy as jnp
    import numpyro
    from numpyro import distributions as dist

    def model(X, rest_idx, day_idx, n_rest, n_days, total, y=None, *,
              use_day: bool = True, varying_dispersion: bool = False,
              prior_sigma: float = 1.0, prior_beta: float = 2.5, log_res_col: int = 0):
        alpha = numpyro.sample("alpha", dist.Normal(-2.0, 1.5))
        beta = numpyro.sample("beta", dist.Normal(0.0, prior_beta).expand([X.shape[1]]).to_event(1))

        # پارامترسازی غیرمرکزی — بند 7.17.2
        sigma_rest = numpyro.sample("sigma_rest", dist.HalfNormal(prior_sigma))
        z_rest = numpyro.sample("z_rest", dist.Normal(0.0, 1.0).expand([n_rest]).to_event(1))
        eta = alpha + X @ beta + sigma_rest * z_rest[rest_idx]

        if use_day:
            sigma_day = numpyro.sample("sigma_day", dist.HalfNormal(prior_sigma))
            z_day = numpyro.sample("z_day", dist.Normal(0.0, 1.0).expand([n_days]).to_event(1))
            eta = eta + sigma_day * z_day[day_idx]

        mu = jax.nn.sigmoid(eta)
        if varying_dispersion:
            # φ = exp(a + b·log_res) — بند 7.17.1 عضو ۶، پاسخ مستقیم به F06/F07
            phi_a = numpyro.sample("log_phi_intercept", dist.Normal(2.5, 1.0))
            phi_b = numpyro.sample("log_phi_slope", dist.Normal(0.0, 0.5))
            phi = jnp.exp(phi_a + phi_b * X[:, log_res_col])
        else:
            phi = numpyro.sample("phi", dist.Gamma(2.0, 0.1))
        numpyro.sample("obs", dist.BetaBinomial(mu * phi, (1.0 - mu) * phi, total_count=total), obs=y)

    return model


class BayesModel:
    def __init__(self, samples: dict, prep: BayesPrep, config: dict, diagnostics: dict):
        self.samples = samples
        self.prep = prep
        self.config = config
        self.diagnostics = diagnostics

    def posterior_rho_samples(self, test: pd.DataFrame, seed: int = 0,
                              max_draws: int = 600) -> np.ndarray:
        """ماتریس (draw × ردیف) از نرخ شبیه‌سازی‌شده — پایه‌ی هر کوانتایل/پوشش/PPC.

        ⭐ **اثر روزِ آزمون از پیشین قرعه می‌خورد، نه از پسین.** روز آزمون در آموزش
        نبوده؛ استفاده از میانگین صفر یعنی وانمود کردن که «شوک روز آینده صفر است» و
        دقیقاً همان چیزی که ۸۳٪ واریانس را می‌سازد (F59) از عدم‌قطعیت حذف می‌شود.
        همین یک خط، تفاوت کالیبراسیون بیزی با مدل نقطه‌ای است.
        """
        rng = np.random.default_rng(seed)
        X = self.prep.design(test)
        rest_idx = self.prep.restaurant_index(test)
        total = np.maximum(test["Res"].to_numpy(dtype=float), 1.0)

        alpha = np.asarray(self.samples["alpha"])
        beta = np.asarray(self.samples["beta"])
        n_draws = min(len(alpha), max_draws)
        sel = rng.choice(len(alpha), size=n_draws, replace=False) if len(alpha) > n_draws else np.arange(len(alpha))

        sigma_rest = np.asarray(self.samples["sigma_rest"])[sel]
        z_rest = np.asarray(self.samples["z_rest"])[sel]
        eta = alpha[sel, None] + (beta[sel] @ X.T)

        u_rest = np.empty((n_draws, len(test)))
        seen = rest_idx >= 0
        if seen.any():
            u_rest[:, seen] = sigma_rest[:, None] * z_rest[:, rest_idx[seen]]
        if (~seen).any():   # سلف دیده‌نشده ⇒ قرعه از پیشین با همان σ
            u_rest[:, ~seen] = sigma_rest[:, None] * rng.standard_normal((n_draws, int((~seen).sum())))
        eta = eta + u_rest

        if "sigma_day" in self.samples:
            sigma_day = np.asarray(self.samples["sigma_day"])[sel]
            day_codes = pd.factorize(test[DATE_COL].to_numpy())[0]
            eps = rng.standard_normal((n_draws, day_codes.max() + 1))
            eta = eta + sigma_day[:, None] * eps[:, day_codes]

        mu = 1.0 / (1.0 + np.exp(-eta))
        if "log_phi_intercept" in self.samples:
            phi = np.exp(np.asarray(self.samples["log_phi_intercept"])[sel, None]
                         + np.asarray(self.samples["log_phi_slope"])[sel, None]
                         * X[:, self.prep.log_res_col][None, :])
        else:
            phi = np.asarray(self.samples["phi"])[sel, None]

        a = np.clip(mu * phi, 1e-6, 1e6)
        b = np.clip((1.0 - mu) * phi, 1e-6, 1e6)
        p = rng.beta(a, b)
        k = rng.binomial(np.broadcast_to(total.astype(np.int64), p.shape), p)
        return k / total[None, :]

    def predict(self, test: pd.DataFrame, tau: float, seed: int = 0) -> np.ndarray:
        return np.clip(np.quantile(self.posterior_rho_samples(test, seed=seed), tau, axis=0), 0.0, 1.0)


# ---------------------------------------------------------------------------
# تشخیص همگرایی (بند 7.17.3) — جایگزین قاعده‌ی A6
# ---------------------------------------------------------------------------

def convergence_report(mcmc) -> dict:
    """R̂ · ESS · تعداد divergence — همان چک‌لیست بند 7.17.3، به‌صورت عدد نه چشمی."""
    from numpyro.diagnostics import summary as _summary

    summary = _summary(mcmc.get_samples(group_by_chain=True), prob=0.9)
    r_hats, ess = [], []
    for stats_ in summary.values():
        r_hats.append(float(np.nanmax(np.atleast_1d(stats_["r_hat"]))))
        ess.append(float(np.nanmin(np.atleast_1d(stats_["n_eff"]))))
    extra = mcmc.get_extra_fields() if hasattr(mcmc, "get_extra_fields") else {}
    diverging = extra.get("diverging")
    n_div = int(np.asarray(diverging).sum()) if diverging is not None else -1
    n_total = int(np.asarray(diverging).size) if diverging is not None else 0
    return {
        "max_r_hat": max(r_hats) if r_hats else float("nan"),
        "min_ess": min(ess) if ess else float("nan"),
        "n_divergences": n_div,
        "divergence_rate": (n_div / n_total) if n_total else float("nan"),
        "passes_rhat": bool(r_hats and max(r_hats) < 1.01),
        "passes_ess": bool(ess and min(ess) > 400),
        "passes_divergence": bool(n_div == 0 or (n_total and n_div / n_total < 0.001)),
    }


def posterior_predictive_check(model: BayesModel, frame: pd.DataFrame, seed: int = 0) -> dict:
    """بند 7.17.3 بند ششم — مهم‌ترین بند چک‌لیست: چولگی و تورم صفر باید بازتولید شوند."""
    sim = model.posterior_rho_samples(frame, seed=seed)
    actual = frame["rho"].to_numpy(dtype=float)
    sim_flat = sim.reshape(-1)
    return {
        "sim_skew": float(stats.skew(sim_flat)),
        "actual_skew": float(stats.skew(actual)),
        "target_skew_F02": PPC_TARGET_SKEW,
        "sim_zero_share": float((sim_flat <= 1e-9).mean()),
        "actual_zero_share": float((actual <= 1e-9).mean()),
        "target_zero_share_F03": PPC_TARGET_ZERO_SHARE,
        "sim_mean": float(sim_flat.mean()),
        "actual_mean": float(actual.mean()),
    }


# ---------------------------------------------------------------------------
# برازش
# ---------------------------------------------------------------------------

def _fit_generic(train: pd.DataFrame, tau: float, *, use_day: bool, varying_dispersion: bool,
                 seed: int = 42, **hp) -> BayesModel:
    import jax
    import numpyro
    from numpyro.infer import MCMC, NUTS

    numpyro.set_host_device_count(int(hp.get("num_chains", 2)))
    prep = BayesPrep().fit(train)
    X = prep.design(train)
    rest_idx = np.clip(prep.restaurant_index(train), 0, None)
    day_idx = pd.factorize(train[DATE_COL].to_numpy())[0]
    total = np.maximum(train["Res"].to_numpy(dtype=np.int32), 1)
    y = np.clip(train["NoRecv"].to_numpy(dtype=np.int32), 0, total)

    prior = PRIOR_PRESETS[str(hp.get("prior_preset", "default"))]
    kernel = NUTS(_numpyro_model(), target_accept_prob=float(hp.get("target_accept", 0.9)),
                  max_tree_depth=int(hp.get("max_tree_depth", 10)))
    mcmc = MCMC(kernel, num_warmup=int(hp.get("num_warmup", 500)),
                num_samples=int(hp.get("num_samples", 500)),
                num_chains=int(hp.get("num_chains", 2)),
                chain_method="vectorized", progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), X, rest_idx, day_idx,
             len(prep.rest_vocab), int(day_idx.max()) + 1, total, y,
             use_day=use_day, varying_dispersion=varying_dispersion,
             prior_sigma=prior["prior_sigma"], prior_beta=prior["prior_beta"],
             log_res_col=prep.log_res_col, extra_fields=("diverging",))

    samples = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
    config = {"use_day": use_day, "varying_dispersion": varying_dispersion,
              "prior_preset": str(hp.get("prior_preset", "default")), **prior,
              "num_warmup": int(hp.get("num_warmup", 500)),
              "num_samples": int(hp.get("num_samples", 500)),
              "num_chains": int(hp.get("num_chains", 2)),
              "target_accept": float(hp.get("target_accept", 0.9)),
              "n_train_rows": int(len(train))}
    return BayesModel(samples, prep, config, convergence_report(mcmc))


def _predict(model: BayesModel, test: pd.DataFrame, tau: float) -> np.ndarray:
    return model.predict(test, tau)


def _save(model: BayesModel, stem) -> list:
    """نمونه‌های پسین در `.npz` + پیش‌پردازش در `.pkl` + تشخیص همگرایی در `.json`.

    نمونه‌های پسین **خودِ مدل‌اند** (مدل بیزی وزن ندارد، توزیع دارد) — با همین فایل
    می‌شود بدون هیچ نمونه‌گیری دوباره، برای هر داده‌ی جدید و هر τ پیش‌بینی گرفت.
    """
    import pickle

    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    npz = Path(f"{stem}.npz")
    np.savez_compressed(npz, **model.samples)
    prep_path = Path(f"{stem}__prep.pkl")
    prep_path.write_bytes(pickle.dumps(model.prep))
    meta = Path(f"{stem}.json")
    meta.write_text(json.dumps({"config": model.config, "diagnostics": model.diagnostics},
                               ensure_ascii=False, indent=2) + "\n")
    return [npz, prep_path, meta]


def _load(stem) -> BayesModel:
    import pickle

    stem = Path(stem)
    blob = np.load(Path(f"{stem}.npz"))
    prep = pickle.loads(Path(f"{stem}__prep.pkl").read_bytes())
    meta = json.loads(Path(f"{stem}.json").read_text())
    return BayesModel({k: blob[k] for k in blob.files}, prep, meta["config"], meta["diagnostics"])


FITTERS: dict[str, FamilyFitter] = {
    "bhm_beta_binomial_restaurant": FamilyFitter(
        "bhm_beta_binomial_restaurant",
        fit=lambda tr, tau, **hp: _fit_generic(tr, tau, use_day=False, varying_dispersion=False, **hp),
        predict=_predict, save=_save, load=_load,
        defaults={"num_warmup": 400, "num_samples": 400, "num_chains": 2}),
    "bhm_beta_binomial_crossed": FamilyFitter(
        "bhm_beta_binomial_crossed",
        fit=lambda tr, tau, **hp: _fit_generic(tr, tau, use_day=True, varying_dispersion=False, **hp),
        predict=_predict, save=_save, load=_load,
        defaults={"num_warmup": 500, "num_samples": 500, "num_chains": 2}),
    "bhm_varying_dispersion": FamilyFitter(
        "bhm_varying_dispersion",
        fit=lambda tr, tau, **hp: _fit_generic(tr, tau, use_day=True, varying_dispersion=True, **hp),
        predict=_predict, save=_save, load=_load,
        defaults={"num_warmup": 500, "num_samples": 500, "num_chains": 2}),
}

MODELS = models_from_fitters(FITTERS)

_ALGORITHMS = {
    "bhm_beta_binomial_restaurant": "numpyro NUTS · BetaBinomial + RE(سلف)",
    "bhm_beta_binomial_crossed": "numpyro NUTS · BetaBinomial + RE(روز×سلف متقاطع)",
    "bhm_varying_dispersion": "numpyro NUTS · BetaBinomial با φ=f(log Res) + RE متقاطع",
}
for _mid, _algo in _ALGORITHMS.items():
    register(ModelSpec(model_id=_mid, family=FAMILY, levels=(LEVEL,), quantile_route="Q4",
                       algorithm=_algo))


def _space_common(trial: optuna.Trial) -> dict:
    return {
        "prior_preset": trial.suggest_categorical("prior_preset", list(PRIOR_PRESETS)),
        "target_accept": trial.suggest_float("target_accept", 0.8, 0.99),
        "num_warmup": trial.suggest_categorical("num_warmup", [300, 500, 800]),
        "num_samples": trial.suggest_categorical("num_samples", [400, 600, 1000]),
    }


# ⚠️ کاردینالیتی صریح اعلام شده (بند 7.6.2 بازنویسی‌شده): فضای این خانواده عملاً
# گسسته و کوچک است (۳ پیشین × ۳ warmup × ۳ sample × یک پیوسته). بدون این اعلام،
# جدول بودجه ۶۰ trial می‌داد که با NUTS یعنی چند ده ساعت — دقیقاً اشتباه یافته‌ی ۱۲.
for _mid in _ALGORITHMS:
    register_space(_mid, version=1, n_hyperparams=4, cardinality=27)(_space_common)

QUANTREG_MODEL_IDS: frozenset[str] = frozenset()
TUNING_EXCLUDED: frozenset[str] = frozenset()
