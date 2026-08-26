"""بند 7.15 سند فاز ۷ — خانواده‌ی ۶: کرنل و بردار پشتیبان، شاخه‌ی **فرایند گاوسی (GP)**.

⚠️ **دامنه‌ی این ماژول**: از ۱۰ عضو بند 7.15.1 فقط شاخه‌ی GP (اعضای ۶، ۷ و جدول
ترکیب کرنل 7.15.3) پیاده شده — طبق فهرست کوتاه اسپرینت C («خ۶ کرنل: GP روی L3»).
شاخه‌ی SVR/SVQR/KRR روی CPU جای خودش را دارد و این‌جا نیست: آن‌ها از GPU سودی
نمی‌برند (sklearn روی CPU اجرا می‌شود)، ولی GP دقیقاً همان‌جایی است که GPU برنده
است — تجزیه‌ی چولسکی/CG روی ماتریس ۷۵۷۹×۷۵۷۹.

## چرا L1 و نه فقط L3

جدول 7.15.4 می‌گوید GP روی L1 (۷٬۵۷۹) «در مرز» است و ~۱۰ دقیقه روی CPU می‌گیرد.
روی GPU همان کار ثانیه‌ای است — پس دلیلِ محدودکردن به L3 (۲۵۶ نقطه) از بین می‌رود
و می‌شود GP را روی همان سطحی اجرا کرد که همه‌ی خانواده‌های دیگر با آن سنجیده شدند
(بند 7.1.2). این خودش یکی از دلایل فرستادن این خانواده به GPU است.

## دو خروجی اجباری بند 7.15.6

1. **جدول مقایسه‌ی ۷ ترکیب کرنل با درست‌نمایی حاشیه‌ای لگاریتمی** — `KERNEL_COMBOS`
   دقیقاً همان K1…K7 جدول 7.15.3 است و `log_marginal_likelihood` هر برازش ثبت می‌شود.
2. **نمودار/جدول طول‌مقیاس ARD هر فیچر** — `ard_lengthscales()`؛ خروجی تفسیری این
   خانواده: طول‌مقیاس کوچک یعنی فیچر مهم. این خودش یک روش انتخاب فیچر است (بند 7.5.4).

## کوانتایل از پسین (مسیر Q4)

GP پسین گاوسی می‌دهد: $\\rho \\mid x \\sim \\mathcal{N}(m(x), s^2(x) + \\sigma_n^2)$.
پس $\\hat\\rho_\\tau = m(x) + z_\\tau\\sqrt{s^2+\\sigma_n^2}$ — **بدون** نیاز به آفست
تجربی باقیمانده. عضو K7 (`gp_heteroscedastic`) یک قدم جلوتر می‌رود و
$\\sigma_n^2$ را تابع $\\log Res$ می‌کند — پاسخ مستقیم به F06 (ناهم‌واریانسی
تأییدشده: نسبت std چارک کوچک به بزرگ ۳.۰۳).
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

FAMILY = "F06"
LEVEL = "L1"
FEATURE_SET = "FS_F06_gp_v1"

#: بند 7.15.5 «کاهش بعد پیش از کرنل» — GP با ۱۵۴ ستون یک‌هاتِ فیچرست کامل نه
#: می‌آموزد نه تفسیرپذیر می‌ماند (ARD روی ۱۵۴ طول‌مقیاس عملاً بهینه‌سازی‌ناپذیر است).
GP_NUMERIC = [
    "log_res", "res_vs_history", "res_vs_dow_history", "day_shock_lag1", "day_shock_roll_mean_7",
    "cell_expanding_rate", "cell_shrunk_rate", "cell_dow_shrunk_rate",
    "rho_roll_mean_7", "rho_roll_std_7", "rho_cell_lag1", "rho_cell_lag7",
    "food_shrunk_rate", "competitor_food_rate", "temp_min", "week_of_semester",
    "days_to_next_holiday", "is_holiday_any", "is_day_before_holiday", "is_exam_period",
    "is_ramadan", "is_lunch", "is_tehran",
]
GP_ONEHOT = ["RestaurantName"]

#: جدول 7.15.3 — هفت ترکیب کرنل، دقیقاً با همان نام‌ها
KERNEL_COMBOS = ("K1_rbf", "K2_periodic", "K3_rbf_x_periodic", "K4_matern",
                 "K5_full", "K6_full_plus_linear", "K7_heteroscedastic")


@lru_cache(maxsize=1)
def _numeric_cols() -> list[str]:
    return list(GP_NUMERIC)


class GPPrep:
    """استانداردسازی مقاوم + یک‌هات سلف + یک محور زمانی خام برای کرنل دوره‌ای.

    ⭐ `t_index` (شماره‌ی روز از ابتدای داده) عمداً **استاندارد نمی‌شود**: کرنل
    ExpSineSquared با `period_length=7` فقط وقتی معنا دارد که واحد محور زمان «روز»
    باشد. استانداردکردنش، دوره‌ی ۷ روزه را به عددی بی‌معنا تبدیل می‌کند — و همین
    ریزه‌کاری تفاوت K2/K3/K5 با یک RBF ساده است (F18: روزهفته p=۳.۶e−۸۸).
    """

    def __init__(self):
        self.num_cols: list[str] = []
        self.medians: dict = {}
        self.center: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.rest_levels: list[str] = []
        self.t0: pd.Timestamp | None = None
        self.y_mean: float = 0.0
        self.y_std: float = 1.0

    def fit(self, train: pd.DataFrame) -> "GPPrep":
        self.num_cols = [c for c in _numeric_cols() if c in train.columns]
        X = train[self.num_cols].astype(float)
        self.medians = X.median().to_dict()
        X = X.fillna(pd.Series(self.medians))
        q25, q75 = X.quantile(0.25).to_numpy(), X.quantile(0.75).to_numpy()
        self.center = X.median().to_numpy()
        self.scale = np.where((q75 - q25) > 1e-9, q75 - q25, 1.0)
        self.rest_levels = sorted(map(str, pd.unique(train[GP_ONEHOT[0]])))
        self.t0 = pd.Timestamp(pd.to_datetime(train[DATE_COL]).min())
        y = train["rho"].to_numpy(dtype=float)
        self.y_mean, self.y_std = float(y.mean()), float(max(y.std(), 1e-6))
        return self

    @property
    def t_index_col(self) -> int:
        """اندیس ستون محور زمان در ماتریس طراحی — کرنل دوره‌ای فقط روی همین می‌نشیند."""
        return len(self.num_cols) + len(self.rest_levels)

    @property
    def log_res_col(self) -> int:
        return self.num_cols.index("log_res") if "log_res" in self.num_cols else 0

    @property
    def feature_names(self) -> list[str]:
        return self.num_cols + [f"{GP_ONEHOT[0]}={v}" for v in self.rest_levels] + ["t_index"]

    def design(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.num_cols].astype(float).fillna(pd.Series(self.medians))
        num = (X.to_numpy() - self.center) / self.scale
        oh = np.zeros((len(df), len(self.rest_levels)), dtype=float)
        index = {v: i for i, v in enumerate(self.rest_levels)}
        for r, v in enumerate(df[GP_ONEHOT[0]].astype(str)):
            j = index.get(v)
            if j is not None:
                oh[r, j] = 1.0
        t = (pd.to_datetime(df[DATE_COL]) - self.t0).dt.days.to_numpy(dtype=float).reshape(-1, 1)
        return np.concatenate([num, oh, t], axis=1).astype(np.float64)

    def standardize_y(self, y: np.ndarray) -> np.ndarray:
        return (y - self.y_mean) / self.y_std

    def destandardize(self, mean: np.ndarray, sd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return mean * self.y_std + self.y_mean, sd * self.y_std


def build_kernel(combo: str, n_dims: int, t_col: int):
    """ترکیب کرنل طبق جدول 7.15.3."""
    import gpytorch
    from gpytorch.kernels import (AdditiveKernel, LinearKernel, MaternKernel, PeriodicKernel,
                                  RBFKernel, ScaleKernel)

    other = [d for d in range(n_dims) if d != t_col]
    rbf = ScaleKernel(RBFKernel(ard_num_dims=len(other), active_dims=other))
    matern = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=len(other), active_dims=other))
    periodic = ScaleKernel(PeriodicKernel(active_dims=[t_col]))
    periodic.base_kernel.period_length = 7.0     # ⭐ فصلی هفتگی صریح — F18

    if combo == "K1_rbf":
        return rbf
    if combo == "K2_periodic":
        return periodic
    if combo == "K3_rbf_x_periodic":
        return rbf * periodic
    if combo == "K4_matern":
        return matern
    if combo in ("K5_full", "K7_heteroscedastic"):
        return AdditiveKernel(rbf * periodic, matern)
    if combo == "K6_full_plus_linear":
        return AdditiveKernel(rbf * periodic, matern, ScaleKernel(LinearKernel()))
    raise ValueError(f"ترکیب کرنل ناشناخته: {combo!r} (مجاز: {KERNEL_COMBOS})")


def _build_gp_cls():
    import gpytorch

    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, x, y, likelihood, kernel):
            super().__init__(x, y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = kernel

        def forward(self, x):
            return gpytorch.distributions.MultivariateNormal(self.mean_module(x),
                                                             self.covar_module(x))

    return ExactGPModel


class GPModel:
    def __init__(self, state: dict, prep: GPPrep, config: dict, train_x: np.ndarray,
                 train_y: np.ndarray, lml: float, noise_model: dict | None = None):
        self.state = state
        self.prep = prep
        self.config = config
        self.train_x = train_x        # GP بدون داده‌ی آموزش قابل پیش‌بینی نیست — بخشی از مدل است
        self.train_y = train_y
        self.lml = lml
        self.noise_model = noise_model

    # -- بازسازی شیء gpytorch از حالت ذخیره‌شده ---------------------------------
    def _materialize(self):
        import gpytorch
        import torch

        from src.models.gpu_runner import torch_device

        device = torch_device()
        ExactGPModel = _build_gp_cls()
        x = torch.as_tensor(self.train_x, dtype=torch.float64, device=device)
        y = torch.as_tensor(self.train_y, dtype=torch.float64, device=device)
        kernel = build_kernel(self.config["kernel"], x.shape[1], self.prep.t_index_col)
        if self.noise_model is not None:
            noise = torch.as_tensor(self._noise_vector(self.train_x), dtype=torch.float64, device=device)
            likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
                noise=noise, learn_additional_noise=True)
        else:
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(x, y, likelihood, kernel).to(device).double()
        model.load_state_dict(self.state)
        return model, likelihood, device

    def _noise_vector(self, X: np.ndarray) -> np.ndarray:
        """K7 — واریانس نویز به‌عنوان تابع نمایی $\\log Res$ (پاسخ به F06)."""
        a, b = self.noise_model["intercept"], self.noise_model["slope"]
        return np.clip(np.exp(a + b * X[:, self.prep.log_res_col]), 1e-8, 10.0)

    def posterior(self, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        import gpytorch
        import torch

        model, likelihood, device = self._materialize()
        model.eval()
        likelihood.eval()
        Xte = self.prep.design(test)
        xt = torch.as_tensor(Xte, dtype=torch.float64, device=device)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            post = model(xt)
            mean = post.mean.cpu().numpy()
            var = post.variance.cpu().numpy()
        noise = (self._noise_vector(Xte) if self.noise_model is not None
                 else float(likelihood.noise.detach().cpu().numpy().ravel()[0]))
        return self.prep.destandardize(mean, np.sqrt(np.clip(var + noise, 1e-12, None)))

    def predict(self, test: pd.DataFrame, tau: float) -> np.ndarray:
        mean, sd = self.posterior(test)
        return np.clip(mean + stats.norm.ppf(tau) * sd, 0.0, 1.0)

    def ard_lengthscales(self) -> pd.DataFrame:
        """خروجی تفسیری اجباری بند 7.15.6 — طول‌مقیاس کوچک‌تر ⇒ فیچر مهم‌تر."""
        rows = []
        names = self.prep.feature_names
        other = [i for i in range(len(names)) if i != self.prep.t_index_col]
        for key, value in self.state.items():
            if "raw_lengthscale" not in key:
                continue
            ls = np.asarray(value).ravel()
            ls = np.log1p(np.exp(ls))          # معکوس softplus خامِ gpytorch
            if len(ls) == len(other):
                for pos, i in enumerate(other):
                    rows.append({"kernel_param": key, "feature": names[i], "lengthscale": float(ls[pos])})
        return (pd.DataFrame(rows).sort_values("lengthscale").reset_index(drop=True)
                if rows else pd.DataFrame(columns=["kernel_param", "feature", "lengthscale"]))


# ---------------------------------------------------------------------------
# برازش
# ---------------------------------------------------------------------------

def _train_exact_gp(X: np.ndarray, y_std: np.ndarray, kernel_combo: str, t_col: int,
                    n_iters: int, learning_rate: float, noise_vector: np.ndarray | None,
                    seed: int):
    import gpytorch
    import torch

    from src.models.gpu_runner import torch_device

    device = torch_device()
    torch.manual_seed(seed)
    ExactGPModel = _build_gp_cls()
    x = torch.as_tensor(X, dtype=torch.float64, device=device)
    y = torch.as_tensor(y_std, dtype=torch.float64, device=device)

    if noise_vector is not None:
        likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
            noise=torch.as_tensor(noise_vector, dtype=torch.float64, device=device),
            learn_additional_noise=True)
    else:
        likelihood = gpytorch.likelihoods.GaussianLikelihood()

    kernel = build_kernel(kernel_combo, x.shape[1], t_col)
    model = ExactGPModel(x, y, likelihood, kernel).to(device).double()
    model.train()
    likelihood.train()
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

    lml = float("nan")
    for _ in range(n_iters):
        opt.zero_grad(set_to_none=True)
        loss = -mll(model(x), y)
        loss.backward()
        opt.step()
        lml = -float(loss.item())
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    return state, lml


def fit_gp(train: pd.DataFrame, tau: float, *, seed: int = 42, kernel: str = "K5_full",
           n_iters: int = 150, learning_rate: float = 0.05, heteroscedastic: bool = False,
           **hp) -> GPModel:
    prep = GPPrep().fit(train)
    X = prep.design(train)
    y = prep.standardize_y(train["rho"].to_numpy(dtype=float))

    state, lml = _train_exact_gp(X, y, kernel, prep.t_index_col, n_iters, learning_rate, None, seed)
    noise_model = None

    if heteroscedastic:
        # مرحله‌ی ۲ (K7): باقیمانده‌ی مرحله‌ی ۱ ⇒ رگرسیون log(residual²) روی log_res
        # ⇒ نویز نقطه‌به‌نقطه ⇒ بازبرازش با FixedNoiseGaussianLikelihood.
        tmp = GPModel(state, prep, {"kernel": kernel}, X, y, lml)
        mean, _ = tmp.posterior(train)
        resid = train["rho"].to_numpy(dtype=float) - mean
        log_r2 = np.log(np.clip(resid ** 2, 1e-10, None)) - 2 * np.log(prep.y_std)
        z = X[:, prep.log_res_col]
        slope, intercept = np.polyfit(z, log_r2, 1)
        noise_model = {"intercept": float(intercept), "slope": float(slope)}
        noise_vec = np.clip(np.exp(intercept + slope * z), 1e-8, 10.0)
        state, lml = _train_exact_gp(X, y, kernel, prep.t_index_col, n_iters, learning_rate,
                                     noise_vec, seed)

    config = {"kernel": kernel, "n_iters": int(n_iters), "learning_rate": float(learning_rate),
              "heteroscedastic": bool(heteroscedastic), "n_train": int(len(train)),
              "n_dims": int(X.shape[1])}
    return GPModel(state, prep, config, X, y, lml, noise_model)


def _predict(model: GPModel, test: pd.DataFrame, tau: float) -> np.ndarray:
    return model.predict(test, tau)


def _save(model: GPModel, stem) -> list:
    """⚠️ GP **پارامتری نیست**: بدون داده‌ی آموزش، هایپرپارامترها بی‌فایده‌اند. پس
    ماتریس آموزش هم در همان فایل ذخیره می‌شود (نه فقط `state_dict`)."""
    import pickle

    import torch

    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    path = Path(f"{stem}.pt")
    torch.save({"state": model.state, "prep": pickle.dumps(model.prep), "config": model.config,
                "train_x": model.train_x, "train_y": model.train_y, "lml": model.lml,
                "noise_model": model.noise_model}, path)
    meta = Path(f"{stem}.json")
    meta.write_text(json.dumps({"config": model.config, "log_marginal_likelihood": model.lml,
                                "noise_model": model.noise_model}, ensure_ascii=False, indent=2) + "\n")
    ard = Path(f"{stem}__ard.csv")
    model.ard_lengthscales().to_csv(ard, index=False)
    return [path, meta, ard]


def _load(stem) -> GPModel:
    import pickle

    import torch

    blob = torch.load(Path(f"{stem}.pt"), map_location="cpu", weights_only=False)
    return GPModel(blob["state"], pickle.loads(blob["prep"]), blob["config"],
                   blob["train_x"], blob["train_y"], blob["lml"], blob["noise_model"])


FITTERS: dict[str, FamilyFitter] = {
    "gp_quantile": FamilyFitter(
        "gp_quantile", fit=fit_gp, predict=_predict, save=_save, load=_load,
        defaults={"kernel": "K5_full", "n_iters": 150, "learning_rate": 0.05}),
    "gp_heteroscedastic": FamilyFitter(
        "gp_heteroscedastic",
        fit=lambda tr, tau, **hp: fit_gp(tr, tau, **{**hp, "heteroscedastic": True}),
        predict=_predict, save=_save, load=_load,
        defaults={"kernel": "K5_full", "n_iters": 150, "learning_rate": 0.05}),
}

MODELS = models_from_fitters(FITTERS)

register(ModelSpec(model_id="gp_quantile", family=FAMILY, levels=(LEVEL,), quantile_route="Q4",
                   algorithm="gpytorch ExactGP (ARD) + کوانتایل از پسین گاوسی"))
register(ModelSpec(model_id="gp_heteroscedastic", family=FAMILY, levels=(LEVEL,),
                   quantile_route="Q4",
                   algorithm="gpytorch ExactGP + FixedNoiseGaussianLikelihood(σ²=f(log Res))"))


@register_space("gp_quantile", version=1, n_hyperparams=3)
def _space_gp(trial: optuna.Trial) -> dict:
    return {"kernel": trial.suggest_categorical("kernel", list(KERNEL_COMBOS[:-1])),
            "n_iters": trial.suggest_int("n_iters", 60, 400, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True)}


@register_space("gp_heteroscedastic", version=1, n_hyperparams=2)
def _space_gp_het(trial: optuna.Trial) -> dict:
    return {"n_iters": trial.suggest_int("n_iters", 60, 400, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True)}


QUANTREG_MODEL_IDS: frozenset[str] = frozenset()
TUNING_EXCLUDED: frozenset[str] = frozenset()
