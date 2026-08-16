"""بند 7.18 سند فاز ۷ — خانواده‌ی ۹: رگرسیون توزیعی و شمارشی (F09).

⚠️ **فقط یک نماینده**، طبق فهرست کوتاه اسپرینت C (`doc/decisions/37-phase7-rescope.md`
بند ۵؛ بودجه‌اش هم از دروازه‌ی منفی ARCH-LM خ۵ منتقل شد، ردیف ۳۸ decision_log).

سند اصلی «LightGBMLSS یا NGBoost» پیشنهاد داده بود؛ هر دو پکیج در این محیط قابل‌نصب
نبودند — `ngboost` به `lifelines` وابسته است که نصبش timeout شد، `lightgbmlss` به
`torch` (~۹۰۰ مگابایت) وابسته است که برای پروژه‌ی صرفاً-CPU (بند 7.28.1) نامتناسب
است. **جایگزین معادل**: دو LightGBM مستقل (یکی برای μ، یکی برای φ/دقت) که خروجی‌شان
پارامترهای توزیع Beta را می‌سازد — دقیقاً همان ایده‌ی GAMLSS/LSS (بند ۵ سند تصمیم ۳۷:
«تنها خانواده‌ای که واریانس را صریح مدل می‌کند»، پاسخ به F06/F07/F09) بدون وابستگی سنگین.

## چرا Beta

خ۱ (F01، بند 7.10.1) قبلاً Beta را رقیب جدی GLM Gamma دیده بود (F04: KS نزدیک).
`common.beta_quantile` هم از قبل آماده است (بند 7.5.1 خ۱).
"""

import numpy as np
import optuna
import pandas as pd
from scipy import stats

from src.models.families import common
from src.models.families.f01_linear import _design_s2
from src.models.registry import ModelSpec, register
from src.models.spaces import register_space

FAMILY = "F09"
LEVEL = "L1"

#: زیر این آستانه، φ برآوردی بی‌معناست (واریانس منفی/صفر از رگرسیون σ) — سقوط امن به φ سراسری
_MIN_PHI = 2.0


def fit_predict_lightgbm_lss_beta(train: pd.DataFrame, test: pd.DataFrame, tau: float,
                                  num_leaves: int = 15, learning_rate: float = 0.05,
                                  n_estimators: int = 200, **hp) -> np.ndarray:
    """گام ۱: LightGBM برای μ (میانگین شرطی ρ). گام ۲: LightGBM دوم برای log-واریانس
    باقیمانده‌ی گام ۱ (تابعی از همان فیچرها) — یعنی خودِ واریانس هم شرطی به کوواریت‌هاست،
    نه یک عدد سراسری (تفاوت اصلی با GLM Gamma/Beta خ۱ که φ سراسری داشتند). از μ̂ و
    φ̂=μ̂(1-μ̂)/Var̂-1 کوانتایل Beta گرفته می‌شود (مسیر Q2، `common.beta_quantile`)."""
    import lightgbm as lgb

    Xtr, Xte = _design_s2(train, test, quantreg=False)
    y = np.clip(train["rho"].to_numpy(), 1e-4, 1 - 1e-4)

    mu_model = lgb.LGBMRegressor(objective="regression", num_leaves=num_leaves,
                                 learning_rate=learning_rate, n_estimators=n_estimators,
                                 min_child_samples=10, verbosity=-1, random_state=42)
    mu_model.fit(Xtr, y)
    mu_tr = np.clip(mu_model.predict(Xtr), 1e-4, 1 - 1e-4)
    mu_te = np.clip(mu_model.predict(Xte), 1e-4, 1 - 1e-4)

    # گام ۲: مدل واریانس شرطی — هدف log(باقیمانده‌ی مربعی) روی همان فیچرها
    sq_resid = np.clip((y - mu_tr) ** 2, 1e-8, None)
    var_model = lgb.LGBMRegressor(objective="regression", num_leaves=max(3, num_leaves // 2),
                                  learning_rate=learning_rate, n_estimators=n_estimators,
                                  min_child_samples=20, verbosity=-1, random_state=43)
    var_model.fit(Xtr, np.log(sq_resid))
    var_te = np.clip(np.exp(var_model.predict(Xte)), 1e-8, None)

    # φ از رابطه‌ی واریانس Beta: Var = mu(1-mu)/(1+phi) ⇒ phi = mu(1-mu)/Var - 1
    phi_te = mu_te * (1 - mu_te) / var_te - 1.0
    # سقوط امن: φ نامعتبر (منفی/خیلی کوچک) ⇒ φ سراسری تجربی از باقیمانده‌ی train
    phi_global = float(np.clip(np.mean(mu_tr * (1 - mu_tr)) / np.mean(sq_resid) - 1.0, _MIN_PHI, None))
    phi_te = np.where(np.isfinite(phi_te) & (phi_te >= _MIN_PHI), phi_te, phi_global)

    a = mu_te * phi_te
    b = (1 - mu_te) * phi_te
    return np.clip(stats.beta.ppf(tau, a, b), 0.0, 1.0)


MODELS = {"lightgbm_lss_beta": fit_predict_lightgbm_lss_beta}

register(ModelSpec(model_id="lightgbm_lss_beta", family=FAMILY, levels=(LEVEL,), quantile_route="Q2",
                   algorithm="۲×lightgbm.LGBMRegressor (μ و واریانس شرطی) → Beta"))


@register_space("lightgbm_lss_beta", version=1, n_hyperparams=3)
def _space_lightgbm_lss_beta(trial: optuna.Trial) -> dict:
    return {
        "num_leaves": trial.suggest_int("num_leaves", 7, 63, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 400, log=True),
    }


def main() -> None:
    from src.config import set_global_seed
    from src.cv import load_cv_folds, sha256_file
    from src.features.build import FEATURES_A_PATH
    from src.models.s0_runner import RESULTS_MD, baseline_reference, print_summary, run_family_s0, save_results

    set_global_seed()
    df = pd.read_parquet(FEATURES_A_PATH).sort_values("date_gregorian").reset_index(drop=True)
    folds, cv_folds_hash = load_cv_folds()
    f = folds[0]
    tr_mask, te_mask = f.masks(df["date_gregorian"])
    train, test = df.loc[tr_mask], df.loc[te_mask]
    data_snapshot_hash = sha256_file(FEATURES_A_PATH)

    print(f"R0 — {FAMILY} ({LEVEL}) — {f} (train={len(train):,}, test={len(test):,})")
    results = run_family_s0(FAMILY, LEVEL, MODELS, train, test, feature_set="FS_day",
                            data_snapshot_hash=data_snapshot_hash, cv_folds_hash=cv_folds_hash,
                            dataset_source=str(FEATURES_A_PATH))
    baseline = baseline_reference(train, test)
    ok = print_summary(results, baseline)
    save_results(results, baseline)
    print(f"\nذخیره شد در {RESULTS_MD}")
    if not ok:
        raise AssertionError("lightgbm_lss_beta در R0 شکست خورد")


def run_s1(n_jobs: int | None = None) -> None:
    from src.config import set_global_seed
    from src.cv import load_cv_folds, sha256_file
    from src.features.build import FEATURES_A_PATH
    from src.models.s1_runner import (
        DEFAULT_N_JOBS,
        N_SCREENING_FOLDS,
        baseline_reference_multi_fold,
        run_family_s1,
        save_results,
    )

    set_global_seed()
    df = pd.read_parquet(FEATURES_A_PATH).sort_values("date_gregorian").reset_index(drop=True)
    folds, cv_folds_hash = load_cv_folds()
    data_snapshot_hash = sha256_file(FEATURES_A_PATH)

    screening_folds = []
    for f in folds[:N_SCREENING_FOLDS]:
        tr_mask, te_mask = f.masks(df["date_gregorian"])
        screening_folds.append((df.loc[tr_mask], df.loc[te_mask]))

    jobs = n_jobs or DEFAULT_N_JOBS
    results = run_family_s1(FAMILY, LEVEL, "src.models.families.f09_distributional", list(MODELS),
                            screening_folds, feature_set="FS_day", data_snapshot_hash=data_snapshot_hash,
                            cv_folds_hash=cv_folds_hash, dataset_source=str(FEATURES_A_PATH), n_jobs=jobs)
    baseline = baseline_reference_multi_fold(screening_folds)
    save_results(results, FAMILY, LEVEL, baseline_pinball=baseline)
    n_fail = sum(1 for r in results if r.status == "fail")
    print(f"{len(results)} trial تمام شد · {n_fail} شکست")


def run_s2(n_jobs: int | None = None) -> None:
    from src.config import set_global_seed
    from src.cv import load_cv_folds, sha256_file
    from src.features.build import FEATURES_A_PATH
    from src.models.s2_runner import (
        N_S2_FOLDS,
        baseline_reference_5fold,
        run_family_s2,
        save_one_result,
        save_results,
    )

    set_global_seed()
    df = pd.read_parquet(FEATURES_A_PATH).sort_values("date_gregorian").reset_index(drop=True)
    folds, cv_folds_hash = load_cv_folds()
    data_snapshot_hash = sha256_file(FEATURES_A_PATH)
    if len(folds) != N_S2_FOLDS:
        raise AssertionError(f"cv_folds.json باید {N_S2_FOLDS} fold داشته باشد")

    tuning_folds = []
    for f in folds:
        tr_mask, te_mask = f.masks(df["date_gregorian"])
        tuning_folds.append((df.loc[tr_mask], df.loc[te_mask]))

    def _on_result(model_id: str, result) -> None:
        save_one_result(model_id, result, FAMILY, LEVEL)

    results = run_family_s2(FAMILY, LEVEL, "src.models.families.f09_distributional", list(MODELS),
                            tuning_folds, data_snapshot_hash=data_snapshot_hash, cv_folds_hash=cv_folds_hash,
                            dataset_source=str(FEATURES_A_PATH), n_jobs=n_jobs or 1, on_result=_on_result)
    baseline = baseline_reference_5fold(tuning_folds)
    save_results(results, FAMILY, LEVEL, baseline_pinball=baseline)
    n_fail = sum(r.n_fail for r in results.values())
    print(f"{len(results)} مدل تنظیم شد · {n_fail} trial شکست‌خورده")


# ⚠️ عمداً بدون بلوک ``if __name__ == "__main__":`` — بند ۱.۵ استاندارد اجرا.
