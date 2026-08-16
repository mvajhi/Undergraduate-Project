"""بند 7.20 سند فاز ۷ — خانواده‌ی ۱۱: تصمیم-محور (Newsvendor یادگیرنده).

بند ۵ سند تصمیم ۳۷ («فهرست کوتاه اسپرینت C»): این خانواده «تقریباً رایگان» است چون
`l1_quantile_regression` (خ۱، از قبل تنظیم‌شده) عملاً یک برآوردگر ERM-Newsvendor است —
کاری که اینجا واقعاً اضافه می‌شود، **آزمون هم‌ارزی نظری بند 7.20.3** است، نه یک مدل
جدید.

## آزمون هم‌ارزی نظری (7.20.3) — نتیجه و یک یافته‌ی واقعی

هزینه‌ی نیوزوندور در واحد پرس، با $\\hat\\rho$ خطی و $\\widehat{CookQty}=Res(1-\\hat\\rho)$:

$$\\text{Cost}_i = C_u\\max(Recv_i-Q_i,0) + C_o\\max(Q_i-Recv_i,0)
= Res_i\\big[C_u\\max(\\hat\\rho_i-\\rho_i,0) + C_o\\max(\\rho_i-\\hat\\rho_i,0)\\big]$$

با $\\tau=C_o/(C_u+C_o)$ (ردیف ۳۶ decision_log)، این دقیقاً برابر است با
$Res_i\\cdot(C_u+C_o)\\cdot\\text{pinball}(\\rho_i,\\hat\\rho_i,\\tau)$ — یعنی هدف واقعی
ERM-Newsvendor **pinball وزن‌دار با $Res$** است، نه pinball خام‌ای که
`l1_quantile_regression` در S2 (بند 7.6) با آن انتخاب و تنظیم شد.

⚠️ **این خودش یک یافته‌ی واقعی است، نه انطباق کامل** (دقیقاً همان چیزی که بند 7.20.3
پیش‌بینی کرده بود: «اگر واگرا شدند، یا یکی باگ دارد یا خودِ واگرایی آموزنده است»).
آزمایش مستقیم (بازبرازش با `sample_weight=Res` در برابر بدون‌وزن، هر دو با همان α
بهینه‌ی S2، روی ۵ fold رسمی):

| مدل | pinball خام (میانگین ردیف) | هزینه‌ی کل نیوزوندور ($\\sum Res_i\\cdot\\text{pinball}_i$) |
|---|---|---|
| `l1_quantile_regression` (S2، بدون‌وزن) | **0.01319** (بهتر به‌ظاهر) | 7445.35 |
| نسخه‌ی Res-وزن‌دار (ERM-NV واقعی) | 0.01358 (بدتر به‌ظاهر) | **7239.52** (۲.۷۶٪ کمتر) |

یعنی مدلی که بند 7.6 (معیار انتخاب S2) به‌عنوان بهتر انتخاب می‌کند، از منظر **هزینه‌ی
واقعی ریالی** (که بند 7.20 دنبالش است) ۲.۷۶٪ بدتر از رقیبش عمل می‌کند — چون معیار
انتخاب S2 هر ردیف را یکسان می‌شمارد، در حالی‌که هزینه‌ی واقعی به‌اندازه‌ی $Res$ هر
سلول وزن دارد (سلف/وعده‌ی پرحجم، خطای یکسانِ نسبی، هزینه‌ی مطلق بیشتری دارد).

**پیامد عملیاتی:** برای تصمیم نهایی سایز پخت (نه رتبه‌بندی مدل‌ها در جدول مقایسه)،
باید از نسخه‌ی Res-وزن‌دار استفاده شود. این دقیقاً همان چیزی است که بند ۲-۲ سند
تعریف مسئله با وزن‌دهی $\\sqrt{Res}$ در pinball loss خواسته بود (ردیف ۸ decision_log) —
اینجا با $Res$ خطی (نه $\\sqrt{Res}$) چون اشتقاق مستقیم هزینه، نه یک انتخاب دلبخواه.
"""

import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import StandardScaler

from src.models.families import common
from src.models.families.f01_linear import _design_s2
from src.models.registry import ModelSpec, register
from src.models.spaces import register_space

FAMILY = "F11"
LEVEL = "L1"


def fit_predict_erm_newsvendor(train: pd.DataFrame, test: pd.DataFrame, tau: float,
                               alpha: float = 0.01, **hp) -> np.ndarray:
    """رگرسیون کوانتایل **وزن‌دار با Res** — برآوردگر درستِ ERM-Newsvendor (نه
    `l1_quantile_regression` بدون‌وزن که واحد pinball را به‌جای واحد ریالی کمینه می‌کند).
    فیچرست و ساختار دقیقاً همان `l1_quantile_regression` (بند 7.5.3 ردیف رگرسیون
    کوانتایل خطی) است — تنها تفاوت، وزن نمونه."""
    Xtr, Xte = _design_s2(train, test, quantreg=True)
    scaler = StandardScaler().fit(Xtr)
    model = QuantileRegressor(quantile=tau, alpha=alpha, solver="highs")
    model.fit(scaler.transform(Xtr), train["rho"], sample_weight=train["Res"].to_numpy(float))
    return np.clip(model.predict(scaler.transform(Xte)), 0.0, 1.0)


def newsvendor_cost(actual_rho: np.ndarray, pred_rho: np.ndarray, res: np.ndarray, tau: float) -> float:
    """هزینه‌ی کل نیوزوندور در واحد $Res\\cdot\\text{pinball}$ — معیار واقعی بند 7.20،
    نه pinball خام."""
    from src.baselines import pinball_loss
    return float((pinball_loss(actual_rho, pred_rho, tau) * res).sum())


MODELS = {"erm_newsvendor": fit_predict_erm_newsvendor}

register(ModelSpec(model_id="erm_newsvendor", family=FAMILY, levels=(LEVEL,), quantile_route="Q1",
                   algorithm="sklearn.QuantileRegressor (sample_weight=Res — ERM-Newsvendor واقعی)"))


@register_space("erm_newsvendor", version=1, n_hyperparams=1)
def _space_erm_newsvendor(trial: optuna.Trial) -> dict:
    return {"alpha": trial.suggest_float("alpha", 1e-5, 1e0, log=True)}


def main() -> None:
    """R0 — بند 7.20 («تقریباً رایگان»، یک مدل)."""
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
        raise AssertionError("erm_newsvendor در R0 شکست خورد")


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
    results = run_family_s1(FAMILY, LEVEL, "src.models.families.f11_decision", list(MODELS),
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

    results = run_family_s2(FAMILY, LEVEL, "src.models.families.f11_decision", list(MODELS), tuning_folds,
                            data_snapshot_hash=data_snapshot_hash, cv_folds_hash=cv_folds_hash,
                            dataset_source=str(FEATURES_A_PATH), n_jobs=n_jobs or 1, on_result=_on_result)
    baseline = baseline_reference_5fold(tuning_folds)
    save_results(results, FAMILY, LEVEL, baseline_pinball=baseline)
    n_fail = sum(r.n_fail for r in results.values())
    print(f"{len(results)} مدل تنظیم شد · {n_fail} trial شکست‌خورده")


# ⚠️ عمداً بدون بلوک ``if __name__ == "__main__":`` — بند ۱.۵ استاندارد اجرا.
# اجرا: ``python -m src.models.run_family src.models.families.f11_decision --stage {s0,s1,s2}``
