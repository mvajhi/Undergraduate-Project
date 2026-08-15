"""بند 7.11 سند فاز ۷ — خانواده‌ی ۲: درختی و بوستینگ (F02).

⚠️ **این ماژول کل بند 7.11 (۱۶ مدل) را پیاده نمی‌کند** — طبق پیمایش محور-اول
(بند 7.9.1 بازنویسی‌شده) و فهرست کوتاه اسپرینت C (`doc/decisions/37-phase7-rescope.md`
بند ۵)، فقط سه نماینده ساخته می‌شوند: **LightGBM-quantile**، **CatBoost** (هر دو با
کمینه‌سازی بومی pinball)، و **QRF** (Quantile Regression Forest، `quantile-forest`).
مابقی ۱۳ مدل (Decision Tree خام، RF/ET پایه، GRF/DRF، HistGB، NGBoost، LSS، EBM،
RuleFit) طبق یافته‌ی ۱۴ (`doc/progress/07-*.md`) زیرمجموعه‌ی عملکردی همین سه‌اند و در
فهرست کوتاه کنار گذاشته شدند — نه چون آزموده‌نشده، چون *انتخاب‌نشده با دلیل مستند*.

**دامنه‌ی مدل: سراسری** (نه per_restaurant) — طبق یافته‌ی ۱۴ اسپرینت A: مدل‌های
درختی/بوستینگ خودشان تعامل سلف/شهر را از فیچرهای دسته‌ای کشف می‌کنند؛ تفکیک صریح فقط
داده را قطعه‌قطعه و ضعیف‌تر می‌کند (LightGBM کاوشگر: Δ مثبت روی هر سه دامنه‌ی محلی).

هر ``fit_predict_*`` امضای ``(train, test, tau, **hyperparams) -> np.ndarray`` دارد،
مطابق بند ۱.۱ `doc/phase7-execution-standard.md`.
"""

import json
from functools import lru_cache

import numpy as np
import optuna
import pandas as pd

from src.models.families import common
from src.models.registry import ModelSpec, register
from src.models.spaces import register_space

FAMILY = "F02"
LEVEL = "L1"
FEATURE_SET_S0 = "FS_day"


@lru_cache(maxsize=1)
def _feature_cols() -> list[str]:
    from src.features.build import FEATURE_SETS_PATH
    return json.loads(FEATURE_SETS_PATH.read_text())[FEATURE_SET_S0]


@lru_cache(maxsize=1)
def _feature_cols_s2() -> list[str]:
    """بند 7.5.3 ردیف «درختی/بوستینگ» — `FS_full_A` کامل، بدون حذف زودهنگام (درخت
    خودش برهم‌کنش/غیرخطی را کشف می‌کند؛ هرس فقط با SHAP-RFE گام ۳ که در فهرست کوتاه
    اسپرینت C نیست — بودجه‌ی محدود صرف مدل‌های بیشتر، نه هرس فیچر عمیق‌تر شد)."""
    from src.features.build import FEATURE_SETS_PATH
    return json.loads(FEATURE_SETS_PATH.read_text())["FS_full_A"]


def _design(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return common.design_matrix(train, test, _feature_cols())


def _design_s2(train: pd.DataFrame, test: pd.DataFrame, quantreg: bool = False
              ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """رابط مشترک s2_runner (بند 7.6.3) — ``quantreg`` اینجا بی‌معناست (هر سه مدل Q1
    بومی‌اند، مثل خ۱)، فقط برای سازگاری امضا نگه داشته شده."""
    return common.design_matrix(train, test, _feature_cols_s2())


def _raw_categorical_design(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]
                            ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """بند 7.5.3: دسته‌ای‌ها **خام** (نه یک‌هات) برای LightGBM/CatBoost — رمزگذاری بومی
    این کتابخانه‌ها معمولاً از یک‌هات بهتر عمل می‌کند، به‌خصوص روی `RestaurantName` (۳۰
    سطح)."""
    tr, te = train[cols].copy(), test[cols].copy()
    for c in cols:
        if tr[c].dtype == object:
            tr[c] = tr[c].astype("category")
            te[c] = pd.Categorical(te[c], categories=tr[c].cat.categories)
    return tr, te


# ---------------------------------------------------------------------------
# ۱: LightGBM-quantile — کمینه‌سازی بومی pinball (بند 7.11.1 عضو ۷)
# ---------------------------------------------------------------------------

def fit_predict_lightgbm_quantile(train: pd.DataFrame, test: pd.DataFrame, tau: float,
                                  num_leaves: int = 31, learning_rate: float = 0.05,
                                  min_child_samples: int = 20, feature_fraction: float = 1.0,
                                  n_estimators: int = 300, **hp) -> np.ndarray:
    import lightgbm as lgb

    Xtr, Xte = _raw_categorical_design(train, test, _feature_cols_s2())
    model = lgb.LGBMRegressor(objective="quantile", alpha=tau, num_leaves=num_leaves,
                              learning_rate=learning_rate, min_child_samples=min_child_samples,
                              feature_fraction=feature_fraction, n_estimators=n_estimators,
                              verbosity=-1, random_state=42)
    model.fit(Xtr, train["rho"])
    return np.clip(model.predict(Xte), 0.0, 1.0)


# ---------------------------------------------------------------------------
# ۲: CatBoost — رمزگذاری دسته‌ای بومی (بند 7.11.1 عضو ۹)
# ---------------------------------------------------------------------------

def fit_predict_catboost_quantile(train: pd.DataFrame, test: pd.DataFrame, tau: float,
                                  depth: int = 6, learning_rate: float = 0.05,
                                  l2_leaf_reg: float = 3.0, iterations: int = 300, **hp) -> np.ndarray:
    from catboost import CatBoostRegressor

    cols = _feature_cols_s2()
    cat_idx = [i for i, c in enumerate(cols) if train[c].dtype == object]
    Xtr, Xte = train[cols].copy(), test[cols].copy()
    for i in cat_idx:
        Xtr.iloc[:, i] = Xtr.iloc[:, i].astype(str)
        Xte.iloc[:, i] = Xte.iloc[:, i].astype(str)

    model = CatBoostRegressor(loss_function=f"Quantile:alpha={tau}", depth=depth,
                              learning_rate=learning_rate, l2_leaf_reg=l2_leaf_reg,
                              iterations=iterations, cat_features=cat_idx,
                              verbose=False, random_seed=42, allow_writing_files=False)
    model.fit(Xtr, train["rho"])
    return np.clip(np.asarray(model.predict(Xte)), 0.0, 1.0)


# ---------------------------------------------------------------------------
# ۳: QRF — Quantile Regression Forest (بند 7.11.1 عضو ۴، Meinshausen 2006)
# ---------------------------------------------------------------------------

def fit_predict_qrf(train: pd.DataFrame, test: pd.DataFrame, tau: float,
                    n_estimators: int = 200, max_depth: int | None = None,
                    min_samples_leaf: int = 5, **hp) -> np.ndarray:
    from quantile_forest import RandomForestQuantileRegressor

    Xtr, Xte = common.design_matrix(train, test, _feature_cols_s2())
    model = RandomForestQuantileRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                          min_samples_leaf=min_samples_leaf,
                                          random_state=42, n_jobs=1)
    model.fit(Xtr, train["rho"])
    pred = model.predict(Xte, quantiles=[tau])
    return np.clip(np.asarray(pred).ravel(), 0.0, 1.0)


# ---------------------------------------------------------------------------
# رجیستری
# ---------------------------------------------------------------------------

MODELS = {
    "lightgbm_quantile": fit_predict_lightgbm_quantile,
    "catboost_quantile": fit_predict_catboost_quantile,
    "qrf": fit_predict_qrf,
}

_QUANTILE_ROUTES = {  # هر سه Q1 — کمینه‌سازی مستقیم pinball
    "lightgbm_quantile": "Q1", "catboost_quantile": "Q1", "qrf": "Q1",
}

_ALGORITHMS = {
    "lightgbm_quantile": "lightgbm.LGBMRegressor(objective=quantile)",
    "catboost_quantile": "catboost.CatBoostRegressor(loss=Quantile)",
    "qrf": "quantile_forest.RandomForestQuantileRegressor",
}

#: هیچ مدلی از قیف تنظیم مستثنا نیست (برخلاف خ۱ که glm_binomial رد‌شده داشت)
QUANTREG_MODEL_IDS: frozenset[str] = frozenset()
TUNING_EXCLUDED: frozenset[str] = frozenset()

for _model_id, _route in _QUANTILE_ROUTES.items():
    register(ModelSpec(model_id=_model_id, family=FAMILY, levels=(LEVEL,), quantile_route=_route,
                       algorithm=_ALGORITHMS[_model_id]))


# ---------------------------------------------------------------------------
# فضای هایپرپارامتر — بند 7.11.2 (فضای پیوسته، بدون کِران کاردینالیتی)
# ---------------------------------------------------------------------------

@register_space("lightgbm_quantile", version=1, n_hyperparams=5)
def _space_lightgbm_quantile(trial: optuna.Trial) -> dict:
    return {
        "num_leaves": trial.suggest_int("num_leaves", 7, 127, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "n_estimators": trial.suggest_int("n_estimators", 50, 500, log=True),
    }


@register_space("catboost_quantile", version=1, n_hyperparams=4)
def _space_catboost_quantile(trial: optuna.Trial) -> dict:
    return {
        "depth": trial.suggest_int("depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        "iterations": trial.suggest_int("iterations", 100, 500, log=True),
    }


@register_space("qrf", version=1, n_hyperparams=3)
def _space_qrf(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 400, log=True),
        "max_depth": trial.suggest_categorical("max_depth", [None, 5, 10, 15, 20]),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 30, log=True),
    }


def main() -> None:
    """اجرای R0 (بند 7.3.1/7.3.2) برای هر ۳ عضو کوتاه‌فهرست‌شده‌ی F02 روی نخستین fold."""
    from src.config import set_global_seed
    from src.cv import load_cv_folds, sha256_file
    from src.features.build import FEATURES_A_PATH
    from src.models.s0_runner import (
        RESULTS_MD,
        baseline_reference,
        print_summary,
        run_family_s0,
        save_results,
    )

    set_global_seed()
    df = pd.read_parquet(FEATURES_A_PATH).sort_values("date_gregorian").reset_index(drop=True)
    folds, cv_folds_hash = load_cv_folds()
    f = folds[0]
    tr_mask, te_mask = f.masks(df["date_gregorian"])
    train, test = df.loc[tr_mask], df.loc[te_mask]
    data_snapshot_hash = sha256_file(FEATURES_A_PATH)

    print(f"R0 — {FAMILY} ({LEVEL}) — {f} (train={len(train):,}, test={len(test):,})")
    print("فهرست کوتاه اسپرینت C (بند ۵ سند تصمیم ۳۷): فقط ۳ عضو از ۱۶\n")
    results = run_family_s0(FAMILY, LEVEL, MODELS, train, test, feature_set=FEATURE_SET_S0,
                            data_snapshot_hash=data_snapshot_hash, cv_folds_hash=cv_folds_hash,
                            dataset_source=str(FEATURES_A_PATH))
    baseline = baseline_reference(train, test)
    ok = print_summary(results, baseline)
    save_results(results, baseline)
    print(f"\nذخیره شد در {RESULTS_MD}")
    if not ok:
        raise AssertionError("یک یا چند مدل L1×F02 در R0 شکست خوردند")


def run_s1(n_jobs: int | None = None) -> None:
    """اجرای R1 (بند 7.3.2 بازنویسی‌شده) — ۱۰ trial تصادفی × ۳ fold، سقف ۶۰۰ ثانیه/trial."""
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

    model_ids = sorted(MODELS)
    jobs = n_jobs or DEFAULT_N_JOBS
    print(f"R1 — {FAMILY} ({LEVEL}) — {len(model_ids)} مدل، {N_SCREENING_FOLDS} fold نخست، "
         f"{jobs} worker موازی\n")

    results = run_family_s1(FAMILY, LEVEL, "src.models.families.f02_tree", model_ids, screening_folds,
                            feature_set=FEATURE_SET_S0, data_snapshot_hash=data_snapshot_hash,
                            cv_folds_hash=cv_folds_hash, dataset_source=str(FEATURES_A_PATH),
                            n_jobs=jobs)
    baseline = baseline_reference_multi_fold(screening_folds)
    save_results(results, FAMILY, LEVEL, baseline_pinball=baseline)

    n_fail = sum(1 for r in results if r.status == "fail")
    print(f"\n{len(results)} trial تمام شد · {n_fail} شکست · "
         f"ذخیره شد در reports/phase7/S1_screening_{FAMILY}.md")


def run_s2(n_jobs: int | None = None) -> None:
    """اجرای R2 (بند 7.3.2/7.6.2 بازنویسی‌شده) — Optuna TPE، هر ۵ fold، بودجه‌ی کِران‌شده
    به کاردینالیتی (اینجا بی‌اثر — هر سه فضا پیوسته‌اند)، سقف زمانی ۹۰ دقیقه/مدل."""
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
        raise AssertionError(f"cv_folds.json باید {N_S2_FOLDS} fold داشته باشد، نه {len(folds)}")

    tuning_folds = []
    for f in folds:
        tr_mask, te_mask = f.masks(df["date_gregorian"])
        tuning_folds.append((df.loc[tr_mask], df.loc[te_mask]))

    model_ids = sorted(MODELS)
    jobs = n_jobs or min(6, len(model_ids))
    print(f"R2 — {FAMILY} ({LEVEL}) — {len(model_ids)} مدل، هر {N_S2_FOLDS} fold، {jobs} worker موازی\n")

    def _on_result(model_id: str, result) -> None:
        save_one_result(model_id, result, FAMILY, LEVEL)

    results = run_family_s2(FAMILY, LEVEL, "src.models.families.f02_tree", model_ids, tuning_folds,
                            data_snapshot_hash=data_snapshot_hash, cv_folds_hash=cv_folds_hash,
                            dataset_source=str(FEATURES_A_PATH), n_jobs=jobs, on_result=_on_result)

    baseline = baseline_reference_5fold(tuning_folds)
    save_results(results, FAMILY, LEVEL, baseline_pinball=baseline)

    n_fail = sum(r.n_fail for r in results.values())
    print(f"\n{len(results)} مدل تنظیم شد · {n_fail} trial شکست‌خورده · "
         f"ذخیره شد در reports/phase7/S2_tuning_{FAMILY}.md")


# ⚠️ عمداً بدون بلوک ``if __name__ == "__main__":`` — بند ۱.۵ استاندارد اجرا (رفع
# BrokenProcessPool زیر spawn). همیشه از ``python -m src.models.run_family
# src.models.families.f02_tree --stage {s0,s1,s2}`` اجرا شود.
