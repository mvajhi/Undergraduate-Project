"""زیرساخت مشترک فاز ۹ (تفسیرپذیری) — برازش قهرمان روی مرز holdout بند ۸.۱ و بازگرداندن
شیء مدل واقعی (نه فقط پیش‌بینی) برای SHAP/PDP/ICE/ALE/اهمیت ویژگی.

قهرمان سبک است (۰.۳ ثانیه هر برازش) — هر اسکریپت فاز ۹ خودش دوباره برازش می‌کند، به‌جای
اشتراک شیء بین پردازش‌ها؛ دقیقاً همان هایپرپارامتر/فیچرست S2 بند ۸.۱ (بدون بازتنظیم).
"""

import importlib

import lightgbm as lgb
import pandas as pd

from src.models.card_writer import load_s2_result
from src.models.run_phase8_holdout_eval import load_holdout

CHAMPION = "lightgbm_quantile"
CHAMPION_FAMILY = "F02"


def _raw_categorical_design(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]
                            ) -> tuple[pd.DataFrame, pd.DataFrame]:
    tr, te = train[cols].copy(), test[cols].copy()
    for c in cols:
        if tr[c].dtype == object:
            tr[c] = tr[c].astype("category")
            te[c] = pd.Categorical(te[c], categories=tr[c].cat.categories)
    return tr, te


def fit_champion(tau: float, seed: int = 42):
    """برمی‌گرداند: (model, Xtr, Xte, train, test, cols) — روی مرز holdout بند ۸.۱."""
    train, test, fold = load_holdout()
    result = load_s2_result(CHAMPION, CHAMPION_FAMILY)
    hp = result["best_hyperparams"]
    mod = importlib.import_module("src.models.families.f02_tree")
    cols = mod._feature_cols_s2()

    Xtr, Xte = _raw_categorical_design(train, test, cols)
    model = lgb.LGBMRegressor(objective="quantile", alpha=tau, num_leaves=hp["num_leaves"],
                              learning_rate=hp["learning_rate"], min_child_samples=hp["min_child_samples"],
                              feature_fraction=hp["feature_fraction"], n_estimators=hp["n_estimators"],
                              verbosity=-1, random_state=seed)
    model.fit(Xtr, train["rho"])
    return model, Xtr, Xte, train, test, cols
