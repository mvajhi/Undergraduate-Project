"""بازتنظیم هایپرپارامتر قهرمان روی پنجره‌ی تازه — ابهام (ث) بند ۱۳-۹.

**پرسش.** بند ۱۳-۹ قهرمان را با هایپرپارامتر **تنظیم‌نشده** روی پنجره‌ی تازه اجرا
کرد و ۶.۸٪ هزینه‌ی انتقال گزارش داد. آن عدد دو چیز را با هم اندازه می‌گیرد که باید
جدا شوند: «معماری روی سال بعد ضعیف‌تر است» و «هایپرپارامتر برای این پنجره تنظیم
نشده». تا وقتی جدا نشوند، پاسخ «آیا بازآموزی لازم است؟» قابل دفاع نیست.

**سه پیکربندی، نه دو.**

| شناسه | منشأ |
|---|---|
| `default` | همان مقادیر سخت‌کدشده‌ی `run_phase13_oos.py` — چیزی که تا امروز گزارش شده |
| `phase7` | هایپرپارامتر تنظیم‌شده‌ی قهرمان فاز ۷، روی پنجره‌ی **قفل‌شده** |
| `retuned` | تنظیم‌شده اینجا، روی سال آموزشِ پنجره‌ی تازه |

`default` عمداً در جدول می‌ماند: بدون آن معلوم نمی‌شود چه بخشی از تفاوت کارِ
«تنظیم روی پنجره‌ی درست» است و چه بخشی صرفاً کارِ «تنظیم‌شدن، هر جا» — و اتفاقاً
مقادیر سخت‌کدشده هیچ‌وقت هایپرپارامتر قهرمان فاز ۷ نبودند، پیش‌فرض کتابخانه‌ای بودند.

⚠️ **تنظیم هیچ‌کدام از سه مجموعه‌ی ارزیابی را نمی‌بیند.** سال آموزش سه‌تکه می‌شود:
۰–۶۴٪ برازشِ trial، ۶۴–۸۰٪ اعتبارسنجی trial، ۸۰–۱۰۰٪ **دست‌نخورده** (همان مرجع
درون‌پنجره‌ای `build_splits`). دو سال بعد اصلاً وارد تنظیم نمی‌شوند. بدون این
تفکیک، «بهبود» صرفاً بیش‌برازش روی مجموعه‌ی گزارش بود.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import optuna
import pandas as pd

from src.baselines import operational_metrics, pinball_loss
from src.config import DATA_PROCESSED, REPORTS_DIR, set_global_seed
from src.features.build_v4 import OUTPUT_PATH as FEATURES_PATH
from src.features.build_v4 import feature_set
from src.models.run_phase13_oos import (
    CHAMPION_PARAMS,
    DATA_WINDOW,
    EXPERIMENT,
    TAU_OPERATIONAL,
    TRAIN_YEAR,
    _design,
    build_splits,
)
from src.models.spaces import sample

# رجیستری فضاها هنگام import خانواده پر می‌شود — بدون این خط `sample` خطا می‌دهد.
# عمداً همان فضای فاز ۷ استفاده می‌شود تا مقایسه‌ی `phase7` و `retuned` منصفانه بماند.
import src.models.families.f02_tree  # noqa: F401  (side-effect: register_space)

logger = logging.getLogger(__name__)

OUTPUT_DIR = REPORTS_DIR / "phase13"
BEST_PARAMS_PATH = DATA_PROCESSED / "phase13_best_params.json"

N_TRIALS = 80
SEED = 42

#: مقادیر سخت‌کدشده‌ی `run_phase13_oos.lightgbm_quantile` — مبنای مقایسه.
DEFAULT_PARAMS = {
    "num_leaves": 31, "learning_rate": 0.05, "min_child_samples": 20,
    "n_estimators": 300, "feature_fraction": 1.0,
}

#: قهرمان فاز ۷، تنظیم‌شده روی پنجره‌ی قفل‌شده. یک منبع، تا جدول این سند و مدلی که
#: `run_phase13_oos` واقعاً اجرا می‌کند نتوانند از هم جدا بیفتند.
PHASE7_PARAMS = CHAMPION_PARAMS


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, tau: float, params: dict,
                 seed: int = SEED) -> np.ndarray:
    import lightgbm as lgb

    cols = [c for c in feature_set("full") if c in train.columns]
    Xtr, Xte = _design(train[cols]), _design(test[cols])
    Xte = Xte.reindex(columns=Xtr.columns, fill_value=np.nan)

    model = lgb.LGBMRegressor(
        objective="quantile", alpha=tau, verbosity=-1, random_state=seed, **params
    )
    model.fit(Xtr, train["rho"])
    return np.clip(model.predict(Xte), 0.0, 1.0)


def tune(df: pd.DataFrame, tau: float = TAU_OPERATIONAL, n_trials: int = N_TRIALS) -> dict:
    """جست‌وجوی TPE روی برشِ درونیِ سال آموزش. فضا **همان فضای ثبت‌شده‌ی فاز ۷** است."""
    train_all = df[df["academic_year"] == TRAIN_YEAR].sort_values("date_gregorian")
    modeling = train_all.iloc[: int(0.8 * len(train_all))]
    inner_cut = int(0.8 * len(modeling))
    inner_train, inner_valid = modeling.iloc[:inner_cut], modeling.iloc[inner_cut:]
    logger.info(
        "تنظیم روی %d ردیف، اعتبارسنجی روی %d — مرجع درون‌پنجره‌ای (%d ردیف) دست‌نخورده",
        len(inner_train), len(inner_valid), len(train_all) - len(modeling),
    )

    def objective(trial: optuna.Trial) -> float:
        params = sample("lightgbm_quantile", trial)
        pred = _fit_predict(inner_train, inner_valid, tau, params)
        return float(pinball_loss(inner_valid["rho"].to_numpy(float), pred, tau).mean())

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info("بهترین trial: pinball=%.6f · %s", study.best_value, study.best_params)
    return {"best_params": study.best_params, "best_value": float(study.best_value),
            "n_trials": n_trials, "tau": tau,
            "n_inner_train": len(inner_train), "n_inner_valid": len(inner_valid)}


def evaluate(df: pd.DataFrame, param_sets: dict[str, dict],
             tau: float = TAU_OPERATIONAL) -> pd.DataFrame:
    """هر پیکربندی روی هر سه تقسیم."""
    splits = build_splits(df)
    rows = []
    for label, params in param_sets.items():
        for split_name, split in splits.items():
            pred = _fit_predict(split["train"], split["test"], tau, params)
            m = operational_metrics(split["test"], pred, tau)
            rows.append({"params": label, "split": split_name, "tau": tau,
                         "n_train": len(split["train"]), "n_test": len(split["test"]), **m})
            logger.info("%s · %s → pinball %.5f", label, split_name, m["pinball"])
    return pd.DataFrame(rows)


def _transfer_table(results: pd.DataFrame) -> pd.DataFrame:
    """هزینه‌ی انتقال هر پیکربندی: افت نسبت به مرجع درون‌پنجره‌ای **خودش**."""
    out = []
    for label, block in results.groupby("params"):
        base = float(block.loc[block.split == "in_window", "pinball"].iloc[0])
        row = {"params": label, "in_window": base}
        for sp in block.split.unique():
            if sp == "in_window":
                continue
            row[f"{sp}_pinball"] = float(block.loc[block.split == sp, "pinball"].iloc[0])
            row[f"{sp}_transfer_%"] = 100 * (row[f"{sp}_pinball"] / base - 1)
        out.append(row)
    return pd.DataFrame(out)


def run(n_trials: int = N_TRIALS, *, log_mlflow: bool = True) -> tuple[pd.DataFrame, dict]:
    df = pd.read_parquet(FEATURES_PATH)
    study = tune(df, n_trials=n_trials)

    param_sets = {
        "default": DEFAULT_PARAMS,
        "phase7": PHASE7_PARAMS,
        "retuned": {**study["best_params"]},
    }
    results = evaluate(df, param_sets)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_DIR / "hyperparameter_retune.csv", index=False)
    BEST_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    BEST_PARAMS_PATH.write_text(
        json.dumps({"study": study, "param_sets": param_sets}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if log_mlflow:
        _log(results, param_sets)
    return results, study


def _log(results: pd.DataFrame, param_sets: dict) -> None:
    try:
        import mlflow
    except ImportError:
        logger.warning("mlflow نصب نیست — ثبت رد شد")
        return

    from src.config import MLFLOW_TRACKING_URI

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    for _, row in results.iterrows():
        name = (f"F02_lightgbm_L1_rho_FSfull_t{int(row.tau * 1000):03d}"
                f"_{row.split}_hp-{row.params}_s{SEED}")
        with mlflow.start_run(run_name=name):
            mlflow.set_tags({
                "data_window": DATA_WINDOW,       # ⚠️ اجباری — بند ۱۳-۱۲ پ
                "split": row.split, "model_family": "F02", "phase": "13",
                "hp_source": row.params,
            })
            mlflow.log_params({"model_name": "lightgbm_quantile", "tau": row.tau,
                               "level": "L1", "seed": SEED,
                               **param_sets[row.params]})
            mlflow.log_metrics({k: float(row[k]) for k in results.columns
                                if k not in {"params", "split"} and pd.notna(row[k])})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    set_global_seed()
    results, study = run()

    pd.set_option("display.width", 220)
    print("\n=== pinball@τ به تفکیک پیکربندی و تقسیم ===")
    print(results.pivot_table(index="split", columns="params", values="pinball")
          .to_string(float_format=lambda v: f"{v:.5f}"))
    print("\n=== هزینه‌ی انتقال (نسبت به مرجع درون‌پنجره‌ایِ خودِ همان پیکربندی) ===")
    print(_transfer_table(results).to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print(f"\nبهترین هایپرپارامتر: {study['best_params']}")
