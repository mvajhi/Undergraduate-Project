"""اعتبارسنجی برون‌نمونه‌ای روی سال‌های تحصیلی — بند ۱۳-۹، دروازه‌ی M6.

**سؤالی که تا امروز قابل پرسیدن نبود.** مطالعه‌ی اصلی یک پنجره‌ی ۱۴۲ روزه در یک سال
تحصیلی داشت؛ با یک پنجره نمی‌شود پرسید «آیا این نتیجه تعمیم‌پذیر است؟» — نه اینکه
پاسخ منفی باشد، اصلاً **قابل آزمون نبود**. سه سال تحصیلی این را ممکن می‌کند.

**پروتکل: holdout زمانی واقعی، نه fold تصادفی.**

```
آموزش: ۱۴۰۲-۱۴۰۳  →  آزمون: ۱۴۰۳-۱۴۰۴   (یک سال جلوتر)
آموزش: ۱۴۰۲-۱۴۰۳  →  آزمون: ۱۴۰۴-۱۴۰۵   (دو سال جلوتر)
```

محور تقسیم **سال تحصیلی** است نه تاریخ خام (F103): مرز سال تحصیلی همان‌جایی است که
جمعیت دانشجویی عوض می‌شود، پس تقسیم روی تاریخِ وسط ترم دو رژیم را قاطی می‌کند.

**مرجع درون‌پنجره‌ای، بدون آن عدد بی‌معناست.** هر افت عملکرد روی سال‌های بعد دو علت
ممکن دارد: مدل ذاتاً ضعیف است، یا انتقال سخت است. برای تفکیکشان، همان مدل با همان
پروتکل روی یک holdout **درون** ۱۴۰۲-۱۴۰۳ هم اجرا می‌شود. تفاضل این دو، خودِ «هزینه‌ی
انتقال» است.

⚠️ **دروازه‌ی انصاف (`AGENTS.md`).** پنجره‌ی داده عوض شده، پس foldها و `cv_folds_hash`
هم عوض شده‌اند و **هیچ‌کدام از ~۱۶۹ پیکربندی فاز ۷ با runهای اینجا در یک جدول
قابل‌مقایسه نیستند**. این ناسازگاری درست رخ می‌دهد ولی بی‌صدا هم رخ می‌دهد، پس هر run
اینجا experiment جدا (`phase13_oos`) و tag اجباری `data_window=v4_extended` دارد.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from src.baselines import TAU_GRID, operational_metrics
from src.config import DATA_PROCESSED, REPORTS_DIR, set_global_seed
from src.features.build_v4 import OUTPUT_PATH as FEATURES_PATH
from src.features.build_v4 import feature_set

logger = logging.getLogger(__name__)

EXPERIMENT = "phase13_oos"
DATA_WINDOW = "v4_extended"
OUTPUT_DIR = REPORTS_DIR / "phase13"
FOLDS_PATH = DATA_PROCESSED / "cv_folds_v4.json"

TRAIN_YEAR = "1402-1403"
TEST_YEARS = ["1403-1404", "1404-1405"]

#: نقطه‌ی عملیاتی پروژه (برآورد ذی‌نفع $C_u \\approx 4C_o$). شبکه‌ی کامل هم اجرا
#: می‌شود تا پرسش «آیا ۰.۲۰ هنوز درست است؟» با عدد پاسخ بگیرد، نه با فرض.
TAU_OPERATIONAL = 0.20


# ---------------------------------------------------------------------------
# تقسیم‌ها
# ---------------------------------------------------------------------------


def build_splits(df: pd.DataFrame) -> dict[str, dict]:
    """تقسیم‌های برون‌نمونه‌ای + مرجع درون‌پنجره‌ای.

    مرجع درون‌پنجره‌ای، **۲۰٪ پایانی خودِ سال آموزش** است (تقسیم زمانی، نه تصادفی) —
    تا با آزمون‌های برون‌نمونه‌ای از یک جنس باشد و تفاضلشان فقط «انتقال» را نشان دهد.
    """
    train_all = df[df["academic_year"] == TRAIN_YEAR].sort_values("date_gregorian")
    cut = int(0.8 * len(train_all))
    in_window_train = train_all.iloc[:cut]
    in_window_test = train_all.iloc[cut:]

    splits = {
        "in_window": {"train": in_window_train, "test": in_window_test,
                      "label": f"{TRAIN_YEAR} (۲۰٪ پایانی)"},
    }
    for year in TEST_YEARS:
        test = df[df["academic_year"] == year]
        if len(test):
            splits[f"oos_{year}"] = {"train": train_all, "test": test, "label": year}
    return splits


def write_folds_manifest(df: pd.DataFrame, splits: dict, path=FOLDS_PATH) -> dict:
    """ثبت مرز foldها با هش — معادل `cv_folds.json` برای پنجره‌ی تازه.

    ⚠️ این هش **عمداً** با `cv_folds_hash` فاز ۷ نمی‌خواند. ناسازگاری واقعی است و
    ثبتش تنها راهی است که در جدول مقایسه بی‌صدا نماند.
    """
    import hashlib

    manifest = {
        "data_window": DATA_WINDOW,
        "split_axis": "academic_year",
        "rationale": "F103 — مرز سال تحصیلی همان‌جایی است که جمعیت دانشجویی عوض می‌شود",
        "n_rows_total": int(len(df)),
        "splits": {
            name: {
                "label": s["label"],
                "n_train": int(len(s["train"])), "n_test": int(len(s["test"])),
                "train_range": [str(s["train"]["date_gregorian"].min().date()),
                                str(s["train"]["date_gregorian"].max().date())],
                "test_range": [str(s["test"]["date_gregorian"].min().date()),
                               str(s["test"]["date_gregorian"].max().date())],
            }
            for name, s in splits.items()
        },
    }
    payload = json.dumps(manifest, ensure_ascii=False, sort_keys=True)
    manifest["cv_folds_hash"] = hashlib.sha256(payload.encode()).hexdigest()[:16]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


# ---------------------------------------------------------------------------
# مدل‌ها
# ---------------------------------------------------------------------------

#: کلیدهای گروه خط پایه، نگاشت‌شده به نام ستون‌های v4. خط پایه‌های `src/baselines.py`
#: با نام‌های پنجره‌ی قفل‌شده نوشته شده‌اند (`RestaurantName`, `Meal`, `dow`)؛ به‌جای
#: تغییرشان — که شاهد تاریخی گزارش را دست می‌زند — اینجا نگاشت می‌شوند.
GROUP_KEYS = ["restaurant", "meal", "day_of_week"]


def b0_cook_all(train, test, tau):
    """وضع موجود: پخت = کل رزرو."""
    return np.zeros(len(test))


def b1_global_mean(train, test, tau):
    return np.full(len(test), train["NoRecv"].sum() / train["Res"].sum())


def b3_empirical_quantile(train, test, tau):
    """کوانتایل تجربی گروه — رقیب سرسختِ مستند فاز ۶/۷."""
    q = train.groupby(GROUP_KEYS, observed=True)["rho"].quantile(tau)
    fallback = train["rho"].quantile(tau)
    idx = pd.MultiIndex.from_frame(test[GROUP_KEYS])
    return q.reindex(idx).fillna(fallback).to_numpy()


def lightgbm_quantile(train, test, tau, seed: int = 42):
    """قهرمان فاز ۷ (`lightgbm_quantile`)، بازآموزی‌شده روی پنجره‌ی تازه.

    وزن‌های فاز ۷ بارگذاری **نمی‌شوند**: آن‌ها روی فیچرست و شبکه‌ی پنجره‌ی قفل‌شده
    برازش شده‌اند و ماتریس v4 ستون‌های دیگری دارد. پرسش بند ۱۳-۹ هم همین است —
    «آیا این **معماری** روی سال بعد کار می‌کند؟» نه «آیا این فایل وزن کار می‌کند؟».
    """
    import lightgbm as lgb

    cols = [c for c in feature_set("full") if c in train.columns]
    Xtr, Xte = _design(train[cols]), _design(test[cols])
    Xte = Xte.reindex(columns=Xtr.columns, fill_value=np.nan)

    model = lgb.LGBMRegressor(
        objective="quantile", alpha=tau, num_leaves=31, learning_rate=0.05,
        min_child_samples=20, n_estimators=300, verbosity=-1, random_state=seed,
    )
    model.fit(Xtr, train["rho"])
    return np.clip(model.predict(Xte), 0.0, 1.0)


def _design(X: pd.DataFrame) -> pd.DataFrame:
    out = X.copy()
    for col in out.columns:
        if out[col].dtype == object or isinstance(out[col].dtype, pd.CategoricalDtype):
            out[col] = out[col].astype("category").cat.codes.replace(-1, np.nan)
        elif out[col].dtype == bool:
            out[col] = out[col].astype("int8")
    return out


MODELS = {
    "B0_cook_all": b0_cook_all,
    "B1_global_mean": b1_global_mean,
    "B3_empirical_quantile": b3_empirical_quantile,
    "lightgbm_quantile": lightgbm_quantile,
}


# ---------------------------------------------------------------------------
# اجرا
# ---------------------------------------------------------------------------


def run(taus: list[float] | None = None, *, log_mlflow: bool = True) -> pd.DataFrame:
    taus = taus or TAU_GRID
    df = pd.read_parquet(FEATURES_PATH)
    splits = build_splits(df)
    manifest = write_folds_manifest(df, splits)
    logger.info("cv_folds_hash (v4) = %s", manifest["cv_folds_hash"])

    rows = []
    for split_name, split in splits.items():
        train, test = split["train"], split["test"]
        for model_name, fn in MODELS.items():
            for tau in taus:
                pred = fn(train, test, tau)
                metrics = operational_metrics(test, pred, tau)
                rows.append({
                    "split": split_name, "split_label": split["label"],
                    "model": model_name, "tau": tau,
                    "n_train": len(train), **metrics,
                })
                logger.info(
                    "%s · %s · τ=%.2f → pinball %.5f · coverage %.3f · shortage %.3f",
                    split_name, model_name, tau,
                    metrics["pinball"], metrics["coverage"], metrics["shortage_rate"],
                )

    results = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_DIR / "oos_results.csv", index=False)

    if log_mlflow:
        _log_to_mlflow(results, manifest)
    return results


def _log_to_mlflow(results: pd.DataFrame, manifest: dict) -> None:
    """هر ردیف یک run، با tagهای اجباری دروازه‌ی انصاف."""
    try:
        import mlflow
    except ImportError:
        logger.warning("mlflow نصب نیست — ثبت رد شد")
        return

    from src.config import MLFLOW_TRACKING_URI

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    for _, row in results.iterrows():
        name = f"{row.model}_L1_rho_FSfull_t{int(row.tau * 1000):03d}_{row.split}_s42"
        with mlflow.start_run(run_name=name):
            mlflow.set_tags({
                "data_window": DATA_WINDOW,           # ⚠️ اجباری — بند ۱۳-۱۲ پ
                "split": row.split,
                "model_family": "F02" if "lightgbm" in row.model else "baseline",
                "phase": "13",
            })
            mlflow.log_params({
                "model_name": row.model, "tau": row.tau, "level": "L1",
                "cv_folds_hash": manifest["cv_folds_hash"],
                "split_axis": manifest["split_axis"],
                "n_train": int(row.n_train), "seed": 42,
            })
            mlflow.log_metrics({
                k: float(row[k]) for k in results.columns
                if k not in {"split", "split_label", "model", "tau", "n_train"}
                and pd.notna(row[k])
            })
    logger.info("ثبت در MLflow: experiment=%s، %d run", EXPERIMENT, len(results))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    set_global_seed()
    results = run()

    pd.set_option("display.width", 220)
    at_tau = results[results.tau == TAU_OPERATIONAL]

    print(f"\n=== pinball در τ={TAU_OPERATIONAL} ===")
    print(at_tau.pivot(index="model", columns="split", values="pinball").round(5).to_string())

    print(f"\n=== پوشش تجربی (باید ≈ τ={TAU_OPERATIONAL} باشد) ===")
    print(at_tau.pivot(index="model", columns="split", values="coverage").round(3).to_string())

    print("\n=== نرخ کمبود ===")
    print(at_tau.pivot(index="model", columns="split", values="shortage_rate").round(3).to_string())

    print("\n=== کاهش دورریز (٪) ===")
    print((100 * at_tau.pivot(index="model", columns="split", values="waste_reduction_pct")).round(1).to_string())

    champ = results[results.model == "lightgbm_quantile"]
    print("\n=== آیا τ=۰.۲۰ هنوز نقطه‌ی عملیاتی درست است؟ (قهرمان) ===")
    print(champ.pivot(index="tau", columns="split", values="coverage").round(3).to_string())
