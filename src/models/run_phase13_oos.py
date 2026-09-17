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

#: هایپرپارامتر قهرمان فاز ۷ (`reports/phase7/tau_sensitivity_champion.md`).
#:
#: ⚠️ تا پیش از این، اینجا پیش‌فرض کتابخانه‌ای بود نه پیکربندی قهرمان — ناسازگاری
#: بی‌صدایی که نام «قهرمان فاز ۷» پنهانش می‌کرد. `src/models/tune_phase13.py` هر سه
#: گزینه را سنجید و نتیجه (F136) این بود که **همین مقادیر** روی هر سه تقسیم بهترین‌اند،
#: حتی بهتر از تنظیم تازه روی خودِ پنجره‌ی جدید. ردیف ۸۱ `decision_log`.
CHAMPION_PARAMS = {
    "num_leaves": 19, "learning_rate": 0.10511867551641824, "min_child_samples": 89,
    "feature_fraction": 0.922812809422756, "n_estimators": 101,
}


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


def lightgbm_quantile(train, test, tau, seed: int = 42, feature_set_name: str = "full"):
    """قهرمان فاز ۷ (`lightgbm_quantile`)، بازآموزی‌شده روی پنجره‌ی تازه.

    وزن‌های فاز ۷ بارگذاری **نمی‌شوند**: آن‌ها روی فیچرست و شبکه‌ی پنجره‌ی قفل‌شده
    برازش شده‌اند و ماتریس v4 ستون‌های دیگری دارد. پرسش بند ۱۳-۹ هم همین است —
    «آیا این **معماری** روی سال بعد کار می‌کند؟» نه «آیا این فایل وزن کار می‌کند؟».
    """
    import lightgbm as lgb

    cols = [c for c in feature_set(feature_set_name) if c in train.columns]
    Xtr, Xte = _design(train[cols]), _design(test[cols])
    Xte = Xte.reindex(columns=Xtr.columns, fill_value=np.nan)

    model = lgb.LGBMRegressor(
        objective="quantile", alpha=tau, verbosity=-1, random_state=seed,
        **CHAMPION_PARAMS,
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


# ---------------------------------------------------------------------------
# نقطه‌ی عملیاتی و سهم فیچر ترکیب
# ---------------------------------------------------------------------------

#: نسبت‌های هزینه‌ای که حساسیت روی آن‌ها سنجیده می‌شود. ۴ برآورد ذی‌نفع است
#: ($C_u \approx 4C_o \Rightarrow \tau = C_o/(C_u+C_o) = 0.20$)؛ ۳ و ۵ کران‌های
#: معقولِ همان برآوردند و کارشان نشان‌دادن این است که تصمیم به عدد دقیق وابسته نباشد.
COST_RATIOS = (3.0, 4.0, 5.0)


def tau_optimality(ratios=COST_RATIOS) -> pd.DataFrame:
    r"""هزینه‌ی نسبی هر $\tau$: $\frac{C_u}{C_o}\cdot\text{کمبود} + \text{مازاد}$.

    پرسش صریح بند ۱۳-۹: «آیا $\tau=0.20$ با داده‌ی تازه هنوز درست است؟» — که فقط با
    **منحنی** پاسخ می‌گیرد نه با یک نقطه، چون اگر کمینه تخت باشد، «کمینه بودن» خبر
    کم‌ارزشی است. هزینه در فضای **پرس** حساب می‌شود نه نرخ (بند ۶.۴).
    """
    results = pd.read_csv(OUTPUT_DIR / "oos_results.csv")
    champ = results[results.model == "lightgbm_quantile"]
    rows = []
    for ratio in ratios:
        for split_name, block in champ.groupby("split"):
            cost = ratio * block["shortage_portions"] + block["surplus_portions"]
            best_tau = float(block.loc[cost.idxmin(), "tau"])
            at_020 = float(cost[block["tau"].round(4) == 0.20].iloc[0])
            rows.append({
                "cost_ratio": ratio, "split": split_name,
                "best_tau": best_tau, "cost_at_best": float(cost.min()),
                "cost_at_020": at_020,
                "excess_of_020_%": 100 * (at_020 / cost.min() - 1),
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "tau_optimality.csv", index=False)
    return out


def composition_ablation(tau: float = TAU_OPERATIONAL, n_boot: int = 500) -> pd.DataFrame:
    """سهم خانواده‌ی فیچر ترکیب — همان قهرمان، با و بدون آن.

    با CI بلوکی می‌آید چون بهبود کوچک است (~۱-۲٪) و بدون فاصله‌ی اطمینان، تفکیکش از
    نویز بذر ممکن نیست.
    """
    from src.baselines import pinball_loss
    from src.cv import block_bootstrap_2d

    df = pd.read_parquet(FEATURES_PATH)
    splits = build_splits(df)
    rows = []
    for split_name, split in splits.items():
        test = split["test"].reset_index(drop=True)
        y = test["rho"].to_numpy(float)
        preds = {
            fs: lightgbm_quantile(split["train"], test, tau, feature_set_name=fs)
            for fs in ("full", "no_composition")
        }
        work = test[["date_gregorian", "restaurant"]].copy()
        work["_d"] = (pinball_loss(y, preds["full"], tau)
                      - pinball_loss(y, preds["no_composition"], tau))
        base = float(pinball_loss(y, preds["no_composition"], tau).mean())
        delta, lo, hi = block_bootstrap_2d(
            work, lambda s: s["_d"].mean(), day_col="date_gregorian",
            unit_col="restaurant", n_boot=n_boot,
        )
        rows.append({"split": split_name, "tau": tau,
                     "pinball_no_composition": base,
                     "pinball_full": float(pinball_loss(y, preds["full"], tau).mean()),
                     "delta": delta, "ci_low": lo, "ci_high": hi,
                     "delta_pct": 100 * delta / base, "significant": bool(hi < 0)})
        logger.info("ترکیب · %s → Δ=%.6f [%.6f, %.6f]", split_name, delta, lo, hi)
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "composition_ablation.csv", index=False)
    return out


# ---------------------------------------------------------------------------
# معناداری
# ---------------------------------------------------------------------------


def significance(tau: float = TAU_OPERATIONAL, n_boot: int = 500) -> pd.DataFrame:
    """برتری قهرمان بر B3، با بوت‌استرپ بلوکیِ **دوبعدی** (روز × سلف).

    ⚠️ نسخه‌ی نخست این گزارش بوت‌استرپ سطر‌به‌سطر زده بود. استاندارد خودِ پروژه آن را
    رد می‌کند: ICC(روز)=۰.۲۲۵ و ICC(سلف)=۰.۲۰۵ (F10)، پس سطرهای یک روز مستقل نیستند
    و CI سطری **باریک‌تر از واقعیت** است. `src/cv.py::block_bootstrap_2d` همان ابزاری
    است که فاز ۶ برای همین ساخته بود؛ اینجا هم همان استفاده می‌شود.
    """
    from src.baselines import pinball_loss
    from src.cv import block_bootstrap_2d

    df = pd.read_parquet(FEATURES_PATH)
    splits = build_splits(df)

    rows = []
    for split_name, split in splits.items():
        test = split["test"].reset_index(drop=True)
        champ = MODELS["lightgbm_quantile"](split["train"], test, tau)
        base = MODELS["B3_empirical_quantile"](split["train"], test, tau)
        y = test["rho"].to_numpy(float)
        work = test[["date_gregorian", "restaurant"]].copy()
        work["_d"] = pinball_loss(y, champ, tau) - pinball_loss(y, base, tau)
        work["_b"] = pinball_loss(y, base, tau)

        delta, lo, hi = block_bootstrap_2d(
            work, lambda s: s["_d"].mean(), day_col="date_gregorian",
            unit_col="restaurant", n_boot=n_boot,
        )
        rows.append({
            "split": split_name, "tau": tau, "n_test": len(test),
            "delta_pinball": delta, "ci_low": lo, "ci_high": hi,
            "delta_pct": 100 * delta / work["_b"].mean(),
            "significant": bool(hi < 0),
        })
        logger.info("%s → Δ=%.6f [%.6f, %.6f] %s", split_name, delta, lo, hi,
                    "معنادار" if hi < 0 else "نامعنادار")

    out = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_DIR / "oos_significance.csv", index=False)
    return out


# ---------------------------------------------------------------------------
# حساسیت به پنجره‌ی جنگ
# ---------------------------------------------------------------------------

#: ماتریسی که پنجره‌ی شوک را **نگه داشته** — فقط برای همین سنجش ساخته می‌شود
#: (`build_features(exclude_anomalies=False)`).
ANOMALY_FEATURES_PATH = DATA_PROCESSED / "features_L1_v4_with_anomalies.parquet"


def war_window_sensitivity(tau: float = TAU_OPERATIONAL) -> pd.DataFrame:
    """همان مدل، همان آموزش — فقط با و بدون ۱۳ روز جنگ در مجموعه‌ی **آزمون**.

    **چرا نباید فقط عدد تمیز گزارش شود.** حذف یک بازه از آزمون، عملکرد را بهتر
    نشان می‌دهد؛ اگر فقط عدد بعد از حذف بیاید، خواننده راهی برای سنجیدن اثر حذف
    ندارد. پس هر دو در یک جدول می‌آیند. آموزش در هر دو حالت یکسان است — سال
    ۱۴۰۲-۱۴۰۳ اصلاً جنگ ندارد — پس تفاوت **فقط** ترکیب مجموعه‌ی آزمون است.
    """
    from src.features.build_v4 import ANOMALY_WINDOWS

    if not ANOMALY_FEATURES_PATH.exists():
        logger.warning("ماتریسِ با-شوک ساخته نشده (%s) — سنجش رد شد", ANOMALY_FEATURES_PATH)
        return pd.DataFrame()

    war_start = pd.Timestamp(ANOMALY_WINDOWS[0][0])
    df = pd.read_parquet(ANOMALY_FEATURES_PATH)
    splits = build_splits(df)

    rows = []
    for split_name, split in splits.items():
        train, test = split["train"], split["test"]
        in_war = test["date_gregorian"].between(
            war_start, pd.Timestamp(ANOMALY_WINDOWS[0][1])
        ).to_numpy()
        if not in_war.any():
            continue
        for model_name, fn in MODELS.items():
            if model_name == "B0_cook_all":
                continue
            pred = fn(train, test, tau)
            for variant, mask in (("with_war", np.ones(len(test), bool)),
                                  ("without_war", ~in_war)):
                m = operational_metrics(test[mask], pred[mask], tau)
                rows.append({"split": split_name, "model": model_name, "variant": variant,
                             "n_test": int(mask.sum()), "tau": tau,
                             "pinball": m["pinball"], "coverage": m["coverage"],
                             "shortage_rate": m["shortage_rate"]})
    out = pd.DataFrame(rows)
    if len(out):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out.to_csv(OUTPUT_DIR / "war_window_sensitivity.csv", index=False)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    set_global_seed()
    results = run()

    sens = war_window_sensitivity()
    if len(sens):
        pd.set_option("display.width", 200)
        print("\n=== حساسیت به ۱۳ روز جنگ (همان آموزش، آزمونِ با/بدون) ===")
        print(sens.pivot_table(index=["split", "model"], columns="variant", values="pinball")
              .to_string(float_format=lambda v: f"{v:.5f}"))

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
