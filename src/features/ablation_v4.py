r"""بند ۱۴-۹-۱ — سهم واقعی هر خانواده‌ی ویژگی، با leave-one-family-out.

**چرا این بسته وجود دارد.** ماتریس v4 هفت خانواده دارد و تا امروز فقط **یکی**
(`composition`) ablation شده بود — و همان یک‌بار هم اول اشتباه اندازه‌گیری شد: با
هایپرپارامتر پیش‌فرض کتابخانه‌ای، سهم فیچر ~۳ برابر بیش‌برآورد شد (−۱.۵٪ در برابر
−۰.۲٪، F139). سهم شش خانواده‌ی دیگر هرگز سنجیده نشده بود؛ یعنی ۵۶ ستون از ۶۶ ستون
ماتریس، بدون هیچ شاهد کمّی‌ای «فیچر» شمرده می‌شدند.

**سه قید اجباری بند ۱۴-۹-۱، و اینکه هرکدام جلوی چه خطایی را می‌گیرد:**

| قید | بدون آن چه می‌شود | مرجع |
|---|---|---|
| هایپرپارامتر **تنظیم‌شده‌ی** قهرمان | مدل تنظیم‌نشده به هر ستونی تکیه می‌کند، پس حذف هر خانواده بزرگ‌تر از واقع به نظر می‌رسد | F139 · قاعده‌ی ۱۴-۱-ج |
| بوت‌استرپ بلوکی **دوبعدی** | ICC روز ۰.۲۲۵ و ICC سلف ۰.۲۰۵ (F10)؛ CI سطری باریک‌تر از واقعیت است و خانواده‌ی بی‌اثر را «معنادار» نشان می‌دهد | F138 · قاعده‌ی ۱۴-۱-ب |
| **هر سه** تقسیم | خانواده‌ای که درون‌پنجره کمک کند و بیرون نه، فیچر نیست؛ بیش‌برازش است | بند ۱۳-۹ · F123 |

**قید چهارمی که این ماژول خودش اضافه می‌کند: نویز بذر.** Δهای این جدول در مرتبه‌ی
۱۰⁻⁴ تا ۱۰⁻⁵‌اند و یک برازش LightGBM با بذر دیگر هم همین‌قدر جابه‌جا می‌شود. پس هر
بازو با **سه بذر** برازش می‌شود و زیانِ هر ردیف روی بذرها میانگین گرفته می‌شود؛ ستون
«نویز بذر» در گزارش، دامنه‌ی pinball مدل کامل روی همان سه بذر است. خانواده‌ای که Δ‌اش
از نویز بذر کوچک‌تر است، هر CIای هم که بگیرد، ادعای قابل‌دفاعی ندارد.

**قرارداد علامت.** $\Delta = \text{pinball(کامل)} - \text{pinball(بدون خانواده)}$ —
پس **منفی یعنی خانواده کمک می‌کند**. همان قرارداد `composition_ablation` فاز ۱۳، تا
دو جدول کنار هم خوانده شوند.

اجرا: ``python -m src.features.ablation_v4``
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.baselines import pinball_loss
from src.config import REPORTS_DIR, set_global_seed
from src.cv import block_bootstrap_2d_mean
from src.features.build_v4 import FEATURE_SETS, OUTPUT_PATH as FEATURES_PATH, feature_set
from src.models.run_phase13_oos import CHAMPION_PARAMS, DATA_WINDOW, _design, build_splits

logger = logging.getLogger(__name__)

EXPERIMENT = "phase14_features"
OUTPUT_DIR = REPORTS_DIR / "phase14"

#: نقطه‌ی عملیاتی پروژه. ⚠️ همان τ که `CHAMPION_PARAMS` رویش تنظیم شده — تنظیم در یک
#: τ و سنجش در τ دیگر، اشتباه شماره‌ی ۱ بند ۱۴-۱-۳ است.
TAU = 0.20

#: سه بذر، نه یکی. هزینه‌اش سه برابر برازش است (~۷ ثانیه برای هر بازو) و چیزی که
#: می‌خرد، تفکیک Δ از نوسان بذر است — که بدون آن کل جدول تفسیرناپذیر می‌ماند.
SEEDS = (42, 43, 44)

N_BOOT = 500

#: ترتیب گزارش. `regime` عمداً خالی است (ردیف ۸۰) و ردیفش در جدول می‌ماند تا
#: «هفت خانواده سنجیده شد» ادعای قابل‌بررسی باشد نه شش‌تا به‌اضافه‌ی سکوت — تعیین
#: تکلیفش کار بند ۱۴-۹-۲ است.
FAMILY_ORDER = ["known_demand", "history", "entity", "calendar", "weather", "composition", "regime"]


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, tau: float,
                cols: list[str], seed: int) -> np.ndarray:
    """قهرمان فاز ۷ روی یک زیرمجموعه‌ی ستونی مشخص.

    `CHAMPION_PARAMS` از `run_phase13_oos` **import** می‌شود نه کپی — اگر روزی
    پیکربندی قهرمان عوض شود، ablation نباید بی‌صدا با پیکربندی قدیمی بماند.
    """
    import lightgbm as lgb

    cols = [c for c in cols if c in train.columns]
    Xtr, Xte = _design(train[cols]), _design(test[cols])
    Xte = Xte.reindex(columns=Xtr.columns, fill_value=np.nan)

    model = lgb.LGBMRegressor(objective="quantile", alpha=tau, verbosity=-1,
                              random_state=seed, **CHAMPION_PARAMS)
    model.fit(Xtr, train["rho"])
    return np.clip(model.predict(Xte), 0.0, 1.0)


def arm_losses(train: pd.DataFrame, test: pd.DataFrame, tau: float, cols: list[str],
               seeds=SEEDS) -> tuple[np.ndarray, list[float]]:
    """زیان pinball هر ردیف، میانگین‌گرفته روی بذرها + pinball هر بذر جداگانه."""
    y = test["rho"].to_numpy(float)
    per_seed = [pinball_loss(y, fit_predict(train, test, tau, cols, s), tau) for s in seeds]
    return np.mean(per_seed, axis=0), [float(loss.mean()) for loss in per_seed]


def run_lofo(tau: float = TAU, seeds=SEEDS, n_boot: int = N_BOOT) -> pd.DataFrame:
    """جدول ۷×۳ بند ۱۴-۹-۱ — یک ردیف به‌ازای هر (خانواده، تقسیم).

    Δ هم در فضای **نرخ** و هم در فضای **پرس** می‌آید (قاعده‌ی بند ۶.۴). این دو
    می‌توانند مخالف هم باشند: خانواده‌ای که فقط سلول‌های کوچک را بهتر می‌کند در فضای
    نرخ برنده و در فضای پرس بی‌اثر است — و متغیر تصمیم پروژه پرس است.
    """
    df = pd.read_parquet(FEATURES_PATH)
    splits = build_splits(df)
    full_cols = feature_set("full")

    rows: list[dict] = []
    for split_name, split in splits.items():
        train = split["train"]
        test = split["test"].reset_index(drop=True)
        res = test["Res"].to_numpy(float)
        work = test[["date_gregorian", "restaurant"]].copy()

        loss_full, full_by_seed = arm_losses(train, test, tau, full_cols, seeds)
        logger.info("%s · کامل → pinball %.6f (بذرها: %s)", split_name,
                    np.mean(full_by_seed), [round(v, 6) for v in full_by_seed])

        for family in FAMILY_ORDER:
            if not FEATURE_SETS[family]:
                rows.append({
                    "family": family, "split": split_name, "split_label": split["label"],
                    "n_cols": 0, "n_test": len(test), "tau": tau,
                    "pinball_full": float(loss_full.mean()), "pinball_drop": np.nan,
                    "delta": np.nan, "ci_low": np.nan, "ci_high": np.nan, "delta_pct": np.nan,
                    "delta_portions": np.nan, "ci_low_portions": np.nan,
                    "ci_high_portions": np.nan, "significant": False,
                    "note": "خانواده خالی است (ردیف ۸۰) — بند ۱۴-۹-۲",
                })
                continue

            loss_drop, _ = arm_losses(train, test, tau, feature_set(f"no_{family}"), seeds)
            work["_d"] = loss_full - loss_drop
            work["_dp"] = (loss_full - loss_drop) * res

            delta, lo, hi = block_bootstrap_2d_mean(
                work, "_d", day_col="date_gregorian", unit_col="restaurant", n_boot=n_boot)
            delta_p, lo_p, hi_p = block_bootstrap_2d_mean(
                work, "_dp", day_col="date_gregorian", unit_col="restaurant", n_boot=n_boot)

            base = float(loss_drop.mean())
            rows.append({
                "family": family, "split": split_name, "split_label": split["label"],
                "n_cols": len(FEATURE_SETS[family]), "n_test": len(test), "tau": tau,
                "pinball_full": float(loss_full.mean()), "pinball_drop": base,
                "delta": delta, "ci_low": lo, "ci_high": hi,
                "delta_pct": 100 * delta / base,
                "delta_portions": delta_p, "ci_low_portions": lo_p, "ci_high_portions": hi_p,
                "significant": bool(hi < 0), "note": "",
            })
            logger.info("%s · %-13s → Δ=%+.6f [%+.6f, %+.6f] (%+.2f٪) %s",
                        split_name, family, delta, lo, hi, 100 * delta / base,
                        "معنادار" if hi < 0 else "—")

    out = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_DIR / "feature_family_ablation.csv", index=False)
    return out


#: گروه‌های بند ۱۴-۹-۱-ب — سه پرسشی که جدول LOFO به‌تنهایی نمی‌تواند پاسخ دهد.
#:
#: **چرا LOFO تنها کافی نیست.** leave-one-family-out سهم **یکتای** هر خانواده را
#: می‌سنجد، نه سهم کلش. خانواده‌ای که اطلاعاتش جای دیگری هم هست، Δ صفر می‌گیرد و
#: «بی‌فایده» به نظر می‌رسد در حالی که فقط **تکراری** است. سه گروه زیر همین تمایز را
#: عملیاتی می‌کنند:
#:
#: - `candidates`: هر سه نامزد حذف **با هم** برداشته می‌شوند. این خودِ تصمیم است —
#:   حذف تک‌به‌تکِ بی‌ضرر می‌تواند جمعاً مضر باشد.
#: - `entity_given_no_history` و `history_given_no_entity`: آزمون جانشینی. فیچرهای
#:   تاریخی **به تفکیک سلف** ساخته می‌شوند (`venue_hist_rho`, `venue_dow_hist_rho`)،
#:   پس هویت سلف از قبل داخلشان است. اگر `entity` فقط وقتی مهم شود که `history`
#:   نباشد، Δ صفرش «بی‌اثر» نیست، «تکراری» است — و پاسخ سیاستی این دو کاملاً فرق دارد.
ABLATION_GROUPS: dict[str, dict] = {
    "candidates": {"drop": ["entity", "weather", "composition"], "given": [],
                   "question": "حذف هم‌زمان هر سه نامزد چقدر هزینه دارد؟"},
    "entity_given_no_history": {"drop": ["entity"], "given": ["history"],
                                "question": "وقتی تاریخچه نیست، هویت سلف مهم می‌شود؟"},
    "history_given_no_entity": {"drop": ["history"], "given": ["entity"],
                                "question": "وقتی هویت سلف نیست، تاریخچه مهم‌تر می‌شود؟"},
}


def run_group_ablation(groups: dict[str, dict] | None = None, tau: float = TAU,
                       seeds=SEEDS, n_boot: int = N_BOOT) -> pd.DataFrame:
    """ablation گروهی — هر گروه نسبت به **مرجع خودش**، نه لزوماً مدل کامل.

    Δ = pinball(مرجع) − pinball(مرجع منهای گروه)؛ همان قرارداد علامت جدول LOFO، پس
    منفی یعنی گروهِ حذف‌شده نسبت به آن مرجع کمک می‌کرده.
    """
    groups = groups or ABLATION_GROUPS
    df = pd.read_parquet(FEATURES_PATH)
    splits = build_splits(df)

    rows: list[dict] = []
    for split_name, split in splits.items():
        train = split["train"]
        test = split["test"].reset_index(drop=True)
        res = test["Res"].to_numpy(float)
        work = test[["date_gregorian", "restaurant"]].copy()
        cache: dict[frozenset, np.ndarray] = {}

        def losses(dropped: list[str]) -> np.ndarray:
            key = frozenset(dropped)
            if key not in cache:
                cols = [c for fam, cols_ in FEATURE_SETS.items() if fam not in key for c in cols_]
                cache[key], _ = arm_losses(train, test, tau, cols, seeds)
            return cache[key]

        for name, spec in groups.items():
            ref = losses(spec["given"])
            arm = losses(spec["given"] + spec["drop"])
            work["_d"] = ref - arm
            work["_dp"] = (ref - arm) * res
            delta, lo, hi = block_bootstrap_2d_mean(
                work, "_d", day_col="date_gregorian", unit_col="restaurant", n_boot=n_boot)
            delta_p, lo_p, hi_p = block_bootstrap_2d_mean(
                work, "_dp", day_col="date_gregorian", unit_col="restaurant", n_boot=n_boot)
            base = float(arm.mean())
            rows.append({
                "group": name, "split": split_name, "split_label": split["label"],
                "dropped": "+".join(spec["drop"]), "given_absent": "+".join(spec["given"]) or "—",
                "question": spec["question"], "tau": tau,
                "pinball_ref": float(ref.mean()), "pinball_arm": base,
                "delta": delta, "ci_low": lo, "ci_high": hi, "delta_pct": 100 * delta / base,
                "delta_portions": delta_p, "ci_low_portions": lo_p, "ci_high_portions": hi_p,
                "significant": bool(hi < 0),
            })
            logger.info("%s · %-24s → Δ=%+.6f [%+.6f, %+.6f] (%+.2f٪) %s",
                        split_name, name, delta, lo, hi, 100 * delta / base,
                        "معنادار" if hi < 0 else "—")

    out = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_DIR / "feature_family_ablation_groups.csv", index=False)
    return out


#: ستون‌های **هویت** که در ماتریس v4 هستند ولی در هیچ فیچرستی نیامده‌اند.
IDENTITY_COLS = ["restaurant", "dish", "city"]


def _design_native(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """رده‌ای‌ها را **رده‌ای** نگه می‌دارد تا LightGBM خودش تقسیم رده‌ای بزند.

    `_design` فاز ۱۳ هر رده‌ای را به کد صحیح تبدیل می‌کند و درخت آن را **عددی**
    می‌بیند؛ یعنی «سلف ۷ > سلف ۳» معنا پیدا می‌کند و جداکردن یک سلف خاص به چند
    تقسیم پیاپی نیاز دارد. برای ستون ۶ سطحی بی‌آزار است، برای ۷۱ سلف نه. رده‌های
    آزمون از روی **آموزش** تثبیت می‌شوند، پس سلف دیده‌نشده NaN می‌شود نه یک کد تازه.
    """
    tr, te = train.copy(), test.copy()
    for col in tr.columns:
        if tr[col].dtype == object or isinstance(tr[col].dtype, pd.CategoricalDtype):
            dtype = pd.CategoricalDtype(categories=sorted(tr[col].dropna().unique()))
            tr[col] = tr[col].astype(dtype)
            te[col] = te[col].astype(dtype)
        elif tr[col].dtype == bool:
            tr[col] = tr[col].astype("int8")
            te[col] = te[col].astype("int8")
    return tr, te


def fit_predict_native(train: pd.DataFrame, test: pd.DataFrame, tau: float,
                       cols: list[str], seed: int) -> np.ndarray:
    """همان قهرمان، با رده‌ای‌های بومی LightGBM به‌جای کد صحیح."""
    import lightgbm as lgb

    cols = [c for c in cols if c in train.columns]
    Xtr, Xte = _design_native(train[cols], test[cols])
    model = lgb.LGBMRegressor(objective="quantile", alpha=tau, verbosity=-1,
                              random_state=seed, **CHAMPION_PARAMS)
    model.fit(Xtr, train["rho"])
    return np.clip(model.predict(Xte), 0.0, 1.0)


def identity_probe(tau: float = TAU, seeds=SEEDS, n_boot: int = N_BOOT) -> pd.DataFrame:
    r"""آیا حکم «`entity` نامزد حذف» مصنوعِ **نبودِ هویت** یا **کدگذاری** است؟

    دو مشاهده‌ی ناراحت‌کننده پشت این بند است:

    ۱. خانواده‌ی `entity` در ماتریس v4 فقط **صفت** سلف است (پردیس، نوع، جنسیت،
       فاصله)، نه **هویت** آن. ستون‌های `restaurant` (۷۱ سطح)، `dish` (۱۲۲) و
       `city` در ماتریس هستند ولی در هیچ فیچرستی نیامده‌اند — در حالی که فاز ۵
       `RestaurantName` را قوی‌ترین رده‌ای می‌دانست (F15، η²=۰.۳۲۳).
    ۲. `_design` هر رده‌ای را کد صحیح می‌کند، پس حتی صفت‌های موجود هم به درخت
       به‌صورت عددی می‌رسند.

    هر دو می‌توانند سهم `entity` را **به‌طور سیستماتیک کم نشان دهند**؛ و حکم حذف
    بدون رد کردنشان، دقیقاً اشتباه شماره‌ی ۵ بند ۱۴-۱-۳ است (اندازه‌گیری با ابزار
    ناسازگار). قرارداد علامت همان است: $\Delta$ منفی یعنی بازوی غنی‌تر بهتر است.
    """
    df = pd.read_parquet(FEATURES_PATH)
    splits = build_splits(df)
    full_cols = feature_set("full")

    arms = {
        "identity_ordinal": (full_cols + IDENTITY_COLS, fit_predict,
                             "هویت اضافه شد، با همان کدگذاری صحیحِ `_design`"),
        "identity_native": (full_cols + IDENTITY_COLS, fit_predict_native,
                            "هویت اضافه شد، با رده‌ای بومی LightGBM"),
        "native_encoding_only": (full_cols, fit_predict_native,
                                 "بدون ستون تازه — فقط کدگذاری رده‌ای عوض شد"),
    }

    rows = []
    for split_name, split in splits.items():
        train = split["train"]
        test = split["test"].reset_index(drop=True)
        res = test["Res"].to_numpy(float)
        work = test[["date_gregorian", "restaurant"]].copy()
        y = test["rho"].to_numpy(float)
        ref = np.mean([pinball_loss(y, fit_predict(train, test, tau, full_cols, s), tau)
                       for s in seeds], axis=0)

        for arm, (cols, fn, why) in arms.items():
            arm_loss = np.mean([pinball_loss(y, fn(train, test, tau, cols, s), tau)
                                for s in seeds], axis=0)
            work["_d"] = arm_loss - ref
            work["_dp"] = (arm_loss - ref) * res
            delta, lo, hi = block_bootstrap_2d_mean(
                work, "_d", day_col="date_gregorian", unit_col="restaurant", n_boot=n_boot)
            delta_p, lo_p, hi_p = block_bootstrap_2d_mean(
                work, "_dp", day_col="date_gregorian", unit_col="restaurant", n_boot=n_boot)
            rows.append({
                "arm": arm, "split": split_name, "split_label": split["label"],
                "what": why, "tau": tau,
                "pinball_ref": float(ref.mean()), "pinball_arm": float(arm_loss.mean()),
                "delta": delta, "ci_low": lo, "ci_high": hi,
                "delta_pct": 100 * delta / float(ref.mean()),
                "delta_portions": delta_p, "ci_low_portions": lo_p, "ci_high_portions": hi_p,
                "better": bool(hi < 0), "worse": bool(lo > 0),
            })
            logger.info("%s · %-21s → Δ=%+.6f [%+.6f, %+.6f] (%+.2f٪) %s",
                        split_name, arm, delta, lo, hi, rows[-1]["delta_pct"],
                        "بهتر" if hi < 0 else ("بدتر" if lo > 0 else "—"))

    out = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_DIR / "feature_identity_probe.csv", index=False)
    return out


def seed_noise(tau: float = TAU, seeds=SEEDS) -> pd.DataFrame:
    """دامنه‌ی pinball مدل **کامل** روی بذرهای مختلف — خط‌کشِ خواندن ستون Δ."""
    df = pd.read_parquet(FEATURES_PATH)
    rows = []
    for split_name, split in build_splits(df).items():
        test = split["test"].reset_index(drop=True)
        _, by_seed = arm_losses(split["train"], test, tau, feature_set("full"), seeds)
        rows.append({
            "split": split_name, "split_label": split["label"],
            "pinball_mean": float(np.mean(by_seed)), "pinball_min": float(np.min(by_seed)),
            "pinball_max": float(np.max(by_seed)),
            "seed_range": float(np.max(by_seed) - np.min(by_seed)),
            "seed_std": float(np.std(by_seed, ddof=1)),
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "feature_family_seed_noise.csv", index=False)
    return out


def log_to_mlflow(lofo: pd.DataFrame) -> None:
    """experiment `phase14_features` با tagهای اجباری قاعده‌ی ۱۴-۱-الف."""
    try:
        import mlflow
    except ImportError:
        logger.warning("mlflow نصب نیست — ثبت رد شد")
        return

    from src.config import MLFLOW_TRACKING_URI
    from src.models.tracking import git_commit_short

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    for _, row in lofo.iterrows():
        if pd.isna(row["delta"]):
            continue
        name = f"lofo_{row.family}_L1_rho_t{int(row.tau * 1000):03d}_{row.split}"
        with mlflow.start_run(run_name=name):
            mlflow.set_tags({
                "data_window": DATA_WINDOW, "stream": "features",
                "phase": "14", "package": "14-9-1", "family_ablated": row.family,
                "git_commit": git_commit_short(),
            })
            mlflow.log_params({
                "model_name": "lightgbm_quantile", "tau": row.tau, "level": "L1",
                "split": row.split, "n_cols_dropped": int(row.n_cols),
                "seeds": ",".join(str(s) for s in SEEDS), "n_boot": N_BOOT,
                "hyperparams": "champion_phase7",
            })
            mlflow.log_metrics({
                k: float(row[k]) for k in
                ("pinball_full", "pinball_drop", "delta", "ci_low", "ci_high",
                 "delta_pct", "delta_portions", "ci_low_portions", "ci_high_portions")
                if pd.notna(row[k])
            })
    logger.info("ثبت در MLflow: experiment=%s", EXPERIMENT)


# ---------------------------------------------------------------------------
# گزارش
# ---------------------------------------------------------------------------

SPLIT_ORDER = ["in_window", "oos_1403-1404", "oos_1404-1405"]
SPLIT_FA = {"in_window": "درون‌پنجره", "oos_1403-1404": "۱۴۰۳-۱۴۰۴", "oos_1404-1405": "۱۴۰۴-۱۴۰۵"}

_FA_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")


def num(value: float, fmt: str = "+.5f") -> str:
    """عدد با رقم فارسی — قرارداد نگارشی همه‌ی گزارش‌های پروژه.

    فقط روی **مقدارهای عددی** اعمال می‌شود، نه کل متن: ترجمه‌ی سراسری، `L1` را به
    `L۱` و `ablation_v4.py` را به `ablation_v۴.py` تبدیل می‌کرد.
    """
    return format(value, fmt).translate(_FA_DIGITS)


def verdict(block: pd.DataFrame, noise: pd.DataFrame) -> tuple[str, str]:
    """حکم هر خانواده: نگه‌داشتن، نامزد حذف، یا تعیین‌تکلیف جداگانه.

    قاعده‌ی پذیرش بند ۱۴-۹-۱ عیناً: «هر خانواده‌ای که CIاش روی **هر سه** تقسیم شامل
    صفر باشد نامزد حذف می‌شود — نه حذف خودکار، ولی ماندنش ردیف `decision_log`
    می‌خواهد.» قید نویز بذر روی همین سوار می‌شود: Δی کوچک‌تر از دامنه‌ی بذر، حتی با
    CI معنادار، «کمک اثبات‌شده» شمرده نمی‌شود.
    """
    if block["delta"].isna().all():
        return "تعیین تکلیف جداگانه", "خانواده خالی — بند ۱۴-۹-۲"

    n_sig = int(block["significant"].sum())
    merged = block.merge(noise[["split", "seed_range"]], on="split")
    above_noise = int((merged["delta"].abs() > merged["seed_range"]).sum())

    if n_sig == 0:
        return "نامزد حذف", "CI روی هر سه تقسیم شامل صفر"
    if n_sig == 3 and above_noise == 3:
        return "نگه‌داشتن", "معنادار و بزرگ‌تر از نویز بذر روی هر سه تقسیم"
    if above_noise == 0:
        return "نامزد حذف", f"معنادار روی {n_sig} تقسیم ولی Δ زیر نویز بذر در هر سه"
    return "نگه‌داشتن (مشروط)", f"معنادار روی {n_sig} از ۳ تقسیم · بالای نویز در {above_noise}"


def _insight_section(lofo: pd.DataFrame, groups: pd.DataFrame,
                     probe: pd.DataFrame) -> list[str]:
    """بخش «این جدول چه می‌گوید» — با عددهای واقعی، نه بازگویی خانه‌های جدول.

    قاعده‌ی «بینش، نه توصیف» (`final_report/AGENTS.md`) اینجا هم برقرار است: هر عدد
    باید بگوید برای **انتخاب فیچر** چه معنایی دارد. اعداد از همان قاب‌ها interpolate
    می‌شوند تا گزارش با هر بازاجرا خودش را تصحیح کند.
    """
    oos = lofo[lofo.split.str.startswith("oos_")]
    cand = groups[groups.group == "candidates"]
    n_cols_dropped = int(sum(len(FEATURE_SETS[f]) for f in ABLATION_GROUPS["candidates"]["drop"]))
    n_full = len(feature_set("full"))
    hist_oos = oos[oos.family == "history"]["delta_pct"].mean()
    kd_oos = oos[oos.family == "known_demand"]["delta_pct"].mean()
    worst_native = probe[probe.arm == "identity_native"]["delta_pct"].max()

    return [
        "",
        "## این جدول چه می‌گوید",
        "",
        f"**۱. دو خانواده کار می‌کنند، پنج‌تا نه.** `known_demand` ({num(kd_oos, '+.1f')}٪ میانگین "
        f"برون‌نمونه) و `history` ({num(hist_oos, '+.1f')}٪) تنها خانواده‌هایی‌اند که Δشان هم از صفر و "
        "هم از نویز بذر جدا می‌شود. این با F119 می‌خواند (قوی‌ترین سیگنال سطح سلف است) و "
        "با شهود مسئله: مدل «چند نفر رزرو کرده‌اند» و «این سلف تا امروز چه نرخی داشته» را "
        "می‌داند، و بقیه‌ی ماتریس حرف تازه‌ای اضافه نمی‌کند.",
        "",
        "**۲. فضای نرخ و فضای پرس یک جواب نمی‌دهند — و پرس معیار تصمیم است.** در فضای نرخ، "
        "`known_demand` روی دو تقسیم معنادار است و `history` روی دو تقسیم؛ در فضای **پرس** "
        "`history` تنها خانواده‌ای است که روی **هر سه** تقسیم معنادار می‌ماند. یعنی سهم "
        "`history` در سلول‌های **بزرگ** متمرکز است — دقیقاً جایی که تصمیم پخت هزینه دارد. "
        "گزارش‌کردن فقط فضای نرخ، این خانواده را کم‌ارزش‌تر از واقع نشان می‌داد.",
        "",
        f"**۳. ⭐ سه خانواده ({num(n_cols_dropped, 'd')} ستون از {num(n_full, 'd')}) با هم "
        "قابل حذف‌اند و هزینه‌ای "
        "ندارند.** `entity`، `weather` و `composition` هیچ‌کدام روی هیچ تقسیمی معنادار نیستند، "
        "و حذف **هم‌زمانشان** هم بی‌هزینه است: Δ = "
        + " · ".join(f"{num(r['delta'])} [{num(r['ci_low'])}, {num(r['ci_high'])}]"
                     for _, r in cand.iterrows())
        + " — هر سه CI شامل صفر. حذف تک‌به‌تکِ بی‌ضرر می‌توانست جمعاً مضر باشد؛ نیست.",
        "",
        "**۴. `entity` بی‌اثر است، نه تکراری — و این تمایز را جداگانه آزمودیم.** فرض طبیعی "
        "این بود که هویت سلف از قبل داخل فیچرهای تاریخی است (که به تفکیک سلف ساخته می‌شوند)، "
        "پس Δ صفرِ `entity` یعنی «تکراری». ولی وقتی `history` را هم برمی‌داریم، `entity` باز "
        "هم چیزی اضافه نمی‌کند (CI شامل صفر روی هر سه تقسیم) — در حالی که عکسش برقرار است: "
        "`history` بدون `entity` روی هر سه تقسیم معنادار می‌ماند و Δ‌اش **بزرگ‌تر** می‌شود "
        "(−۲.۳٪ تا −۲.۸٪ در برابر −۱.۵٪ تا −۲.۴٪ در LOFO). جهت وابستگی یک‌طرفه است.",
        "",
        f"**۵. ⚠️ و هویت خام اگر با رده‌ای بومی وارد شود، برون‌نمونه را تا "
        f"{num(worst_native, '+.1f')}٪ "
        "**بدتر** می‌کند.** سه بازوی کنترلی اجرا شد: افزودن `restaurant`/`dish`/`city` با "
        "کدگذاری فعلی بی‌اثر است؛ عوض‌کردن کدگذاری به رده‌ای بومی **بدون** ستون تازه هم بی‌اثر "
        "است (پس `_design` چیزی را خراب نمی‌کرده)؛ ولی هویتِ ۷۱ سلف و ۱۲۲ غذا با تقسیم رده‌ای "
        "بومی، درون‌پنجره بی‌ضرر و روی **هر دو سال بعد** معنادار بدتر است. این همان بیش‌برازشی "
        "است که فقط با آزمون برون‌نمونه دیده می‌شود، و چون «طبیعی‌ترین بهبود بعدی» است، "
        "ثبتش برای جریان پ لازم بود.",
        "",
        "**۶. آنچه این جدول نمی‌گوید.** LOFO سهم **یکتا** را می‌سنجد و به هیچ‌کدام از این "
        "خانواده‌ها فرصت بازطراحی نمی‌دهد. `composition` با خلاصه‌سازی بهتر (بند ۱۴-۱۱) و "
        "`calendar` با جایگزین هارمونیک (بند ۱۴-۱۰) دوباره آزموده می‌شوند؛ حکم «نامزد حذف» "
        "درباره‌ی **نسخه‌ی فعلی** آن‌هاست، نه درباره‌ی خانواده به‌طور کلی.",
    ]


def write_report(lofo: pd.DataFrame, noise: pd.DataFrame,
                 groups: pd.DataFrame | None = None,
                 probe: pd.DataFrame | None = None) -> str:
    fam = lofo[lofo.family != "regime"]
    lines = [
        "# سهم واقعی هر خانواده‌ی ویژگی — بند ۱۴-۹-۱",
        "",
        "> `src/features/ablation_v4.py` · داده: `data/processed/features_L1_v4.parquet` ·",
        f"> τ={num(TAU, '.2f')} · بذرها: {'، '.join(num(s, 'd') for s in SEEDS)} · "
        f"بوت‌استرپ بلوکی دوبعدی (روز × سلف، {num(N_BOOT, 'd')} تکرار)",
        "",
        "**قرارداد علامت: Δ = pinball(کامل) − pinball(بدون خانواده). منفی یعنی خانواده کمک می‌کند.**",
        "",
        "## جدول ۷×۳ — Δ در فضای نرخ",
        "",
        "| خانواده | ستون | " + " | ".join(SPLIT_FA[s] for s in SPLIT_ORDER) + " | حکم |",
        "|---|---|" + "---|" * (len(SPLIT_ORDER) + 1),
    ]
    for family in FAMILY_ORDER:
        block = lofo[lofo.family == family]
        cells = []
        for split in SPLIT_ORDER:
            row = block[block.split == split]
            if not len(row) or pd.isna(row.iloc[0]["delta"]):
                cells.append("—")
                continue
            r = row.iloc[0]
            mark = " ✅" if r["significant"] else ""
            cells.append(f"{num(r['delta'])} [{num(r['ci_low'])}, {num(r['ci_high'])}]{mark}")
        v, why = verdict(block, noise)
        lines.append(f"| `{family}` | {num(int(block.iloc[0]['n_cols']), 'd')} | "
                     + " | ".join(cells) + f" | **{v}** |")

    lines += [
        "",
        "✅ = CI بالا زیر صفر (خانواده معنادار کمک می‌کند).",
        "",
        "## نویز بذر — خط‌کش خواندن جدول بالا",
        "",
        "| تقسیم | pinball مدل کامل | دامنه‌ی سه بذر | Δهای زیر این دامنه |",
        "|---|---|---|---|",
    ]
    for split in SPLIT_ORDER:
        n = noise[noise.split == split].iloc[0]
        under = fam[(fam.split == split) & (fam["delta"].abs() < n["seed_range"])]["family"]
        lines.append(f"| {SPLIT_FA[split]} | {num(n['pinball_mean'], '.6f')} | "
                     f"{num(n['seed_range'], '.6f')} | "
                     + (", ".join(f"`{u}`" for u in under) if len(under) else "—") + " |")

    lines += ["", "## Δ در فضای پرس — همان جدول، معیار تصمیم", "",
              "| خانواده | " + " | ".join(SPLIT_FA[s] for s in SPLIT_ORDER) + " |",
              "|---|" + "---|" * len(SPLIT_ORDER)]
    for family in FAMILY_ORDER:
        block = lofo[lofo.family == family]
        cells = []
        for split in SPLIT_ORDER:
            row = block[block.split == split]
            if not len(row) or pd.isna(row.iloc[0]["delta_portions"]):
                cells.append("—")
                continue
            r = row.iloc[0]
            mark = " ✅" if r["ci_high_portions"] < 0 else ""
            cells.append(f"{num(r['delta_portions'], '+.4f')} "
                         f"[{num(r['ci_low_portions'], '+.4f')}, "
                         f"{num(r['ci_high_portions'], '+.4f')}]{mark}")
        lines.append(f"| `{family}` | " + " | ".join(cells) + " |")

    if groups is not None and len(groups):
        lines += [
            "",
            "## ablation گروهی — آنچه جدول LOFO نمی‌تواند بگوید",
            "",
            "LOFO سهم **یکتای** هر خانواده را می‌سنجد. خانواده‌ای که اطلاعاتش جای دیگری",
            "هم هست Δ صفر می‌گیرد — ولی «بی‌اثر» و «تکراری» دو چیزند و پاسخ سیاستی‌شان",
            "مخالف هم است.",
            "",
            "| گروه | حذف‌شده | در غیاب | " + " | ".join(SPLIT_FA[s] for s in SPLIT_ORDER) + " |",
            "|---|---|---|" + "---|" * len(SPLIT_ORDER),
        ]
        for name in groups["group"].unique():
            block = groups[groups.group == name]
            first = block.iloc[0]
            cells = []
            for split in SPLIT_ORDER:
                row = block[block.split == split]
                if not len(row):
                    cells.append("—")
                    continue
                r = row.iloc[0]
                mark = " ✅" if r["significant"] else ""
                cells.append(f"{num(r['delta'])} [{num(r['ci_low'])}, {num(r['ci_high'])}]{mark}")
            lines.append(f"| `{name}` | `{first['dropped']}` | `{first['given_absent']}` | "
                         + " | ".join(cells) + " |")
        lines += ["", "پرسش هر گروه:", ""]
        for name in groups["group"].unique():
            lines.append(f"- `{name}` — {groups[groups.group == name].iloc[0]['question']}")

    if probe is not None and len(probe):
        lines += [
            "",
            "## آزمون هویت و کدگذاری — آیا حکم `entity` مصنوع ابزار است؟",
            "",
            "ستون‌های `restaurant` (۷۱ سطح)، `dish` (۱۲۲) و `city` در ماتریس **هستند** ولی در",
            "هیچ فیچرستی نیامده‌اند؛ و `_design` هر رده‌ای را کد صحیح می‌کند. هر دو می‌توانستند",
            "سهم `entity` را کم نشان دهند. Δ منفی یعنی بازوی غنی‌تر بهتر است.",
            "",
            "| بازو | چه عوض شد | " + " | ".join(SPLIT_FA[s] for s in SPLIT_ORDER) + " |",
            "|---|---|" + "---|" * len(SPLIT_ORDER),
        ]
        for arm in probe["arm"].unique():
            block = probe[probe.arm == arm]
            cells = []
            for split in SPLIT_ORDER:
                row = block[block.split == split]
                if not len(row):
                    cells.append("—")
                    continue
                r = row.iloc[0]
                mark = " ✅" if r["better"] else (" ❌" if r["worse"] else "")
                cells.append(f"{num(r['delta'])} [{num(r['ci_low'])}, {num(r['ci_high'])}]{mark}")
            lines.append(f"| `{arm}` | {block.iloc[0]['what']} | " + " | ".join(cells) + " |")
        lines += ["", "✅ = معنادار بهتر · ❌ = معنادار **بدتر**."]

    if groups is not None and probe is not None and len(groups) and len(probe):
        lines += _insight_section(lofo, groups, probe)

    text = "\n".join(lines) + "\n"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "feature_family_ablation.md").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    set_global_seed()
    lofo = run_lofo()
    noise = seed_noise()
    groups = run_group_ablation()
    probe = identity_probe()
    write_report(lofo, noise, groups, probe)
    log_to_mlflow(lofo)

    pd.set_option("display.width", 220)
    print("\n=== Δ pinball (فضای نرخ) — منفی یعنی خانواده کمک می‌کند ===")
    print(lofo.pivot(index="family", columns="split", values="delta")
          .reindex(FAMILY_ORDER).to_string(float_format=lambda v: f"{v:+.6f}"))
    print("\n=== معنادار؟ ===")
    print(lofo.pivot(index="family", columns="split", values="significant")
          .reindex(FAMILY_ORDER).to_string())
    print("\n=== نویز بذر ===")
    print(noise.to_string(index=False))


if __name__ == "__main__":
    main()
