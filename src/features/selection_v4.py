r"""بند ۱۴-۱۲ — انتخاب ویژگی، هم‌خطی و بودجه‌ی ستون.

**قاعده‌ی پذیرش این بند، بار اثبات را وارونه می‌کند.** تا اینجای جریان ب هر بند
می‌پرسید «آیا این ستون کمک می‌کند؟» و ستونِ بی‌اثر می‌ماند. بند ۱۴-۱۲-۲ می‌گوید:

> «هر ستون یا در یک فیچرست مدلی هست یا صریح کنار گذاشته شده. **ماند چون ضرری
> نداشت** دلیل پذیرفته‌شده‌ای نیست.»

پس این ماژول یک **سرشماری** است: هر ستونِ ماتریس و هر ستونِ نامزدی که جریان ب
ساخت، باید دقیقاً یک وضعیت داشته باشد، با دلیل و شناسه‌ی شاهد. شرط بسته‌شدن،
**صفر ستون بی‌وضعیت** است.

**سه قید هم‌خطی که از کاوش می‌آیند و اینجا اعمال می‌شوند:**

| قید | شاهد | اثر |
|---|---|---|
| `city` ≡ `campus` | F206 (شهر و پردیس یکی‌اند؛ F12 بازتولید نشد) | فقط یکی می‌ماند |
| سلسله‌مراتب مکان | بند ۱۴-۷-۲ | صفت‌های تودرتوی سلف هم‌زمان وارد نمی‌شوند |
| هارمونیک هفتگی ≡ `venue_dow_hist_rho` | **F291** | نگه‌داشتن هر دو انباشت است نه جایگزینی |

⚠️ **قید دامنه‌ی کل جریان ب، که همین‌جا باید صریح ثبت شود.** هر Δی که بندهای ۱۴-۹
تا ۱۴-۱۱ گزارش کردند با **یک مدل** اندازه‌گیری شد: قهرمان LightGBM. ارزش یک فیچر
به ذات مدل وابسته است — جدول 7.5.3 خودش می‌گوید مدل خطی پایه‌ی فوریه **لازم** دارد
در حالی که درخت همان اطلاعات را از شاخص صحیح روز هفته می‌سازد. پس «هارمونیک کمکی
نکرد» (F290) یعنی **برای درخت** کمکی نکرد؛ برای خانواده‌های خطی و کرنلی جریان پ
باید دوباره سنجیده شود. این محدودیت، نتیجه‌ها را باطل نمی‌کند ولی دامنه‌شان را
تعیین می‌کند.

اجرا: ``python -m src.features.selection_v4``
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import DOCS_DIR, REPORTS_DIR, set_global_seed
from src.features.build_v4 import (
    FEATURE_SETS, LEAN_FAMILIES, OUTPUT_PATH as FEATURES_PATH, feature_set,
)

logger = logging.getLogger(__name__)

OUTPUT_DIR = REPORTS_DIR / "phase14"

#: ستون‌هایی که با **پوشش** رد می‌شوند، نه با Δ.
#:
#: قاعده‌ی این بند اجازه نمی‌دهد ستونی چون «ضرری ندارد» بماند. `cell_rho_lag7` روی
#: **۹۹٪** ردیف‌ها تهی است (F289) و `cell_rho_lag1` روی ۸۵٪ (F220)؛ برداشتن هر دو
#: با هم روی **هر سه** تقسیم علامت منفی داد (یعنی بی‌ضرر یا کمی مفید). ستونی که نه
#: پوشش دارد و نه Δ، جایی در فیچرست ندارد.
COVERAGE_DROPS = ["cell_rho_lag1", "cell_rho_lag7"]

#: مجموعه‌ی مرجع نهایی جریان ب: `lean` منهای دو ستون بالا.
CORE_FAMILIES = LEAN_FAMILIES


def core_columns() -> list[str]:
    return [c for c in feature_set("lean") if c not in COVERAGE_DROPS]


# ---------------------------------------------------------------------------
# سرشماری ستون‌ها
# ---------------------------------------------------------------------------

#: (وضعیت، دلیل، شاهد) به‌ازای هر گروه ستون. ترتیب مهم است: نخستین تطابق برنده است.
STATUS_RULES: list[tuple[str, list[str], str, str, str]] = [
    ("هدف", ["rho", "NoRecv", "Recv", "n_reservations"],
     "متغیر هدف و اجزای آن — فیچر شدنشان نشتی کامل است", "قاعده‌ی پایداری هدف · F90"),
    ("کلید", ["date_gregorian", "restaurant", "dish", "meal_seq", "academic_year",
              "semester", "year_jalali"],
     "کلید شبکه یا محور تقسیم — نه فیچر است نه دورریختنی", "بند ۱۳-۹ · F103"),
    ("فیلتر", ["is_war_window", "is_structural_gap"],
     "پرچم پنجره‌ی شوک؛ نقشش **فیلتر** است نه فیچر", "ردیف ۸۰ · F135"),
]

#: ستون‌هایی که پس از سنجش **رد** شدند — هرکدام با شناسه‌ی یافته‌ی خودش.
REJECTED: dict[str, tuple[str, str]] = {
    **{c: ("خانواده‌ی `entity` — Δ روی هیچ تقسیمی معنادار نیست؛ و در غیاب "
           "`history` هم چیزی اضافه نمی‌کند", "F276 · F277 · F279")
       for c in FEATURE_SETS["entity"]},
    **{c: ("خانواده‌ی `weather` — Δ بی‌اثر، و تنها خانواده‌ای که در استقرار به "
           "**پیش‌بینی** نیاز دارد", "F276 · F277")
       for c in FEATURE_SETS["weather"]},
    **{c: ("خانواده‌ی `composition` — نسل اول و نسل دوم هر دو رد شدند", "F277 · F295")
       for c in FEATURE_SETS["composition"]},
}

#: ستون‌هایی که در ماتریس هستند ولی هرگز در فیچرستی نبودند (F282) — اینجا تعیین
#: تکلیف می‌شوند. هر کدام یا نماینده‌ی خوشه‌ای است که نماینده‌اش از قبل هست، یا
#: پشتوانه‌ی محاسبه‌ی ستون دیگری است، یا رده‌ی رد‌شده.
ORPHAN_RULES: dict[str, tuple[str, str, str]] = {
    "city": ("هم‌خط", "با `campus` یکی است — نگه‌داشتن هر دو تکرار محض است", "F206"),
    "is_friday": ("هم‌خط", "زیرمجموعه‌ی `day_of_week`", "—"),
    "is_holiday_national": ("هم‌خط", "زیرمجموعه‌ی `is_holiday_any`", "—"),
    "is_named_holiday": ("هم‌خط", "زیرمجموعه‌ی `is_holiday_any`", "—"),
    "is_final_exam_period": ("هم‌خط", "زیرمجموعه‌ی `is_exam_period`", "—"),
    "is_midterm_period": ("هم‌خط", "زیرمجموعه‌ی `is_exam_period`", "—"),
    "is_evaluation_period": ("هم‌خط", "زیرمجموعه‌ی `is_exam_period`", "—"),
    "is_add_drop_period": ("هم‌خط", "زیرمجموعه‌ی `is_in_session`", "—"),
    "is_nowruz_block": ("رد‌شده", "رژیم تکرارشونده — خانواده‌ی `regime` خالی ماند", "F283"),
    "is_inter_semester_break": ("رد‌شده", "همان بالا", "F283"),
    "is_summer_break": ("رد‌شده", "همان بالا", "F283"),
    "holiday_block_length": ("رد‌شده", "منشأ **۴۷٪** زیان مؤلفه‌ی تعطیلات؛ برون‌یابی "
                             "روی بلوک نوروز خرابش می‌کند", "F188 · F293"),
    "days_since_semester_start": ("هم‌خط", "با `semester_progress` و `week_of_semester` "
                                  "یک اطلاع را می‌دهد", "—"),
    "days_to_semester_end": ("هم‌خط", "همان بالا", "—"),
    "cell_hist_res_mean": ("پشتوانه", "مخرج `res_vs_cell_hist` و ورودی بازسازی "
                           "جمع‌های کوچک‌سازی", "F287"),
    "venue_hist_res_mean": ("پشتوانه", "مخرج `res_vs_venue_hist`", "—"),
    "venue_dow_hist_res_mean": ("پشتوانه", "پشتوانه‌ی محاسبه، نه فیچر", "—"),
    "dish_hist_res_mean": ("پشتوانه", "پشتوانه‌ی محاسبه، نه فیچر", "—"),
    "global_hist_res_mean": ("پشتوانه", "پشتوانه‌ی محاسبه، نه فیچر", "—"),
    "venue_dow_hist_n": ("هم‌خط", "شمارنده‌ی هم‌خانواده؛ `venue_hist_n` نماینده است", "—"),
    "dish_hist_n": ("هم‌خط", "همان بالا", "—"),
    "global_hist_n": ("هم‌خط", "همان بالا", "—"),
}

#: ستون‌های نامزدی که جریان ب ساخت و در ماتریس ذخیره‌شده نیستند.
CANDIDATE_GROUPS: dict[str, tuple[list[str], str, str]] = {}


def _register_candidates() -> None:
    """نامزدهای ساخته‌شده‌ی بندهای ۱۴-۹ تا ۱۴-۱۱ — همه رد شدند، ولی همه ثبت می‌شوند."""
    from src.features.composition_v4_g2 import G2_ALL, G2_INTERACTIONS
    from src.features.history_v4 import ROLL_WINDOWS
    from src.features.regime_v4 import REGIME_FEATURES
    from src.features.seasonal_v4 import HARM_ALL

    roll = [f"{lvl}_roll{w}_rho" for w in ROLL_WINDOWS for lvl in ("cell", "venue")]
    CANDIDATE_GROUPS.update({
        "پنجره‌ی متحرک": (roll, "قوی‌ترین نامزد بود ولی فقط روی **یک** تقسیم "
                          "برون‌نمونه‌ای معنادار شد", "F286"),
        "کوچک‌سازی بیزی": (["cell_shrunk_rho", "venue_shrunk_rho"],
                           "با پیشین بازبرازش‌شده؛ از سد قاعده رد نشد", "F286 · F287"),
        "آستانه‌ی lag تازه": (["cell_rho_lag1_b27", "cell_rho_lag7_b27"],
                              "پوشش بالاتر، Δ بهتر نشد", "F289"),
        "رژیم": (REGIME_FEATURES, "رژیم تکرارشونده هم پرچم نمی‌خواهد — "
                 "۰ از ۲ تقسیم", "F283 · F284"),
        "فصلی و هارمونیک": (HARM_ALL + ["seas_holiday_signed", "seas_weekly_safe"],
                            "نه جایگزین `calendar` شد نه کنارش نشست", "F290 · F293"),
        "⛔ مؤلفه‌ی نشتی": (["seas_weekly_leaky"],
                            "برازش روی کل سری — **هرگز فیچر نمی‌شود**؛ فقط برای "
                            "اندازه‌گیری نشت ساخته شد", "F292"),
        "ترکیب نسل دوم": (G2_ALL + G2_INTERACTIONS,
                          "سقف F59 را جابه‌جا نکرد", "F295 · F297"),
    })


def census(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """هر ستون، دقیقاً یک وضعیت — شرط بسته‌شدن بند ۱۴-۱۲-۲."""
    df = pd.read_parquet(FEATURES_PATH) if df is None else df
    _register_candidates()
    core = set(core_columns())

    rows: list[dict] = []
    seen: set[str] = set()

    def add(col: str, status: str, reason: str, evidence: str, group: str) -> None:
        if col in seen:
            return
        seen.add(col)
        cov = 100 * df[col].notna().mean() if col in df.columns else np.nan
        rows.append({"column": col, "status": status, "group": group,
                     "reason": reason, "evidence": evidence,
                     "coverage_pct": cov, "in_matrix": col in df.columns})

    for status, cols, reason, evidence in [(a, b, c, d) for a, b, c, d in
                                           [(r[0], r[1], r[2], r[3]) for r in STATUS_RULES]]:
        for col in cols:
            add(col, status, reason, evidence, "—")

    for fam in CORE_FAMILIES:
        for col in FEATURE_SETS[fam]:
            if col in core:
                add(col, "مرجع", "عضو مجموعه‌ی مرجع جریان ب", "F276 · F278 · ردیف ۱۱۴", fam)

    for col in COVERAGE_DROPS:
        add(col, "حذف — پوشش", "پوشش ناکافی؛ برداشتنش روی هر سه تقسیم بی‌ضرر بود",
            "F220 · F289", "history")

    for col, (reason, evidence) in REJECTED.items():
        add(col, "رد‌شده", reason, evidence, "ablation ۱۴-۹-۱")

    for col, (status, reason, evidence) in ORPHAN_RULES.items():
        add(col, "هم‌خط" if status == "هم‌خط" else status, reason, evidence, "بی‌صاحب (F282)")

    for group, (cols, reason, evidence) in CANDIDATE_GROUPS.items():
        for col in cols:
            add(col, "نامزدِ رد‌شده", reason, evidence, group)

    for col in df.columns:
        add(col, "⚠️ بی‌وضعیت", "هیچ قاعده‌ای این ستون را پوشش نمی‌دهد", "—", "—")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# هم‌خطی
# ---------------------------------------------------------------------------

VIF_THRESHOLD = 5.0
PAIR_CORR_THRESHOLD = 0.90


def collinearity(df: pd.DataFrame | None = None, cols: list[str] | None = None) -> dict:
    """VIF و همبستگی جفتی روی مجموعه‌ی مرجع.

    ⚠️ VIF برای مدل **خطی** معنا دارد، نه برای درخت — و قهرمان فعلی درخت است. پس
    این جدول برای رد ستون در مجموعه‌ی مرجع استفاده نمی‌شود؛ ورودی **جریان پ** است،
    جایی که خانواده‌های خطی و کوانتایل خطی اجرا می‌شوند و جدول 7.5.3 صریحاً
    می‌گوید «یکی از هر خوشه‌ی هم‌خط (VIF>۵) حذف شود».
    """
    df = pd.read_parquet(FEATURES_PATH) if df is None else df
    cols = cols or core_columns()
    numeric = [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
               and not pd.api.types.is_bool_dtype(df[c])]
    X = df[numeric].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).astype(float)
    keep = [c for c in numeric if X[c].std() > 0]
    X = X[keep]

    corr = X.corr()
    pairs = []
    for i, a in enumerate(keep):
        for b in keep[i + 1:]:
            r = float(corr.loc[a, b])
            if abs(r) >= PAIR_CORR_THRESHOLD:
                pairs.append({"a": a, "b": b, "r": r})

    # VIF از معکوس ماتریس همبستگی — پایدارتر و سریع‌تر از رگرسیون هر ستون.
    vif = {}
    try:
        inv = np.linalg.pinv(corr.to_numpy())
        vif = {c: float(inv[i, i]) for i, c in enumerate(keep)}
    except np.linalg.LinAlgError:
        logger.warning("ماتریس همبستگی معکوس‌پذیر نیست — VIF رد شد")

    return {"vif": pd.Series(vif).sort_values(ascending=False),
            "pairs": pd.DataFrame(pairs).sort_values("r", key=abs, ascending=False)
            if pairs else pd.DataFrame(columns=["a", "b", "r"]),
            "n_cols": len(keep)}


# ---------------------------------------------------------------------------
# آزمون «آیا ستونِ هم‌خط جایش را توجیه می‌کند؟»
# ---------------------------------------------------------------------------

#: جفت‌های هم‌خطِ **داخل خودِ `FS_core`** که جدول VIF پیدا کرد. برای هرکدام، عضو
#: کم‌اطلاع‌تر نامزد حذف است و Δ حذفش سنجیده می‌شود — چون قاعده‌ی این بند می‌گوید
#: ماندن باید توجیه شود، نه رفتن.
COLLINEAR_DROPS: dict[str, dict] = {
    "drop_n_people": {"drop": ["n_people"],
                      "why": "r=۰.۹۹۹۸ با `Res` · VIF ۲٬۶۵۱ — و نسبتشان هم جدا هست"},
    "drop_semester_progress": {"drop": ["semester_progress"],
                               "why": "r=۰.۹۹۶ با `week_of_semester` · VIF ۲۴۸"},
    "drop_venue_rho_lag1": {"drop": ["venue_rho_lag1"],
                            "why": "r=۰.۹۵۷ با `venue_last_rho` · VIF ۱۲.۸"},
    "drop_all_three": {"drop": ["n_people", "semester_progress", "venue_rho_lag1"],
                       "why": "هر سه با هم"},
}


def test_collinear_drops(tau: float | None = None, n_boot: int | None = None) -> pd.DataFrame:
    """Δ حذف هر ستون هم‌خط نسبت به `FS_core` کامل — منفی یعنی حذف بهتر است."""
    from src.baselines import pinball_loss
    from src.cv import block_bootstrap_2d_mean
    from src.features.ablation_v4 import N_BOOT, SEEDS, TAU, fit_predict
    from src.models.run_phase13_oos import build_splits

    tau = TAU if tau is None else tau
    n_boot = N_BOOT if n_boot is None else n_boot
    df = pd.read_parquet(FEATURES_PATH)
    base = core_columns()

    rows = []
    for split_name, split in build_splits(df).items():
        train, test = split["train"], split["test"].reset_index(drop=True)
        y = test["rho"].to_numpy(float)
        work = test[["date_gregorian", "restaurant"]].copy()
        ref = np.mean([pinball_loss(y, fit_predict(train, test, tau, base, s), tau)
                       for s in SEEDS], axis=0)
        for name, spec in COLLINEAR_DROPS.items():
            cols = [c for c in base if c not in spec["drop"]]
            arm = np.mean([pinball_loss(y, fit_predict(train, test, tau, cols, s), tau)
                           for s in SEEDS], axis=0)
            work["_d"] = arm - ref
            delta, lo, hi = block_bootstrap_2d_mean(
                work, "_d", day_col="date_gregorian", unit_col="restaurant", n_boot=n_boot)
            rows.append({"variant": name, "why": spec["why"], "split": split_name,
                         "n_dropped": len(spec["drop"]), "delta": delta,
                         "ci_low": lo, "ci_high": hi,
                         "delta_pct": 100 * delta / float(ref.mean()),
                         "better": bool(hi < 0), "worse": bool(lo > 0)})
            logger.info("%s · %-24s → Δ=%+.6f [%+.6f, %+.6f] (%+.2f٪) %s", split_name, name,
                        delta, lo, hi, rows[-1]["delta_pct"],
                        "بهتر" if hi < 0 else ("بدتر" if lo > 0 else "—"))
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "feature_collinear_drops.csv", index=False)
    return out


# ---------------------------------------------------------------------------
# گزارش
# ---------------------------------------------------------------------------

SPLIT_ORDER = ["in_window", "oos_1403-1404", "oos_1404-1405"]
SPLIT_FA = {"in_window": "درون‌پنجره", "oos_1403-1404": "۱۴۰۳-۱۴۰۴", "oos_1404-1405": "۱۴۰۴-۱۴۰۵"}


def write_report(cen: pd.DataFrame, coll: dict, drops: pd.DataFrame | None = None) -> str:
    from src.features.ablation_v4 import num

    unclassified = cen[cen.status.str.contains("بی‌وضعیت")]
    counts = cen["status"].value_counts()
    lines = [
        "# انتخاب ویژگی، هم‌خطی و بودجه‌ی ستون — بند ۱۴-۱۲",
        "",
        "> `src/features/selection_v4.py` · سرشماری کامل: "
        f"`reports/phase14/feature_census.csv`",
        "",
        "## قاعده‌ای که این بند اعمال می‌کند",
        "",
        "> «هر ستون یا در یک فیچرست مدلی هست یا صریح کنار گذاشته شده. **ماند چون",
        "> ضرری نداشت** دلیل پذیرفته‌شده‌ای نیست.»",
        "",
        "بار اثبات وارونه می‌شود: تا اینجا ستون باید ثابت می‌کرد **مضر** است تا برود؛",
        "از اینجا باید ثابت کند **مفید** است تا بماند.",
        "",
        "## سرشماری",
        "",
        "| وضعیت | تعداد ستون |",
        "|---|---|",
    ]
    for status, n in counts.items():
        lines.append(f"| {status} | {num(int(n), 'd')} |")
    lines += [
        f"| **جمع** | **{num(int(len(cen)), 'd')}** |",
        "",
        ("✅ **صفر ستون بی‌وضعیت** — شرط بسته‌شدن بند ۱۴-۱۲-۲ برقرار است."
         if unclassified.empty else
         f"❌ **{num(len(unclassified), 'd')} ستون بی‌وضعیت** — بند بسته نمی‌شود: "
         + "، ".join(f"`{c}`" for c in unclassified["column"])),
        "",
        "## مجموعه‌ی مرجع نهایی جریان ب",
        "",
        f"`FS_core` = **{num(len(core_columns()), 'd')} ستون** = `lean` "
        f"({num(len(feature_set('lean')), 'd')}) منهای دو ستون کم‌پوشش.",
        "",
        "| ستون حذف‌شده | پوشش | چرا |",
        "|---|---|---|",
    ]
    for col in COVERAGE_DROPS:
        row = cen[cen.column == col]
        cov = row["coverage_pct"].iloc[0] if len(row) else np.nan
        lines.append(f"| `{col}` | {num(cov, '.2f')}٪ | پوشش ناکافی؛ برداشتنش روی هر سه "
                     "تقسیم بی‌ضرر بود (F220، F289) |")

    lines += [
        "",
        "## هم‌خطی — ورودی جریان پ، نه معیار حذف اینجا",
        "",
        "⚠️ VIF برای مدل **خطی** معنا دارد نه درخت، و قهرمان فعلی درخت است. پس این",
        "جدول ستونی را از `FS_core` حذف نمی‌کند؛ ورودی جدول 7.5.3 است، که برای",
        "خانواده‌های خطی می‌گوید «یکی از هر خوشه‌ی هم‌خط (VIF>۵) حذف شود».",
        "",
        f"روی {num(coll['n_cols'], 'd')} ستون عددی `FS_core`:",
        "",
        "| ستون | VIF |",
        "|---|---|",
    ]
    for col, v in coll["vif"].head(10).items():
        flag = " ⚠️" if v > VIF_THRESHOLD else ""
        lines.append(f"| `{col}` | {num(v, '.2f')}{flag} |")

    pairs = coll["pairs"]
    lines += ["", f"جفت‌های با |r| ≥ {num(PAIR_CORR_THRESHOLD, '.2f')}: "
              f"**{num(len(pairs), 'd')}**", ""]
    if len(pairs):
        lines += ["| ستون الف | ستون ب | r |", "|---|---|---|"]
        for _, r in pairs.head(12).iterrows():
            lines.append(f"| `{r['a']}` | `{r['b']}` | {num(r['r'], '+.3f')} |")

    if drops is not None and len(drops):
        lines += [
            "",
            "### آیا این ستون‌ها جایشان را توجیه می‌کنند؟",
            "",
            "قاعده‌ی این بند می‌گوید ماندن باید توجیه شود. پس Δ **حذف** هر عضو هم‌خط",
            "سنجیده شد؛ منفی یعنی حذف بهتر است.",
            "",
            "| حذف | چرا نامزد شد | " + " | ".join(
                SPLIT_FA[s] for s in SPLIT_ORDER) + " |",
            "|---|---|" + "---|" * len(SPLIT_ORDER),
        ]
        for name, spec in COLLINEAR_DROPS.items():
            block = drops[drops.variant == name]
            cells = []
            for split in SPLIT_ORDER:
                row = block[block.split == split]
                if not len(row):
                    cells.append("—")
                    continue
                r = row.iloc[0]
                mark = " ✅" if r["better"] else (" ❌" if r["worse"] else "")
                cells.append(f"{num(r['delta'])} [{num(r['ci_low'])}, {num(r['ci_high'])}]{mark}")
            lines.append(f"| `{'`، `'.join(spec['drop'])}` | {spec['why']} | "
                         + " | ".join(cells) + " |")
        lines += ["", "✅ = حذف معنادار **بهتر** · ❌ = حذف معنادار **بدتر**.", ""]

    lines += [
        "",
        "## ⚠️ دامنه‌ی اعتبار کل جریان ب",
        "",
        "هر Δی که بندهای ۱۴-۹ تا ۱۴-۱۱ گزارش کردند با **یک مدل** اندازه‌گیری شد:",
        "قهرمان LightGBM. ارزش فیچر به ذات مدل وابسته است — جدول 7.5.3 خودش می‌گوید",
        "مدل خطی پایه‌ی فوریه **لازم** دارد در حالی که درخت همان اطلاعات را از شاخص",
        "صحیح روز هفته می‌سازد. پس «هارمونیک کمکی نکرد» (F290) یعنی **برای درخت**",
        "کمکی نکرد.",
        "",
        "این نتیجه‌ها را باطل نمی‌کند، **دامنه‌شان را تعیین می‌کند**: `FS_core` فیچرست",
        "مرجع خانواده‌های **درختی** است. جریان پ برای هر ذات مدل دیگر باید گام ۱ و ۲",
        "پروتکل 7.5.2 را از نو اجرا کند — و ستون‌های رد‌شده‌ی اینجا برایش **نامزدند،",
        "نه ممنوع**. تنها استثنا `seas_weekly_leaky` است که برای هیچ مدلی مجاز نیست.",
        "",
    ]
    text = "\n".join(lines) + "\n"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "feature_selection.md").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    set_global_seed()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(FEATURES_PATH)
    cen = census(df)
    cen.to_csv(OUTPUT_DIR / "feature_census.csv", index=False)
    coll = collinearity(df)
    drops = test_collinear_drops()
    write_report(cen, coll, drops)

    pd.set_option("display.width", 200)
    print("=== سرشماری ستون‌ها ===")
    print(cen["status"].value_counts().to_string())
    unclassified = cen[cen.status.str.contains("بی‌وضعیت")]
    print(f"\nبی‌وضعیت: {len(unclassified)}")
    if len(unclassified):
        print(unclassified[["column", "coverage_pct"]].to_string(index=False))
    print(f"\nFS_core = {len(core_columns())} ستون")
    print(f"\nجفت‌های هم‌خط (|r| ≥ {PAIR_CORR_THRESHOLD}): {len(coll['pairs'])}")
    if len(coll["pairs"]):
        print(coll["pairs"].head(10).to_string(index=False))
    print("\n=== بالاترین VIF ===")
    print(coll["vif"].head(8).round(2).to_string())


if __name__ == "__main__":
    main()
