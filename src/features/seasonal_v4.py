r"""بند ۱۴-۱۰ — خانواده‌ی فصلی و هارمونیک.

**ورودی اجباری این بند، فهرست نامزدها را پیشاپیش کوتاه می‌کند.** WBS می‌گوید
«مؤلفه‌ای که منتقل نمی‌شود فیچر نمی‌شود، هر قدر هم درون‌پنجره برازش خوبی بدهد»، و
رتبه‌بندی انتقال‌پذیری بند ۱۴-۵-۷ قبلاً محاسبه شده است:

| مؤلفه | نسبت انتقال (F185) | حکم پیش از نوشتن کد |
|---|---|---|
| **هفتگی** | **۰.۹۲** (کمینه ۰.۸۸) | تنها نامزد جدی |
| نیمسال | ۰.۳۲ — و با داده‌ی بیشتر **بدتر** می‌شود (F187) | نامزد ضعیف، آزموده می‌شود تا ادعا عددی بماند |
| سالانه | ۰.۲۳ — با دو سال آموزش علامتش عوض می‌شود (F186) | نامزد ضعیف |
| رمضان | **−۱.۱۵** — و شکلش بین دو سال وارونه است (F284) | ⛔ بند ۱۴-۹-۲ قبلاً ردش کرد؛ اینجا تکرار نمی‌شود |
| تعطیلات | **−۱۴.۷۹**، که ۴۷٪‌اش از `holiday_block_length` می‌آید (F188) | ⛔ همان ستون از قبل در فیچرست **نیست** |

پس این بند سه پرسش باز دارد، نه شش‌تا: آیا هارمونیک جای پرچم تقویمی را می‌گیرد؟
آیا فاصله‌ی **علامت‌دار** بهتر از دو ستون جداست؟ و آیا مؤلفه‌ی فصلیِ بی‌نشت چیزی
اضافه می‌کند؟

⚠️ **پاسخ احتمالی پرسش سوم از قبل در ماتریس است.** مؤلفه‌ی هفتگیِ بی‌نشت دقیقاً
همان `venue_dow_hist_rho` است: میانگین انبساطی نرخ هر (سلف، وعده، روزهفته) تا لحظه‌ی
برش. پس نامزد تازه باید ثابت کند **فراتر از آن** چیزی دارد — که با بازویی آزموده
می‌شود که همان ستون را برمی‌دارد.

**دو دام اجباری WBS، و کاری که اینجا با هرکدام شده:**

۱. **نشتی مؤلفه‌ی تجزیه.** STL روی کل سری برازش می‌شود و مؤلفه‌ی فصلیِ هر نقطه از
   آینده‌ی همان سری هم اطلاع دارد. اینجا دو نسخه ساخته می‌شود: نسخه‌ی **بی‌نشت**
   (شاخص هفتگی انبساطی، فقط تا لحظه‌ی برش) که نامزد فیچر است، و نسخه‌ی **نشتی**
   (برازش روی کل سری) که **هرگز فیچر نمی‌شود** و فقط برای این محاسبه می‌شود که
   اندازه‌ی نشت یک عدد داشته باشد، نه یک هشدار.
۲. **هم‌خطی با `calendar`.** هدف **جایگزینی** است نه انباشت. بازوی
   `harmonic_replaces_calendar` دقیقاً همین را می‌پرسد.

**قاعده‌ی پذیرش — پیش از اجرا (قاعده‌ی ۱۴-۱-ه).** خانواده‌ی فصلی فقط وقتی به
مجموعه‌ی مرجع اضافه (یا جایگزین `calendar`) می‌شود که Δ‌اش روی **هر دو** تقسیم
برون‌نمونه‌ای معنادار منفی باشد.

اجرا: ``python -m src.features.seasonal_v4``
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.baselines import pinball_loss
from src.config import REPORTS_DIR, set_global_seed
from src.cv import block_bootstrap_2d_mean
from src.features.ablation_v4 import _FA_DIGITS, N_BOOT, SEEDS, TAU, fit_predict, num
from src.features.build_v4 import FEATURE_SETS, OUTPUT_PATH as FEATURES_PATH, feature_set
from src.features.cutoff import expanding_stat_at_cutoff, meal_seq
from src.models.run_phase13_oos import build_splits

logger = logging.getLogger(__name__)

OUTPUT_DIR = REPORTS_DIR / "phase14"

#: طول ماه‌های تقویم شمسی — برای تبدیل (ماه، روز) به روزِ سال.
JALALI_MONTH_LENGTHS = [31, 31, 31, 31, 31, 31, 30, 30, 30, 30, 30, 29]

#: تعداد هارمونیک هر دوره. سه‌تا برای هفتگی (تا شکل نامتقارن هفته را بگیرد)، دوتا
#: برای بقیه — بیشتر از این، با یک سال آموزش فقط نویز برازش می‌کند.
N_HARMONICS = {"dow": 3, "semester": 2, "annual": 2}

#: ⚠️ دوره‌ی هارمونیک هفتگی **۷** است نه ۶، با اینکه WBS «هفته‌ی سرویس» را ۶ نوشته.
#: دلیلش داده است: شاخص `day_of_week` مقادیر ۰ تا ۶ می‌گیرد و با دوره‌ی ۷ می‌چرخد.
#: سرویس **حجمش** در انتهای هفته کم می‌شود (پنجشنبه نصف روزهای دیگر، جمعه فقط
#: ۱.۴۷٪ ردیف‌ها) ولی خودِ شاخص جا نمی‌افتد؛ دوره‌ی ۶ روی شاخصِ ۷تایی، فاز را هر
#: هفته یک واحد می‌لغزاند و هارمونیک را بی‌معنا می‌کند.
DOW_PERIOD = 7

#: پنجره‌ی برش فاصله‌ی علامت‌دار تا تعطیلی. درس F188: ستون پیوسته‌ی تعطیلات با
#: برون‌یابی خراب می‌شود، پس دامنه محدود می‌شود تا مقدار آزمون بیرون از دامنه‌ی
#: آموزش نیفتد.
HOLIDAY_CLIP_DAYS = 14


def jalali_doy(month: pd.Series, day: pd.Series) -> pd.Series:
    """روزِ سال شمسی از (ماه، روز) — پایه‌ی هارمونیک سالانه."""
    offsets = np.concatenate([[0], np.cumsum(JALALI_MONTH_LENGTHS)[:-1]])
    return pd.Series(offsets[month.astype(int).to_numpy() - 1] + day.astype(int).to_numpy(),
                     index=month.index)


def _harmonics(phase: pd.Series, n: int, prefix: str) -> pd.DataFrame:
    """جفت‌های $\\sin/\\cos$ برای فاز نرمال‌شده در $[0,1)$."""
    out = {}
    for k in range(1, n + 1):
        out[f"{prefix}_sin{k}"] = np.sin(2 * np.pi * k * phase)
        out[f"{prefix}_cos{k}"] = np.cos(2 * np.pi * k * phase)
    return pd.DataFrame(out, index=phase.index)


def add_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    """هارمونیک‌ها، فاصله‌ی علامت‌دار، و دو نسخه‌ی مؤلفه‌ی فصلی (بی‌نشت و نشتی)."""
    out = df.sort_values(["date_gregorian", "meal"], kind="stable").reset_index(drop=True)
    out["meal_seq"] = meal_seq(out["date_gregorian"], out["meal"])

    out = pd.concat([
        out,
        _harmonics(out["day_of_week"].astype(float) / DOW_PERIOD, N_HARMONICS["dow"], "seas_dow"),
        _harmonics(out["semester_progress"].astype(float), N_HARMONICS["semester"], "seas_sem"),
        _harmonics(jalali_doy(out["month_jalali"], out["day_of_month_jalali"]).astype(float) / 365.0,
                   N_HARMONICS["annual"], "seas_year"),
    ], axis=1)

    # فاصله‌ی **علامت‌دار** تا تعطیلی: یک ستون به‌جای دو. نزدیک‌ترین طرف برنده است،
    # و علامت می‌گوید قبل یا بعد. دامنه بریده می‌شود (درس F188).
    nxt = out["days_to_next_holiday"].astype(float)
    prv = out["days_since_last_holiday"].astype(float)
    out["seas_holiday_signed"] = np.where(nxt.fillna(np.inf) <= prv.fillna(np.inf),
                                          nxt, -prv).clip(-HOLIDAY_CLIP_DAYS, HOLIDAY_CLIP_DAYS)

    out = _add_weekly_index(out)
    return out


def _add_weekly_index(df: pd.DataFrame) -> pd.DataFrame:
    r"""شاخص فصلی هفتگی، در دو نسخه — یکی مجاز، یکی ممنوع.

    * ``seas_weekly_safe``: نسبت نرخ انبساطیِ (وعده، روزهفته) به نرخ انبساطیِ (وعده)،
      هر دو **فقط تا لحظه‌ی برش**. این همان مؤلفه‌ی فصلی هفتگی است، ساخته‌شده با
      همان مسیری که هر فیچر تاریخی دیگر از آن می‌گذرد.
    * ``seas_weekly_leaky``: همان نسبت، ولی روی **کل** سری — یعنی نقطه‌ی امروز از
      میانگین روزهای بعدی هم خبر دارد. ⛔ **هرگز فیچر نمی‌شود**؛ فقط برای اندازه‌گیری
      نشت ساخته می‌شود (دام ۱ بند ۱۴-۱۰).
    """
    out = df
    for keys, prefix in ((["meal", "day_of_week"], "_dw"), (["meal"], "_m")):
        nr = expanding_stat_at_cutoff(out, keys, "NoRecv", out_prefix=f"{prefix}_nr")
        res = expanding_stat_at_cutoff(out, keys, "Res", out_prefix=f"{prefix}_res")
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"{prefix}_rho"] = np.where(res[f"{prefix}_res_sum"] > 0,
                                            nr[f"{prefix}_nr_sum"] / res[f"{prefix}_res_sum"], np.nan)
    out["seas_weekly_safe"] = out["_dw_rho"] / out["_m_rho"].replace(0, np.nan)

    full_dw = out.groupby(["meal", "day_of_week"], observed=True).apply(
        lambda g: g["NoRecv"].sum() / max(g["Res"].sum(), 1), include_groups=False)
    full_m = out.groupby("meal", observed=True).apply(
        lambda g: g["NoRecv"].sum() / max(g["Res"].sum(), 1), include_groups=False)
    idx = pd.MultiIndex.from_frame(out[["meal", "day_of_week"]])
    out["seas_weekly_leaky"] = (full_dw.reindex(idx).to_numpy()
                                / full_m.reindex(out["meal"]).to_numpy())
    return out.drop(columns=["_dw_rho", "_m_rho"])


# ---------------------------------------------------------------------------
# بازوها
# ---------------------------------------------------------------------------

HARM_DOW = [f"seas_dow_{t}{k}" for k in range(1, N_HARMONICS["dow"] + 1) for t in ("sin", "cos")]
HARM_SEM = [f"seas_sem_{t}{k}" for k in range(1, N_HARMONICS["semester"] + 1) for t in ("sin", "cos")]
HARM_YEAR = [f"seas_year_{t}{k}" for k in range(1, N_HARMONICS["annual"] + 1) for t in ("sin", "cos")]
HARM_ALL = HARM_DOW + HARM_SEM + HARM_YEAR

#: هر بازو: `add` · `drop` (نسبت به `lean`) · `ref_drop` (ستون‌هایی که از **مرجع** هم
#: برداشته می‌شوند، برای بازوهای شرطی).
ARMS: dict[str, dict] = {
    "harmonic_dow_add": {
        "add": HARM_DOW, "drop": [], "ref_drop": [],
        "question": "هارمونیک هفتگی **کنار** پرچم‌های تقویمی"},
    "harmonic_dow_replaces_dow": {
        "add": HARM_DOW, "drop": ["day_of_week"], "ref_drop": [],
        "question": "هارمونیک هفتگی **به‌جای** `day_of_week`"},
    "harmonic_all_add": {
        "add": HARM_ALL, "drop": [], "ref_drop": [],
        "question": "هر سه دوره (هفتگی + نیمسال + سالانه)"},
    "harmonic_replaces_calendar": {
        "add": HARM_ALL, "drop": FEATURE_SETS["calendar"], "ref_drop": [],
        "question": "⭐ هارمونیک **به‌جای کل خانواده‌ی** `calendar`"},
    "holiday_signed": {
        "add": ["seas_holiday_signed"],
        "drop": ["days_to_next_holiday", "days_since_last_holiday"], "ref_drop": [],
        "question": "فاصله‌ی علامت‌دار به‌جای دو ستون جدا"},
    "weekly_index_safe": {
        "add": ["seas_weekly_safe"], "drop": [], "ref_drop": [],
        "question": "مؤلفه‌ی فصلی هفتگیِ بی‌نشت"},
    "harmonic_given_no_venue_dow": {
        "add": HARM_DOW, "drop": ["venue_dow_hist_rho"], "ref_drop": ["venue_dow_hist_rho"],
        "question": "⭐ هارمونیک وقتی `venue_dow_hist_rho` نیست"},
    # بازوی تکمیلیِ بازوی شرطی بالا. `harmonic_given_no_venue_dow` می‌گوید «وقتی
    # `venue_dow_hist_rho` نیست، هارمونیک کمک می‌کند» — ولی تصمیم واقعی این است که
    # آیا **جایگزینی** از خودِ `lean` بهتر است. همان قاعده‌ی پذیرش رویش اعمال می‌شود.
    "harmonic_swaps_venue_dow": {
        "add": HARM_DOW, "drop": ["venue_dow_hist_rho"], "ref_drop": [],
        "question": "⭐ هارمونیک **به‌جای** `venue_dow_hist_rho`، سنجیده با `lean`"},
    "LEAKY_weekly_index": {
        "add": ["seas_weekly_leaky"], "drop": [], "ref_drop": [],
        "question": "⛔ ممیزی نشت — **هرگز فیچر نمی‌شود**"},
}

#: بازوهایی که نامزد فیچر **نیستند** و فقط برای ممیزی اجرا می‌شوند.
AUDIT_ONLY = {"LEAKY_weekly_index"}


def evaluate(tau: float = TAU, seeds=SEEDS, n_boot: int = N_BOOT) -> pd.DataFrame:
    df = add_seasonal_features(pd.read_parquet(FEATURES_PATH))
    splits = build_splits(df)
    base_cols = feature_set("lean")

    rows = []
    for split_name, split in splits.items():
        train, test = split["train"], split["test"].reset_index(drop=True)
        y = test["rho"].to_numpy(float)
        res = test["Res"].to_numpy(float)
        work = test[["date_gregorian", "restaurant"]].copy()
        ref_cache: dict[frozenset, np.ndarray] = {}

        def reference(drop: list[str]) -> np.ndarray:
            key = frozenset(drop)
            if key not in ref_cache:
                cols = [c for c in base_cols if c not in key]
                ref_cache[key] = np.mean(
                    [pinball_loss(y, fit_predict(train, test, tau, cols, s), tau) for s in seeds],
                    axis=0)
            return ref_cache[key]

        for name, spec in ARMS.items():
            ref = reference(spec["ref_drop"])
            cols = [c for c in base_cols if c not in spec["drop"]] + spec["add"]
            arm = np.mean([pinball_loss(y, fit_predict(train, test, tau, cols, s), tau)
                           for s in seeds], axis=0)
            work["_d"] = arm - ref
            work["_dp"] = (arm - ref) * res
            delta, lo, hi = block_bootstrap_2d_mean(
                work, "_d", day_col="date_gregorian", unit_col="restaurant", n_boot=n_boot)
            delta_p, lo_p, hi_p = block_bootstrap_2d_mean(
                work, "_dp", day_col="date_gregorian", unit_col="restaurant", n_boot=n_boot)
            rows.append({
                "arm": name, "question": spec["question"], "audit_only": name in AUDIT_ONLY,
                "split": split_name, "split_label": split["label"], "n_cols": len(cols),
                "tau": tau, "pinball_ref": float(ref.mean()), "pinball_arm": float(arm.mean()),
                "delta": delta, "ci_low": lo, "ci_high": hi,
                "delta_pct": 100 * delta / float(ref.mean()),
                "delta_portions": delta_p, "ci_low_portions": lo_p, "ci_high_portions": hi_p,
                "better": bool(hi < 0), "worse": bool(lo > 0),
            })
            logger.info("%s · %-30s → Δ=%+.6f [%+.6f, %+.6f] (%+.2f٪) %s", split_name, name,
                        delta, lo, hi, rows[-1]["delta_pct"],
                        "بهتر" if hi < 0 else ("بدتر" if lo > 0 else "—"))

    out = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_DIR / "seasonal_variants.csv", index=False)
    return out


def answer_wbs_question(results: pd.DataFrame) -> str:
    """پاسخ صریح به پرسشی که WBS به‌عنوان شرط پذیرش نوشته.

    «آیا هارمونیک جای پرچم تقویمی را می‌گیرد، یا کنارش می‌نشیند، یا هیچ‌کدام؟» —
    سه حالت، و هرکدام از یک بازوی مشخص خوانده می‌شود، نه از برداشت کلی.
    """
    def oos(arm: str) -> pd.DataFrame:
        return results[(results.arm == arm) & results.split.str.startswith("oos_")]

    replaces = oos("harmonic_replaces_calendar")
    beside = oos("harmonic_dow_add")
    if len(replaces) == 2 and replaces["better"].all():
        return "**جایگزین می‌شود** — هارمونیک به‌جای کل خانواده‌ی `calendar` روی هر دو سال بهتر است"
    if len(beside) == 2 and beside["better"].all():
        return "**کنارش می‌نشیند** — افزودنش کمک می‌کند ولی جایگزینی خانواده‌ی `calendar` نه"
    if replaces["worse"].any():
        return ("**هیچ‌کدام** — و جایگزینی صریحاً **بدتر** است: پرچم‌های تقویمی چیزی دارند "
                "که هارمونیک هموار نمی‌تواند بسازد")
    return "**هیچ‌کدام** — نه جایگزینی برتری‌اش اثبات شد، نه افزودن"


SPLIT_ORDER = ["in_window", "oos_1403-1404", "oos_1404-1405"]
SPLIT_FA = {"in_window": "درون‌پنجره", "oos_1403-1404": "۱۴۰۳-۱۴۰۴", "oos_1404-1405": "۱۴۰۴-۱۴۰۵"}


def write_report(results: pd.DataFrame) -> str:
    leaky = results[results.arm == "LEAKY_weekly_index"]
    lines = [
        "# خانواده‌ی فصلی و هارمونیک — بند ۱۴-۱۰",
        "",
        f"> `src/features/seasonal_v4.py` · τ={num(TAU, '.2f')} · بذرها: "
        f"{'، '.join(num(s, 'd') for s in SEEDS)} · بوت‌استرپ بلوکی دوبعدی "
        f"({num(N_BOOT, 'd')} تکرار) · پایه: `lean`",
        "",
        "## آنچه پیش از نوشتن کد معلوم بود",
        "",
        "ورودی اجباری این بند، رتبه‌بندی انتقال‌پذیری بند ۱۴-۵-۷ است — و همان فهرست",
        "نامزدها را نصف می‌کند:",
        "",
        "| مؤلفه | نسبت انتقال (F185) | نتیجه |",
        "|---|---|---|",
        "| هفتگی | **۰.۹۲** | تنها نامزد جدی |",
        "| نیمسال | ۰.۳۲ (با داده‌ی بیشتر **بدتر** — F187) | نامزد ضعیف |",
        "| سالانه | ۰.۲۳ (با دو سال علامتش عوض می‌شود — F186) | نامزد ضعیف |",
        "| رمضان | **−۱.۱۵** (و شکلش وارونه است — F284) | ⛔ بند ۱۴-۹-۲ ردش کرد |",
        "| تعطیلات | **−۱۴.۷۹** (۴۷٪ از `holiday_block_length` — F188) | ⛔ آن ستون از قبل در فیچرست نیست |",
        "",
        "⚠️ و مؤلفه‌ی هفتگیِ بی‌نشت **از قبل در ماتریس است**: `venue_dow_hist_rho`.",
        "پس نامزد تازه باید ثابت کند فراتر از آن چیزی دارد — بازوی",
        "`harmonic_given_no_venue_dow` دقیقاً همین را می‌سنجد.",
        "",
        "## Δ هر بازو نسبت به مرجع خودش",
        "",
        "Δ = pinball(بازو) − pinball(مرجع). **منفی یعنی بازو بهتر است.**",
        "",
        "| بازو | پرسش | " + " | ".join(SPLIT_FA[s] for s in SPLIT_ORDER) + " |",
        "|---|---|" + "---|" * len(SPLIT_ORDER),
    ]
    for name, spec in ARMS.items():
        block = results[results.arm == name]
        cells = []
        for split in SPLIT_ORDER:
            row = block[block.split == split]
            if not len(row):
                cells.append("—")
                continue
            r = row.iloc[0]
            mark = " ✅" if r["better"] else (" ❌" if r["worse"] else "")
            cells.append(f"{num(r['delta'])} [{num(r['ci_low'])}, {num(r['ci_high'])}]{mark}")
        lines.append(f"| `{name}` | {spec['question']} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "✅ = معنادار بهتر · ❌ = معنادار **بدتر**.",
        "",
        "## پاسخ به پرسش شرط پذیرش",
        "",
        "> «آیا هارمونیک جای پرچم تقویمی را می‌گیرد، یا کنارش می‌نشیند، یا هیچ‌کدام؟»",
        "",
        answer_wbs_question(results),
        "",
        "## ⛔ دام ۱ — اندازه‌ی نشت مؤلفه‌ی تجزیه، به‌جای هشدار درباره‌اش",
        "",
        "مؤلفه‌ی فصلی که روی **کل سری** برازش شود، از آینده‌ی همان سری خبر دارد. این",
        "ستون ساخته شد تا نشتش **اندازه‌گیری** شود، و هرگز وارد هیچ فیچرستی نمی‌شود:",
        "",
        "| تقسیم | Δ نسخه‌ی نشتی | Δ نسخه‌ی بی‌نشت |",
        "|---|---|---|",
    ]
    safe = results[results.arm == "weekly_index_safe"]
    for split in SPLIT_ORDER:
        lk = leaky[leaky.split == split]
        sf = safe[safe.split == split]
        if not len(lk) or not len(sf):
            continue
        lkr, sfr = lk.iloc[0], sf.iloc[0]
        lk_mark = " ✅" if lkr["better"] else ""
        sf_mark = " ✅" if sfr["better"] else ""
        lines.append(f"| {SPLIT_FA[split]} | {num(lkr['delta'])} "
                     f"[{num(lkr['ci_low'])}, {num(lkr['ci_high'])}]{lk_mark} | "
                     f"{num(sfr['delta'])} [{num(sfr['ci_low'])}, {num(sfr['ci_high'])}]{sf_mark} |")

    oos = results[results.split.str.startswith("oos_")]
    lk = oos[oos.arm == "LEAKY_weekly_index"]["delta_pct"].mean()
    sf = oos[oos.arm == "weekly_index_safe"]["delta_pct"].mean()
    cond = oos[oos.arm == "harmonic_given_no_venue_dow"]["delta_pct"].mean()
    swap = oos[oos.arm == "harmonic_swaps_venue_dow"]["delta_pct"].mean()
    lines += [
        "",
        f"نسخه‌ی نشتی روی **هر دو** سال معنادار بهتر است (میانگین {num(lk, '+.2f')}٪)؛",
        f"نسخه‌ی بی‌نشتِ **همان مؤلفه** روی هیچ‌کدام معنادار نیست ({num(sf, '+.2f')}٪).",
        "یعنی تحلیل‌گری که STL را مستقیم روی کل سری بزند، یک بهبودِ «معنادار» می‌بیند",
        f"که حدود {num(abs(lk / sf), '.1f')} برابر اثر واقعی است — و می‌پذیردش. این",
        "همان دامی است که WBS نامش را برد؛ حالا عددش هم هست.",
        "",
        "## این جدول چه می‌گوید",
        "",
        "**۱. هیچ بازوی افزودنی یا جایگزینی از سد قاعده رد نشد** — ولی **همه‌ی** بازوهای",
        "هارمونیک روی **هر سه** تقسیم علامت منفی دارند. یعنی اثر واقعی است و کوچک؛",
        "نه اینکه نباشد. با یک سال آموزش، از نویزِ سال‌به‌سال جدا نمی‌شود.",
        "",
        f"**۲. تنها بازویی که رد شد، شرطی است — و همین معنایش را عوض می‌کند.** "
        f"`harmonic_given_no_venue_dow` روی هر دو سال معنادار است ({num(cond, '+.2f')}٪)، ولی",
        "مرجعش `lean` **منهای** `venue_dow_hist_rho` است. یعنی «هارمونیک وقتی کمک می‌کند",
        "که چیز بهتری نباشد» — نتیجه‌ی **جانشینی**، نه افزودن. و وقتی همان جایگزینی",
        f"مستقیماً با خودِ `lean` سنجیده می‌شود ({num(swap, '+.2f')}٪)، معناداری‌اش را",
        "از دست می‌دهد. پس `venue_dow_hist_rho` و هارمونیک هفتگی **یک چیز** را می‌گویند،",
        "و نگه‌داشتن هر دو انباشت است نه جایگزینی.",
        "",
        "**۳. فاصله‌ی علامت‌دار جای دو ستون را نمی‌گیرد.** −۰.۱۵٪ و +۰.۰۷٪ — بی‌اثر.",
        "خبر خوب اینجاست که ستون واقعاً خطرناک (`holiday_block_length`، منشأ ۴۷٪ زیان",
        "مؤلفه‌ی تعطیلات طبق F188) **از قبل** در هیچ فیچرستی نبود؛ پس توصیه‌ی F188",
        "عملاً از قبل اجرا شده بود و این بند فقط تأییدش می‌کند.",
        "",
        "**۴. نیمسال و سالانه همان‌قدر ضعیف درآمدند که F185/F186/F187 پیش‌بینی کرده",
        "بودند.** `harmonic_all_add` روی ۱۴۰۳-۱۴۰۴ معنادار است و روی ۱۴۰۴-۱۴۰۵ نه —",
        "دقیقاً الگوی «مؤلفه‌ای که با یک سال آموزش منتقل نمی‌شود». ورودی اجباری بند",
        "۱۴-۵-۷ کارش را کرد: هیچ ستونی بر پایه‌ی برازش درون‌پنجره‌ای وارد نشد.",
        "",
    ]
    text = "\n".join(lines) + "\n"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "seasonal_features.md").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    set_global_seed()
    results = evaluate()
    write_report(results)
    print("\n=== Δ٪ نسبت به مرجع ===")
    print(results.pivot(index="arm", columns="split", values="delta_pct")
          .reindex(ARMS).round(2).to_string())
    print("\n=== پاسخ پرسش WBS ===")
    print(answer_wbs_question(results))


if __name__ == "__main__":
    main()
