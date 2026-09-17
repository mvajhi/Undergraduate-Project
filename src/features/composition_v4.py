"""پل سطح فرد → سطح سلول برای پنجره‌ی v4 (معادل بند ۵.۱۹ روی شبکه‌ی تازه).

**چرا مجاز است.** رزرو ۷۲ ساعت پیش از سرو بسته می‌شود، پس در لحظه‌ی برش **هویت
رزروکنندگان کاملاً معلوم است**. آنچه معلوم نیست *نتیجه*ی آن‌هاست — و اینجا فقط از
تاریخچه‌ی **پیش‌از‌برشِ** همان افراد استفاده می‌شود، از همان `cutoff.py` که بقیه‌ی
فیچرهای تاریخی از آن می‌گذرند.

**چرا میانگین ساده کافی نیست.** فاز ۴ نشان داد میانگینِ تاریخچه‌ها در سطح سلول تقریباً
هیچ اضافه نمی‌کند (ΔR²=+۰.۰۰۲۴، F59) چون اثر فرد و روز **ضرب‌شونده** است نه جمع‌شونده
(F62). پس کنار میانگین، **پراکندگی** هم استخراج می‌شود (انحراف معیار، صدک ۹۰، سهم
افراد پرریسک): اگر اثر ضربی باشد، دُم توزیع مهم‌تر از مرکزش است.

⚠️ در پنجره‌ی قفل‌شده این فیچر فقط برای **شام** ارزش داشت (شام r=+۰.۵۱۹، ناهار
r=−۰.۰۲۰ — F63). اینجا همان تفکیک ساخته می‌شود ولی **دوباره آزموده** می‌شود، چون
پنجره‌ی تازه ۵.۸ برابر بزرگ‌تر است و F63 روی ۱۴۲ روز به‌دست آمده بود.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.baselines import BETA_ALPHA, BETA_BETA
from src.config import DATA_PROCESSED
from src.data.build_grid_v4 import apply_domain_filters, load_person_level
from src.features.cutoff import expanding_stat_at_cutoff, meal_seq

logger = logging.getLogger(__name__)

OUTPUT_PATH = DATA_PROCESSED / "composition_L2_v4.parquet"

#: آستانه‌ی «سابقه‌ی کافی» برای اینکه تاریخچه‌ی فرد معنادار شمرده شود (بند ۵.۱۹)
MIN_HIST = 10
#: آستانه‌ی «فرد پرریسک» برای فیچر سهم دُم
HIGH_RISK_RATE = 0.20


def person_history_at_cutoff(person: pd.DataFrame) -> pd.DataFrame:
    """نرخ عدم‌دریافت تاریخی هر فرد، **فقط تا لحظه‌ی برشِ همان ردیف**.

    نرخ **کوچک‌شده‌ی بیزی** است نه خام: فردی با ۲ رزرو قبلی که هر دو را نگرفته،
    نرخ خام ۱.۰ می‌دهد و تجمیع سطح سلول را منفجر می‌کند. پیشین Beta از بند ۴.۸
    (F8.3) همان پیشینی است که خط پایه‌ی B2 استفاده می‌کند.
    """
    work = person.sort_values(["date_gregorian", "meal"], kind="stable").reset_index(drop=True)
    work["meal_seq"] = meal_seq(work["date_gregorian"], work["meal"])
    work["_no_recv"] = work["Count"].where(work["dont_receive"], 0)

    no_recv = expanding_stat_at_cutoff(work, ["PersonId"], "_no_recv", out_prefix="_p_nr")
    total = expanding_stat_at_cutoff(work, ["PersonId"], "Count", out_prefix="_p_n")

    work["person_n_prior"] = total["_p_n_sum"].fillna(0)
    work["person_shrunk_rate"] = (
        (no_recv["_p_nr_sum"].fillna(0) + BETA_ALPHA)
        / (work["person_n_prior"] + BETA_ALPHA + BETA_BETA)
    )
    return work


def build_composition(person: pd.DataFrame) -> pd.DataFrame:
    """تجمیع تاریخچه‌ی رزروکنندگان به سطح (روز، وعده، سلف)."""
    p = person.copy()
    p["_has_hist"] = (p["person_n_prior"] >= MIN_HIST).astype(float)
    p["_high_risk"] = (p["person_shrunk_rate"] >= HIGH_RISK_RATE).astype(float)

    keys = ["date_gregorian", "meal", "restaurant"]
    comp = p.groupby(keys, observed=True).agg(
        composition_n=("person_shrunk_rate", "size"),
        composition_mean=("person_shrunk_rate", "mean"),
        composition_std=("person_shrunk_rate", "std"),
        composition_p90=("person_shrunk_rate", lambda s: s.quantile(0.90)),
        composition_high_risk_share=("_high_risk", "mean"),
        composition_coverage=("_has_hist", "mean"),
    ).reset_index()
    return comp


def add_day_shock(features: pd.DataFrame) -> pd.DataFrame:
    """شوک روزِ آخرین وعده‌ی مجاز، نسبت به میانگین تاریخی سراسری.

    مخرج و صورت **هر دو** پیش از لحظه‌ی برش‌اند: `venue_last_rho` از
    `last_value_at_cutoff` می‌آید و `global_hist_rho` از `expanding_stat_at_cutoff`.
    این همان چیزی است که برهم‌کنش ضرب‌شونده‌ی F62 به آن نیاز دارد.
    """
    out = features.copy()
    out["day_shock_lag1"] = out["venue_last_rho"] - out["global_hist_rho"]
    return out


def add_composition(features: pd.DataFrame, comp: pd.DataFrame) -> pd.DataFrame:
    """الحاق به ماتریس ویژگی + برهم‌کنش ضرب‌شونده‌ی بند ۵.۱۸."""
    out = features.merge(comp, on=["date_gregorian", "meal", "restaurant"],
                         how="left", validate="many_to_one")
    out = add_day_shock(out)

    # ⭐ برهم‌کنش ضرب‌شونده (F62): در روزهای بد، بی‌ثبات‌ها ۳.۳۶ برابر بیشتر می‌ریزند.
    # جمع‌شونده نیست، پس خودِ حاصل‌ضرب باید فیچر باشد نه دو ستون جدا.
    for col in ("composition_mean", "composition_p90", "composition_high_risk_share"):
        out[f"{col}_x_dayshock"] = out[col] * out["day_shock_lag1"]
    return out


COMPOSITION_FEATURES = [
    "composition_n", "composition_mean", "composition_std", "composition_p90",
    "composition_high_risk_share", "composition_coverage", "day_shock_lag1",
    "composition_mean_x_dayshock", "composition_p90_x_dayshock",
    "composition_high_risk_share_x_dayshock",
]


def build(output_path=OUTPUT_PATH, *, exclude_anomalies: bool = True) -> pd.DataFrame:
    from src.features.build_v4 import anomaly_mask

    person = load_person_level()
    person, _ = apply_domain_filters(person)
    logger.info("سطح فرد پس از فیلتر دامنه: %d ردیف", len(person))

    # پنجره‌ی شوک از **تاریخچه‌ی فرد** هم برداشته می‌شود، به همان دلیل شبکه: نرخ
    # انبساطی است، پس «نگرفتنِ غذا در روزهای تخلیه» تا آخر پنجره روی نرخ فرد
    # می‌ماند و او را برای سال بعد پرریسک نشان می‌دهد (ردیف ۸۰ `decision_log`).
    if exclude_anomalies:
        drop = anomaly_mask(person)
        if drop.any():
            logger.info("پنجره‌ی شوک از سطح فرد حذف شد: %d رزرو", int(drop.sum()))
            person = person[~drop].reset_index(drop=True)

    person = person_history_at_cutoff(person)
    comp = build_composition(person)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    comp.to_parquet(output_path, index=False)
    logger.info("فیچر ترکیب ذخیره شد: %s (%d سلول سلف×وعده×روز)", output_path, len(comp))
    return comp


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    from src.config import set_global_seed
    from src.features.build_v4 import OUTPUT_PATH as FEATURES_PATH

    set_global_seed()
    comp = build()

    features = pd.read_parquet(FEATURES_PATH)
    # ماتریسِ روی دیسک ممکن است از اجرای قبلی ستون ترکیب داشته باشد؛ بدون این حذف،
    # merge پسوند `_x`/`_y` می‌سازد و برهم‌کنش‌ها روی ستون ناموجود می‌افتند.
    features = features.drop(columns=[c for c in COMPOSITION_FEATURES if c in features.columns])
    merged = add_composition(features, comp)

    pd.set_option("display.width", 200)
    print(f"\nپوشش الحاق: {100 * merged.composition_mean.notna().mean():.1f}٪")
    print("\n=== همبستگی با هدف، کل و به تفکیک وعده (آزمون دوباره‌ی F63) ===")
    rows = []
    for col in COMPOSITION_FEATURES:
        rows.append({
            "feature": col,
            "coverage_%": round(100 * merged[col].notna().mean(), 1),
            "r_all": round(float(merged[col].corr(merged["rho"])), 3),
            "r_lunch": round(float(merged[merged.meal == "lunch"][col].corr(
                merged[merged.meal == "lunch"]["rho"])), 3),
            "r_dinner": round(float(merged[merged.meal == "dinner"][col].corr(
                merged[merged.meal == "dinner"]["rho"])), 3),
        })
    print(pd.DataFrame(rows).to_string(index=False))
