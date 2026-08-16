"""اسپرینت B — بند 7.9.1 بازنویسی‌شده (`doc/decisions/37-phase7-rescope.md` بند ۵):
پل L5→L1. چون رزرو ۷۲ ساعت زودتر بسته می‌شود و لغو ممکن نیست (بند ۴-۲ سند تعریف
مسئله)، در لحظه‌ی برش هر سلولِ هدف $(d,m,r)$ دقیقاً می‌دانیم **چه کسانی** رزرو
کرده‌اند. تاریخچه‌ی رفتاری آن‌ها (که خودش leakage-safe است — بند پایین) به یک
کوواریت **آینده‌ی معلوم** سطح سلول تبدیل می‌شود، بدون نیاز به مدل کامل سطح فرد و
بدون هزینه‌ی تجمیع پواسون-دوجمله‌ای (بند 7.24.3).

⚠️ **چرا این از ساختن یک مدل کامل L5 ارزان‌تر و امن‌تر است** (بند ۵ سند تصمیم ۳۷):
مدل L5 کامل استقلال افراد را فرض می‌کند، ولی F59 می‌گوید ۸۳٪ واریانس سلول شوکِ
**مشترک** روزانه است — یعنی خطاهای فردی همبسته‌اند و عدم‌قطعیت تجمیعی کم‌برآورد
می‌شود. اینجا این مشکل اصلاً مطرح نیست چون فیچر کوهورت مستقیماً در مدل L1 مصرف
می‌شود، نه این‌که خودش خروجی نهایی باشد.

## ایمنی نشت (leakage safety)

`person_features_v1.parquet::person_expanding_norecv_rate` قبلاً برای «مدل B»
(طبقه‌بندی سطح فرد) با قاعده‌ی برش استاندارد پروژه ساخته شده — expanding‌ **قبل**
از رزرو جاری (تأیید عملی: `person_n_prior_reservations=0` و
`person_expanding_norecv_rate=NaN` روی اولین رزرو هر فرد؛ `person_n_prior_*`
درون یک روز هم به‌ترتیب ناهار→شام درست افزایش می‌یابد، منطبق با لحظه‌ی برش هر
وعده). تجمیع اینجا فقط این ستون‌های از-قبل-ایمن را روی هر سلول $(d,m,r)$ میانگین
می‌گیرد — هیچ اطلاعات جدیدی از آینده وارد نمی‌شود.
"""

import numpy as np
import pandas as pd

PERSON_FEATURES_PATH = "data/processed/person_features_v1.parquet"

#: بند ۵ سند تصمیم ۳۷ — آستانه‌ی «مزمن» (نرخ عدم‌دریافت تاریخی بالا)
_CHRONIC_THRESHOLD = 0.3


def _binary_entropy(p: pd.Series) -> pd.Series:
    p = p.clip(1e-6, 1 - 1e-6)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def build_cohort_features(person_path: str = PERSON_FEATURES_PATH) -> pd.DataFrame:
    """یک ردیف به‌ازای هر $(d,m,r)$ — سطح L2 (سلف×وعده)، چون رزرو در این سطح ثبت
    می‌شود (فیچر فردی `FoodType` ندارد). در L1 با merge روی `(date_gregorian, Meal,
    RestaurantName)` به همه‌ی سطرهای هم‌غذا اعمال می‌شود."""
    person = pd.read_parquet(person_path)
    keys = ["date_gregorian", "Meal", "restaurant_canonical"]

    g = person.groupby(keys, observed=True)
    rate = person["person_expanding_norecv_rate"]

    out = g.agg(
        cohort_n_people=("PersonId", "count"),
        cohort_norecv_mean=("person_expanding_norecv_rate", "mean"),
        cohort_norecv_p75=("person_expanding_norecv_rate", lambda s: s.quantile(0.75)),
        cohort_norecv_std=("person_expanding_norecv_rate", "std"),
        cohort_coldstart_share=("is_cold_start", "mean"),
        cohort_tenure_mean=("person_n_prior_reservations", "mean"),
        cohort_dorm_share=("is_dorm_resident", "mean"),
    ).reset_index()

    chronic = (rate > _CHRONIC_THRESHOLD).groupby([person[k] for k in keys], observed=True).mean()
    out = out.merge(chronic.rename("cohort_chronic_share").reset_index(), on=keys, how="left")

    out["cohort_dorm_entropy"] = _binary_entropy(out["cohort_dorm_share"])
    out = out.rename(columns={"restaurant_canonical": "RestaurantName"})

    # سلول‌هایی که همه‌ی رزروکنندگانشان cold-start بودند ⇒ mean/std/p75 روی رزرو
    # NaN می‌شود (چون person_expanding_norecv_rate خودش NaN است) — با ۰ پر می‌شود
    # (فرض محافظه‌کارانه: بدون شواهد خلاف، نرخ پیش‌فرض)
    for c in ["cohort_norecv_mean", "cohort_norecv_p75", "cohort_norecv_std", "cohort_chronic_share"]:
        out[c] = out[c].fillna(0.0)
    return out


def merge_cohort_onto_l1(l1: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    """LEFT JOIN روی (date_gregorian, Meal, RestaurantName) — سلول‌های L1 بدون
    رزرو-فردی‌ثبت‌شده (باید نادر باشد، طبق F53) با میانه‌ی ستون پر می‌شوند."""
    merged = l1.merge(cohort, on=["date_gregorian", "Meal", "RestaurantName"], how="left")
    cohort_cols = [c for c in cohort.columns if c not in ("date_gregorian", "Meal", "RestaurantName")]
    n_missing = int(merged[cohort_cols[0]].isna().sum())
    for c in cohort_cols:
        merged[c] = merged[c].fillna(merged[c].median())
    return merged, n_missing


COHORT_FEATURE_COLS = [
    "cohort_n_people", "cohort_norecv_mean", "cohort_norecv_p75", "cohort_norecv_std",
    "cohort_coldstart_share", "cohort_chronic_share", "cohort_tenure_mean",
    "cohort_dorm_share", "cohort_dorm_entropy",
]
