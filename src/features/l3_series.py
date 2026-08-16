"""سطح L3 (عامل روز، $(d,m)$) برای خ۳ سری‌زمانی (بند 7.12 WBS).

هدف $y_{d,m}$ = ``day_shock`` (`aggregate_features.build_day_factor`، بند ۵.۱۷) — انحراف
میانگین سلف‌ها از عادت خودشان در آن روز-وعده، نه نرخ خام. این انتخاب عمدی است: خودِ نام
سطح («عامل روز») در سراسر مستندات پروژه دقیقاً به همین کمیت اشاره دارد (F59: ۸۳٪ واریانس
سلول همین شوک مشترک است)، و `build_day_factor` از قبل leakage-safe ساخته شده.

## دو تصمیم روش‌شناختی که اینجا اجرا شده‌اند

۱. **ناهار و شام جدا** (F33: دو دینامیک متفاوت) — دو سری مستقل برمی‌گردد، نه یک سری مشترک.
۲. **بدون حذف/درون‌یابی شکاف‌ها** (F35: گمشدگی MCAR نیست، ۸۰.۸٪ جمعه‌ها بدون سرو؛ F38:
   شکاف رمضان ۲۹ روزه) — ایندکس تاریخ **روزانه‌ی کامل** ساخته می‌شود و روزهای بدون سرویس
   ``NaN`` صریح می‌گیرند؛ فیلتر کالمن `statsmodels` این را بومی مدیریت می‌کند (بند 7.12.1).

⚠️ رگرسور برون‌زا فقط از `calendar_tehran.csv` (بند 7.9.2، پوشش کامل ۲۴۳ روز تقویمی
بدون هیچ وابستگی به سرویس‌دهی) گرفته می‌شود — نه $\\log Res$/`day_shock_lag1`/`is_ramadan`/
`is_snow_day` که فقط در `FEATURES_A_PATH` و فقط برای روزهای **سرویس‌داده‌شده** موجودند.
SARIMAX به exog کامل روی کل افق پیش‌بینی نیاز دارد؛ این محدودیت دامنه‌ی فهرست کوتاه
است، نه نقص پیاده‌سازی.
"""

import pandas as pd

from src.cv import DATE_COL
from src.eda_lib.runners._common import CALENDAR_PATH
from src.features.aggregate_features import build_day_factor
from src.features.build import FEATURES_A_PATH

#: از `calendar_tehran.csv` — تعریف‌شده برای هر ۲۴۳ روز تقویمی، بدون هیچ NaN
CALENDAR_EXOG = ["is_holiday_any", "is_day_before_holiday", "is_exam_period",
                "is_final_exam_period", "is_nowruz_block"]

MEALS = ("lunch", "dinner")


def build_l3_series() -> dict[str, pd.DataFrame]:
    """برمی‌گرداند ``{"lunch": df, "dinner": df}`` — هرکدام با ایندکس تاریخ روزانه‌ی
    کامل (بدون شکاف)، ستون‌های ``day_shock`` (هدف، NaN روی روزهای بدون سرویس)،
    ``n_cells``، و رگرسور تقویمی (هرگز NaN)."""
    df = pd.read_parquet(FEATURES_A_PATH).sort_values(DATE_COL).reset_index(drop=True)
    cell = df.rename(columns={"rho": "rho_cell"})
    day = build_day_factor(cell)[[DATE_COL, "Meal", "day_shock", "n_cells"]]

    cal_raw = pd.read_csv(CALENDAR_PATH, parse_dates=[DATE_COL])
    calendar = cal_raw[[DATE_COL, *CALENDAR_EXOG]].drop_duplicates(subset=DATE_COL).set_index(DATE_COL)
    full_dates = pd.date_range(df[DATE_COL].min(), df[DATE_COL].max(), freq="D")
    calendar = calendar.reindex(full_dates)
    if calendar[CALENDAR_EXOG].isna().any().any():
        raise ValueError("رگرسور تقویمی نباید NaN داشته باشد — مغایر با مستندسازی این ماژول")

    out = {}
    for meal in MEALS:
        sub = day[day["Meal"] == meal].set_index(DATE_COL)[["day_shock", "n_cells"]]
        sub = sub.reindex(full_dates)
        merged = sub.join(calendar)
        merged.index.name = DATE_COL
        out[meal] = merged.reset_index()
    return out
