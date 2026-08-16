"""سطح L4 (سری $(m,r)$، ۴۱ سری) برای خ۴ چندمتغیره/پنلی (بند 7.13 WBS).

برخلاف L3 (`day_shock`، انحراف)، هدف اینجا **خودِ نرخ $\\rho$** است — چون DFM دقیقاً
هم‌حرکتی سطحِ سری‌ها را مدل می‌کند (F60: همبستگی باقیمانده‌ی جفت‌سلف=+۰.۴۲۱، PC1=۴۰.۵٪)،
نه انحراف هرکدام از عادت خودش.

خروجی یک پنل **wide** (ایندکس=تاریخ روزانه‌ی کامل، ستون=هر (RestaurantName, Meal))
است — قالب مورد نیاز `statsmodels.tsa.statespace.dynamic_factor_mq.DynamicFactorMQ`
که NaN را بومی می‌پذیرد (طراحی‌شده برای داده‌ی گمشده/فرکانس‌آمیخته).
"""

import pandas as pd

from src.cv import DATE_COL
from src.features.build import FEATURES_A_PATH

SEP = "__"


def build_l4_panel() -> pd.DataFrame:
    """برمی‌گرداند DataFrame wide: ایندکس=تاریخ روزانه‌ی کامل، ۴۱ ستون
    ``{RestaurantName}__{Meal}``، مقدار=میانگین $\\rho$ آن (d,m,r) (میانگین روی
    FoodName، چون L4 در سطح سلف×وعده است نه غذا — F39)."""
    df = pd.read_parquet(FEATURES_A_PATH)
    series_id = df["RestaurantName"].astype(str) + SEP + df["Meal"].astype(str)
    wide = df.assign(series_id=series_id).pivot_table(
        index=DATE_COL, columns="series_id", values="rho", aggfunc="mean")
    full_dates = pd.date_range(df[DATE_COL].min(), df[DATE_COL].max(), freq="D")
    wide = wide.reindex(full_dates)
    wide.index.name = DATE_COL
    return wide


def series_columns(wide: pd.DataFrame) -> list[str]:
    return list(wide.columns)
