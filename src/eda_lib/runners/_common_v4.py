"""بارگذاری مشترک داده‌ی فاز ۱۴ (پنجره‌ی v4) و ابزار چاپ.

همه‌ی اجراکننده‌های دور ۷ داده را از اینجا می‌گیرند تا نسخه‌ی داده در **یک** نقطه
تعریف شود. تفاوت بنیادی با `_common.py` (فاز ۴): آنجا مرجع `dataset_v2.csv` بود —
۱۴۲ روز، نگاشت ۳۱ سلفه. اینجا شبکه‌ی v4 است — ۵۹۳ روز، ۷۱ سلف واقعی.

⚠️ ماتریس پیش‌فرض `features_L1_v4.parquet` است که پنجره‌ی شوک از آن **حذف شده**
(ردیف ۸۰، F135). تنها بندی که باید نسخه‌ی `_with_anomalies` را بخواند ۱۴-۸-۱ است.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import DATA_EXTERNAL, DATA_PROCESSED, FIGURES_DIR, set_global_seed

FEATURES_L1 = DATA_PROCESSED / "features_L1_v4.parquet"
FEATURES_L1_ANOM = DATA_PROCESSED / "features_L1_v4_with_anomalies.parquet"
GRID_L1 = DATA_PROCESSED / "grid_L1_v4.parquet"
GRID_L2 = DATA_PROCESSED / "grid_L2_v4.parquet"
GRID_L3 = DATA_PROCESSED / "grid_L3_v4.parquet"
COMPOSITION_L2 = DATA_PROCESSED / "composition_L2_v4.parquet"
RESTAURANT_DIM = DATA_EXTERNAL / "restaurant_dim_v4.csv"
CALENDAR = DATA_EXTERNAL / "calendar_tehran_v4.csv"
EVENTS = DATA_EXTERNAL / "events_v4.csv"
WEATHER = DATA_EXTERNAL / "weather_aqi_by_city_v4.csv"

ROUND7_FIGURES = FIGURES_DIR / "round7"

DOW_FA = {0: "شنبه", 1: "یکشنبه", 2: "دوشنبه", 3: "سه‌شنبه", 4: "چهارشنبه", 5: "پنجشنبه", 6: "جمعه"}
MEAL_FA = {"lunch": "ناهار", "dinner": "شام"}


def load_features(with_anomalies: bool = False, columns: list[str] | None = None) -> pd.DataFrame:
    """ماتریس ویژگی L1 پنجره‌ی v4.

    `with_anomalies=True` فقط برای بند ۱۴-۸-۱ مجاز است — همان‌جا که پنجره‌ی شوک
    **موضوع** مطالعه است، نه آلودگی.
    """
    path = FEATURES_L1_ANOM if with_anomalies else FEATURES_L1
    return pd.read_parquet(path, columns=columns)


def load_grid(level: str = "L1") -> pd.DataFrame:
    return pd.read_parquet({"L1": GRID_L1, "L2": GRID_L2, "L3": GRID_L3}[level])


def load_restaurant_dim(core_only: bool = True) -> pd.DataFrame:
    dim = pd.read_csv(RESTAURANT_DIM)
    return dim[dim.restaurant_class == "core"] if core_only else dim


def add_legacy_name(df: pd.DataFrame, col: str = "restaurant") -> pd.DataFrame:
    """ستون `legacy_restaurant` را می‌چسباند — نگاشت **مرده**‌ی ۳۱ سلفه‌ی فاز ۴.

    تنها کاربرد مجازش بند ۱۴-۳-۲ است: بازمحاسبه‌ی یک آماره زیر هر دو نگاشت روی
    **همان داده**، تا تفاوت، مصنوعِ نگاشت باشد نه مصنوعِ پنجره. سلفی که نگاشت قدیمی
    اصلاً نمی‌شناخت مقدار `«خارج از نگاشت قدیمی»` می‌گیرد — حذفش اریبی انتخاب می‌سازد.
    """
    dim = load_restaurant_dim()[["raw_name", "legacy_canonical_name"]]
    out = df.merge(dim, left_on=col, right_on="raw_name", how="left")
    out["legacy_restaurant"] = out["legacy_canonical_name"].fillna("«خارج از نگاشت قدیمی»")
    return out.drop(columns=["raw_name", "legacy_canonical_name"])


def academic_year_order() -> list[str]:
    return ["1402-1403", "1403-1404", "1404-1405"]


# ---------------------------------------------------------------------------
# چاپ
# ---------------------------------------------------------------------------

def header(title: str, level: int = 1) -> None:
    bar = {1: "=", 2: "-", 3: "."}[level]
    print(f"\n{bar * 78}\n{title}\n{bar * 78}")


def show(df: pd.DataFrame, n: int = 30, float_fmt: str = "%.4f") -> None:
    with pd.option_context("display.max_rows", n, "display.width", 200,
                           "display.max_columns", 50, "display.float_format", lambda v: float_fmt % v):
        print(df)


def md_table(df: pd.DataFrame, float_fmt: str = "{:.4f}") -> str:
    """جدول Markdown از یک DataFrame — برای چسباندن مستقیم در سند دور ۷."""
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else float_fmt.format(v))
    head = "| " + " | ".join(str(c) for c in d.columns) + " |"
    sep = "|" + "|".join(["---"] * len(d.columns)) + "|"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join([head, sep, *rows])
