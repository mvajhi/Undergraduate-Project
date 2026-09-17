"""ماتریس ویژگی روی شبکه‌ی v4 — شش خانواده، همه مقید به قاعده‌ی برش اطلاعاتی.

**قاعده‌ی حاکم، بدون استثنا.** هر فیچر تاریخی از `src/features/cutoff.py` می‌گذرد، که
قاعده‌ی برش را به یک نامساوی واحد فرو می‌ریزد
($\\text{seq}_{obs} \\le \\text{seq}_{target} - 2$). هیچ فیچری اینجا با
`shift()` یا `expanding()` خام ساخته نمی‌شود — همان الگویی که در دور ۳ نشتی ساخت و در
`doc/leakage_audit.md` کمّی شد.

**چه چیزی در لحظه‌ی برش معلوم است و شهود خلافش را می‌گوید.** `Res` (تعداد رزرو) برای
روز هدف **معلوم است**، چون رزرو ۷۲ ساعت زودتر بسته می‌شود (بند ۴-۲ سند مسئله). پس
مدل «چند نفر رزرو کرده‌اند» را می‌داند و فقط «چند درصدشان نمی‌آیند» را پیش‌بینی می‌کند.
همین `Res` قوی‌ترین فیچر غیرتاریخی است و نبودنش در یک ماتریس ویژگی، خطای طراحی است نه
احتیاط.

| خانواده | نمونه | ریسک زمانی |
|---|---|---|
| تقویمی | `dow`, `week_of_semester`, `is_ramadan` | ندارد — از پیش معلوم |
| تقاضای معلوم | `Res`, `n_people`, `portions_per_person` | ندارد — رزرو ۷۲ ساعت زودتر بسته می‌شود |
| موجودیت | `campus`, `venue_type`, `food_group` | ندارد — ثابت |
| تاریخی | `cell_hist_rho`, `venue_dow_hist_rho`, `rho_lag1` | **مقید به `cutoff.py`** |
| شوک تقاضا | `res_vs_venue_hist` | مقید — مخرج تاریخی است |
| هواشناسی | `temp_mean`, `aqi_us_max` | ⚠️ **مقدار واقعی روز $d$** — در استقرار باید پیش‌بینی باشد |

⚠️ خانواده‌ی هواشناسی عمداً با همان هشدار فاز ۵ نگه داشته شد: مقدار واقعیِ روز هدف در
لحظه‌ی برش در دسترس **نیست**، ولی پیش‌بینی هواشناسی هست. نگه‌داشتنش با برچسب صریح،
قابل‌دفاع‌تر از حذف بی‌صداست؛ فیچرست `no_weather` برای سنجش سهمش هست.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import DATA_EXTERNAL, DATA_PROCESSED
from src.features.cutoff import (
    expanding_stat_at_cutoff,
    last_value_at_cutoff,
    meal_seq,
    same_meal_lag,
)
from src.data.build_grid_v4 import OUTPUT_PATHS as GRID_PATHS
from src.data.food_dim_v4 import OUTPUT_PATH as FOOD_DIM_PATH
from src.data.restaurant_dim_v4 import OUTPUT_PATH as RESTAURANT_DIM_PATH

logger = logging.getLogger(__name__)

CALENDAR_PATH = DATA_EXTERNAL / "calendar_tehran_v4.csv"
WEATHER_PATH = DATA_EXTERNAL / "weather_aqi_by_city_v4.csv"
OUTPUT_PATH = DATA_PROCESSED / "features_L1_v4.parquet"

CALENDAR_COLUMNS = [
    "date_gregorian", "day_of_week", "month_jalali", "day_of_month_jalali", "year_jalali",
    "is_friday", "is_holiday_national", "is_named_holiday", "is_holiday_any",
    "days_to_next_holiday", "days_since_last_holiday", "is_bridge_day",
    "is_day_before_holiday", "is_day_after_holiday", "holiday_block_length",
    "academic_year", "semester", "week_of_semester", "semester_progress", "is_in_session",
    "is_add_drop_period", "is_midterm_period", "is_final_exam_period", "is_exam_period",
    "is_evaluation_period", "days_to_exam_start", "days_since_semester_start",
    "days_to_semester_end", "is_nowruz_block", "is_inter_semester_break", "is_summer_break",
    "is_ramadan", "days_into_ramadan",
]

WEATHER_COLUMNS = [
    "temp_mean", "temp_min", "temp_max", "feels_like_min",
    "precipitation_sum", "snowfall_sum", "wind_speed_max",
    "relative_humidity_mean", "aqi_us_max",
]

#: شکاف ساختاری خرداد/تیر ۱۴۰۴ (یافته F108، ردیف ۷۰ `decision_log`).
#: بازه از نخستین روزِ ریزش شروع می‌شود نه از نخستین روز صفر — روزهای ۲۰۲۵-۰۶-۱۶ تا
#: ۲۰۲۵-۰۶-۲۰ در شبکه **هستند** (با حجم ۳٬۳۵۱ تا ۱۱ پرس) و بدون پرچم، مدل آن‌ها را
#: روزهای عادیِ کم‌تقاضا می‌بیند.
#: ⚠️ علت ثبت نشده — پرچم ساختاری است، نه انتساب علّی.
STRUCTURAL_GAPS: list[tuple[str, str]] = [
    ("2025-06-16", "2025-07-03"),
]


def _load_grid(level: str = "L1") -> pd.DataFrame:
    grid = pd.read_parquet(GRID_PATHS[level])
    grid["date_gregorian"] = pd.to_datetime(grid["date_gregorian"])
    return grid


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    cal = pd.read_csv(CALENDAR_PATH, usecols=CALENDAR_COLUMNS)
    cal["date_gregorian"] = pd.to_datetime(cal["date_gregorian"])
    out = df.merge(cal, on="date_gregorian", how="left")
    if out["day_of_week"].isna().any():
        raise ValueError("روزهایی خارج از پنجره‌ی تقویم v4 در شبکه هست")
    return out


def add_entity_features(df: pd.DataFrame) -> pd.DataFrame:
    restaurants = pd.read_csv(RESTAURANT_DIM_PATH)[
        ["raw_name", "campus", "city", "venue_type", "gender", "is_tehran",
         "distance_km_to_tehran_campus"]
    ]
    foods = pd.read_csv(FOOD_DIM_PATH)
    # سطح غذا در شبکه `canonical_dish` است، پس ویژگی‌های غذا هم باید در همان سطح
    # جمع شوند: رده‌ی آشپزی ثابت است و میانگین تعداد مخلفات وزن‌دار با حجم.
    dish = foods.groupby("canonical_dish").apply(
        lambda g: pd.Series({
            "food_group": g["food_group"].iloc[0],
            "n_extras_mean": np.average(g["n_extras"], weights=g["portions"].clip(lower=1)),
            "dish_total_portions": g["portions"].sum(),
        }),
        include_groups=False,
    ).reset_index()

    out = df.merge(restaurants, left_on="restaurant", right_on="raw_name", how="left")
    if "dish" in out.columns:
        out = out.merge(dish, left_on="dish", right_on="canonical_dish", how="left")
    return out.drop(columns=[c for c in ("raw_name", "canonical_dish") if c in out.columns])


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    weather = pd.read_csv(WEATHER_PATH, usecols=["city", "date_gregorian", *WEATHER_COLUMNS])
    weather["date_gregorian"] = pd.to_datetime(weather["date_gregorian"])
    out = df.merge(weather, on=["city", "date_gregorian"], how="left")
    missing = out["temp_mean"].isna().mean()
    if missing > 0:
        raise ValueError(f"{missing:.2%} ردیف بدون هواشناسی — شهر/تاریخ نمی‌خواند")
    return out


def add_known_demand_features(df: pd.DataFrame) -> pd.DataFrame:
    """تقاضای **معلوم در لحظه‌ی برش** — رزرو ۷۲ ساعت زودتر بسته می‌شود."""
    out = df.copy()
    out["portions_per_person"] = out["Res"] / out["n_people"].clip(lower=1)
    out["log_res"] = np.log1p(out["Res"])
    return out


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """فیچرهای تاریخی — **همه** از `cutoff.py` می‌گذرند.

    چهار سطح تجمیع، از خاص به عام. سطح خاص دقیق‌تر است ولی کم‌نمونه؛ سطح عام همیشه
    پر است. مدل خودش وزن‌دهی می‌کند، ولی هر چهار سطح باید موجود باشد چون **۳۶.۹٪
    سلول‌های L1 کمتر از ۱۰ رزرو دارند** (F104) و برای آن‌ها تاریخچه‌ی سطح سلول نویز است.
    """
    out = df.sort_values(["date_gregorian", "meal"], kind="stable").reset_index(drop=True)
    out["meal_seq"] = meal_seq(out["date_gregorian"], out["meal"])

    # `NoRecv` جمع می‌شود و `Res` هم، تا نرخ تاریخی **وزن‌دار با حجم** باشد نه
    # میانگینِ نرخ‌ها — سلولی با ۳ رزرو نباید هم‌وزن سلولی با ۳۰۰ رزرو باشد.
    levels: dict[str, list[str]] = {
        "cell": ["restaurant", "dish", "meal"],
        "venue_dow": ["restaurant", "meal", "day_of_week"],
        "venue": ["restaurant", "meal"],
        "dish": ["dish", "meal"],
        "global": ["meal"],
    }
    for name, keys in levels.items():
        keys = [k for k in keys if k in out.columns]
        no_recv = expanding_stat_at_cutoff(out, keys, "NoRecv", out_prefix=f"_{name}_nr")
        res = expanding_stat_at_cutoff(out, keys, "Res", out_prefix=f"_{name}_res")
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"{name}_hist_rho"] = np.where(
                res[f"_{name}_res_sum"] > 0,
                no_recv[f"_{name}_nr_sum"] / res[f"_{name}_res_sum"],
                np.nan,
            )
        out[f"{name}_hist_n"] = res[f"_{name}_res_count"]
        out[f"{name}_hist_res_mean"] = res[f"_{name}_res_mean"]

    # شوک تقاضا: رزروِ **معلومِ** امروز نسبت به میانگین تاریخی همان سلف. مخرج تاریخی
    # است و صورت در لحظه‌ی برش معلوم — پس نشتی ندارد.
    out["res_vs_venue_hist"] = out["Res"] / out["venue_hist_res_mean"].replace(0, np.nan)
    out["res_vs_cell_hist"] = out["Res"] / out["cell_hist_res_mean"].replace(0, np.nan)

    # آخرین سروِ همین غذا در همین سلف — و اینکه چند روز پیش بوده.
    # در سطح سلول این جای `same_meal_lag` را می‌گیرد: یک غذا هر دو تا چهار هفته
    # یک‌بار سرو می‌شود، پس فیلترِ فاصله ۸۵٪ ردیف‌ها را خالی می‌کرد (پوشش ۱۴.۷٪).
    # اینجا فاصله از فیلتر به **فیچر** تبدیل می‌شود.
    # ⚠️ فقط در سطح سلول معنا دارد. کلید L1 «(روز، وعده، سلف، غذا)» یکتاست، پس
    # «آخرین مشاهده» تعریف روشنی دارد. همین تابع روی گروه «(سلف، وعده)» **بی‌معنا**
    # می‌شود، چون چند غذا در یک روز هم‌`meal_seq`اند و «آخرین» یکی از آن‌ها را
    # دلبخواهی برمی‌دارد؛ ممیزی ساختاری همین را با خطای ۱.۰ گرفت. نسخه‌ی سطح سلف
    # پایین‌تر، روی نرخِ **وزن‌دار با حجم** سلف ساخته می‌شود.
    out = pd.concat([
        out,
        last_value_at_cutoff(out, ["restaurant", "dish", "meal"], ["rho", "Res"], out_prefix="cell_last"),
    ], axis=1)

    # lag در توالی هم‌وعده — آستانه‌ی فاصله به‌ازای هر lag، نه سراسری (تله‌ی رمضان).
    # در سطح سلف پوشش بالاست چون هر سلف تقریباً هر روز سرو دارد.
    lags = same_meal_lag(out, ["restaurant", "dish", "meal"], "rho", [1, 7])
    out = pd.concat([out, lags.add_prefix("cell_")], axis=1)
    venue_rho = (
        out.groupby(["restaurant", "meal", "date_gregorian"], observed=True)
        .agg(_nr=("NoRecv", "sum"), _res=("Res", "sum")).reset_index()
    )
    venue_rho["venue_rho"] = venue_rho["_nr"] / venue_rho["_res"].replace(0, np.nan)
    venue_rho["meal_seq"] = meal_seq(
        venue_rho["date_gregorian"], venue_rho["meal"], origin=out["date_gregorian"].min()
    )
    venue_lags = same_meal_lag(venue_rho, ["restaurant", "meal"], "venue_rho", [1, 7])
    venue_last = last_value_at_cutoff(
        venue_rho, ["restaurant", "meal"], ["venue_rho"], out_prefix="venue_last"
    ).rename(columns={"venue_last_venue_rho": "venue_last_rho"})
    venue_rho = pd.concat([venue_rho, venue_lags, venue_last], axis=1)
    out = out.merge(
        venue_rho[["restaurant", "meal", "date_gregorian", "venue_rho_lag1", "venue_rho_lag7",
                   "venue_last_rho", "venue_last_days_since"]],
        on=["restaurant", "meal", "date_gregorian"], how="left",
    )

    return out.drop(columns=[c for c in out.columns if c.startswith("_")])


def add_regime_flags(df: pd.DataFrame) -> pd.DataFrame:
    """پرچم شکاف ساختاری (F108) — بدون انتساب علّی."""
    out = df.copy()
    flag = pd.Series(False, index=out.index)
    for start, end in STRUCTURAL_GAPS:
        flag |= out["date_gregorian"].between(pd.Timestamp(start), pd.Timestamp(end))
    out["is_structural_gap"] = flag
    return out


def build_features(level: str = "L1", output_path=OUTPUT_PATH) -> pd.DataFrame:
    df = _load_grid(level)
    df = add_calendar_features(df)
    df = add_entity_features(df)
    df = add_weather_features(df)
    df = add_known_demand_features(df)
    df = add_history_features(df)
    df = add_regime_flags(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info("ماتریس ویژگی ذخیره شد: %s (%d ردیف × %d ستون)", output_path, *df.shape)
    return df


#: فیچرست‌ها. `no_weather` عمداً هست تا سهم خانواده‌ی هواشناسی — تنها خانواده‌ای که در
#: استقرار به پیش‌بینی نیاز دارد — قابل‌سنجش باشد، نه فرض‌شده.
FEATURE_SETS: dict[str, list[str]] = {
    "calendar": [
        "day_of_week", "month_jalali", "day_of_month_jalali", "is_holiday_any",
        "days_to_next_holiday", "days_since_last_holiday", "is_bridge_day",
        "is_day_before_holiday", "is_day_after_holiday", "week_of_semester",
        "semester_progress", "is_exam_period", "is_in_session", "is_ramadan",
        "days_into_ramadan", "days_to_exam_start",
    ],
    "entity": [
        "campus", "venue_type", "gender", "is_tehran", "distance_km_to_tehran_campus",
        "food_group", "n_extras_mean", "dish_total_portions", "meal",
    ],
    "known_demand": ["Res", "log_res", "n_people", "portions_per_person"],
    "history": [
        "cell_hist_rho", "cell_hist_n", "venue_hist_rho", "venue_hist_n",
        "venue_dow_hist_rho", "dish_hist_rho", "global_hist_rho",
        "cell_last_rho", "cell_last_Res", "cell_last_days_since",
        "venue_last_rho", "venue_last_days_since",
        "cell_rho_lag1", "cell_rho_lag7", "venue_rho_lag1", "venue_rho_lag7",
        "res_vs_venue_hist", "res_vs_cell_hist",
    ],
    "weather": WEATHER_COLUMNS,
    "regime": ["is_structural_gap"],
}


def feature_set(name: str) -> list[str]:
    """`full` = همه · `no_weather` = همه جز هواشناسی."""
    if name == "full":
        return [c for cols in FEATURE_SETS.values() for c in cols]
    if name == "no_weather":
        return [c for key, cols in FEATURE_SETS.items() if key != "weather" for c in cols]
    return FEATURE_SETS[name]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    from src.config import set_global_seed

    set_global_seed()
    feats = build_features()

    pd.set_option("display.width", 200)
    print(f"\n{feats.shape[0]:,} ردیف × {feats.shape[1]} ستون")

    print("\n=== پوشش فیچرهای تاریخی (NaN یعنی هنوز تاریخچه نداشته) ===")
    hist = [c for c in feature_set("history")]
    cov = pd.DataFrame({
        "coverage_pct": (100 * feats[hist].notna().mean()).round(1),
        "corr_with_rho": [round(float(feats[c].corr(feats["rho"])), 3) for c in hist],
    })
    print(cov.to_string())

    print("\n=== همبستگی فیچرهای کلیدی با هدف ===")
    keys = ["Res", "log_res", "portions_per_person", "res_vs_venue_hist",
            "temp_mean", "aqi_us_max", "week_of_semester", "days_to_exam_start"]
    print(pd.Series(
        {k: round(float(feats[k].corr(feats["rho"])), 3) for k in keys}
    ).sort_values().to_string())

    print("\n=== rho بر حسب سال تحصیلی (ورودی بند ۱۳-۹) ===")
    print(feats.groupby("academic_year").apply(
        lambda g: pd.Series({
            "cells": len(g), "portions": g.Res.sum(), "rho": g.NoRecv.sum() / g.Res.sum(),
        }), include_groups=False,
    ).round(4).to_string())
