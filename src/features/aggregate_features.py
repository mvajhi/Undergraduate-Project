"""بندهای ۵.۱ تا ۵.۹ و ۵.۱۷ — فیچرهای سطح تجمیعی $(d,m,r,f)$ برای مدل A.

هر خانواده‌ی فیچر به ردیف پشتیبانش در `doc/data_facts_register.md` ارجاع دارد. فیچرهایی
که فاز ۴ رد کرد (AQI، فاصله‌ی تکرار غذا، `has_extras`، …) اینجا **ساخته نمی‌شوند** —
فهرست کامل در بند ۵.۰ WBS.

⭐ مهم‌ترین بخش این ماژول `build_day_factor` (بند ۵.۱۷) است: ۸۳٪ واریانس نرخ سلول شوک
مشترک روزانه است (F59) و همین شوک با اطلاعات لحظه‌ی برش R²خارج‌نمونه=۰.۶۰ دارد (F61).
"""

import logging

import numpy as np
import pandas as pd

from src.eda_lib.runners._common import CALENDAR_PATH, EVENTS_PATH, WEATHER_BY_CITY_PATH
from src.features.cutoff import (
    CUTOFF_LAG,
    expanding_stat_at_cutoff,
    meal_seq,
    same_meal_lag,
    shrunk_rate,
)

logger = logging.getLogger(__name__)

BETA_ALPHA, BETA_BETA = 0.9758, 11.1506

#: تأخیرهای هم‌وعده. فاز ۴ (F33/F34) نشان داد ناهار و شام دینامیک متفاوت دارند؛
#: هر دو خانواده ساخته می‌شود و انتخاب نهایی به فاز ۷ سپرده می‌شود، ولی lag۷ ناهار
#: با علم به اینکه مصنوع ترکیب سلف بود ساخته می‌شود (نه به‌عنوان کاندید قوی).
LAGS = [1, 2, 7, 14, 28]
#: بافر فاصله‌ی تقویمی به‌ازای هر lag (نه یک آستانه‌ی سراسری — نگاه کنید به
#: `same_meal_lag` برای اینکه چرا آستانه‌ی ثابت خودش یک منبع نشت بود)
GAP_BUFFER_DAYS = 6


# ---------------------------------------------------------------------------
# ۵.۱ تقویمی
# ---------------------------------------------------------------------------

CAL_COLS = [
    "date_gregorian", "is_holiday_any", "is_day_before_holiday", "is_day_after_holiday",
    "is_bridge_day", "days_to_next_holiday", "days_since_last_holiday",
    "holiday_block_length", "is_exam_period", "is_final_exam_period",
    "days_to_exam_start", "week_of_semester", "semester",
]


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """بند ۵.۱ — فیچرهای تقویمی. بدون ریسک نشت (همه از پیش معلوم‌اند)."""
    cal = pd.read_csv(CALENDAR_PATH, parse_dates=["date_gregorian"])
    out = df.merge(cal[CAL_COLS], on="date_gregorian", how="left", validate="many_to_one")
    out["dow"] = (out["date_gregorian"].dt.dayofweek + 2) % 7
    out["jmonth"] = out["DateReserve"].str.slice(5, 7).astype(int)
    out["day_of_month"] = out["DateReserve"].str.slice(8, 10).astype(int)

    for c in ["is_holiday_any", "is_day_before_holiday", "is_day_after_holiday",
              "is_bridge_day", "is_exam_period", "is_final_exam_period"]:
        out[c] = out[c].fillna(False).astype(int)

    # برهم‌کنش تأییدشده: شدت اثر «روز قبل از تعطیلی» با طول بلوک رابطه دارد (F19)
    out["pre_holiday_x_block_len"] = out["is_day_before_holiday"] * out["holiday_block_length"].fillna(0)

    # رمضان — فقط فلگ گزارشی، نه فیچر یادگیری‌شونده (F22)
    ev = pd.read_csv(EVENTS_PATH, parse_dates=["date_start", "date_end"])
    ram = ev[ev["event_id"].astype(str).str.contains("ramadan", na=False)]
    out["is_ramadan"] = 0
    for _, e in ram.iterrows():
        m = (out["date_gregorian"] >= e["date_start"]) & (out["date_gregorian"] <= e["date_end"])
        out.loc[m, "is_ramadan"] = 1

    # ۵.۹ فوریه‌ای — رمزگذاری پیوسته‌ی فصلی هفتگی
    for k in (1, 2, 3):
        out[f"dow_sin{k}"] = np.sin(2 * np.pi * k * out["dow"] / 7)
        out[f"dow_cos{k}"] = np.cos(2 * np.pi * k * out["dow"] / 7)
    return out


# ---------------------------------------------------------------------------
# ۵.۸ خارجی — به‌شدت کوتاه‌شده (AQI حذف، بارش دسته‌ای)
# ---------------------------------------------------------------------------

def add_external(df: pd.DataFrame) -> pd.DataFrame:
    """بند ۵.۸ — فقط `precip_type` دسته‌ای و `temp_min`.

    ⚠️ الحاق با کلید `(city, date)` نه فقط `date` (ردیف ۲۱ decision_log).
    ⚠️ AQI/PM ساخته نمی‌شود (F25 — همبستگی کاذب بین‌شهری).
    ⚠️ مقدار **واقعی** روز d در لحظه‌ی برش معلوم نیست؛ اینها به‌عنوان جای‌نگه‌دارِ
       «پیش‌بینی هواشناسی» استفاده می‌شوند و محدودیتشان در ممیزی نشت ثبت می‌شود.
    """
    w = pd.read_csv(WEATHER_BY_CITY_PATH, parse_dates=["date_gregorian"])
    w = w[["city", "date_gregorian", "temp_min", "rain_sum", "snowfall_sum", "precipitation_sum"]]
    out = df.merge(w, on=["city", "date_gregorian"], how="left", validate="many_to_one")
    out["precip_type"] = np.select(
        [out["snowfall_sum"] > 0.1, out["rain_sum"] > 0.5, out["precipitation_sum"] > 0],
        ["snow", "rain", "trace"], default="none")
    out["is_snow_day"] = (out["precip_type"] == "snow").astype(int)
    return out.drop(columns=["rain_sum", "snowfall_sum", "precipitation_sum"])


# ---------------------------------------------------------------------------
# ۵.۲ / ۵.۳ / ۵.۴ — lag، پنجره‌ی متحرک، انبساطی (همه در سطح سلف×وعده)
# ---------------------------------------------------------------------------

def build_cell_series(df: pd.DataFrame) -> pd.DataFrame:
    """سری (سلف، وعده، روز) با نرخ وزنی — واحدی که فیچرهای زمانی روی آن ساخته می‌شوند.

    چرا این سطح و نه $(m,r,f)$؟ سری‌های سطح غذا میانه‌ی **۴ نقطه** دارند و ۱۰۰٪شان زیر
    ۳۰ نقطه‌اند (F39)، پس lag اغلب NaN می‌شود. سطح (سلف، وعده) میانه‌ی ۱۹۷ نقطه دارد (F40).
    """
    cell = (df.groupby(["RestaurantName", "Meal", "date_gregorian"], as_index=False)
              .agg(Res=("Res", "sum"), NoRecv=("NoRecv", "sum")))
    cell["rho_cell"] = cell["NoRecv"] / cell["Res"]
    cell["meal_seq"] = meal_seq(cell["date_gregorian"], cell["Meal"])
    return cell.sort_values(["RestaurantName", "Meal", "date_gregorian"], kind="stable")


def add_cell_time_features(df: pd.DataFrame, cell: pd.DataFrame) -> pd.DataFrame:
    """lag هم‌وعده (۵.۲) + پنجره‌ی متحرک (۵.۳) + انبساطی (۵.۴)، الحاق‌شده به df."""
    c = cell.copy()
    key = ["RestaurantName", "Meal"]

    lag = same_meal_lag(c, key, "rho_cell", LAGS, gap_buffer_days=GAP_BUFFER_DAYS)
    c = pd.concat([c, lag], axis=1)

    grp = c.groupby(key, observed=True, sort=False)["rho_cell"]
    for w in (7, 14, 28):
        c[f"rho_roll_mean_{w}"] = grp.transform(lambda s: s.shift(1).rolling(w, min_periods=3).mean())
    c["rho_roll_std_7"] = grp.transform(lambda s: s.shift(1).rolling(7, min_periods=3).std())

    # انبساطی با مرز برش + کوچک‌سازی بیزی برای سلف‌های کم‌حجم (F8.3)
    h = expanding_stat_at_cutoff(c, key, "NoRecv", out_prefix="cn")
    hres = expanding_stat_at_cutoff(c, key, "Res", out_prefix="cr")
    c["cell_expanding_rate"] = np.where(hres["cr_sum"] > 0, h["cn_sum"] / hres["cr_sum"], np.nan)
    c["cell_shrunk_rate"] = shrunk_rate(h["cn_sum"], hres["cr_sum"], BETA_ALPHA, BETA_BETA)
    # ⚠️ «تعداد روزهای تاریخچه» به‌عنوان فیچر ساخته **نمی‌شود**: Spearman آن با روزِ
    # تقویم ۰.۹۵ است، یعنی عملاً همان زمان — و به‌تنهایی R² خارج‌نمونه را از +۰.۰۸ به
    # −۱.۰۹ می‌برد (توضیح کامل در `cutoff.usage_rate`). فقط برای تشخیص نگه داشته می‌شود.
    c["_cell_n_prior_days_diag"] = h["cn_count"]

    keep = (key + ["date_gregorian"] + list(lag.columns)
            + [f"rho_roll_mean_{w}" for w in (7, 14, 28)]
            + ["rho_roll_std_7", "cell_expanding_rate", "cell_shrunk_rate"])
    return df.merge(c[keep], on=key + ["date_gregorian"], how="left", validate="many_to_one")


def add_dow_expanding(df: pd.DataFrame) -> pd.DataFrame:
    """۵.۴ — نرخ تاریخی همان سلف×وعده×روزهفته (معادل باینشده‌ی baseline بند ۲.۳)."""
    d = df.copy()
    d["meal_seq"] = meal_seq(d["date_gregorian"], d["Meal"])
    key = ["RestaurantName", "Meal", "dow"]
    hn = expanding_stat_at_cutoff(d, key, "NoRecv", out_prefix="dn")
    hr = expanding_stat_at_cutoff(d, key, "Res", out_prefix="dr")
    d["cell_dow_expanding_rate"] = np.where(hr["dr_sum"] > 0, hn["dn_sum"] / hr["dr_sum"], np.nan)
    d["cell_dow_shrunk_rate"] = shrunk_rate(hn["dn_sum"], hr["dr_sum"], BETA_ALPHA, BETA_BETA)
    return d


# ---------------------------------------------------------------------------
# ۵.۵ مقیاس رزرو
# ---------------------------------------------------------------------------

def add_scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """بند ۵.۵ — `Res` در لحظه‌ی برش کاملاً معلوم است (رزرو ۷۲ ساعت زودتر بسته می‌شود)."""
    d = df.copy()
    d["log_res"] = np.log1p(d["Res"])
    d["meal_seq"] = meal_seq(d["date_gregorian"], d["Meal"])

    key = ["RestaurantName", "Meal"]
    h = expanding_stat_at_cutoff(d, key, "Res", out_prefix="rs")
    d["res_vs_history"] = np.where(h["rs_mean"] > 0, d["Res"] / h["rs_mean"], np.nan)

    kd = ["RestaurantName", "Meal", "dow"]
    hd = expanding_stat_at_cutoff(d, kd, "Res", out_prefix="rd")
    d["res_vs_dow_history"] = np.where(hd["rd_mean"] > 0, d["Res"] / hd["rd_mean"], np.nan)

    # ➕ حجم کل دانشگاه در آن روز-وعده — معلوم در لحظه‌ی برش، R² شوک را ۰.۳۹۶→۰.۴۲۳ می‌برد (F61)
    daily = (d.groupby(["date_gregorian", "Meal"], as_index=False)["Res"].sum()
               .rename(columns={"Res": "daily_total_res"}))
    d = d.merge(daily, on=["date_gregorian", "Meal"], how="left", validate="many_to_one")
    d["log_daily_total_res"] = np.log1p(d["daily_total_res"])
    return d


# ---------------------------------------------------------------------------
# ۵.۶ منو و غذا — کوتاه‌شده
# ---------------------------------------------------------------------------

def add_food_features(df: pd.DataFrame) -> pd.DataFrame:
    """بند ۵.۶ — اثر **عمومی** غذا (F67). ترجیح شخصی و فاصله‌ی تکرار ساخته نمی‌شوند (F27, F66)."""
    d = df.copy()
    d["meal_seq"] = meal_seq(d["date_gregorian"], d["Meal"])
    hn = expanding_stat_at_cutoff(d, ["FoodName"], "NoRecv", out_prefix="fn")
    hr = expanding_stat_at_cutoff(d, ["FoodName"], "Res", out_prefix="fr")
    d["food_expanding_rate"] = np.where(hr["fr_sum"] > 0, hn["fn_sum"] / hr["fr_sum"], np.nan)
    d["food_shrunk_rate"] = shrunk_rate(hn["fn_sum"], hr["fr_sum"], BETA_ALPHA, BETA_BETA)
    d["_food_n_prior_diag"] = hn["fn_count"]  # تشخیصی، فیچر نیست (همان دلیل بالا)
    d["is_new_food"] = (hn["fn_count"] == 0).astype(int)

    # غذای رقیب: هر (روز، وعده، سلف) دقیقاً دو گزینه دارد
    g = d.groupby(["date_gregorian", "Meal", "RestaurantName"])["food_expanding_rate"]
    d["competitor_food_rate"] = (g.transform("sum") - d["food_expanding_rate"].fillna(0)) / \
                                (g.transform("count") - 1).replace(0, np.nan)
    d["food_rate_minus_competitor"] = d["food_expanding_rate"] - d["competitor_food_rate"]
    return d


# ---------------------------------------------------------------------------
# ⭐ ۵.۱۷ عامل روز — مهم‌ترین افزوده‌ی این فاز
# ---------------------------------------------------------------------------

def build_day_factor(cell: pd.DataFrame) -> pd.DataFrame:
    """عامل روز: میانگین انحراف سلف‌ها از عادت خودشان در آن روز-وعده.

    $$\\text{shock}_{d,m} = \\frac{1}{|R|}\\sum_{r}(\\rho_{d,m,r} - \\bar\\rho_{m,r})$$

    ⚠️ خودِ `shock` روز هدف **در لحظه‌ی برش معلوم نیست** (از outcome همان روز ساخته
    می‌شود). فقط `lag`های آن فیچر مجازند — و چون گروه‌بندی روی وعده است، `lag1` یعنی
    «همان وعده در روز قبل» که طبق قاعده‌ی برش در دسترس است.
    """
    c = cell.copy()
    c["rho_dev"] = c["rho_cell"] - c.groupby(["RestaurantName", "Meal"], observed=True)["rho_cell"].transform("mean")
    day = (c.groupby(["date_gregorian", "Meal"], as_index=False)
             .agg(day_shock=("rho_dev", "mean"), n_cells=("rho_dev", "size")))
    day.loc[day["n_cells"] < 3, "day_shock"] = np.nan
    day = day.sort_values(["Meal", "date_gregorian"], kind="stable")
    g = day.groupby("Meal", observed=True)["day_shock"]
    gd = day.groupby("Meal", observed=True)["date_gregorian"]
    # همان قاعده‌ی per-lag که در `same_meal_lag` توضیح داده شد: بدون این، lag۱ عامل روز
    # می‌تواند از شکاف رمضان بپرد و «۳۴ روز پیش» را به اسم «دیروز» تحویل مدل بدهد.
    for k in (1, 2, 7):
        col = f"day_shock_lag{k}"
        day[col] = g.shift(k)
        gap = (day["date_gregorian"] - gd.shift(k)).dt.days
        day.loc[gap > k + GAP_BUFFER_DAYS, col] = np.nan
    day["day_shock_roll_mean_7"] = g.transform(lambda s: s.shift(1).rolling(7, min_periods=2).mean())
    return day


def add_day_factor(df: pd.DataFrame, day: pd.DataFrame) -> pd.DataFrame:
    cols = ["date_gregorian", "Meal", "day_shock_lag1", "day_shock_lag2", "day_shock_lag7",
            "day_shock_roll_mean_7"]
    return df.merge(day[cols], on=["date_gregorian", "Meal"], how="left", validate="many_to_one")


# ---------------------------------------------------------------------------
# ۵.۱۸ برهم‌کنش‌های تأییدشده
# ---------------------------------------------------------------------------

def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """چهار برهم‌کنش با ΔAIC معنادار در F42 + برهم‌کنش ضرب‌شونده‌ی F62."""
    d = df.copy()
    d["is_khabgah"] = (d["RestaurantType"] == "khabgah").astype(int)
    d["is_lunch"] = (d["Meal"] == "lunch").astype(int)
    d["dow_x_type"] = d["dow"] * 10 + d["is_khabgah"]           # دسته‌ای ترکیبی
    d["meal_x_type"] = d["is_lunch"] * 2 + d["is_khabgah"]
    d["dow_x_city"] = d["dow"].astype(str) + "|" + d["city"].astype(str)
    d["city_x_meal"] = d["city"].astype(str) + "|" + d["Meal"].astype(str)
    return d
