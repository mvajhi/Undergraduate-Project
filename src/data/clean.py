"""پاکسازی و یکپارچه‌سازی فایل تجمیعی (مدل A) — بند ۳.۵, ۳.۸, ۳.۹, ۳.۱۰ WBS.

ترتیب صحیح فراخوانی (رعایت‌شده در `make_dataset.py`): نرمال‌سازی متن (`text_normalize`)
→ `parse_dates` → `drop_exact_duplicates` → تست‌های `src/validate.py` → `aggregate_over_gender`
→ `build_full_grid`. نرمال‌سازی باید **قبل از** dedup اجرا شود چون تفاوت رسم‌الخط دو ردیف
را که واقعاً یکسان‌اند از دید `duplicated()` مخفی می‌کند (بررسی‌شده: نرمال‌سازی `FoodName`
تعداد غذای یکتای فایل تجمیعی را از ۹۱ به ۷۸ می‌رساند).
"""

import pandas as pd

from src.config import DATA_EXTERNAL

AGGREGATE_KEY_COLS = ["DateReserve", "Meal", "RestaurantName", "FoodName", "Gender"]
COUNT_COLS = ["Reservation", "ReceiveWithCard", "ReceiveWithCode", "DontReceive"]

# ستون‌های مشتق‌شده‌ی از پیش‌موجود در فایل خام که در سطح روز ثابت‌اند (بند ۲.۱.۳ WBS) —
# پس از تجمیع روی جنسیت با first() نگه‌داشته می‌شوند (تغییری در مقدارشان بین گروه‌های
# همان روز وجود ندارد).
DAY_LEVEL_COLS = [
    "DayOfWeek",
    "HolidayInWeekCount",
    "HolidayInPrevWeekCount",
    "HolidayInNextWeekCount",
    "NextHoliday_1",
    "NextHoliday_2",
    "PreviousHoliday_1",
    "PreviousHoliday_2",
]


# ---------------------------------------------------------------------------
# ۳.۵ مدیریت زمان
# ---------------------------------------------------------------------------


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """`DateReserveGregorian` (رشته‌ی ISO) را به `datetime64` واقعی تبدیل می‌کند؛ `DateReserve`
    (نسخه‌ی شمسی) بدون تغییر برای گزارش نگه داشته می‌شود. بررسی تاریخ نامعتبر: `pd.to_datetime`
    با `errors='raise'` هر مقدار غیرقابل‌تجزیه را همان‌جا متوقف می‌کند.
    """
    df = df.copy()
    df["date_gregorian"] = pd.to_datetime(df["DateReserveGregorian"], format="%Y-%m-%d", errors="raise")
    return df


# ---------------------------------------------------------------------------
# ۳.۸ داده تکراری
# ---------------------------------------------------------------------------


def drop_exact_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """تکرار کامل (تمام ستون‌ها) حذف می‌شود؛ بند ۳.۸: «تکرار کامل → حذف»."""
    n_before = len(df)
    df_out = df.drop_duplicates().reset_index(drop=True)
    n_dropped = n_before - len(df_out)
    return df_out, n_dropped


# ---------------------------------------------------------------------------
# ۳.۹ تجمیع روی جنسیت
# ---------------------------------------------------------------------------


def aggregate_over_gender(df: pd.DataFrame) -> pd.DataFrame:
    """فرمول دقیق بند ۳.۹ WBS: از سطح (d,m,r,f,g) به سطح (d,m,r,f)."""
    group_cols = ["DateReserve", "date_gregorian", "Meal", "RestaurantName", "RestaurantType", "FoodName", "FoodType"]

    g = df.groupby(group_cols, as_index=False)
    res_g = g.apply(
        lambda gr: pd.Series(
            {
                "Res": gr["Reservation"].sum(),
                "Recv": (gr["ReceiveWithCard"] + gr["ReceiveWithCode"]).sum(),
                "NoRecv": (gr["Reservation"] - gr["ReceiveWithCard"] - gr["ReceiveWithCode"]).sum(),
                "gender_ratio": gr.loc[gr["Gender"] == "woman", "Reservation"].sum() / gr["Reservation"].sum(),
                "card_ratio": (
                    gr["ReceiveWithCard"].sum() / (gr["ReceiveWithCard"] + gr["ReceiveWithCode"]).sum()
                    if (gr["ReceiveWithCard"] + gr["ReceiveWithCode"]).sum() > 0
                    else pd.NA
                ),
            }
        ),
        include_groups=False,
    )

    day_level = df.groupby("DateReserve")[DAY_LEVEL_COLS].first().reset_index()
    out = res_g.merge(day_level, on="DateReserve", how="left")
    out["rho"] = out["NoRecv"] / out["Res"]
    return out


# ---------------------------------------------------------------------------
# ۳.۱۰ ساخت شبکه کامل و تشخیص عدم سرو
# ---------------------------------------------------------------------------


def build_full_grid(df_agg_gender: pd.DataFrame, df_calendar: pd.DataFrame) -> pd.DataFrame:
    """حاصل‌ضرب دکارتی (تاریخ × سلف × وعده × غذا) در بازه‌ی مشاهده‌شده‌ی داده، مقایسه با
    ترکیب‌های واقعی، و طبقه‌بندی غیاب.

    تعریف عملیاتی «غذا در منو نبوده» (بند ۳.۱۰ WBS): چون تنها منبع «منوی واقعی هر روز»
    خودِ داده‌ی رزرو است (منوی رسمی هنوز دریافت نشده — بند ۲.۱.۲، ردیف 🟡)، وقتی سلف/وعده‌ی
    مشخصی در روزی *عملاً سرویس داده* (غذای دیگری از همان سلف/وعده در همان روز موجود است)
    ولی غذای مشخصی غایب است، این غیاب = «آن غذا آن روز جزو منو نبود»، نه نشت داده یا خطا —
    این دقیقاً معنایی است که داده در اختیار می‌گذارد.

    ۴ دسته‌ی غیاب (بند ۳.۱۰ WBS):
      - holiday: `is_holiday_any` تقویم تهران (`src.data.calendar`) برای آن روز True است.
      - restaurant_meal_not_offered: آن جفت (سلف، وعده) در **کل بازه‌ی داده** هرگز رخ نداده
        (مثلاً سلف‌های دانشکده‌ای که هرگز شام سرو نمی‌کنند — ساختاری، نه غیبت یک‌روزه).
      - restaurant_closed_that_day: آن (سلف، وعده) معمولاً فعال است ولی **هیچ** غذایی از آن
        در این روز خاص سرو نشده (نه تعطیل رسمی) — تعطیلی موردی/غیررسمی یا شکاف داده.
      - food_not_on_menu_that_day: آن (سلف، وعده) همان روز فعال بوده (غذای دیگری سرو شده) ولی
        این غذای مشخص در منو نبوده.
    """
    dates = df_calendar.loc[
        (df_calendar["date_gregorian"] >= df_agg_gender["date_gregorian"].min().strftime("%Y-%m-%d"))
        & (df_calendar["date_gregorian"] <= df_agg_gender["date_gregorian"].max().strftime("%Y-%m-%d")),
        ["date_gregorian", "is_holiday_any"],
    ].copy()
    dates["date_gregorian"] = pd.to_datetime(dates["date_gregorian"])

    restaurants = df_agg_gender["RestaurantName"].unique()
    meals = df_agg_gender["Meal"].unique()
    foods = df_agg_gender["FoodName"].unique()

    grid = pd.MultiIndex.from_product(
        [dates["date_gregorian"], restaurants, meals, foods],
        names=["date_gregorian", "RestaurantName", "Meal", "FoodName"],
    ).to_frame(index=False)
    grid = grid.merge(dates, on="date_gregorian", how="left")

    actual_food = df_agg_gender[["date_gregorian", "RestaurantName", "Meal", "FoodName"]].drop_duplicates()
    actual_food["is_served"] = True
    grid = grid.merge(actual_food, on=["date_gregorian", "RestaurantName", "Meal", "FoodName"], how="left")
    grid["is_served"] = grid["is_served"].eq(True)

    # (سلف، وعده) که در کل بازه هرگز رخ نداده = ساختاری
    active_rm = df_agg_gender[["RestaurantName", "Meal"]].drop_duplicates()
    active_rm["_rm_active"] = True
    grid = grid.merge(active_rm, on=["RestaurantName", "Meal"], how="left")
    grid["_rm_active"] = grid["_rm_active"].eq(True)

    # (سلف، وعده، روز) که حداقل یک غذا در آن سرو شده = آن روز فعال بوده
    actual_rmd = df_agg_gender[["date_gregorian", "RestaurantName", "Meal"]].drop_duplicates()
    actual_rmd["_rmd_active"] = True
    grid = grid.merge(actual_rmd, on=["date_gregorian", "RestaurantName", "Meal"], how="left")
    grid["_rmd_active"] = grid["_rmd_active"].eq(True)

    def classify(row):
        if row["is_served"]:
            return "served"
        if row["is_holiday_any"]:
            return "holiday"
        if not row["_rm_active"]:
            return "restaurant_meal_not_offered"
        if not row["_rmd_active"]:
            return "restaurant_closed_that_day"
        return "food_not_on_menu_that_day"

    grid["absence_reason"] = grid.apply(classify, axis=1)
    return grid.drop(columns=["_rm_active", "_rmd_active"])


def coverage_report(grid: pd.DataFrame) -> pd.DataFrame:
    """خلاصه‌ی توزیع دسته‌های غیاب — بند ۳.۱۰ «یک گزارش پوشش»."""
    return grid["absence_reason"].value_counts().rename_axis("category").reset_index(name="n_rows")


if __name__ == "__main__":
    import pandas as pd

    from src.data.inspect_raw import load_aggregate
    from src.data.text_normalize import normalize_columns

    df = load_aggregate()
    df = normalize_columns(df, ["RestaurantName", "FoodName"])
    df = parse_dates(df)
    df, n_dropped = drop_exact_duplicates(df)
    print(f"dropped {n_dropped} exact-duplicate rows -> {len(df)} rows remain")

    df_gender = aggregate_over_gender(df)
    print(f"aggregated over gender -> {len(df_gender)} rows at (d,m,r,f) grain")
    print(df_gender.head())

    df_cal = pd.read_csv(DATA_EXTERNAL / "calendar_tehran.csv")
    grid = build_full_grid(df_gender, df_cal)
    print("\ncoverage report:")
    print(coverage_report(grid))
