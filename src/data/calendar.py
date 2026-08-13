"""Build the daily calendar/holiday/academic-calendar feature table for the study window (WBS بند ۲.۲).

Data sources:
- Holidays: `holidays` package, Iran subdivision (national + Hijri/religious holidays).
- Academic calendar: data/raw/UT_calender_1402_1043.pdf (University of Tehran, both semesters
  1402-1403), manually transcribed via `pdftotext -layout` — see doc/decision_log.md row 15.

`is_holiday_any` (includes Friday) explains non-serving days; `is_named_holiday` (excludes
Friday) mirrors the convention of the raw aggregate file's own `NextHoliday_1`/`PreviousHoliday_1`
columns (validated in WBS بند ۲.۱: those columns reach values >30, so they do not count the
weekly Friday as a "holiday" — only named/calendar holidays).
"""

import logging

import holidays
import jdatetime
import pandas as pd

from src.config import DATA_EXTERNAL
from src.data.weather import gregorian_to_jalali

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_PATH = DATA_EXTERNAL / "calendar_tehran.csv"

START_DATE = "2023-11-01"
END_DATE = "2024-06-30"


def _jd(y: int, m: int, d: int) -> pd.Timestamp:
    """Jalali (y, m, d) -> Gregorian pd.Timestamp."""
    return pd.Timestamp(jdatetime.date(y, m, d).togregorian())


# ---------------------------------------------------------------------------
# تقویم آموزشی دانشگاه تهران — استخراج دستی از data/raw/UT_calender_1402_1043.pdf
# (صفحه ۱ = نیمسال اول ۱۴۰۲-۱۴۰۳، صفحه ۲ = نیمسال دوم ۱۴۰۲-۱۴۰۳)
# ---------------------------------------------------------------------------

SEMESTER_1 = {
    "class_start": _jd(1402, 7, 1),
    # اصلاح‌شده ۲۰۲۶-۰۸-۱۳ پس از بازبینی تصویر با وضوح بالا (رندر ۳۰۰dpi): مقدار قبلی (۱۷ تا ۱۹ مهر) اشتباه بود
    "add_drop": (_jd(1402, 7, 15), _jd(1402, 7, 17)),
    "midterm_exams": (_jd(1402, 8, 20), _jd(1402, 9, 2)),
    # اصلاح‌شده ۲۰۲۶-۰۸-۱۳: مقدار قبلی (دی ۲۷) اشتباه بود — سلول «پایان کلاسها» واقعاً دی ۲۰ است
    "class_end": _jd(1402, 10, 20),
    "final_exams": (_jd(1402, 10, 28), _jd(1402, 11, 14)),
}

SEMESTER_2 = {
    "class_start": _jd(1402, 11, 21),
    # اصلاح‌شده ۲۰۲۶-۰۸-۱۳: مقدار قبلی (بهمن ۲۸ تا اسفند ۵) کاملاً اشتباه بود — سلول‌های زرد «حذف و اضافه»
    # واقعاً اسفند ۵، ۷، ۸ هستند (اسفند ۶ به‌دلیل «ولادت حضرت قائم» رنگ متفاوت دارد ولی احتمالاً همچنان
    # داخل همان بازه است)
    "add_drop": (_jd(1402, 12, 5), _jd(1402, 12, 8)),
    "nowruz_block": (_jd(1402, 12, 26), _jd(1403, 1, 10)),
    "midterm_exams": (_jd(1403, 1, 25), _jd(1403, 2, 6)),
    "class_end": _jd(1403, 3, 24),
    # امتحانات پایان‌ترم نیمسال دوم بعد از بازه‌ی داده‌ی ما شروع می‌شود (تیر ۱۴۰۳) — دیدن یادداشت زیر
    "final_exams": (_jd(1403, 3, 31), _jd(1403, 4, 14)),
}

# فاصله‌ی بین دو نیمسال: پایان امتحانات نیمسال اول (بهمن ۱۴) تا شروع کلاس‌های نیمسال دوم (بهمن ۲۱)،
# منهای خودِ روز ثبت‌نام/شروع کلاس — بازه‌ی واقعاً بدون سرویس‌دهی: بهمن ۱۵ تا ۲۰.
INTER_SEMESTER_BREAK = (_jd(1402, 11, 15), _jd(1402, 11, 20))

ADD_DROP_PERIODS = [SEMESTER_1["add_drop"], SEMESTER_2["add_drop"]]
MIDTERM_PERIODS = [SEMESTER_1["midterm_exams"], SEMESTER_2["midterm_exams"]]
FINAL_EXAM_PERIODS = [SEMESTER_1["final_exams"], SEMESTER_2["final_exams"]]


def _in_any(date: pd.Timestamp, periods: list[tuple[pd.Timestamp, pd.Timestamp]]) -> bool:
    return any(start <= date <= end for start, end in periods)


# ---------------------------------------------------------------------------
# ۲.۲.۱ تقویم پایه
# ---------------------------------------------------------------------------


def build_base_calendar(start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    dates = pd.date_range(start, end, freq="D")
    rows = []
    for d in dates:
        jd = jdatetime.date.fromgregorian(date=d.date())
        rows.append(
            {
                "date_gregorian": d.strftime("%Y-%m-%d"),
                "date_jalali": gregorian_to_jalali(d.year, d.month, d.day),
                "year_jalali": jd.year,
                "month_jalali": jd.month,
                "day_of_month_jalali": jd.day,
                # شنبه=۰ ... جمعه=۶ (بند ۲.۱ WBS: نرخ تطابق ۱.۰۰ با ستون DayOfWeek فایل تجمیعی)
                "day_of_week": (d.weekday() + 2) % 7,
                "week_of_year_jalali": jd.isocalendar()[1] if hasattr(jd, "isocalendar") else None,
            }
        )
    df = pd.DataFrame(rows)
    df["_ts"] = dates
    return df


# ---------------------------------------------------------------------------
# ۲.۲.۲ تعطیلات
# ---------------------------------------------------------------------------


def classify_holidays(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    years = sorted(df["_ts"].dt.year.unique().tolist())
    ir_holidays = holidays.Iran(years=years)

    df["is_friday"] = df["day_of_week"] == 6
    df["holiday_name"] = df["_ts"].apply(lambda d: ir_holidays.get(d.date()))
    df["is_holiday_national"] = df["holiday_name"].notna()
    df["is_nowruz_block"] = df["_ts"].apply(
        lambda d: SEMESTER_2["nowruz_block"][0] <= d <= SEMESTER_2["nowruz_block"][1]
    )
    df["is_inter_semester_break"] = df["_ts"].apply(
        lambda d: INTER_SEMESTER_BREAK[0] <= d <= INTER_SEMESTER_BREAK[1]
    )

    # نامزد تعطیلی اضطراری (آلودگی هوا) که خبرگزاری‌ها تأیید کرده‌اند ولی هیچ‌کدام با شکاف
    # واقعی سرو در فایل تجمیعی هم‌پوشانی ندارد — عمداً is_holiday نمی‌شود (دیدن decision_log #15).
    # این تاریخ‌ها فقط برای مستندسازی نگه داشته می‌شوند، نه فیچرسازی این بند.
    df["is_named_holiday"] = df["is_holiday_national"] | df["is_nowruz_block"] | df["is_inter_semester_break"]
    df["is_holiday_any"] = df["is_friday"] | df["is_named_holiday"]
    return df


# ---------------------------------------------------------------------------
# ۲.۲.۳ فیچرهای مشتق تقویمی
# ---------------------------------------------------------------------------


def compute_holiday_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    named_holiday_dates = df.loc[df["is_named_holiday"], "_ts"].tolist()

    def days_to_next(d):
        future = [h for h in named_holiday_dates if h > d]
        return (min(future) - d).days if future else None

    def days_since_last(d):
        past = [h for h in named_holiday_dates if h < d]
        return (d - max(past)).days if past else None

    df["days_to_next_holiday"] = df["_ts"].apply(days_to_next)
    df["days_since_last_holiday"] = df["_ts"].apply(days_since_last)

    is_off = df["is_holiday_any"].to_numpy()
    n = len(df)
    is_bridge = [False] * n
    block_length = [None] * n
    day_before = [False] * n
    day_after = [False] * n

    # طول بلوک تعطیلی متصل (شامل جمعه‌ها) که هر روز در آن قرار دارد
    i = 0
    while i < n:
        if is_off[i]:
            j = i
            while j < n and is_off[j]:
                j += 1
            for k in range(i, j):
                block_length[k] = j - i
            i = j
        else:
            i += 1

    for i in range(n):
        prev_off = is_off[i - 1] if i > 0 else False
        next_off = is_off[i + 1] if i < n - 1 else False
        if not is_off[i] and prev_off and next_off:
            is_bridge[i] = True
        if not is_off[i] and next_off:
            day_before[i] = True
        if not is_off[i] and prev_off:
            day_after[i] = True

    df["is_bridge_day"] = is_bridge
    df["is_day_before_holiday"] = day_before
    df["is_day_after_holiday"] = day_after
    df["holiday_block_length"] = block_length
    return df


# ---------------------------------------------------------------------------
# ۲.۲.۴ تقویم آموزشی دانشگاه
# ---------------------------------------------------------------------------


def academic_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def semester_of(d):
        # بازه‌ی «تعلق به نیمسال» تا پایان امتحانات پایان‌ترم گسترده می‌شود (نه فقط پایان کلاس‌ها)
        # چون هفته‌ی «کلاس‌های جبرانی»/امتحانات هم به‌لحاظ اداری بخشی از همان نیمسال است.
        if SEMESTER_1["class_start"] <= d <= SEMESTER_1["final_exams"][1]:
            return 1
        if SEMESTER_2["class_start"] <= d <= SEMESTER_2["final_exams"][1]:
            return 2
        return None

    def week_of_semester(d):
        # فقط در بازه‌ی کلاس‌های عادی معنا دارد؛ در هفته‌ی جبرانی/امتحانات None برمی‌گردد.
        info = None
        if SEMESTER_1["class_start"] <= d <= SEMESTER_1["class_end"]:
            info, sem = SEMESTER_1, 1
        elif SEMESTER_2["class_start"] <= d <= SEMESTER_2["class_end"]:
            info, sem = SEMESTER_2, 2
        if info is None:
            return None
        if sem == 2 and info["nowruz_block"][0] <= d <= info["nowruz_block"][1]:
            return None  # هفته‌ی نوروز، هفته‌ی کلاسی شمرده نمی‌شود
        days = (d - info["class_start"]).days
        if sem == 2 and d > info["nowruz_block"][1]:
            days -= (info["nowruz_block"][1] - info["nowruz_block"][0]).days + 1
        return days // 7 + 1

    df["semester"] = df["_ts"].apply(semester_of)
    df["week_of_semester"] = df["_ts"].apply(week_of_semester)
    df["is_add_drop_period"] = df["_ts"].apply(lambda d: _in_any(d, ADD_DROP_PERIODS))
    df["is_midterm_period"] = df["_ts"].apply(lambda d: _in_any(d, MIDTERM_PERIODS))
    df["is_final_exam_period"] = df["_ts"].apply(lambda d: _in_any(d, FINAL_EXAM_PERIODS))
    df["is_exam_period"] = df["is_midterm_period"] | df["is_final_exam_period"]

    exam_starts = sorted(s for s, _ in MIDTERM_PERIODS + FINAL_EXAM_PERIODS)

    def days_to_exam_start(d):
        future = [s for s in exam_starts if s >= d]
        return (min(future) - d).days if future else None

    df["days_to_exam_start"] = df["_ts"].apply(days_to_exam_start)
    return df


# ---------------------------------------------------------------------------
# اعتبارسنجی در برابر فایل تجمیعی (بند ۲.۱)
# ---------------------------------------------------------------------------


def validate_against_raw(df_calendar: pd.DataFrame, df_aggregate: pd.DataFrame) -> dict:
    """مقایسه‌ی تقویم مستقل با ستون‌های تقویمی از‌پیش‌موجود فایل تجمیعی."""
    merged = df_aggregate[["DateReserveGregorian", "DayOfWeek"]].drop_duplicates().merge(
        df_calendar[["date_gregorian", "day_of_week"]],
        left_on="DateReserveGregorian",
        right_on="date_gregorian",
        how="left",
    )
    dow_match_rate = (merged["DayOfWeek"] == merged["day_of_week"]).mean()

    served_dates = set(df_aggregate["DateReserveGregorian"].unique())
    cal_window = df_calendar[
        (df_calendar["date_gregorian"] >= df_aggregate["DateReserveGregorian"].min())
        & (df_calendar["date_gregorian"] <= df_aggregate["DateReserveGregorian"].max())
    ]
    missing_dates = set(cal_window["date_gregorian"]) - served_dates
    explained_by_holiday = cal_window[cal_window["date_gregorian"].isin(missing_dates) & cal_window["is_holiday_any"]]
    explained_by_bridge = cal_window[cal_window["date_gregorian"].isin(missing_dates) & cal_window["is_bridge_day"]]
    unexplained = missing_dates - set(explained_by_holiday["date_gregorian"]) - set(explained_by_bridge["date_gregorian"])

    return {
        "day_of_week_match_rate": round(float(dow_match_rate), 4),
        "n_missing_days": len(missing_dates),
        "n_explained_by_holiday": len(explained_by_holiday),
        "n_explained_by_bridge_day": len(explained_by_bridge),
        "n_unexplained": len(unexplained),
        "unexplained_dates": sorted(unexplained),
    }


# ---------------------------------------------------------------------------
# اسمبل نهایی
# ---------------------------------------------------------------------------


def build_full_calendar(output_path=OUTPUT_PATH) -> pd.DataFrame:
    df = build_base_calendar()
    df = classify_holidays(df)
    df = compute_holiday_derived_features(df)
    df = academic_calendar_features(df)
    df_out = df.drop(columns=["_ts"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)
    logger.info(f"Saved calendar dataset ({len(df_out)} rows) to {output_path}")
    return df_out


if __name__ == "__main__":
    from src.data.inspect_raw import load_aggregate

    df_cal_full = build_base_calendar()
    df_cal_full = classify_holidays(df_cal_full)
    df_cal_full = compute_holiday_derived_features(df_cal_full)
    df_cal_full = academic_calendar_features(df_cal_full)

    df_agg = load_aggregate()
    print("Validation against raw aggregate file:")
    print(validate_against_raw(df_cal_full, df_agg))

    df_result = build_full_calendar()
    print()
    print("Dataset sample:")
    print(df_result.head())
    print()
    print("Holiday type counts:")
    print(
        df_result[
            ["is_friday", "is_holiday_national", "is_nowruz_block", "is_inter_semester_break", "is_bridge_day"]
        ].sum()
    )
