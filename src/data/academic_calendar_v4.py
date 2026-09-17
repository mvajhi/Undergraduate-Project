"""تقویم روزانه‌ی پنجره‌ی v4 — تعطیلات + تقویم آموزشی سه سال تحصیلی (بند ۱۳-۶).

این ماژول قلم بیرونیِ **مسدود** بند ۱۳-۶ را باز می‌کند. تا پیش از این، `is_exam_period`،
`week_of_semester` و `semester` فقط برای ۱۴۰۲-۱۴۰۳ ساخته می‌شدند چون PDF دو سال بعد
در دسترس نبود؛ با رسیدن سه PDF تازه هر سه فیچر روی کل پنجره ساخته می‌شوند.

**نسبتش با `src/data/calendar.py`:** آن ماژول برای پنجره‌ی قفل‌شده‌ی v2 است، مقادیر
تقویم آموزشی‌اش دستی رونویسی شده و **دست نمی‌خورد** (شاهد تاریخی گزارش نهایی است،
قاعده‌ی پسوند `_v4` در `doc/phase13-execution-standard.md` بند ۳). این ماژول همان
ستون‌ها را می‌سازد به‌علاوه‌ی چند ستون تازه، ولی مقادیر تقویم آموزشی را از PDF
استخراج می‌کند (`academic_calendar_pdf`) نه از ثابت داخل کد.

استخراج خودکار هر شش مقدارِ دستیِ ۱۴۰۲-۱۴۰۳ را عیناً بازتولید کرد — از جمله هر دو
مقداری که در ۲۰۲۶-۰۸-۱۳ اصلاح شده بودند و بلوک نوروز (`۱۴۰۲-۱۲-۲۶` تا `۱۴۰۳-۰۱-۱۰`).
یعنی تقویم دو سال تازه با همان دقتی ساخته می‌شود که تقویم سال اول، بدون رونویسی دستی.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache

import holidays
import jdatetime
import pandas as pd

from src.config import DATA_EXTERNAL, DATA_RAW
from src.data.academic_calendar_pdf import (
    REQUIRED_EVENT_CODES,
    CalendarEvent,
    parse_calendar_page,
)
from src.data.calendar import compute_holiday_derived_features

logger = logging.getLogger(__name__)

EVENTS_PATH = DATA_EXTERNAL / "academic_calendar_events_v4.csv"
DAILY_PATH = DATA_EXTERNAL / "calendar_tehran_v4.csv"

#: پنجره‌ی تقویم — عمداً هم‌مرز `weather_aqi_by_city_v4.csv` (۹۴۳ روز) گرفته شده تا
#: الحاق دو قلم بیرونی هیچ روزِ بی‌زوج نسازد. داده‌ی رزرو خودش باریک‌تر است
#: (۲۰۲۳-۰۹-۲۵ تا ۲۰۲۶-۰۲-۲۰، یافته F88).
START_DATE = "2023-09-01"
END_DATE = "2026-03-31"


# ---------------------------------------------------------------------------
# منابع PDF
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalendarSource:
    pdf: str
    page: int              # صفر-پایه
    academic_year: str
    semester: int
    layout: str            # "grid" یا "event_column"
    superseded_by: str | None = None


#: ⚠️ `تقویم_آموزشی.pdf` صفحه‌ی ۲ و `اصلاحیه_سوم…` **هر دو** نیمسال دوم ۱۴۰۴-۱۴۰۵‌اند و
#: یکی نیستند: اصلاحیه کل نیمسال را یک هفته جلو می‌برد (شروع کلاس‌ها ۲۵ بهمن → ۲ اسفند،
#: حذف‌واضافه ۹-۱۱ اسفند → ۱۶-۱۸ اسفند، میان‌ترم ۲۹ فروردین → ۵ اردیبهشت). اصلاحیه
#: متأخرتر است (ساخت ۲۰۲۶-۰۶-۲۲ در برابر ۲۰۲۵-۰۵-۰۳) و مرجع است؛ نسخه‌ی اول فقط برای
#: ردگیری نگه داشته می‌شود و به تقویم روزانه راه نمی‌یابد.
CALENDAR_SOURCES: tuple[CalendarSource, ...] = (
    CalendarSource("UT_calender_1402_1043.pdf", 0, "1402-1403", 1, "grid"),
    CalendarSource("UT_calender_1402_1043.pdf", 1, "1402-1403", 2, "grid"),
    CalendarSource("تقویم آموزشی-1403-1404.pdf", 0, "1403-1404", 1, "grid"),
    CalendarSource("تقویم آموزشی-1403-1404.pdf", 1, "1403-1404", 2, "grid"),
    CalendarSource("تقویم_آموزشی.pdf", 0, "1404-1405", 1, "event_column"),
    CalendarSource(
        "تقویم_آموزشی.pdf", 1, "1404-1405", 2, "event_column",
        superseded_by="اصلاحیه_سوم__تقویم_دانشگاهی_نیمسال_2-1404 (1).pdf",
    ),
    CalendarSource("اصلاحیه_سوم__تقویم_دانشگاهی_نیمسال_2-1404 (1).pdf", 0, "1404-1405", 2, "event_column"),
)


# ---------------------------------------------------------------------------
# نیمسال
# ---------------------------------------------------------------------------


@dataclass
class Semester:
    academic_year: str
    semester: int
    class_start: pd.Timestamp
    class_end: pd.Timestamp
    final_end: pd.Timestamp
    periods: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = field(default_factory=dict)
    nowruz: set[str] = field(default_factory=set)   # تاریخ میلادی ISO

    @property
    def key(self) -> str:
        return f"{self.academic_year}-{self.semester}"

    def contains(self, ts: pd.Timestamp) -> bool:
        """تعلق اداری به نیمسال — تا پایان امتحانات، نه فقط پایان کلاس‌ها."""
        return self.class_start <= ts <= self.final_end


def _j2ts(date_jalali: str) -> pd.Timestamp:
    y, m, d = (int(p) for p in date_jalali.split("-"))
    return pd.Timestamp(jdatetime.date(y, m, d).togregorian())


@lru_cache(maxsize=None)
def _parse_source(src: CalendarSource):
    """تجزیه‌ی یک صفحه، با cache — بدون این، ساخت کامل تقویم هر PDF را پنج بار
    می‌خواند و هشدار غلط چاپی را پنج بار چاپ می‌کند."""
    return parse_calendar_page(
        DATA_RAW / src.pdf, src.page, src.academic_year, src.semester, src.layout
    )


def parse_all_events(*, include_superseded: bool = False) -> list[CalendarEvent]:
    """همه‌ی رویدادهای تقویم آموزشی از همه‌ی PDFها."""
    events: list[CalendarEvent] = []
    for src in CALENDAR_SOURCES:
        if src.superseded_by and not include_superseded:
            continue
        parsed = _parse_source(src)
        found = {e.event_code for e in parsed.events}
        missing = REQUIRED_EVENT_CODES - found
        if missing:
            raise ValueError(
                f"{src.pdf} صفحه‌ی {src.page + 1} ({src.academic_year} نیمسال "
                f"{src.semester}): رویدادهای الزامی پیدا نشدند: {sorted(missing)}"
            )
        events.extend(parsed.events)
    return events


def build_semesters(events: list[CalendarEvent] | None = None) -> list[Semester]:
    """رویدادهای نقطه‌ای را به بازه‌های نیمسال تبدیل می‌کند.

    هر کد رویداد می‌تواند چند تاریخ داشته باشد (مثلاً `add_drop` سه روز)؛ بازه‌اش
    «کمینه تا بیشینه» است. `midterm_mark` وقتی به‌کار می‌آید که PDF به‌جای
    «شروع امتحانات میان‌ترم» فقط «امتحان میان‌ترم» نوشته باشد.
    """
    events = events if events is not None else parse_all_events()

    by_sem: dict[tuple[str, int], dict[str, list[str]]] = {}
    for e in events:
        by_sem.setdefault((e.academic_year, e.semester), {}).setdefault(e.event_code, []).append(e.date_jalali)

    nowruz_by_sem: dict[tuple[str, int], set[str]] = {}
    for src in CALENDAR_SOURCES:
        if src.superseded_by:
            continue
        parsed = _parse_source(src)
        nowruz_by_sem.setdefault((src.academic_year, src.semester), set()).update(
            _j2ts(d).strftime("%Y-%m-%d") for d in parsed.nowruz_dates
        )

    def span(codes: dict[str, list[str]], *names: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        dates = sorted(d for n in names for d in codes.get(n, []))
        return (_j2ts(dates[0]), _j2ts(dates[-1])) if dates else None

    semesters = []
    for (year, sem), codes in sorted(by_sem.items()):
        periods = {}
        for name, sources in {
            "add_drop": ("add_drop",),
            "final_control": ("final_control",),
            "midterm": ("midterm_start", "midterm_mark", "midterm_end"),
            "final_exam": ("final_start", "general_exam", "final_end"),
            "emergency_drop": ("emergency_drop",),
            "evaluation": ("evaluation",),
        }.items():
            got = span(codes, *sources)
            if got:
                periods[name] = got

        semesters.append(
            Semester(
                academic_year=year,
                semester=sem,
                class_start=span(codes, "class_start")[0],
                class_end=span(codes, "class_end")[1],
                final_end=periods["final_exam"][1] if "final_exam" in periods else span(codes, "class_end")[1],
                periods=periods,
                nowruz=nowruz_by_sem.get((year, sem), set()),
            )
        )
    return semesters


# ---------------------------------------------------------------------------
# تقویم روزانه
# ---------------------------------------------------------------------------


def _base_calendar(start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    dates = pd.date_range(start, end, freq="D")
    rows = []
    for d in dates:
        jd = jdatetime.date.fromgregorian(date=d.date())
        rows.append(
            {
                "date_gregorian": d.strftime("%Y-%m-%d"),
                "date_jalali": f"{jd.year:04d}-{jd.month:02d}-{jd.day:02d}",
                "year_jalali": jd.year,
                "month_jalali": jd.month,
                "day_of_month_jalali": jd.day,
                # شنبه=۰ … جمعه=۶ — همان قرارداد v1 (نرخ تطابق ۱.۰۰ با `DayOfWeek`)
                "day_of_week": (d.weekday() + 2) % 7,
            }
        )
    df = pd.DataFrame(rows)
    df["_ts"] = dates
    return df


def _ramadan_windows(years: list[int]) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """بازه‌ی تقریبی ماه رمضان، از تاریخ عید فطر عقب‌شمرده.

    ⚠️ **تقریبی است.** پروژه کتابخانه‌ی تبدیل تاریخ هجری قمری ندارد؛ تنها لنگر قمریِ
    در دسترس، خودِ عید فطر در پکیج `holidays` است. رمضان ۲۹ یا ۳۰ روز است و اینجا
    ۳۰ فرض می‌شود، پس مرز آغاز می‌تواند یک روز خطا داشته باشد. مرز پایان (روز پیش از
    عید فطر) دقیق است.

    چرا با این تقریب هم ارزش دارد: پنجره‌ی v4 **سه** رمضان دارد نه یکی (بند ۱۳-۶) و
    رمضان ناهار دانشجویی را ساختاراً عوض می‌کند — بی‌فیچرماندنش یعنی مدل این افت را
    به تقویم تحصیلی یا فصل نسبت می‌دهد.
    """
    ir = holidays.Iran(years=years)
    windows = []
    for date, name in sorted(ir.items()):
        if "Eid al-Fitr" in name and "Holiday" not in name:
            end = pd.Timestamp(date) - pd.Timedelta(1, unit="D")
            windows.append((end - pd.Timedelta(29, unit="D"), end))
    return windows


def _classify_holidays(df: pd.DataFrame, semesters: list[Semester]) -> pd.DataFrame:
    df = df.copy()
    ir = holidays.Iran(years=sorted(df["_ts"].dt.year.unique().tolist()))

    nowruz = {d for s in semesters for d in s.nowruz}

    # بین دو نیمسال: از روز پس از پایان امتحانات هر نیمسال تا روز پیش از شروع کلاس‌های
    # نیمسال بعد. تعطیلات تابستان هم همین است، فقط بلندتر — با آستانه‌ی ۴۵ روز جدا می‌شود.
    ordered = sorted(semesters, key=lambda s: s.class_start)
    breaks, summer = [], []
    for prev, nxt in zip(ordered, ordered[1:]):
        gap = (prev.final_end + pd.Timedelta(1, unit="D"), nxt.class_start - pd.Timedelta(1, unit="D"))
        (summer if (gap[1] - gap[0]).days >= 45 else breaks).append(gap)

    def in_spans(ts, spans):
        return any(a <= ts <= b for a, b in spans)

    df["is_friday"] = df["day_of_week"] == 6
    df["holiday_name"] = df["_ts"].apply(lambda d: ir.get(d.date()))
    df["is_holiday_national"] = df["holiday_name"].notna()
    df["is_nowruz_block"] = df["date_gregorian"].isin(nowruz)
    df["is_inter_semester_break"] = df["_ts"].apply(lambda d: in_spans(d, breaks))
    df["is_summer_break"] = df["_ts"].apply(lambda d: in_spans(d, summer))

    df["is_named_holiday"] = df["is_holiday_national"] | df["is_nowruz_block"] | df["is_inter_semester_break"]
    df["is_holiday_any"] = df["is_friday"] | df["is_named_holiday"]

    ramadan = _ramadan_windows(sorted(df["_ts"].dt.year.unique().tolist()))
    df["is_ramadan"] = df["_ts"].apply(lambda d: in_spans(d, ramadan))
    df["days_into_ramadan"] = df["_ts"].apply(
        lambda d: next(((d - a).days + 1 for a, b in ramadan if a <= d <= b), pd.NA)
    )
    return df


def _academic_features(df: pd.DataFrame, semesters: list[Semester]) -> pd.DataFrame:
    df = df.copy()

    def sem_of(ts: pd.Timestamp) -> Semester | None:
        return next((s for s in semesters if s.contains(ts)), None)

    def week_of_semester(ts: pd.Timestamp) -> int | float:
        """شماره‌ی هفته‌ی کلاسی. فقط در بازه‌ی کلاس‌های عادی معنا دارد؛ هفته‌های نوروز
        شمرده نمی‌شوند، پس هفته‌ی پس از نوروز ادامه‌ی هفته‌ی پیش از آن است."""
        s = sem_of(ts)
        if s is None or not (s.class_start <= ts <= s.class_end):
            return pd.NA
        if ts.strftime("%Y-%m-%d") in s.nowruz:
            return pd.NA
        skipped = sum(
            1 for d in s.nowruz if s.class_start <= pd.Timestamp(d) < ts
        )
        return (ts - s.class_start).days // 7 + 1 - skipped // 7

    def period_flag(name: str):
        def flag(ts: pd.Timestamp) -> bool:
            s = sem_of(ts)
            if s is None or name not in s.periods:
                return False
            a, b = s.periods[name]
            return a <= ts <= b
        return flag

    df["academic_year"] = df["_ts"].apply(lambda d: (s := sem_of(d)) and s.academic_year)
    df["semester"] = df["_ts"].apply(lambda d: (s := sem_of(d)) and s.semester)
    df["week_of_semester"] = df["_ts"].apply(week_of_semester)
    df["is_in_session"] = df["_ts"].apply(
        lambda d: (s := sem_of(d)) is not None
        and s.class_start <= d <= s.class_end
        and d.strftime("%Y-%m-%d") not in s.nowruz
    )

    for name, column in {
        "add_drop": "is_add_drop_period",
        "final_control": "is_final_control_period",
        "midterm": "is_midterm_period",
        "final_exam": "is_final_exam_period",
        "emergency_drop": "is_emergency_drop_period",
        "evaluation": "is_evaluation_period",
    }.items():
        df[column] = df["_ts"].apply(period_flag(name))

    df["is_exam_period"] = df["is_midterm_period"] | df["is_final_exam_period"]

    exam_starts = sorted(
        s.periods[name][0] for s in semesters for name in ("midterm", "final_exam") if name in s.periods
    )
    df["days_to_exam_start"] = df["_ts"].apply(
        lambda d: next(((x - d).days for x in exam_starts if x >= d), pd.NA)
    )

    df["days_since_semester_start"] = df["_ts"].apply(
        lambda d: (d - s.class_start).days if (s := sem_of(d)) else pd.NA
    )
    df["days_to_semester_end"] = df["_ts"].apply(
        lambda d: (s.final_end - d).days if (s := sem_of(d)) else pd.NA
    )

    def progress(ts: pd.Timestamp) -> float | pd.NA:
        """جای نسبی روز در بخش کلاسیِ نیمسال (۰ تا ۱).

        چرا علاوه بر `week_of_semester`: طول نیمسال‌ها یکسان نیست (۱۱۳ تا ۱۲۹ روز
        کلاسی در همین شش نیمسال)، پس «هفته‌ی دوازدهم» در دو سال به یک نقطه از ترم
        اشاره نمی‌کند ولی `۰٫۷۵` می‌کند.
        """
        s = sem_of(ts)
        if s is None or not (s.class_start <= ts <= s.class_end):
            return pd.NA
        total = (s.class_end - s.class_start).days
        return round((ts - s.class_start).days / total, 4) if total else pd.NA

    df["semester_progress"] = df["_ts"].apply(progress)
    return df


def build_daily_calendar(output_path=DAILY_PATH) -> pd.DataFrame:
    semesters = build_semesters()
    df = _base_calendar()
    df = _classify_holidays(df, semesters)
    df = compute_holiday_derived_features(df)
    df = _academic_features(df, semesters)

    out = df.drop(columns=["_ts"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    logger.info("تقویم روزانه‌ی v4 ذخیره شد: %s (%d روز)", output_path, len(out))
    return out


def build_events_table(output_path=EVENTS_PATH) -> pd.DataFrame:
    """جدول خام رویدادها — شاهد ردیف‌به‌ردیف تقویم روزانه.

    نسخه‌ی جایگزین‌شده‌ی نیمسال دوم ۱۴۰۴-۱۴۰۵ **هم** ثبت می‌شود (با ستون
    `is_superseded`) تا معلوم باشد اصلاحیه دقیقاً چه چیزی را عوض کرده است.
    """
    rows = []
    for src in CALENDAR_SOURCES:
        parsed = _parse_source(src)
        for e in parsed.events:
            rows.append({**e.__dict__, "is_superseded": src.superseded_by is not None})

    df = pd.DataFrame(rows).sort_values(["academic_year", "semester", "date_jalali", "event_code"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("جدول رویدادها ذخیره شد: %s (%d ردیف)", output_path, len(df))
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    from src.config import set_global_seed

    set_global_seed()

    events = build_events_table()
    daily = build_daily_calendar()

    print("\n=== نیمسال‌ها ===")
    for s in build_semesters():
        print(
            f"{s.key}  کلاس {s.class_start.date()} → {s.class_end.date()}  "
            f"پایان امتحانات {s.final_end.date()}  نوروز {len(s.nowruz)} روز"
        )
    print("\n=== پوشش تقویم روزانه ===")
    print(daily[["semester", "is_in_session", "is_exam_period", "is_ramadan"]].notna().sum())
