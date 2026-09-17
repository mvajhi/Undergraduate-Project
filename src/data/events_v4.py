"""دفتر رویدادهای پنجره‌ی v4 — تمدید `events.csv` به ۹۴۳ روز و شش شهر (بند ۱۳-۶).

**سه چیزی که نسخه‌ی قبلی نمی‌توانست داشته باشد:**

۱. **شش شهر، نه یک شهر.** `events.csv` قدیمی فقط تهران را می‌دید چون داده‌ی هواشناسی
   فقط تهران بود. حالا `weather_aqi_by_city_v4.csv` هر شش پردیس را دارد، و یک دوره‌ی
   آلودگی در کرج برای سلف‌های کشاورزی رویداد است حتی اگر تهران پاک باشد.
۲. **سه ماه رمضان، نه یکی** — پنجره‌ی تازه سه رمضان دارد (F110).
۳. **شکاف ساختاری خرداد/تیر ۱۴۰۴** (F108) که اصلاً در پنجره‌ی قفل‌شده نبود.

⚠️ **این فایل فیچر نمی‌شود** (بند ۲-۴ WBS). کارش توضیح پرت‌هاست: وقتی یک روز نرخ
عجیبی دارد، اینجا باید بشود فهمید چرا. رویدادهای «مشتق‌شده» (آلودگی، برف) از خودِ
داده‌ی پروژه می‌آیند نه از خبر، پس تاریخ‌هایشان قطعی و بازتولیدپذیرند.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.config import DATA_EXTERNAL
from src.data.events import (
    AQI_MIN_RUN_DAYS,
    AQI_UNHEALTHY,
    HEAVY_SNOW_CM,
    _consecutive_runs,
    _jalali,
)

logger = logging.getLogger(__name__)

OUTPUT_PATH = DATA_EXTERNAL / "events_v4.csv"
WEATHER_V4_PATH = DATA_EXTERNAL / "weather_aqi_by_city_v4.csv"
CALENDAR_V4_PATH = DATA_EXTERNAL / "calendar_tehran_v4.csv"

#: پنجره‌ی داده‌ی رزرو (F88) — رویداد بیرون از این بازه ثبت می‌شود ولی علامت می‌خورد.
DATA_START, DATA_END = "2023-09-25", "2026-02-20"

COLUMNS = [
    "event_id", "date_start", "date_end", "date_jalali_start", "date_jalali_end",
    "event_type", "scope", "affected_meal", "service_status", "confidence",
    "description", "expected_effect", "observed_in_data", "source_url",
]


def _weather() -> pd.DataFrame:
    return pd.read_csv(WEATHER_V4_PATH)


def derive_pollution_episodes_by_city(weather: pd.DataFrame) -> list[dict]:
    """دوره‌های آلودگی «ناسالم برای همه»، **به تفکیک شهر**."""
    events = []
    for city, block in weather.groupby("city"):
        hits = block.loc[block.aqi_us_max >= AQI_UNHEALTHY, "date_gregorian"].tolist()
        runs = [
            (s, e) for s, e in _consecutive_runs(hits)
            if (pd.Timestamp(e) - pd.Timestamp(s)).days + 1 >= AQI_MIN_RUN_DAYS
        ]
        for i, (start, end) in enumerate(runs, start=1):
            n_days = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
            window = block[(block.date_gregorian >= start) & (block.date_gregorian <= end)]
            events.append({
                "event_id": f"aqi_{city}_{i:02d}",
                "date_start": start, "date_end": end,
                "event_type": "آلودگی_هوای_شدید", "scope": city, "affected_meal": "هردو",
                "confidence": "بالا",
                "description": (
                    f"دوره‌ی پیوسته‌ی {n_days} روزه در {city} با AQI حداکثر روزانه ≥ "
                    f"{AQI_UNHEALTHY}؛ اوج دوره: AQI={int(window.aqi_us_max.max())}. "
                    "استخراج‌شده از داده‌ی Open-Meteo پروژه، نه از خبر"
                ),
                "expected_effect": (
                    "کاهش تردد اختیاری و احتمال افزایش نرخ عدم‌دریافت؛ ⚠️ جایگزین «تعطیلی رسمی» نیست"
                ),
                "source_url": "derived:weather_aqi_by_city_v4.csv",
            })
    return events


def derive_snow_episodes_by_city(weather: pd.DataFrame) -> list[dict]:
    events = []
    for city, block in weather.groupby("city"):
        hits = block.loc[block.snowfall_sum >= HEAVY_SNOW_CM, "date_gregorian"].tolist()
        for i, (start, end) in enumerate(_consecutive_runs(hits), start=1):
            window = block[(block.date_gregorian >= start) & (block.date_gregorian <= end)]
            events.append({
                "event_id": f"snow_{city}_{i:02d}",
                "date_start": start, "date_end": end,
                "event_type": "بارش_برف_سنگین", "scope": city, "affected_meal": "هردو",
                "confidence": "بالا",
                "description": (
                    f"بارش برف مجموعاً {round(float(window.snowfall_sum.sum()), 2)} سانتی‌متر در "
                    f"{city} با کمینه‌ی دمای {float(window.temp_min.min())}°C"
                ),
                "expected_effect": "اختلال تردد و احتمال افزایش نرخ عدم‌دریافت، به‌ویژه برای غیرخوابگاهی‌ها",
                "source_url": "derived:weather_aqi_by_city_v4.csv",
            })
    return events


def derive_ramadan_periods() -> list[dict]:
    """سه ماه رمضان پنجره (F110) — از ستون `is_ramadan` تقویم v4."""
    cal = pd.read_csv(CALENDAR_V4_PATH, usecols=["date_gregorian", "is_ramadan"])
    hits = cal.loc[cal.is_ramadan.astype(bool), "date_gregorian"].tolist()
    return [
        {
            "event_id": f"ramadan_{i:02d}",
            "date_start": start, "date_end": end,
            "event_type": "ماه_رمضان", "scope": "سراسری", "affected_meal": "ناهار",
            "confidence": "متوسط",
            "description": (
                "ماه رمضان؛ ناهار برای روزه‌داران عملاً حذف می‌شود. ⚠️ مرز آغاز **تقریبی** "
                "است (F110): پروژه کتابخانه‌ی تبدیل هجری قمری ندارد و آغاز از عید فطر "
                "منهای ۲۹ روز زده شده — مرز پایان دقیق، مرز آغاز ±۱ روز"
            ),
            "expected_effect": "افت شدید حجم رزرو ناهار و تغییر ترکیب مراجعان؛ شام کمتر متأثر",
            "source_url": "derived:calendar_tehran_v4.csv",
        }
        for i, (start, end) in enumerate(_consecutive_runs(hits), start=1)
    ]


def structural_gap_events() -> list[dict]:
    """پنجره‌ی درگیری خرداد ۱۴۰۴ — یافته F108 و F135، ردیف ۸۰ `decision_log`.

    **علت، دیگر ثبت‌نشده نیست.** ردیف ۷۰ این بازه را با علت ناشناخته ثبت کرده بود
    چون داده خودش نمی‌گفت و پروژه سابقه‌ی صریح دارد که علت خبری بی‌شاهد ثبت نشود
    (ردیف ۱۵). ذی‌نفع علت را تأیید کرد، و مهم‌تر: **خودِ داده حالا شاهد دارد** —
    پنجره‌ی ۷۲ ساعته‌ی رزرو، انتساب را آزمون‌پذیر می‌کند و آزمون را پاس می‌کند.
    """
    return [{
        "event_id": "war_1404_03",
        "date_start": "2025-06-13", "date_end": "2025-07-03",
        "event_type": "درگیری_نظامی", "scope": "سراسری", "affected_meal": "هردو",
        "confidence": "بالا",
        "description": (
            "درگیری خرداد ۱۴۰۴ (آغاز ۲۰۲۵-۰۶-۱۳). امضای داده دو مرحله دارد و هر دو با "
            "پنجره‌ی ۷۲ ساعته‌ی رزرو می‌خوانند: (۱) روزهای ۱۴ تا ۱۶ ژوئن که رزروشان "
            "**پیش از** آغاز درگیری قفل شده بود، در داده هستند ولی ρ آن‌ها ۰.۴۷ تا ۰.۶۷ "
            "است در برابر ۰.۰۸ عادی — یعنی رزروکننده شهر را ترک کرده؛ (۲) از ۱۷ ژوئن "
            "خودِ رزرو فرومی‌ریزد (۵۴۰ → ۳۲۸ → ۱۶۰ → ۶) و از ۲۰۲۵-۰۶-۲۱ صفر می‌ماند تا "
            "پایان نیمسال. تقویم آموزشی این بازه را توضیح نمی‌دهد: امتحانات پایان‌ترم "
            "سایر سال‌ها سرویس **دارند**"
        ),
        "expected_effect": (
            "نیمسال دوم ۱۴۰۳-۱۴۰۴ ~۲ هفته زودتر عملاً تمام شد. این پنجره از آموزش و "
            "ارزیابی **حذف** می‌شود نه پرچم‌گذاری: یک رویداد یکتاست، تعمیم‌پذیر نیست، "
            "و پرچمش در استقرار برای شوک بعدی مقدارش False است (ردیف ۸۰)"
        ),
        "source_url": "derived:StudentReserve.parquet (F108/F135) + تأیید ذی‌نفع",
    }]


def build_events_v4(output_path=OUTPUT_PATH) -> pd.DataFrame:
    weather = _weather()
    events = (
        derive_pollution_episodes_by_city(weather)
        + derive_snow_episodes_by_city(weather)
        + derive_ramadan_periods()
        + structural_gap_events()
    )

    df = pd.DataFrame(events)
    df["date_jalali_start"] = df["date_start"].map(_jalali)
    df["date_jalali_end"] = df["date_end"].map(_jalali)
    # رویدادی که بیرون از پنجره‌ی داده بیفتد حذف نمی‌شود — علامت می‌خورد، چون
    # پنجره‌ی هواشناسی عمداً بلندتر از پنجره‌ی رزرو است.
    df["service_status"] = (
        (df["date_end"] < DATA_START) | (df["date_start"] > DATA_END)
    ).map({True: "بیرون_از_پنجره_داده", False: "داخل_پنجره_داده"})
    df["observed_in_data"] = ""

    df = df.reindex(columns=COLUMNS).sort_values(["date_start", "event_id"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("دفتر رویدادهای v4 ذخیره شد: %s (%d رویداد)", output_path, len(df))
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    from src.config import set_global_seed

    set_global_seed()
    df = build_events_v4()

    pd.set_option("display.width", 200)
    print("\n=== رویداد بر حسب نوع ===")
    print(df.event_type.value_counts().to_string())
    print("\n=== آلودگی و برف بر حسب شهر ===")
    print(pd.crosstab(df.scope, df.event_type).to_string())
    print("\n=== داخل/بیرون پنجره‌ی داده ===")
    print(df.service_status.value_counts().to_string())
    print("\n=== رمضان و شکاف ساختاری ===")
    print(df[df.event_type.isin(["ماه_رمضان", "درگیری_نظامی"])][
        ["event_id", "date_start", "date_end", "date_jalali_start", "service_status"]
    ].to_string(index=False))
