"""موقعیت جغرافیایی پردیس‌ها و فهرست سلف‌های کنارگذاشته‌شده (اصلاحیه‌ی فاز ۴).

**چرا این ماژول وجود دارد؟** دو اصلاح دامنه‌ای که در فاز ۳ دیده نشده بود و در ابتدای
فاز ۴ توسط ذی‌نفع پروژه گزارش شد (ردیف‌های ۲۱ و ۲۲ `doc/decision_log.md`):

۱. **پنج سلف در تهران نیستند.** «کشاورزی» در کرج، «فارابی» در قم، «ابوریحان» در
   پاکدشت، «کاسپین» در رضوانشهر (گیلان) و «فومن» در فومن (گیلان) قرار دارند. تا پیش
   از این اصلاح، داده‌ی هواشناسی/AQI *تهران* (`weather_aqi_tehran.csv`) به همه‌ی
   رکوردها الحاق می‌شد — یعنی برای این پنج پردیس، متغیرهای خارجی عملاً نویز بودند
   (قم و رضوانشهر بیش از ۱۳۰ و ۳۳۰ کیلومتر با تهران فاصله دارند و اقلیم متفاوتی
   دارند: رضوانشهر/فومن اقلیم معتدل خزری، قم اقلیم کویری).

۲. **«دوره حوزه علوم انسانی» داده‌ی پرت است و کاملاً حذف می‌شود** — نه یک سلف عملیاتی
   با سری زمانی معنادار، بلکه یک ردیف کم‌حجم و بی‌ثبات (۱۲ رکورد تجمیعی، میانگین
   Res≈۱۶، ρ تا ۰.۵) که در بند ۴.۱ هم (F1.12) به‌عنوان نمونه‌ی به‌شدت ناکافی پرچم
   شده بود.

مطابق قاعده‌ی مخزن (بند ۳.۳)، جدول‌ها به فایل صریح `data/external/*.csv` نوشته
می‌شوند و پایپ‌لاین فقط همان CSV را می‌خواند — دانش دامنه‌ای اینجا فقط *منبع تولید*
فایل است، نه چیزی که در زمان اجرا hardcode شود.
"""

import logging

import pandas as pd

from src.config import DATA_EXTERNAL

logger = logging.getLogger(__name__)

CAMPUS_GEO_PATH = DATA_EXTERNAL / "campus_geo.csv"
EXCLUDED_RESTAURANTS_PATH = DATA_EXTERNAL / "excluded_restaurants.csv"

# مختصات مرجع هر شهر (نقطه‌ی پردیس، نه مرکز شهر) — ورودی فراخوانی Open-Meteo
_CITY_COORDS: dict[str, tuple[str, float, float]] = {
    # city: (province, lat, lon)
    "تهران": ("تهران", 35.705, 51.396),
    "کرج": ("البرز", 35.8043, 50.9722),
    "قم": ("قم", 34.6416, 50.8746),
    "پاکدشت": ("تهران", 35.4694, 51.6836),
    "رضوانشهر": ("گیلان", 37.5528, 49.1417),
    "فومن": ("گیلان", 37.2244, 49.3125),
}

# سلف‌هایی که شهرشان تهران نیست (نام کانونیک فایل تجمیعی). هر نام ممکن است هم نسخه‌ی
# `daneshgah` و هم `khabgah` داشته باشد — شهر به نام سلف وابسته است نه به نوع آن.
_NON_TEHRAN: dict[str, str] = {
    "کشاورزی": "کرج",
    "فارابی": "قم",
    "ابوریحان": "پاکدشت",
    "کاسپین": "رضوانشهر",
    "فومن": "فومن",
}

# فاصله‌ی تقریبی جاده‌ای تا پردیس مرکزی تهران (کیلومتر) — فقط برای گزارش/تفسیر،
# نه فیچر مدل.
_DISTANCE_KM: dict[str, int] = {
    "تهران": 0,
    "پاکدشت": 40,
    "کرج": 45,
    "قم": 140,
    "فومن": 300,
    "رضوانشهر": 335,
}

# سلف‌های کنارگذاشته‌شده: نام کانونیک → دلیل
_EXCLUDED: dict[str, str] = {
    "دوره حوزه علوم انسانی": "داده پرت / سری زمانی ناکافی (ردیف ۲۲ decision_log)",
    "دوره حوزه علوم انسانی ( مهر)": "داده پرت / سری زمانی ناکافی (ردیف ۲۲ decision_log)",
}


def build_campus_geo(canonical_restaurant_names) -> pd.DataFrame:
    """جدول `canonical_name → city/province/lat/lon/is_tehran` را می‌سازد و ذخیره می‌کند."""
    names = sorted({str(n).strip() for n in canonical_restaurant_names if str(n).strip() and str(n) != "nan"})
    rows = []
    for name in names:
        city = _NON_TEHRAN.get(name, "تهران")
        province, lat, lon = _CITY_COORDS[city]
        rows.append(
            {
                "canonical_name": name,
                "city": city,
                "province": province,
                "lat": lat,
                "lon": lon,
                "is_tehran": city == "تهران",
                "distance_km_to_tehran_campus": _DISTANCE_KM[city],
            }
        )
    df = pd.DataFrame(rows)
    CAMPUS_GEO_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CAMPUS_GEO_PATH, index=False)
    logger.info(f"Saved campus geo table ({len(df)} restaurants, {df['city'].nunique()} cities) to {CAMPUS_GEO_PATH}")
    return df


def build_excluded_restaurants() -> pd.DataFrame:
    """فهرست صریح سلف‌های کنارگذاشته‌شده را می‌سازد و ذخیره می‌کند."""
    df = pd.DataFrame(
        [{"canonical_name": k, "reason": v, "decided_on": "2026-08-13"} for k, v in _EXCLUDED.items()]
    )
    EXCLUDED_RESTAURANTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(EXCLUDED_RESTAURANTS_PATH, index=False)
    logger.info(f"Saved excluded-restaurant list ({len(df)} rows) to {EXCLUDED_RESTAURANTS_PATH}")
    return df


def load_campus_geo() -> pd.DataFrame:
    return pd.read_csv(CAMPUS_GEO_PATH)


def load_excluded_restaurants() -> list[str]:
    if not EXCLUDED_RESTAURANTS_PATH.exists():
        return []
    return pd.read_csv(EXCLUDED_RESTAURANTS_PATH)["canonical_name"].astype(str).tolist()


def drop_excluded_restaurants(df: pd.DataFrame, name_col: str) -> tuple[pd.DataFrame, int]:
    """ردیف‌های سلف‌های کنارگذاشته‌شده را حذف می‌کند و تعداد حذف‌شده را برمی‌گرداند."""
    excluded = load_excluded_restaurants()
    if not excluded:
        return df, 0
    mask = df[name_col].astype(str).isin(excluded)
    n = int(mask.sum())
    return df.loc[~mask].copy(), n


def attach_city(df: pd.DataFrame, name_col: str) -> pd.DataFrame:
    """ستون‌های `city`, `province`, `is_tehran` را بر اساس نام کانونیک سلف الحاق می‌کند."""
    geo = load_campus_geo()[["canonical_name", "city", "province", "is_tehran", "distance_km_to_tehran_campus"]]
    out = df.merge(geo, left_on=name_col, right_on="canonical_name", how="left")
    if "canonical_name" in out.columns and name_col != "canonical_name":
        out = out.drop(columns=["canonical_name"])
    n_missing = int(out["city"].isna().sum())
    if n_missing:
        missing_names = sorted(set(out.loc[out["city"].isna(), name_col].astype(str)))
        logger.warning(f"{n_missing} rows have no city mapping; restaurants: {missing_names}")
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    agg = pd.read_csv(DATA_EXTERNAL.parent / "processed" / "dataset_v1.csv", usecols=["RestaurantName"])
    geo_df = build_campus_geo(agg["RestaurantName"])
    excl_df = build_excluded_restaurants()
    print(geo_df.groupby(["city", "province"]).size().to_string())
    print()
    print(excl_df.to_string(index=False))
