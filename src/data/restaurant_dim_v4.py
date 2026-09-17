"""جدول بعد سلف‌ها برای snapshot نسخه‌ی ۴ — جایگزین `restaurant_mapping.csv` + `campus_geo.csv`.

**چرا نگاشت قدیمی بازسازی می‌شود و نه وصله (یافته F91).** `restaurant_mapping.csv` هر نام
خام را به یکی از ۳۱ نام کانونیکِ **فایل تجمیعی** می‌برد. آن مرجع در v4 مرده است: داده‌ی
تازه ۱۲۴ نام دارد و فایل تجمیعی فقط ۴۷ تایشان را می‌شناسد. وصله‌زدن یعنی ۷۷ سلف بی‌نگاشت
بمانند و در تجمیع **بی‌صدا** حذف شوند.

**چرا جدول بعد و نه یک نگاشت دیگر.** آزمون شباهت فازی روی هر ۷۳ نام `core` نشان داد هیچ
دو نامی یک موجودیت نیستند — پنج زوج پرشباهت هم واقعاً جدا بودند (`فیزیک`/`ژئوفیزیک`،
`عطارشرقی`/`عطارغربی`، `قدس`/`قدس ۳`، `نصرت`/`نصرت ۲۰`، `اندیشه قم`/`دانش قم`). پس کار
واقعی، **جمع‌کردن نام‌ها نیست، چسباندن ویژگی به آن‌هاست**: پردیس، شهر، مختصات، نوع
مکان، جنسیت. همان نتیجه‌ای که `build_dorm_mapping` برای `GroupName` گرفت، اینجا هم
داده‌محور تکرار شد.

**نتیجه‌ی جانبی که یک انسداد را باز می‌کند.** خودِ نام سلف پردیسش را اعلام می‌کند
(`کشاورزی-…`، `فارابی-…`، `فومن-…`). یعنی تمدید `campus_geo.csv` از ۳۰ به ۱۲۴ سلف —
که بدون آن هواشناسی به سلف‌های تازه الحاق نمی‌شد — نیازی به فهرست دستی ندارد و از
همین‌جا مشتق می‌شود. هر شش پردیس دقیقاً به شش شهر `weather_aqi_by_city_v4.csv` می‌خورند.
"""

from __future__ import annotations

import logging
import re

import pandas as pd

from src.config import DATA_EXTERNAL
from src.data.load_v4 import _normalize_categories, load_raw_v4
from src.eda_lib.restaurant_taxonomy import classify_restaurants

logger = logging.getLogger(__name__)

OUTPUT_PATH = DATA_EXTERNAL / "restaurant_dim_v4.csv"
LEGACY_GEO_PATH = DATA_EXTERNAL / "campus_geo.csv"

#: مختصات و فاصله عیناً از `campus_geo.csv` گرفته شده‌اند تا عدد تازه‌ای وارد پروژه نشود.
#: کلید `city` دقیقاً همان مقدار ستون `city` در `weather_aqi_by_city_v4.csv` است — الحاق
#: هواشناسی روی همین ستون انجام می‌شود.
CAMPUS_GEO: dict[str, dict] = {
    "تهران":    {"city": "تهران",    "province": "تهران", "lat": 35.7050, "lon": 51.3960, "is_tehran": True,  "distance_km_to_tehran_campus": 0},
    "کشاورزی":  {"city": "کرج",      "province": "البرز", "lat": 35.8043, "lon": 50.9722, "is_tehran": False, "distance_km_to_tehran_campus": 45},
    "فارابی":   {"city": "قم",       "province": "قم",    "lat": 34.6416, "lon": 50.8746, "is_tehran": False, "distance_km_to_tehran_campus": 140},
    "ابوریحان": {"city": "پاکدشت",   "province": "تهران", "lat": 35.4694, "lon": 51.6836, "is_tehran": False, "distance_km_to_tehran_campus": 40},
    "فومن":     {"city": "فومن",     "province": "گیلان", "lat": 37.2244, "lon": 49.3125, "is_tehran": False, "distance_km_to_tehran_campus": 300},
    "کاسپین":   {"city": "رضوانشهر", "province": "گیلان", "lat": 37.5528, "lon": 49.1417, "is_tehran": False, "distance_km_to_tehran_campus": 335},
}

DEFAULT_CAMPUS = "تهران"

#: نشانه‌ی پردیس‌های غیرتهران. `دانشکدگان فارابی` هم به همین قاعده می‌افتد چون واژه‌ی
#: `فارابی` را دارد.
#:
#: ⚠️ تطبیق **واژه‌ای** است نه زیررشته‌ای، و این تفاوت واقعاً مهم است:
#: `رستوران مکمل-شماره (1)-بلوار کشاورز(بخش خصوصی)` در تهران است، ولی تطبیق زیررشته‌ای
#: `کشاورز` را نشانه‌ی پردیس کرج می‌گرفت و هواشناسی کرج را به یک رستوران تهران می‌چسباند.
_CAMPUS_TOKENS = ("کشاورزی", "فارابی", "ابوریحان", "فومن", "کاسپین")
_FA_LETTER = r"؀-ۿ"


def detect_campus(name: str) -> str:
    for token in _CAMPUS_TOKENS:
        if re.search(rf"(?<![{_FA_LETTER}]){token}(?![{_FA_LETTER}])", name):
            return token
    return DEFAULT_CAMPUS


#: نوع مکان. ترتیب مهم است — نخستین تطابق برنده است، پس `خرید از بوفه` باید پیش از
#: `خوابگاه` بیاید (نامش هر دو را دارد: `کشاورزی-خرید از بوفه(خوابگاه خواهران)`).
VENUE_PATTERNS: list[tuple[str, str]] = [
    ("buffet", "خرید از بوفه"),
    ("guest_quota", "^مهمان"),
    ("private", "رستوران مکمل"),
    ("temporary_course", "حوزه"),
    ("system_test", "تست اپ"),
    ("free_restaurant", "رستوران آزاد"),
    ("dorm_married", "^متاهلین"),
    ("dorm_city", "^سطح شهر"),
    ("faculty", "^دانشکده"),
    ("campus_central", "سلف مرکزی|پردیس فنی"),
    ("dorm", "خوابگاه"),
]


def detect_venue_type(name: str) -> str:
    for label, pattern in VENUE_PATTERNS:
        if re.search(pattern, name):
            return label
    return "other"


_GENDER_TOKENS = {
    "male": ("برادران", "پسران"),
    "female": ("خواهران", "دختران"),
}


def detect_gender(name: str) -> str:
    """جنسیت مخاطب سلف.

    `mixed` یعنی نام **هر دو** نشانه را دارد، نه اینکه هیچ‌کدام را نداشته باشد — مثل
    `دانشکده-سلف فنی2-امیرآباد-خواهران-دکترا-برادران-کارکنان` که واقعاً چند گروه را
    سرو می‌کند. نبودِ هر دو `unknown` است، چون سلف دانشکده‌ای معمولاً جنسیت را در نام
    نمی‌آورد و «مختلط» دانستنش یک فرض است نه یک مشاهده.
    """
    has = {g: any(t in name for t in tokens) for g, tokens in _GENDER_TOKENS.items()}
    if has["male"] and has["female"]:
        return "mixed"
    if has["male"]:
        return "male"
    if has["female"]:
        return "female"
    return "unknown"


#: وعده‌ی مقیدشده در نام. فقط شعب مکمل این را دارند؛ هر کسب‌وکار خصوصی با دو ردیف
#: (`-ناهار` و `-شام`) ثبت می‌شود.
_MEAL_TOKENS = [("lunch", ("ناهار", "نهار")), ("dinner", ("شام",))]


def detect_meal_lock(name: str) -> str | None:
    found = [meal for meal, tokens in _MEAL_TOKENS if any(t in name for t in tokens)]
    return found[0] if len(found) == 1 else None


_PAREN_RE = re.compile(r"\([^)]*\)")
_NOISE_RE = re.compile(
    r"^(مهمان|رستوران مکمل|دانشکده|سطح شهر|متاهلین|خوابگاه)\s*-?\s*|"
    r"\b(سلف|شماره|بخش خصوصی|روزانه \d+ ?نفر)\b|"
    r"\b(برادران|خواهران|دختران|پسران|کارکنان|دکترا|ناهار|نهار|شام)\b"
)


def business_key(name: str) -> str:
    """کلید «کسب‌وکار» — نامِ بی‌وعده و بی‌شماره، برای شمردن شعب یکتا.

    لازمِ فیچر بافتاری مکمل (ردیف ۶۷ `decision_log`: «تعداد شعبه‌ی فعال در پردیس»).
    بدون این، ۲۵ نامِ مکمل ۲۵ شعبه شمرده می‌شوند در حالی که واقعاً ~۱۴ کسب‌وکارند؛ هر
    کدام با دو ردیف `-ناهار` و `-شام` ثبت شده‌اند و شمردن نام‌ها تعداد را دو برابر
    می‌کند.
    """
    cleaned = _PAREN_RE.sub(" ", name)
    cleaned = _NOISE_RE.sub(" ", cleaned)
    cleaned = re.sub(r"[\-–—/\d]+", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


_TOKEN_RE = re.compile(rf"[{_FA_LETTER}\w]+")


def _tokens(name: str) -> set[str]:
    """واژه‌های یک نام. `فنی2` به `فنی` و `2` شکسته می‌شود تا `فنی امیرآباد` قدیمی با
    `دانشکده-سلف فنی2-امیرآباد-…` تازه بخواند."""
    out: set[str] = set()
    for chunk in _TOKEN_RE.findall(name):
        out.update(p for p in re.split(r"(\d+)", chunk) if p)
    return out


def _legacy_links(names: list[str]) -> dict[str, tuple[str | None, int | None]]:
    """پیوند هر نام v4 به نام کانونیک پنجره‌ی قفل‌شده (۳۱ نام `campus_geo.csv`).

    فقط برای **ردیابی** است: بدون آن هیچ جدولی از گزارش نهایی به جدول تازه وصل
    نمی‌شود. مبنای مدل‌سازی نیست — نام نرمال‌شده‌ی v4 خودش کلید کانونیک است.

    ⚠️ **تطبیق فازی عمداً استفاده نشد.** نسخه‌ی اول این تابع `partial_ratio` می‌زد و
    دقیقاً همان سه خطایی را ساخت که `mapping.py` سال‌ها پیش دستی تصحیحشان کرده بود:
    `علوم اجتماعی`→`علوم`، `ژئوفیزیک`→`فیزیک`، و
    `علوم و فناوری های میان رشته ای`→`علوم`. علتش ساختاری است — `partial_ratio` به هر
    زیررشته امتیاز کامل می‌دهد، پس نام کوتاه‌تر همیشه برنده می‌شود.

    قاعده‌ی جایگزین قطعی است: نام قدیمی وقتی معادل است که **همه‌ی واژه‌هایش** در نام
    تازه باشند، و بین چند نامزد، آنکه واژه‌ی بیشتری دارد برنده است. `ژئوفیزیک` یک
    واژه است و `فیزیک` را در خود ندارد، پس تطبیق غلط اصلاً ممکن نمی‌شود.

    پیوند **چند-به-یک** است و این درست است، نه نقص: پنجره‌ی قفل‌شده ۲۵ خوابگاه سطح
    شهر را در یک نام `سطح شهر` جمع کرده بود و داده‌ی تازه از هم تفکیکشان می‌کند.
    """
    if not LEGACY_GEO_PATH.exists():
        return {n: (None, None) for n in names}
    legacy = pd.read_csv(LEGACY_GEO_PATH)["canonical_name"].dropna().unique().tolist()
    legacy_tokens = {name: _tokens(name) for name in legacy}

    out: dict[str, tuple[str | None, int | None]] = {}
    for name in names:
        have = _tokens(name)
        hits = [(len(toks), cand) for cand, toks in legacy_tokens.items() if toks and toks <= have]
        out[name] = (max(hits)[1], max(hits)[0]) if hits else (None, None)
    return out


def build_restaurant_dim(output_path=OUTPUT_PATH) -> pd.DataFrame:
    """جدول بعد سلف‌ها را از خودِ snapshot می‌سازد و ذخیره می‌کند."""
    df = load_raw_v4(["RestaurantName", "resturantId", "Count", "miladi"], meals=(1, 3))
    df["RestaurantName"] = _normalize_categories(df["RestaurantName"])

    agg = df.groupby("RestaurantName", observed=True).agg(
        rows=("Count", "size"),
        portions=("Count", "sum"),
        first_day=("miladi", "min"),
        last_day=("miladi", "max"),
        active_days=("miladi", "nunique"),
        n_ids=("resturantId", "nunique"),
    )
    agg.index.name = "raw_name"
    out = agg.reset_index()
    out["raw_name"] = out["raw_name"].astype(str)

    out["restaurant_class"] = classify_restaurants(out["raw_name"])
    out["campus"] = out["raw_name"].map(detect_campus)
    out["venue_type"] = out["raw_name"].map(detect_venue_type)
    out["gender"] = out["raw_name"].map(detect_gender)
    out["meal_lock"] = out["raw_name"].map(detect_meal_lock)
    out["business_key"] = out["raw_name"].map(business_key)

    geo = out["campus"].map(CAMPUS_GEO).apply(pd.Series)
    out = pd.concat([out, geo], axis=1)

    links = _legacy_links(out["raw_name"].tolist())
    out["legacy_canonical_name"] = out["raw_name"].map(lambda n: links[n][0])
    out["legacy_match_tokens"] = out["raw_name"].map(lambda n: links[n][1])

    for col in ("first_day", "last_day"):
        out[col] = pd.to_datetime(out[col]).dt.strftime("%Y-%m-%d")

    out = out.sort_values(["restaurant_class", "campus", "portions"], ascending=[True, True, False])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    logger.info("جدول بعد سلف‌ها ذخیره شد: %s (%d سلف)", output_path, len(out))
    return out


def coverage_report(dim: pd.DataFrame) -> pd.DataFrame:
    """دروازه‌ی بند ۱۳-۵: نرخ پوشش هر ویژگی.

    هیچ نامی نباید بی‌صدا بی‌ویژگی بماند — به همین دلیل این گزارش جزو خروجی است،
    نه یک بررسی اختیاری.
    """
    total_portions = dim["portions"].sum()
    rows = []
    for col, unknown in [
        ("campus", None), ("venue_type", "other"), ("gender", "unknown"),
        ("city", None), ("legacy_canonical_name", None),
    ]:
        bad = dim[col].isna() if unknown is None else dim[col].eq(unknown)
        rows.append({
            "attribute": col,
            "n_missing": int(bad.sum()),
            "pct_names": round(100 * bad.mean(), 2),
            "pct_portions": round(100 * dim.loc[bad, "portions"].sum() / total_portions, 2),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    from src.config import set_global_seed

    set_global_seed()
    dim = build_restaurant_dim()

    pd.set_option("display.width", 200)
    print("\n=== رده × پردیس ===")
    print(pd.crosstab(dim["restaurant_class"], dim["campus"]))
    print("\n=== نوع مکان ===")
    print(dim["venue_type"].value_counts().to_string())
    print("\n=== پوشش ویژگی‌ها (دروازه‌ی بند ۱۳-۵) ===")
    print(coverage_report(dim).to_string(index=False))
    print("\n=== شعب یکتای مکمل بر حسب کلید کسب‌وکار ===")
    mok = dim[dim.restaurant_class == "mokammel"]
    print(f"{len(mok)} نام → {mok['business_key'].nunique()} کسب‌وکار یکتا")
