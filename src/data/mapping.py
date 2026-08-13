"""نگاشت نام سلف/خوابگاه (بند ۳.۳) و تجزیه‌ی نام غذا از مخلفات (بند ۳.۴).

روش بند ۳.۳: نرمال‌سازی متن (`text_normalize`) → تطبیق فازی (`rapidfuzz`) → **بازبینی
دستی کامل** جدول نگاشت. بازبینی دستی روی خروجی خام فازی این ماژول انجام شد (نتایج در
`doc/decision_log.md`)؛ دو خطای واقعی پیدا شد که با وجود امتیاز بالا (۹۰) غلط بودند
(WRatio به‌خاطر partial-ratio به رشته‌ی زیرمجموعه امتیاز بالا می‌دهد، حتی وقتی معنایی
متفاوت است) — این دو در `_RESTAURANT_MANUAL_OVERRIDES` تصحیح شده‌اند. جدول نهایی همیشه
به فایل صریح ذخیره می‌شود (`data/external/*.csv`)؛ پایپ‌لاین (`clean_individual.py`)
هرگز این نگاشت را hardcode نمی‌کند، فقط فایل CSV را می‌خواند.
"""

import re

import pandas as pd
from rapidfuzz import fuzz, process

from src.config import DATA_EXTERNAL
from src.data.text_normalize import normalize_persian_text

RESTAURANT_MAPPING_PATH = DATA_EXTERNAL / "restaurant_mapping.csv"
DORM_MAPPING_PATH = DATA_EXTERNAL / "dorm_mapping.csv"
FOOD_MAPPING_PATH = DATA_EXTERNAL / "food_mapping.csv"

FUZZY_SCORE_THRESHOLD = 90.0

# ---------------------------------------------------------------------------
# ۳.۳ نگاشت نام سلف (فایل فردی، ۷۳ نام خام → ۳۲ نام کانونیک فایل تجمیعی)
# ---------------------------------------------------------------------------

# ردیف‌هایی که تطبیق فازی خودکار (WRatio) با وجود امتیاز بالا (۹۰) اشتباه تشخیص داد
# (بازبینی دستی کامل جدول، طبق قاعده‌ی WBS ۳.۳):
#   - 'دانشکده-سلف علوم اجتماعی' → فازی به 'علوم' می‌رفت (باید 'علوم اجتماعی' باشد)
#   - 'دانشکده-سلف ژئوفیزیک'     → فازی به 'فیزیک' می‌رفت (باید 'ژئوفیزیک' باشد)
#   - 'تست اپ' یک ردیف تستی/زباله‌ی سامانه است، نه یک سلف واقعی → از نگاشت کنار گذاشته می‌شود
_RESTAURANT_MANUAL_OVERRIDES: dict[str, str | None] = {
    "دانشکده-سلف علوم اجتماعی": "علوم اجتماعی",
    "دانشکده-سلف ژئوفیزیک": "ژئوفیزیک",
    "تست اپ": None,
}


def _unique_normalized(values) -> list[str]:
    return sorted({v for v in (normalize_persian_text(x) for x in values) if v})


def build_restaurant_mapping(
    individual_restaurant_names,
    aggregate_restaurant_names,
) -> pd.DataFrame:
    """نگاشت `raw_name → canonical_name` را می‌سازد و در `restaurant_mapping.csv` ذخیره می‌کند.

    ورودی‌ها ستون خام (`RestaurantName`) هر دو منبع‌اند (قبل یا بعد از نرمال‌سازی، فرقی
    نمی‌کند — این تابع خودش نرمال می‌کند).
    """
    canonical = _unique_normalized(aggregate_restaurant_names)
    raw_names = _unique_normalized(individual_restaurant_names)

    # نکته‌ی مهم از بازبینی دستی: امتیاز WRatio پایین (مثلاً ۶۰-۸۵.۵) непременно به معنی
    # نگاشت غلط نیست — رشته‌های خام طولانی با پسوندهایی مثل «سلف مهمان-روزانه ۵۰نفر» امتیاز
    # را پایین می‌آورند بدون اینکه match اشتباه شود (بازبینی دستی این ۹ مورد تأیید کرد که
    # match صحیح است). عکسش هم رخ داد: دو مورد با امتیاز ۹۰ (`علوم اجتماعی`→`علوم`,
    # `ژئوفیزیک`→`فیزیک`) غلط بودند. پس امتیاز فقط برای **اولویت‌بندی بازبینی دستی** استفاده
    # شد، نه به‌عنوان قطع خودکار — چون بازبینی کامل ۷۳ ردیف انجام شد، نتیجه‌ی فازی (پس از
    # اعمال override های بالا) مستقیماً به‌عنوان canonical_name نهایی پذیرفته می‌شود.
    rows = []
    for raw in raw_names:
        if raw in _RESTAURANT_MANUAL_OVERRIDES:
            target = _RESTAURANT_MANUAL_OVERRIDES[raw]
            match_score = None
            method = "excluded" if target is None else "manual_override"
        else:
            target, score, _ = process.extractOne(raw, canonical, scorer=fuzz.WRatio)
            match_score = round(float(score), 2)
            method = "fuzzy_reviewed" if score >= FUZZY_SCORE_THRESHOLD else "fuzzy_reviewed_low_score"
        rows.append({"raw_name": raw, "canonical_name": target, "match_score": match_score, "method": method})

    df = pd.DataFrame(rows)
    DATA_EXTERNAL.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESTAURANT_MAPPING_PATH, index=False)
    return df


# ---------------------------------------------------------------------------
# ۳.۳ نگاشت نام خوابگاه/گروه (فایل فردی، ستون GroupName)
# ---------------------------------------------------------------------------


def build_dorm_mapping(group_names) -> pd.DataFrame:
    """نگاشت `raw_name → canonical_name` برای `GroupName`.

    برخلاف نام سلف، `GroupName` معادل خارجی در فایل تجمیعی ندارد (فقط ویژگی جمعیتی
    فایل فردی است) — پس نگاشت به یک لیست مرجع بیرونی معنا ندارد؛ هدف فقط یکسان‌سازی
    نگارش‌های مختلف *درون همین ستون* است. بررسی فازی خودهمبستگی (`rapidfuzz`، آستانه‌ی
    بالا) نشان داد پس از نرمال‌سازی متن، هیچ دو مقدار واقعاً یک موجودیت نیستند (بالاترین
    شباهت‌ها بین ساختمان‌های مجزای هم‌نام‌خانواده مثل «قدس» و «قدس ۳» است، نه تکرار واقعی)
    — بنابراین نگاشت نهایی، پس از نرمال‌سازی متن، همانی (identity) است؛ این یافته‌ی
    داده‌محور است، نه فرض ازپیش‌تعیین‌شده.
    """
    raw_names = _unique_normalized(group_names)
    df = pd.DataFrame({"raw_name": raw_names, "canonical_name": raw_names, "method": "normalized_identity"})
    DATA_EXTERNAL.mkdir(parents=True, exist_ok=True)
    df.to_csv(DORM_MAPPING_PATH, index=False)
    return df


# ---------------------------------------------------------------------------
# ۳.۴ تجزیه‌ی نام غذا از مخلفات
# ---------------------------------------------------------------------------

# فایل تجمیعی هرگز از این جداکننده‌ها در FoodName استفاده نمی‌کند (بررسی‌شده)؛ فایل فردی
# مخلفات را با '+' (غالب) یا '،' جدا می‌کند. پرانتز معمولاً توضیح غذای اصلی است
# (مثلاً «فسنجان(مرغ)»، «(تکنفره)»)، نه یک قلم مخلفات جدا — پس از غذای اصلی حذف می‌شود،
# نه به extras_list اضافه.
_EXTRA_SPLIT_RE = re.compile(r"[+،]")
_PAREN_RE = re.compile(r"\s*\([^)]*\)\s*")
_MULTI_SPACE_RE = re.compile(r"\s+")

# مورد بازبینی‌شده‌ی مشکوک: تطبیق فازی «خوراک دلمه بادمجان» را به «خوراک بادمجان» می‌برد
# (امتیاز ۹۵) ولی دلمه (محتوای پرشده) و خوراک بادمجان (خورش) دو غذای متفاوت‌اند —
# عمداً نامنطبق نگه داشته می‌شود تا با غذای اشتباه یکی نشود.
_FOOD_MANUAL_OVERRIDES: dict[str, str | None] = {
    "خوراک دلمه بادمجان": None,
}


def parse_food_name(s: str | float | None) -> tuple[str | None, bool, list[str]]:
    """یک `FoodName` خام را به (غذای اصلی نرمال‌شده, has_extras, extras_list) می‌شکند."""
    s = normalize_persian_text(s)
    if not isinstance(s, str):
        return None, False, []
    parts = [p.strip() for p in _EXTRA_SPLIT_RE.split(s) if p.strip()]
    main = parts[0] if parts else s
    main = _PAREN_RE.sub(" ", main)
    main = _MULTI_SPACE_RE.sub(" ", main).strip()
    extras = parts[1:]
    return main, len(extras) > 0, extras


def build_food_mapping(individual_food_names, aggregate_food_names) -> pd.DataFrame:
    """`main_food` هر نام غذای فایل فردی را (در صورت وجود شواهد کافی) به نام کانونیک فایل
    تجمیعی نگاشت می‌کند؛ خروجی در `food_mapping.csv`.

    زیر آستانه یا در فهرست override با مقدار `None`: نگاشت اجباری انجام نمی‌شود (`canonical_food`
    = خود `main_food`، `matched=False`) — چون غذاهایی مثل نسخه‌ی سویا/پیتزا/صبحانه/خریدهای
    بوفه‌ای واقعاً معادلی در فایل تجمیعی (فقط ناهار/شام، ۹۱ غذا) ندارند؛ نگاشت اجباری آن‌ها
    به نزدیک‌ترین غذای لغوی نادرست و گمراه‌کننده است.
    """
    canonical = _unique_normalized(aggregate_food_names)
    raw_foods = _unique_normalized(individual_food_names)

    rows = []
    for raw in raw_foods:
        main, has_extras, extras = parse_food_name(raw)
        if main in _FOOD_MANUAL_OVERRIDES:
            canonical_food = _FOOD_MANUAL_OVERRIDES[main]
            score = None
            matched = False
        else:
            match, s, _ = process.extractOne(main, canonical, scorer=fuzz.WRatio)
            score = round(float(s), 2)
            matched = s >= FUZZY_SCORE_THRESHOLD
            canonical_food = match if matched else main
        rows.append(
            {
                "raw_food_name": raw,
                "main_food": main,
                "canonical_food": canonical_food,
                "matched": matched,
                "match_score": score,
                "has_extras": has_extras,
                "extras_list": "|".join(extras) if extras else "",
            }
        )

    df = pd.DataFrame(rows)
    DATA_EXTERNAL.mkdir(parents=True, exist_ok=True)
    df.to_csv(FOOD_MAPPING_PATH, index=False)
    return df


if __name__ == "__main__":
    from src.data.inspect_raw import load_aggregate, load_individual_all, load_individual_by_file

    df_agg = load_aggregate()
    by_file = load_individual_by_file()
    df_ind = load_individual_all(by_file)

    df_rest = build_restaurant_mapping(df_ind["RestaurantName"], df_agg["RestaurantName"])
    print(f"restaurant_mapping.csv: {len(df_rest)} rows -> {RESTAURANT_MAPPING_PATH}")
    print(df_rest["method"].value_counts())

    df_dorm = build_dorm_mapping(df_ind["GroupName"])
    print(f"\ndorm_mapping.csv: {len(df_dorm)} rows -> {DORM_MAPPING_PATH}")

    df_food = build_food_mapping(df_ind["FoodName"], df_agg["FoodName"])
    print(f"\nfood_mapping.csv: {len(df_food)} rows -> {FOOD_MAPPING_PATH}")
    print("matched:", df_food["matched"].sum(), "/ unmatched:", (~df_food["matched"]).sum())
