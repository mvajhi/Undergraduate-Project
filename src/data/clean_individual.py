"""پاکسازی اختصاصی داده سطح فردی (مدل B) — بند ۳.۱۲ WBS.

پیش‌نیازها (باید قبلاً اجرا شده باشند): `src/data/mapping.py` → `restaurant_mapping.csv`,
`dorm_mapping.csv`, `food_mapping.csv`.
"""

import jdatetime
import pandas as pd

from src.data.mapping import DORM_MAPPING_PATH, FOOD_MAPPING_PATH, RESTAURANT_MAPPING_PATH
from src.data.text_normalize import normalize_columns

# `ReserveDay1402-part1.xlsx` و `ReserveDay1402-part2.xlsx` در بازرسی این فاز byte-for-byte
# یکسان تشخیص داده شدند (۴۳۷٬۴۷۰ ردیف، `DataFrame.equals()` == True) — نه دو نیمه‌ی جدا
# طبق نامشان، بلکه فایل تکراری. بارگذاری هر دو باعث دوبارشماری کامل دی‌ماه در هر آمار
# سطح فردی می‌شود (ردیف ۱۸ `doc/decision_log.md`). فقط part1 در پایپ‌لاین استفاده می‌شود.
EXCLUDED_INDIVIDUAL_FILES = {"ReserveDay1402-part2.xlsx"}

MEAL_NAME_MAP = {"ناهار": "lunch", "شام": "dinner", "صبحانه": "breakfast", "سحر": "sahar"}

TEXT_COLS = [
    "Gender",
    "EducationSession",
    "CollegeName",
    "FieldName",
    "DegreeName",
    "GroupName",
    "RestaurantName",
    "FoodName",
    "Comment",
    "Reception",
]

# نگاشت آماری `ReserveStatus` → معادل باینری `DontReceive` (بند ۳.۱۲ WBS، «حدس‌زدن ممنوع»).
# روش: زیرمجموعه‌ی رزروهای فایل فردی که پس از نگاشت سلف/غذا (بند ۳.۳-۳.۴) با کلید
# (d,m,r,f) به `dataset_v1.csv` می‌پیوندند (۹۸٫۵٪ نرخ تطابق کلید، در برابر ۰٪/۱٫۵۸٪ فاز ۲)،
# سپس مقایسه‌ی Σ Count هر مقدار ReserveStatus با Recv/NoRecv واقعی همان گروه:
#   فرضیه «دریافت‌شده=Recv، بقیه=NoRecv» → نرخ تطابق دقیق ۹۰٫۸٪ برای NoRecv (MAE=۰٫۳۶),
#   ۸۵٫۵٪ برای Recv (MAE=۳٫۴) روی ۷٬۵۳۸ گروه؛ فرضیه‌ی جایگزین (کنار گذاشتن «ارسال نشده»
#   از NoRecv) نرخ تطابق را به ۶۳٫۹٪ می‌رساند — پس شواهد آماری قاطعانه «ارسال نشده» را هم
#   در دسته‌ی عدم‌دریافت قرار می‌دهد، نه دریافت. جزئیات کامل: `doc/decision_log.md` ردیف ۱۹.
_DONT_RECEIVE_STATUSES = {"ارسال نشده", "دریافت نشده", "منقضی شده"}
_RECEIVE_STATUSES = {"دریافت شده"}


def load_raw_individual() -> pd.DataFrame:
    from src.data.inspect_raw import load_individual_all, load_individual_by_file

    by_file = load_individual_by_file()
    for excluded in EXCLUDED_INDIVIDUAL_FILES:
        by_file.pop(excluded, None)
    return load_individual_all(by_file)


def harmonize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """اسکیمای ۶ فایل باقیمانده را یکسان می‌کند: `ReceptionType` فقط در برخی فایل‌ها وجود
    دارد (بند ۲.۱.۴ WBS) — این‌جا صریح به NaN تبدیل می‌شود، نه فرض ضمنی وجود ستون."""
    df = df.copy()
    if "ReceptionType" not in df.columns:
        df["ReceptionType"] = pd.NA
    df = df.rename(columns={"Name": "Meal", "_source_file": "source_file"})
    df["Meal"] = df["Meal"].map(MEAL_NAME_MAP)
    return df


def normalize_individual_text(df: pd.DataFrame) -> pd.DataFrame:
    return normalize_columns(df, TEXT_COLS)


def parse_individual_dates(df: pd.DataFrame) -> pd.DataFrame:
    """`DateReserve` خام (`'1402/9/1'`, بدون صفر-پد) → `date_jalali` هم‌فرمت با فایل تجمیعی
    (`'1402-09-01'`) + `date_gregorian` واقعی."""
    df = df.copy()

    def _parse(s: str) -> tuple[str, pd.Timestamp]:
        y, m, d = (int(x) for x in s.split("/"))
        greg = jdatetime.date(y, m, d).togregorian()
        return f"{y:04d}-{m:02d}-{d:02d}", pd.Timestamp(greg)

    parsed = df["DateReserve"].map(_parse)
    df["date_jalali"] = parsed.map(lambda t: t[0])
    df["date_gregorian"] = parsed.map(lambda t: t[1])
    return df


def apply_restaurant_dorm_mapping(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rest_map = pd.read_csv(RESTAURANT_MAPPING_PATH).set_index("raw_name")["canonical_name"]
    dorm_map = pd.read_csv(DORM_MAPPING_PATH).set_index("raw_name")["canonical_name"]
    df["restaurant_canonical"] = df["RestaurantName"].map(rest_map)
    df["dorm_canonical"] = df["GroupName"].map(dorm_map)
    return df


def apply_food_mapping(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    food_map = pd.read_csv(FOOD_MAPPING_PATH).set_index("raw_food_name")
    df["main_food"] = df["FoodName"].map(food_map["main_food"])
    df["food_canonical"] = df["FoodName"].map(food_map["canonical_food"])
    df["food_matched"] = df["FoodName"].map(food_map["matched"]).fillna(False)
    df["has_extras"] = df["FoodName"].map(food_map["has_extras"]).fillna(False)
    df["extras_list"] = df["FoodName"].map(food_map["extras_list"]).fillna("")
    return df


def map_reserve_status(df: pd.DataFrame) -> pd.DataFrame:
    """`ReserveStatus` → `dont_receive` (bool) طبق نگاشت آماری بالا."""
    df = df.copy()
    unexpected = set(df["ReserveStatus"].unique()) - _DONT_RECEIVE_STATUSES - _RECEIVE_STATUSES
    if unexpected:
        raise ValueError(f"Unmapped ReserveStatus values found (data drift?): {unexpected}")
    df["dont_receive"] = df["ReserveStatus"].isin(_DONT_RECEIVE_STATUSES)
    return df


# ---------------------------------------------------------------------------
# تفکیک fact/dimension (بند ۳.۱۲ WBS)
# ---------------------------------------------------------------------------

PERSON_DIM_COLS = [
    "PersonId",
    "Gender",
    "CollegeCode",
    "CollegeName",
    "FieldCode",
    "FieldName",
    "DegreeCode",
    "DegreeName",
    "dorm_canonical",
    "EducationSession",
    "PersonType",
]

FACT_COLS = [
    "Reserveid",
    "source_file",
    "PersonId",
    "date_jalali",
    "date_gregorian",
    "Meal",
    "restaurant_canonical",
    "FoodName",
    "main_food",
    "food_canonical",
    "food_matched",
    "has_extras",
    "extras_list",
    "Count",
    "ReserveStatus",
    "dont_receive",
    "Price",
    "Reception",
    "ReceptionType",
    "Comment",
]


def build_person_dim(df: pd.DataFrame) -> pd.DataFrame:
    """یک ردیف به ازای هر `PersonId` — **آخرین** مقدار شناخته‌شده (بر اساس تاریخ) برای هر
    ویژگی نسبتاً ثابت، چون ممکن است در طول ۷ ماه تغییر کند (مثلاً جابه‌جایی خوابگاه)."""
    df_sorted = df.sort_values("date_gregorian")
    dim = df_sorted.groupby("PersonId")[
        [c for c in PERSON_DIM_COLS if c != "PersonId"]
    ].last()
    return dim.reset_index()


def build_person_reservation_fact(df: pd.DataFrame) -> pd.DataFrame:
    """یک ردیف به ازای هر قلم غذای رزروشده؛ کلید یکتا = (`Reserveid`, `source_file`, `FoodName`).

    چرا نه فقط (`Reserveid`, `source_file`)؟ بازرسی ۷ نقض T7 نشان داد در موارد نادر (۷ از
    ~۲٫۱ میلیون) یک `Reserveid` واحد یک تراکنش را روی دو ردیف غذای متفاوت می‌شکند (غذای
    اصلی + قلم همراه، با `Count=1` روی یکی و `Count=0` روی دیگری) — این خطای تکرار نیست،
    ویژگی واقعی سامانه‌ی مبدأ است؛ هر دو ردیف نگه داشته می‌شوند.
    """
    return df[FACT_COLS].copy()
