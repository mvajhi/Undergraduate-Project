"""Profile and integrity-check the raw nutrition-system data (WBS بند ۲.۱).

Two sources:
- Aggregate: data/raw/raw_data_14020901_14030301.csv        -> (d, m, r, f) گرانولاریتی
- Individual: data/raw/per_person_raw_data/Reserve*.xlsx      -> سطح رزرو (PersonId)

Running this module end-to-end reproduces every number quoted in
doc/data_dictionary.md, doc/data_dictionary_individual.md and the
preliminary section of doc/leakage_audit.md.
"""

import glob
import logging
from pathlib import Path

import jdatetime
import pandas as pd

from src.config import DATA_RAW

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

AGGREGATE_PATH = DATA_RAW / "raw_data_14020901_14030301.csv"
INDIVIDUAL_DIR = DATA_RAW / "per_person_raw_data"

# فقط دو وعده‌ای که در فایل تجمیعی هم هستند قابل تطبیق‌اند؛ صبحانه/سحر معادلی در تجمیعی ندارند.
MEAL_MAP = {"شام": "dinner", "ناهار": "lunch"}


# ---------------------------------------------------------------------------
# فایل تجمیعی
# ---------------------------------------------------------------------------


def load_aggregate() -> pd.DataFrame:
    return pd.read_csv(AGGREGATE_PATH)


def profile_columns(df: pd.DataFrame) -> pd.DataFrame:
    """پروفایل عمومی هر ستون: نوع، ٪گمشده، n-unique، دامنه/نمونه — برای هر دو منبع قابل استفاده است."""
    rows = []
    for col in df.columns:
        s = df[col]
        is_num = pd.api.types.is_numeric_dtype(s)
        rows.append(
            {
                "column": col,
                "dtype": str(s.dtype),
                "pct_missing": round(100 * s.isnull().mean(), 3),
                "n_unique": s.nunique(),
                "min": s.min() if is_num and s.notna().any() else None,
                "max": s.max() if is_num and s.notna().any() else None,
                "example": s.dropna().iloc[0] if s.notna().any() else None,
            }
        )
    return pd.DataFrame(rows)


def integrity_checks_aggregate(df: pd.DataFrame) -> dict:
    """T1، T3، T4، T5 (بند ۳-۶ WBS) + شمارش ردیف‌های کاملاً تکراری."""
    t1_violations = (
        df["Reservation"] != (df["ReceiveWithCard"] + df["ReceiveWithCode"] + df["DontReceive"])
    ).sum()
    t3_violations = (df["DontReceive"] > df["Reservation"]).sum()
    t4_violations = (df["Reservation"] <= 0).sum()
    key_cols = ["DateReserve", "Meal", "RestaurantName", "FoodName", "Gender"]
    t5_violations = df.duplicated(subset=key_cols).sum()
    full_dup = df.duplicated().sum()
    return {
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "date_range_jalali": (df["DateReserve"].min(), df["DateReserve"].max()),
        "date_range_gregorian": (df["DateReserveGregorian"].min(), df["DateReserveGregorian"].max()),
        "T1_violations": int(t1_violations),
        "T3_violations": int(t3_violations),
        "T4_violations": int(t4_violations),
        "T5_dup_key_violations": int(t5_violations),
        "full_duplicate_rows": int(full_dup),
    }


def check_dayofweek_convention(df: pd.DataFrame) -> float:
    """تأیید قرارداد DayOfWeek: شنبه=۰ ... جمعه=۶ (نسبت تطابق با گرگوری، باید ۱.۰ باشد)."""
    tmp = df[["DateReserveGregorian", "DayOfWeek"]].drop_duplicates()
    py_weekday = pd.to_datetime(tmp["DateReserveGregorian"]).dt.weekday  # Mon=0..Sun=6
    iran_weekday_guess = (py_weekday + 2) % 7  # Sat=0..Fri=6
    return float((iran_weekday_guess == tmp["DayOfWeek"]).mean())


# ---------------------------------------------------------------------------
# فایل‌های سطح فردی
# ---------------------------------------------------------------------------


def individual_files() -> list[str]:
    return sorted(glob.glob(str(INDIVIDUAL_DIR / "Reserve*.xlsx")))


def dailysell_files() -> list[str]:
    return sorted(glob.glob(str(INDIVIDUAL_DIR / "dailysell" / "*.xlsx")))


def load_individual_by_file() -> dict[str, pd.DataFrame]:
    """هر فایل ماهانه را جدا برمی‌گرداند (لازم برای بررسی drift بین ماه‌ها)."""
    out = {}
    for f in individual_files():
        logger.info(f"Loading {f} ...")
        out[Path(f).name] = pd.read_excel(f)
    return out


def load_individual_all(by_file: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    by_file = by_file or load_individual_by_file()
    frames = []
    for name, df in by_file.items():
        df = df.copy()
        df["_source_file"] = name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def reserve_status_by_month(by_file: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """مقادیر یکتای ReserveStatus به‌تفکیک فایل ماهانه — بررسی drift معنایی."""
    rows = []
    for name, df in by_file.items():
        rows.append({"file": name, "reserve_status_values": sorted(df["ReserveStatus"].unique().tolist())})
    return pd.DataFrame(rows)


def check_t7(df_all: pd.DataFrame) -> dict:
    """T7: یکتایی سراسری Reserveid در تمام فایل‌های ماهانه."""
    n = len(df_all)
    n_unique = df_all["Reserveid"].nunique()
    return {"n_rows": n, "n_unique_reserveid": n_unique, "violations": n - n_unique}


def check_t7_per_file(by_file: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """T7 درون هر فایل به‌تنهایی — برای تفکیک «تکرار درونی» از «همپوشانی بین فایل‌ها»."""
    rows = []
    for name, df in by_file.items():
        n = len(df)
        n_unique = df["Reserveid"].nunique()
        rows.append({"file": name, "n_rows": n, "n_unique": n_unique, "internal_duplicates": n - n_unique})
    return pd.DataFrame(rows)


def check_reception_type_mapping(by_file: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """رابطه‌ی Reception (متنی) و ReceptionType (عددی) در فایل‌هایی که هر دو را دارند."""
    frames = []
    for name, df in by_file.items():
        if "ReceptionType" not in df.columns:
            continue
        ct = pd.crosstab(df["Reception"].fillna("NaN"), df["ReceptionType"].fillna(-1))
        ct.insert(0, "_source_file", name)
        frames.append(ct)
    return pd.concat(frames) if frames else pd.DataFrame()


def check_code_name_consistency(df: pd.DataFrame, code_col: str, name_col: str) -> dict:
    g = df.groupby(code_col)[name_col].nunique()
    return {"n_codes": len(g), "codes_with_gt1_name": int((g > 1).sum())}


def _normalize_jalali_slash(s: str) -> str:
    """'1402/9/1' -> '1402-09-01' (فرمت فایل تجمیعی)."""
    y, m, d = s.split("/")
    return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"


def build_t6_comparison(df_individual_all: pd.DataFrame, df_aggregate: pd.DataFrame) -> dict:
    """تست T6: مقایسه‌ی Σ Count فایل فردی (فقط ناهار/شام) با Reservation فایل تجمیعی.

    دو سطح گزارش می‌شود:
    - دقیق (d, meal, restaurant, food): محدود به تفاوت نام‌گذاری سلف در دو فایل.
    - درشت‌تر (d, meal, food) بدون سلف: برای سنجش اینکه آیا خودِ تفاوت نام سلف عامل
      اصلی عدم‌تطابق است یا مشکل داده‌ای واقعی‌تری هم هست.
    """
    ind = df_individual_all.copy()
    ind["meal_en"] = ind["Name"].map(MEAL_MAP)
    excluded = ind[ind["meal_en"].isna()]["Name"].value_counts().to_dict()
    ind_matched = ind[ind["meal_en"].notna()].copy()
    ind_matched["date_jalali"] = ind_matched["DateReserve"].apply(_normalize_jalali_slash)

    fine = (
        ind_matched.groupby(["date_jalali", "meal_en", "RestaurantName", "FoodName"])["Count"]
        .sum()
        .reset_index()
        .rename(columns={"Count": "sum_count_individual", "meal_en": "Meal"})
    )
    agg_fine = (
        df_aggregate.groupby(["DateReserve", "Meal", "RestaurantName", "FoodName"])["Reservation"]
        .sum()
        .reset_index()
        .rename(columns={"DateReserve": "date_jalali", "Reservation": "reservation_aggregate"})
    )
    merged_fine = pd.merge(
        fine, agg_fine, on=["date_jalali", "Meal", "RestaurantName", "FoodName"], how="outer", indicator=True
    )
    exact_match = merged_fine[merged_fine["_merge"] == "both"]
    exact_match_rate_rows = len(exact_match) / len(merged_fine)
    exact_value_match_rate = (
        exact_match["sum_count_individual"] == exact_match["reservation_aggregate"]
    ).mean()

    coarse_ind = (
        ind_matched.groupby(["date_jalali", "meal_en", "FoodName"])["Count"]
        .sum()
        .reset_index()
        .rename(columns={"Count": "sum_count_individual", "meal_en": "Meal"})
    )
    coarse_agg = (
        df_aggregate.groupby(["DateReserve", "Meal", "FoodName"])["Reservation"]
        .sum()
        .reset_index()
        .rename(columns={"DateReserve": "date_jalali", "Reservation": "reservation_aggregate"})
    )
    merged_coarse = pd.merge(
        coarse_ind, coarse_agg, on=["date_jalali", "Meal", "FoodName"], how="outer", indicator=True
    )
    coarse_match = merged_coarse[merged_coarse["_merge"] == "both"]
    coarse_match_rate_rows = len(coarse_match) / len(merged_coarse)
    coarse_value_match_rate = (
        coarse_match["sum_count_individual"] == coarse_match["reservation_aggregate"]
    ).mean()

    ind_restaurants = set(ind_matched["RestaurantName"].unique())
    agg_restaurants = set(df_aggregate["RestaurantName"].unique())

    return {
        "excluded_meal_row_counts": excluded,
        "n_individual_rows_matched_meal": len(ind_matched),
        "fine_grain_key_match_rate": round(exact_match_rate_rows, 4),
        "fine_grain_value_match_rate_when_key_matches": round(exact_value_match_rate, 4),
        "coarse_grain_key_match_rate": round(coarse_match_rate_rows, 4),
        "coarse_grain_value_match_rate_when_key_matches": round(coarse_value_match_rate, 4),
        "individual_restaurant_names_not_in_aggregate": len(ind_restaurants - agg_restaurants),
        "n_individual_restaurant_names": len(ind_restaurants),
        "n_aggregate_restaurant_names": len(agg_restaurants),
    }


def check_dailysell_overlap() -> dict:
    """بررسی ابهام‌های ثبت‌شده در data_manifest.md: هم‌پوشانی dailysell/ با Reserve*، و دو فایل Bahman."""
    files = dailysell_files()
    bahman_a = [f for f in files if f.endswith("DailySellBahman1402.xlsx")][0]
    bahman_b = [f for f in files if f.endswith("DailySellBahman1402(1).xlsx")][0]
    df_a = pd.read_excel(bahman_a)
    df_b = pd.read_excel(bahman_b)
    key_cols = ["DateDailySellReserve", "RestaurantName", "FoodName", "Meal"]
    a_keys = set(map(tuple, df_a[key_cols].values))
    b_keys = set(map(tuple, df_b[key_cols].values))
    return {
        "bahman_a_rows": len(df_a),
        "bahman_b_rows": len(df_b),
        "bahman_a_sum_count": int(df_a["Count"].sum()),
        "bahman_b_sum_count": int(df_b["Count"].sum()),
        "shared_keys": len(a_keys & b_keys),
        "a_only_keys": len(a_keys - b_keys),
        "b_only_keys": len(b_keys - a_keys),
    }


if __name__ == "__main__":
    print("=" * 70)
    print("AGGREGATE FILE:", AGGREGATE_PATH)
    print("=" * 70)
    df_agg = load_aggregate()
    print(profile_columns(df_agg).to_string())
    print()
    print("Integrity checks:", integrity_checks_aggregate(df_agg))
    print("DayOfWeek convention match rate (Saturday=0):", check_dayofweek_convention(df_agg))

    print()
    print("=" * 70)
    print("INDIVIDUAL FILES:", INDIVIDUAL_DIR)
    print("=" * 70)
    by_file = load_individual_by_file()
    for name, df in by_file.items():
        print(f"{name}: {len(df)} rows")
    df_ind_all = load_individual_all(by_file)
    print()
    print("Column profile (concatenated):")
    print(profile_columns(df_ind_all.drop(columns=["_source_file"])).to_string())
    print()
    print("ReserveStatus by month:")
    print(reserve_status_by_month(by_file).to_string())
    print()
    print("T7 (Reserveid global uniqueness):", check_t7(df_ind_all))
    print("T7 per file (internal duplicates only):")
    print(check_t7_per_file(by_file).to_string())
    print()
    print("Reception <-> ReceptionType crosstab (files with both columns):")
    print(check_reception_type_mapping(by_file))
    print()
    print("CollegeCode<->CollegeName:", check_code_name_consistency(df_ind_all, "CollegeCode", "CollegeName"))
    print("FieldCode<->FieldName:", check_code_name_consistency(df_ind_all, "FieldCode", "FieldName"))
    print("DegreeCode<->DegreeName:", check_code_name_consistency(df_ind_all, "DegreeCode", "DegreeName"))
    print()
    print("Name (meal) unique values:", sorted(df_ind_all["Name"].unique().tolist()))
    print("PersonType unique values:", sorted(df_ind_all["PersonType"].unique().tolist()))
    print("Comment value counts:")
    print(df_ind_all["Comment"].value_counts())
    print("Reception value counts (incl. NaN):")
    print(df_ind_all["Reception"].value_counts(dropna=False))
    print("Price describe:", df_ind_all["Price"].describe().to_dict(), "nunique:", df_ind_all["Price"].nunique())
    print("GroupName (dorm) n_unique raw variants:", df_ind_all["GroupName"].nunique())
    print("n_unique PersonId:", df_ind_all["PersonId"].nunique())
    print("CollegeCode null rows sample:")
    print(df_ind_all[df_ind_all["CollegeCode"].isna()][["CollegeName", "FieldName", "_source_file"]].head())

    print()
    print("=" * 70)
    print("T6 cross-check (individual Σ Count vs aggregate Reservation)")
    print("=" * 70)
    print(build_t6_comparison(df_ind_all, df_agg))

    print()
    print("=" * 70)
    print("dailysell/ overlap check (Bahman duplicate-file ambiguity)")
    print("=" * 70)
    print(check_dailysell_overlap())
