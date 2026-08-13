"""بند ۳.۱۲ WBS — قفل کردن snapshot سطح فردی (مدل B).

از ۶ فایل خام باقیمانده (`ReserveDay1402-part2.xlsx` کنار گذاشته شده — تکراری تأییدشده)
تا `data/processed/person_dim_v1.csv` + `data/processed/person_reservation_fact_v1.csv`
با یک دستور: `python -m src.data.make_dataset_individual`.
"""

import logging

from src.config import DATA_PROCESSED
from src.data.clean_individual import (
    apply_food_mapping,
    apply_restaurant_dorm_mapping,
    build_person_dim,
    build_person_reservation_fact,
    harmonize_schema,
    load_raw_individual,
    map_reserve_status,
    normalize_individual_text,
    parse_individual_dates,
)
from src.validate import check_t2_nonnegative, check_t7

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PERSON_DIM_PATH = DATA_PROCESSED / "person_dim_v1.csv"
PERSON_FACT_PATH = DATA_PROCESSED / "person_reservation_fact_v1.csv"


def build_datasets():
    logger.info("Loading raw individual files (excluding confirmed duplicate part2) ...")
    df = load_raw_individual()
    n_raw = len(df)
    logger.info(f"{n_raw} raw rows loaded")

    logger.info("Harmonizing schema (ReceptionType, Name->Meal) ...")
    df = harmonize_schema(df)

    logger.info("Normalizing Persian text ...")
    df = normalize_individual_text(df)

    logger.info("Parsing dates ...")
    df = parse_individual_dates(df)

    logger.info("Applying restaurant/dorm mapping ...")
    df = apply_restaurant_dorm_mapping(df)

    logger.info("Applying food mapping ...")
    df = apply_food_mapping(df)

    logger.info("Mapping ReserveStatus -> dont_receive (statistical mapping, WBS 3.12) ...")
    df = map_reserve_status(df)

    logger.info("Running validation gate (T2, T7 with composite key) ...")
    t2 = check_t2_nonnegative(df, ["Count"])
    # (Reserveid, source_file) به تنهایی کافی نیست: بررسی دستی ۷ نقض کشف کرد که Reserveid
    # گاهی یک تراکنش را روی چند ردیف غذای مجزا می‌شکند (غذای اصلی + قلم همراه، با Count=1
    # روی یکی و Count=0 روی دیگری) — نه خطای تکرار. افزودن FoodName کلید را یکتا می‌کند
    # بدون حذف هیچ ردیف واقعی (بند ۳.۱۲/یافته‌ی T7، ردیف ۱۹ decision_log).
    t7 = check_t7(df, key_cols=["Reserveid", "source_file", "FoodName"])
    logger.info(str(t2))
    logger.info(str(t7))
    if not t2.passed:
        raise AssertionError(f"Validation failed: {t2}")
    if not t7.passed:
        raise AssertionError(f"Validation failed: {t7}: {t7.violations[['Reserveid', 'source_file']].head()}")

    logger.info("Splitting fact/dimension (WBS 3.12) ...")
    dim = build_person_dim(df)
    fact = build_person_reservation_fact(df)
    logger.info(f"person_dim: {len(dim)} rows | person_reservation_fact: {len(fact)} rows")

    return dim, fact


def save_datasets(dim, fact):
    PERSON_DIM_PATH.parent.mkdir(parents=True, exist_ok=True)
    dim.to_csv(PERSON_DIM_PATH, index=False)
    fact.to_csv(PERSON_FACT_PATH, index=False)
    logger.info(f"Saved {PERSON_DIM_PATH}")
    logger.info(f"Saved {PERSON_FACT_PATH}")


if __name__ == "__main__":
    dim_out, fact_out = build_datasets()
    save_datasets(dim_out, fact_out)
    print("\nperson_dim sample:")
    print(dim_out.head())
    print("\nperson_reservation_fact sample:")
    print(fact_out.head())
    print("\ndont_receive rate:", fact_out["dont_receive"].mean())
