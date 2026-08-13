"""بند ۳.۱۱ WBS — قفل کردن snapshot پردازش‌شده (مدل A، معیار پایان M1).

از `data/raw/raw_data_14020901_14030301.csv` تا `data/processed/dataset_v1.csv` با یک
دستور: `python -m src.data.make_dataset`. زنجیره: نرمال‌سازی متن → مدیریت زمان →
حذف تکراری کامل (۳.۸) → gate تست‌های T1-T5 (۳.۶، fail-fast) → تجمیع روی جنسیت (۳.۹) →
شبکه کامل/`is_served` (۳.۱۰، در `data/interim/`، نه بخشی از snapshot قفل‌شده چون داده‌ی
مشتق‌شده‌ی حجیم است نه داده‌ی مدل‌سازی مستقیم).
"""

import logging

from src.config import DATA_EXTERNAL, DATA_INTERIM, DATA_PROCESSED
from src.data.clean import aggregate_over_gender, build_full_grid, coverage_report, drop_exact_duplicates, parse_dates
from src.data.inspect_raw import load_aggregate
from src.data.text_normalize import normalize_columns
from src.validate import assert_suite_passes, run_aggregate_suite

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_PATH = DATA_PROCESSED / "dataset_v1.csv"
COVERAGE_GRID_PATH = DATA_INTERIM / "coverage_grid_v1.csv"


def build_dataset() -> "pd.DataFrame":  # noqa: F821 - typing only
    import pandas as pd

    logger.info("Loading raw aggregate file ...")
    df = load_aggregate()
    n_raw = len(df)

    logger.info("Normalizing Persian text (RestaurantName, FoodName) ...")
    df = normalize_columns(df, ["RestaurantName", "FoodName"])

    logger.info("Parsing dates ...")
    df = parse_dates(df)

    logger.info("Dropping exact-duplicate rows (WBS 3.8) ...")
    df, n_dropped = drop_exact_duplicates(df)
    logger.info(f"{n_raw} raw rows -> {n_dropped} exact duplicates dropped -> {len(df)} rows")

    logger.info("Running T1-T5 validation gate (WBS 3.6) ...")
    results = run_aggregate_suite(df)
    for r in results:
        logger.info(str(r))
    assert_suite_passes(results)

    logger.info("Aggregating over gender (WBS 3.9) ...")
    df_gender = aggregate_over_gender(df)
    logger.info(f"-> {len(df_gender)} rows at (d,m,r,f) grain")

    logger.info("Building full grid / is_served classification (WBS 3.10) ...")
    df_cal = pd.read_csv(DATA_EXTERNAL / "calendar_tehran.csv")
    grid = build_full_grid(df_gender, df_cal)
    DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    grid.to_csv(COVERAGE_GRID_PATH, index=False)
    logger.info(f"Saved coverage grid ({len(grid)} rows) to {COVERAGE_GRID_PATH}")
    logger.info(f"Coverage report:\n{coverage_report(grid).to_string(index=False)}")

    df_gender["is_served"] = True
    return df_gender


def save_dataset(df, output_path=OUTPUT_PATH):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved locked dataset ({len(df)} rows) to {output_path}")


if __name__ == "__main__":
    df_out = build_dataset()
    save_dataset(df_out)
    print("\nSample:")
    print(df_out.head())
    print("\ndtypes:")
    print(df_out.dtypes)
