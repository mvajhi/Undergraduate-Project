"""بند ۵.۱۴ — خط لوله‌ی ساخت فیچر و تولید فیچرست‌های بند ۵.۱۲.

اجرا: `python -m src.features.build`

خروجی:
- `data/processed/features_A_v1.parquet` — ماتریس کامل فیچر سطح $(d,m,r,f)$ برای مدل A
- `data/processed/feature_sets_v1.json` — تعریف هر فیچرست (`FS_baseline` … `FS_bridge`)

ترتیب مراحل عمدی است: فیچرهای زمانی باید **پیش از** برهم‌کنش‌ها ساخته شوند، و ترکیب
رزروکنندگان پس از عامل روز (چون برهم‌کنش ضرب‌شونده به هر دو نیاز دارد).
"""

import json
import logging

import pandas as pd

from src.config import DATA_PROCESSED, set_global_seed
from src.eda_lib.runners._common import load_dataset
from src.features import aggregate_features as agg
from src.features.composition import add_composition, build_composition
from src.features.person_features import OUTPUT_PATH as PERSON_FEATURES_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

FEATURES_A_PATH = DATA_PROCESSED / "features_A_v1.parquet"
FEATURE_SETS_PATH = DATA_PROCESSED / "feature_sets_v1.json"

TARGET = "rho"
#: ستون‌هایی که هرگز فیچر نمی‌شوند (کلید، یا خروجی همان وعده)
NEVER_FEATURES = {
    "rho", "NoRecv", "Recv", "Res", "card_ratio", "is_served",
    "DateReserve", "date_gregorian", "ym", "jyear", "dow_name",
    "day_shock", "n_cells", "daily_total_res",
}


def build_feature_matrix() -> pd.DataFrame:
    df = load_dataset()
    logger.info(f"Loaded aggregate dataset: {len(df):,} rows")

    logger.info("5.1 calendar ...")
    df = agg.add_calendar(df)
    logger.info("5.8 external (precip categorical, no AQI) ...")
    df = agg.add_external(df)
    logger.info("5.5 reservation scale ...")
    df = agg.add_scale_features(df)
    logger.info("5.6 food (universal effect only) ...")
    df = agg.add_food_features(df)

    logger.info("5.2-5.4 cell-level lag / rolling / expanding ...")
    cell = agg.build_cell_series(df)
    df = agg.add_cell_time_features(df, cell)
    df = agg.add_dow_expanding(df)

    logger.info("5.17 day factor (the big one) ...")
    day = agg.build_day_factor(cell)
    df = agg.add_day_factor(df, day)

    logger.info("5.18 interactions ...")
    df = agg.add_interactions(df)

    logger.info("5.19 composition bridge (from person features) ...")
    if PERSON_FEATURES_PATH.exists():
        pf = pd.read_parquet(PERSON_FEATURES_PATH,
                             columns=["date_gregorian", "Meal", "restaurant_canonical",
                                      "person_shrunk_norecv_rate", "person_n_prior_reservations"])
        comp = build_composition(pf)
        df = add_composition(df, comp)
        logger.info(f"  composition coverage: {df['composition_mean'].notna().mean():.1%} of rows")
    else:
        logger.warning(f"{PERSON_FEATURES_PATH} not found — run person_features first; skipping 5.19")
    return df


def define_feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    """بند ۵.۱۲ — فیچرست‌های نام‌گذاری‌شده که محور آزمایش فاز ۷ می‌شوند."""
    have = set(df.columns)

    def keep(cols):
        return [c for c in cols if c in have]

    baseline = keep(["cell_dow_shrunk_rate"])
    ident = keep(["RestaurantName", "Meal", "FoodType", "RestaurantType", "city", "is_khabgah",
                  "is_lunch", "log_res"])
    calendar = keep(["dow", "jmonth", "day_of_month", "week_of_semester", "is_holiday_any",
                     "is_day_before_holiday", "is_day_after_holiday", "is_bridge_day",
                     "days_to_next_holiday", "days_since_last_holiday", "holiday_block_length",
                     "pre_holiday_x_block_len", "is_exam_period", "is_final_exam_period",
                     "days_to_exam_start", "is_ramadan"]
                    + [f"dow_{t}{k}" for t in ("sin", "cos") for k in (1, 2, 3)])
    lag = keep([f"rho_cell_lag{k}" for k in agg.LAGS]
               + [f"rho_roll_mean_{w}" for w in (7, 14, 28)]
               + ["rho_roll_std_7", "cell_expanding_rate", "cell_shrunk_rate",
                  "cell_dow_expanding_rate"])
    scale = keep(["res_vs_history", "res_vs_dow_history", "log_daily_total_res"])
    food = keep(["food_expanding_rate", "food_shrunk_rate",
                 "is_new_food", "competitor_food_rate", "food_rate_minus_competitor"])
    external = keep(["temp_min", "precip_type", "is_snow_day"])
    day = keep(["day_shock_lag1", "day_shock_lag2", "day_shock_lag7", "day_shock_roll_mean_7"])
    inter = keep(["dow_x_type", "meal_x_type", "dow_x_city", "city_x_meal"])
    comp = keep(["composition_mean", "composition_std", "composition_p90",
                 "composition_high_risk_share", "composition_coverage", "composition_n",
                 "composition_mean_dinner_only",
                 "composition_mean_x_dayshock", "composition_p90_x_dayshock",
                 "composition_high_risk_share_x_dayshock"])

    fs = {
        "FS_baseline": baseline,
        "FS_calendar": baseline + ident + calendar,
        "FS_lag": baseline + ident + calendar + lag,
        "FS_day": baseline + ident + calendar + lag + day,
        "FS_full_A": baseline + ident + calendar + lag + day + scale + food + external + inter,
        "FS_bridge": baseline + ident + calendar + lag + day + scale + food + external + inter + comp,
    }
    return {k: sorted(set(v)) for k, v in fs.items()}


def main() -> None:
    set_global_seed()
    df = build_feature_matrix()
    fs = define_feature_sets(df)

    FEATURES_A_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FEATURES_A_PATH, index=False)
    FEATURE_SETS_PATH.write_text(json.dumps(fs, ensure_ascii=False, indent=1))
    logger.info(f"Saved {FEATURES_A_PATH} ({len(df):,} rows × {df.shape[1]} cols)")

    print("\nفیچرست‌ها:")
    for k, v in fs.items():
        print(f"  {k:<14s} {len(v):>3d} فیچر")
    print("\nپوشش فیچرهای کلیدی (سهم غیرتهی):")
    for c in ["cell_dow_shrunk_rate", "rho_cell_lag1", "rho_cell_lag14", "day_shock_lag1",
              "composition_mean", "food_shrunk_rate", "res_vs_history"]:
        if c in df.columns:
            print(f"  {c:<28s} {df[c].notna().mean():>6.1%}")


if __name__ == "__main__":
    main()
