"""تست‌های صحت T1-T7 (بند ۳.۶ WBS) — assert های ساده، بدون `pandera`/`great_expectations`.

هر `check_*` یک `ValidationResult` برمی‌گرداند: تعداد نقض + جدول ردیف‌های ناقض (برای بررسی
الگو/تصمیم مستند طبق بند ۳.۶). قابل‌استفاده هم روی فایل تجمیعی (T1-T5) هم فایل فردی (T6-T7).
"""

from dataclasses import dataclass

import pandas as pd


@dataclass
class ValidationResult:
    name: str
    n_violations: int
    violations: pd.DataFrame

    @property
    def passed(self) -> bool:
        return self.n_violations == 0

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"<{self.name} {status}: {self.n_violations} violations>"


def check_t1(df: pd.DataFrame) -> ValidationResult:
    """T1: Reservation == ReceiveWithCard + ReceiveWithCode + DontReceive."""
    mismatch = df["Reservation"] != (df["ReceiveWithCard"] + df["ReceiveWithCode"] + df["DontReceive"])
    return ValidationResult("T1_reservation_balance", int(mismatch.sum()), df[mismatch])


def check_t2_nonnegative(df: pd.DataFrame, count_cols: list[str]) -> ValidationResult:
    """T2: تمام ستون‌های شمارشی >= 0."""
    mask = pd.Series(False, index=df.index)
    for c in count_cols:
        mask = mask | (df[c] < 0)
    return ValidationResult("T2_nonnegative_counts", int(mask.sum()), df[mask])


def check_t3(df: pd.DataFrame) -> ValidationResult:
    """T3: DontReceive <= Reservation."""
    mask = df["DontReceive"] > df["Reservation"]
    return ValidationResult("T3_dontreceive_le_reservation", int(mask.sum()), df[mask])


def check_t4(df: pd.DataFrame) -> ValidationResult:
    """T4: Reservation > 0 (وگرنه rho تعریف‌نشده است)."""
    mask = df["Reservation"] <= 0
    return ValidationResult("T4_reservation_positive", int(mask.sum()), df[mask])


def check_t5(df: pd.DataFrame, key_cols: list[str]) -> ValidationResult:
    """T5: عدم وجود کلید تکراری در سطح (d, m, r, f, g)."""
    dup = df.duplicated(subset=key_cols, keep=False)
    return ValidationResult("T5_no_duplicate_key", int(df.duplicated(subset=key_cols).sum()), df[dup])


def check_t6(
    df_individual: pd.DataFrame,
    df_aggregate: pd.DataFrame,
    key_cols: list[str],
    individual_count_col: str = "Count",
    aggregate_count_col: str = "Reservation",
) -> ValidationResult:
    """T6: Σ Count فایل فردی == Reservation فایل تجمیعی به‌ازای هر (d,m,r,f)."""
    ind_sum = df_individual.groupby(key_cols)[individual_count_col].sum().reset_index()
    agg_sum = df_aggregate.groupby(key_cols)[aggregate_count_col].sum().reset_index()
    merged = ind_sum.merge(agg_sum, on=key_cols, how="outer", indicator=True)
    merged[individual_count_col] = merged[individual_count_col].fillna(0)
    merged[aggregate_count_col] = merged[aggregate_count_col].fillna(0)
    mismatch = (merged["_merge"] != "both") | (merged[individual_count_col] != merged[aggregate_count_col])
    return ValidationResult("T6_individual_vs_aggregate_sum", int(mismatch.sum()), merged[mismatch])


def check_t7(df: pd.DataFrame, key_cols: list[str] | str = "Reserveid") -> ValidationResult:
    """T7: هر Reserveid (یا کلید مرکب Reserveid+فایل/ماه) دقیقاً یک‌بار ظاهر می‌شود."""
    if isinstance(key_cols, str):
        key_cols = [key_cols]
    dup = df.duplicated(subset=key_cols, keep=False)
    return ValidationResult("T7_unique_reserveid", int(df.duplicated(subset=key_cols).sum()), df[dup])


def run_aggregate_suite(df: pd.DataFrame) -> list[ValidationResult]:
    """T1-T5 روی فایل تجمیعی (یا `dataset_v1.csv` پس از پاکسازی)."""
    key_cols = ["DateReserve", "Meal", "RestaurantName", "FoodName", "Gender"]
    return [
        check_t1(df),
        check_t2_nonnegative(df, ["Reservation", "ReceiveWithCard", "ReceiveWithCode", "DontReceive"]),
        check_t3(df),
        check_t4(df),
        check_t5(df, key_cols),
    ]


def assert_suite_passes(results: list[ValidationResult]) -> None:
    failed = [r for r in results if not r.passed]
    if failed:
        detail = "; ".join(repr(r) for r in failed)
        raise AssertionError(f"Validation suite failed: {detail}")


if __name__ == "__main__":
    from src.data.inspect_raw import load_aggregate

    df_agg = load_aggregate()
    results = run_aggregate_suite(df_agg)
    for r in results:
        print(r)
