"""ممیزی نشت داده روی ماتریس ویژگی v4 — دروازه‌ی اجباری پیش از هر آموزش.

همان سه لایه‌ی `src/features/audit.py` (بند ۵.۱۳)، اجراشده روی
`features_L1_v4.parquet`. وجودِ نسخه‌ی جدا به این دلیل است که ماتریس v4 فیچرهای تازه
دارد (`cell_last_*`, `venue_last_*`, `res_vs_*_hist`) که ممیزی قدیمی نمی‌شناسدشان و
**بی‌صدا از قلم می‌اندازد**.

۱. **آزمون ساختاری** — بازسازی مستقل و brute-force هر فیچر تاریخی از روی تعریف
   ($\\text{seq}_{obs} \\le \\text{seq}_{target} - 2$) و مقایسه با مقدار ذخیره‌شده.
   این آزمون به داده وابسته نیست؛ یا تعریف رعایت شده یا نشده.
۲. **همبستگی مشکوک** — هیچ فیچری نباید $|r| > 0.95$ با هدف داشته باشد.
۳. **سقف $R^2$** — فاز ۴ نشان داد ۸۳٪ واریانس سلول شوک روزانه است (F59)، پس سقف
   واقع‌بینانه ~۰.۴–۰.۵ است. عدد بالای ۰.۹ تقریباً قطعاً نشتی است.

اجرا: `python -m src.features.audit_v4`
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import set_global_seed
from src.features.build_v4 import OUTPUT_PATH as FEATURES_PATH
from src.features.build_v4 import feature_set
from src.features.cutoff import CUTOFF_LAG

logger = logging.getLogger(__name__)

TARGET = "rho"

#: سطوح تجمیع تاریخی، دقیقاً همان‌هایی که `build_v4.add_history_features` می‌سازد.
HISTORY_LEVELS: dict[str, list[str]] = {
    "cell": ["restaurant", "dish", "meal"],
    "venue_dow": ["restaurant", "meal", "day_of_week"],
    "venue": ["restaurant", "meal"],
    "dish": ["dish", "meal"],
    "global": ["meal"],
}

SUSPICIOUS_CORR = 0.95
R2_CEILING = 0.90


def test_cutoff_invariant(df: pd.DataFrame, n_sample: int = 250, seed: int = 42) -> tuple[bool, pd.DataFrame]:
    """هر فیچر تاریخی را از روی **تعریف** بازسازی می‌کند و با مقدار ذخیره‌شده می‌سنجد.

    عمداً کُند و ساده است (فیلتر کامل به‌ازای هر ردیف نمونه) — هدف این است که مسیر
    بازسازی هیچ کدی با مسیر تولید مشترک نداشته باشد، وگرنه یک باگ مشترک هر دو را
    هم‌زمان غلط می‌کند و آزمون بی‌معنا می‌شود.
    """
    rng = np.random.default_rng(seed)
    sample = df.iloc[rng.choice(len(df), size=min(n_sample, len(df)), replace=False)]

    rows = []
    for name, keys in HISTORY_LEVELS.items():
        max_rho_err = max_n_err = 0.0
        for _, row in sample.iterrows():
            mask = np.ones(len(df), dtype=bool)
            for key in keys:
                mask &= (df[key].to_numpy() == row[key])
            mask &= df["meal_seq"].to_numpy() <= row["meal_seq"] - CUTOFF_LAG

            res_sum = df.loc[mask, "Res"].sum()
            expected_rho = df.loc[mask, "NoRecv"].sum() / res_sum if res_sum > 0 else np.nan
            expected_n = int(mask.sum())

            got_rho, got_n = row[f"{name}_hist_rho"], row[f"{name}_hist_n"]
            if not (pd.isna(expected_rho) and pd.isna(got_rho)):
                max_rho_err = max(max_rho_err, abs(float(expected_rho) - float(got_rho)))
            max_n_err = max(max_n_err, abs(expected_n - int(got_n)))

        rows.append({
            "feature": f"{name}_hist_rho/n", "max_rho_abs_err": round(max_rho_err, 12),
            "max_count_err": int(max_n_err), "passed": max_rho_err < 1e-9 and max_n_err == 0,
        })

    # `cell_last_rho`: آخرین مشاهده‌ی مجاز در همان سلول. کلید L1 یکتاست پس «آخرین»
    # تعریف روشنی دارد؛ `venue_last_rho` عمداً اینجا نیست چون در سطح سلف ساخته
    # می‌شود، نه سطح سلول (بند بعدی).
    max_err = 0.0
    for _, row in sample.iterrows():
        mask = np.ones(len(df), dtype=bool)
        for key in ["restaurant", "dish", "meal"]:
            mask &= (df[key].to_numpy() == row[key])
        mask &= df["meal_seq"].to_numpy() <= row["meal_seq"] - CUTOFF_LAG
        prior = df.loc[mask]
        expected = prior.loc[prior["meal_seq"].idxmax(), TARGET] if len(prior) else np.nan
        got = row["cell_last_rho"]
        if not (pd.isna(expected) and pd.isna(got)):
            max_err = max(max_err, abs(float(expected) - float(got)))
    rows.append({
        "feature": "cell_last_rho", "max_rho_abs_err": round(max_err, 12),
        "max_count_err": 0, "passed": max_err < 1e-9,
    })

    # `venue_last_rho`: نرخ **وزن‌دار با حجم** سلف در آخرین وعده‌ی مجاز.
    venue = (
        df.groupby(["restaurant", "meal", "date_gregorian", "meal_seq"], observed=True)
        .agg(nr=("NoRecv", "sum"), res=("Res", "sum")).reset_index()
    )
    venue["vrho"] = venue["nr"] / venue["res"].replace(0, np.nan)
    max_err = 0.0
    for _, row in sample.iterrows():
        mask = (
            (venue["restaurant"].to_numpy() == row["restaurant"])
            & (venue["meal"].to_numpy() == row["meal"])
            & (venue["meal_seq"].to_numpy() <= row["meal_seq"] - CUTOFF_LAG)
        )
        prior = venue.loc[mask]
        expected = prior.loc[prior["meal_seq"].idxmax(), "vrho"] if len(prior) else np.nan
        got = row["venue_last_rho"]
        if not (pd.isna(expected) and pd.isna(got)):
            max_err = max(max_err, abs(float(expected) - float(got)))
    rows.append({
        "feature": "venue_last_rho", "max_rho_abs_err": round(max_err, 12),
        "max_count_err": 0, "passed": max_err < 1e-9,
    })

    report = pd.DataFrame(rows)
    return bool(report["passed"].all()), report


def test_suspicious_correlation(df: pd.DataFrame, features: list[str]) -> tuple[bool, pd.DataFrame]:
    numeric = [c for c in features if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    corr = pd.DataFrame({
        "feature": numeric,
        "r_with_target": [round(float(df[c].corr(df[TARGET])), 4) for c in numeric],
    })
    corr["suspicious"] = corr["r_with_target"].abs() > SUSPICIOUS_CORR
    return bool(~corr["suspicious"].any()), corr.sort_values("r_with_target", key=abs, ascending=False)


def test_r2_ceiling(df: pd.DataFrame, features: list[str]) -> tuple[bool, float]:
    """برازش سریع روی تقسیم **زمانی** — نه تصادفی، وگرنه خودِ آزمون نشتی می‌سازد."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import r2_score

    work = df.sort_values("date_gregorian")
    cut = int(0.8 * len(work))
    train, test = work.iloc[:cut], work.iloc[cut:]

    cols = [c for c in features if c in df.columns]
    X_train, X_test = _encode(train[cols]), _encode(test[cols])
    X_test = X_test.reindex(columns=X_train.columns, fill_value=np.nan)

    model = HistGradientBoostingRegressor(max_iter=120, random_state=42)
    model.fit(X_train, train[TARGET])
    r2 = float(r2_score(test[TARGET], model.predict(X_test)))
    return r2 < R2_CEILING, round(r2, 4)


def _encode(X: pd.DataFrame) -> pd.DataFrame:
    out = X.copy()
    for col in out.columns:
        if out[col].dtype == object or isinstance(out[col].dtype, pd.CategoricalDtype):
            out[col] = out[col].astype("category").cat.codes.replace(-1, np.nan)
        elif out[col].dtype == bool:
            out[col] = out[col].astype("int8")
    return out


def main() -> None:
    set_global_seed()
    df = pd.read_parquet(FEATURES_PATH)
    features = feature_set("full")

    pd.set_option("display.width", 200)

    ok_struct, struct_report = test_cutoff_invariant(df)
    print("=== ۱. آزمون ساختاری قاعده‌ی برش ===")
    print(struct_report.to_string(index=False))

    ok_corr, corr_report = test_suspicious_correlation(df, features)
    print("\n=== ۲. همبستگی با هدف (۱۰ تای برتر) ===")
    print(corr_report.head(10).to_string(index=False))

    ok_r2, r2 = test_r2_ceiling(df, features)
    print(f"\n=== ۳. سقف R² (تقسیم زمانی ۸۰/۲۰) ===\n  R² = {r2}  (سقف {R2_CEILING})")

    all_ok = ok_struct and ok_corr and ok_r2
    print(f"\n{'✅ ممیزی نشت: قبول' if all_ok else '❌ ممیزی نشت: رد'}")
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
