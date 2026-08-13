"""توابع کمکی کاوش اکتشافی سطح فرد (بند ۴.۱۳ WBS) — استفاده در notebooks/04_08_individual_level_model_b.ipynb.

این ماژول جدید است (نه ویرایش فایل موجود) طبق دستور پروژه برای جلوگیری از تداخل
با سایر subagent هایی که هم‌زمان روی src/ کار می‌کنند.

نکته حافظه: person_reservation_fact_v1.csv خام ۶۸۴ مگابایت است؛ `load_fact_reduced`
همیشه با `usecols`/`dtype` محدود صدا زده می‌شود (بنچمارک: با ۶ ستون لازم، بار کامل فایل
~۳۰-۴۰ مگابایت حافظه و چند ثانیه زمان می‌گیرد — نیازی به chunksize نیست).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pingouin as pg
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# لود سبک فایل فردی
# ---------------------------------------------------------------------------

FACT_USECOLS = ["PersonId", "date_gregorian", "Meal", "restaurant_canonical", "Count", "dont_receive"]
FACT_DTYPES = {
    "PersonId": "int32",
    "Meal": "category",
    "restaurant_canonical": "category",
    "Count": "int16",
    "dont_receive": "bool",
}


def load_fact_reduced(
    path: str | Path,
    usecols: list[str] | None = None,
    dtype: dict | None = None,
) -> pd.DataFrame:
    """لود person_reservation_fact_v1.csv فقط با ستون‌های لازم + dtype فشرده (بند ۴.۱۳ WBS)."""
    usecols = usecols if usecols is not None else FACT_USECOLS
    dtype = dtype if dtype is not None else {k: v for k, v in FACT_DTYPES.items() if k in usecols}
    return pd.read_csv(path, usecols=usecols, dtype=dtype, parse_dates=["date_gregorian"])


# ---------------------------------------------------------------------------
# ۴.۱۳.۱ — توزیع تاریخچه فردی
# ---------------------------------------------------------------------------


def person_history_stats(fact: pd.DataFrame) -> pd.DataFrame:
    """به ازای هر PersonId: تعداد رزرو قبلی (سطح ردیف غذا، `groupby('PersonId').size()`
    طبق روش صریح WBS 4.13.1) و تعداد روز یکتای حضور در بازه‌ی داده."""
    n_res = fact.groupby("PersonId").size().rename("n_reservations")
    n_days = fact.groupby("PersonId")["date_gregorian"].nunique().rename("n_unique_days")
    return pd.concat([n_res, n_days], axis=1).reset_index()


# ---------------------------------------------------------------------------
# ۴.۱۳.۲ — H15: فیچر تاریخچه leakage-safe + مقایسه AUC
# ---------------------------------------------------------------------------


def expanding_person_rate(fact_sorted_by_date: pd.DataFrame, target_col: str = "dont_receive") -> pd.Series:
    """فیچر leakage-safe `person_expanding_norecv_rate`: نرخ عدم‌دریافت فرد **پیش از** رزرو
    جاری (`shift(1).expanding().mean()` روی تاریخچه‌ی مرتب‌شده‌ی زمانی هر فرد، طبق H15 WBS).

    ورودی باید از قبل با `sort_values('date_gregorian')` مرتب شده باشد. اولین رزرو هر فرد
    همیشه NaN می‌شود (هیچ تاریخچه‌ای پیش از آن نیست) — کاربر باید صریحاً fillna کند.
    """
    out = fact_sorted_by_date.groupby("PersonId")[target_col].apply(lambda s: s.shift(1).expanding().mean())
    return out.reset_index(level=0, drop=True)


def freq_encode_fit(series: pd.Series) -> pd.Series:
    """نگاشت فراوانی نسبی هر مقدار دسته‌ای — محاسبه‌شده فقط روی داده train تا نشتی توزیع
    test در فیچر جمعیتی رخ ندهد."""
    return series.value_counts(normalize=True)


def freq_encode_apply(series: pd.Series, mapping: pd.Series, default: float = 0.0) -> pd.Series:
    """اعمال نگاشت فراوانی fit‌شده روی train؛ مقادیر دیده‌نشده (فقط در test) → `default`."""
    return series.map(mapping).fillna(default)


def time_split(df: pd.DataFrame, date_col: str, train_frac: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """جداسازی ساده train/test بر مبنای چارک زمانی (نه k-fold تصادفی، چون داده سری زمانی است)."""
    cutoff = df[date_col].quantile(train_frac)
    train = df[df[date_col] <= cutoff].copy()
    test = df[df[date_col] > cutoff].copy()
    return train, test, cutoff


def bootstrap_auc_diff(
    y_true: np.ndarray,
    p_base: np.ndarray,
    p_extended: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """فاصله اطمینان بوت‌استرپ (۹۵٪) برای تفاوت AUC (extended - base) روی همان مجموعه آزمون
    (resample هم‌زمان y_true/دو بردار احتمال با اندیس مشترک، طبق قاعده H15)."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    p_base = np.asarray(p_base)
    p_extended = np.asarray(p_extended)
    n = len(y_true)
    diffs = np.empty(n_boot)
    idx_all = np.arange(n)
    for i in range(n_boot):
        idx = rng.choice(idx_all, size=n, replace=True)
        yt = y_true[idx]
        if yt.sum() == 0 or yt.sum() == len(yt):
            diffs[i] = np.nan
            continue
        diffs[i] = roc_auc_score(yt, p_extended[idx]) - roc_auc_score(yt, p_base[idx])
    diffs = diffs[~np.isnan(diffs)]
    return {
        "n_boot_valid": int(len(diffs)),
        "mean_diff": float(np.mean(diffs)),
        "ci_low_2.5": float(np.percentile(diffs, 2.5)),
        "ci_high_97.5": float(np.percentile(diffs, 97.5)),
        "pct_boot_gt_zero": float((diffs > 0).mean()),
    }


# ---------------------------------------------------------------------------
# ۴.۱۳.۲/۴.۱۳.۳ — آزمون‌ها و اندازه اثر
# ---------------------------------------------------------------------------


def mannwhitney_effect(x: np.ndarray, y: np.ndarray, x_name: str = "A", y_name: str = "B") -> dict:
    """آزمون Mann-Whitney U + اندازه اثر (rank-biserial correlation از pingouin) بین دو گروه مستقل."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    res = pg.mwu(x, y)
    return {
        f"n_{x_name}": int(len(x)),
        f"n_{y_name}": int(len(y)),
        f"median_{x_name}": float(np.median(x)),
        f"median_{y_name}": float(np.median(y)),
        "U": float(res["U_val"].iloc[0]),
        "p": float(res["p_val"].iloc[0]),
        "RBC": float(res["RBC"].iloc[0]),
        "CLES": float(res["CLES"].iloc[0]),
    }


# ---------------------------------------------------------------------------
# ۴.۱۳.۳ — منحنی لورنز / نابرابری عدم‌دریافت بین افراد
# ---------------------------------------------------------------------------


def lorenz_curve(counts: pd.Series) -> tuple[np.ndarray, np.ndarray, float]:
    """منحنی لورنز + ضریب جینی برای یک سری شمارشی ناموزون بین افراد (مثلاً تعداد no-recv
    هر فرد). برمی‌گرداند: (کسر تجمعی جمعیت مرتب‌شده صعودی، کسر تجمعی مقدار، ضریب جینی)."""
    x = np.sort(np.asarray(counts, dtype=float))
    n = len(x)
    total = x.sum()
    cum_x = np.cumsum(x)
    cum_frac = np.concatenate([[0.0], cum_x / total]) if total > 0 else np.zeros(n + 1)
    pop_frac = np.linspace(0, 1, n + 1)
    gini = 1 - 2 * np.trapezoid(cum_frac, pop_frac)
    return pop_frac, cum_frac, float(gini)


def pareto_share(counts: pd.Series, top_person_fracs: list[float]) -> pd.DataFrame:
    """جدول Pareto: به ازای هر کسر جمعیت (مرتب‌شده نزولی بر اساس شمارش)، چند درصد از کل
    مقدار را آن جمعیت تولید می‌کند."""
    sorted_desc = np.sort(np.asarray(counts, dtype=float))[::-1]
    n = len(sorted_desc)
    total = sorted_desc.sum()
    cum = np.cumsum(sorted_desc)
    rows = []
    for frac in top_person_fracs:
        k = max(1, int(round(frac * n)))
        rows.append({"pct_people": frac * 100, "n_people": k, "pct_of_total_norecv": float(cum[k - 1] / total * 100)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# ۴.۱۳.۴ — سازگاری بین دو منبع داده
# ---------------------------------------------------------------------------


def restaurant_daily_agg_from_fact(fact: pd.DataFrame) -> pd.DataFrame:
    """تجمیع $Res_{d,m,r}$ از فایل فردی (فقط ناهار/شام، سازگار با گرانولاریت dataset_v1)."""
    sub = fact[fact["Meal"].isin(["lunch", "dinner"])]
    return (
        sub.groupby(["date_gregorian", "Meal", "restaurant_canonical"], observed=True)["Count"]
        .sum()
        .reset_index()
        .rename(columns={"restaurant_canonical": "RestaurantName", "Count": "Res_individual"})
    )


def restaurant_daily_agg_from_dataset_v1(ds1: pd.DataFrame) -> pd.DataFrame:
    """تجمیع $Res_{d,m,r}$ از dataset_v1.csv — جمع روی FoodName چون واحد مقایسه‌ی این بند
    (طبق بند ۴.۱۳.۴ WBS) سطح رستوران است، نه رستوران×غذا."""
    return (
        ds1.groupby(["date_gregorian", "Meal", "RestaurantName"])["Res"]
        .sum()
        .reset_index()
        .rename(columns={"Res": "Res_aggregate"})
    )
