"""کمک‌کدهای بند ۴.۲ WBS (تحلیل گروهی و آزمون فرضیه) — دفترچه فرضیه‌های H1-H12.

این ماژول شامل: اندازه‌ی اثرهای ناپارامتری (Cliff's delta، rank-biserial،
eta-squared برای Kruskal-Wallis و Levene)، فاصله‌اطمینان bootstrap برای تفاوت
میانه‌ها، پیاده‌سازی دستی آزمون روند Mann-Kendall (چون ``pymannkendall`` نصب
نیست)، و ساخت فیچر ``days_since_same_food`` برای H5.

فایل‌های اصلی ``src/`` (``config.py``, ``viz_fa.py``, ...) دست‌نخورده مانده‌اند؛
طبق دستور کار، منطق کمکی این بند اینجا و فقط اینجا اضافه شده است.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# --------------------------------------------------------------------------
# اندازه‌ی اثر برای آزمون‌های ناپارامتری
# --------------------------------------------------------------------------


def cliffs_delta(x, y) -> float:
    """Cliff's delta برای دو نمونه‌ی مستقل x و y.

    delta = P(X > Y) - P(X < Y)، در بازه‌ی [-1, 1]. مقدار مثبت یعنی x
    سیستماتیک بزرگ‌تر از y است. پیاده‌سازی O(n log n) با رتبه‌بندی مشترک،
    بدون نیاز به کتابخانه‌ی خارجی.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return float("nan")

    all_vals = np.concatenate([x, y])
    order = np.argsort(all_vals, kind="mergesort")
    ranks = np.empty(len(all_vals))
    # رتبه‌ی میانگین برای مقادیر مساوی (هم‌طراز/tie-aware)
    sorted_vals = all_vals[order]
    ranks_sorted = np.arange(1, len(all_vals) + 1, dtype=float)
    i = 0
    while i < len(sorted_vals):
        j = i
        while j + 1 < len(sorted_vals) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks_sorted[i : j + 1] = ranks_sorted[i : j + 1].mean()
        i = j + 1
    ranks[order] = ranks_sorted

    rank_x_sum = ranks[:nx].sum()
    # U از رابطه‌ی رتبه‌ای معادل Mann-Whitney؛ delta = 2U/(nx*ny) - 1
    u_stat = rank_x_sum - nx * (nx + 1) / 2
    delta = 2 * u_stat / (nx * ny) - 1
    return float(delta)


def rank_biserial_from_u(u_stat: float, n1: int, n2: int) -> float:
    """همبستگی rank-biserial از آماره‌ی U من‌ویتنی: r = 1 - 2U/(n1*n2).

    از نظر عددی با cliff's delta معادل است (r == delta وقتی U متناظر با
    نمونه‌ی اول باشد)؛ اینجا برای وقتی جداگانه نگه داشته شده که فقط آماره‌ی
    U از ``scipy.stats.mannwhitneyu`` در دست است.
    """
    return float(1 - (2 * u_stat) / (n1 * n2))


def eta_squared_kruskal(h_stat: float, n_total: int, k_groups: int) -> float:
    """اندازه‌ی اثر eta-squared برای Kruskal-Wallis (تقریب رایج در ادبیات).

    eta^2_H = (H - k + 1) / (n - k)، بازه‌ی تفسیر مشابه ANOVA (کوچک ~0.01،
    متوسط ~0.06، بزرگ ~0.14).
    """
    if n_total <= k_groups:
        return float("nan")
    return float((h_stat - k_groups + 1) / (n_total - k_groups))


def eta_squared_levene(values: pd.Series, groups: pd.Series) -> float:
    """اندازه‌ی اثر eta-squared برای آزمون Levene.

    آزمون Levene خودش یک ANOVA یک‌طرفه روی قدرمطلق انحراف از میانه‌ی هر گروه
    است؛ این تابع همان ANOVA کمکی را می‌سازد و eta^2 = SS_between/SS_total
    را برمی‌گرداند (سهم واریانس ρ که با تفاوت گروه‌های Res توضیح داده می‌شود).
    """
    df = pd.DataFrame({"value": values, "group": groups}).dropna()
    centered = df.groupby("group")["value"].transform(lambda s: np.abs(s - s.median()))
    grand_mean = centered.mean()
    ss_between = df.assign(z=centered).groupby("group")["z"].apply(
        lambda z: len(z) * (z.mean() - grand_mean) ** 2
    ).sum()
    ss_total = ((centered - grand_mean) ** 2).sum()
    if ss_total == 0:
        return float("nan")
    return float(ss_between / ss_total)


# --------------------------------------------------------------------------
# فاصله‌اطمینان bootstrap برای تفاوت میانه‌ها
# --------------------------------------------------------------------------


def bootstrap_median_diff_ci(
    x,
    y,
    n_boot: int = 4000,
    alpha: float = 0.05,
    random_state: int | None = None,
) -> tuple[float, float, float]:
    """فاصله‌اطمینان bootstrap (روش percentile) برای median(x) - median(y).

    خروجی: (نقطه‌برآورد، حد پایین، حد بالا) با سطح اطمینان ``1 - alpha``.
    از ``scipy.stats.bootstrap`` استفاده می‌کند (نمونه‌گیری مستقل هر گروه).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(random_state)

    def _stat(a, b, axis=-1):
        return np.median(a, axis=axis) - np.median(b, axis=axis)

    res = stats.bootstrap(
        (x, y),
        _stat,
        n_resamples=n_boot,
        confidence_level=1 - alpha,
        method="percentile",
        random_state=rng,
        paired=False,
    )
    point = float(np.median(x) - np.median(y))
    return point, float(res.confidence_interval.low), float(res.confidence_interval.high)


# --------------------------------------------------------------------------
# آزمون روند Mann-Kendall (پیاده‌سازی دستی — pymannkendall نصب نیست)
# --------------------------------------------------------------------------


def mann_kendall(x) -> dict:
    """آزمون ناپارامتری روند Mann-Kendall، پیاده‌سازی دستی با فرمول S-statistic.

    x باید یک سری زمانی مرتب (بر اساس زمان) و بدون NaN باشد. برای n بزرگ از
    تقریب نرمال (با تصحیح tie) برای p-value استفاده می‌شود.

    خروجی: دیکشنری با کلیدهای s, var_s, z, p, tau, trend.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 4:
        return {"s": np.nan, "var_s": np.nan, "z": np.nan, "p": np.nan, "tau": np.nan, "trend": "insufficient_data"}

    s = 0
    for k in range(n - 1):
        s += np.sum(np.sign(x[k + 1 :] - x[k]))

    # تصحیح واریانس برای مقادیر تکراری (tie correction)
    _, counts = np.unique(x, return_counts=True)
    tie_term = np.sum(counts * (counts - 1) * (2 * counts + 5))
    var_s = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0

    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0

    p = 2 * (1 - stats.norm.cdf(abs(z)))
    tau = s / (0.5 * n * (n - 1))
    trend = "increasing" if (p < 0.05 and s > 0) else "decreasing" if (p < 0.05 and s < 0) else "no_trend"

    return {"s": int(s), "var_s": float(var_s), "z": float(z), "p": float(p), "tau": float(tau), "trend": trend}


# --------------------------------------------------------------------------
# فیچر H5: days_since_same_food
# --------------------------------------------------------------------------


def build_days_since_same_food(
    df: pd.DataFrame,
    restaurant_col: str = "RestaurantName",
    food_col: str = "FoodName",
    date_col: str = "date_gregorian",
) -> pd.Series:
    """برای هر (سلف, غذا)، فاصله‌ی روزی از آخرین سرو قبلی همان غذا در همان سلف.

    طبق تعریف دفترچه فرضیه‌ها (H5)، گروه‌بندی فقط روی (سلف, غذا) است — نه
    وعده — یعنی سرو یک غذا در ناهارِ یک روز، "سرو قبلی" برای شامِ روزهای بعد
    هم محسوب می‌شود. اولین سرو هر (سلف, غذا) مقدار NaN می‌گیرد (بدون سابقه).
    نتیجه به‌عنوان Series هم‌ترازشده با اندیس df بازگردانده می‌شود.
    """
    tmp = df[[restaurant_col, food_col, date_col]].copy()
    tmp["_date"] = pd.to_datetime(tmp[date_col])
    tmp["_orig_idx"] = df.index

    tmp = tmp.sort_values([restaurant_col, food_col, "_date"])
    grp = tmp.groupby([restaurant_col, food_col])["_date"]
    tmp["days_since_same_food"] = grp.diff().dt.days

    tmp = tmp.set_index("_orig_idx")
    return tmp["days_since_same_food"].reindex(df.index)


# --------------------------------------------------------------------------
# ابزار تصحیح چندگانگی
# --------------------------------------------------------------------------


def fdr_correct(p_values, method: str = "fdr_bh", alpha: float = 0.05) -> np.ndarray:
    """رَپِر نازک روی ``statsmodels.stats.multitest.multipletests`` که فقط
    آرایه‌ی p-value تصحیح‌شده را برمی‌گرداند (برای الحاق ساده به DataFrame)."""
    from statsmodels.stats.multitest import multipletests

    p_values = np.asarray(p_values, dtype=float)
    mask = ~np.isnan(p_values)
    corrected = np.full_like(p_values, np.nan)
    if mask.sum() > 0:
        _, corrected[mask], _, _ = multipletests(p_values[mask], alpha=alpha, method=method)
    return corrected
