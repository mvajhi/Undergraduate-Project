"""کمک‌کدهای بندهای ۴.۶ (خوشه‌بندی و کشف الگو) و ۴.۷ (توازن/پوشش/کفایت داده) WBS.

این ماژول جدید است (نه ویرایش فایل موجود) طبق دستور پروژه برای جلوگیری از تداخل
با سایر subagent هایی که هم‌زمان روی src/ کار می‌کنند. شامل:
- ساخت بردار ویژگی سلف/غذا برای خوشه‌بندی (بند ۴.۶.۱ و ۴.۶.۲)
- اسکن کیفیت خوشه‌بندی K-Means روی بازه‌ای از k (Elbow/Silhouette/Davies-Bouldin)
- ماتریس پوشش سلف×هفته و آمار طول سری زمانی (بند ۴.۷)

توابع ``save_fig`` و مشابه که قبلاً در ``src/eda_lib/univariate_helpers.py``
ساخته شده‌اند از همان‌جا import می‌شوند تا کد تکرار نشود.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score


# ---------------------------------------------------------------------------
# ۴.۶.۱ بردار ویژگی سلف‌ها
# ---------------------------------------------------------------------------


def _ols_trend_slope(dates: pd.Series, values: pd.Series) -> float:
    """شیب رگرسیون ساده‌ی OLS مقدار روی زمان (واحد: تغییر در روز، یعنی per day).

    اگر کمتر از ۳ نقطه یا واریانس زمانی صفر باشد NaN برمی‌گرداند.
    """
    d = pd.to_datetime(dates)
    day_idx = (d - d.min()).dt.days.to_numpy(dtype=float)
    y = np.asarray(values, dtype=float)
    mask = ~np.isnan(y)
    day_idx, y = day_idx[mask], y[mask]
    if len(y) < 3 or np.ptp(day_idx) == 0:
        return float("nan")
    return float(stats.linregress(day_idx, y).slope)


def restaurant_feature_matrix(
    df: pd.DataFrame,
    restaurant_col: str = "RestaurantName",
    type_col: str = "RestaurantType",
    meal_col: str = "Meal",
    rho_col: str = "rho",
    res_col: str = "Res",
    dow_col: str = "DayOfWeek",
    date_col: str = "date_gregorian",
) -> pd.DataFrame:
    """بردار ویژگی هر سلف برای خوشه‌بندی (بند ۴.۶.۱ WBS).

    ستون‌های خروجی (اندیس = نام سلف):
    - ``mean_rho``, ``std_rho``: میانگین/انحراف‌معیار نرخ عدم‌دریافت
    - ``dow_0`` .. ``dow_6``: میانگین rho هر روز هفته تقسیم‌بر میانگین کل سلف
      (پروفایل هفتگی نرمال‌شده؛ مقدار ۱٫۰ یعنی آن روز مثل میانگین است)
    - ``lunch_dinner_diff``: میانگین rho ناهار منهای میانگین rho شام؛ برای
      سلف‌هایی که فقط یک وعده سرو می‌کنند NaN است (بعداً با ۰ پر می‌شود چون
      نبود تفاوت تعریف‌نشده است، نه صفر واقعی — پرچم ``serves_both_meals``
      این حالت را جدا نگه می‌دارد)
    - ``serves_both_meals``: ۱ اگر سلف هم ناهار هم شام دارد، وگرنه ۰
    - ``log_res_mean``: میانگین log1p(Res) — حجم رزرو در مقیاس لگاریتمی
    - ``trend_slope``: شیب OLS ساده‌ی rho روی زمان (per day)
    - ``n_obs``, ``total_res``, ``restaurant_type``: فراداده (metadata)، نه
      بخشی از فیچرهای خوشه‌بندی
    """
    rows = []
    for name, g in df.groupby(restaurant_col):
        overall_mean = g[rho_col].mean()
        dow_means = g.groupby(dow_col)[rho_col].mean()
        dow_profile = {
            f"dow_{d}": (dow_means.get(d, np.nan) / overall_mean if overall_mean != 0 else np.nan)
            for d in range(7)
        }

        meals_present = g[meal_col].unique()
        serves_both = int({"lunch", "dinner"}.issubset(set(meals_present)))
        if serves_both:
            lunch_mean = g.loc[g[meal_col] == "lunch", rho_col].mean()
            dinner_mean = g.loc[g[meal_col] == "dinner", rho_col].mean()
            lunch_dinner_diff = lunch_mean - dinner_mean
        else:
            lunch_dinner_diff = np.nan

        row = {
            restaurant_col: name,
            "restaurant_type": g[type_col].iloc[0],
            "n_obs": len(g),
            "total_res": g[res_col].sum(),
            "mean_rho": overall_mean,
            "std_rho": g[rho_col].std(ddof=1) if len(g) > 1 else np.nan,
            **dow_profile,
            "lunch_dinner_diff": lunch_dinner_diff,
            "serves_both_meals": serves_both,
            "log_res_mean": np.log1p(g[res_col]).mean(),
            "trend_slope": _ols_trend_slope(g[date_col], g[rho_col]),
        }
        rows.append(row)

    out = pd.DataFrame(rows).set_index(restaurant_col)
    return out


# ---------------------------------------------------------------------------
# ۴.۶.۲ بردار ویژگی غذاها
# ---------------------------------------------------------------------------


def food_feature_matrix(
    df: pd.DataFrame,
    food_col: str = "FoodName",
    food_type_col: str = "FoodType",
    restaurant_col: str = "RestaurantName",
    meal_col: str = "Meal",
    rho_col: str = "rho",
    res_col: str = "Res",
) -> pd.DataFrame:
    """بردار ویژگی هر غذا برای خوشه‌بندی (بند ۴.۶.۲ WBS).

    ``relative_volume_mean`` حجم رزرو غذا را نسبت به میانگین (سلف, وعده)ی
    میزبانش نرمال می‌کند (مقدار ۱٫۰ یعنی حجم رزرو معمولی برای آن زمینه)، تا
    اندازه‌ی ذاتی سلف باعث تورشدگی مقایسه‌ی محبوبیت غذاها نشود.

    **هشدار نشت (leakage):** این ماتریس از **کل بازه‌ی ۵ ماهه** ساخته شده و
    فقط برای EDA/کشف الگو مجاز است. اگر «شاخص محبوبیت غذا» به فیچر مدل در
    فاز ۵ تبدیل شود، باید برای هر رکورد فقط از رکوردهای **قبل از لحظه‌ی برش
    اطلاعاتی همان رکورد** بازمحاسبه شود، وگرنه نشت زمانی (future leakage) رخ
    می‌دهد.
    """
    context_mean = df.groupby([restaurant_col, meal_col])[res_col].transform("mean")
    rel_volume = df[res_col] / context_mean

    tmp = df.assign(_rel_volume=rel_volume)
    agg = tmp.groupby(food_col).agg(
        food_type=(food_type_col, "first"),
        n_served=(rho_col, "size"),
        mean_rho=(rho_col, "mean"),
        std_rho=(rho_col, "std"),
        relative_volume_mean=("_rel_volume", "mean"),
        total_res=(res_col, "sum"),
    )
    agg["log_n_served"] = np.log1p(agg["n_served"])
    return agg


# ---------------------------------------------------------------------------
# اسکن کیفیت خوشه‌بندی K-Means
# ---------------------------------------------------------------------------


def cluster_quality_scan(
    X: np.ndarray, k_range: range, random_state: int = 42, n_init: int = 20
) -> pd.DataFrame:
    """K-Means را برای هر k در ``k_range`` اجرا می‌کند و inertia/silhouette/
    Davies-Bouldin را برمی‌گرداند (برای نمودار Elbow و انتخاب k بهینه).

    k باید حداقل ۲ و حداکثر n_samples-1 باشد؛ مقادیر خارج از این بازه نادیده
    گرفته می‌شوند.
    """
    n = X.shape[0]
    rows = []
    for k in k_range:
        if k < 2 or k > n - 1:
            continue
        km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        labels = km.fit_predict(X)
        rows.append(
            {
                "k": k,
                "inertia": km.inertia_,
                "silhouette": silhouette_score(X, labels),
                "davies_bouldin": davies_bouldin_score(X, labels),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# ۴.۷ پوشش و کفایت داده
# ---------------------------------------------------------------------------


def relative_week_index(dates: pd.Series) -> pd.Series:
    """اندیس هفته‌ی نسبی (۰ = هفته‌ی اول بازه) از روی ``date_gregorian``.

    به‌جای ``isocalendar().week`` خام استفاده می‌شود چون بازه‌ی داده (آذر
    ۱۴۰۲ تا خرداد ۱۴۰۳) از مرز سال میلادی (دسامبر→ژانویه) عبور می‌کند و شماره
    هفته‌ی ISO در آن مرز ریست می‌شود؛ اندیس نسبی این مشکل را ندارد و پیوسته
    می‌ماند.
    """
    d = pd.to_datetime(dates)
    return ((d - d.min()).dt.days // 7).astype(int)


def coverage_matrix(
    df: pd.DataFrame,
    group_col: str = "RestaurantName",
    date_col: str = "date_gregorian",
) -> pd.DataFrame:
    """جدول محوری تعداد مشاهدات به ازای ``group_col`` × هفته‌ی نسبی (برای heatmap پوشش)."""
    tmp = df.assign(week_index=relative_week_index(df[date_col]))
    return tmp.pivot_table(
        index=group_col, columns="week_index", values=date_col, aggfunc="count", fill_value=0
    )


def series_length_distribution(
    df: pd.DataFrame,
    group_cols: list[str] = ["Meal", "RestaurantName", "FoodName"],
) -> pd.DataFrame:
    """تعداد مشاهدات هر سری زمانی یکتای ``group_cols`` (پیش‌فرض (m,r,f)) — برای
    بررسی اینکه چند سری کمتر از یک آستانه (مثلاً ۳۰) نقطه دارند."""
    return df.groupby(group_cols).size().reset_index(name="n_obs").sort_values("n_obs")
