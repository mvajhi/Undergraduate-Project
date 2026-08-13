"""توابع کمکی همبستگی/هم‌خطی/ناهم‌واریانسی (بندهای ۴.۴ و ۴.۸ WBS) —
استفاده در notebooks/04_04_correlation_vif_heteroscedasticity.ipynb.

این ماژول جدید است (نه ویرایش فایل موجود) طبق دستور پروژه برای جلوگیری از تداخل
با سایر subagent هایی که هم‌زمان روی src/ کار می‌کنند.

توضیح Distance Correlation (Székely–Rizzo, 2007): چون کتابخانه‌ی `dcor` نصب نیست،
پیاده‌سازی برداری با فرمول استاندارد double-centering انجام شده است:
    dCov²(X,Y) = mean(A ⊙ B),   dVar²(X) = mean(A ⊙ A)
    dCor(X,Y)  = dCov(X,Y) / sqrt(dVar(X)·dVar(Y))
که A و B ماتریس فاصله‌ی دوبه‌دو (Euclidean) هستند پس از double-centering (کم‌کردن
میانگین سطر، میانگین ستون، و افزودن میانگین کل).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

from src.eda_lib.figio import save_fig  # noqa: F401 — پیاده‌سازی واحد، بازصادر می‌شود



def _dcor_from_arrays(x: np.ndarray, y: np.ndarray) -> float:
    """محاسبه‌ی خام dCor روی دو آرایه‌ی از پیش نمونه‌برداری‌شده (بدون subsample داخلی)."""
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    a = squareform(pdist(x, metric="euclidean"))
    b = squareform(pdist(y, metric="euclidean"))
    A = a - a.mean(axis=0, keepdims=True) - a.mean(axis=1, keepdims=True) + a.mean()
    B = b - b.mean(axis=0, keepdims=True) - b.mean(axis=1, keepdims=True) + b.mean()
    dcov2 = (A * B).mean()
    dvarx2 = (A * A).mean()
    dvary2 = (B * B).mean()
    if dvarx2 <= 0 or dvary2 <= 0:
        return 0.0
    dcov2 = max(dcov2, 0.0)
    return float(np.sqrt(dcov2) / np.sqrt(np.sqrt(dvarx2) * np.sqrt(dvary2)))


def distance_correlation(x: np.ndarray, y: np.ndarray, max_n: int = 2000, random_state: int = 42) -> float:
    """Distance Correlation بین دو بردار عددی (Székely–Rizzo).

    برای n بزرگ (>max_n)، برای سرعت (پیچیدگی O(n²)) روی یک نمونه‌ی تصادفی محاسبه می‌شود.
    مقدار در [0,1]؛ صفر فقط در صورت استقلال کامل رخ می‌دهد (برخلاف Pearson/Spearman
    که فقط رابطه‌ی خطی/یکنوا را می‌بینند).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n > max_n:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, size=max_n, replace=False)
        x, y = x[idx], y[idx]
    return _dcor_from_arrays(x, y)


def distance_correlation_matrix(
    df: pd.DataFrame, columns: list[str], max_n: int = 1500, random_state: int = 42
) -> tuple[pd.DataFrame, int]:
    """ماتریس Distance Correlation برای چند ستون هم‌زمان.

    برای سازگاری، همه‌ی ستون‌ها روی **یک** زیرنمونه‌ی تصادفی مشترک از ردیف‌ها محاسبه
    می‌شوند (نه subsample جدا برای هر جفت)، تا ساختار مشترک بین متغیرها حفظ شود.
    """
    data = df[columns].dropna()
    if len(data) > max_n:
        data = data.sample(n=max_n, random_state=random_state)
    cols = list(columns)
    n = len(cols)
    mat = np.eye(n)
    arrays = {c: data[c].to_numpy(dtype=float) for c in cols}
    for i in range(n):
        for j in range(i + 1, n):
            v = _dcor_from_arrays(arrays[cols[i]], arrays[cols[j]])
            mat[i, j] = mat[j, i] = v
    return pd.DataFrame(mat, index=cols, columns=cols), len(data)


def mutual_info_target(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> pd.Series:
    """امتیاز Mutual Information هر ستون X نسبت به هدف y (نزولی مرتب‌شده)."""
    from sklearn.feature_selection import mutual_info_regression

    mi = mutual_info_regression(X.to_numpy(), y.to_numpy(), random_state=random_state)
    return pd.Series(mi, index=X.columns, name="mutual_info").sort_values(ascending=False)


# ---------------------------------------------------------------------------
# ۴.۴.۲ — هم‌خطی (VIF + خوشه‌بندی)
# ---------------------------------------------------------------------------

def vif_table(X: pd.DataFrame) -> pd.DataFrame:
    """جدول VIF برای همه‌ی ستون‌های X (با افزودن عرض از مبدأ طبق رویه‌ی استاندارد)."""
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    Xc = sm.add_constant(X, has_constant="add")
    rows = []
    for i, col in enumerate(Xc.columns):
        if col == "const":
            continue
        v = variance_inflation_factor(Xc.to_numpy(), i)
        rows.append({"feature": col, "VIF": v})
    return pd.DataFrame(rows).sort_values("VIF", ascending=False).reset_index(drop=True)


def cluster_representatives(corr_df: pd.DataFrame, t: float = 0.3):
    """خوشه‌بندی Ward-average روی فاصله‌ی 1-|corr| و انتخاب یک نماینده از هر خوشه
    (نماینده = عضوی با بیشترین میانگین |corr| نسبت به بقیه‌ی اعضای همان خوشه).

    خروجی: (جدول feature->cluster، فهرست نماینده‌ها، linkage matrix Z برای دندروگرام)
    """
    from scipy.cluster.hierarchy import fcluster, linkage

    dist = 1 - corr_df.abs()
    dist_vals = dist.to_numpy()
    np.fill_diagonal(dist_vals, 0.0)
    dist_vals = np.clip(dist_vals, 0, None)
    condensed = squareform(dist_vals, checks=False)
    Z = linkage(condensed, method="average")
    clusters = fcluster(Z, t=t, criterion="distance")
    out = pd.DataFrame({"feature": corr_df.columns, "cluster": clusters})

    reps = []
    for c, grp in out.groupby("cluster"):
        members = grp["feature"].tolist()
        if len(members) == 1:
            reps.append(members[0])
        else:
            avg_abs_corr = corr_df.loc[members, members].abs().mean().sort_values(ascending=False)
            reps.append(avg_abs_corr.index[0])
    return out, reps, Z


# ---------------------------------------------------------------------------
# ۴.۴.۳ — همبستگی متقابل با تأخیر (CCF)
# ---------------------------------------------------------------------------

def manual_ccf(x: pd.Series, y: pd.Series, max_lag: int = 5) -> pd.DataFrame:
    """Cross-correlation دستی: corr(x_{t-lag}, y_t) برای lag=0..max_lag.

    x و y باید هم‌ایندکس (روی یک تقویم روزانه‌ی کامل، شامل روزهای بدون سرو با NaN)
    باشند؛ در هر lag، جفت‌های NaN با pairwise-deletion کنار گذاشته می‌شوند.
    lag>0 یعنی «آیا x چند روز قبل با y امروز مرتبط است» (اثر تأخیری x روی y).
    """
    rows = []
    for lag in range(0, max_lag + 1):
        x_shifted = x.shift(lag)
        aligned = pd.concat([x_shifted, y], axis=1, keys=["x", "y"]).dropna()
        if len(aligned) < 3:
            corr = np.nan
        else:
            corr = aligned["x"].corr(aligned["y"])
        rows.append({"lag": lag, "pearson_r": corr, "n_pairs": len(aligned)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# ۴.۸ — ناهم‌واریانسی و Empirical Bayes Shrinkage
# ---------------------------------------------------------------------------

def theoretical_binomial_variance(p_bar: float, res: np.ndarray) -> np.ndarray:
    """واریانس نظری نرخ تحت مدل دوجمله‌ای: Var(ρ) ≈ p̄(1-p̄)/Res."""
    res = np.asarray(res, dtype=float)
    return p_bar * (1 - p_bar) / res


def fit_beta_moments_overdispersion(no_recv: np.ndarray, res: np.ndarray) -> tuple[float, float]:
    """برازش Beta(α,β) با روش گشتاورها، با تصحیح واریانس نمونه‌گیری دوجمله‌ای شناخته‌شده.

    برازش ساده‌ی Beta مستقیم روی ρ مشاهده‌شده (مثلاً scipy.stats.beta.fit) واریانس
    نویز نمونه‌گیری (که خودش تابعی از Res است، دقیقاً موضوع بند ۴.۸) را با واریانس
    واقعی p_i بین سلول‌ها قاطی می‌کند. روش صحیح‌تر (Efron–Morris / Empirical Bayes
    کلاسیک): واریانس نمونه‌گیری مورد انتظار را از واریانس مشاهده‌شده‌ی ρ̂ کم می‌کنیم:

        Var_obs(ρ̂) ≈ Var_true(p) + E[p̄(1-p̄)/Res]
        Var_true(p) = Var_obs(ρ̂) - mean(p̄(1-p̄)/Res)
        S = α+β = p̄(1-p̄)/Var_true(p) - 1  ,  α = p̄·S  ,  β = (1-p̄)·S

    اگر Var_true(p) <= 0 شود (نویز نمونه‌گیری کل واریانس مشاهده را توضیح می‌دهد)،
    S به یک مقدار بزرگ محدود می‌شود (کوچک‌سازی شدید به سمت میانگین کل).
    """
    no_recv = np.asarray(no_recv, dtype=float)
    res = np.asarray(res, dtype=float)
    p_hat = no_recv / res
    p_bar = no_recv.sum() / res.sum()
    var_obs = np.var(p_hat, ddof=1)
    mean_binom_var = np.mean(p_bar * (1 - p_bar) / res)
    var_true = var_obs - mean_binom_var
    if var_true <= 0:
        S = 1000.0  # کوچک‌سازی شدید — تقریباً کل نویز مشاهده‌شده نمونه‌گیری است
    else:
        S = p_bar * (1 - p_bar) / var_true - 1
        S = max(S, 1e-3)
    alpha = p_bar * S
    beta = (1 - p_bar) * S
    return float(alpha), float(beta)


def fit_beta_simple(rho: np.ndarray) -> tuple[float, float]:
    """برازش MLE ساده‌ی Beta روی ρ مستقیماً (scipy.stats.beta.fit، بدون تصحیح Res).

    برای مقایسه با fit_beta_moments_overdispersion آورده شده — انتظار می‌رود S این
    روش کوچک‌تر باشد (کوچک‌سازی ضعیف‌تر) چون نویز نمونه‌گیری را جزو تنوع واقعی می‌بیند.
    """
    from scipy import stats

    rho = np.asarray(rho, dtype=float)
    n = len(rho)
    # تبدیل Smithson-Verkuilen برای فشردن نقاط مرزی دقیق ۰/۱ به داخل بازه‌ی باز
    rho_sv = (rho * (n - 1) + 0.5) / n
    a, b, _, _ = stats.beta.fit(rho_sv, floc=0, fscale=1)
    return float(a), float(b)


def empirical_bayes_shrink(no_recv: np.ndarray, res: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """نرخ کوچک‌سازی‌شده‌ی بیزی تجربی: ρ̃ = (NoRecv+α)/(Res+α+β)."""
    no_recv = np.asarray(no_recv, dtype=float)
    res = np.asarray(res, dtype=float)
    return (no_recv + alpha) / (res + alpha + beta)


def gini(weights: np.ndarray) -> float:
    """ضریب جینی یک بردار وزن نامنفی (۰=توزیع کاملاً برابر، نزدیک ۱=تمرکز شدید)."""
    w = np.sort(np.asarray(weights, dtype=float))
    n = len(w)
    if n == 0 or w.sum() == 0:
        return 0.0
    cum = np.cumsum(w)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def top_decile_weight_share(weights: np.ndarray, frac: float = 0.10) -> float:
    """سهم مجموع وزن که به‌دست پرحجم‌ترین `frac` رکوردها (بر اساس خودِ وزن) می‌افتد."""
    w = np.asarray(weights, dtype=float)
    n_top = max(1, int(np.ceil(frac * len(w))))
    w_sorted = np.sort(w)[::-1]
    return float(w_sorted[:n_top].sum() / w.sum())
