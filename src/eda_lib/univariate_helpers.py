"""توابع کمکی تحلیل تک‌متغیره (بند ۴.۱ WBS) — استفاده در notebooks/04_01_univariate.ipynb.

این ماژول جدید است (نه ویرایش فایل موجود) طبق دستور پروژه برای جلوگیری از تداخل
با سایر subagent هایی که هم‌زمان روی src/ کار می‌کنند.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def descriptive_stats(x: np.ndarray) -> dict:
    """میانگین، میانه، مد، انحراف معیار، IQR، چولگی، کشیدگی و صدک‌های استاندارد."""
    x = np.asarray(x, dtype=float)
    mode_res = stats.mode(x, keepdims=False)
    q1, q3 = np.percentile(x, [25, 75])
    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    return {
        "n": len(x),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "mode": float(mode_res.mode),
        "mode_count": int(mode_res.count),
        "std": float(np.std(x, ddof=1)),
        "iqr": float(q3 - q1),
        "q1": float(q1),
        "q3": float(q3),
        "skewness": float(stats.skew(x)),
        "kurtosis_excess": float(stats.kurtosis(x)),
        "percentiles": {p: float(np.percentile(x, p)) for p in pcts},
    }


def zero_one_inflation(x: np.ndarray) -> dict:
    """درصد رکوردهای دقیقاً صفر و دقیقاً یک (تورم مرزی برای متغیر [0,1])."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    return {
        "n_zero": int((x == 0).sum()),
        "pct_zero": float((x == 0).mean() * 100),
        "n_one": int((x == 1).sum()),
        "pct_one": float((x == 1).mean() * 100),
        "n_interior": int(((x > 0) & (x < 1)).sum()),
    }


def sarle_bimodality_coefficient(x: np.ndarray) -> float:
    """ضریب دومُدی بودن Sarle. مقدار > 0.555 (مرجع توزیع یکنواخت) نشانه‌ی احتمال دومُدی بودن است.

    مرجع: SAS Institute (1990); Pfister et al. (2013) کاربرد در روانشناسی/رفتار.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    skew = stats.skew(x)
    kurt_pearson = stats.kurtosis(x, fisher=False)
    return float((skew**2 + 1) / (kurt_pearson + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))))


def kde_peak_count(x: np.ndarray, grid_points: int = 2000, bw_method: str = "scott"):
    """تعداد و مکان قله‌های KDE (برای بررسی چندمُدی بودن اکتشافی، مکمل ضریب Sarle)."""
    from scipy.signal import find_peaks

    x = np.asarray(x, dtype=float)
    kde = stats.gaussian_kde(x, bw_method=bw_method)
    xs = np.linspace(x.min(), x.max(), grid_points)
    ys = kde(xs)
    peak_idx, _ = find_peaks(ys)
    return xs[peak_idx], ys[peak_idx]


def ks_against(data: np.ndarray, dist, params: tuple):
    """آزمون KS در برابر یک توزیع فریز‌شده (پارامترها از قبل برازش شده‌اند)."""
    frozen = dist(*params)
    return stats.kstest(data, frozen.cdf)


def fit_candidate_distributions(rho: np.ndarray, interior_only: bool = False) -> pd.DataFrame:
    """برازش MLE چهار توزیع کاندید (Normal, Beta, Gamma, Log-normal) روی rho و آزمون KS.

    چون rho در [0,1] است و مقادیر دقیقاً ۰/۱ برای Beta/Gamma/Lognormal مشکل مرزی ایجاد
    می‌کنند (چگالی MLE در مرز نامعین)، دو حالت پشتیبانی می‌شود:
    - interior_only=False (پیش‌فرض): تبدیل Smithson-Verkuilen برای Beta (فشردن به سمت
      داخل بازه)، و epsilon-shift برای Gamma/Lognormal، روی کل داده.
    - interior_only=True: فقط مشاهدات (0,1) باز نگه داشته می‌شود (۰ و ۱ دقیق کنار
      گذاشته می‌شوند) — برای مقایسه‌ی «شکل» توزیع جدا از تورم مرزی.
    """
    rho = np.asarray(rho, dtype=float)
    n = len(rho)
    rows = []

    if interior_only:
        data = rho[(rho > 0) & (rho < 1)]
        note = f"فقط داخل بازه باز (۰,۱) — {n - len(data)} مشاهده‌ی مرزی کنار گذاشته شد"
    else:
        data = rho
        note = "کل داده؛ Beta با تبدیل Smithson-Verkuilen، Gamma/Lognormal با epsilon-shift=1e-6"

    # Normal
    mean_, std_ = np.mean(data), np.std(data, ddof=1)
    ks_n = ks_against(data, stats.norm, (mean_, std_))
    rows.append({"dist": "Normal", "params": f"mu={mean_:.4f}, sigma={std_:.4f}",
                 "ks_stat": ks_n.statistic, "ks_pvalue": ks_n.pvalue, "note": note})

    # Beta
    if interior_only:
        a, b, loc, scale = stats.beta.fit(data, floc=0, fscale=1)
        ks_b = ks_against(data, stats.beta, (a, b, 0, 1))
    else:
        data_sv = (data * (n - 1) + 0.5) / n
        a, b, loc, scale = stats.beta.fit(data_sv, floc=0, fscale=1)
        ks_b = ks_against(data_sv, stats.beta, (a, b, 0, 1))
    rows.append({"dist": "Beta", "params": f"a={a:.4f}, b={b:.4f}",
                 "ks_stat": ks_b.statistic, "ks_pvalue": ks_b.pvalue, "note": note})

    # Gamma
    data_g = data if interior_only else data + 1e-6
    ag, locg, scaleg = stats.gamma.fit(data_g, floc=0)
    ks_g = ks_against(data_g, stats.gamma, (ag, locg, scaleg))
    rows.append({"dist": "Gamma", "params": f"a={ag:.4f}, scale={scaleg:.4f}",
                 "ks_stat": ks_g.statistic, "ks_pvalue": ks_g.pvalue, "note": note})

    # Log-normal
    data_l = data if interior_only else data + 1e-6
    s, locl, scalel = stats.lognorm.fit(data_l, floc=0)
    ks_l = ks_against(data_l, stats.lognorm, (s, locl, scalel))
    rows.append({"dist": "Log-normal", "params": f"s={s:.4f}, scale={scalel:.4f}",
                 "ks_stat": ks_l.statistic, "ks_pvalue": ks_l.pvalue, "note": note})

    return pd.DataFrame(rows)


def compare_transforms(x: np.ndarray) -> pd.DataFrame:
    """مقایسه‌ی AD-statistic (در برابر Normal) برای raw / log1p / Box-Cox / Yeo-Johnson.

    Box-Cox فقط وقتی x کاملاً مثبت باشد قابل‌اجراست (وگرنه ردیف N/A برمی‌گردد).
    AD-statistic پایین‌تر یعنی نزدیک‌تر به نرمال.
    """
    x = np.asarray(x, dtype=float)
    rows = []

    ad_raw = stats.anderson(x, dist="norm")
    rows.append({"transform": "raw", "lambda": None, "skewness": stats.skew(x),
                 "ad_stat": ad_raw.statistic})

    log1p_x = np.log1p(x)
    ad_log = stats.anderson(log1p_x, dist="norm")
    rows.append({"transform": "log1p", "lambda": None, "skewness": stats.skew(log1p_x),
                 "ad_stat": ad_log.statistic})

    if x.min() > 0:
        bc_x, lam_bc = stats.boxcox(x)
        ad_bc = stats.anderson(bc_x, dist="norm")
        rows.append({"transform": "Box-Cox", "lambda": lam_bc, "skewness": stats.skew(bc_x),
                     "ad_stat": ad_bc.statistic})
    else:
        rows.append({"transform": "Box-Cox", "lambda": None, "skewness": None,
                     "ad_stat": None})

    yj_x, lam_yj = stats.yeojohnson(x)
    ad_yj = stats.anderson(yj_x, dist="norm")
    rows.append({"transform": "Yeo-Johnson", "lambda": lam_yj, "skewness": stats.skew(yj_x),
                 "ad_stat": ad_yj.statistic})

    return pd.DataFrame(rows)


def save_fig(fig, name: str, figures_dir: Path | str, dpi: int = 150) -> Path:
    """ذخیره‌ی شکل matplotlib با نام یکتا (پیشوند بند WBS) در reports/figures/."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / name
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    return out_path
