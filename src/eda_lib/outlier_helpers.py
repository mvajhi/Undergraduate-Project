"""کمک‌کدهای بند ۴.۵ (داده پرت) و ۴.۹ (تحلیل تعامل) WBS.

این ماژول شامل: پیاده‌سازی چهار روش تشخیص پرت روی ``rho`` (IQR، Z-score،
Modified Z-score مبتنی بر MAD، و دو روش چندمتغیره Isolation Forest/LOF)،
ابزار مقایسه‌ی هم‌پوشانی این روش‌ها (ماتریس Jaccard + شمارش اجماع)، و توابع
join دستی به ``events.csv``/``calendar_tehran.csv`` برای تحلیل ریشه‌ای پرت‌ها
(بند ۴.۵.۲). بخش پایانی، ابزار کمکی heatmap دوبعدی و اجرای ANOVA دوطرفه با
جمله‌ی تعامل برای بند ۴.۹ را فراهم می‌کند.

فایل‌های اصلی ``src/`` (``config.py``, ``viz_fa.py``, ...) دست‌نخورده مانده‌اند؛
طبق دستور کار پروژه برای اجرای موازی subagent ها، منطق کمکی این دو بند اینجا
و فقط اینجا اضافه شده است.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# ۴.۵.۱ روش‌های تک‌متغیره
# ---------------------------------------------------------------------------


def iqr_outlier_mask(x: pd.Series, k: float = 1.5) -> tuple[pd.Series, tuple[float, float]]:
    """پرت‌های IQR (ضریب پیش‌فرض ۱.۵×). خروجی: (mask بولی, (کران پایین, کران بالا))."""
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    return (x < lower) | (x > upper), (lower, upper)


def zscore_outlier_mask(x: pd.Series, threshold: float = 3.0) -> tuple[pd.Series, pd.Series]:
    """پرت‌های Z-score (میانگین/انحراف‌معیار کلاسیک، حساس به خود پرت‌ها). خروجی: (mask, z)."""
    z = (x - x.mean()) / x.std(ddof=1)
    return z.abs() > threshold, z


def modified_zscore(x: pd.Series) -> pd.Series:
    """Modified Z-score مبتنی بر MAD (Iglewicz & Hoaglin, 1993): 0.6745*(x-median)/MAD.

    مقاوم‌تر از Z-score کلاسیک چون میانه/MAD کمتر تحت‌تأثیر خود پرت‌ها قرار می‌گیرند.
    اگر MAD=0 (بیش از نیمی از داده مقدار یکسان دارند)، به‌جای تقسیم‌بر‌صفر مقدار inf
    برمی‌گرداند و هشدار در docstring صریح است.
    """
    median = x.median()
    mad = (x - median).abs().median()
    if mad == 0:
        return pd.Series(np.where(x == median, 0.0, np.inf), index=x.index)
    return 0.6745 * (x - median) / mad


def modified_zscore_outlier_mask(x: pd.Series, threshold: float = 3.5) -> tuple[pd.Series, pd.Series]:
    """پرت‌های Modified Z (آستانه‌ی استاندارد ۳.۵ طبق Iglewicz & Hoaglin). خروجی: (mask, modz)."""
    modz = modified_zscore(x)
    return modz.abs() > threshold, modz


# ---------------------------------------------------------------------------
# ۴.۵.۱ روش‌های چندمتغیره
# ---------------------------------------------------------------------------


def isolation_forest_mask(
    features: pd.DataFrame, contamination: float = 0.03, random_state: int = 42, n_estimators: int = 200
) -> np.ndarray:
    """پرت‌های چندمتغیره با Isolation Forest. ``contamination`` سهم تقریبی پرت‌ها را کنترل می‌کند
    (نه یک آستانه‌ی دقیق آماری) — برای مقایسه‌ی نسبی بین روش‌ها انتخاب شده، نه ادعای دقت مطلق.
    """
    from sklearn.ensemble import IsolationForest

    model = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=random_state)
    pred = model.fit_predict(features)
    return pred == -1


def lof_mask(features: pd.DataFrame, n_neighbors: int = 20, contamination: float = 0.03) -> np.ndarray:
    """پرت‌های محلی با Local Outlier Factor — نسبت چگالی هر نقطه به همسایه‌هایش، نه به کل داده.
    برخلاف Isolation Forest که پرت سراسری می‌گیرد، LOF می‌تواند نقطه‌ای را که در خوشه‌ی محلی
    خودش (مثلاً یک سلف با حجم رزرو مشابه) نامتعارف است ولی نسبت به کل داده نامتعارف نیست، بگیرد.
    """
    from sklearn.neighbors import LocalOutlierFactor

    model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    pred = model.fit_predict(features)
    return pred == -1


# ---------------------------------------------------------------------------
# مقایسه‌ی هم‌پوشانی روش‌ها
# ---------------------------------------------------------------------------


def build_outlier_flag_table(masks: dict[str, np.ndarray], index: pd.Index) -> pd.DataFrame:
    """جدول بولی یک ستون به‌ازای هر روش + ستون ``consensus`` (تعداد روش‌هایی که آن ردیف را
    پرت تشخیص داده‌اند). ورودی مناسب برای مرتب‌سازی/انتخاب پرت‌های برجسته در بند ۴.۵.۲.
    """
    flags = pd.DataFrame({name: np.asarray(mask, dtype=bool) for name, mask in masks.items()}, index=index)
    flags["consensus"] = flags.sum(axis=1)
    return flags


def pairwise_jaccard(flags: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    """ماتریس Jaccard دوبه‌دوی بین ستون‌های بولی ``methods`` — هم‌پوشانی نسبی هر جفت روش
    (۱.۰ = دقیقاً همان ردیف‌ها را پرت می‌دانند، ۰.۰ = هیچ اشتراکی ندارند).
    """
    mat = pd.DataFrame(index=methods, columns=methods, dtype=float)
    for a, b in itertools.product(methods, methods):
        va, vb = flags[a].values, flags[b].values
        union = (va | vb).sum()
        mat.loc[a, b] = (va & vb).sum() / union if union > 0 else 0.0
    return mat


# ---------------------------------------------------------------------------
# ۴.۵.۲ تحلیل ریشه‌ای — join به events.csv و calendar_tehran.csv
# ---------------------------------------------------------------------------


def nearest_event(date: pd.Timestamp, events: pd.DataFrame, window_days: int = 5) -> str:
    """توصیف رویدادهای مرتبط با ``date``: اول رویدادهایی که بازه‌شان شامل تاریخ است
    (``date_start <= date <= date_end``)، وگرنه نزدیک‌ترین رویداد در ``window_days`` روز
    قبل/بعد با فاصله‌ی روز صریح (مثلاً «۱ روز پیش از شروع ramadan_1445»). این تابع دقیقاً
    همان تله‌ای را می‌پوشاند که فقط containment را چک می‌کند: یک رویداد ساختاری (مثل شروع
    رمضان) می‌تواند روز *قبل* از تاریخ شروعش را هم تحت تأثیر قرار دهد.
    """
    contained = events[(events["date_start"] <= date) & (events["date_end"] >= date)]
    if not contained.empty:
        return "; ".join(f"{r.event_id} ({r.event_type})" for r in contained.itertuples())

    candidates = []
    for r in events.itertuples():
        if r.date_start > date:
            gap = (r.date_start - date).days
            if gap <= window_days:
                candidates.append((gap, f"{gap} روز پیش از شروع {r.event_id} ({r.event_type})"))
        elif r.date_end < date:
            gap = (date - r.date_end).days
            if gap <= window_days:
                candidates.append((gap, f"{gap} روز پس از پایان {r.event_id} ({r.event_type})"))
    if not candidates:
        return "بدون رویداد ثبت‌شده در شعاع ±{}روز".format(window_days)
    candidates.sort(key=lambda t: t[0])
    return "; ".join(text for _, text in candidates[:2])


def calendar_flags(date: pd.Timestamp, calendar: pd.DataFrame) -> str:
    """خلاصه‌ی متنی فلگ‌های مرتبط تقویم روزانه برای یک تاریخ مشخص (تعطیلات/امتحانات/پل/نوروز)."""
    row = calendar.loc[calendar["date_gregorian"] == date]
    if row.empty:
        return "بدون رکورد تقویم"
    r = row.iloc[0]
    flags = []
    if bool(r.get("is_holiday_national", False)):
        flags.append(f"تعطیل رسمی: {r.get('holiday_name', '')}")
    if bool(r.get("is_friday", False)):
        flags.append("جمعه")
    if bool(r.get("is_exam_period", False)):
        flags.append("ایام امتحانات")
    if bool(r.get("is_nowruz_block", False)):
        flags.append("بلوک نوروز")
    if bool(r.get("is_inter_semester_break", False)):
        flags.append("تعطیلات بین‌ترم")
    if bool(r.get("is_bridge_day", False)):
        flags.append("روز پل")
    if bool(r.get("is_day_before_holiday", False)):
        flags.append("روز قبل از تعطیلی")
    if bool(r.get("is_day_after_holiday", False)):
        flags.append("روز بعد از تعطیلی")
    return "؛ ".join(flags) if flags else "بدون فلگ خاص"


def same_day_context(
    row: pd.Series, df: pd.DataFrame, date_col: str = "date_gregorian", meal_col: str = "Meal"
) -> dict:
    """میانه‌ی rho همان روز/همان وعده (بدون خود ردیف) — برای تشخیص «آیا کل روز غیرعادی
    بود» (اثر سراسری/تقویمی) در برابر «فقط همین سلف غیرعادی بود» (تعطیلی/خطای محلی).
    """
    same = df[(df[date_col] == row[date_col]) & (df[meal_col] == row[meal_col])]
    other = same.drop(index=row.name, errors="ignore")
    return {
        "n_same_day_meal": int(len(same)),
        "median_rho_others": float(other["rho"].median()) if len(other) else float("nan"),
    }


# ---------------------------------------------------------------------------
# ۴.۹ تحلیل تعامل
# ---------------------------------------------------------------------------


def two_way_anova_interaction(df: pd.DataFrame, target: str, factor_a: str, factor_b: str):
    """ANOVA دوطرفه با جمله‌ی تعامل (``target ~ C(a) * C(b)``) + مقایسه‌ی AIC با/بدون تعامل.

    خروجی: (جدول anova_lm جمع‌کاسته‌شده‌ی مدل کامل, AIC مدل کامل, AIC مدل بدون تعامل,
    eta^2 جمله‌ی تعامل). eta^2 = sum_sq(تعامل) / sum_sq(کل) — سهم واریانس توضیح‌داده‌شده
    توسط خودِ جمله‌ی تعامل (فراتر از دو اثر اصلی).
    """
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm

    formula_full = f'{target} ~ C({factor_a}) * C({factor_b})'
    formula_additive = f'{target} ~ C({factor_a}) + C({factor_b})'
    model_full = smf.ols(formula_full, data=df).fit()
    model_additive = smf.ols(formula_additive, data=df).fit()
    table = anova_lm(model_full)

    interaction_rows = [i for i in table.index if ":" in i]
    interaction_row = interaction_rows[0] if interaction_rows else None
    eta_sq = table.loc[interaction_row, "sum_sq"] / table["sum_sq"].sum() if interaction_row else float("nan")

    return {
        "anova_table": table,
        "aic_with_interaction": model_full.aic,
        "aic_without_interaction": model_additive.aic,
        "interaction_row": interaction_row,
        "eta_sq_interaction": float(eta_sq),
        "p_value_interaction": float(table.loc[interaction_row, "PR(>F)"]) if interaction_row else float("nan"),
    }


def save_fig(fig, name: str, figures_dir: Path | str, dpi: int = 150) -> Path:
    """ذخیره‌ی شکل matplotlib با نام یکتا (پیشوند بند WBS) در reports/figures/."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / name
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    return out_path
