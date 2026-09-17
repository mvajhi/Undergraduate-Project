"""ممیزی اثر **نگاشت مرده‌ی ۳۱ سلفه** بر آماره‌های سلف‌محور (بند ۱۴-۳-۲).

## چرا این ماژول وجود دارد

F133 نشان داد نگاشت قدیمی ۲۵ خوابگاه «سطح شهر» را **یک** سلف می‌شمرد و همین،
همبستگی درون‌گروهی F63 را از ۰.۰۸۶ به ۰.۱۴۵ متورم کرده بود. آن نگاشت فقط F63 را
آلوده نکرده — **هر آماره‌ای که گروه‌بندی‌اش سلف بوده** مظنون است.

## طراحی — چرا این طراحی و نه بازاجرای فاز ۴

مقایسه‌ی «عدد فاز ۴ روی v2» با «عدد فاز ۱۴ روی v4» دو چیز را هم‌زمان عوض می‌کند
(پنجره و نگاشت) و تفکیک‌ناپذیر است. اینجا **داده ثابت** نگه داشته می‌شود (همان
شبکه‌ی v4) و فقط ستون گروه‌بندی عوض می‌شود. هر تفاوتی که بماند، **مصنوع نگاشت** است.

## جهت اریبی موردانتظار

تجمیع، تغییرات *بین* گروه‌ها را داخل گروه نگه می‌دارد. پس:
- آماره‌های **درون‌گروهی** (همبستگی درون سلف، واریانس درون سلف) → اریبی **به بالا**
- آماره‌های **بین‌گروهی** (ICC سلف، $\\eta^2$ سلف، دامنه‌ی ناهمگنی) → اریبی **به پایین**
- سهم «شوک روز» در تجزیه‌ی واریانس → اریبی **به بالا** (واریانس بین‌سلفیِ جذب‌شده
  جایی برای رفتن ندارد جز باقی‌مانده و اثر روز)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# مؤلفه‌های واریانس
# ---------------------------------------------------------------------------

def icc_oneway(values: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    """ICC(1) با برآوردگر ANOVA یک‌طرفه‌ی اثر تصادفی.

    برآوردگر ANOVA (نه REML) انتخاب شده چون بسته‌بندی‌اش سبک است و همان چیزی است که
    F10 استفاده کرده بود — مقایسه‌پذیری با عدد ثبت‌شده مهم‌تر از کارایی برآورد است.
    اندازه‌های گروه نابرابرند، پس $n_0$ هارمونیک‌وار محاسبه می‌شود.
    """
    s = pd.Series(values).groupby(pd.Series(groups)).agg(["count", "mean"])
    s = s[s["count"] > 0]
    k, n = len(s), int(s["count"].sum())
    if k < 2 or n <= k:
        return {"icc": np.nan, "var_between": np.nan, "var_within": np.nan, "k": k, "n": n}
    grand = float(np.average(s["mean"], weights=s["count"]))
    ss_between = float((s["count"] * (s["mean"] - grand) ** 2).sum())
    gmean = pd.Series(groups).map(s["mean"]).to_numpy()
    ss_within = float(((values - gmean) ** 2).sum())
    ms_between = ss_between / (k - 1)
    ms_within = ss_within / (n - k)
    n0 = (n - float((s["count"] ** 2).sum()) / n) / (k - 1)
    var_b = max((ms_between - ms_within) / n0, 0.0)
    icc = var_b / (var_b + ms_within) if (var_b + ms_within) > 0 else np.nan
    return {"icc": icc, "var_between": var_b, "var_within": ms_within, "k": k, "n": n}


def eta_squared(values: np.ndarray, groups: np.ndarray) -> float:
    """$\\eta^2$ = سهم واریانس توضیح‌داده‌شده‌ی یک عامل دسته‌ای (بدون تصحیح)."""
    s = pd.Series(values).groupby(pd.Series(groups)).agg(["count", "mean"])
    grand = float(values.mean())
    ss_between = float((s["count"] * (s["mean"] - grand) ** 2).sum())
    ss_total = float(((values - grand) ** 2).sum())
    return ss_between / ss_total if ss_total > 0 else np.nan


def omega_squared(values: np.ndarray, groups: np.ndarray) -> float:
    """$\\omega^2$ — نسخه‌ی تصحیح‌شده‌ی $\\eta^2$.

    ⚠️ چرا لازم است: $\\eta^2$ با **تعداد گروه** بالا می‌رود حتی وقتی اثری نیست.
    نگاشت مرده تعداد گروه را از ۷۱ به ۳۰ می‌برد، پس مقایسه‌ی دو $\\eta^2$ با
    کاردینالیتی متفاوت ذاتاً ناعادلانه است و باید کنار $\\omega^2$ خوانده شود.
    """
    s = pd.Series(values).groupby(pd.Series(groups)).agg(["count", "mean"])
    k, n = len(s), len(values)
    if k < 2 or n <= k:
        return np.nan
    grand = float(values.mean())
    ss_between = float((s["count"] * (s["mean"] - grand) ** 2).sum())
    ss_total = float(((values - grand) ** 2).sum())
    ms_within = (ss_total - ss_between) / (n - k)
    denom = ss_total + ms_within
    return (ss_between - (k - 1) * ms_within) / denom if denom > 0 else np.nan


def heterogeneity_range(df: pd.DataFrame, group_col: str, min_portions: int = 1000) -> dict[str, float]:
    """دامنه‌ی ناهمگنی $\\rho$ بین گروه‌ها — همان آماره‌ای که F101 «۲۷ برابر» گزارش کرد.

    وزن‌دهی با `Res` انجام می‌شود نه میانگین ساده‌ی سلول‌ها (مصنوع میانگین ساده‌ی دور ۴).
    """
    g = df.groupby(group_col, observed=True).agg(Res=("Res", "sum"), NoRecv=("NoRecv", "sum"))
    g = g[g.Res >= min_portions]
    g["rho"] = g.NoRecv / g.Res
    return {"k": len(g), "min": float(g.rho.min()), "max": float(g.rho.max()),
            "ratio": float(g.rho.max() / g.rho.min()) if g.rho.min() > 0 else np.inf,
            "iqr": float(g.rho.quantile(0.75) - g.rho.quantile(0.25)),
            "sd": float(g.rho.std())}


def day_vs_venue_variance(df: pd.DataFrame, venue_col: str, value_col: str = "rho") -> dict[str, float]:
    """تجزیه‌ی واریانس سلول به «اثر روز»، «اثر سلف» و باقی‌مانده — نسخه‌ی سبک F59.

    ⚠️ دو عامل هم‌پوشان نیستند ولی متعامد هم نیستند؛ پس سهم هرکدام هم به‌صورت
    **حاشیه‌ای** (تنها عامل در مدل) و هم **افزایشی** (پس از ورود عامل دیگر) گزارش
    می‌شود. گزارش‌کردن فقط یکی از این دو، همان اشتباهی است که بند ۱۴-۱-۳-۵ از آن
    برحذر می‌دارد.
    """
    y = df[value_col].to_numpy()
    grand = y.mean()
    ss_total = float(((y - grand) ** 2).sum())

    def _resid(vals, keys):
        m = pd.Series(vals).groupby(pd.Series(keys)).transform("mean").to_numpy()
        return vals - m

    day = df["date_gregorian"].astype(str) + "|" + df["meal"].astype(str)
    venue = df[venue_col].astype(str)

    r_day = _resid(y, day.to_numpy())
    r_venue = _resid(y, venue.to_numpy())
    r_both = _resid(r_day, venue.to_numpy())

    ss_day = ss_total - float((r_day ** 2).sum())
    ss_venue = ss_total - float((r_venue ** 2).sum())
    ss_both = ss_total - float((r_both ** 2).sum())
    return {
        "day_marginal": ss_day / ss_total,
        "venue_marginal": ss_venue / ss_total,
        "venue_incremental": (ss_both - ss_day) / ss_total,
        "day_incremental": (ss_both - ss_venue) / ss_total,
        "joint": ss_both / ss_total,
        "residual": 1 - ss_both / ss_total,
    }


# ---------------------------------------------------------------------------
# اجراکننده‌ی مقایسه
# ---------------------------------------------------------------------------

def compare_mappings(df: pd.DataFrame, real_col: str = "restaurant",
                     legacy_col: str = "legacy_restaurant") -> pd.DataFrame:
    """هر آماره‌ی سلف‌محور را زیر **هر دو** نگاشت روی همان داده محاسبه می‌کند."""
    rows = []

    def add(name, real, legacy, direction):
        shift = legacy - real
        rel = shift / real if real not in (0, np.nan) and np.isfinite(real) and real != 0 else np.nan
        rows.append({"آماره": name, "نگاشت واقعی (۷۱)": real, "نگاشت مرده (۳۰)": legacy,
                     "تفاوت": shift, "تفاوت نسبی": rel, "جهت موردانتظار": direction})

    y = df["rho"].to_numpy()
    r_icc = icc_oneway(y, df[real_col].to_numpy())
    l_icc = icc_oneway(y, df[legacy_col].to_numpy())
    add("ICC سلف", r_icc["icc"], l_icc["icc"], "↓ پایین")
    add("واریانس بین‌سلفی", r_icc["var_between"], l_icc["var_between"], "↓ پایین")
    add("واریانس درون‌سلفی", r_icc["var_within"], l_icc["var_within"], "↑ بالا")
    add("η² سلف", eta_squared(y, df[real_col].to_numpy()), eta_squared(y, df[legacy_col].to_numpy()), "↓ پایین")
    add("ω² سلف", omega_squared(y, df[real_col].to_numpy()), omega_squared(y, df[legacy_col].to_numpy()), "↓ پایین")

    for col, lab in [(real_col, "real"), (legacy_col, "legacy")]:
        pass
    hr_r = heterogeneity_range(df, real_col)
    hr_l = heterogeneity_range(df, legacy_col)
    add("نسبت بیشینه/کمینه‌ی ρ سلف", hr_r["ratio"], hr_l["ratio"], "↓ پایین")
    add("انحراف معیار ρ بین سلف‌ها", hr_r["sd"], hr_l["sd"], "↓ پایین")
    add("IQR ρ بین سلف‌ها", hr_r["iqr"], hr_l["iqr"], "↓ پایین")

    vd_r = day_vs_venue_variance(df, real_col)
    vd_l = day_vs_venue_variance(df, legacy_col)
    add("سهم حاشیه‌ای اثر روز", vd_r["day_marginal"], vd_l["day_marginal"], "↑ بالا")
    add("سهم حاشیه‌ای اثر سلف", vd_r["venue_marginal"], vd_l["venue_marginal"], "↓ پایین")
    add("سهم افزایشی اثر سلف", vd_r["venue_incremental"], vd_l["venue_incremental"], "↓ پایین")
    add("باقی‌مانده", vd_r["residual"], vd_l["residual"], "↑ بالا")

    out = pd.DataFrame(rows)
    out["تعداد گروه"] = [f"{r_icc['k']} → {l_icc['k']}"] * len(out)
    return out
