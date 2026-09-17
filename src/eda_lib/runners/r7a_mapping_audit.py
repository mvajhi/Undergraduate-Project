"""۱۴-۳-۲ · ممیزی اثر نگاشت مرده بر یافته‌های سلف‌محور فازهای ۴ تا ۹.

اجرا: ``python -m src.eda_lib.runners.r7a_mapping_audit``
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import set_global_seed
from src.eda_lib.mapping_audit_v4 import (
    compare_mappings, heterogeneity_range, icc_oneway, day_vs_venue_variance,
)
from src.eda_lib.runners._common_v4 import add_legacy_name, header, load_features, md_table, show


def main() -> None:
    set_global_seed()
    df = load_features(columns=["date_gregorian", "meal", "restaurant", "dish", "Res",
                                "NoRecv", "rho", "campus", "city", "venue_type", "academic_year"])
    df = add_legacy_name(df)

    header("۰ · ساختار فروپاشی نگاشت مرده")
    collapse = (df.groupby("legacy_restaurant", observed=True)
                  .agg(سلف_واقعی=("restaurant", "nunique"), سلول=("rho", "size"),
                       پرس=("Res", "sum"))
                  .sort_values("سلف_واقعی", ascending=False))
    show(collapse.head(12), float_fmt="%.0f")
    n_real, n_legacy = df.restaurant.nunique(), df.legacy_restaurant.nunique()
    absorbed = int((collapse["سلف_واقعی"] - 1).clip(lower=0).sum())
    print(f"\n{n_real} سلف واقعی → {n_legacy} نام قدیمی | {absorbed} موجودیت جذب شد "
          f"({absorbed / n_real:.1%} از دامنه)")
    share = collapse.loc[collapse["سلف_واقعی"] > 1, "پرس"].sum() / df.Res.sum()
    print(f"سهم پرس‌هایی که زیر نگاشت مرده در یک گروه تجمیع‌شده می‌افتادند: {share:.1%}")

    header("۱ · مقایسه‌ی آماره‌های سلف‌محور زیر دو نگاشت — کل پنجره")
    cmp_all = compare_mappings(df)
    show(cmp_all, n=20)
    print("\n--- Markdown ---\n" + md_table(cmp_all.drop(columns=["تعداد گروه"]), "{:.4f}"))

    header("۲ · همان مقایسه به تفکیک سال تحصیلی — آیا اریبی پایدار است؟")
    parts = []
    for yr, g in df.groupby("academic_year", observed=True):
        c = compare_mappings(g)
        c.insert(0, "سال", yr)
        parts.append(c[c["آماره"].isin(["ICC سلف", "η² سلف", "سهم حاشیه‌ای اثر روز",
                                        "سهم افزایشی اثر سلف"])])
    by_year = pd.concat(parts, ignore_index=True)
    show(by_year, n=20)

    header("۳ · کدام گروه‌ها بیشترین ناهمگنی جذب‌شده را دارند")
    rows = []
    for legacy, g in df.groupby("legacy_restaurant", observed=True):
        if g.restaurant.nunique() < 2:
            continue
        hr = heterogeneity_range(g, "restaurant", min_portions=500)
        if not np.isfinite(hr["ratio"]) or hr["k"] < 2:
            continue
        rows.append({"نام قدیمی": legacy, "سلف واقعی": g.restaurant.nunique(),
                     "ρ گروه": g.NoRecv.sum() / g.Res.sum(),
                     "کمینه ρ عضو": hr["min"], "بیشینه ρ عضو": hr["max"],
                     "نسبت": hr["ratio"], "SD اعضا": hr["sd"]})
    inner = pd.DataFrame(rows).sort_values("نسبت", ascending=False)
    show(inner)
    print("\n--- Markdown ---\n" + md_table(inner, "{:.4f}"))

    header("۴ · سلسله‌مراتب مکان — کدام سطح واقعاً سهم دارد (ورودی ۱۴-۷-۲)")
    y = df.rho.to_numpy()
    lv = []
    for name, col in [("شهر", "city"), ("پردیس", "campus"), ("نوع سلف", "venue_type"),
                      ("نام قدیمی (نگاشت مرده)", "legacy_restaurant"), ("سلف واقعی", "restaurant")]:
        r = icc_oneway(y, df[col].astype(str).to_numpy())
        lv.append({"سطح": name, "تعداد گروه": r["k"], "ICC": r["icc"],
                   "η²": float(np.nan) if r["k"] < 2 else 0.0})
    lv = pd.DataFrame(lv)
    from src.eda_lib.mapping_audit_v4 import eta_squared
    lv["η²"] = [eta_squared(y, df[c].astype(str).to_numpy())
                for c in ["city", "campus", "venue_type", "legacy_restaurant", "restaurant"]]
    show(lv)
    print("\n--- Markdown ---\n" + md_table(lv, "{:.4f}"))

    header("۵ · تجزیه‌ی واریانس کامل زیر هر دو نگاشت")
    vd = pd.DataFrame([{"نگاشت": "واقعی (۷۱)", **day_vs_venue_variance(df, "restaurant")},
                       {"نگاشت": "مرده (۳۰)", **day_vs_venue_variance(df, "legacy_restaurant")}])
    show(vd)

    grid_comparison()
    structural_exposure()
    confounding_checks()



def grid_comparison() -> None:
    """۶ · چرا ICC ثبت‌شده‌ی F10 بازتولید نمی‌شود — تفکیک پنجره از شبکه از نگاشت."""
    from src.eda_lib.mapping_audit_v4 import icc_oneway

    header("۶ · بازتولید ICC ثبت‌شده‌ی F10 — سه عامل، تفکیک‌شده")
    v2 = pd.read_csv("data/processed/dataset_v2.csv", parse_dates=["date_gregorian"])
    v4 = add_legacy_name(load_features(columns=["date_gregorian", "meal", "restaurant", "dish",
                                               "Res", "NoRecv", "rho"]))
    w0, w1 = v2.date_gregorian.min(), v2.date_gregorian.max()
    v4w = v4[(v4.date_gregorian >= w0) & (v4.date_gregorian <= w1)].copy()

    # «شبکه‌ی سبک v2»: سلول‌های v4 را دقیقاً همان‌طور که v2 می‌ساخت ادغام کن
    merged = (v4w.groupby(["date_gregorian", "meal", "legacy_restaurant", "dish"], observed=True)
                 .agg(Res=("Res", "sum"), NoRecv=("NoRecv", "sum"), n_sub=("restaurant", "nunique"))
                 .reset_index())
    merged["rho"] = merged.NoRecv / merged.Res

    def pair(df, col, dc, mc):
        y = df["rho"].to_numpy()
        d = (df[dc].astype(str) + "|" + df[mc].astype(str)).to_numpy()
        return icc_oneway(y, df[col].astype(str).to_numpy())["icc"], icc_oneway(y, d)["icc"]

    rows = []
    for lab, df, col, dc, mc in [
        ("v2 — فایل تجمیعی اصلی (مرجع F10)", v2, "RestaurantName", "date_gregorian", "Meal"),
        ("v4 ← ادغام‌شده به سبک v2", merged, "legacy_restaurant", "date_gregorian", "meal"),
        ("v4 — سلول واقعی، برچسب قدیمی", v4w, "legacy_restaurant", "date_gregorian", "meal"),
        ("v4 — سلول واقعی، سلف واقعی", v4w, "restaurant", "date_gregorian", "meal"),
        ("v4 — کل پنجره، سلف واقعی", v4, "restaurant", "date_gregorian", "meal"),
    ]:
        iv, idd = pair(df, col, dc, mc)
        rows.append({"پیکربندی": lab, "سلول": len(df), "میانه Res": float(df.Res.median()),
                     "ICC سلف": iv, "ICC روز×وعده": idd})
    tab = pd.DataFrame(rows)
    show(tab)
    print("\n--- Markdown ---\n" + md_table(tab, "{:.4f}"))

    # اثر سلف که *بین* شبکه‌ها ثابت می‌ماند
    r2 = v2.groupby("RestaurantName").apply(lambda g: g.NoRecv.sum() / g.Res.sum(), include_groups=False)
    r4 = v4w.groupby("legacy_restaurant").apply(lambda g: g.NoRecv.sum() / g.Res.sum(), include_groups=False)
    c = pd.DataFrame({"v2": r2, "v4": r4}).dropna()
    print(f"\nاثر سلف، وزن‌دار با پرس — آماره‌ی شبکه‌ناوردا:")
    print(f"  همبستگی ρ سطح سلف بین دو شبکه = {c.v2.corr(c.v4):.4f}")
    print(f"  SD بین سلف‌ها: v2={c.v2.std():.4f}  v4={c.v4.std():.4f}  (اختلاف {abs(c.v2.std()-c.v4.std()):.4f})")
    print(f"  سهم سلول‌های ρ=0: v2={(v2.rho==0).mean():.1%}  v4={(v4w.rho==0).mean():.1%}")
    print(f"  SD ρ سلولی:      v2={v2.rho.std():.4f}  v4={v4w.rho.std():.4f}")


def structural_exposure() -> None:
    """۷ · نگاشت مرده چه چیزی را حفظ و چه چیزی را نابود کرد — قاعده‌ی معافیت."""
    header("۷ · دامنه‌ی آسیب نگاشت مرده — کدام سطح حفظ شد")
    d = pd.read_csv("data/external/restaurant_dim_v4.csv")
    d = d[d.restaurant_class == "core"].copy()
    d["legacy"] = d.legacy_canonical_name.fillna("«خارج از نگاشت قدیمی»")
    g = d.groupby("legacy").agg(سلف=("raw_name", "nunique"), پردیس=("campus", "nunique"),
                                شهر=("city", "nunique"), نوع=("venue_type", "nunique"),
                                جنسیت=("gender", "nunique"))
    print(f"گروه‌های نگاشت مرده که بیش از یک مقدار در هر بُعد دارند:")
    for col in ["پردیس", "شهر", "نوع", "جنسیت"]:
        print(f"  {col:8s}: {int((g[col] > 1).sum())} گروه از {len(g)}")
    show(g[(g["نوع"] > 1) | (g["جنسیت"] > 1)].sort_values("سلف", ascending=False), float_fmt="%.0f")
    print("\n⇒ قاعده‌ی معافیت: یافته‌ای که گروه‌بندی یا کنترلش **شهر یا پردیس** است معاف است؛")
    print("  یافته‌ای که گروه‌بندی یا کنترلش **سلف، نوع سلف یا جنسیت** است باید بازمحاسبه شود.")


def confounding_checks() -> None:
    """۸ · دو یافته‌ی پرمصرف که قاعده‌ی بند ۷ آن‌ها را «در معرض» علامت می‌زند."""
    from scipy.stats import chi2_contingency

    header("۸ · F83/F43 — هم‌آمیختگی وعده با نوع سلف، و F17 که بر آن سوار است")
    v4 = load_features(columns=["date_gregorian", "meal", "restaurant", "Res", "NoRecv", "rho",
                                "venue_type", "academic_year"])

    def cramers_v(a, b, w):
        t = pd.crosstab(a, b, values=w, aggfunc="sum").fillna(0)
        chi2 = chi2_contingency(t)[0]
        return float(np.sqrt(chi2 / (t.values.sum() * (min(t.shape) - 1))))

    print(f"کرامر V (نوع سلف × وعده، وزن‌دار با پرس) = {cramers_v(v4.venue_type, v4.meal, v4.Res):.3f}"
          f"   [F83 روی v2: 0.732]")
    t = v4.pivot_table(index="venue_type", columns="meal", values="Res", aggfunc="sum").fillna(0)
    t["سهم ناهار"] = t.lunch / (t.lunch + t.dinner)
    t["کل پرس"] = t.lunch + t.dinner
    show(t.sort_values("کل پرس", ascending=False), float_fmt="%.3f")
    fd = v4[(v4.venue_type == "faculty") & (v4.meal == "dinner")]
    print(f"\n⚠️ سلف دانشکده‌ای با سرویس شام: {fd.restaurant.nunique()} سلف، {fd.Res.sum():,.0f} پرس")
    print("   F43 روی v2 گفته بود «فقط یک سلف»؛ روی v4 **تفکیک کامل** است، نه ۹۸.۱٪.")

    header("۹ · F17 — آیا «ناهار بالاتر از شام» مصنوع ترکیب نوع سلف است؟", 2)
    tot = lambda g: g.NoRecv.sum() / g.Res.sum()
    b = v4.groupby("meal").apply(tot, include_groups=False)
    raw = b["lunch"] - b["dinner"]
    print(f"  خام: ناهار {b['lunch']:.4f} · شام {b['dinner']:.4f} · Δ={raw:+.4f}   [F17 روی v2: +0.0204]")

    both = v4.groupby("restaurant").meal.nunique()
    sub = v4[v4.restaurant.isin(set(both[both == 2].index))]
    p = (sub.groupby(["restaurant", "meal"])
            .apply(lambda g: pd.Series({"rho": tot(g), "Res": g.Res.sum()}), include_groups=False)
            .unstack().dropna())
    dd = p[("rho", "lunch")] - p[("rho", "dinner")]
    ww = p[("Res", "lunch")] + p[("Res", "dinner")]
    print(f"  درون‌سلفی ({len(dd)} سلف دو‌وعده‌ای): میانگین وزن‌دار Δ={np.average(dd, weights=ww):+.4f} · "
          f"میانه {dd.median():+.4f} · {int((dd > 0).sum())}/{len(dd)} سلف ناهارشان بالاتر")

    lu, di = v4[v4.meal == "lunch"], v4[v4.meal == "dinner"]
    idx = sorted(set(v4.venue_type.unique()))
    wl = (lu.groupby("venue_type").Res.sum() / lu.Res.sum()).reindex(idx).fillna(0)
    wd = (di.groupby("venue_type").Res.sum() / di.Res.sum()).reindex(idx).fillna(0)
    rl = lu.groupby("venue_type").apply(tot, include_groups=False).reindex(idx).fillna(0)
    rd = di.groupby("venue_type").apply(tot, include_groups=False).reindex(idx).fillna(0)
    comp = float(((wl - wd) * ((rl + rd) / 2)).sum())
    rate = float((((wl + wd) / 2) * (rl - rd)).sum())
    print(f"  تجزیه‌ی Oaxaca‌وار: Δ خام {raw:+.4f} = ترکیبِ نوع سلف {comp:+.4f} ({comp/raw:+.0%}) "
          f"+ نرخ واقعی {rate:+.4f} ({rate/raw:+.0%})")
    print("  ⇒ ترکیب **علیه** شکاف کار می‌کند؛ اثر واقعی وعده بزرگ‌تر از عدد خام است، نه کوچک‌تر.")


if __name__ == "__main__":
    main()
