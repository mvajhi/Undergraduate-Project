"""۱۴-۴-۱ · توزیع هدف روی ۵۹۳ روز — شکل، دم، تراکم صفر، پایداری بین سال.

اجرا: ``python -m src.eda_lib.runners.r7b_target_distribution``
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import set_global_seed
from src.eda_lib.distribution_v4 import fit_families, zero_excess, zero_profile
from src.eda_lib.runners._common_v4 import (
    academic_year_order, header, load_features, load_grid, md_table, show,
)


def _levels() -> dict[str, pd.DataFrame]:
    """چهار سطح تجمیعی — همان تعریف جدول سطوح `AGENTS.md`."""
    l1 = load_grid("L1")
    return {
        "L1 سلول (d,m,r,f)": l1,
        "L2 سلف×وعده (d,m,r)": load_grid("L2"),
        "L3 عامل روز (d,m)": load_grid("L3"),
        "L4 سری (m,r) — نقطه‌ها": load_grid("L2"),
    }


def main() -> None:
    set_global_seed()

    header("۱ · شکل توزیع در هر سطح")
    rows = []
    for name, d in _levels().items():
        rho = d.NoRecv / d.Res
        rows.append({"سطح": name, "n": len(d), "میانه Res": float(d.Res.median()),
                     "ρ وزنی": float(d.NoRecv.sum() / d.Res.sum()), "میانگین ساده": float(rho.mean()),
                     "میانه": float(rho.median()), "SD": float(rho.std()),
                     "چولگی": float(rho.skew()), "کشیدگی": float(rho.kurtosis()),
                     "P(ρ=0)": float((rho == 0).mean()), "P(ρ>0.5)": float((rho > 0.5).mean())})
    tab = pd.DataFrame(rows)
    show(tab)
    print("\n--- Markdown ---\n" + md_table(tab, "{:.4f}"))

    header("۲ · برازش خانواده‌ی توزیعی روی L1 — کل پنجره")
    l1 = load_grid("L1")
    fam = fit_families(l1.NoRecv.to_numpy(), l1.Res.to_numpy())
    show(fam)
    print("\n--- Markdown ---\n" + md_table(fam[["خانواده", "k_par", "logL", "AIC", "BIC", "ΔAIC", "μ", "φ", "π₀"]], "{:.2f}"))
    best = fam.loc[fam.AIC.idxmin()]
    print(f"\nبرنده: {best['خانواده']}  (μ={best['μ']:.4f}, φ={best['φ']:.2f}, π₀={best['π₀']:.4f})")

    header("۳ · ⭐ تراکم صفر — نمونه‌گیری یا ساختار؟")
    bb = fam[fam["خانواده"] == "بتا-دوجمله‌ای"].iloc[0]
    ze = zero_excess(l1.NoRecv.to_numpy(), l1.Res.to_numpy(), bb["μ"], bb["φ"])
    print("  " + " · ".join(f"{k}={v:.4f}" for k, v in ze.items()))
    zp = zero_profile(l1, mu=bb["μ"], phi=bb["φ"])
    show(zp)
    print("\n--- Markdown ---\n" + md_table(zp.reset_index(), "{:.4f}"))

    header("۴ · پایداری خانواده بین سه سال تحصیلی")
    f = load_features(columns=["date_gregorian", "meal", "restaurant", "dish", "Res", "NoRecv",
                               "academic_year"])
    parts = []
    for yr in academic_year_order():
        g = f[f.academic_year == yr]
        if len(g) < 500:
            continue
        ff = fit_families(g.NoRecv.to_numpy(), g.Res.to_numpy())
        ff.insert(0, "سال", yr)
        parts.append(ff)
    by_year = pd.concat(parts, ignore_index=True)
    show(by_year[["سال", "خانواده", "logL", "AIC", "ΔAIC", "μ", "φ", "π₀", "N"]], n=20)

    piv = by_year[by_year["خانواده"] == "ZI بتا-دوجمله‌ای"].set_index("سال")[["μ", "φ", "π₀"]]
    print("\nپارامترهای ZI-BB بین سال‌ها:")
    show(piv)
    rng = (piv.max() - piv.min()) / piv.mean()
    print("\nدامنه‌ی نسبی هر پارامتر (max−min)/mean:")
    print("  " + " · ".join(f"{c}={rng[c]:.1%}" for c in piv.columns))

    header("۵ · دم بالا — کدام سلول‌ها ρ>0.5 دارند")
    f["rho"] = f.NoRecv / f.Res
    tail = f[f.rho > 0.5]
    print(f"  {len(tail):,} سلول ({len(tail)/len(f):.2%}) · {tail.Res.sum():,} پرس ({tail.Res.sum()/f.Res.sum():.3%})")
    print(f"  میانه‌ی Res در دم: {tail.Res.median():.0f} در برابر {f.Res.median():.0f} کل")
    print("\n  توزیع اندازه‌ی سلول در دم:")
    show(tail.Res.describe().to_frame("Res دم"), float_fmt="%.1f")
    print("\n  ⚠️ سلول‌های دم با Res≥30 (یعنی دمِ *واقعی*، نه نویز سلول کوچک):")
    real = tail[tail.Res >= 30]
    print(f"  {len(real):,} سلول · {real.Res.sum():,} پرس · "
          f"{real.restaurant.nunique()} سلف · {real.date_gregorian.nunique()} روز")
    if len(real):
        show(real.groupby("restaurant").agg(سلول=("rho", "size"), پرس=("Res", "sum"),
                                            ρ=("rho", "mean")).sort_values("سلول", ascending=False).head(10))


if __name__ == "__main__":
    main()
