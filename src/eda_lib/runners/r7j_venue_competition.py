"""۱۴-۷-۱ اثر سلف · ۱۴-۷-۲ هم‌خطی مکان · ۱۴-۷-۳ رقابت مکمل · ۱۴-۷-۴ آب‌وهوا.

اجرا: ``python -m src.eda_lib.runners.r7j_venue_competition``
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import set_global_seed
from src.eda_lib.individual_v4 import build_person_level_v4
from src.eda_lib.mapping_audit_v4 import eta_squared, icc_oneway
from src.eda_lib.runners._common_v4 import WEATHER, header, load_features, md_table, show


def main() -> None:
    set_global_seed()
    f = load_features(columns=["date_gregorian", "meal", "restaurant", "dish", "Res", "NoRecv",
                               "academic_year", "venue_type", "campus", "city", "gender",
                               "day_of_week", "is_holiday_any", "is_tehran",
                               "distance_km_to_tehran_campus", "temp_mean", "temp_max", "temp_min",
                               "precipitation_sum", "snowfall_sum", "aqi_us_max",
                               "relative_humidity_mean"])
    f["rho"] = f.NoRecv / f.Res
    tot = lambda g: g.NoRecv.sum() / g.Res.sum()
    base = tot(f)

    header("۱۴-۷-۱ · ۱ · اثر سلف روی ۷۱ موجودیت واقعی")
    v = (f.groupby("restaurant", observed=True)
           .agg(نوع=("venue_type", "first"), پردیس=("campus", "first"), پرس=("Res", "sum"),
                روز=("date_gregorian", "nunique"))
           .assign(ρ=f.groupby("restaurant", observed=True).apply(tot, include_groups=False)))
    v = v[v.پرس >= 1000].sort_values("ρ")
    print(f"  {len(v)} سلف با ≥۱۰۰۰ پرس · دامنه ρ: {v.ρ.min():.4f} تا {v.ρ.max():.4f} "
          f"({v.ρ.max()/v.ρ.min():.1f} برابر)")
    print("\n  پنج پایین‌ترین و پنج بالاترین:")
    show(pd.concat([v.head(5), v.tail(5)]), float_fmt="%.4f")

    header("۲ · تفکیک به نوع سلف", 2)
    t = (f.groupby("venue_type", observed=True)
           .agg(سلف=("restaurant", "nunique"), پرس=("Res", "sum"))
           .assign(ρ=f.groupby("venue_type", observed=True).apply(tot, include_groups=False)))
    t["نسبت"] = t.ρ / base
    t["SD بین سلف"] = f.groupby("venue_type", observed=True).apply(
        lambda g: g.groupby("restaurant", observed=True).apply(tot, include_groups=False).std(),
        include_groups=False)
    show(t.sort_values("ρ"), float_fmt="%.4f")
    print("\n--- Markdown ---\n" + md_table(t.reset_index(), "{:.4f}"))

    header("۱۴-۷-۲ · ۳ · ⚠️ سلسله‌مراتب مکان — کدام سطح سهم مستقل دارد")
    y = f.rho.to_numpy()
    rows = []
    for name, col in [("شهر", "city"), ("پردیس", "campus"), ("نوع سلف", "venue_type"),
                      ("سلف", "restaurant")]:
        k = f[col].astype(str).to_numpy()
        rows.append({"سطح": name, "گروه": f[col].nunique(), "ICC": icc_oneway(y, k)["icc"],
                     "η²": eta_squared(y, k)})
    # سهم افزایشی: پردیس پس از شهر، نوع پس از پردیس، سلف پس از نوع
    order = [("city", "شهر"), ("campus", "پردیس"), ("venue_type", "نوع سلف"), ("restaurant", "سلف")]
    r = y.copy()
    ss0 = float(np.var(y))
    inc = []
    for col, name in order:
        before = float(np.var(r))
        r = r - pd.Series(r).groupby(f[col].astype(str).to_numpy()).transform("mean").to_numpy()
        inc.append((before - float(np.var(r))) / ss0)
    lv = pd.DataFrame(rows)
    lv["سهم افزایشی (به ترتیب لانه)"] = inc
    show(lv)
    print("\n--- Markdown ---\n" + md_table(lv, "{:.4f}"))
    ct = pd.crosstab(f.city, f.campus)
    print(f"\n  جدول تقاطعی شهر × پردیس: {ct.shape} · خانه‌های ناصفر: {(ct>0).values.sum()}")
    print("  ⇒ اگر برابر تعداد شهر باشد، نگاشت یک‌به‌یک است و دو متغیر یکی‌اند (F146).")

    header("۴ · بازآزمون F84 — چقدر از اختلاف دو نوع سلف «ترکیب افراد» است؟")
    d = build_person_level_v4()
    vt = f[["restaurant", "venue_type"]].drop_duplicates().set_index("restaurant").venue_type
    d = d.assign(venue_type=d.RestaurantName.map(vt))
    d = d[d.venue_type.notna()]
    dorm = d.venue_type.isin(["dorm", "dorm_city", "dorm_married"])
    raw_gap = d[dorm].dont_receive.mean() - d[~dorm].dont_receive.mean()
    print(f"  نرخ خوابگاهی {d[dorm].dont_receive.mean():.4f} · "
          f"غیرخوابگاهی {d[~dorm].dont_receive.mean():.4f} · شکاف خام {raw_gap:+.4f}")
    # افرادی که در هر دو نوع رزرو دارند ⇒ مقایسه‌ی درون‌فردی
    both = d.assign(is_dorm=dorm).groupby("PersonId").is_dorm.nunique()
    ids = set(both[both == 2].index)
    w = d[d.PersonId.isin(ids)].assign(is_dorm=lambda x: x.venue_type.isin(
        ["dorm", "dorm_city", "dorm_married"]))
    per = w.groupby(["PersonId", "is_dorm"]).dont_receive.agg(["mean", "size"]).unstack()
    per = per[(per[("size", True)] >= 5) & (per[("size", False)] >= 5)]
    within = float((per[("mean", True)] - per[("mean", False)]).mean())
    print(f"  {len(per):,} فرد در **هر دو** نوع سلف رزرو دارند (≥۵ رزرو در هرکدام)")
    print(f"  شکاف **درون‌فردی** (همان افراد، دو مکان): {within:+.4f}")
    if raw_gap != 0:
        print(f"  ⇒ سهم «مکان» = {within/raw_gap:.1%} · سهم «ترکیب افراد» = {1-within/raw_gap:.1%}"
              "   [F84 روی v2: ترکیب ۶۱٪]")

    header("۱۴-۷-۴ · ۵ · ⚠️ آب‌وهوا — قید شناسایی پیش از هر عدد")
    print("  شهر ≡ پردیس (F146). پس هر برآوردی که از تفاوت **بین شهرها** بیاید،")
    print("  اثر ثابت پردیس است نه اثر آب‌وهوا. تنها شناسایی معتبر از تغییرات")
    print("  **زمانی درون یک شهر** می‌آید.\n")
    wcols = ["temp_mean", "temp_max", "precipitation_sum", "snowfall_sum", "aqi_us_max",
             "relative_humidity_mean"]
    rows = []
    for c in wcols:
        x = f[c].to_numpy(float)
        ok = ~np.isnan(x)
        raw = float(np.corrcoef(x[ok], y[ok])[0, 1])
        # درون شهر×ماه: حذف اثر ثابت شهر×ماه از هر دو
        key = (f.city.astype(str) + "|" + f.date_gregorian.dt.to_period("M").astype(str)).to_numpy()
        xr = x - pd.Series(x).groupby(key).transform("mean").to_numpy()
        yr = y - pd.Series(y).groupby(key).transform("mean").to_numpy()
        ok2 = ~np.isnan(xr)
        # + حذف روز هفته و سلف
        for k in [f.day_of_week.astype(str).to_numpy(), f.restaurant.astype(str).to_numpy()]:
            yr = yr - pd.Series(yr).groupby(k).transform("mean").to_numpy()
            xr = xr - pd.Series(xr).groupby(k).transform("mean").to_numpy()
        ctrl = float(np.corrcoef(xr[ok2], yr[ok2])[0, 1])
        rows.append({"متغیر": c, "r خام": raw, "r درون شهر×ماه + کنترل": ctrl,
                     "افت": abs(ctrl) / abs(raw) if raw else np.nan})
    wt = pd.DataFrame(rows)
    show(wt)
    print("\n--- Markdown ---\n" + md_table(wt, "{:.4f}"))

    header("۶ · آیا اثر آب‌وهوا برای خوابگاهی و غیرخوابگاهی فرق دارد؟", 2)
    f["is_dorm_venue"] = f.venue_type.isin(["dorm", "dorm_city", "dorm_married"])
    key = (f.city.astype(str) + "|" + f.date_gregorian.dt.to_period("M").astype(str)).to_numpy()
    yr = y - pd.Series(y).groupby(key).transform("mean").to_numpy()
    for k in [f.day_of_week.astype(str).to_numpy(), f.restaurant.astype(str).to_numpy()]:
        yr = yr - pd.Series(yr).groupby(k).transform("mean").to_numpy()
    for c in ["temp_mean", "precipitation_sum", "snowfall_sum", "aqi_us_max"]:
        x = f[c].to_numpy(float)
        xr = x - pd.Series(x).groupby(key).transform("mean").to_numpy()
        out = []
        for lab, m in [("خوابگاهی", f.is_dorm_venue.to_numpy()),
                       ("غیرخوابگاهی", ~f.is_dorm_venue.to_numpy())]:
            ok = m & ~np.isnan(xr)
            out.append(f"{lab}={float(np.corrcoef(xr[ok], yr[ok])[0,1]):+.4f}")
        print(f"  {c:22s} " + " · ".join(out))
    print("\n  ⚠️ فرضیه‌ی مستند: غیرخوابگاهی باید حساس‌تر باشد چون باید تردد کند.")


if __name__ == "__main__":
    main()
