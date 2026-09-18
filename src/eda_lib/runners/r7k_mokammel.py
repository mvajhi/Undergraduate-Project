"""۱۴-۷-۳ · رقابت مکمل — آزمون روند موازی و بستن ابهام (ت).

اجرا: ``python -m src.eda_lib.runners.r7k_mokammel``

F100 اثر جانشینی را **−۳.۴۴ وعده در ۳۰ روز (−۱۵.۰٪)** برآورد کرد، با قید صریح که
این **کران بالا**ست نه برآورد علّی، چون پذیرندگان مکمل پایه‌ی مصرف بسیار بالاتری
دارند (۲۲.۸ در برابر ۱۲.۷) و فرض روند موازی بی‌خطر نیست.

⭐ با سه سال داده، آن فرض **قابل آزمون** است — و این تنها کاری است که این بند
می‌کند. تصمیم ورود فیچر همچنان ذی‌نفع است (ردیف ۶۷).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import set_global_seed
from src.eda_lib.runners._common_v4 import header, md_table, show


def main() -> None:
    set_global_seed()
    from src.data.load_v4 import load_clean_v4
    from src.eda_lib.restaurant_taxonomy import add_restaurant_class

    df = add_restaurant_class(load_clean_v4(columns=["PersonId", "miladi", "DateReserve",
                                                     "mealid", "ReserveStatus", "Count",
                                                     "RestaurantName"]))
    dim = pd.read_csv("data/external/restaurant_dim_v4.csv")
    camp = dim.set_index("raw_name").campus

    mok = df[df.restaurant_class == "mokammel"].copy()
    core = df[df.restaurant_class == "core"].copy()
    mok["campus"] = mok.RestaurantName.map(camp)
    core["campus"] = core.RestaurantName.map(camp)

    header("۱ · دامنه و زمان‌بندی ورود شعبه‌های مکمل")
    ent = (mok.groupby("RestaurantName")
              .agg(پردیس=("campus", "first"), نخستین=("date_gregorian", "min"),
                   آخرین=("date_gregorian", "max"), رزرو=("Count", "sum"))
              .sort_values("نخستین"))
    print(f"  {len(ent)} شعبه‌ی مکمل · {mok.Count.sum():,} رزرو · "
          f"{mok.PersonId.nunique():,} فرد")
    show(ent, n=30, float_fmt="%.0f")

    header("۲ · تعداد شعبه‌ی فعال در هر پردیس، ماه به ماه (فیچر پیشنهادی ابهام ت)")
    m = mok.assign(p=mok.date_gregorian.dt.to_period("M"))
    act = (m.groupby(["p", "campus"], observed=True).RestaurantName.nunique().unstack().fillna(0))
    show(act.astype(int), n=35, float_fmt="%.0f")

    header("۳ · ⭐ آزمون روند موازی — شرط اعتبار برآورد F100")
    # گروه تیمار: افرادی که در طول پنجره مکمل را پذیرفتند. زمان تیمار = نخستین رزرو مکمل.
    first_mok = mok.groupby("PersonId").date_gregorian.min().rename("t0")
    adopters = set(first_mok.index)
    print(f"  پذیرندگان مکمل: {len(adopters):,} از {df.PersonId.nunique():,} فرد "
          f"({len(adopters)/df.PersonId.nunique():.1%})")

    c = core.assign(p=core.date_gregorian.dt.to_period("M"))
    vol = c.groupby(["PersonId", "p"], observed=True).Count.sum().rename("وعده").reset_index()
    vol["پذیرنده"] = vol.PersonId.isin(adopters)
    vol = vol.merge(first_mok, left_on="PersonId", right_index=True, how="left")
    vol["t0p"] = vol.t0.dt.to_period("M")
    vol["rel"] = (vol.p.astype("int64") - vol.t0p.astype("int64"))

    print("\n  مصرف ماهانه‌ی سلف اصلی، به تفکیک ماه نسبت به پذیرش مکمل:")
    prof = (vol[vol["پذیرنده"] & vol.rel.between(-8, 6)]
              .groupby("rel").وعده.agg(["mean", "median", "size"]))
    prof.columns = ["میانگین", "میانه", "n"]
    show(prof, n=20)

    print("\n  گروه کنترل (غیرپذیرندگان) در همان ماه‌های تقویمی:")
    ctrl_months = vol[vol["پذیرنده"] & vol.rel.between(-8, 6)][["PersonId", "p", "rel"]]
    cal = (vol[~vol["پذیرنده"]].groupby("p").وعده.mean().rename("کنترل"))
    tr = ctrl_months.merge(cal, on="p").groupby("rel").کنترل.mean()
    both = prof.join(tr)
    both["نسبت تیمار/کنترل"] = both["میانگین"] / both["کنترل"]
    show(both, n=20)
    print("\n--- Markdown ---\n" + md_table(both.reset_index(), "{:.3f}"))

    pre = both.loc[-8:-1, "نسبت تیمار/کنترل"].dropna()
    post = both.loc[1:6, "نسبت تیمار/کنترل"].dropna()
    if len(pre) >= 3:
        from scipy.stats import linregress
        lr = linregress(pre.index.astype(float), pre.values)
        print(f"\n  ⭐ آزمون pre-trend روی نسبت تیمار/کنترل ({len(pre)} ماه پیش از پذیرش):")
        print(f"     شیب = {lr.slope:+.4f} در ماه · p = {lr.pvalue:.4f} · R² = {lr.rvalue**2:.3f}")
        print(f"     {'✅ روند موازی رد نمی‌شود' if lr.pvalue > 0.05 else '❌ روند موازی رد می‌شود ⇒ برآورد F100 کران بالاست'}")
        print(f"\n     میانگین نسبت پیش از پذیرش: {pre.mean():.3f} · پس از پذیرش: {post.mean():.3f}")
        print(f"     تفاضل‌در‌تفاضل (نسبت): {post.mean()-pre.mean():+.3f} "
              f"({(post.mean()-pre.mean())/pre.mean():+.1%})")

    header("۴ · اثر بر **ظرفیت** سلف، نه فقط بر نرخ فرد")
    cm = c.groupby(["campus", "p"], observed=True).agg(پرس=("Count", "sum"),
                                                       فرد=("PersonId", "nunique")).reset_index()
    am = act.stack().rename("شعبه").reset_index()
    am.columns = ["p", "campus", "شعبه"]
    cm = cm.merge(am, on=["p", "campus"], how="left").fillna({"شعبه": 0})
    cm["پرس_به_فرد"] = cm.پرس / cm.فرد
    print("  همبستگی «تعداد شعبه‌ی فعال» با حجم سلف اصلی، درون هر پردیس:")
    for camp_name, g in cm.groupby("campus"):
        if len(g) < 8 or g.شعبه.nunique() < 2:
            continue
        x = g.شعبه.to_numpy(float)
        for col in ["پرس", "پرس_به_فرد"]:
            yv = g[col].to_numpy(float)
            # حذف روند زمانی خطی تا «رشد کلی» به رقابت نسبت داده نشود
            t = np.arange(len(g), dtype=float)
            xr = x - np.polyval(np.polyfit(t, x, 1), t)
            yr = yv - np.polyval(np.polyfit(t, yv, 1), t)
            r = float(np.corrcoef(xr, yr)[0, 1]) if xr.std() > 0 else np.nan
            print(f"    {camp_name:10s} n={len(g):3d} شعبه {int(g.شعبه.min())}–{int(g.شعبه.max())} · "
                  f"{col:10s} r(روندزدوده)={r:+.3f}")


if __name__ == "__main__":
    main()
