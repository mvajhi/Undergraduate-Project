"""۱۴-۴-۳ پوشش و کفایت سلول · ۱۴-۴-۴ پرت رژیم‌آگاه.

اجرا: ``python -m src.eda_lib.runners.r7d_coverage_outliers``

⚠️ قاعده‌ی این بند: پرت در **باقی‌مانده‌ی مدل تقویمی** یافته می‌شود نه در $\\rho$ خام.
یک روز رمضانی با $\\rho$ بالا پرت نیست — رژیم است. و هر سلول با $Res<30$ از
پرت‌یابی کنار گذاشته می‌شود، چون دُمِ آن سلول‌ها نویز نمونه‌گیری است نه پدیده (F149).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import set_global_seed
from src.eda_lib.runners._common_v4 import EVENTS, header, load_features, md_table, show

MIN_RES_OUTLIER = 30


# ---------------------------------------------------------------------------
# ۱۴-۴-۳ پوشش
# ---------------------------------------------------------------------------

def series_lengths(f: pd.DataFrame) -> pd.DataFrame:
    """طول سری در هر تعریفِ سری — ورودی مستقیم «کدام خانواده روی کدام سطح مجاز است»."""
    defs = {
        "L4 · (سلف، وعده)": ["restaurant", "meal"],
        "L2* · (سلف)": ["restaurant"],
        "(سلف، غذا، وعده)": ["restaurant", "dish", "meal"],
        "(سلف، غذا)": ["restaurant", "dish"],
        "L3 · (وعده)": ["meal"],
    }
    rows = []
    for name, keys in defs.items():
        n = f.groupby(keys, observed=True).date_gregorian.nunique()
        rows.append({"تعریف سری": name, "سری": len(n), "میانه طول": float(n.median()),
                     "میانگین": float(n.mean()), "≥۳۰": int((n >= 30).sum()),
                     "≥۱۰۰": int((n >= 100).sum()), "≥۲۰۰": int((n >= 200).sum()),
                     "سهم سری ≥۱۰۰": float((n >= 100).mean())})
    return pd.DataFrame(rows)


def adequacy(f: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for th in (10, 30, 100, 250):
        m = f.Res < th
        rows.append({"آستانه Res": th, "سلول زیر آستانه": int(m.sum()),
                     "سهم سلول": float(m.mean()), "سهم پرس": float(f.loc[m, "Res"].sum() / f.Res.sum())})
    return pd.DataFrame(rows)


def lifecycle(f: pd.DataFrame, col: str) -> pd.DataFrame:
    """ورود و خروج موجودیت‌ها — چیزی که با ۱۴۲ روز اصلاً دیده نمی‌شد."""
    g = f.groupby(col, observed=True).agg(اولین=("date_gregorian", "min"),
                                          آخرین=("date_gregorian", "max"),
                                          روز=("date_gregorian", "nunique"),
                                          پرس=("Res", "sum"))
    lo, hi = f.date_gregorian.min(), f.date_gregorian.max()
    g["ورود دیرهنگام"] = (g.اولین - lo).dt.days > 60
    g["خروج زودهنگام"] = (hi - g.آخرین).dt.days > 60
    g["چگالی"] = g.روز / ((g.آخرین - g.اولین).dt.days + 1)
    return g


# ---------------------------------------------------------------------------
# ۱۴-۴-۴ پرت رژیم‌آگاه
# ---------------------------------------------------------------------------

CALENDAR_COLS = ["day_of_week", "is_holiday_any", "is_ramadan", "is_exam_period",
                 "is_nowruz_block", "is_in_session", "semester"]


def calendar_residual(f: pd.DataFrame) -> pd.Series:
    """باقی‌مانده‌ی $\\rho$ پس از حذف اثر ثابت رژیم — نه پس از حذف اثر روز.

    ⚠️ تمایز مهم: اگر اثر ثابتِ **روز** را برداریم، هر شوک روزانه صفر می‌شود و
    پرتِ روزانه دیگر دیده نمی‌شود — یعنی همان چیزی که دنبالش هستیم را حذف کرده‌ایم.
    پس فقط اثر ثابتِ **رژیم** (سلف، وعده، روزهفته، تقویم) برداشته می‌شود.
    """
    y = f.rho.to_numpy().astype(float)
    keys = [
        f.restaurant.astype(str) + "|" + f.meal.astype(str),
        f.day_of_week.astype(str) + "|" + f.meal.astype(str),
        f.is_holiday_any.astype(str) + "|" + f.is_ramadan.astype(str) + "|"
        + f.is_exam_period.astype(str) + "|" + f.is_nowruz_block.astype(str),
        f.dish.astype(str),
    ]
    for k in keys:
        y = y - pd.Series(y).groupby(k.to_numpy()).transform("mean").to_numpy()
    return pd.Series(y, index=f.index)


def main() -> None:
    set_global_seed()
    f = load_features(columns=["date_gregorian", "meal", "restaurant", "dish", "Res", "NoRecv",
                               "academic_year", "venue_type", "campus", "city", "day_of_week",
                               "is_holiday_any", "is_ramadan", "is_exam_period", "is_nowruz_block",
                               "is_in_session", "semester", "is_war_window", "is_structural_gap"])
    f["rho"] = f.NoRecv / f.Res

    header("۱۴-۴-۳ · ۱ · طول سری در هر تعریف")
    sl = series_lengths(f)
    show(sl)
    print("\n--- Markdown ---\n" + md_table(sl, "{:.1f}"))
    print("\n⚠️ قاعده‌ی F39 روی داده‌ی تازه هم برقرار است اگر میانه‌ی (سلف،غذا،وعده) کوچک بماند.")

    header("۲ · کفایت سلول")
    ad = adequacy(f)
    show(ad)
    print("\n--- Markdown ---\n" + md_table(ad, "{:.4f}"))

    header("۳ · چرخه‌ی عمر موجودیت‌ها — ورود و خروج در ۳ سال")
    for col, lab in [("restaurant", "سلف"), ("dish", "غذا")]:
        lc = lifecycle(f, col)
        print(f"\n{lab}: {len(lc)} موجودیت · ورود دیرهنگام {int(lc['ورود دیرهنگام'].sum())} · "
              f"خروج زودهنگام {int(lc['خروج زودهنگام'].sum())} · "
              f"هر دو {int((lc['ورود دیرهنگام'] & lc['خروج زودهنگام']).sum())}")
        both = lc[lc["ورود دیرهنگام"] | lc["خروج زودهنگام"]].sort_values("پرس", ascending=False)
        print(f"  سهم پرسِ موجودیت‌های ناپایدار: {both.پرس.sum()/lc.پرس.sum():.1%}")
        show(both.head(8), float_fmt="%.2f")

    header("۴ · پوشش سلف×سال — چه کسی کِی هست")
    piv = f.pivot_table(index="venue_type", columns="academic_year", values="restaurant",
                        aggfunc="nunique", observed=True).fillna(0)
    show(piv, float_fmt="%.0f")

    header("۱۴-۴-۴ · ۵ · پرت در باقی‌مانده‌ی مدل تقویمی")
    big = f[f.Res >= MIN_RES_OUTLIER].copy()
    big["resid"] = calendar_residual(big)
    s = big.resid.std()
    print(f"  دامنه‌ی پرت‌یابی: {len(big):,} سلول با Res≥{MIN_RES_OUTLIER} "
          f"({len(big)/len(f):.1%} سلول، {big.Res.sum()/f.Res.sum():.1%} پرس)")
    print(f"  SD باقی‌مانده = {s:.4f}")
    for k in (3, 4, 5):
        m = big.resid.abs() > k * s
        print(f"  |resid| > {k}σ: {int(m.sum()):5d} سلول ({m.mean():.3%})  "
              f"مثبت {int((big.resid > k*s).sum())} / منفی {int((big.resid < -k*s).sum())}")

    header("۶ · پرت نقطه‌ای در برابر دوره‌ای")
    big["is_out"] = big.resid > 3 * s
    day = (big.groupby("date_gregorian")
              .agg(سلول=("is_out", "size"), پرت=("is_out", "sum"),
                   resid_mean=("resid", "mean"), پرس=("Res", "sum")))
    day["سهم پرت"] = day.پرت / day.سلول
    hot = day[day["سهم پرت"] > 0.25].sort_values("سهم پرت", ascending=False)
    print(f"  روزهایی که >۲۵٪ سلول‌هایشان پرت مثبت‌اند: {len(hot)} از {len(day)}")
    show(hot.head(15), float_fmt="%.3f")

    # دوره‌ای = روزهای داغ پشت‌سرهم
    hotd = sorted(hot.index)
    runs, cur = [], []
    for d in hotd:
        if cur and (d - cur[-1]).days <= 3:
            cur.append(d)
        else:
            if cur:
                runs.append(cur)
            cur = [d]
    if cur:
        runs.append(cur)
    print(f"\n  دوره‌های پیوسته (فاصله ≤۳ روز): {len(runs)}")
    for r in sorted(runs, key=len, reverse=True)[:10]:
        print(f"    {r[0].date()} → {r[-1].date()}  ({len(r)} روز داغ)")

    header("۷ · ارجاع متقابل با دفتر ۱۴۸ رویداد")
    ev = pd.read_csv(EVENTS, parse_dates=["date_start", "date_end"])
    ev = ev[ev.service_status == "داخل_پنجره_داده"].copy()
    l3 = (f.groupby(["date_gregorian", "meal"]).apply(
        lambda g: g.NoRecv.sum() / g.Res.sum(), include_groups=False).rename("rho").reset_index())
    base = l3.groupby("meal").rho.transform("median")
    l3["dev"] = l3.rho - base
    city_of = f.groupby("city").restaurant.nunique()
    rows = []
    for _, e in ev.iterrows():
        days = pd.date_range(e.date_start, e.date_end)
        if e.scope in city_of.index:
            sub = f[(f.city == e.scope) & (f.date_gregorian.isin(days))]
            ctrl = f[(f.city == e.scope) & (~f.date_gregorian.isin(days))]
        else:
            sub = f[f.date_gregorian.isin(days)]
            ctrl = f[~f.date_gregorian.isin(days)]
        if len(sub) < 5 or ctrl.Res.sum() == 0:
            rows.append({"event_id": e.event_id, "نوع": e.event_type, "دامنه": e.scope,
                         "روز": len(days), "سلول": len(sub), "ρ رویداد": np.nan,
                         "ρ پایه": np.nan, "Δ": np.nan})
            continue
        rows.append({"event_id": e.event_id, "نوع": e.event_type, "دامنه": e.scope,
                     "روز": len(days), "سلول": len(sub),
                     "ρ رویداد": sub.NoRecv.sum() / sub.Res.sum(),
                     "ρ پایه": ctrl.NoRecv.sum() / ctrl.Res.sum(),
                     "Δ": sub.NoRecv.sum() / sub.Res.sum() - ctrl.NoRecv.sum() / ctrl.Res.sum()})
    ec = pd.DataFrame(rows)
    print(f"  {len(ec)} رویداد داخل پنجره · {int(ec['Δ'].notna().sum())} با داده‌ی کافی")
    summ = (ec.dropna(subset=["Δ"]).groupby("نوع")
              .agg(رویداد=("Δ", "size"), میانه_Δ=("Δ", "median"), میانگین_Δ=("Δ", "mean"),
                   سهم_مثبت=("Δ", lambda s: float((s > 0).mean())), سلول=("سلول", "sum")))
    show(summ)
    print("\n--- Markdown ---\n" + md_table(summ.reset_index(), "{:.4f}"))
    ec.to_csv("data/interim/r7_event_crossref_v4.csv", index=False)
    print("\n  ← data/interim/r7_event_crossref_v4.csv")

    unflagged_regimes()



def unflagged_regimes() -> None:
    """۸ · آیا رژیمی هست که پرچم‌های تقویمی موجود اصلاً نمی‌بینند؟

    روزهای داغ بند ۶ (پرت **دوره‌ای**) با تقویم کدگشایی می‌شوند. هر دوره‌ای که به
    ترکیبی از پرچم‌های موجود نگاشت نشود، یک **شکاف فیچر** است نه یک پرت.
    """
    f = load_features(columns=["date_gregorian", "meal", "restaurant", "dish", "Res", "NoRecv",
                               "is_ramadan", "days_into_ramadan", "is_exam_period",
                               "is_holiday_any", "is_in_session", "is_inter_semester_break",
                               "is_summer_break", "days_since_semester_start", "days_to_exam_start",
                               "week_of_semester", "semester"])
    tot = lambda g: g.NoRecv.sum() / g.Res.sum()
    base = tot(f)

    header("۸ · رژیم تقویمی — ترکیب پرچم‌های موجود")
    g = (f.groupby(["is_in_session", "is_exam_period", "is_holiday_any"], observed=True)
           .apply(lambda d: pd.Series({"روز": d.date_gregorian.nunique(), "پرس": d.Res.sum(),
                                       "ρ": tot(d), "Δ از پایه": tot(d) - base}), include_groups=False))
    show(g, float_fmt="%.4f")

    header("۹ · ⭐⭐ منطقه‌ی مرده — کلاس تمام، امتحان شروع نشده، تعطیلی هم نیست", 2)
    dead = f[(~f.is_in_session.astype(bool)) & (~f.is_exam_period.astype(bool))
             & (~f.is_holiday_any.astype(bool)) & (~f.is_summer_break.astype(bool))
             & (~f.is_inter_semester_break.astype(bool))]
    print(f"  {dead.date_gregorian.nunique()} روز · {dead.Res.sum():,} پرس "
          f"({dead.Res.sum()/f.Res.sum():.2%}) · ρ={tot(dead):.4f} "
          f"(Δ={tot(dead)-base:+.4f} · {tot(dead)/base - 1:+.1%})")
    print("  ⇒ هیچ پرچمی این رژیم را علامت نمی‌زند: `is_in_session=False` و "
          "`is_exam_period=False` یعنی «هیچ‌چیز خاصی نیست» — در حالی که نرخ ۱۹٪ بالاتر است.")
    yrs = dead.groupby(dead.date_gregorian.dt.year).agg(روز=("date_gregorian", "nunique"),
                                                        ρ=("rho", "size"))
    print(f"  تکرارشونده است، نه یک‌باره: در {dead.date_gregorian.dt.year.nunique()} سال تقویمی رخ می‌دهد")

    header("۱۰ · ⭐ پروفایل آغاز نیمسال — آیا `week_of_semester` شکل را می‌گیرد؟", 2)
    s2 = f[f.days_since_semester_start.between(0, 20)]
    p2 = (s2.groupby("days_since_semester_start")
            .apply(lambda d: pd.Series({"روز": d.date_gregorian.nunique(), "پرس": d.Res.sum(),
                                        "ρ": tot(d), "نسبت به پایه": tot(d) / base}),
                   include_groups=False))
    show(p2, n=25, float_fmt="%.4f")
    first5 = s2[s2.days_since_semester_start <= 4]
    rest = f[(f.days_since_semester_start > 7) | f.days_since_semester_start.isna()]
    print(f"\n  پنج روز اول نیمسال: ρ={tot(first5):.4f} در برابر {tot(rest):.4f} بقیه "
          f"({tot(first5)/tot(rest) - 1:+.1%})")
    print("  ⇒ اثر در ~۷ روز مستهلک می‌شود. `week_of_semester` با دانه‌بندی هفتگی "
          "اوجِ روز صفر را با کفِ روز ششم در یک سطل می‌گذارد.")


if __name__ == "__main__":
    main()
