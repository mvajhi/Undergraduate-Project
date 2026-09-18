"""۱۴-۵-۳ پروفایل درون‌نیمسال · ۱۴-۵-۴ رمضان · ۱۴-۵-۵ تعطیلات.

اجرا: ``python -m src.eda_lib.runners.r7f_calendar_profiles``

قاعده‌ی مشترک هر سه بند: پروفایل روی **باقی‌مانده‌ی پس از حذف اثر ثابت سلف×وعده و
روز هفته** ساخته می‌شود، نه روی $\\rho$ خام. وگرنه ترکیبِ سلف‌هایی که در آن بازه سرویس
دارند با خودِ اثر تقویمی قاطی می‌شود — همان تله‌ای که F145 در مورد وعده نشان داد.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import set_global_seed
from src.eda_lib.runners._common_v4 import header, load_features, md_table, show

BASE_COLS = ["date_gregorian", "meal", "restaurant", "dish", "Res", "NoRecv", "academic_year",
             "day_of_week", "semester", "week_of_semester", "days_since_semester_start",
             "days_to_semester_end", "is_exam_period", "is_final_exam_period", "is_in_session",
             "is_ramadan", "days_into_ramadan", "is_holiday_any", "is_nowruz_block",
             "days_to_next_holiday", "days_since_last_holiday", "is_bridge_day",
             "holiday_block_length", "is_named_holiday", "venue_type", "month_jalali",
             "day_of_month_jalali"]


def add_adjusted(f: pd.DataFrame) -> pd.DataFrame:
    """$\\rho$ تعدیل‌شده: باقی‌مانده پس از اثر ثابت سلف×وعده و روز هفته، + میانگین کل.

    برگرداندن میانگین کل فقط برای خوانایی است (عدد در مقیاس $\\rho$ می‌ماند)؛ هیچ
    استنباطی به آن وابسته نیست.
    """
    f = f.copy()
    f["rho"] = f.NoRecv / f.Res
    y = f.rho.to_numpy().astype(float)
    w = f.Res.to_numpy().astype(float)
    for key in [(f.restaurant.astype(str) + "|" + f.meal.astype(str)).to_numpy(),
                (f.day_of_week.astype(str) + "|" + f.meal.astype(str)).to_numpy()]:
        s = pd.DataFrame({"y": y, "w": w, "k": key})
        m = s.groupby("k").apply(lambda g: np.average(g.y, weights=g.w), include_groups=False)
        y = y - s.k.map(m).to_numpy()
    f["rho_adj"] = y + float(np.average(f.rho, weights=w))
    return f


def _profile(df: pd.DataFrame, by: str, min_days: int = 2) -> pd.DataFrame:
    g = (df.groupby(by, observed=True)
           .apply(lambda d: pd.Series({
               "روز": d.date_gregorian.nunique(), "پرس": d.Res.sum(),
               "ρ خام": d.NoRecv.sum() / d.Res.sum(),
               "ρ تعدیل‌شده": float(np.average(d.rho_adj, weights=d.Res))}), include_groups=False))
    return g[g["روز"] >= min_days]


def main() -> None:
    set_global_seed()
    f = add_adjusted(load_features(columns=BASE_COLS))
    base = float(np.average(f.rho, weights=f.Res))
    print(f"ρ پایه (وزن‌دار) = {base:.4f}\n")

    # ------------------------------------------------------------------ ۱۴-۵-۳
    header("۱۴-۵-۳ · ۱ · پروفایل درون‌نیمسال بر حسب هفته")
    ses = f[f.week_of_semester.notna()]
    p = _profile(ses, "week_of_semester")
    p["نسبت"] = p["ρ تعدیل‌شده"] / base
    show(p, n=25)
    print("\n--- Markdown ---\n" + md_table(p.reset_index(), "{:.4f}"))

    header("۲ · همان پروفایل، نیمسال اول در برابر دوم", 2)
    piv = (ses.groupby(["semester", "week_of_semester"], observed=True)
              .apply(lambda d: float(np.average(d.rho_adj, weights=d.Res)), include_groups=False)
              .unstack(0))
    piv.columns = [f"نیمسال {int(c)}" for c in piv.columns]
    show(piv, n=25)
    c = piv.dropna()
    if len(c) > 3:
        print(f"\n  همبستگی پروفایل دو نیمسال: {c.iloc[:, 0].corr(c.iloc[:, 1]):.4f}")

    header("۳ · پروفایل روزانه‌ی آغاز و پایان نیمسال", 2)
    for col, lab, rng in [("days_since_semester_start", "روز از آغاز نیمسال", (0, 14)),
                          ("days_to_semester_end", "روز تا پایان نیمسال", (0, 14))]:
        sub = f[f[col].between(*rng)]
        pp = _profile(sub, col)
        pp["نسبت"] = pp["ρ تعدیل‌شده"] / base
        print(f"\n{lab}:")
        show(pp, n=20)

    header("۴ · دوره‌ی امتحانات و دُم پس از آن", 2)
    rows = []
    for lab, m in [("در جلسه، بدون امتحان", f.is_in_session & ~f.is_exam_period),
                   ("میان‌ترم", f.is_exam_period & ~f.is_final_exam_period),
                   ("پایان‌ترم", f.is_final_exam_period),
                   ("پس از پایان کلاس، پیش از امتحان", ~f.is_in_session & ~f.is_exam_period
                    & ~f.is_holiday_any)]:
        d = f[m.fillna(False)]
        if len(d) == 0:
            continue
        rows.append({"رژیم": lab, "روز": d.date_gregorian.nunique(), "پرس": int(d.Res.sum()),
                     "ρ خام": d.NoRecv.sum() / d.Res.sum(),
                     "ρ تعدیل‌شده": float(np.average(d.rho_adj, weights=d.Res)),
                     "نسبت": float(np.average(d.rho_adj, weights=d.Res)) / base})
    show(pd.DataFrame(rows))

    # ------------------------------------------------------------------ ۱۴-۵-۴
    header("۱۴-۵-۴ · ۵ · ⭐ رمضان — از پرچم به شکل")
    ram = f[f.is_ramadan.astype(bool)]
    print(f"  {ram.date_gregorian.nunique()} روز رمضانی · {ram.Res.sum():,} پرس · "
          f"ρ={ram.NoRecv.sum()/ram.Res.sum():.4f} در برابر پایه {base:.4f} "
          f"({ram.NoRecv.sum()/ram.Res.sum()/base - 1:+.1%})")
    prof = (ram.groupby(["days_into_ramadan", "meal"], observed=True)
               .apply(lambda d: float(np.average(d.rho_adj, weights=d.Res)), include_groups=False)
               .unstack())
    cnt = ram.groupby("days_into_ramadan").date_gregorian.nunique().rename("روز")
    out = prof.join(cnt)
    show(out, n=35)
    print("\n--- Markdown ---\n" + md_table(out.reset_index(), "{:.4f}"))

    header("۶ · آیا اثر در طول ماه تشدید یا تضعیف می‌شود؟", 2)
    for meal in ("lunch", "dinner"):
        d = ram[ram.meal == meal].dropna(subset=["days_into_ramadan"])
        if len(d) < 50:
            continue
        x = d.days_into_ramadan.to_numpy(float)
        y = d.rho_adj.to_numpy()
        w = d.Res.to_numpy(float)
        b = np.polyfit(x, y, 1, w=np.sqrt(w))
        from scipy.stats import spearmanr
        agg = (d.groupby("days_into_ramadan")
                 .apply(lambda g: float(np.average(g.rho_adj, weights=g.Res)), include_groups=False))
        r = spearmanr(agg.index, agg.values)
        print(f"  {meal:7s}: شیب روزانه = {b[0]:+.5f}  ({b[0]*29/base:+.1%} در کل ماه) · "
              f"اسپیرمن(روز, ρ) = {r.statistic:+.3f} p={r.pvalue:.4f}")

    header("۷ · ⚠️ ابهام (ب) — حساسیت به جابه‌جایی ±۱ روز مرز آغاز", 2)
    for shift in (-1, 0, 1):
        d = f.copy()
        d["dir"] = d.days_into_ramadan + shift
        m = d.dir.between(1, 29)
        sub = d[m]
        print(f"  شیفت {shift:+d} روز: {sub.date_gregorian.nunique():3d} روز · "
              f"ρ تعدیل‌شده={float(np.average(sub.rho_adj, weights=sub.Res)):.4f} · "
              f"نسبت به پایه {float(np.average(sub.rho_adj, weights=sub.Res))/base:.3f}")
    early = f[f.days_into_ramadan.between(1, 5)]
    print(f"\n  ⚠️ حساس‌ترین ناحیه (۵ روز اول): ρ تعدیل‌شده="
          f"{float(np.average(early.rho_adj, weights=early.Res)):.4f}")

    header("۸ · بازگشت پس از عید فطر", 2)
    ends = (f[f.is_ramadan.astype(bool)].groupby(f.date_gregorian.dt.year)
              .date_gregorian.max())
    rows = []
    for yr, end in ends.items():
        for k in range(-3, 15):
            d = f[f.date_gregorian == end + pd.Timedelta(days=k)]
            if len(d) == 0:
                continue
            rows.append({"سال": yr, "روز نسبت به پایان رمضان": k,
                         "ρ تعدیل‌شده": float(np.average(d.rho_adj, weights=d.Res))})
    rec = pd.DataFrame(rows).pivot_table(index="روز نسبت به پایان رمضان", columns="سال",
                                         values="ρ تعدیل‌شده")
    show(rec, n=20)

    # ------------------------------------------------------------------ ۱۴-۵-۵
    header("۱۴-۵-۵ · ۹ · تعطیلات — نامتقارنی پیش و پس")
    f["signed_dist"] = np.where(f.days_to_next_holiday <= f.days_since_last_holiday,
                                -f.days_to_next_holiday, f.days_since_last_holiday)
    sub = f[f.signed_dist.between(-5, 5) & ~f.is_holiday_any.astype(bool)]
    p = _profile(sub, "signed_dist")
    p["نسبت"] = p["ρ تعدیل‌شده"] / base
    show(p, n=15)
    print("\n--- Markdown ---\n" + md_table(p.reset_index(), "{:.4f}"))
    pre = sub[sub.signed_dist.between(-3, -1)]
    post = sub[sub.signed_dist.between(1, 3)]
    a = float(np.average(pre.rho_adj, weights=pre.Res))
    b_ = float(np.average(post.rho_adj, weights=post.Res))
    print(f"\n  پیش (d−۳..d−۱): {a:.4f}  ·  پس (d+۱..d+۳): {b_:.4f}  ·  "
          f"نامتقارنی = {a - b_:+.4f} ({(a - b_)/base:+.1%} از پایه)")

    header("۱۰ · تعطیلی تک‌روزه در برابر بلوک", 2)
    rows = []
    for lab, m in [("تک‌روزه (بلوک=۱)", f.holiday_block_length == 1),
                   ("بلوک ۲–۳ روزه", f.holiday_block_length.between(2, 3)),
                   ("بلوک ۴+ روزه", f.holiday_block_length >= 4),
                   ("بلوک نوروز", f.is_nowruz_block.astype(bool)),
                   ("روز پل", f.is_bridge_day.astype(bool))]:
        d = f[m.fillna(False)]
        if len(d) == 0:
            continue
        rows.append({"نوع": lab, "روز": d.date_gregorian.nunique(), "پرس": int(d.Res.sum()),
                     "ρ تعدیل‌شده": float(np.average(d.rho_adj, weights=d.Res)),
                     "نسبت": float(np.average(d.rho_adj, weights=d.Res)) / base})
    show(pd.DataFrame(rows))


if __name__ == "__main__":
    main()
