"""۱۴-۶-۱ بازآزمون F46/F55/F56 · ۱۴-۶-۲ ⭐⭐ دوره‌ی غیبت (دروازه‌ی بند ۱۴-۱۶).

اجرا: ``python -m src.eda_lib.runners.r7h_individual``
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import set_global_seed
from src.eda_lib.individual_v4 import (
    build_person_level_v4, hazard_of_ending, simulate_null_spells, spells,
)
from src.eda_lib.runners._common_v4 import header, md_table, show

MIN_RES = 20


def main() -> None:
    set_global_seed()
    d = build_person_level_v4()
    print(f"{len(d):,} رزرو · {d.PersonId.nunique():,} فرد · {d.meal_seq.max()+1} فرصت سرویس · "
          f"نرخ کل {d.dont_receive.mean():.4f}\n")

    # ------------------------------------------------------------------ ۱۴-۶-۱
    header("۱۴-۶-۱ · ۱ · بازآزمون F46 — پایداری نیمه‌اول/نیمه‌دوم")
    d = d.sort_values(["PersonId", "meal_seq"])
    g = d.groupby("PersonId")
    n = g.PersonId.transform("size")
    sub = d[n >= 20].copy()
    sub["half"] = (sub.groupby("PersonId").cumcount()
                   < sub.groupby("PersonId").PersonId.transform("size") / 2)
    h = (sub.groupby(["PersonId", "half"]).dont_receive.mean().unstack())
    h.columns = ["نیمه‌ی دوم", "نیمه‌ی اول"]
    h = h.dropna()
    print(f"  {len(h):,} فرد با ≥۲۰ رزرو")
    print(f"  همبستگی پیرسون نیمه‌اول/نیمه‌دوم: {h['نیمه‌ی اول'].corr(h['نیمه‌ی دوم']):.4f}")
    print(f"  همبستگی اسپیرمن: {h['نیمه‌ی اول'].corr(h['نیمه‌ی دوم'], method='spearman'):.4f}"
          "   [F46 روی v2: ~۰.۵]")

    header("۲ · همان پایداری، این بار **بین سال‌های تحصیلی**", 2)
    yr = (d.assign(ay=np.where(d.jmonth >= 7, d.jyear, d.jyear - 1))
            .groupby(["PersonId", "ay"]).dont_receive.agg(["mean", "size"]))
    yr = yr[yr["size"] >= 20]["mean"].unstack()
    yr.columns = [f"۱۴{int(c)-1400:02d}" for c in yr.columns]
    print(f"  ماتریس فرد × سال: {yr.shape}")
    cols = list(yr.columns)
    for i in range(len(cols) - 1):
        for j in range(i + 1, len(cols)):
            c = yr[[cols[i], cols[j]]].dropna()
            if len(c) < 50:
                continue
            print(f"  {cols[i]} ↔ {cols[j]}: n={len(c):5d}  پیرسون={c.iloc[:,0].corr(c.iloc[:,1]):.4f}  "
                  f"اسپیرمن={c.iloc[:,0].corr(c.iloc[:,1], method='spearman'):.4f}")

    header("۳ · بازآزمون F55 — همبستگی درون‌فردی lag-1 پس از حذف اثر ثابت فرد")
    s = d[n >= MIN_RES].copy()
    s["dev"] = s.dont_receive.astype(float) - s.groupby("PersonId").dont_receive.transform("mean")
    for lag in (1, 2, 3, 5, 10):
        s[f"prev{lag}"] = s.groupby("PersonId").dev.shift(lag)
        ok = s[[f"prev{lag}", "dev"]].dropna()
        print(f"  lag-{lag:<2d}: r = {ok['dev'].corr(ok[f'prev{lag}']):+.4f}   (n={len(ok):,})"
              + ("   [F55 روی v2: +۰.۰۴۹]" if lag == 1 else ""))

    # ------------------------------------------------------------------ ۱۴-۶-۲
    header("۱۴-۶-۲ · ۴ · ⭐⭐ دوره‌ی غیبت — توزیع طول در برابر استقلال")
    sp = spells(d, min_reservations=MIN_RES)
    print(f"  {len(sp):,} دوره از {sp.PersonId.nunique():,} فرد")
    null = simulate_null_spells(sp.groupby("PersonId").base_rate.first().to_numpy(),
                                sp.groupby("PersonId").n.first().to_numpy(), seed=42)
    cmp = pd.DataFrame({
        "مشاهده‌شده": sp.length.value_counts(normalize=True).sort_index(),
        "تحت استقلال": null.length.value_counts(normalize=True).sort_index()}).head(15).fillna(0)
    cmp["نسبت"] = cmp["مشاهده‌شده"] / cmp["تحت استقلال"].replace(0, np.nan)
    show(cmp)
    print("\n--- Markdown ---\n" + md_table(cmp.reset_index(), "{:.5f}"))
    print(f"\n  میانگین طول دوره: مشاهده‌شده {sp.length.mean():.3f} · "
          f"تحت استقلال {null.length.mean():.3f}")
    for k in (2, 3, 4, 5):
        o = float((sp.length >= k).mean())
        e = float((null.length >= k).mean())
        print(f"  سهم دوره‌های ≥{k}: مشاهده {o:.4f} · استقلال {e:.4f} · نسبت {o/e:.2f}×")

    header("۵ · تابع مخاطره‌ی پایان دوره", 2)
    hz = hazard_of_ending(sp)
    show(hz)
    print("\n--- Markdown ---\n" + md_table(hz, "{:.4f}"))
    print("  ⇒ مخاطره‌ی ثابت = بی‌حافظه (استقلال). مخاطره‌ی نزولی = «چسبندگی» دوره.")

    header("۶ · ⭐ سهم **رزروها**یی که داخل دوره‌ی معنادارند — قاعده‌ی تصمیم بند ۱۴-۶-۲")
    total_dont = int(sp.length.sum())
    total_res = int(sp.groupby("PersonId").n.first().sum())
    for k in (2, 3, 4):
        inside = int(sp.loc[sp.length >= k, "length"].sum())
        print(f"  دوره‌های با طول ≥{k}: {int((sp.length>=k).sum()):6,} دوره · "
              f"{inside:8,} رزرو · {inside/total_res:.3%} از کل رزروها · "
              f"{inside/total_dont:.1%} از کل عدم‌دریافت‌ها")
    print(f"\n  کل رزرو افراد واجد شرایط: {total_res:,} · کل عدم‌دریافت: {total_dont:,}")
    print("\n  📌 قاعده‌ی از‌پیش‌ثبت‌شده: اگر سهم رزروهای داخل دوره‌های معنادار < ۵٪ باشد،")
    print("     بند ۱۴-۱۶ بودجه‌ی حداقلی می‌گیرد؛ اگر ≥ ۵٪، بودجه‌ی کامل.")

    header("۷ · هم‌زمانی دوره‌ها با تقویم — آیا «دوره» همان شوک روز F59 است؟")
    starts = sp[sp.length >= 3].start.value_counts().sort_index()
    slot = (d[["meal_seq", "date_gregorian", "meal"]].drop_duplicates()
              .set_index("meal_seq"))
    st = slot.join(starts.rename("شروع دوره"), how="left").fillna({"شروع دوره": 0})
    active = d.groupby("meal_seq").PersonId.nunique().rename("فرد فعال")
    st = st.join(active)
    st["نرخ شروع"] = st["شروع دوره"] / st["فرد فعال"]
    print(f"  همبستگی «نرخ شروع دوره» با «نرخ عدم‌دریافت روز»:")
    day_rate = d.groupby("meal_seq").dont_receive.mean().rename("ρ روز")
    st = st.join(day_rate)
    ok = st[["نرخ شروع", "ρ روز"]].dropna()
    print(f"    پیرسون={ok['نرخ شروع'].corr(ok['ρ روز']):.4f} · "
          f"اسپیرمن={ok['نرخ شروع'].corr(ok['ρ روز'], method='spearman'):.4f}  (n={len(ok)})")
    print("\n  ده فرصتِ سرویس با بیشترین نرخ شروع دوره:")
    show(st.nlargest(10, "نرخ شروع")[["date_gregorian", "meal", "شروع دوره", "فرد فعال",
                                      "نرخ شروع", "ρ روز"]], float_fmt="%.4f")


if __name__ == "__main__":
    main()
