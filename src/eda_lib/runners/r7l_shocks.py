"""۱۴-۸-۱ پنجره‌ی جنگ به‌عنوان آزمایش طبیعی · ۱۴-۸-۲ شوک تکرارشونده و بازیابی.

اجرا: ``python -m src.eda_lib.runners.r7l_shocks``

⚠️ تنها بندی که عمداً با `features_L1_v4_with_anomalies.parquet` کار می‌کند: پنجره‌ی
شوک از **مدل** حذف شد (ردیف ۸۰) ولی به‌عنوان **موضوع مطالعه** حذف نشد.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import set_global_seed
from src.eda_lib.runners._common_v4 import EVENTS, header, load_features, md_table, show

WAR_START = pd.Timestamp("2025-06-13")


def main() -> None:
    set_global_seed()
    a = load_features(with_anomalies=True,
                      columns=["date_gregorian", "meal", "restaurant", "dish", "Res", "NoRecv",
                               "campus", "city", "venue_type", "is_war_window",
                               "is_structural_gap", "day_of_week", "academic_year"])
    a["rho"] = a.NoRecv / a.Res
    tot = lambda g: g.NoRecv.sum() / g.Res.sum()

    header("۱۴-۸-۱ · ۱ · ⭐ آناتومی شوک — سرعت و شکل فروپاشی")
    win = a[(a.date_gregorian >= WAR_START - pd.Timedelta(days=21))
            & (a.date_gregorian <= WAR_START + pd.Timedelta(days=21))]
    d = (win.groupby(win.date_gregorian.dt.date)
            .agg(سلف=("restaurant", "nunique"), سلول=("Res", "size"), Res=("Res", "sum"),
                 NoRecv=("NoRecv", "sum")))
    d["ρ"] = d.NoRecv / d.Res
    pre = a[(a.date_gregorian >= WAR_START - pd.Timedelta(days=30))
            & (a.date_gregorian < WAR_START)]
    base_vol = float(pre.groupby(pre.date_gregorian.dt.date).Res.sum().median())
    base_rho = tot(pre)
    d["حجم نسبی"] = d.Res / base_vol
    d["ρ نسبی"] = d.ρ / base_rho
    show(d, n=30, float_fmt="%.3f")
    print(f"\n  خط پایه‌ی ۳۰ روز پیش: حجم روزانه {base_vol:,.0f} · ρ={base_rho:.4f}")

    header("۲ · ⭐⭐ تفکیک اثر بر رزروِ **قفل‌شده** از رزروِ **تازه**", 2)
    print("  پنجره‌ی ۷۲ ساعته‌ی رزرو این تفکیک را ممکن می‌کند (F135):")
    print("  · رزروی که پیش از شوک قفل شده ⇒ ρ منفجر می‌شود (کسی نمی‌آید)")
    print("  · رزروی که باید تازه ثبت می‌شد ⇒ حجم صفر می‌شود\n")
    war = a[a.is_war_window.astype(bool)]
    for day, g in war.groupby(war.date_gregorian.dt.date):
        lock = (WAR_START - pd.Timestamp(day)).days
        print(f"    {day} · {'قفل‌شده پیش از شوک' if lock >= -3 else 'ثبت‌شده پس از شوک'} · "
              f"حجم {g.Res.sum():6,} ({g.Res.sum()/base_vol:5.1%}) · ρ={tot(g):.4f} "
              f"({tot(g)/base_rho:5.1f}×)")

    header("۳ · ناهمگونی بین پردیس و نوع سلف", 2)
    for col in ["campus", "venue_type"]:
        t = (war.groupby(col, observed=True)
               .agg(سلول=("Res", "size"), پرس=("Res", "sum"))
               .assign(ρ_شوک=war.groupby(col, observed=True).apply(tot, include_groups=False)))
        pv = pre.groupby(col, observed=True).apply(tot, include_groups=False).rename("ρ_پایه")
        vol_pre = pre.groupby(col, observed=True).Res.sum().rename("پرس_پایه")
        t = t.join(pv).join(vol_pre)
        t["نسبت ρ"] = t.ρ_شوک / t.ρ_پایه
        t["افت حجم"] = 1 - (t.پرس / (t.پرس_پایه * len(war.date_gregorian.unique()) / 21))
        show(t[t.پرس > 100].sort_values("نسبت ρ", ascending=False), float_fmt="%.3f")
        print()

    header("۴ · بازیابی در نیمسال بعد")
    nxt = a[(a.date_gregorian >= pd.Timestamp("2025-09-01"))
            & (a.date_gregorian <= pd.Timestamp("2025-12-31"))]
    prev_year = a[(a.date_gregorian >= pd.Timestamp("2024-09-01"))
                  & (a.date_gregorian <= pd.Timestamp("2024-12-31"))]
    print(f"  مهر–آذر ۱۴۰۳ (پیش از شوک): ρ={tot(prev_year):.4f} · "
          f"حجم روزانه {prev_year.groupby(prev_year.date_gregorian.dt.date).Res.sum().median():,.0f}")
    print(f"  مهر–آذر ۱۴۰۴ (پس از شوک) : ρ={tot(nxt):.4f} · "
          f"حجم روزانه {nxt.groupby(nxt.date_gregorian.dt.date).Res.sum().median():,.0f}")
    print(f"  ⇒ نرخ {tot(nxt)/tot(prev_year):.3f}× · حجم "
          f"{nxt.groupby(nxt.date_gregorian.dt.date).Res.sum().median()/prev_year.groupby(prev_year.date_gregorian.dt.date).Res.sum().median():.3f}×")

    header("۱۴-۸-۲ · ۵ · مطالعه‌ی رویداد — پنجره‌ی −۷ تا +۱۴ روز")
    c = load_features(columns=["date_gregorian", "meal", "restaurant", "Res", "NoRecv",
                               "city", "day_of_week", "venue_type"])
    c["rho"] = c.NoRecv / c.Res
    # باقی‌مانده پس از اثر ثابت سلف×وعده و روزهفته×وعده — تا اثر رویداد خالص باشد
    y = c.rho.to_numpy().astype(float)
    w = c.Res.to_numpy().astype(float)
    for key in [(c.restaurant.astype(str) + "|" + c.meal.astype(str)).to_numpy(),
                (c.day_of_week.astype(str) + "|" + c.meal.astype(str)).to_numpy()]:
        s = pd.DataFrame({"y": y, "w": w, "k": key})
        m = s.groupby("k").apply(lambda g: np.average(g.y, weights=g.w), include_groups=False)
        y = y - s.k.map(m).to_numpy()
    c["adj"] = y

    ev = pd.read_csv(EVENTS, parse_dates=["date_start", "date_end"])
    ev = ev[ev.service_status == "داخل_پنجره_داده"]
    rows = []
    for _, e in ev.iterrows():
        sub = c[c.city == e.scope] if e.scope in set(c.city.unique()) else c
        for k in range(-7, 15):
            day = e.date_start + pd.Timedelta(days=k)
            g = sub[sub.date_gregorian == day]
            if len(g) < 3:
                continue
            rows.append({"نوع": e.event_type, "event_id": e.event_id, "k": k,
                         "adj": float(np.average(g.adj, weights=g.Res)), "پرس": int(g.Res.sum())})
    es = pd.DataFrame(rows)
    piv = es.pivot_table(index="k", columns="نوع", values="adj", aggfunc="mean")
    show(piv, n=25)
    print("\n--- Markdown ---\n" + md_table(piv.reset_index(), "{:.5f}"))

    header("۶ · اندازه‌ی اثر و طول بازیابی، با CI بلوکی")
    from src.cv import block_bootstrap_2d
    for typ, g in es.groupby("نوع"):
        pre_w = g[g.k.between(-7, -1)].adj
        ev_w = g[g.k.between(0, 4)].adj
        post = g[g.k.between(5, 14)].adj
        if len(ev_w) < 10:
            continue
        eff = float(ev_w.mean() - pre_w.mean())
        # CI بلوکی روی رویداد (هر رویداد یک بلوک مستقل)
        ids = g.event_id.unique()
        rng = np.random.default_rng(42)
        boot = []
        for _ in range(500):
            pick = rng.choice(ids, size=len(ids), replace=True)
            s = pd.concat([g[g.event_id == i] for i in pick])
            a_ = s[s.k.between(0, 4)].adj.mean()
            b_ = s[s.k.between(-7, -1)].adj.mean()
            if np.isfinite(a_) and np.isfinite(b_):
                boot.append(a_ - b_)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        sig = "✅" if lo > 0 or hi < 0 else "❌ بی‌اثر"
        rec = float(post.mean() - pre_w.mean())
        print(f"  {typ:18s} رویداد={len(ids):3d} · اثر={eff:+.5f} CI=[{lo:+.5f},{hi:+.5f}] {sig}"
              f" · باقی‌مانده در +۵..+۱۴: {rec:+.5f}")

    header("۷ · پرکردن ستون `observed_in_data` دفتر رویدادها")
    out = []
    for _, e in ev.iterrows():
        g = es[es.event_id == e.event_id]
        if len(g) < 8:
            out.append("داده‌ی ناکافی")
            continue
        eff = g[g.k.between(0, 4)].adj.mean() - g[g.k.between(-7, -1)].adj.mean()
        out.append(f"اثر تعدیل‌شده {eff:+.4f}")
    ev = ev.assign(observed_in_data=out)
    ev.to_csv("data/interim/events_v4_observed.csv", index=False)
    print(f"  {len(ev)} رویداد · {sum('ناکافی' in o for o in out)} با داده‌ی ناکافی")
    print("  ← data/interim/events_v4_observed.csv")


if __name__ == "__main__":
    main()
