"""۱۴-۶-۳ پویایی کوهورت · ۱۴-۶-۴ بازآزمون F59/F62 · ۱۴-۶-۱ (مسابقه‌ی فیچر فردی).

اجرا: ``python -m src.eda_lib.runners.r7i_variance_cohort``

⚠️ قاعده‌ی ۱۴-۱-د: مسابقه‌ی فیچر فردی **هم** با AUC سطح فرد و **هم** در سطح سلول
داوری می‌شود. AUC فردی یک تشخیص است، نه معیار انتخاب.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import set_global_seed
from src.eda_lib.individual_v4 import build_person_level_v4
from src.eda_lib.runners._common_v4 import header, md_table, show

MIN_RES = 20


def expanding_prior(values: pd.Series, by: pd.Series, prior_mean: float,
                    prior_n: float = 10.0) -> pd.Series:
    """میانگین انبساطی هر فرد با هموارسازی بیزی، **کاملاً پیش از لحظه‌ی برش**.

    ⚠️ تله‌ای که نسخه‌ی نخست این بند در آن افتاد: `grouped.shift(1).expanding()`
    گروه‌بندی را **از دست می‌دهد** — `shift` یک Series برمی‌گرداند و `expanding`
    روی کل قاب اجرا می‌شود، نه درون هر فرد. نتیجه AUC ۰.۵۰۸ بود (تصادف محض) و
    به‌راحتی می‌شد آن را «یافته» خواند. همیشه با `cumsum` صریحِ درون‌گروهی.
    """
    sh = values.astype(float).groupby(by).shift(1)
    csum = sh.fillna(0).groupby(by).cumsum()
    cnt = sh.notna().astype(float).groupby(by).cumsum()
    return (csum + prior_mean * prior_n) / (cnt + prior_n)


def main() -> None:
    set_global_seed()
    d = build_person_level_v4().sort_values(["PersonId", "meal_seq"])
    base = float(d.dont_receive.mean())

    header("۱۴-۶-۱ · ۱ · مسابقه‌ی فیچر فردی — AUC سطح فرد (بازآزمون F56)")
    n = d.groupby("PersonId").PersonId.transform("size")
    s = d[n >= MIN_RES].copy()
    y = s.dont_receive.astype(float)
    g = s.groupby("PersonId").dont_receive
    feats = {
        "انبساطی (هموار)": expanding_prior(s.dont_receive, s.PersonId, base),
        "EWM(۵)": g.transform(lambda x: x.astype(float).shift(1).ewm(span=5, min_periods=1).mean()),
        "rolling(۱۰)": g.transform(lambda x: x.astype(float).shift(1).rolling(10, min_periods=1).mean()),
        "rolling(۳)": g.transform(lambda x: x.astype(float).shift(1).rolling(3, min_periods=1).mean()),
        "آخرین مقدار": g.transform(lambda x: x.astype(float).shift(1)),
    }
    from sklearn.metrics import roc_auc_score
    rows = []
    for name, f in feats.items():
        ok = f.notna() & y.notna()
        rows.append({"فیچر": name, "n": int(ok.sum()), "AUC": roc_auc_score(y[ok], f[ok]),
                     "پوشش": float(ok.mean())})
    auc = pd.DataFrame(rows).sort_values("AUC", ascending=False)
    show(auc)
    print("   [F56 روی v2: انبساطی AUC ۰.۷۲۰۰ و بهتر از EWM(۵) و rolling(۱۰)]")

    header("۲ · ⚠️ همان مسابقه، **در سطح سلول** (قاعده‌ی ۱۴-۱-د)")
    s = s.assign(**{f"f_{i}": v for i, v in enumerate(feats.values())})
    names = list(feats.keys())
    cell = (s.groupby(["date_gregorian", "meal", "RestaurantName", "FoodName"], observed=True)
              .agg(Res=("dont_receive", "size"), NoRecv=("dont_receive", "sum"),
                   **{f"m{i}": (f"f_{i}", "mean") for i in range(len(names))})
              .reset_index())
    cell["rho"] = cell.NoRecv / cell.Res
    cell = cell[cell.Res >= 10]
    print(f"  {len(cell):,} سلول با ≥۱۰ رزرو")
    rows = []
    for i, name in enumerate(names):
        c = cell[[f"m{i}", "rho", "Res"]].dropna()
        r = float(np.corrcoef(c[f"m{i}"], c.rho)[0, 1])
        # pinball در τ=۰.۲۰ با پیش‌بینی خودِ فیچر (کالیبره‌نشده — فقط رتبه‌بندی نسبی)
        tau = 0.20
        e = c.rho - c[f"m{i}"]
        pin = float(np.average(np.maximum(tau * e, (tau - 1) * e), weights=c.Res))
        rows.append({"فیچر": name, "همبستگی با ρ سلول": r, "pinball@۰.۲۰": pin, "n سلول": len(c)})
    cl = pd.DataFrame(rows).sort_values("همبستگی با ρ سلول", ascending=False)
    show(cl)
    print("\n--- Markdown ---\n" + md_table(auc.merge(cl, on="فیچر"), "{:.5f}"))

    header("۱۴-۶-۴ · ۳ · ⭐ تجزیه‌ی واریانس چهارجزئی — بازآزمون F59")
    c2 = (s.groupby(["date_gregorian", "meal", "RestaurantName", "FoodName"], observed=True)
            .agg(Res=("dont_receive", "size"), NoRecv=("dont_receive", "sum"),
                 comp=("f_0", "mean")).reset_index())
    c2 = c2[c2.Res >= 10].copy()
    c2["rho"] = c2.NoRecv / c2.Res
    y = c2.rho.to_numpy()
    tot = float(np.var(y))
    samp = float(np.mean(y * (1 - y) / c2.Res.to_numpy()))
    day = (c2.date_gregorian.astype(str) + "|" + c2.meal.astype(str)).to_numpy()

    def _r2(x_cols):
        X = np.column_stack([np.ones(len(c2))] + x_cols)
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        return 1 - float(np.var(y - X @ b)) / tot

    comp_only = _r2([c2.comp.to_numpy()])
    day_only = 1 - float(np.var(y - pd.Series(y).groupby(day).transform("mean").to_numpy())) / tot
    r_day = y - pd.Series(y).groupby(day).transform("mean").to_numpy()
    Xc = np.column_stack([np.ones(len(c2)), c2.comp.to_numpy()])
    bc, *_ = np.linalg.lstsq(Xc, r_day, rcond=None)
    comp_after_day = 1 - float(np.var(r_day - Xc @ bc)) / float(np.var(r_day))
    dec = pd.DataFrame([
        {"جزء": "نویز نمونه‌گیری", "سهم": samp / tot},
        {"جزء": "ترکیب رزروکنندگان (تنها)", "سهم": comp_only},
        {"جزء": "اثر ثابت روز×وعده (تنها)", "سهم": day_only},
        {"جزء": "ترکیب، **پس از** اثر روز", "سهم": comp_after_day * (1 - day_only)},
        {"جزء": "باقی‌مانده", "سهم": 1 - day_only - comp_after_day * (1 - day_only)},
    ])
    show(dec)
    print("\n--- Markdown ---\n" + md_table(dec, "{:.4f}"))
    print("   [F59 روی v2: شوک روز ۸۳.۲٪ · ترکیب ۹.۸٪ · ΔR² ترکیب +۰.۰۰۲۴]")
    print(f"   ΔR² ترکیب پس از اثر روز = {comp_after_day*(1-day_only):+.4f}")

    header("۴ · بازآزمون F62 — آیا برهم‌کنش فرد×روز ضرب‌شونده است؟")
    c2["day_shock"] = pd.Series(y).groupby(day).transform("mean").to_numpy()
    q = pd.qcut(c2.comp, 5, labels=["۱ کم‌ریسک", "۲", "۳", "۴", "۵ پرریسک"])
    dq = pd.qcut(c2.day_shock, 5, labels=["۱ روز خوب", "۲", "۳", "۴", "۵ روز بد"])
    piv = c2.assign(q=q, dq=dq).pivot_table(index="q", columns="dq", values="rho",
                                            aggfunc="mean", observed=True)
    show(piv)
    lo, hi = piv.iloc[0], piv.iloc[-1]
    print(f"\n  شکاف پرریسک−کم‌ریسک در روز خوب: {hi.iloc[0]-lo.iloc[0]:+.4f}")
    print(f"  شکاف پرریسک−کم‌ریسک در روز بد  : {hi.iloc[-1]-lo.iloc[-1]:+.4f}")
    print(f"  نسبت شکاف‌ها = {(hi.iloc[-1]-lo.iloc[-1])/(hi.iloc[0]-lo.iloc[0]):.2f}×"
          "   (ضرب‌شونده ⇒ >۱ · جمع‌شونده ⇒ ≈۱)")

    header("۱۴-۶-۳ · ۵ · پویایی کوهورت — تِنور و ورود")
    first = d.groupby("PersonId").agg(first_seq=("meal_seq", "min"), last_seq=("meal_seq", "max"),
                                      n=("meal_seq", "size"), rate=("dont_receive", "mean"))
    lo_edge = 30
    first["سانسور چپ"] = first.first_seq <= lo_edge
    print(f"  {len(first):,} فرد · سانسور چپ (رزرو در ۳۰ فرصت اول): "
          f"{int(first['سانسور چپ'].sum()):,} ({first['سانسور چپ'].mean():.1%})")
    d2 = d.merge(first[["first_seq", "سانسور چپ"]], left_on="PersonId", right_index=True)
    d2["tenure"] = ((d2.meal_seq - d2.first_seq) / 24).astype(int)  # ~ماه (۲۴ فرصت ≈ ۱۲ روز سرویس)
    prof = (d2[~d2["سانسور چپ"]].groupby("tenure")
              .agg(رزرو=("dont_receive", "size"), فرد=("PersonId", "nunique"),
                   ρ=("dont_receive", "mean")))
    prof = prof[prof.رزرو >= 2000].head(25)
    prof["نسبت"] = prof.ρ / base
    show(prof, n=30)
    print("\n--- Markdown ---\n" + md_table(prof.reset_index(), "{:.4f}"))

    header("۶ · تازه‌واردها در برابر باقی — آیا «تازه‌وارد بودن» فیچر است؟", 2)
    newbie = d2[(~d2["سانسور چپ"]) & (d2.meal_seq - d2.first_seq <= 24)]
    rest = d2[(~d2["سانسور چپ"]) & (d2.meal_seq - d2.first_seq > 24)]
    print(f"  ۲۴ فرصت اولِ هر فرد تازه‌وارد: ρ={newbie.dont_receive.mean():.4f} "
          f"({len(newbie):,} رزرو)")
    print(f"  بقیه‌ی رزروهای همان افراد   : ρ={rest.dont_receive.mean():.4f} "
          f"({len(rest):,} رزرو)")
    print(f"  نسبت = {newbie.dont_receive.mean()/rest.dont_receive.mean():.3f}×")
    print(f"\n  ⚠️ افراد با سانسور چپ (تِنور نامعلوم): ρ="
          f"{d2[d2['سانسور چپ']].dont_receive.mean():.4f}")

    header("۷ · نرخ خروج و اثرش بر ترکیب", 2)
    end = int(d.meal_seq.max())
    first["خارج‌شده"] = first.last_seq < end - 60
    ch = first.groupby("خارج‌شده").agg(فرد=("n", "size"), میانه_رزرو=("n", "median"),
                                       ρ=("rate", "mean"))
    show(ch)


if __name__ == "__main__":
    main()
