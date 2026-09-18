"""۱۴-۵-۱ تجزیه‌ی چندفصلی · ۱۴-۵-۲ کدام تقویم · ۱۴-۵-۶ ریتم دوهفتگی.

اجرا: ``python -m src.eda_lib.runners.r7e_seasonal``
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import set_global_seed
from src.eda_lib.runners._common_v4 import DOW_FA, header, load_features, md_table, show
from src.eda_lib.seasonal_v4 import (
    _ols_r2, calendar_positions, fourier_terms, incremental_variance,
)

K_HARMONICS = 4


def _design(f: pd.DataFrame, pos: pd.DataFrame) -> dict[str, np.ndarray]:
    """بلوک‌های ماتریس طرح، به ترتیبِ اعلام‌شده‌ی ورود."""
    n = len(f)
    t = (f.date_gregorian - f.date_gregorian.min()).dt.days.to_numpy() / 365.0
    dow = pd.get_dummies(f.day_of_week, drop_first=True).to_numpy(dtype=float)
    meal = pd.get_dummies(f.meal, drop_first=True).to_numpy(dtype=float)
    return {
        "وعده": meal,
        "روند (چندجمله‌ای ۲)": np.column_stack([t, t ** 2]),
        "هفتگی (روز هفته)": dow,
        "سالانه (۴ هارمونیک شمسی)": fourier_terms(pos.pos_jalali.to_numpy(), K_HARMONICS, "j").to_numpy(),
        "نیم‌سالی (۳ هارمونیک تحصیلی)": fourier_terms(pos.pos_acad.to_numpy(), 3, "a").to_numpy(),
        "تقویمی (پرچم‌ها)": f[["is_holiday_any", "is_ramadan", "is_exam_period",
                               "is_nowruz_block", "is_in_session"]].astype(float).to_numpy(),
    }


def main() -> None:
    set_global_seed()
    f = load_features(columns=["date_gregorian", "meal", "restaurant", "dish", "Res", "NoRecv",
                               "academic_year", "day_of_week", "month_jalali",
                               "day_of_month_jalali", "year_jalali", "is_holiday_any",
                               "is_ramadan", "is_exam_period", "is_nowruz_block", "is_in_session",
                               "venue_type"])
    f["rho"] = f.NoRecv / f.Res

    # سری L3 (روز×وعده) — سطحی که مؤلفه‌ی فصلی روی آن معنا دارد
    l3 = (f.groupby(["date_gregorian", "meal"], observed=True)
            .agg(Res=("Res", "sum"), NoRecv=("NoRecv", "sum"),
                 day_of_week=("day_of_week", "first"), month_jalali=("month_jalali", "first"),
                 day_of_month_jalali=("day_of_month_jalali", "first"),
                 academic_year=("academic_year", "first"),
                 **{c: (c, "first") for c in ["is_holiday_any", "is_ramadan", "is_exam_period",
                                              "is_nowruz_block", "is_in_session"]})
            .reset_index())
    l3["rho"] = l3.NoRecv / l3.Res
    pos = calendar_positions(l3)
    l3["academic_year"] = pos["academic_year_filled"].to_numpy()

    header("۱۴-۵-۱ · ۱ · تجزیه‌ی مؤلفه‌ها روی سری L3 — سهم واریانس افزایشی")
    print(f"  {len(l3):,} نقطه (روز×وعده) · {l3.date_gregorian.nunique()} روز · "
          f"وزن‌دهی با Res (F153)")
    blocks = _design(l3, pos)
    iv = incremental_variance(l3.rho.to_numpy(), blocks, l3.Res.to_numpy().astype(float))
    show(iv)
    print("\n--- Markdown ---\n" + md_table(iv, "{:.4f}"))
    print("\n  ⚠️ ترتیب ورود اعلام‌شده است و اهمیت دارد: مؤلفه‌ها متعامد نیستند.")

    header("۲ · همان تجزیه، به تفکیک سال — آیا سهم مؤلفه‌ها پایدار است؟")
    parts = []
    for yr, g in l3.groupby("academic_year"):
        if len(g) < 100:
            continue
        p = calendar_positions(g)
        r = incremental_variance(g.rho.to_numpy(), _design(g, p), g.Res.to_numpy().astype(float))
        r.insert(0, "سال", yr)
        parts.append(r)
    show(pd.concat(parts, ignore_index=True), n=30)

    header("۱۴-۵-۲ · ۳ · ⭐ کدام تقویم؟ — برازش در برابر انتقال‌پذیری")
    y = l3.rho.to_numpy()
    w = l3.Res.to_numpy().astype(float)
    base = np.column_stack([np.ones(len(l3)),
                            pd.get_dummies(l3.meal, drop_first=True).to_numpy(dtype=float),
                            pd.get_dummies(l3.day_of_week, drop_first=True).to_numpy(dtype=float)])
    yrs = sorted(l3.academic_year.unique())
    Fj = fourier_terms(pos.pos_jalali.to_numpy(), K_HARMONICS, "j").to_numpy()
    Fg = fourier_terms(pos.pos_greg.to_numpy(), K_HARMONICS, "g").to_numpy()
    Fa = fourier_terms(pos.pos_acad.to_numpy(), K_HARMONICS, "a").to_numpy()

    def ridge_oos(X, alpha=1e-3):
        """انتقال‌پذیری: آموزش روی سال $t$، ارزیابی روی $t+1$ و $t+2$.

        ⚠️ ریج مقیاس‌شده اجباری است، نه تزئین: ترکیب دو پایه‌ی فصلی داخل **یک سال
        آموزش** عدد شرطی $10^6$ می‌دهد و OLS خالص ضرایب منفجرشده تولید می‌کند
        (R² برون‌نمونه‌ای منفیِ چندمیلیونی در نسخه‌ی اول همین بند).
        """
        sc = []
        for i, ytr in enumerate(yrs[:-1]):
            m = (l3.academic_year == ytr).to_numpy()
            sw = np.sqrt(w[m])
            A, b = X[m] * sw[:, None], y[m] * sw
            P = np.eye(X.shape[1]) * alpha * float(np.trace(A.T @ A)) / X.shape[1]
            P[0, 0] = 0
            beta = np.linalg.solve(A.T @ A + P, A.T @ b)
            for yte in yrs[i + 1:]:
                t = (l3.academic_year == yte).to_numpy()
                pred = X[t] @ beta
                yb = np.average(y[t], weights=w[t])
                sc.append(1 - float(np.sum(w[t] * (y[t] - pred) ** 2))
                          / float(np.sum(w[t] * (y[t] - yb) ** 2)))
        return float(np.mean(sc)), float(np.min(sc))

    rows = []
    for name, X in [("بدون فصل", base), ("شمسی", np.hstack([base, Fj])),
                    ("میلادی", np.hstack([base, Fg])), ("تحصیلی", np.hstack([base, Fa])),
                    ("شمسی + تحصیلی", np.hstack([base, Fj, Fa])),
                    ("هر سه", np.hstack([base, Fj, Fa, Fg]))]:
        r2_in, _ = _ols_r2(X, y, w)
        m, mn = ridge_oos(X)
        rows.append({"ترکیب": name, "ستون": X.shape[1], "R² درون‌نمونه": r2_in,
                     "R² انتقال (میانگین)": m, "R² انتقال (کمینه)": mn})
    cal = pd.DataFrame(rows)
    show(cal)
    print("\n--- Markdown ---\n" + md_table(cal, "{:.4f}"))

    header("۴ · چرا «هر سه را می‌گذاریم» ممکن نیست — هم‌خطی ساختاری", 2)
    def recon_r2(B, Ref):
        X = np.hstack([np.ones((len(Ref), 1)), Ref])
        out = []
        for i in range(B.shape[1]):
            beta, *_ = np.linalg.lstsq(X, B[:, i], rcond=None)
            r = B[:, i] - X @ beta
            out.append(1 - float(np.sum(r ** 2)) / float(np.sum((B[:, i] - B[:, i].mean()) ** 2)))
        return np.array(out)

    rg = recon_r2(Fg, Fj)
    print(f"  بازسازی پایه‌ی **میلادی** از پایه‌ی **شمسی** روی کل پنجره: "
          f"میانه R²={np.median(rg):.5f} · کمینه={rg.min():.5f}")
    print("  ⇒ دو تقویم خورشیدی فقط **جابه‌جایی فاز**اند؛ پایه‌ی فوریه نسبت به شیفت هم‌ارز است.")
    print("\n  و درون **یک سال آموزش** (جایی که انتقال‌پذیری سنجیده می‌شود):")
    for ytr in yrs[:-1]:
        m = (l3.academic_year == ytr).to_numpy()
        ra = recon_r2(Fa[m], Fj[m])
        sv = np.linalg.svd(np.hstack([Fj[m], Fa[m]]) - np.hstack([Fj[m], Fa[m]]).mean(0),
                           compute_uv=False)
        print(f"    {ytr}: عدد شرطی(شمسی|تحصیلی)={sv[0]/sv[-1]:12.1f} · "
              f"بازسازی تحصیلی از شمسی: میانه R²={np.median(ra):.4f}")
    print("  ⇒ داخل یک سال، موقعیت تحصیلی و موقعیت شمسی **یک چرخه‌ی واحد**اند که یک‌بار"
          " پیموده می‌شود. تفکیکشان با یک سال آموزش ساختاراً ناممکن است.")

    header("۱۴-۵-۶ · ۵ · ریتم دوهفتگی F34 — طیف‌سنجی و آزمون جایگشتی")
    from scipy.signal import lombscargle

    lunch = f[f.meal == "lunch"].copy()
    r = lunch.rho.to_numpy().astype(float)
    for key in [(lunch.restaurant.astype(str) + "|" + lunch.dish.astype(str)).to_numpy(),
                lunch.day_of_week.astype(str).to_numpy()]:
        r = r - pd.Series(r).groupby(key).transform("mean").to_numpy()
    lunch["resid"] = r
    daily = (lunch.groupby(["date_gregorian", "academic_year"])
                  .apply(lambda g: np.average(g.resid, weights=g.Res), include_groups=False)
                  .rename("resid").reset_index())
    print(f"  {len(daily)} روز ناهار · باقی‌مانده پس از حذف اثر ثابت سلف×غذا و روز هفته")
    print("  ⚠️ Lomb-Scargle انتخاب شد نه FFT: سری نامنظم است و FFT درون‌یابی می‌خواهد (ممنوع، F35).")

    periods = np.arange(2.0, 60.0, 0.25)
    freqs = 2 * np.pi / periods
    band = (periods >= 13) & (periods <= 15)
    rng = np.random.default_rng(42)

    def perm_test(g, label):
        t = (g.date_gregorian - g.date_gregorian.min()).dt.days.to_numpy().astype(float)
        v = g.resid.to_numpy() - g.resid.mean()
        obs = float(lombscargle(t, v, freqs, normalize=True)[band].max())
        null = [float(lombscargle(t, rng.permutation(v), freqs, normalize=True)[band].max())
                for _ in range(500)]
        p = (1 + sum(n >= obs for n in null)) / (len(null) + 1)
        best = float(periods[int(np.argmax(lombscargle(t, v, freqs, normalize=True)))])
        print(f"  {label:14s} n={len(g):3d}  توان۱۳–۱۵={obs:.4f}  میانه‌ی تهی={np.median(null):.4f}  "
              f"p={p:.4f}  قوی‌ترین دوره={best:5.1f} روز" + ("  ✅" if p < 0.05 else "  ❌"))

    perm_test(daily, "کل پنجره")
    for yr, g in daily.dropna(subset=["academic_year"]).groupby("academic_year"):
        if len(g) > 80:
            perm_test(g, str(yr))

    header("۶ · آیا خودِ چرخه‌ی منو بلندتر شده؟ (فرضیه‌ی رقیب)", 2)
    lu = lunch.dropna(subset=["academic_year"]).sort_values("date_gregorian")
    for yr, g in lu.groupby("academic_year"):
        gaps = g.groupby(["restaurant", "dish"])["date_gregorian"].diff().dt.days.dropna()
        gaps = gaps[gaps > 0]
        print(f"  {yr}: فاصله‌ی تکرار (سلف،غذا) میانه={gaps.median():5.1f} روز · "
              f"میانگین={gaps.mean():5.1f} · سهم ۱۳–۱۵ روزه={float(gaps.between(13,15).mean()):.1%}")
    print("  ⇒ اگر چرخه بلندتر شده بود، میانه باید رشد می‌کرد. نکرده.")


if __name__ == "__main__":
    main()
