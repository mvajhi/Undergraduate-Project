"""بند ۴.۳.۳ — پیگیری: آیا «نبود فصلی‌بودن هفتگی» واقعی است یا محصول درون‌یابی؟

**چرا این پیگیری لازم شد.** ACF استاندارد روی سری روزانه‌ی درون‌یابی‌شده دو نتیجه داد
که هر دو با انتظار WBS ناسازگارند:
- lag 7/14/21/28 هیچ‌کدام معنادار نیستند (انتظار: قله‌ی هفتگی)،
- در عوض lag 1 = ۰.۸۴ که برای یک *نرخ* روزانه غیرعادی بالاست.

هر دو می‌توانند مصنوع باشند: ۴۰ روز از ۱۸۲ روز (۲۲٪) سرو نداشته‌اند و با درون‌یابی
خطی پر شده‌اند. درون‌یابی خطی ذاتاً همبستگی lag کوتاه می‌سازد (نقاط پرشده روی خط
مستقیم بین دو مشاهده‌اند) و هم‌زمان الگوی هفتگی را می‌شوید — به‌خصوص اینجا که روزهای
گمشده **تصادفی نیستند**: عمدتاً جمعه‌ها، تعطیلات و بلوک نوروز/رمضان‌اند، یعنی گمشدگی
خودش با روز هفته همبسته است.

**روش این پیگیری:** ACF «آگاه به شکاف» — همبستگی در تأخیر k فقط روی جفت‌هایی محاسبه
می‌شود که **هر دو سرِ جفت واقعاً مشاهده شده‌اند** (نه درون‌یابی‌شده). هیچ مقدار ساختگی
وارد محاسبه نمی‌شود؛ در عوض n هر تأخیر متفاوت است و کنار ضریب گزارش می‌شود.

اجرا: `python -m src.eda_lib.runners.s03b_acf_gapaware`
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import acf

from src.eda_lib.runners._common import header, kv, load_dataset, setup
from src.eda_lib.timeseries_helpers import daily_university_series


def gap_aware_acf(s: pd.Series, max_lag: int = 30) -> pd.DataFrame:
    """ACF فقط روی جفت‌های واقعاً مشاهده‌شده (بدون هیچ درون‌یابی)."""
    s = s.copy()
    rows = []
    for k in range(1, max_lag + 1):
        a, b = s.values[:-k], s.values[k:]
        m = ~(np.isnan(a) | np.isnan(b))
        n = int(m.sum())
        if n < 10:
            rows.append({"lag": k, "n_pairs": n, "r": np.nan, "p": np.nan})
            continue
        r, p = stats.pearsonr(a[m], b[m])
        rows.append({"lag": k, "n_pairs": n, "r": r, "p": p})
    return pd.DataFrame(rows)


def main() -> None:
    setup()
    df = load_dataset()
    daily = daily_university_series(df)

    header("مقایسه‌ی ACF درون‌یابی‌شده در برابر ACF آگاه به شکاف")
    kv("روزهای تقویمی", len(daily))
    kv("روزهای دارای سرو", int(daily["rho"].notna().sum()))
    kv("روزهای درون‌یابی‌شده", f"{int(daily['is_interpolated'].sum())} ({daily['is_interpolated'].mean():.1%})")

    print("\nآیا گمشدگی تصادفی است؟ توزیع روزهای بدون سرو بر حسب روز هفته:")
    miss = daily.copy()
    miss["dow_fa"] = (miss.index.dayofweek + 2) % 7
    names = {0: "شنبه", 1: "یکشنبه", 2: "دوشنبه", 3: "سه‌شنبه", 4: "چهارشنبه", 5: "پنجشنبه", 6: "جمعه"}
    tab = miss.groupby("dow_fa").agg(کل=("rho", "size"), بدون_سرو=("is_interpolated", "sum"))
    tab["نسبت_بدون_سرو"] = (tab["بدون_سرو"] / tab["کل"]).round(3)
    tab.index = tab.index.map(names)
    print(tab.to_string())
    chi2, p, _, _ = stats.chi2_contingency(
        pd.crosstab(miss["dow_fa"], miss["is_interpolated"]))
    print(f"\nChi-square استقلالِ «روز هفته» و «بدون سرو بودن»: chi2={chi2:.1f} p={p:.3g}")
    print("→ گمشدگی MCAR نیست؛ با روز هفته همبسته است، پس درون‌یابی مستقیماً الگوی هفتگی را تحریف می‌کند."
          if p < 0.05 else "→ گمشدگی با روز هفته مستقل است.")

    interp = acf(daily["rho_interp"].dropna(), nlags=30, fft=False)
    ga = gap_aware_acf(daily["rho"], 30)
    ci_i = 1.96 / np.sqrt(int(daily["rho_interp"].notna().sum()))

    print("\nlag | ACF درون‌یابی | ACF آگاه‌به‌شکاف (n جفت) | p")
    for k in [1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 15, 20, 21, 22, 28]:
        row = ga.loc[ga.lag == k].iloc[0]
        mark_i = "*" if abs(interp[k]) > ci_i else " "
        mark_g = "*" if row["p"] < 0.05 else " "
        print(f"{k:3d} | {interp[k]:+.4f}{mark_i}      | {row['r']:+.4f}{mark_g} (n={int(row['n_pairs'])})"
              f"      | {row['p']:.3g}")

    header("همان تحلیل، اما به تفکیک وعده (سری‌های بدون شکاف مصنوعی)", 2)
    for meal in ["lunch", "dinner"]:
        sub = df[df.Meal == meal]
        d = (sub.groupby("date_gregorian").apply(
            lambda x: x["NoRecv"].sum() / x["Res"].sum(), include_groups=False))
        d.index = pd.to_datetime(d.index)
        full = d.reindex(pd.date_range(d.index.min(), d.index.max(), freq="D"))
        g = gap_aware_acf(full, 30)
        sig7 = g.loc[g.lag == 7].iloc[0]
        sig1 = g.loc[g.lag == 1].iloc[0]
        sig14 = g.loc[g.lag == 14].iloc[0]
        print(f"\n{meal}: روزهای سرو={int(full.notna().sum())} از {len(full)} روز تقویمی")
        print(f"  lag 1 : r={sig1['r']:+.4f} p={sig1['p']:.3g} (n={int(sig1['n_pairs'])})")
        print(f"  lag 7 : r={sig7['r']:+.4f} p={sig7['p']:.3g} (n={int(sig7['n_pairs'])})")
        print(f"  lag 14: r={sig14['r']:+.4f} p={sig14['p']:.3g} (n={int(sig14['n_pairs'])})")
        top = g.dropna().nlargest(5, "r")[["lag", "r", "p", "n_pairs"]]
        print("  پنج تأخیر با بیشترین همبستگی:")
        print("   " + top.round(4).to_string(index=False).replace("\n", "\n   "))

    header("همان تحلیل در سطح (سلف×وعده) — واحدی که مدل واقعاً روی آن کار می‌کند", 2)
    # هر (سلف، وعده، روز) دو غذا دارد، پس اول باید به یک ردیف در روز تجمیع شود
    # (نرخ وزنی حجم)، وگرنه ایندکس تاریخ تکراری می‌شود.
    daily_rm = (df.groupby(["RestaurantName", "Meal", "date_gregorian"])
                  .apply(lambda x: x["NoRecv"].sum() / x["Res"].sum(), include_groups=False)
                  .rename("rho").reset_index())
    res = []
    for (r, m), g in daily_rm.groupby(["RestaurantName", "Meal"]):
        if len(g) < 60:
            continue
        s = g.set_index("date_gregorian").sort_index()["rho"]
        s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"))
        ga2 = gap_aware_acf(s, 21)
        res.append({"سلف": r, "وعده": m, "n": len(g),
                    "lag1": ga2.loc[ga2.lag == 1, "r"].iloc[0],
                    "lag7": ga2.loc[ga2.lag == 7, "r"].iloc[0],
                    "lag14": ga2.loc[ga2.lag == 14, "r"].iloc[0]})
    rdf = pd.DataFrame(res)
    print(rdf.round(3).to_string(index=False))
    print(f"\nمیانه‌ی lag1={rdf['lag1'].median():+.3f} · lag7={rdf['lag7'].median():+.3f} "
          f"· lag14={rdf['lag14'].median():+.3f}  (روی {len(rdf)} سری)")
    print(f"سهم سری‌هایی که lag7 > lag1 دارند: {(rdf['lag7'] > rdf['lag1']).mean():.1%}")


if __name__ == "__main__":
    main()
