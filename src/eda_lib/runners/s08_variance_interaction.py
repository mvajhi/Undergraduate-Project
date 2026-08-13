"""بند ۴.۸ (ناهم‌واریانسی) و ۴.۹ (تحلیل تعامل) — دور ۱ روی `dataset_v2`.

بند ۴.۸ خروجی **تصمیمی** دارد: بین سه گزینه‌ی وزن‌دهی نمونه ($w=Res$)، مدل
دوجمله‌ای/Beta-Binomial، و کوچک‌سازی بیزی تجربی کدام؟ برای پاسخ، واریانس مشاهده‌شده با
واریانس نظری دوجمله‌ای $p(1-p)/Res$ مقایسه می‌شود — اگر واریانس واقعی به‌مراتب بیشتر
باشد (overdispersion)، مدل دوجمله‌ای ساده کافی نیست.

اجرا: `python -m src.eda_lib.runners.s08_variance_interaction`
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

from src.config import FIGURES_DIR
from src.eda_lib.correlation_helpers import (
    empirical_bayes_shrink,
    fit_beta_moments_overdispersion,
    theoretical_binomial_variance,
)
from src.eda_lib.figio import save_fig
from src.eda_lib.runners._common import header, kv, load_dataset, setup
from src.viz_fa import fa
from src.viz_fa import setup as viz_setup


def run_heteroscedasticity(df: pd.DataFrame) -> pd.DataFrame:
    header("۴.۸ ناهم‌واریانسی — واریانس مشاهده‌شده در برابر واریانس نظری دوجمله‌ای")
    p_bar = df["NoRecv"].sum() / df["Res"].sum()
    kv("p̄ (نرخ وزنی کل)", f"{p_bar:.4f}")

    df = df.copy()
    df["res_bin"] = pd.cut(df["Res"], bins=[0, 20, 50, 100, 200, 400, 800, 2000],
                           labels=["<20", "20-50", "50-100", "100-200", "200-400", "400-800", ">800"])
    tab = df.groupby("res_bin", observed=True).apply(lambda g: pd.Series({
        "n": len(g),
        "Res_median": g["Res"].median(),
        "var_observed": g["rho"].var(),
        "var_binomial": theoretical_binomial_variance(p_bar, np.array([g["Res"].median()]))[0],
    }), include_groups=False)
    tab["نسبت مشاهده/نظری"] = tab["var_observed"] / tab["var_binomial"]
    print(tab.round(6).to_string())
    print("\n→ نسبت ۱.۰ یعنی تمام واریانس از نمونه‌گیری دوجمله‌ای است؛ نسبت >۱ یعنی "
          "بیش‌پراکندگی واقعی (تفاوت ذاتی بین رکوردها) هم وجود دارد.")

    header("آزمون‌های رسمی ناهم‌واریانسی", 2)
    d = df.dropna(subset=["rho", "Res"]).copy()
    d["log_res"] = np.log(d["Res"])
    model = smf.ols("rho ~ log_res + C(Meal) + C(RestaurantType)", data=d).fit()
    bp = sm.stats.diagnostic.het_breuschpagan(model.resid, model.model.exog)
    wh = sm.stats.diagnostic.het_white(model.resid, model.model.exog)
    print(f"Breusch-Pagan: LM={bp[0]:.2f}  p={bp[1]:.3g}")
    print(f"White        : LM={wh[0]:.2f}  p={wh[1]:.3g}")
    q = pd.qcut(d["Res"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    lev = stats.levene(*[d.loc[q == lv, "rho"].values for lv in q.cat.categories], center="median")
    print(f"Levene بین چارک‌های Res: stat={lev.statistic:.2f} p={lev.pvalue:.3g}")

    header("برازش Beta با تصحیح بیش‌پراکندگی و کوچک‌سازی بیزی تجربی", 2)
    alpha, beta = fit_beta_moments_overdispersion(df["NoRecv"].values, df["Res"].values)
    kv("α برآوردشده", f"{alpha:.4f}")
    kv("β برآوردشده", f"{beta:.4f}")
    kv("میانگین ضمنی α/(α+β)", f"{alpha / (alpha + beta):.4f}")
    kv("«اندازه‌ی نمونه‌ی پیشین» معادل (α+β)", f"{alpha + beta:.1f} رزرو")
    shrunk = empirical_bayes_shrink(df["NoRecv"].values, df["Res"].values, alpha, beta)
    df["rho_shrunk"] = shrunk
    print(f"\nاثر کوچک‌سازی: std(ρ) {df['rho'].std():.4f} → std(ρ_shrunk) {df['rho_shrunk'].std():.4f}")
    print(f"بیشترین جابه‌جایی در رکوردهای کوچک:")
    df["shift"] = (df["rho_shrunk"] - df["rho"]).abs()
    top = df.nlargest(6, "shift")[["DateReserve", "RestaurantName", "Res", "rho", "rho_shrunk", "shift"]]
    print(top.round(4).to_string(index=False))
    print("\nمیانگین |جابه‌جایی| به تفکیک اندازه‌ی رزرو:")
    print(df.groupby("res_bin", observed=True)["shift"].agg(["mean", "max"]).round(4).to_string())

    print("\nهمبستگی Res با ρ، قبل و بعد از کوچک‌سازی (باید کاهش یابد):")
    print(f"  خام    : Spearman={stats.spearmanr(df['Res'], df['rho'])[0]:+.4f}")
    print(f"  کوچک‌شده: Spearman={stats.spearmanr(df['Res'], df['rho_shrunk'])[0]:+.4f}")

    viz_setup()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.loglog(tab["Res_median"], tab["var_observed"], "o-", label=fa("واریانس مشاهده‌شده"), color="#C44E52")
    ax.loglog(tab["Res_median"], tab["var_binomial"], "s--", label=fa("واریانس نظری دوجمله‌ای"), color="#4C72B0")
    ax.set_xlabel(fa("اندازه‌ی رزرو (میانه‌ی هر دسته)")); ax.set_ylabel(fa("واریانس نرخ عدم‌دریافت"))
    ax.set_title(fa("واریانس نرخ با بزرگ‌شدن رزرو افت می‌کند اما بیش از حد دوجمله‌ای باقی می‌ماند"))
    ax.legend(); ax.grid(alpha=.3, which="both")
    fig.tight_layout()
    print("\n" + str(save_fig(fig, "4.8_variance_vs_res", FIGURES_DIR)))
    plt.close(fig)
    return df


def run_interactions(df: pd.DataFrame) -> None:
    header("۴.۹ تحلیل تعامل")
    d = df.copy()
    print("جدول محوری ρ وزنی — سلف‌نوع × روز هفته:")
    piv = d.pivot_table(index="RestaurantType", columns="dow_name",
                        values=["NoRecv", "Res"], aggfunc="sum")
    print((piv["NoRecv"] / piv["Res"]).round(4).to_string())

    print("\nρ وزنی — وعده × نوع غذا:")
    piv = d.pivot_table(index="Meal", columns="FoodType", values=["NoRecv", "Res"], aggfunc="sum")
    print((piv["NoRecv"] / piv["Res"]).round(4).to_string())

    print("\nρ وزنی — شهر × وعده:")
    piv = d.pivot_table(index="city", columns="Meal", values=["NoRecv", "Res"], aggfunc="sum")
    print((piv["NoRecv"] / piv["Res"]).round(4).to_string())

    header("آزمون رسمی: ANOVA دوطرفه با/بدون جمله‌ی تعامل (مقایسه‌ی AIC)", 2)
    d["log_res"] = np.log(d["Res"])
    tests = [
        ("Meal × RestaurantType", "rho ~ C(Meal) + C(RestaurantType)", "rho ~ C(Meal) * C(RestaurantType)"),
        ("Meal × FoodType", "rho ~ C(Meal) + C(FoodType)", "rho ~ C(Meal) * C(FoodType)"),
        ("dow × RestaurantType", "rho ~ C(DayOfWeek) + C(RestaurantType)", "rho ~ C(DayOfWeek) * C(RestaurantType)"),
        ("city × Meal", "rho ~ C(city) + C(Meal)", "rho ~ C(city) * C(Meal)"),
        ("dow × city", "rho ~ C(DayOfWeek) + C(city)", "rho ~ C(DayOfWeek) * C(city)"),
    ]
    rows = []
    for name, f0, f1 in tests:
        m0, m1 = smf.ols(f0, data=d).fit(), smf.ols(f1, data=d).fit()
        # آزمون F تودرتو (نه compare_lr_test که برای OLS با scale نامعلوم NaN می‌دهد)
        av = sm.stats.anova_lm(m0, m1)
        p_f = av["Pr(>F)"].iloc[-1]
        rows.append({"تعامل": name, "AIC بدون": m0.aic, "AIC با": m1.aic,
                     "ΔAIC": m0.aic - m1.aic, "p (آزمون F تودرتو)": p_f,
                     "R² بدون": m0.rsquared, "R² با": m1.rsquared})
    res = pd.DataFrame(rows).sort_values("ΔAIC", ascending=False)
    print(res.round(4).to_string(index=False))
    print("\n→ ΔAIC مثبت یعنی مدل *با* تعامل بهتر است (AIC کمتر). ΔAIC>10 شواهد قوی.")

    header("اثر دما به تفکیک وعده (تعامل دما×وعده، فقط تهران)", 2)
    from src.eda_lib.runners._common import load_weather
    w = load_weather().drop(columns=["province", "date_jalali"], errors="ignore")
    dt = d.merge(w, on=["city", "date_gregorian"], how="left")
    teh = dt[dt.city == "تهران"].dropna(subset=["temp_min"])
    teh = teh.assign(temp_bin=pd.qcut(teh["temp_min"], 4, labels=["سرد", "خنک", "معتدل", "گرم"]))
    piv = teh.pivot_table(index="temp_bin", columns="Meal", values=["NoRecv", "Res"],
                          aggfunc="sum", observed=True)
    print((piv["NoRecv"] / piv["Res"]).round(4).to_string())


def main() -> None:
    setup()
    df = load_dataset()
    df = run_heteroscedasticity(df)
    run_interactions(df)


if __name__ == "__main__":
    main()
