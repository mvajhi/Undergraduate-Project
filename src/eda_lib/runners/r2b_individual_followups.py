"""دور ۲ (ب) — پاسخ به سؤال‌های باز سطح فردِ دور ۱ (از `doc/eda/round1/S8_individual_level.md`).

- Q8: منشأ اختلاف +۲.۳٪ و ۳۸۸ گروه اضافه‌ی فایل فردی چیست؟
- Q9 (H16): آیا جمعیت cold-start رفتاری نزدیک میانگین دارد یا خوشه‌ی خاص؟
- Q10: کدام سطح بیشترین سهم را در بیش‌پراکندگی ۲۱.۸ برابری دارد — روز، سلف، یا خوابگاه؟
  (تعیین‌کننده‌ی واحد بوت‌استرپ خوشه‌ای در فاز ۶)
- Q11: آیا اثر «ساکن خوابگاه» پس از کنترل دانشکده/مقطع/جنسیت باقی می‌ماند؟

اجرا: `python -m src.eda_lib.runners.r2b_individual_followups`
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from src.eda_lib.group_test_helpers import cliffs_delta
from src.eda_lib.runners._common import PERSON_DIM_PATH, header, kv, load_dataset, pct, setup
from src.eda_lib.runners.s13_individual import load_fact


def q8_source_of_gap(fact: pd.DataFrame) -> None:
    header("Q8 — منشأ اختلاف +۲.۳٪ و ۳۸۸ گروه اضافه‌ی فایل فردی")
    agg = load_dataset()
    ind = fact.groupby(["date_gregorian", "Meal", "restaurant_canonical"], observed=True).agg(
        Res_ind=("Count", "sum"), n_rows=("Count", "size")).reset_index()
    aggr = agg.groupby(["date_gregorian", "Meal", "RestaurantName"]).agg(
        Res_agg=("Res", "sum")).reset_index()
    m = aggr.merge(ind, left_on=["date_gregorian", "Meal", "RestaurantName"],
                   right_on=["date_gregorian", "Meal", "restaurant_canonical"],
                   how="outer", indicator=True)

    only_ind = m[m["_merge"] == "right_only"].copy()
    kv("گروه‌های فقط-فردی", len(only_ind))
    kv("مجموع Res این گروه‌ها", f"{only_ind['Res_ind'].sum():,.0f}")
    kv("سهم از کل Res فردی", pct(only_ind["Res_ind"].sum() / ind["Res_ind"].sum()))
    print("\nتوزیع تاریخی این گروه‌ها (۱۰ تاریخ پرتکرار):")
    print(only_ind["date_gregorian"].dt.strftime("%Y-%m-%d").value_counts().head(10).to_string())
    print("\nسلف‌های درگیر:")
    print(only_ind["restaurant_canonical"].value_counts().head(12).to_string())
    print("\nوعده:")
    print(only_ind["Meal"].value_counts().to_string())
    print("\nتوزیع اندازه‌ی این گروه‌ها:")
    print(only_ind["Res_ind"].describe().round(1).to_string())

    both = m[m["_merge"] == "both"].copy()
    both["diff"] = both["Res_ind"] - both["Res_agg"]
    big = both[both["diff"].abs() > 20].sort_values("diff", ascending=False)
    print(f"\nگروه‌های منطبق با اختلاف >۲۰ پرس: {len(big)} از {len(both)}")
    print(big.head(10)[["date_gregorian", "Meal", "RestaurantName", "Res_agg", "Res_ind", "diff"]]
          .to_string(index=False))
    print("\nآیا اختلاف در تاریخ خاصی متمرکز است؟")
    print(big["date_gregorian"].dt.strftime("%Y-%m").value_counts().head(8).to_string())


def q9_cold_start(fact: pd.DataFrame) -> None:
    header("Q9 (H16) — رفتار جمعیت cold-start در برابر کل")
    per = fact.groupby("PersonId").agg(n=("dont_receive", "size"), rate=("dont_receive", "mean"))
    bins = pd.cut(per["n"], [0, 4, 9, 29, 99, 10000],
                  labels=["۱-۴ رزرو", "۵-۹", "۱۰-۲۹", "۳۰-۹۹", "۱۰۰+"])
    tab = per.groupby(bins, observed=True).agg(
        افراد=("rate", "size"), نرخ_میانگین=("rate", "mean"), نرخ_میانه=("rate", "median"),
        انحراف=("rate", "std"))
    tab["سهم_رزروها"] = per.groupby(bins, observed=True)["n"].sum() / per["n"].sum()
    print(tab.round(4).to_string())
    overall = fact["dont_receive"].mean()
    kv("\nنرخ کل (وزنی)", f"{overall:.4f}")
    cold = per.loc[per["n"] < 10, "rate"]
    warm = per.loc[per["n"] >= 30, "rate"]
    u, p = stats.mannwhitneyu(cold, warm)
    print(f"cold-start (<۱۰ رزرو، n={len(cold)}) نرخ میانگین={cold.mean():.4f}")
    print(f"باسابقه (≥۳۰ رزرو، n={len(warm)}) نرخ میانگین={warm.mean():.4f}")
    print(f"Mann-Whitney p={p:.3g} · Cliff's δ={cliffs_delta(cold.values, warm.values):+.3f}")
    print("\n→ H16 می‌گوید cold-start باید نزدیک میانگین کل باشد، نه یک خوشه‌ی خاص.")
    print("⚠️ توجه: نرخ افراد کم‌رزرو ذاتاً پرواریانس است (مخرج کوچک) — انحراف معیار را ببینید.")


def q10_variance_decomposition(fact: pd.DataFrame, dim: pd.DataFrame) -> None:
    header("Q10 — بیش‌پراکندگی ۲۱.۸ برابری از کدام سطح می‌آید؟")
    print("روش: واریانس نرخ گروهی در سطوح مختلف تجمیع، نسبت به واریانس دوجمله‌ای متناظر.\n")
    p_bar = fact["dont_receive"].mean()
    levels = {
        "روز × وعده × سلف (پایه)": ["date_gregorian", "Meal", "restaurant_canonical"],
        "روز × وعده (کل دانشگاه)": ["date_gregorian", "Meal"],
        "سلف × وعده (در طول زمان)": ["restaurant_canonical", "Meal"],
        "روز (همه‌ی وعده‌ها)": ["date_gregorian"],
    }
    for name, keys in levels.items():
        g = fact.groupby(keys, observed=True).agg(n=("dont_receive", "size"), k=("dont_receive", "sum"))
        g = g[g["n"] >= 30]
        if len(g) < 10:
            continue
        g["p_hat"] = g["k"] / g["n"]
        obs = g["p_hat"].var()
        exp = (p_bar * (1 - p_bar) / g["n"]).mean()
        chi2 = (((g["k"] - g["n"] * p_bar) ** 2) / (g["n"] * p_bar * (1 - p_bar))).sum()
        print(f"  {name:<28s} گروه={len(g):>5d}  نسبت={obs / exp:>6.2f}×  chi²/df={chi2 / (len(g) - 1):>6.2f}")

    header("تجزیه‌ی واریانس با مدل اثرات: هر عامل چقدر از تغییرپذیری را می‌گیرد؟", 2)
    g = fact.groupby(["date_gregorian", "Meal", "restaurant_canonical"], observed=True).agg(
        n=("dont_receive", "size"), k=("dont_receive", "sum")).reset_index()
    g = g[g["n"] >= 30].copy()
    g["rate"] = g["k"] / g["n"]
    g["dow"] = (g["date_gregorian"].dt.dayofweek + 2) % 7
    base = smf.ols("rate ~ 1", data=g).fit()
    for name, f in [
        ("+ سلف", "rate ~ C(restaurant_canonical)"),
        ("+ سلف + وعده", "rate ~ C(restaurant_canonical) + C(Meal)"),
        ("+ سلف + وعده + روزهفته", "rate ~ C(restaurant_canonical) + C(Meal) + C(dow)"),
        ("+ سلف + وعده + روزهفته + تاریخ", "rate ~ C(restaurant_canonical) + C(Meal) + C(dow) + C(date_gregorian)"),
    ]:
        m = smf.ols(f, data=g).fit()
        resid_var = m.resid.var()
        exp = (p_bar * (1 - p_bar) / g["n"]).mean()
        print(f"  {name:<34s} R²={m.rsquared:.4f}  واریانس باقیمانده/دوجمله‌ای={resid_var / exp:>6.2f}×")
    print("\n→ هر مقدار از این نسبت که پس از افزودن همه‌ی عوامل باقی بماند، بیش‌پراکندگی")
    print("  توضیح‌ناپذیر است و باید در محاسبه‌ی کوانتایل (فاز ۶) لحاظ شود.")

    print("\nICC در سطح روز (آیا روزها منبع اصلی همبستگی‌اند؟):")
    for key, lbl in [("date_gregorian", "روز"), ("restaurant_canonical", "سلف")]:
        groups = [x["rate"].values for _, x in g.groupby(key, observed=True) if len(x) >= 5]
        if len(groups) < 3:
            continue
        k_bar = np.mean([len(x) for x in groups])
        grand = np.concatenate(groups).mean()
        msb = sum(len(x) * (x.mean() - grand) ** 2 for x in groups) / (len(groups) - 1)
        msw = sum(((x - x.mean()) ** 2).sum() for x in groups) / (sum(len(x) for x in groups) - len(groups))
        icc = (msb - msw) / (msb + (k_bar - 1) * msw)
        print(f"  ICC({lbl}) = {icc:+.4f}  ({len(groups)} گروه)")


def q11_dorm_controlled(fact: pd.DataFrame, dim: pd.DataFrame) -> None:
    header("Q11 — آیا اثر «ساکن خوابگاه» پس از کنترل دانشکده/مقطع/جنسیت باقی می‌ماند؟")
    per = fact.groupby("PersonId").agg(n=("dont_receive", "size"), rate=("dont_receive", "mean"),
                                       is_tehran=("is_tehran", "mean"))
    d = per.join(dim.set_index("PersonId"), how="left")
    d = d[(d["n"] >= 10) & d["is_dorm_resident"].notna() & d["CollegeName"].notna()].copy()
    d["is_dorm"] = d["is_dorm_resident"].astype(int)
    d["is_grad"] = d["DegreeName"].astype(str).str.contains("ارشد|دکتری|PhD", na=False).astype(int)
    d["is_female"] = (d["Gender"].astype(str).str.strip() == "زن").astype(int)
    d["teh"] = (d["is_tehran"] > 0.5).astype(int)
    kv("پایه‌ی تحلیل", len(d))

    for name, f in [
        ("خوابگاه تنها", "rate ~ is_dorm"),
        ("+ جنسیت + مقطع", "rate ~ is_dorm + is_female + is_grad"),
        ("+ شهر", "rate ~ is_dorm + is_female + is_grad + teh"),
        ("+ دانشکده (اثر ثابت)", "rate ~ is_dorm + is_female + is_grad + teh + C(CollegeName)"),
    ]:
        m = smf.ols(f, data=d).fit()
        print(f"  {name:<24s} R²={m.rsquared:.4f}  ضریب خوابگاه="
              f"{m.params['is_dorm']:+.5f} (p={m.pvalues['is_dorm']:.3g})")
    print("\n→ اگر ضریب پس از افزودن اثر ثابت دانشکده هم معنادار و هم‌علامت بماند،")
    print("  «ساکن خوابگاه بودن» اطلاعات مستقل دارد و فیچر معتبر مدل B است.")

    print("\nمقایسه‌ی درون‌دانشکده‌ای (فقط دانشکده‌هایی که هر دو گروه را دارند، ≥۳۰ نفر در هر گروه):")
    rows = []
    for col, gg in d.groupby("CollegeName"):
        a = gg.loc[gg.is_dorm == 1, "rate"]
        b = gg.loc[gg.is_dorm == 0, "rate"]
        if len(a) >= 30 and len(b) >= 30:
            u, p = stats.mannwhitneyu(a, b)
            rows.append({"دانشکده": col, "n_خوابگاهی": len(a), "n_غیر": len(b),
                         "نرخ_خوابگاهی": a.mean(), "نرخ_غیر": b.mean(),
                         "تفاوت": a.mean() - b.mean(), "p": p})
    res = pd.DataFrame(rows).sort_values("تفاوت")
    print(res.round(4).to_string(index=False))
    print(f"\nدانشکده‌هایی که نرخ خوابگاهی‌شان کمتر است: {(res['تفاوت'] < 0).sum()} از {len(res)}")
    w = stats.wilcoxon(res["نرخ_خوابگاهی"], res["نرخ_غیر"])
    print(f"Wilcoxon زوجی روی {len(res)} دانشکده: p={w.pvalue:.3g}")


def main() -> None:
    setup()
    fact = load_fact()
    dim = pd.read_csv(PERSON_DIM_PATH)
    q8_source_of_gap(fact)
    q9_cold_start(fact)
    q10_variance_decomposition(fact, dim)
    q11_dorm_controlled(fact, dim)


if __name__ == "__main__":
    main()
