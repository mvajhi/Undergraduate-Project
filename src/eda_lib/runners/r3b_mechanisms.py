"""دور ۳ (ب) — سازوکارهای پنهان و پل بین مدل A و B (فقط تهران).

دور ۳ الف نشان داد عدم‌دریافت عمدتاً **صفت** است (میانگین بلندمدت AUC=۰.۷۲، تازگی
تقریباً چیزی اضافه نمی‌کند). حالا سؤال بعدی: *این صفت از کجا می‌آید و چه چیزی آن را
در یک روز مشخص فعال می‌کند؟* پنج سازوکار کاندید آزمون می‌شوند:

- Q14 **«غیبت از دانشگاه»**: اگر فرد در یک روز هم ناهار و هم شام رزرو کرده باشد، آیا
  سرنوشتشان به هم گره خورده است؟ اگر بله، متغیر پنهان «حضور فیزیکی» است نه «تصمیم
  درباره‌ی غذا».
- Q15 **ترجیح غذایی آشکارشده**: آیا افراد روی غذاهای خاصی بیشتر غیبت می‌کنند؟
- Q16 **قدرت عادت**: آیا فردِ منظم (الگوی رزرو قابل‌پیش‌بینی) کمتر غیبت می‌کند؟
- Q17 **سیگنال تعهد**: قیمت، مخلفات و تعداد پرس.
- Q18 **⭐ پل A↔B — اثر ترکیب رزروکنندگان**: رزرو ۷۲ ساعت زودتر بسته می‌شود، پس در
  لحظه‌ی برش **دقیقاً می‌دانیم چه کسانی رزرو کرده‌اند**. اگر میانگین تاریخچه‌ی همین
  افراد، نرخ آن وعده را پیش‌بینی کند، یک فیچر کاملاً مجاز و بسیار قوی برای مدل A داریم.

اجرا: `python -m src.eda_lib.runners.r3b_mechanisms`
"""

import numpy as np
import pandas as pd
from scipy import stats

from src.eda_lib.runners._common import PERSON_DIM_PATH, header, kv, pct, setup
from src.eda_lib.runners.s13_individual import load_fact


def load_tehran():
    fact = load_fact()
    fact = fact[fact["is_tehran"]].copy()
    dim = pd.read_csv(PERSON_DIM_PATH)
    return fact, dim


def q14_same_day_coupling(fact: pd.DataFrame) -> None:
    header("Q14 — «غیبت از دانشگاه»: آیا ناهار و شام یک روزِ یک فرد به هم گره خورده‌اند؟")
    d = (fact.groupby(["PersonId", "date_gregorian", "Meal"], observed=True)["dont_receive"]
              .max().unstack("Meal"))
    both = d.dropna(subset=["lunch", "dinner"])
    kv("روز-فردهایی با هر دو وعده", f"{len(both):,}")
    l, dn = both["lunch"].astype(bool), both["dinner"].astype(bool)
    kv("P(غیبت ناهار)", f"{l.mean():.4f}")
    kv("P(غیبت شام)", f"{dn.mean():.4f}")
    kv("P(غیبت شام | غیبت ناهار)", f"{dn[l].mean():.4f}")
    kv("P(غیبت شام | حضور ناهار)", f"{dn[~l].mean():.4f}")
    rr = dn[l].mean() / dn[~l].mean()
    kv("نسبت خطر", f"{rr:.2f}")
    kv("P(هر دو غیبت)", f"{(l & dn).mean():.4f}")
    kv("P(هر دو) اگر مستقل بودند", f"{l.mean() * dn.mean():.4f}")
    kv("نسبت مشاهده/مستقل", f"{(l & dn).mean() / (l.mean() * dn.mean()):.2f}×")
    phi = stats.pearsonr(l.astype(float), dn.astype(float))
    kv("همبستگی phi", f"{phi[0]:+.4f} (p={phi[1]:.3g})")

    print("\nکنترل حیاتی: آیا این فقط تفاوت بین افراد است یا واقعاً «روز» است؟")
    b2 = both.copy()
    b2["l"] = l.astype(float); b2["d"] = dn.astype(float)
    b2["l_r"] = b2["l"] - b2.groupby("PersonId")["l"].transform("mean")
    b2["d_r"] = b2["d"] - b2.groupby("PersonId")["d"].transform("mean")
    r_within = stats.pearsonr(b2["l_r"], b2["d_r"])
    kv("همبستگی درون‌فردی (اثر ثابت فرد حذف‌شده)", f"{r_within[0]:+.4f} (p={r_within[1]:.3g})")
    print("→ اگر درون‌فردی هم قوی بماند، متغیر پنهان «حضور فیزیکی آن روز» است،")
    print("  نه «سلیقه‌ی غذایی» یا «نوع شخصیت». این یعنی غیبت یک رخداد *روزانه* است.")


def q15_food_preference(fact: pd.DataFrame) -> None:
    header("Q15 — ترجیح غذایی آشکارشده: آیا غذا بر تصمیم فرد اثر دارد؟")
    f = fact.dropna(subset=["food_canonical"]).copy()
    f["p_mean"] = f.groupby("PersonId")["dont_receive"].transform("mean")
    f["resid"] = f["dont_receive"].astype(float) - f["p_mean"]

    g = f.groupby("food_canonical").agg(n=("resid", "size"), نرخ_خام=("dont_receive", "mean"),
                                        باقیمانده=("resid", "mean"))
    g = g[g["n"] >= 2000].sort_values("باقیمانده")
    g["se"] = f.groupby("food_canonical")["resid"].sem()
    print("۸ غذایی که مردم بیشتر از عادت خودشان تحویل *می‌گیرند* (باقیمانده منفی):")
    print(g.head(8).round(4).to_string())
    print("\n۸ غذایی که مردم بیشتر از عادت خودشان غیبت می‌کنند:")
    print(g.tail(8).round(4).to_string())
    kv("\nدامنه‌ی اثر غذا (باقیمانده)", f"{g['باقیمانده'].max() - g['باقیمانده'].min():.4f}")
    groups = [x["resid"].values for _, x in f.groupby("food_canonical") if len(x) >= 2000]
    h, p = stats.kruskal(*groups)
    kv("Kruskal-Wallis بین غذاها (روی باقیمانده)", f"H={h:.1f} p={p:.3g}")

    print("\nآیا ترجیح غذایی *شخصی* است؟ (آیا فرد X روی غذای Y سازگارانه غیبت می‌کند؟)")
    pf = f.groupby(["PersonId", "food_canonical"], observed=True)["dont_receive"].agg(["mean", "size"])
    pf = pf[pf["size"] >= 4]
    kv("جفت‌های (فرد، غذا) با ≥۴ مشاهده", f"{len(pf):,}")
    var_between_pf = pf.groupby("PersonId")["mean"].var().mean()
    kv("میانگین واریانس نرخ فرد بین غذاهای مختلف", f"{var_between_pf:.5f}")
    # آزمون: تقسیم مشاهدات هر جفت به دو نیمه و بررسی همبستگی
    f2 = f.sort_values("date_gregorian").copy()
    f2["half"] = f2.groupby(["PersonId", "food_canonical"], observed=True).cumcount()
    sz = f2.groupby(["PersonId", "food_canonical"], observed=True)["dont_receive"].transform("size")
    f2 = f2[sz >= 6]
    f2["is_second"] = f2["half"] >= (sz[f2.index] / 2)
    sp = (f2.groupby(["PersonId", "food_canonical", "is_second"], observed=True)["dont_receive"]
            .mean().unstack("is_second").dropna())
    if len(sp) > 100:
        r = stats.pearsonr(sp[False], sp[True])
        kv("همبستگی نیمه‌اول/نیمه‌دوم جفت (فرد،غذا)", f"{r[0]:+.4f} (n={len(sp)}, p={r[1]:.3g})")
        # مقایسه با پایداری خودِ فرد (بدون تفکیک غذا)
        pf2 = f2.groupby(["PersonId", "is_second"], observed=True)["dont_receive"].mean().unstack("is_second").dropna()
        r2 = stats.pearsonr(pf2[False], pf2[True])
        kv("همبستگی نیمه‌اول/نیمه‌دوم خودِ فرد", f"{r2[0]:+.4f} (n={len(pf2)})")
        print("→ اگر همبستگی جفت (فرد،غذا) از پایداری خودِ فرد بیشتر نباشد،")
        print("  «ترجیح غذایی شخصی» چیزی فراتر از «صفت شخصی» اضافه نمی‌کند.")


def q16_habit_regularity(fact: pd.DataFrame) -> None:
    header("Q16 — قدرت عادت: آیا فرد منظم کمتر غیبت می‌کند؟")
    f = fact.copy()
    f["dow"] = (f["date_gregorian"].dt.dayofweek + 2) % 7

    def _entropy(s):
        p = s.value_counts(normalize=True).values
        return float(-(p * np.log(p + 1e-12)).sum())

    per = f.groupby("PersonId").agg(n=("dont_receive", "size"), rate=("dont_receive", "mean"),
                                    n_days=("date_gregorian", "nunique"))
    per = per[per["n"] >= 30]
    ent_dow = f[f.PersonId.isin(per.index)].groupby("PersonId")["dow"].apply(_entropy)
    ent_rest = f[f.PersonId.isin(per.index)].groupby("PersonId")["restaurant_canonical"].apply(_entropy)
    per = per.join(ent_dow.rename("entropy_dow")).join(ent_rest.rename("entropy_restaurant"))
    # شدت استفاده: چند رزرو در هر روزِ حضور
    per["intensity"] = per["n"] / per["n_days"]
    kv("افراد در تحلیل", len(per))

    for col, lbl in [("entropy_dow", "آنتروپی روز هفته (بی‌نظمی)"),
                     ("entropy_restaurant", "آنتروپی سلف (پراکندگی مکانی)"),
                     ("n", "تعداد کل رزرو (شدت استفاده)"),
                     ("intensity", "رزرو در هر روزِ حضور")]:
        r = stats.spearmanr(per[col], per["rate"])
        print(f"  {lbl:<38s} Spearman با نرخ = {r[0]:+.4f}  (p={r[1]:.3g})")

    print("\nنرخ به تفکیک چارک بی‌نظمی روز هفته:")
    q = pd.qcut(per["entropy_dow"], 4, labels=["Q1 منظم‌ترین", "Q2", "Q3", "Q4 بی‌نظم‌ترین"], duplicates="drop")
    print(per.groupby(q, observed=True).agg(افراد=("rate", "size"), نرخ=("rate", "mean"),
                                            رزرو=("n", "mean")).round(4).to_string())
    print("\nنرخ به تفکیک چارک تعداد رزرو:")
    q2 = pd.qcut(per["n"], 4, labels=["Q1 کم‌مصرف", "Q2", "Q3", "Q4 پرمصرف"], duplicates="drop")
    print(per.groupby(q2, observed=True).agg(افراد=("rate", "size"), نرخ=("rate", "mean"),
                                             بینظمی=("entropy_dow", "mean")).round(4).to_string())


def q17_commitment_signals(fact: pd.DataFrame) -> None:
    header("Q17 — سیگنال‌های تعهد: قیمت، مخلفات، تعداد پرس")
    from src.eda_lib.runners._common import PERSON_FACT_PATH
    f = pd.read_csv(
        PERSON_FACT_PATH,
        usecols=["PersonId", "date_gregorian", "Meal", "is_tehran", "is_main_meal",
                 "dont_receive", "Price", "has_extras", "Count", "food_canonical"],
        parse_dates=["date_gregorian"])
    f = f[(f["is_tehran"] == True) & (f["is_main_meal"] == True)].copy()
    f["p_mean"] = f.groupby("PersonId")["dont_receive"].transform("mean")
    f["resid"] = f["dont_receive"].astype(float) - f["p_mean"]

    print("### مخلفات (دوغ/نوشابه/سالاد همراه غذا)")
    t = f.groupby("has_extras").agg(n=("dont_receive", "size"), نرخ_خام=("dont_receive", "mean"),
                                    باقیمانده=("resid", "mean"))
    print(t.round(4).to_string())
    a = f.loc[f.has_extras == True, "resid"]; b = f.loc[f.has_extras == False, "resid"]
    if len(a) > 100 and len(b) > 100:
        u, p = stats.mannwhitneyu(a, b)
        kv("Mann-Whitney روی باقیمانده", f"p={p:.3g} · تفاوت میانگین={a.mean() - b.mean():+.5f}")

    print("\n### قیمت پرداختی")
    kv("مقادیر یکتای قیمت", f.Price.nunique())
    t = f.groupby("Price").agg(n=("dont_receive", "size"), نرخ=("dont_receive", "mean"),
                               باقیمانده=("resid", "mean"))
    print(t[t["n"] >= 3000].round(4).to_string())
    m = f["Price"].notna()
    r = stats.spearmanr(f.loc[m, "Price"], f.loc[m, "resid"])
    kv("Spearman(قیمت، باقیمانده)", f"{r[0]:+.4f} (p={r[1]:.3g})")

    print("\n### تعداد پرس در یک رزرو")
    t = f.groupby("Count").agg(n=("dont_receive", "size"), نرخ=("dont_receive", "mean"),
                               باقیمانده=("resid", "mean"))
    print(t[t["n"] >= 500].round(4).to_string())


def q18_composition_bridge(fact: pd.DataFrame) -> None:
    header("Q18 — ⭐ پل A↔B: آیا «ترکیب رزروکنندگان» نرخ آن وعده را پیش‌بینی می‌کند؟")
    print("قاعده‌ی برش: رزرو ۷۲ ساعت زودتر بسته می‌شود ⇒ در لحظه‌ی تصمیم دقیقاً می‌دانیم")
    print("چه کسانی رزرو کرده‌اند. پس «میانگین تاریخچه‌ی این افراد» یک فیچر مجاز است.\n")

    f = fact.sort_values(["PersonId", "date_gregorian"]).copy()
    # میانگین انبساطی سختگیرانه: فقط رزروهای *قبلی* همان فرد
    csum = f.groupby("PersonId")["dont_receive"].cumsum() - f["dont_receive"]
    cnt = f.groupby("PersonId").cumcount()
    f["hist_rate"] = np.where(cnt > 0, csum / cnt.replace(0, np.nan), np.nan)
    f["has_hist"] = cnt >= 5

    cell = f.groupby(["date_gregorian", "Meal", "restaurant_canonical"], observed=True).agg(
        n=("dont_receive", "size"),
        actual=("dont_receive", "mean"),
        predicted=("hist_rate", "mean"),
        cover=("has_hist", "mean")).reset_index()
    cell = cell[(cell["n"] >= 30) & (cell["cover"] >= 0.7) & cell["predicted"].notna()]
    kv("سلول‌های (روز، وعده، سلف) واجد شرایط", f"{len(cell):,}")

    r_p = stats.pearsonr(cell["predicted"], cell["actual"])
    r_s = stats.spearmanr(cell["predicted"], cell["actual"])
    kv("Pearson(ترکیب پیش‌بینی‌شده، نرخ واقعی)", f"{r_p[0]:+.4f} (p={r_p[1]:.3g})")
    kv("Spearman", f"{r_s[0]:+.4f}")
    kv("R²", f"{r_p[0] ** 2:.4f}")
    kv("MAE", f"{(cell['predicted'] - cell['actual']).abs().mean():.4f}")
    kv("اریبی (میانگین پیش‌بینی منهای واقعی)", f"{(cell['predicted'] - cell['actual']).mean():+.4f}")

    print("\n### مقایسه با خط پایه‌های ساده (روی همان سلول‌ها)")
    import statsmodels.formula.api as smf
    cell["dow"] = (cell["date_gregorian"].dt.dayofweek + 2) % 7
    base = {
        "میانگین کل (ثابت)": "actual ~ 1",
        "سلف": "actual ~ C(restaurant_canonical)",
        "سلف + وعده + روزهفته": "actual ~ C(restaurant_canonical) + C(Meal) + C(dow)",
        "**فقط ترکیب رزروکنندگان**": "actual ~ predicted",
        "ترکیب + سلف + وعده + روزهفته": "actual ~ predicted + C(restaurant_canonical) + C(Meal) + C(dow)",
    }
    for name, fo in base.items():
        m = smf.ols(fo, data=cell).fit()
        print(f"  {name:<34s} R²={m.rsquared:.4f}  R²adj={m.rsquared_adj:.4f}")

    print("\n### آیا ترکیب، *تغییرات روزانه* را هم می‌گیرد یا فقط تفاوت بین سلف‌ها؟")
    cell["a_res"] = cell["actual"] - cell.groupby(["restaurant_canonical", "Meal"], observed=True)["actual"].transform("mean")
    cell["p_res"] = cell["predicted"] - cell.groupby(["restaurant_canonical", "Meal"], observed=True)["predicted"].transform("mean")
    r_w = stats.pearsonr(cell["p_res"], cell["a_res"])
    kv("همبستگی درون (سلف×وعده)", f"{r_w[0]:+.4f} (p={r_w[1]:.3g})")
    print("→ اگر این هم معنادار بماند، ترکیب رزروکنندگان اطلاعات *پویا* دارد،")
    print("  نه فقط بازتاب هویت ثابت سلف.")
    cell.to_csv("/tmp/claude-1000/-home-mvajhi-code-new-project/"
                "4b42a398-d5e8-407f-82dc-3b3b666126de/scratchpad/composition_cells.csv", index=False)


def q21_social_contagion(fact: pd.DataFrame, dim: pd.DataFrame) -> None:
    header("Q21 — سرایت اجتماعی: آیا هم‌خوابگاهی‌ها در یک روز با هم غیبت می‌کنند؟")
    d = dim.set_index("PersonId")["dorm_canonical"]
    f = fact.join(d, on="PersonId").dropna(subset=["dorm_canonical"])
    f = f[f["dorm_canonical"].astype(str).str.contains("خوابگاه", na=False)]
    kv("رزروهای ساکنان خوابگاه", f"{len(f):,}")

    # باقیمانده پس از حذف اثر فرد و اثر (روز×وعده×سلف)
    f = f.copy()
    f["r1"] = f["dont_receive"].astype(float) - f.groupby("PersonId")["dont_receive"].transform("mean")
    cellmean = f.groupby(["date_gregorian", "Meal", "restaurant_canonical"], observed=True)["r1"].transform("mean")
    f["r2"] = f["r1"] - cellmean

    print("ICC باقیمانده در سطح (خوابگاه × روز × وعده) — پس از حذف اثر فرد و اثر سلف-روز:")
    for col, lbl, resid in [("r1", "قبل از حذف اثر سلف-روز", "r1"), ("r2", "پس از حذف اثر سلف-روز", "r2")]:
        g = f.groupby(["dorm_canonical", "date_gregorian", "Meal"], observed=True)[resid].agg(["mean", "size"])
        g = g[g["size"] >= 10]
        if len(g) < 50:
            continue
        msb = g["size"].mul(g["mean"] ** 2).sum() / (len(g) - 1)
        within = f.merge(g["mean"].rename("gm"), left_on=["dorm_canonical", "date_gregorian", "Meal"],
                         right_index=True, how="inner")
        msw = ((within[resid] - within["gm"]) ** 2).sum() / (len(within) - len(g))
        k_bar = g["size"].mean()
        icc = (msb - msw) / (msb + (k_bar - 1) * msw)
        print(f"  {lbl:<28s} گروه={len(g):>5d}  ICC={icc:+.4f}")
    print("→ ICC مثبتِ باقی‌مانده پس از حذف اثر سلف-روز یعنی خوابگاه یک منبع همبستگی")
    print("  *مستقل* است (مثلاً سرویس ایاب‌وذهاب، برنامه‌ی مشترک، یا هم‌رفتاری اجتماعی).")


def main() -> None:
    setup()
    fact, dim = load_tehran()
    header(f"دور ۳ (ب) — تهران: {len(fact):,} رزرو · {fact['PersonId'].nunique():,} فرد")
    q14_same_day_coupling(fact)
    q15_food_preference(fact)
    q16_habit_regularity(fact)
    q17_commitment_signals(fact)
    q18_composition_bridge(fact)
    q21_social_contagion(fact, dim)


if __name__ == "__main__":
    main()
