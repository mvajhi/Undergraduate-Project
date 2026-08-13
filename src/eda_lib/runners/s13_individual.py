"""بند ۴.۱۳ — کاوش اکتشافی سطح فرد (ورودی مستقیم مدل B)، دور ۱ روی نسخه‌ی v2.

این بند به درخواست صریح ذی‌نفع پروژه عمیق‌تر از نسخه‌ی WBS اجرا می‌شود، با این استدلال
که «از داده‌ی فردی می‌توان داده‌ی کلی را هم درآورد» — یعنی هر یافته‌ی سطح تجمیعی
(بندهای ۴.۱ تا ۴.۱۰) باید در سطح فرد **سازوکارش** دیده شود:

- آیا نرخ کل، رفتار *همه* است یا رفتار *اقلیتی* از دانشجویان؟ (۴.۱۳.۳)
- «تهران بالاتر از پردیس‌های دیگر» ترکیب افراد است یا رفتار افراد؟
- «ناهار بالاتر از شام» در سطح فرد هم برقرار است یا اثر ترکیب جمعیت؟
- ⚠️ **فرض استقلال** (هشدار سطر ۸۶۲ WBS): آیا تصمیم افراد مستقل است؟ اگر هم‌خوابگاهی‌ها
  رفتار همبسته داشته باشند، واریانس واقعی مجموع بزرگ‌تر از پواسون-دوجمله‌ای است و
  کوانتایل محاسبه‌شده **خوش‌بینانه** می‌شود — یعنی ریسک کمبود غذا کم‌برآورد می‌شود.

اجرا: `python -m src.eda_lib.runners.s13_individual`
"""

import numpy as np
import pandas as pd
from scipy import stats

from src.eda_lib.group_test_helpers import cliffs_delta, fdr_correct
from src.eda_lib.individual_helpers import lorenz_curve, pareto_share
from src.eda_lib.runners._common import (
    PERSON_DIM_PATH,
    PERSON_FACT_PATH,
    boot_ci,
    header,
    kv,
    load_dataset,
    pct,
    setup,
)

FACT_COLS = ["PersonId", "date_gregorian", "Meal", "restaurant_canonical", "city",
             "is_tehran", "Count", "dont_receive", "food_canonical", "is_main_meal"]


def load_fact() -> pd.DataFrame:
    """فایل ۲.۱ میلیون‌ردیفی را فقط با ستون‌های لازم و dtype فشرده می‌خواند.

    `is_tehran` را نمی‌توان مستقیم bool خواند: دو ردیف `restaurant_canonical` گمشده
    دارند (نام سلفشان در فایل خام خالی بوده) و در نتیجه شهرشان هم NA است. این دو ردیف
    (از ۲٫۱ میلیون) کنار گذاشته می‌شوند.
    """
    df = pd.read_csv(PERSON_FACT_PATH, usecols=FACT_COLS, parse_dates=["date_gregorian"],
                     dtype={"PersonId": "int32", "Meal": "category",
                            "restaurant_canonical": "category", "city": "category",
                            "Count": "int16", "dont_receive": "bool"})
    n0 = len(df)
    df = df.dropna(subset=["restaurant_canonical", "is_tehran"])
    if len(df) < n0:
        print(f"[info] {n0 - len(df)} ردیف بدون نگاشت سلف/شهر کنار گذاشته شد (از {n0:,})")
    df["is_tehran"] = df["is_tehran"].astype(bool)
    # فقط ناهار و شام: واحد تحلیل پروژه همین دو وعده است و فایل تجمیعی هم فقط همین‌ها
    # را دارد. صبحانه و سحری (۲.۹٪ ردیف‌ها) کشف دور ۲ بودند و خارج از دامنه‌اند.
    n1 = len(df)
    df = df[df["is_main_meal"].astype(bool)]
    print(f"[info] {n1 - len(df):,} ردیف صبحانه/سحری کنار گذاشته شد → {len(df):,} ردیف ناهار/شام")
    return df.drop(columns=["is_main_meal"])


def run_history(fact: pd.DataFrame, dim: pd.DataFrame) -> pd.DataFrame:
    header("۴.۱۳.۱ توزیع تاریخچه‌ی فردی")
    per = fact.groupby("PersonId").agg(
        n_res=("dont_receive", "size"), n_noshow=("dont_receive", "sum"),
        n_days=("date_gregorian", "nunique"),
        first=("date_gregorian", "min"), last=("date_gregorian", "max"))
    per["rate"] = per["n_noshow"] / per["n_res"]
    per["tenure_days"] = (per["last"] - per["first"]).dt.days + 1
    kv("تعداد افراد یکتا", len(per))
    kv("کل رزروها", len(fact))
    print("\nتوزیع «تعداد رزرو هر فرد»:")
    print(per["n_res"].describe(percentiles=[.05, .25, .5, .75, .95, .99]).round(1).to_string())
    for thr in [1, 5, 10, 30]:
        n = int((per["n_res"] < thr).sum())
        share_res = fact["PersonId"].isin(per.index[per["n_res"] < thr]).mean()
        print(f"  افراد با کمتر از {thr:>2d} رزرو: {n:>6d} ({n / len(per):>5.1%} افراد، "
              f"{share_res:>5.1%} کل رزروها)  ← جمعیت cold-start")
    print("\nتوزیع «طول سابقه (روز)»:")
    print(per["tenure_days"].describe(percentiles=[.05, .25, .5, .75, .95]).round(1).to_string())
    print("\nتوزیع «تعداد روز یکتای حضور»:")
    print(per["n_days"].describe(percentiles=[.05, .25, .5, .75, .95]).round(1).to_string())
    return per


def run_imbalance(fact: pd.DataFrame, per: pd.DataFrame) -> None:
    header("۴.۱۳.۳ کمّی‌سازی عدم‌توازن — آیا نرخ کل رفتار همه است یا اقلیت؟")
    overall = fact["dont_receive"].mean()
    kv("نرخ عدم‌دریافت سطح رزرو منفرد", f"{overall:.4f}")
    kv("نرخ تجمیعی سطح (d,m,r,f) برای مقایسه", "0.0805 (ρ وزنی، S1)")

    print("\nتوزیع نرخ شخصی (فقط افراد با ≥۱۰ رزرو، تا نرخ‌های ۰/۱ ساختگی حذف شوند):")
    act = per[per["n_res"] >= 10]
    kv("  تعداد این افراد", f"{len(act)} ({len(act) / len(per):.1%} افراد)")
    print(act["rate"].describe(percentiles=[.1, .25, .5, .75, .9, .95, .99]).round(4).to_string())
    kv("  سهم افراد با نرخ صفر", pct((act["rate"] == 0).mean()))
    kv("  سهم افراد با نرخ >۰.۳", pct((act["rate"] > 0.3).mean()))

    header("منحنی لورنتس: چند درصد افراد چند درصد کل عدم‌دریافت را می‌سازند؟", 2)
    w = per["n_noshow"].values.astype(float)
    x, y, gini_vol = lorenz_curve(w)
    pt = pareto_share(w, [0.05, 0.10, 0.20, 0.50])
    for _, r in pt.iterrows():
        print(f"  {r['pct_people']:>4.0f}% پرتکرارترین افراد ({int(r['n_people']):>6d} نفر) "
              f"⇒ {r['pct_of_total_norecv']:>5.1f}% کل موارد عدم‌دریافت")
    kv("\nضریب جینی عدم‌دریافت بین افراد", f"{gini_vol:.4f}")
    print("(⚠️ بخشی از این نابرابری صرفاً به‌خاطر تفاوت *تعداد رزرو* افراد است، نه تفاوت *نرخ*)")

    print("\nهمان تحلیل، اما نرمال‌شده نسبت به تعداد رزرو (نابرابری نرخ، نه حجم):")
    w2 = act["rate"].values
    x2, y2, gini_rate = lorenz_curve(w2)
    kv("  ضریب جینی نرخ شخصی (افراد ≥۱۰ رزرو)", f"{gini_rate:.4f}")
    pt2 = pareto_share(w2, [0.10, 0.20])
    for _, r in pt2.iterrows():
        print(f"  {r['pct_people']:>4.0f}% بدترین افراد (بر اساس نرخ) "
              f"⇒ {r['pct_of_total_norecv']:>5.1f}% مجموع نرخ‌ها")

    header("آیا رفتار فردی در طول زمان **پایدار** است؟ (پیش‌شرط ارزش فیچر تاریخی)", 2)
    fact_s = fact.sort_values("date_gregorian")
    mid = fact_s["date_gregorian"].quantile(0.5)
    h1 = fact_s[fact_s["date_gregorian"] < mid].groupby("PersonId")["dont_receive"].agg(["mean", "size"])
    h2 = fact_s[fact_s["date_gregorian"] >= mid].groupby("PersonId")["dont_receive"].agg(["mean", "size"])
    both = h1.join(h2, lsuffix="_1", rsuffix="_2", how="inner")
    both = both[(both["size_1"] >= 10) & (both["size_2"] >= 10)]
    r_p = stats.pearsonr(both["mean_1"], both["mean_2"])
    r_s = stats.spearmanr(both["mean_1"], both["mean_2"])
    kv("افراد با ≥۱۰ رزرو در هر دو نیمه", len(both))
    print(f"همبستگی نرخ نیمه‌ی اول با نیمه‌ی دوم: Pearson={r_p[0]:+.4f} (p={r_p[1]:.3g}) · "
          f"Spearman={r_s[0]:+.4f}")
    q = pd.qcut(both["mean_1"], 5, labels=["Q1 بهترین", "Q2", "Q3", "Q4", "Q5 بدترین"], duplicates="drop")
    print("\nنرخ نیمه‌ی دوم بر حسب چارک نرخ نیمه‌ی اول:")
    print(both.assign(q=q).groupby("q", observed=True).agg(
        n=("mean_2", "size"), نرخ_نیمه۱=("mean_1", "mean"), نرخ_نیمه۲=("mean_2", "mean")).round(4).to_string())


def run_demographics(fact: pd.DataFrame, dim: pd.DataFrame, per: pd.DataFrame) -> None:
    header("۴.۱۳.۲ نرخ به تفکیک جمعیتی + آزمون‌های H13–H16")
    d = per.join(dim.set_index("PersonId"), how="left")
    act = d[d["n_res"] >= 10].copy()
    kv("پایه‌ی تحلیل (افراد با ≥۱۰ رزرو)", len(act))

    results = []
    for col, label in [("Gender", "جنسیت"), ("is_dorm_resident", "ساکن خوابگاه"),
                       ("DegreeName", "مقطع"), ("EducationSession", "نوع دوره")]:
        print(f"\n--- {label} ({col}) ---")
        g = act.groupby(col, observed=True).agg(
            افراد=("rate", "size"), نرخ_میانگین=("rate", "mean"), نرخ_میانه=("rate", "median"),
            رزرو_میانگین=("n_res", "mean"))
        g = g[g["افراد"] >= 30].sort_values("نرخ_میانگین", ascending=False)
        print(g.round(4).to_string())
        groups = [x["rate"].values for _, x in act.groupby(col, observed=True) if len(x) >= 30]
        if len(groups) == 2:
            u, p = stats.mannwhitneyu(*groups)
            delta = cliffs_delta(groups[0], groups[1])
            print(f"  Mann-Whitney: p={p:.3g}  Cliff's δ={delta:+.3f}")
            results.append({"متغیر": label, "آزمون": "Mann-Whitney", "p": p, "اثر": delta})
        elif len(groups) > 2:
            h, p = stats.kruskal(*groups)
            eta = (h - len(groups) + 1) / (len(act) - len(groups))
            print(f"  Kruskal-Wallis ({len(groups)} گروه): H={h:.1f} p={p:.3g}  η²={eta:.4f}")
            results.append({"متغیر": label, "آزمون": "Kruskal-Wallis", "p": p, "اثر": eta})

    header("H13 — خوابگاه: مقایسه‌ی *سطح فرد* (نه سطح سلف)", 2)
    dorm = act.loc[act["is_dorm_resident"] == True, "rate"]
    nond = act.loc[act["is_dorm_resident"] == False, "rate"]
    if len(dorm) >= 30 and len(nond) >= 30:
        u, p = stats.mannwhitneyu(dorm, nond)
        m1, l1, h1_ = boot_ci(dorm.values)
        m2, l2, h2_ = boot_ci(nond.values)
        print(f"  ساکن خوابگاه   : n={len(dorm):>6d} نرخ={m1:.4f} CI=[{l1:.4f},{h1_:.4f}]")
        print(f"  غیرساکن        : n={len(nond):>6d} نرخ={m2:.4f} CI=[{l2:.4f},{h2_:.4f}]")
        print(f"  p={p:.3g}  Cliff's δ={cliffs_delta(dorm.values, nond.values):+.3f}")
        print("\n  همان مقایسه اما **فقط داخل تهران** (کنترل هم‌آمیختگی شهر):")
        teh_ids = set(fact.loc[fact["is_tehran"], "PersonId"].unique())
        sub = act[act.index.isin(teh_ids)]
        a2 = sub.loc[sub["is_dorm_resident"] == True, "rate"]
        b2 = sub.loc[sub["is_dorm_resident"] == False, "rate"]
        u2, p2 = stats.mannwhitneyu(a2, b2)
        print(f"    خوابگاهی n={len(a2)} نرخ={a2.mean():.4f} · غیرخوابگاهی n={len(b2)} نرخ={b2.mean():.4f}"
              f" · p={p2:.3g} δ={cliffs_delta(a2.values, b2.values):+.3f}")

    header("H14 — مقطع تحصیلی: تحصیلات تکمیلی در برابر کارشناسی", 2)
    act["is_grad"] = act["DegreeName"].astype(str).str.contains("ارشد|دکتری|PhD", na=False)
    a = act.loc[act["is_grad"], "rate"]
    b = act.loc[~act["is_grad"], "rate"]
    u, p = stats.mannwhitneyu(a, b)
    print(f"  تحصیلات تکمیلی: n={len(a)} نرخ={a.mean():.4f} · کارشناسی/سایر: n={len(b)} نرخ={b.mean():.4f}")
    print(f"  p={p:.3g}  Cliff's δ={cliffs_delta(a.values, b.values):+.3f}")

    header("دانشکده — ۱۰ بالاترین و ۵ پایین‌ترین (فقط دانشکده‌های با ≥۱۰۰ فرد)", 2)
    col = act.groupby("CollegeName").agg(افراد=("rate", "size"), نرخ=("rate", "mean"),
                                         رزرو=("n_res", "sum"))
    col = col[col["افراد"] >= 100].sort_values("نرخ", ascending=False)
    print(col.head(10).round(4).to_string())
    print("...")
    print(col.tail(5).round(4).to_string())
    h, p = stats.kruskal(*[x["rate"].values for _, x in act.groupby("CollegeName")
                           if len(x) >= 100])
    print(f"\nKruskal-Wallis بین دانشکده‌ها: H={h:.1f} p={p:.3g}")

    if results:
        res = pd.DataFrame(results)
        res["p_bh"] = fdr_correct(res["p"].values)
        print("\nتصحیح BH روی آزمون‌های جمعیتی:")
        print(res.round(6).to_string(index=False))


def run_mechanism(fact: pd.DataFrame, per: pd.DataFrame, dim: pd.DataFrame) -> None:
    header("سازوکار یافته‌های سطح تجمیعی، دیده‌شده در سطح فرد")

    print("--- آیا «ناهار > شام» (H1) در سطح فرد هم برقرار است؟ ---")
    pm = fact.groupby(["PersonId", "Meal"], observed=True)["dont_receive"].agg(["mean", "size"]).reset_index()
    wide = pm.pivot(index="PersonId", columns="Meal", values=["mean", "size"])
    both = wide.dropna()
    both = both[(both[("size", "lunch")] >= 10) & (both[("size", "dinner")] >= 10)]
    a, b = both[("mean", "lunch")], both[("mean", "dinner")]
    w = stats.wilcoxon(a, b)
    print(f"  افرادی که هم ناهار و هم شام ≥۱۰ رزرو دارند: n={len(both)}")
    print(f"  نرخ ناهارشان={a.mean():.4f} · نرخ شامشان={b.mean():.4f} · "
          f"تفاوت={a.mean() - b.mean():+.4f}")
    print(f"  Wilcoxon زوجی (همان افراد، پس ترکیب جمعیت کنترل شده): p={w.pvalue:.3g}")
    print(f"  سهم افرادی که نرخ ناهارشان بیشتر است: {(a > b).mean():.1%}")
    print("  → اگر معنادار بماند، «ناهار>شام» اثر *وعده* است نه اثر ترکیب جمعیت.")

    print("\n--- آیا اثر شهر (H18) ترکیب افراد است یا رفتار افراد؟ ---")
    pc = fact.groupby("PersonId").agg(city=("city", lambda s: s.mode().iat[0] if len(s) else np.nan),
                                      rate=("dont_receive", "mean"), n=("dont_receive", "size"))
    pc = pc[pc["n"] >= 10]
    g = pc.groupby("city", observed=True).agg(افراد=("rate", "size"), نرخ_میانگین=("rate", "mean"),
                                              نرخ_میانه=("rate", "median"))
    print(g[g["افراد"] >= 30].round(4).to_string())
    groups = [x["rate"].values for _, x in pc.groupby("city", observed=True) if len(x) >= 30]
    h, p = stats.kruskal(*groups)
    print(f"  Kruskal-Wallis روی نرخ *افراد*: H={h:.1f} p={p:.3g}")
    print("  → تفاوت در سطح فرد یعنی افراد این شهرها واقعاً رفتار متفاوتی دارند، "
          "نه اینکه فقط ترکیب سلف‌ها فرق کند.")


def run_independence(fact: pd.DataFrame, dim: pd.DataFrame) -> None:
    header("⚠️ آزمون فرض استقلال (هشدار سطر ۸۶۲ WBS) — حیاتی برای اعتبار کوانتایل")
    print("اگر تصمیم افراد مستقل بود، تعداد عدم‌دریافت در هر (روز، وعده، سلف) باید")
    print("واریانسی نزدیک به دوجمله‌ای n·p·(1−p) داشته باشد. مقایسه:\n")

    grp = fact.groupby(["date_gregorian", "Meal", "restaurant_canonical"], observed=True).agg(
        n=("dont_receive", "size"), k=("dont_receive", "sum"))
    grp = grp[grp["n"] >= 30].copy()
    grp["p_hat"] = grp["k"] / grp["n"]
    p_bar = grp["k"].sum() / grp["n"].sum()
    # آماره‌ی بیش‌پراکندگی: واریانس مشاهده‌شده‌ی p_hat در برابر p(1-p)/n
    grp["var_binom"] = p_bar * (1 - p_bar) / grp["n"]
    obs_var = grp["p_hat"].var()
    exp_var = grp["var_binom"].mean()
    kv("تعداد گروه (روز×وعده×سلف با n≥۳۰)", len(grp))
    kv("p̄ کل", f"{p_bar:.4f}")
    kv("واریانس مشاهده‌شده‌ی نرخ گروهی", f"{obs_var:.6f}")
    kv("واریانس مورد انتظار تحت استقلال", f"{exp_var:.6f}")
    kv("**نسبت بیش‌پراکندگی**", f"{obs_var / exp_var:.2f}×")
    # آماره‌ی chi2 پیرسون برای بیش‌پراکندگی
    chi2 = (((grp["k"] - grp["n"] * p_bar) ** 2) / (grp["n"] * p_bar * (1 - p_bar))).sum()
    dof = len(grp) - 1
    kv("Chi² پیرسون / درجه آزادی", f"{chi2 / dof:.2f}  (۱.۰ = استقلال کامل)")
    print(f"p-value برای H0=استقلال: {1 - stats.chi2.cdf(chi2, dof):.3g}")
    print("\n→ نسبت >۱ یعنی افراد **مستقل نیستند**: واریانس واقعی مجموع بزرگ‌تر از دوجمله‌ای است.")
    print("→ پیامد عملیاتی: کوانتایلی که با فرض استقلال محاسبه شود **خوش‌بینانه** است و")
    print("  ریسک کمبود غذا را کم‌برآورد می‌کند. مدل B نباید احتمالات فردی را ساده جمع بزند.")

    print("\nآیا هم‌خوابگاهی/هم‌دانشکده‌ای‌ها رفتار همبسته دارند؟")
    d2 = dim.set_index("PersonId")
    pr = fact.groupby("PersonId")["dont_receive"].agg(["mean", "size"])
    pr = pr[pr["size"] >= 10].join(d2[["dorm_canonical", "CollegeName", "is_dorm_resident"]])
    for col, label in [("dorm_canonical", "واحد سکونت/سرو"), ("CollegeName", "دانشکده")]:
        sub = pr.dropna(subset=[col])
        grp_sizes = sub.groupby(col, observed=True)["mean"].size()
        keep = grp_sizes[grp_sizes >= 30].index
        sub = sub[sub[col].isin(keep)]
        # ضریب همبستگی درون‌گروهی ICC(1) از ANOVA یک‌طرفه
        groups = [x["mean"].values for _, x in sub.groupby(col, observed=True)]
        k_bar = np.mean([len(g) for g in groups])
        grand = sub["mean"].mean()
        msb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups) / (len(groups) - 1)
        msw = sum(((g - g.mean()) ** 2).sum() for g in groups) / (len(sub) - len(groups))
        icc = (msb - msw) / (msb + (k_bar - 1) * msw)
        f, p = stats.f_oneway(*groups)
        print(f"  {label}: {len(groups)} گروه، ICC(1)={icc:+.4f}، F={f:.1f}، p={p:.3g}")
    print("  → ICC مثبت یعنی افراد هم‌گروه شبیه هم‌اند (نقض استقلال).")


def run_consistency(fact: pd.DataFrame) -> None:
    header("۴.۱۳.۴ سازگاری بین دو منبع داده")
    agg = load_dataset()
    ind = fact.groupby(["date_gregorian", "Meal", "restaurant_canonical"], observed=True).agg(
        Res_ind=("Count", "sum"), NoRecv_ind=("dont_receive", "sum")).reset_index()
    aggr = agg.groupby(["date_gregorian", "Meal", "RestaurantName"]).agg(
        Res_agg=("Res", "sum"), NoRecv_agg=("NoRecv", "sum")).reset_index()
    m = aggr.merge(ind, left_on=["date_gregorian", "Meal", "RestaurantName"],
                   right_on=["date_gregorian", "Meal", "restaurant_canonical"], how="outer",
                   indicator=True)
    print(m["_merge"].value_counts().to_string())
    both = m[m["_merge"] == "both"].copy()
    both["diff_res"] = both["Res_ind"] - both["Res_agg"]
    kv("\nگروه‌های منطبق", len(both))
    kv("همبستگی Res دو منبع", f"{both['Res_ind'].corr(both['Res_agg']):.6f}")
    kv("سهم گروه‌های کاملاً برابر (Res)", pct((both["diff_res"] == 0).mean()))
    print("\nتوزیع اختلاف Res (فردی منهای تجمیعی):")
    print(both["diff_res"].describe().round(2).to_string())
    kv("مجموع Res فردی", f"{both['Res_ind'].sum():,.0f}")
    kv("مجموع Res تجمیعی", f"{both['Res_agg'].sum():,.0f}")
    kv("اختلاف نسبی کل", f"{(both['Res_ind'].sum() / both['Res_agg'].sum() - 1):+.4%}")
    kv("نرخ کل از منبع فردی", f"{both['NoRecv_ind'].sum() / both['Res_ind'].sum():.4f}")
    kv("نرخ کل از منبع تجمیعی", f"{both['NoRecv_agg'].sum() / both['Res_agg'].sum():.4f}")


def main() -> None:
    setup()
    fact = load_fact()
    dim = pd.read_csv(PERSON_DIM_PATH)
    header(f"داده‌ی سطح فرد v2 — {len(fact):,} رزرو · {fact['PersonId'].nunique():,} فرد · "
           f"{len(dim):,} ردیف بُعد")
    per = run_history(fact, dim)
    run_imbalance(fact, per)
    run_demographics(fact, dim, per)
    run_mechanism(fact, per, dim)
    run_independence(fact, dim)
    run_consistency(fact)


if __name__ == "__main__":
    main()
