"""بند ۴.۲ — تحلیل گروهی و دفترچه‌ی فرضیه‌ها (H1–H12 + H17–H20)، دور ۱ روی `dataset_v2`.

**سه اشتباه رایجی که بند ۴.۲.۳ WBS هشدار داده و اینجا صریحاً مدیریت می‌شوند:**

۱. *فقط p-value:* برای هر آزمون اندازه‌ی اثر (Cliff's δ، rank-biserial، η²، ρ اسپیرمن)
   و برای فرضیه‌های دوگروهی فاصله‌ی اطمینان bootstrap تفاوت میانه‌ها گزارش می‌شود.
۲. *آزمون‌های چندگانه:* همه‌ی p-value ها یک‌جا با Benjamini-Hochberg تصحیح می‌شوند
   (`p_bh`)، نه هر آزمون جدا.
۳. *نقض استقلال:* رکوردهای یک سلف در روزهای متوالی مستقل نیستند، پس p-value خام
   خوش‌بینانه است. هر فرضیه **دو بار** آزمون می‌شود: یک بار روی رکوردهای خام
   (`p_raw`) و یک بار روی داده‌ی تجمیع‌شده در سطح سلف×وعده (`p_clust`, n≈۵۰ به‌جای
   ۷۵۷۹). فرضیه‌ای که فقط در آزمون خام معنادار است، **یافته‌ی قابل‌اتکا نیست** و در
   ستون `robust` علامت می‌خورد.

فرضیه‌ها **پیش از دیدن نتیجه** نوشته شده‌اند (ضد-HARKing، بند ۴.۲.۴): H1–H12 از خود
WBS، و H17–H20 از ورودی ذی‌نفع در شروع فاز ۴ (تفکیک خوابگاه/غیرخوابگاه و شهر).

اجرا: `python -m src.eda_lib.runners.s02_hypotheses`
"""

import numpy as np
import pandas as pd
from scipy import stats

from src.eda_lib.group_test_helpers import (
    bootstrap_median_diff_ci,
    build_days_since_same_food,
    cliffs_delta,
    eta_squared_kruskal,
    eta_squared_levene,
    fdr_correct,
    mann_kendall,
    rank_biserial_from_u,
)
from src.eda_lib.runners._common import CALENDAR_PATH, header, load_dataset_with_weather, setup

RESULTS: list[dict] = []


def _record(hid, statement, test, p_raw, effect, effect_name, p_clust=None, extra=""):
    RESULTS.append({
        "H": hid, "فرضیه": statement, "آزمون": test,
        "p_raw": p_raw, "p_clust": p_clust,
        "اندازه اثر": effect, "نوع اثر": effect_name, "توضیح": extra,
    })


def _cluster_frame(df: pd.DataFrame) -> pd.DataFrame:
    """تجمیع در سطح (سلف، وعده) — واحدی که تقریباً مستقل است، برای آزمون مقاوم به خودهمبستگی."""
    return (df.groupby(["RestaurantName", "RestaurantType", "city", "Meal"], as_index=False)
              .agg(rho=("rho", "mean"), Res=("Res", "mean"), n=("rho", "size")))


def _mw(x, y, alternative="two-sided"):
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x[~np.isnan(x)], y[~np.isnan(y)]
    if len(x) < 3 or len(y) < 3:
        return np.nan, np.nan, np.nan
    u, p = stats.mannwhitneyu(x, y, alternative=alternative)
    return p, cliffs_delta(x, y), rank_biserial_from_u(u, len(x), len(y))


def _kw(groups):
    groups = [np.asarray(g, float) for g in groups if len(g) >= 3]
    h, p = stats.kruskal(*groups)
    n = sum(len(g) for g in groups)
    return p, eta_squared_kruskal(h, n, len(groups))


def _spearman(a, b):
    m = ~(pd.isna(a) | pd.isna(b))
    r, p = stats.spearmanr(np.asarray(a)[m], np.asarray(b)[m])
    return p, r


# ---------------------------------------------------------------------------
# فرضیه‌ها
# ---------------------------------------------------------------------------

def h1_meal(df, cl):
    header("H1 — نرخ عدم‌دریافت شام > ناهار", 2)
    lunch, dinner = df.loc[df.Meal == "lunch", "rho"], df.loc[df.Meal == "dinner", "rho"]
    p, d, rb = _mw(dinner, lunch, "greater")
    pc, _, _ = _mw(cl.loc[cl.Meal == "dinner", "rho"], cl.loc[cl.Meal == "lunch", "rho"], "greater")
    md, lo, hi = bootstrap_median_diff_ci(dinner, lunch, random_state=42)
    print(f"median dinner={dinner.median():.4f} lunch={lunch.median():.4f} diff={md:+.4f} CI=[{lo:+.4f},{hi:+.4f}]")
    print(f"p(dinner>lunch)={p:.3g}  Cliff's δ={d:+.3f}  p_cluster={pc:.3g}")
    print("⚠️ جهت فرضیه معکوس است — ناهار بالاتر از شام. آزمون دوطرفه در جهت معکوس:")
    p2, d2, _ = _mw(lunch, dinner, "greater")
    print(f"p(lunch>dinner)={p2:.3g}  Cliff's δ={d2:+.3f}")
    print("\nهمین مقایسه، جداگانه داخل هر نوع سلف (کنترل هم‌آمیختگی وعده×نوع سلف):")
    for rt in ["daneshgah", "khabgah"]:
        s = df[df.RestaurantType == rt]
        pp, dd, _ = _mw(s.loc[s.Meal == "dinner", "rho"], s.loc[s.Meal == "lunch", "rho"])
        print(f"  {rt}: n_dinner={(s.Meal == 'dinner').sum()} n_lunch={(s.Meal == 'lunch').sum()} "
              f"p={pp:.3g} δ={dd:+.3f}")
    _record("H1", "شام > ناهار", "Mann-Whitney (یک‌طرفه)", p, d, "Cliff's δ", pc,
            f"جهت معکوس تأیید شد: p(ناهار>شام)={p2:.2g}")


def h2_dow(df, cl):
    header("H2 — چهارشنبه/پنجشنبه نرخ بالاتری دارند", 2)
    groups = [g["rho"].values for _, g in df.groupby("DayOfWeek")]
    p, eta = _kw(groups)
    pc, etac = _kw([g["rho"].values for _, g in
                    df.groupby(["RestaurantName", "Meal", "DayOfWeek"])["rho"].mean()
                      .reset_index().groupby("DayOfWeek")])
    print(f"Kruskal-Wallis همه‌ی روزها: p={p:.3g}  η²={eta:.4f}  |  خوشه‌ای: p={pc:.3g} η²={etac:.4f}")
    print(df.groupby("dow_name").agg(n=("rho", "size"), median=("rho", "median"),
                                     mean=("rho", "mean")).round(4).to_string())
    late = df[df.DayOfWeek.isin([4, 5])]["rho"]
    early = df[df.DayOfWeek.isin([0, 1, 2, 3])]["rho"]
    p2, d2, _ = _mw(late, early, "greater")
    print(f"\n(چهارشنبه+پنجشنبه) > بقیه: p={p2:.3g}  Cliff's δ={d2:+.3f}")
    _record("H2", "چهارشنبه/پنجشنبه بالاتر", "Kruskal-Wallis + Mann-Whitney", p, eta, "η²", pc,
            f"چهارشنبه+پنجشنبه vs بقیه: p={p2:.2g}, δ={d2:+.2f}")


def h3_preholiday(df, cl):
    header("H3 — نرخ در روز قبل از تعطیلی بالاتر است", 2)
    a = df.loc[df.is_day_before_holiday, "rho"]
    b = df.loc[~df.is_day_before_holiday, "rho"]
    p, d, _ = _mw(a, b, "greater")
    md, lo, hi = bootstrap_median_diff_ci(a, b, random_state=42)
    clm = (df.groupby(["RestaurantName", "Meal", "is_day_before_holiday"])["rho"].mean().reset_index())
    pc, _, _ = _mw(clm.loc[clm.is_day_before_holiday, "rho"], clm.loc[~clm.is_day_before_holiday, "rho"], "greater")
    print(f"n_pre={len(a)} n_other={len(b)}  median diff={md:+.4f} CI=[{lo:+.4f},{hi:+.4f}]")
    print(f"p={p:.3g}  Cliff's δ={d:+.3f}  p_cluster={pc:.3g}")
    _record("H3", "روز قبل از تعطیلی بالاتر", "Mann-Whitney (یک‌طرفه)", p, d, "Cliff's δ", pc,
            f"تفاوت میانه {md:+.4f} CI=[{lo:+.4f},{hi:+.4f}]")


def h4_foodtype(df, cl):
    header("H4 — غذاهای خورشتی نرخ بالاتری دارند", 2)
    a = df.loc[df.FoodType == "khorak", "rho"]
    b = df.loc[df.FoodType == "berenji", "rho"]
    p, d, _ = _mw(a, b, "greater")
    md, lo, hi = bootstrap_median_diff_ci(a, b, random_state=42)
    clm = df.groupby(["RestaurantName", "Meal", "FoodType"])["rho"].mean().reset_index()
    pc, _, _ = _mw(clm.loc[clm.FoodType == "khorak", "rho"], clm.loc[clm.FoodType == "berenji", "rho"], "greater")
    print(f"median khorak={a.median():.4f} berenji={b.median():.4f} diff={md:+.4f} CI=[{lo:+.4f},{hi:+.4f}]")
    print(f"p={p:.3g}  Cliff's δ={d:+.3f}  p_cluster={pc:.3g}")
    _record("H4", "خورشتی > برنجی", "Mann-Whitney (یک‌طرفه)", p, d, "Cliff's δ", pc,
            f"تفاوت میانه {md:+.4f} CI=[{lo:+.4f},{hi:+.4f}]")


def h5_food_repeat(df, cl):
    header("H5 — تکرار غذا در فاصله‌ی کوتاه نرخ را بالا می‌برد", 2)
    dsf = build_days_since_same_food(df)
    p, r = _spearman(dsf.values, df["rho"].values)
    print(f"Spearman(days_since_same_food, ρ) = {r:+.4f}  p={p:.3g}  n={dsf.notna().sum()}")
    print("توزیع فاصله‌ی تکرار:", dsf.describe().round(2).to_dict())
    _record("H5", "تکرار زودهنگام غذا ⇒ نرخ بالاتر", "Spearman", p, r, "ρ اسپیرمن", None,
            "همبستگی منفی مورد انتظار بود (فاصله کمتر ⇒ نرخ بیشتر)")


def h6_restaurant(df, cl):
    header("H6 — سلف‌ها تفاوت پایدار و معنادار دارند", 2)
    p, eta = _kw([g["rho"].values for _, g in df.groupby("RestaurantName")])
    print(f"Kruskal-Wallis روی {df.RestaurantName.nunique()} سلف: p={p:.3g}  η²={eta:.4f}")
    med = df.groupby(["RestaurantName", "RestaurantType"])["rho"].agg(["size", "median"]).round(4)
    print(f"دامنه‌ی میانه‌ی سلف‌ها: {med['median'].min():.4f} تا {med['median'].max():.4f}")
    _record("H6", "تفاوت معنادار بین سلف‌ها", "Kruskal-Wallis", p, eta, "η²", None,
            f"دامنه‌ی میانه {med['median'].min():.3f}–{med['median'].max():.3f}")


def h7_exam(df, cl):
    header("H7 — نرخ در بازه‌ی امتحانات متفاوت است", 2)
    a, b = df.loc[df.is_exam_period, "rho"], df.loc[~df.is_exam_period, "rho"]
    p, d, _ = _mw(a, b)
    md, lo, hi = bootstrap_median_diff_ci(a, b, random_state=42)
    clm = df.groupby(["RestaurantName", "Meal", "is_exam_period"])["rho"].mean().reset_index()
    pc, _, _ = _mw(clm.loc[clm.is_exam_period, "rho"], clm.loc[~clm.is_exam_period, "rho"])
    print(f"n_exam={len(a)} n_other={len(b)}  median diff={md:+.4f} CI=[{lo:+.4f},{hi:+.4f}]")
    print(f"p={p:.3g}  Cliff's δ={d:+.3f}  p_cluster={pc:.3g}")
    _record("H7", "بازه‌ی امتحانات متفاوت است", "Mann-Whitney (دوطرفه)", p, d, "Cliff's δ", pc,
            f"تفاوت میانه {md:+.4f} CI=[{lo:+.4f},{hi:+.4f}]")


def h8_h9_weather(df, cl):
    header("H8/H9 — AQI و دما (با هواشناسی شهر درست هر سلف)", 2)
    for hid, col, label in [("H8", "aqi_us_max", "AQI حداکثر"), ("H9", "temp_min", "دمای کمینه")]:
        p, r = _spearman(df[col].values, df["rho"].values)
        # کنترل روز هفته و سلف: باقیمانده‌ی ρ نسبت به میانگین همان (سلف، وعده، روزهفته)
        resid = df["rho"] - df.groupby(["RestaurantName", "Meal", "DayOfWeek"])["rho"].transform("mean")
        p_ctrl, r_ctrl = _spearman(df[col].values, resid.values)
        print(f"{hid} {label}: Spearman خام r={r:+.4f} p={p:.3g}  |  "
              f"پس از کنترل سلف×وعده×روزهفته r={r_ctrl:+.4f} p={p_ctrl:.3g}")
        _record(hid, f"{label} با نرخ مرتبط است", "Spearman (+کنترل)", p_ctrl, r_ctrl, "ρ اسپیرمن جزئی", None,
                f"خام r={r:+.3f}")
    print("\nمقایسه‌ی حیاتی — همین آزمون با هواشناسی *غلط* (تهران برای همه):")
    teh = df[df.city == "تهران"]
    p_t, r_t = _spearman(teh["aqi_us_max"].values, teh["rho"].values)
    print(f"  فقط تهران: AQI r={r_t:+.4f} p={p_t:.3g}  (n={len(teh)})")


def h10_gender(df, cl):
    header("H10 — نسبت جنسیتی رزرو با نرخ مرتبط است", 2)
    p, r = _spearman(df["gender_ratio"].values, df["rho"].values)
    resid = df["rho"] - df.groupby(["RestaurantName", "Meal"])["rho"].transform("mean")
    p_c, r_c = _spearman(df["gender_ratio"].values, resid.values)
    print(f"Spearman خام r={r:+.4f} p={p:.3g}  |  پس از کنترل سلف×وعده r={r_c:+.4f} p={p_c:.3g}")
    print("توزیع gender_ratio:", df["gender_ratio"].describe().round(3).to_dict())
    _record("H10", "نسبت جنسیتی با نرخ مرتبط است", "Spearman (+کنترل سلف×وعده)", p_c, r_c, "ρ اسپیرمن جزئی", None,
            f"خام r={r:+.3f}")


def h11_variance(df, cl):
    header("H11 — رکوردهای با Res کوچک واریانس ρ بالاتری دارند", 2)
    q = pd.qcut(df["Res"], 4, labels=["Q1 کوچک", "Q2", "Q3", "Q4 بزرگ"])
    groups = [df.loc[q == lv, "rho"].values for lv in q.cat.categories]
    stat, p = stats.levene(*groups, center="median")
    eta = eta_squared_levene(df["rho"], q)
    print(f"Levene بین چارک‌های Res: stat={stat:.2f} p={p:.3g}  η²={eta:.4f}")
    summ = df.assign(q=q).groupby("q", observed=True).agg(
        n=("rho", "size"), Res_median=("Res", "median"), rho_std=("rho", "std"), rho_mean=("rho", "mean"))
    print(summ.round(4).to_string())
    ratio = summ["rho_std"].iloc[0] / summ["rho_std"].iloc[-1]
    print(f"نسبت انحراف معیار چارک کوچک به بزرگ: {ratio:.2f}x")
    _record("H11", "واریانس ρ با Res کوچک بیشتر است", "Levene (چارک‌های Res)", p, eta, "η²", None,
            f"نسبت std چارک۱/چارک۴ = {ratio:.2f}x")


def h12_trend(df, cl):
    header("H12 — نرخ در طول ترم روند دارد", 2)
    daily = df.groupby("date_gregorian")["rho"].mean().sort_index()
    mk = mann_kendall(daily.values)
    print(f"Mann-Kendall روی سری روزانه‌ی دانشگاه (n={len(daily)}): "
          f"trend={mk.get('trend')} p={mk.get('p'):.3g} tau={mk.get('tau'):+.4f} S={mk.get('S')}")
    per_sem = df.groupby(["semester", "week_of_semester"])["rho"].mean()
    print("\nمیانگین ρ بر حسب هفته‌ی ترم (۱۰ ردیف اول):")
    print(per_sem.head(10).round(4).to_string())
    _record("H12", "روند زمانی در طول ترم", "Mann-Kendall", mk.get("p"), mk.get("tau"), "tau کندال", None,
            f"جهت={mk.get('trend')}")


def h17_h20_new(df, cl):
    header("H17–H20 — فرضیه‌های افزوده‌ی شروع فاز ۴ (ورودی ذی‌نفع)", 2)

    # H17: سلف خوابگاهی در برابر دانشگاهی
    a = df.loc[df.RestaurantType == "khabgah", "rho"]
    b = df.loc[df.RestaurantType == "daneshgah", "rho"]
    p, d, _ = _mw(a, b)
    md, lo, hi = bootstrap_median_diff_ci(a, b, random_state=42)
    pc, dc, _ = _mw(cl.loc[cl.RestaurantType == "khabgah", "rho"], cl.loc[cl.RestaurantType == "daneshgah", "rho"])
    print(f"H17 خوابگاهی vs دانشگاهی: median {a.median():.4f} vs {b.median():.4f} "
          f"diff={md:+.4f} CI=[{lo:+.4f},{hi:+.4f}]")
    print(f"     p_raw={p:.3g} δ={d:+.3f}  |  p_cluster={pc:.3g} δ={dc:+.3f}")
    _record("H17", "سلف خوابگاهی با دانشگاهی تفاوت دارد", "Mann-Whitney", p, d, "Cliff's δ", pc,
            f"تفاوت میانه {md:+.4f} CI=[{lo:+.4f},{hi:+.4f}]")

    # H17b: همان مقایسه، فقط داخل تهران و فقط ناهار (کنترل هم‌آمیختگی شهر و وعده)
    sub = df[(df.city == "تهران") & (df.Meal == "lunch")]
    p2, d2, _ = _mw(sub.loc[sub.RestaurantType == "khabgah", "rho"],
                    sub.loc[sub.RestaurantType == "daneshgah", "rho"])
    print(f"H17b همان مقایسه فقط (تهران، ناهار): n={len(sub)} p={p2:.3g} δ={d2:+.3f}")
    _record("H17b", "خوابگاهی vs دانشگاهی — کنترل‌شده (تهران، ناهار)", "Mann-Whitney", p2, d2, "Cliff's δ", None, "")

    # H18: تهران در برابر غیرتهران
    a = df.loc[df.is_tehran, "rho"]
    b = df.loc[~df.is_tehran, "rho"]
    p, d, _ = _mw(a, b, "greater")
    md, lo, hi = bootstrap_median_diff_ci(a, b, random_state=42)
    pc, dc, _ = _mw(cl.loc[cl.city == "تهران", "rho"], cl.loc[cl.city != "تهران", "rho"], "greater")
    print(f"\nH18 تهران > غیرتهران: median {a.median():.4f} vs {b.median():.4f} "
          f"diff={md:+.4f} CI=[{lo:+.4f},{hi:+.4f}]")
    print(f"     p_raw={p:.3g} δ={d:+.3f}  |  p_cluster={pc:.3g} δ={dc:+.3f}")
    _record("H18", "نرخ تهران > پردیس‌های خارج تهران", "Mann-Whitney (یک‌طرفه)", p, d, "Cliff's δ", pc,
            f"تفاوت میانه {md:+.4f} CI=[{lo:+.4f},{hi:+.4f}]")

    # H19: تفاوت بین شهرها به‌طور کلی
    p, eta = _kw([g["rho"].values for _, g in df.groupby("city")])
    print(f"\nH19 تفاوت بین ۶ شهر: Kruskal-Wallis p={p:.3g} η²={eta:.4f}")
    _record("H19", "شهرها تفاوت معنادار دارند", "Kruskal-Wallis", p, eta, "η²", None, "")

    # H20: آیا اثر شهر صرفاً بازتاب اثر سلف است؟ (سلف داخل شهر تودرتو است)
    # هر شهر غیرتهرانی دقیقاً یک سلف دارد، پس Kruskal بین شهرها با واحدِ سلف ناممکن است؛
    # آزمون درست، مقایسه‌ی دوگروهی «سلف‌های تهران» با «سلف‌های غیرتهران» است — با واحد
    # تحلیل *سلف* (n=۳۰) که برخلاف رکورد، همبستگی سریالی درون‌سلفی ندارد.
    print("\nH20 — آیا اثر شهر چیزی فراتر از اثر سلف است؟ (واحد تحلیل = سلف، نه رکورد)")
    within = df.groupby(["city", "RestaurantName", "RestaurantType"])["rho"].mean().reset_index()
    print(within.groupby("city")["rho"].agg(["size", "mean", "std", "min", "max"]).round(4).to_string())
    teh_r = within.loc[within.city == "تهران", "rho"]
    oth_r = within.loc[within.city != "تهران", "rho"]
    p20, d20, _ = _mw(teh_r, oth_r, "greater")
    print(f"سلف‌های تهران (n={len(teh_r)}, میانه {teh_r.median():.4f}) در برابر "
          f"سلف‌های غیرتهران (n={len(oth_r)}, میانه {oth_r.median():.4f})")
    print(f"Mann-Whitney یک‌طرفه با واحد سلف: p={p20:.3g}  Cliff's δ={d20:+.3f}")
    print(f"هم‌پوشانی: بیشینه‌ی غیرتهران={oth_r.max():.4f} · کمینه‌ی تهران={teh_r.min():.4f}")
    _record("H20", "اثر شهر با واحد تحلیلِ سلف هم باقی می‌ماند", "Mann-Whitney (واحد=سلف)",
            p20, d20, "Cliff's δ", None, f"n_tehran={len(teh_r)} n_other={len(oth_r)}")


def main() -> None:
    setup()
    df = load_dataset_with_weather()
    cal = pd.read_csv(CALENDAR_PATH, parse_dates=["date_gregorian"])
    cal_cols = ["date_gregorian", "is_day_before_holiday", "is_day_after_holiday", "is_exam_period",
                "is_final_exam_period", "days_to_next_holiday", "days_since_last_holiday",
                "semester", "week_of_semester", "is_holiday_any"]
    df = df.merge(cal[cal_cols], on="date_gregorian", how="left", validate="many_to_one")
    cl = _cluster_frame(df)

    header(f"دفترچه‌ی فرضیه‌ها — n={len(df)} رکورد، واحد خوشه‌ای n={len(cl)} (سلف×وعده)")

    for fn in [h1_meal, h2_dow, h3_preholiday, h4_foodtype, h5_food_repeat, h6_restaurant,
               h7_exam, h8_h9_weather, h10_gender, h11_variance, h12_trend, h17_h20_new]:
        fn(df, cl)

    header("جمع‌بندی: تصحیح چندگانگی Benjamini-Hochberg روی همه‌ی آزمون‌ها")
    res = pd.DataFrame(RESULTS)
    res["p_bh"] = fdr_correct(res["p_raw"].fillna(1.0).values)
    res["معنادار (BH .05)"] = res["p_bh"] < 0.05
    res["robust"] = np.where(
        res["p_clust"].isna(), "—",
        np.where((res["p_bh"] < 0.05) & (res["p_clust"] < 0.05), "بله",
                 np.where(res["p_bh"] < 0.05, "خیر (فقط خام)", "—")))
    pd.set_option("display.max_colwidth", 46)
    print(res[["H", "فرضیه", "آزمون", "p_raw", "p_bh", "p_clust", "اندازه اثر", "نوع اثر",
               "معنادار (BH .05)", "robust"]].to_string(index=False))
    print("\nتوضیح ستون‌ها: p_raw=روی رکوردهای خام · p_clust=روی داده‌ی تجمیع‌شده در سطح سلف×وعده "
          "(مقاوم به خودهمبستگی) · p_bh=p_raw پس از تصحیح BH-FDR · "
          "robust=«بله» یعنی هر دو معنادارند")
    print("\nیادداشت‌های تکمیلی هر فرضیه:")
    for r in RESULTS:
        if r["توضیح"]:
            print(f"  {r['H']}: {r['توضیح']}")


if __name__ == "__main__":
    main()
