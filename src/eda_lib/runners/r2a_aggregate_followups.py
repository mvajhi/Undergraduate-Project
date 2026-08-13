"""دور ۲ (الف) — پاسخ به سؤال‌های باز سطح تجمیعیِ دور ۱.

هر تابع اینجا **یک سؤال مشخص** را که دور ۱ باز گذاشت جواب می‌دهد. سؤال‌ها از انتهای
سندهای `doc/eda/round1/S*.md` آمده‌اند:

- Q1 (از S1/S2): اثر «شهر» واقعاً شهر است یا نماینده‌ی اندازه‌ی سلف/سهم خوابگاهی؟
- Q2 (از S2/S6): تناقض ظاهری H5 («فاصله‌ی بیشتر ⇒ نرخ بالاتر») با F6.3 («غذای پرتکرار
  نرخ بالاتر») — کدام سازوکار واقعی است؟
- Q3 (از S3): آیا همبستگی تأخیر ۱۴ ناهار پس از کنترل «هویت غذا» باقی می‌ماند؟
- Q4 (از S2): H7 — نرخ در امتحانات کمتر است؛ آیا حجم رزرو هم تغییر می‌کند (خودانتخابی)؟
- Q5 (از S7): آیا اثر برف پس از کنترل «روز قبل از تعطیلی» باقی می‌ماند؟
- Q6 (از S7): نرخ بالای شام رمضان اثر رمضان است یا اثر حجم پایین؟
- Q7 (از S1): تکلیف جمعه با ۸۶ رکورد چیست؟

اجرا: `python -m src.eda_lib.runners.r2a_aggregate_followups`
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from src.eda_lib.group_test_helpers import build_days_since_same_food, cliffs_delta
from src.eda_lib.runners._common import (
    CALENDAR_PATH,
    EVENTS_PATH,
    header,
    kv,
    load_dataset_with_weather,
    setup,
)


def q1_city_or_proxy(df: pd.DataFrame) -> None:
    header("Q1 — اثر «شهر» واقعی است یا نماینده‌ی اندازه‌ی سلف / سهم خوابگاهی؟")
    r = df.groupby(["RestaurantName", "RestaurantType", "city", "is_tehran"], as_index=False).agg(
        rho=("rho", "mean"), res_mean=("Res", "mean"), n=("rho", "size"))
    r["log_res"] = np.log(r["res_mean"])
    r["is_khabgah"] = (r["RestaurantType"] == "khabgah").astype(int)
    r["is_teh"] = r["is_tehran"].astype(int)

    print("سه مدل تودرتو با واحد تحلیل = سلف (n=%d)، تا خودهمبستگی رکوردی وارد نشود:\n" % len(r))
    models = {
        "فقط اندازه": "rho ~ log_res",
        "فقط نوع سلف": "rho ~ is_khabgah",
        "اندازه + نوع سلف": "rho ~ log_res + is_khabgah",
        "اندازه + نوع + تهران": "rho ~ log_res + is_khabgah + is_teh",
    }
    for name, f in models.items():
        m = smf.ols(f, data=r).fit()
        coef = f"is_teh={m.params.get('is_teh', float('nan')):+.4f} (p={m.pvalues.get('is_teh', float('nan')):.3g})" \
            if "is_teh" in m.params else ""
        print(f"  {name:<24s} R²={m.rsquared:.4f}  R²adj={m.rsquared_adj:.4f}   {coef}")

    m_full = smf.ols("rho ~ log_res + is_khabgah + is_teh", data=r).fit()
    print("\nضرایب مدل کامل:")
    print(m_full.summary2().tables[1].round(4).to_string())

    print("\nمقایسه‌ی مستقیم: سلف‌های هم‌اندازه در تهران و خارج تهران")
    r["size_bin"] = pd.qcut(r["res_mean"], 3, labels=["کوچک", "متوسط", "بزرگ"])
    print(r.groupby(["size_bin", "is_tehran"], observed=True).agg(
        سلف=("rho", "size"), نرخ=("rho", "mean"), اندازه=("res_mean", "mean")).round(4).to_string())

    print("\nهمان مقایسه فقط بین سلف‌های خوابگاهی (کنترل نوع):")
    kh = r[r.is_khabgah == 1]
    print(kh.groupby("is_tehran").agg(سلف=("rho", "size"), نرخ=("rho", "mean"),
                                      اندازه=("res_mean", "mean")).round(4).to_string())
    print("\nو فقط بین سلف‌های دانشکده‌ای:")
    da = r[r.is_khabgah == 0]
    print(da.groupby("is_tehran").agg(سلف=("rho", "size"), نرخ=("rho", "mean"),
                                      اندازه=("res_mean", "mean")).round(4).to_string())


def q2_food_novelty(df: pd.DataFrame) -> None:
    header("Q2 — تناقض H5 و F6.3: فاصله‌ی تکرار در برابر تکرارِ کل")
    d = df.copy()
    d["days_since"] = build_days_since_same_food(d)
    freq = d.groupby("FoodName").size().rename("food_total_servings")
    d = d.join(freq, on="FoodName")

    print("همبستگی خام هر دو متغیر با ρ:")
    for c, lbl in [("days_since", "فاصله تا سرو قبلی همان غذا"), ("food_total_servings", "تعداد کل سرو آن غذا")]:
        m = d[c].notna()
        r = stats.spearmanr(d.loc[m, c], d.loc[m, "rho"])
        print(f"  {lbl}: Spearman={r[0]:+.4f} p={r[1]:.3g}")

    print("\n⚠️ این دو متغیر خودشان همبسته‌اند:")
    m = d["days_since"].notna()
    print(f"  Spearman(فاصله, تکرار کل) = {stats.spearmanr(d.loc[m, 'days_since'], d.loc[m, 'food_total_servings'])[0]:+.4f}")

    print("\nمدل مشترک روی باقیمانده‌ی ρ (کنترل سلف×وعده×روزهفته):")
    d["resid"] = d["rho"] - d.groupby(["RestaurantName", "Meal", "DayOfWeek"])["rho"].transform("mean")
    dd = d.dropna(subset=["days_since", "resid"]).copy()
    dd["log_days"] = np.log(dd["days_since"])
    dd["log_freq"] = np.log(dd["food_total_servings"])
    for f in ["resid ~ log_days", "resid ~ log_freq", "resid ~ log_days + log_freq"]:
        mm = smf.ols(f, data=dd).fit()
        parts = " · ".join(f"{k}={v:+.5f}(p={mm.pvalues[k]:.2g})"
                           for k, v in mm.params.items() if k != "Intercept")
        print(f"  {f:<32s} R²={mm.rsquared:.5f}  {parts}")

    print("\nآیا اثر فاصله پس از کنترل *بلوک تقویمی* باقی می‌ماند؟")
    cal = pd.read_csv(CALENDAR_PATH, parse_dates=["date_gregorian"])
    dd = dd.merge(cal[["date_gregorian", "is_day_before_holiday", "is_exam_period",
                       "days_since_last_holiday"]], on="date_gregorian", how="left")
    mm = smf.ols("resid ~ log_days + log_freq + C(is_day_before_holiday) + C(is_exam_period)",
                 data=dd).fit()
    print(f"  با کنترل تقویم: ضریب log_days={mm.params['log_days']:+.5f} "
          f"(p={mm.pvalues['log_days']:.3g}) · R²={mm.rsquared:.5f}")
    print("\nنرخ به تفکیک دسته‌ی فاصله (باقیمانده):")
    dd["gap_bin"] = pd.cut(dd["days_since"], [0, 7, 14, 21, 35, 60, 200],
                           labels=["≤۷", "۸-۱۴", "۱۵-۲۱", "۲۲-۳۵", "۳۶-۶۰", ">۶۰"])
    print(dd.groupby("gap_bin", observed=True).agg(
        n=("resid", "size"), باقیمانده=("resid", "mean")).round(4).to_string())


def q3_lag14_vs_food(df: pd.DataFrame) -> None:
    header("Q3 — آیا همبستگی تأخیر ۱۴ ناهار پس از کنترل «هویت غذا» باقی می‌ماند؟")
    lunch = df[df.Meal == "lunch"].copy()
    # باقیمانده پس از حذف اثر ثابت غذا (و سلف)
    lunch["resid_food"] = lunch["rho"] - lunch.groupby(["RestaurantName", "FoodName"])["rho"].transform("mean")
    lunch["resid_rest"] = lunch["rho"] - lunch.groupby(["RestaurantName"])["rho"].transform("mean")

    for col, lbl in [("rho", "ρ خام"), ("resid_rest", "پس از حذف اثر سلف"),
                     ("resid_food", "پس از حذف اثر سلف×غذا")]:
        daily = lunch.groupby("date_gregorian")[col].mean()
        daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max(), freq="D"))
        out = []
        for k in [1, 7, 14, 28]:
            a, b = daily.values[:-k], daily.values[k:]
            m = ~(np.isnan(a) | np.isnan(b))
            if m.sum() >= 20:
                r, p = stats.pearsonr(a[m], b[m])
                out.append(f"lag{k}={r:+.3f}{'*' if p < 0.05 else ' '}")
            else:
                out.append(f"lag{k}=NA")
        print(f"  {lbl:<26s} {'  '.join(out)}")
    print("\n→ اگر lag14 پس از حذف اثر غذا فرو بریزد، آن همبستگی «گردش منو» بوده نه رفتار.")


def q4_exam_mechanism(df: pd.DataFrame) -> None:
    header("Q4 — H7: در امتحانات نرخ کمتر است؛ حجم چه می‌شود؟")
    cal = pd.read_csv(CALENDAR_PATH, parse_dates=["date_gregorian"])
    d = df.merge(cal[["date_gregorian", "is_exam_period", "is_final_exam_period"]],
                 on="date_gregorian", how="left")
    daily = d.groupby(["date_gregorian", "is_exam_period"]).agg(
        Res=("Res", "sum"), NoRecv=("NoRecv", "sum")).reset_index()
    daily["rho"] = daily["NoRecv"] / daily["Res"]
    g = daily.groupby("is_exam_period").agg(
        روز=("Res", "size"), حجم_میانگین=("Res", "mean"), حجم_میانه=("Res", "median"),
        نرخ=("rho", "mean"))
    print(g.round(2).to_string())
    a = daily.loc[daily.is_exam_period, "Res"]
    b = daily.loc[~daily.is_exam_period, "Res"]
    u, p = stats.mannwhitneyu(a, b)
    print(f"\nحجم روزانه در امتحانات در برابر بقیه: نسبت میانه={a.median() / b.median():.2f} · "
          f"p={p:.3g} · Cliff's δ={cliffs_delta(a.values, b.values):+.3f}")
    print("→ اگر حجم کاهش یابد و نرخ هم کاهش یابد، سازوکار «خودانتخابی» است:")
    print("  فقط کسانی رزرو می‌کنند که واقعاً می‌آیند، پس نرخ عدم‌دریافت پایین می‌آید.")


def q5_snow_holiday(df: pd.DataFrame) -> None:
    header("Q5 — آیا اثر برف پس از کنترل «روز قبل از تعطیلی» باقی می‌ماند؟")
    cal = pd.read_csv(CALENDAR_PATH, parse_dates=["date_gregorian"])
    d = df[df.city == "تهران"].merge(
        cal[["date_gregorian", "is_day_before_holiday", "is_holiday_any"]],
        on="date_gregorian", how="left")
    d["resid"] = d["rho"] - d.groupby(["RestaurantName", "Meal", "DayOfWeek"])["rho"].transform("mean")
    d["is_snow"] = d["snowfall_sum"] > 0.1
    print(pd.crosstab(d["is_snow"], d["is_day_before_holiday"]).to_string())
    m = smf.ols("resid ~ C(is_snow) + C(is_day_before_holiday)", data=d).fit()
    print("\nمدل مشترک روی باقیمانده:")
    print(m.summary2().tables[1].round(5).to_string())
    print("\nاثر برف فقط در روزهای غیرِ قبل‌از‌تعطیلی:")
    sub = d[~d["is_day_before_holiday"].fillna(False)]
    a = sub.loc[sub.is_snow, "resid"]
    b = sub.loc[~sub.is_snow, "resid"]
    if len(a) >= 20:
        u, p = stats.mannwhitneyu(a, b)
        print(f"  برفی n={len(a)} باقیمانده={a.mean():+.4f} · غیربرفی n={len(b)} "
              f"باقیمانده={b.mean():+.4f} · p={p:.3g}")


def q6_ramadan_volume(df: pd.DataFrame) -> None:
    header("Q6 — نرخ بالای شام رمضان: اثر رمضان یا اثر حجم پایین؟")
    events = pd.read_csv(EVENTS_PATH, parse_dates=["date_start", "date_end"])
    ram = events[events["event_id"].str.contains("ramadan", na=False)]
    d = df[df.Meal == "dinner"].copy()
    d["is_ramadan"] = False
    for _, e in ram.iterrows():
        d.loc[(d.date_gregorian >= e.date_start) & (d.date_gregorian <= e.date_end), "is_ramadan"] = True
    d["log_res"] = np.log(d["Res"])
    d["resid"] = d["rho"] - d.groupby(["RestaurantName", "DayOfWeek"])["rho"].transform("mean")
    for f in ["resid ~ C(is_ramadan)", "resid ~ log_res", "resid ~ C(is_ramadan) + log_res"]:
        m = smf.ols(f, data=d).fit()
        k = "C(is_ramadan)[T.True]"
        c = f"{k}={m.params[k]:+.4f} (p={m.pvalues[k]:.3g})" if k in m.params else ""
        print(f"  {f:<36s} R²={m.rsquared:.4f}  {c}")
    print("\nمقایسه‌ی شام‌های رمضان با شام‌های هم‌حجمِ خارج رمضان:")
    ram_d = d[d.is_ramadan]
    lo, hi = ram_d["Res"].quantile([0.1, 0.9])
    matched = d[(~d.is_ramadan) & (d.Res.between(lo, hi))]
    print(f"  رمضان: n={len(ram_d)} Res میانه={ram_d.Res.median():.0f} ρ وزنی="
          f"{ram_d.NoRecv.sum() / ram_d.Res.sum():.4f}")
    print(f"  هم‌حجم غیررمضان: n={len(matched)} Res میانه={matched.Res.median():.0f} ρ وزنی="
          f"{matched.NoRecv.sum() / matched.Res.sum():.4f}")


def q7_friday(df: pd.DataFrame) -> None:
    header("Q7 — تکلیف جمعه (۸۶ رکورد)")
    fri = df[df.DayOfWeek == 6]
    kv("رکوردهای جمعه", f"{len(fri)} ({len(fri) / len(df):.2%})")
    kv("روزهای جمعه‌ی دارای سرو", fri["DateReserve"].nunique())
    kv("سلف‌های درگیر", fri["RestaurantName"].nunique())
    kv("ρ وزنی جمعه", f"{fri.NoRecv.sum() / fri.Res.sum():.4f}")
    kv("ρ وزنی بقیه", f"{df[df.DayOfWeek != 6].NoRecv.sum() / df[df.DayOfWeek != 6].Res.sum():.4f}")
    kv("سهم جمعه از کل Res", f"{fri.Res.sum() / df.Res.sum():.3%}")
    print("\nجمعه‌های دارای سرو و زمینه‌شان:")
    g = fri.groupby("DateReserve").agg(رکورد=("rho", "size"), Res=("Res", "sum"),
                                       سلف=("RestaurantName", "nunique"))
    g["rho"] = fri.groupby("DateReserve").apply(
        lambda x: x.NoRecv.sum() / x.Res.sum(), include_groups=False)
    print(g.round(4).to_string())
    a, b = fri["rho"], df.loc[df.DayOfWeek == 5, "rho"]
    u, p = stats.mannwhitneyu(a, b)
    print(f"\nجمعه در برابر پنجشنبه: p={p:.3g} · Cliff's δ={cliffs_delta(a.values, b.values):+.3f}")


def main() -> None:
    setup()
    df = load_dataset_with_weather()
    q1_city_or_proxy(df)
    q2_food_novelty(df)
    q3_lag14_vs_food(df)
    q4_exam_mechanism(df)
    q5_snow_holiday(df)
    q6_ramadan_volume(df)
    q7_friday(df)


if __name__ == "__main__":
    main()
