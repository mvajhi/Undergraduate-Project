"""بند ۴.۱۲ — گزارش تصویری EDA: ۱۲ نمودار کلیدی برای گزارش نهایی.

استاندارد بند ۴.۱۲ WBS: عنوان هر نمودار **یک جمله‌ی خبری فارسی** است که یافته را
می‌گوید (نه «توزیع rho»)، محورها فارسی برچسب‌خورده‌اند، و هر نمودار به یک ردیف
`doc/data_facts_register.md` وصل است. خروجی: `reports/figures/report_NN_*.png`.

اجرا: `python -m src.eda_lib.runners.s12_visual_report`
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.config import FIGURES_DIR
from src.eda_lib.figio import save_fig
from src.eda_lib.individual_helpers import lorenz_curve
from src.eda_lib.runners._common import (
    CALENDAR_PATH,
    PERSON_DIM_PATH,
    header,
    load_dataset,
    load_dataset_with_weather,
    setup,
)
from src.eda_lib.runners.s13_individual import load_fact
from src.viz_fa import fa
from src.viz_fa import setup as viz_setup

BLUE, RED, GREEN, ORANGE, PURPLE = "#4C72B0", "#C44E52", "#55A868", "#DD8452", "#8172B2"


def _finish(fig, name, caption):
    path = save_fig(fig, name, FIGURES_DIR)
    plt.close(fig)
    print(f"  {path.name:<44s} {caption}")
    return path


def fig01_target(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].hist(df["rho"], bins=90, color=BLUE, edgecolor="white")
    axes[0].axvline(df["rho"].median(), color=RED, ls="--",
                    label=fa(f"میانه {df['rho'].median():.3f}"))
    axes[0].set_xlabel(fa("نرخ عدم‌دریافت")); axes[0].set_ylabel(fa("تعداد رکورد"))
    axes[0].legend(); axes[0].set_title(fa("توزیع کلی — چوله به راست با دُم سنگین"))
    xs = np.sort(df["rho"].values)
    axes[1].plot(xs, np.arange(1, len(xs) + 1) / len(xs), color=RED, lw=2)
    axes[1].axhline(0.049, color="gray", ls=":", label=fa("۴.۹٪ رکوردها نرخ صفر دارند"))
    axes[1].set_xlabel(fa("نرخ عدم‌دریافت")); axes[1].set_ylabel(fa("نسبت تجمعی"))
    axes[1].legend(); axes[1].grid(alpha=.3)
    axes[1].set_title(fa("صدک ۹۹ برابر ۰.۴۰ است — پنج برابر میانه"))
    fig.suptitle(fa("نرخ عدم‌دریافت غذا: میانه ۸ درصد، ولی دُمی که تا ۱۰۰ درصد می‌رسد"),
                 fontsize=13, y=1.02)
    fig.tight_layout()
    return _finish(fig, "report_01_target_distribution", "F02/F03 — شکل متغیر هدف")


def fig02_city(df):
    g = (df.groupby("city").apply(lambda x: x["NoRecv"].sum() / x["Res"].sum(), include_groups=False)
           .sort_values())
    n = df.groupby("city").size()
    fig, ax = plt.subplots(figsize=(9.5, 5))
    colors = [RED if c == "تهران" else BLUE for c in g.index]
    bars = ax.barh([fa(c) for c in g.index], g.values * 100, color=colors)
    for b, c in zip(bars, g.index):
        ax.text(b.get_width() + 0.1, b.get_y() + b.get_height() / 2,
                f"{b.get_width():.1f}%  (n={n[c]})", va="center", fontsize=9)
    ax.set_xlabel(fa("نرخ عدم‌دریافت (درصد)")); ax.set_xlim(0, 11)
    ax.set_title(fa("پردیس‌های خارج تهران نرخ عدم‌دریافت به‌مراتب کمتری دارند — تا ۲.۸ برابر"))
    fig.tight_layout()
    return _finish(fig, "report_02_city_effect", "F12 — بزرگ‌ترین عامل کشف‌شده")


def fig03_aqi_spurious(dfw):
    teh = dfw[dfw.city == "تهران"].copy()
    teh["resid"] = teh["rho"] - teh.groupby(["RestaurantName", "Meal", "DayOfWeek"])["rho"].transform("mean")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    city_ag = dfw.groupby("city").agg(aqi=("aqi_us_max", "mean")).join(
        dfw.groupby("city").apply(lambda x: x["NoRecv"].sum() / x["Res"].sum(),
                                  include_groups=False).rename("rho"))
    axes[0].scatter(city_ag["aqi"], city_ag["rho"] * 100, s=140, color=RED, zorder=3)
    # رضوانشهر و فومن تقریباً روی هم می‌افتند (AQI ~۴۶-۴۹، نرخ ~۳.۱٪) — برچسبشان
    # به دو جهت مخالف جابه‌جا می‌شود تا خوانا بماند.
    offsets = {"رضوانشهر": (8, 10), "فومن": (8, -14)}
    for c, r in city_ag.iterrows():
        axes[0].annotate(fa(c), (r["aqi"], r["rho"] * 100), textcoords="offset points",
                         xytext=offsets.get(c, (8, 5)), fontsize=9)
    axes[0].set_xlabel(fa("میانگین شاخص کیفیت هوا")); axes[0].set_ylabel(fa("نرخ عدم‌دریافت (درصد)"))
    axes[0].grid(alpha=.3)
    axes[0].set_title(fa("بین شهرها: رابطه‌ای قوی به‌نظر می‌رسد"))

    b = pd.qcut(teh["aqi_us_max"], 10, duplicates="drop")
    gg = teh.groupby(b, observed=True).agg(x=("aqi_us_max", "mean"), y=("resid", "mean"),
                                           se=("resid", stats.sem))
    axes[1].errorbar(gg["x"], gg["y"] * 100, yerr=1.96 * gg["se"] * 100, fmt="o-", color=BLUE)
    axes[1].axhline(0, color="k", lw=.9, ls="--")
    axes[1].set_xlabel(fa("شاخص کیفیت هوا")); axes[1].set_ylabel(fa("انحراف نرخ از میانگین (درصد)"))
    axes[1].grid(alpha=.3)
    axes[1].set_title(fa("داخل تهران: هیچ رابطه‌ای نیست (p=۰.۶۵)"))
    fig.suptitle(fa("«اثر آلودگی هوا» یک همبستگی کاذب بود: تهران هم‌زمان آلوده‌تر و پرنرخ‌تر است"),
                 fontsize=13, y=1.03)
    fig.tight_layout()
    return _finish(fig, "report_03_aqi_spurious", "F25 — مهم‌ترین یافته‌ی روش‌شناختی")


def fig04_variance(df):
    p_bar = df["NoRecv"].sum() / df["Res"].sum()
    d = df.copy()
    d["bin"] = pd.cut(d["Res"], [0, 20, 50, 100, 200, 400, 800, 2000])
    t = d.groupby("bin", observed=True).agg(res=("Res", "median"), var=("rho", "var"))
    t["binom"] = p_bar * (1 - p_bar) / t["res"]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.loglog(t["res"], t["var"], "o-", lw=2.2, ms=8, color=RED, label=fa("واریانس واقعی"))
    ax.loglog(t["res"], t["binom"], "s--", lw=2, ms=7, color=BLUE, label=fa("واریانس تحت فرض استقلال"))
    for x, y1, y2 in zip(t["res"], t["var"], t["binom"]):
        ax.annotate(f"{y1 / y2:.0f}×", (x, np.sqrt(y1 * y2)), ha="center", fontsize=9, color="gray")
    ax.set_xlabel(fa("اندازه‌ی رزرو")); ax.set_ylabel(fa("واریانس نرخ عدم‌دریافت"))
    ax.legend(); ax.grid(alpha=.3, which="both")
    ax.set_title(fa("واریانس واقعی ۴ تا ۱۶ برابر بیشتر از فرض استقلال است — و فاصله با بزرگ‌شدن رزرو زیاد می‌شود"))
    fig.tight_layout()
    return _finish(fig, "report_04_overdispersion", "F07/F08 — پایه‌ی طراحی کوانتایل")


def fig05_meal_type(df):
    piv = df.pivot_table(index="RestaurantType", columns="Meal", values=["NoRecv", "Res"], aggfunc="sum")
    rate = (piv["NoRecv"] / piv["Res"]) * 100
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(2); w = 0.36
    labels = {"daneshgah": "سلف دانشکده‌ای", "khabgah": "سلف خوابگاهی"}
    ax.bar(x - w / 2, [rate.loc[r, "lunch"] for r in rate.index], w, label=fa("ناهار"), color=ORANGE)
    ax.bar(x + w / 2, [rate.loc[r, "dinner"] for r in rate.index], w, label=fa("شام"), color=PURPLE)
    for i, r in enumerate(rate.index):
        ax.text(i - w / 2, rate.loc[r, "lunch"] + .12, f"{rate.loc[r, 'lunch']:.1f}%", ha="center")
        ax.text(i + w / 2, rate.loc[r, "dinner"] + .12, f"{rate.loc[r, 'dinner']:.1f}%", ha="center")
    ax.set_xticks(x); ax.set_xticklabels([fa(labels[r]) for r in rate.index])
    ax.set_ylabel(fa("نرخ عدم‌دریافت (درصد)")); ax.legend()
    ax.set_title(fa("ناهار همیشه پرنرخ‌تر از شام است، ولی شکاف در سلف دانشکده‌ای ۲.۴ برابر بزرگ‌تر است"))
    fig.tight_layout()
    return _finish(fig, "report_05_meal_x_type", "F17/F42 — تعامل وعده×نوع سلف")


def fig06_dow(df):
    order = [0, 1, 2, 3, 4, 5, 6]
    names = {0: "شنبه", 1: "یکشنبه", 2: "دوشنبه", 3: "سه‌شنبه", 4: "چهارشنبه", 5: "پنجشنبه", 6: "جمعه"}
    g = df.groupby("DayOfWeek").apply(lambda x: x["NoRecv"].sum() / x["Res"].sum(), include_groups=False)
    n = df.groupby("DayOfWeek").size()
    fig, ax = plt.subplots(figsize=(9.5, 5))
    cols = [RED if d == 4 else BLUE for d in order]
    ax.bar([fa(names[d]) for d in order], [g[d] * 100 for d in order], color=cols)
    for i, d in enumerate(order):
        ax.text(i, g[d] * 100 + .1, f"{g[d] * 100:.1f}%\n(n={n[d]})", ha="center", fontsize=8.5)
    ax.set_ylabel(fa("نرخ عدم‌دریافت (درصد)")); ax.set_ylim(0, 12)
    ax.set_title(fa("چهارشنبه بیشترین و یکشنبه کمترین نرخ را دارد — جمعه فقط ۸۶ رکورد دارد"))
    fig.tight_layout()
    return _finish(fig, "report_06_day_of_week", "F18 — الگوی هفتگی")


def fig07_holiday(dfw):
    cal = pd.read_csv(CALENDAR_PATH, parse_dates=["date_gregorian"])
    d = dfw.merge(cal[["date_gregorian", "days_to_next_holiday"]], on="date_gregorian", how="left")
    d["resid"] = d["rho"] - d.groupby(["RestaurantName", "Meal", "DayOfWeek"])["rho"].transform("mean")
    ks, ms, los, his = [], [], [], []
    for k in [5, 4, 3, 2, 1]:
        s = d.loc[d["days_to_next_holiday"] == k, "resid"].dropna()
        if len(s) >= 20:
            lo, hi = stats.t.interval(0.95, len(s) - 1, loc=s.mean(), scale=stats.sem(s))
            ks.append(f"d−{k}"); ms.append(s.mean() * 100); los.append(lo * 100); his.append(hi * 100)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(ks, ms, yerr=[np.array(ms) - np.array(los), np.array(his) - np.array(ms)],
                fmt="o-", lw=2.2, ms=9, color=RED, capsize=5)
    ax.axhline(0, color="k", ls="--", lw=.9)
    ax.set_xlabel(fa("فاصله تا تعطیلی بعدی")); ax.set_ylabel(fa("انحراف نرخ از میانگین (واحد درصد)"))
    ax.grid(alpha=.3)
    ax.set_title(fa("درست یک روز پیش از تعطیلی، نرخ عدم‌دریافت ۳.۱ واحد درصد بالا می‌پرد"))
    fig.tight_layout()
    return _finish(fig, "report_07_pre_holiday", "F19 — بزرگ‌ترین اثر تقویمی")


def fig08_acf_by_meal(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, meal, title, col in [
        (axes[0], "lunch", "ناهار: ریتم دوهفتگی", ORANGE),
        (axes[1], "dinner", "شام: پایداری روزبه‌روز", PURPLE)]:
        sub = df[df.Meal == meal]
        s = sub.groupby("date_gregorian").apply(
            lambda x: x["NoRecv"].sum() / x["Res"].sum(), include_groups=False)
        s.index = pd.to_datetime(s.index)
        s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"))
        rs, sig = [], []
        for k in range(1, 29):
            a, b = s.values[:-k], s.values[k:]
            m = ~(np.isnan(a) | np.isnan(b))
            r, p = stats.pearsonr(a[m], b[m]) if m.sum() >= 20 else (np.nan, 1)
            rs.append(r); sig.append(p < 0.05)
        ax.bar(range(1, 29), rs, color=[col if s_ else "#cccccc" for s_ in sig])
        ax.axhline(0, color="k", lw=.8)
        for k in (7, 14, 28):
            ax.axvline(k, color="gray", ls=":", lw=.9)
        ax.set_xlabel(fa("تأخیر (روز)")); ax.set_title(fa(title))
        ax.grid(alpha=.25, axis="y")
    axes[0].set_ylabel(fa("خودهمبستگی"))
    fig.suptitle(fa("ناهار و شام دو ساختار زمانی کاملاً متفاوت دارند — پس فیچر تأخیر باید جدا ساخته شود"),
                 fontsize=13, y=1.03)
    fig.tight_layout()
    return _finish(fig, "report_08_acf_by_meal", "F33/F34 — طراحی فیچرهای lag")


def fig09_daily_series(df):
    daily = df.groupby("date_gregorian").agg(Res=("Res", "sum"), NoRecv=("NoRecv", "sum"))
    daily["rho"] = daily["NoRecv"] / daily["Res"]
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(daily.index, daily["rho"] * 100, lw=1.4, color=BLUE)
    top = daily["rho"].nlargest(5)
    axes[0].scatter(top.index, top.values * 100, color=RED, zorder=4, s=55)
    for d, v in top.items():
        axes[0].annotate(d.strftime("%m-%d"), (d, v * 100), textcoords="offset points",
                         xytext=(0, 9), ha="center", fontsize=8.5, color=RED)
    axes[0].set_ylabel(fa("نرخ عدم‌دریافت (درصد)")); axes[0].grid(alpha=.3)
    axes[0].set_title(fa("پنج روز با بالاترین نرخ (قرمز) — همگی روزهای کم‌حجم‌اند، نه روزهای پرهدررفت"))
    axes[1].fill_between(daily.index, daily["Res"], color=GREEN, alpha=.55)
    axes[1].scatter(top.index, daily.loc[top.index, "Res"], color=RED, zorder=4, s=40)
    axes[1].set_ylabel(fa("تعداد کل رزرو")); axes[1].set_xlabel(fa("تاریخ")); axes[1].grid(alpha=.3)
    fig.tight_layout()
    return _finish(fig, "report_09_daily_series_volume", "F54 — اصلاح تفسیر روزهای پرت")


def fig10_dorm(per_dim):
    d = per_dim
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    a = d.loc[d.is_dorm_resident == True, "rate"]
    b = d.loc[d.is_dorm_resident == False, "rate"]
    axes[0].hist(b, bins=60, range=(0, .6), alpha=.62, density=True, color=BLUE, label=fa("غیرساکن خوابگاه"))
    axes[0].hist(a, bins=60, range=(0, .6), alpha=.62, density=True, color=GREEN, label=fa("ساکن خوابگاه"))
    axes[0].axvline(b.mean(), color=BLUE, ls="--"); axes[0].axvline(a.mean(), color=GREEN, ls="--")
    axes[0].set_xlabel(fa("نرخ عدم‌دریافت شخصی")); axes[0].set_ylabel(fa("چگالی"))
    axes[0].legend(); axes[0].set_title(fa(f"میانگین {a.mean():.3f} در برابر {b.mean():.3f}"))

    rows = []
    for col, g in d.dropna(subset=["CollegeName"]).groupby("CollegeName"):
        x = g.loc[g.is_dorm_resident == True, "rate"]
        y = g.loc[g.is_dorm_resident == False, "rate"]
        if len(x) >= 30 and len(y) >= 30:
            rows.append((col, x.mean(), y.mean()))
    r = pd.DataFrame(rows, columns=["c", "dorm", "non"]).sort_values("non")
    axes[1].scatter(r["non"] * 100, r["dorm"] * 100, s=45, color=PURPLE, zorder=3)
    lim = [min(r[["non", "dorm"]].min()) * 100 - .5, max(r[["non", "dorm"]].max()) * 100 + .5]
    axes[1].plot(lim, lim, "k--", lw=1)
    axes[1].set_xlim(lim); axes[1].set_ylim(lim)
    axes[1].set_xlabel(fa("نرخ غیرساکنان (درصد)")); axes[1].set_ylabel(fa("نرخ ساکنان خوابگاه (درصد)"))
    axes[1].grid(alpha=.3)
    axes[1].set_title(fa("۲۲ از ۲۴ دانشکده زیر خط قرار می‌گیرند"))
    fig.suptitle(fa("ساکنان خوابگاه در تقریباً هر دانشکده نرخ عدم‌دریافت کمتری دارند"),
                 fontsize=13, y=1.03)
    fig.tight_layout()
    return _finish(fig, "report_10_dorm_resident", "F16 — قوی‌ترین یافته‌ی جمعیتی")


def fig11_persistence(fact):
    fs = fact.sort_values("date_gregorian")
    mid = fs["date_gregorian"].quantile(0.5)
    h1 = fs[fs.date_gregorian < mid].groupby("PersonId")["dont_receive"].agg(["mean", "size"])
    h2 = fs[fs.date_gregorian >= mid].groupby("PersonId")["dont_receive"].agg(["mean", "size"])
    b = h1.join(h2, lsuffix="_1", rsuffix="_2", how="inner")
    b = b[(b["size_1"] >= 10) & (b["size_2"] >= 10)]
    q = pd.qcut(b["mean_1"], 5, labels=["۲۰٪ بهترین", "دوم", "میانی", "چهارم", "۲۰٪ بدترین"])
    g = b.assign(q=q).groupby("q", observed=True).agg(h1=("mean_1", "mean"), h2=("mean_2", "mean"))
    fig, ax = plt.subplots(figsize=(9.5, 5))
    x = np.arange(len(g)); w = .37
    ax.bar(x - w / 2, g["h1"] * 100, w, label=fa("نیمه‌ی اول بازه"), color=BLUE)
    ax.bar(x + w / 2, g["h2"] * 100, w, label=fa("نیمه‌ی دوم بازه"), color=RED)
    ax.set_xticks(x); ax.set_xticklabels([fa(str(i)) for i in g.index])
    ax.set_xlabel(fa("گروه‌بندی دانشجویان بر اساس رفتارشان در نیمه‌ی اول"))
    ax.set_ylabel(fa("نرخ عدم‌دریافت (درصد)")); ax.legend()
    r = stats.pearsonr(b["mean_1"], b["mean_2"])[0]
    ax.set_title(fa(f"رفتار فردی پایدار است: همبستگی دو نیمه {r:.2f} — بدترین گروه بدترین می‌ماند"))
    fig.tight_layout()
    return _finish(fig, "report_11_person_persistence", "F46 — توجیه اصلی مدل B")


def fig12_lorenz(fact):
    per = fact.groupby("PersonId")["dont_receive"].sum()
    x, y, gini = lorenz_curve(per.values)
    fig, ax = plt.subplots(figsize=(7.6, 6))
    ax.plot(x * 100, y * 100, lw=2.6, color=RED)
    ax.plot([0, 100], [0, 100], "k--", lw=1.2, label=fa("توزیع کاملاً برابر"))
    ax.fill_between(x * 100, y * 100, x * 100, alpha=.16, color=RED)
    for frac, lbl in [(0.90, "۱۰٪ بدترین"), (0.80, "۲۰٪ بدترین")]:
        yi = np.interp(frac, x, y)
        ax.annotate(fa(f"{lbl} ⇐ {100 - yi * 100:.0f}٪ کل عدم‌دریافت"),
                    (frac * 100, yi * 100), textcoords="offset points", xytext=(-150, 14),
                    fontsize=9.5, arrowprops=dict(arrowstyle="->", color="gray"))
        ax.axvline(frac * 100, color="gray", ls=":", lw=.9)
    ax.set_xlabel(fa("درصد تجمعی دانشجویان (از کم‌مصرف‌ترین)"))
    ax.set_ylabel(fa("درصد تجمعی موارد عدم‌دریافت"))
    ax.legend(loc="upper left"); ax.grid(alpha=.3)
    ax.set_title(fa(f"۱۰ درصد دانشجویان ۳۸ درصد کل هدررفت را می‌سازند (جینی {gini:.2f})"))
    fig.tight_layout()
    return _finish(fig, "report_12_lorenz", "F47 — تمرکز عدم‌دریافت")


def main():
    setup()
    viz_setup()
    header("۴.۱۲ — تولید ۱۲ نمودار کلیدی گزارش")
    df = load_dataset()
    dfw = load_dataset_with_weather()
    fig01_target(df); fig02_city(df); fig03_aqi_spurious(dfw); fig04_variance(df)
    fig05_meal_type(df); fig06_dow(df); fig07_holiday(dfw); fig08_acf_by_meal(df)
    fig09_daily_series(df)

    fact = load_fact()
    dim = pd.read_csv(PERSON_DIM_PATH)
    per = fact.groupby("PersonId").agg(n=("dont_receive", "size"), rate=("dont_receive", "mean"))
    per_dim = per[per["n"] >= 10].join(dim.set_index("PersonId"))
    fig10_dorm(per_dim); fig11_persistence(fact); fig12_lorenz(fact)
    print("\n۱۲ نمودار در reports/figures/report_*.png ذخیره شد.")


if __name__ == "__main__":
    main()
