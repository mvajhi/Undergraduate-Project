"""بند ۴.۱۰ — تحلیل اختصاصی متغیرهای خارجی، دور ۱.

هر تحلیل اینجا **پس از کنترل روز هفته و سلف** انجام می‌شود (هشدار صریح WBS)، با روش
باقیمانده: به‌جای خود ρ، انحراف ρ از میانگین همان (سلف، وعده، روز هفته) بررسی می‌شود.
همچنین همه‌ی تحلیل‌های جوی **فقط روی تهران** اجرا می‌شوند مگر خلافش ذکر شود، تا اثر
بین‌شهری (که در S2/S4 نشان داده شد کاذب است) دوباره وارد نشود.

اجرا: `python -m src.eda_lib.runners.s10_external`
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.config import FIGURES_DIR
from src.eda_lib.figio import save_fig
from src.eda_lib.runners._common import (
    CALENDAR_PATH,
    EVENTS_PATH,
    header,
    kv,
    load_dataset_with_weather,
    setup,
)
from src.viz_fa import fa
from src.viz_fa import setup as viz_setup


def add_residual(df: pd.DataFrame) -> pd.DataFrame:
    """باقیمانده‌ی ρ نسبت به میانگین (سلف، وعده، روز هفته) — کنترل هم‌زمان سه عامل."""
    df = df.copy()
    df["rho_resid"] = df["rho"] - df.groupby(["RestaurantName", "Meal", "DayOfWeek"])["rho"].transform("mean")
    return df


def run_temperature(teh: pd.DataFrame) -> None:
    header("۴.۱۰.۱ دما — خطی یا U-شکل؟ (فقط تهران، پس از کنترل سلف×وعده×روزهفته)")
    d = teh.dropna(subset=["temp_min", "rho_resid"])
    bins = pd.qcut(d["temp_min"], 8, duplicates="drop")
    tab = d.groupby(bins, observed=True).agg(
        n=("rho_resid", "size"), temp=("temp_min", "mean"),
        resid=("rho_resid", "mean"), rho=("rho", "mean"))
    print(tab.round(4).to_string())
    r_lin = stats.spearmanr(d["temp_min"], d["rho_resid"])
    # آزمون U-شکل: جمله‌ی درجه‌دو در OLS روی باقیمانده
    import statsmodels.formula.api as smf
    m = smf.ols("rho_resid ~ temp_min + I(temp_min**2)", data=d).fit()
    print(f"\nSpearman خطی روی باقیمانده: r={r_lin[0]:+.4f} p={r_lin[1]:.3g}")
    print(f"جمله‌ی درجه‌دو: ضریب={m.params['I(temp_min ** 2)']:+.3e} p={m.pvalues['I(temp_min ** 2)']:.3g} "
          f"| R²={m.rsquared:.5f}")
    print("→ " + ("شواهد رابطه‌ی U-شکل" if m.pvalues["I(temp_min ** 2)"] < 0.05 and m.rsquared > 0.005
                  else "شواهد کافی برای رابطه‌ی غیرخطی معنادار وجود ندارد (R² ناچیز)"))


def run_aqi_threshold(teh: pd.DataFrame) -> None:
    header("۴.۱۰.۲ AQI — آیا اثر آستانه‌ای دارد؟ (فقط تهران، روی باقیمانده)")
    d = teh.dropna(subset=["aqi_us_max", "rho_resid"])
    cats = pd.cut(d["aqi_us_max"], [0, 100, 150, 200, 300, 1000],
                  labels=["<100 سالم/متوسط", "100-150 ناسالم برای حساس‌ها",
                          "150-200 ناسالم", "200-300 خیلی ناسالم", ">300 خطرناک"])
    tab = d.groupby(cats, observed=True).agg(
        n=("rho_resid", "size"), روز=("date_gregorian", "nunique"),
        باقیمانده=("rho_resid", "mean"), rho=("rho", "mean"))
    print(tab.round(4).to_string())
    groups = [g["rho_resid"].values for _, g in d.groupby(cats, observed=True) if len(g) >= 20]
    if len(groups) >= 2:
        h, p = stats.kruskal(*groups)
        print(f"\nKruskal-Wallis بین دسته‌های AQI: H={h:.2f} p={p:.3g}")
    for thr in [150, 200]:
        a = d.loc[d["aqi_us_max"] >= thr, "rho_resid"]
        b = d.loc[d["aqi_us_max"] < thr, "rho_resid"]
        if len(a) >= 20:
            u, p = stats.mannwhitneyu(a, b, alternative="greater")
            print(f"  AQI≥{thr} (n={len(a)}) در برابر بقیه: تفاوت میانگین باقیمانده="
                  f"{a.mean() - b.mean():+.4f}  p={p:.3g}")


def run_precipitation(teh: pd.DataFrame) -> None:
    header("۴.۱۰.۳ بارش — نوع (باران/برف) مهم‌تر است یا مقدار؟ (فقط تهران، روی باقیمانده)")
    d = teh.dropna(subset=["rho_resid"]).copy()
    d["نوع بارش"] = np.select(
        [d["snowfall_sum"] > 0.1, d["rain_sum"] > 0.5, d["precipitation_sum"] > 0],
        ["برف", "باران قابل‌توجه", "بارش ناچیز"], default="بدون بارش")
    tab = d.groupby("نوع بارش").agg(
        n=("rho_resid", "size"), روز=("date_gregorian", "nunique"),
        باقیمانده=("rho_resid", "mean"), rho=("rho", "mean"))
    print(tab.round(4).to_string())
    groups = [g["rho_resid"].values for _, g in d.groupby("نوع بارش") if len(g) >= 20]
    if len(groups) >= 2:
        h, p = stats.kruskal(*groups)
        print(f"\nKruskal-Wallis بین انواع بارش: H={h:.2f} p={p:.3g}")
    wet = d[d["precipitation_sum"] > 0]
    if len(wet) > 30:
        r = stats.spearmanr(wet["precipitation_sum"], wet["rho_resid"])
        print(f"مقدار بارش (فقط روزهای بارانی، n={len(wet)}): Spearman r={r[0]:+.4f} p={r[1]:.3g}")
    snow = d[d["snowfall_sum"] > 0.1]
    print(f"\nروزهای برفی: {snow['date_gregorian'].nunique()} روز، {len(snow)} رکورد؛ "
          f"میانگین باقیمانده={snow['rho_resid'].mean():+.4f}")
    print("→ اگر «نوع» معنادار ولی «مقدار» بی‌معنا باشد، فیچر باید دسته‌ای باشد نه پیوسته.")
    print("⚠️ یادآوری قاعده‌ی برش: بارش واقعی روز d در ساعت ۱۵ روز d−1 معلوم نیست؛ "
          "فقط پیش‌بینی هواشناسی مجاز است.")


def run_holiday_profile(df: pd.DataFrame) -> None:
    header("۴.۱۰.۴ پروفایل تعطیلات — روزهای d-3 تا d+2 حول بلوک تعطیلی")
    cal = pd.read_csv(CALENDAR_PATH, parse_dates=["date_gregorian"])
    d = df.merge(cal[["date_gregorian", "is_holiday_any", "days_to_next_holiday",
                      "days_since_last_holiday", "holiday_block_length"]],
                 on="date_gregorian", how="left")
    print("میانگین باقیمانده‌ی ρ بر حسب «روز تا تعطیلی بعدی»:")
    for k in [0, 1, 2, 3, 4, 5]:
        sub = d[d["days_to_next_holiday"] == k]
        if len(sub) >= 20:
            m, lo, hi = sub["rho_resid"].mean(), *stats.t.interval(
                0.95, len(sub) - 1, loc=sub["rho_resid"].mean(),
                scale=stats.sem(sub["rho_resid"].dropna()))
            print(f"  d−{k} (n={len(sub):>4d}): {m:+.4f}  CI95=[{lo:+.4f}, {hi:+.4f}]")
    print("\nمیانگین باقیمانده بر حسب «روز از آخرین تعطیلی»:")
    for k in [1, 2, 3, 4, 5]:
        sub = d[d["days_since_last_holiday"] == k]
        if len(sub) >= 20:
            print(f"  d+{k} (n={len(sub):>4d}): {sub['rho_resid'].mean():+.4f}")
    print("\nاثر طول بلوک تعطیلی بر روز قبل از آن:")
    pre = d[d["days_to_next_holiday"] == 1]
    if "holiday_block_length" in pre.columns and pre["holiday_block_length"].notna().any():
        print(pre.groupby("holiday_block_length")["rho_resid"].agg(["size", "mean"]).round(4).to_string())


def run_ramadan_nowruz(df: pd.DataFrame) -> None:
    header("۴.۱۰.۵ رمضان و نوروز — تصمیم درباره‌ی کنارگذاری")
    events = pd.read_csv(EVENTS_PATH, parse_dates=["date_start", "date_end"])
    ram = events[events["event_id"].str.contains("ramadan", na=False)]
    d = df.copy()
    d["is_ramadan"] = False
    for _, e in ram.iterrows():
        d.loc[(d["date_gregorian"] >= e["date_start"]) & (d["date_gregorian"] <= e["date_end"]), "is_ramadan"] = True
    kv("رکوردهای داخل رمضان", f"{int(d['is_ramadan'].sum())} ({d['is_ramadan'].mean():.1%})")
    print(d.groupby(["is_ramadan", "Meal"]).apply(
        lambda x: pd.Series({"n": len(x), "روز": x["date_gregorian"].nunique(),
                             "rho_w": x["NoRecv"].sum() / x["Res"].sum(),
                             "Res_mean": x["Res"].mean()}), include_groups=False).round(4).to_string())
    print("\n→ ناهار در رمضان: اگر تعداد روز به‌شدت کم باشد، تأیید توقف ساختاری سرو ناهار (ردیف ۲۰).")
    a = d.loc[d["is_ramadan"], "rho_resid"].dropna()
    b = d.loc[~d["is_ramadan"], "rho_resid"].dropna()
    if len(a) >= 20:
        u, p = stats.mannwhitneyu(a, b)
        print(f"باقیمانده‌ی ρ داخل رمضان در برابر بیرون: "
              f"{a.mean():+.4f} در برابر {b.mean():+.4f}  p={p:.3g}")


def run_figure(teh: pd.DataFrame) -> None:
    viz_setup()
    d = teh.dropna(subset=["temp_min", "rho_resid"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    b = pd.qcut(d["temp_min"], 10, duplicates="drop")
    g = d.groupby(b, observed=True).agg(x=("temp_min", "mean"), y=("rho_resid", "mean"),
                                        se=("rho_resid", stats.sem))
    axes[0].errorbar(g["x"], g["y"], yerr=1.96 * g["se"], fmt="o-", color="#4C72B0")
    axes[0].axhline(0, color="k", lw=.8, ls="--")
    axes[0].set_xlabel(fa("دمای کمینه (درجه سانتی‌گراد)")); axes[0].set_ylabel(fa("انحراف نرخ از میانگین"))
    axes[0].set_title(fa("دما اثر معناداری بر نرخ ندارد (تهران، کنترل‌شده)")); axes[0].grid(alpha=.3)

    da = d.dropna(subset=["aqi_us_max"])
    b2 = pd.qcut(da["aqi_us_max"], 10, duplicates="drop")
    g2 = da.groupby(b2, observed=True).agg(x=("aqi_us_max", "mean"), y=("rho_resid", "mean"),
                                           se=("rho_resid", stats.sem))
    axes[1].errorbar(g2["x"], g2["y"], yerr=1.96 * g2["se"], fmt="o-", color="#C44E52")
    axes[1].axhline(0, color="k", lw=.8, ls="--")
    axes[1].axvline(150, color="gray", ls=":", label=fa("آستانه‌ی ناسالم"))
    axes[1].set_xlabel(fa("شاخص کیفیت هوا")); axes[1].set_ylabel(fa("انحراف نرخ از میانگین"))
    axes[1].set_title(fa("آلودگی هوا هم اثر معناداری ندارد (تهران، کنترل‌شده)"))
    axes[1].legend(); axes[1].grid(alpha=.3)
    fig.tight_layout()
    print("\n" + str(save_fig(fig, "4.10_weather_controlled_v2", FIGURES_DIR)))
    plt.close(fig)


def main() -> None:
    setup()
    df = add_residual(load_dataset_with_weather())
    teh = df[df.city == "تهران"]
    header(f"n کل={len(df)} · n تهران={len(teh)}")
    run_temperature(teh)
    run_aqi_threshold(teh)
    run_precipitation(teh)
    run_holiday_profile(df)
    run_ramadan_nowruz(df)
    run_figure(teh)


if __name__ == "__main__":
    main()
