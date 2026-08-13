"""بند ۴.۳ — تحلیل سری زمانی (ترسیم، STL، ACF/PACF، ایستایی، نقطه‌ی شکست)، دور ۱.

⚠️ دو ملاحظه‌ی خاص این داده که در تفسیر هر خروجی زیر باید حاضر باشد:
- **شکاف ساختاری رمضان:** ۲۹ روز (۲۲ اسفند ۱۴۰۲ – ۲۱ فروردین ۱۴۰۳) هیچ ناهاری سرو
  نشده (ردیف ۲۰ decision_log). این «داده‌ی گمشده» نیست، «سرو نشده» است — ولی برای
  STL/ACF که فاصله‌ی منظم می‌خواهند باید پر شود، پس هر نتیجه‌ای که به این بازه تکیه
  کند مشکوک است.
- **۱۴۲ روز داده:** فصلی‌بودن **هفتگی** قابل‌تخمین است؛ فصلی‌بودن سالانه/ترمی **نیست**.

اجرا: `python -m src.eda_lib.runners.s03_timeseries`
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

from src.config import FIGURES_DIR
from src.eda_lib.correlation_helpers import save_fig
from src.eda_lib.runners._common import header, kv, load_dataset, setup
from src.eda_lib.timeseries_helpers import (
    cusum,
    daily_university_series,
    group_daily_series,
    seasonal_strength,
    top_volume_groups,
    trend_strength,
)
from src.viz_fa import fa
from src.viz_fa import setup as viz_setup


def run_series(df: pd.DataFrame) -> pd.DataFrame:
    header("۴.۳.۱ سری زمانی تجمیعی دانشگاه")
    daily = daily_university_series(df)
    kv("طول بازه‌ی تقویمی (روز)", len(daily))
    kv("روزهای دارای سرو", int(daily["rho"].notna().sum()))
    kv("روزهای درون‌یابی‌شده", int(daily["is_interpolated"].sum()))
    kv("ρ روزانه: میانگین/میانه/بیشینه",
       f"{daily['rho'].mean():.4f} / {daily['rho'].median():.4f} / {daily['rho'].max():.4f}")
    top = daily["rho"].nlargest(8)
    print("\n۸ روز با بالاترین ρ روزانه (کاندید بند ۴.۵):")
    for d, v in top.items():
        print(f"  {d.date()}  ρ={v:.4f}")
    print("\n۵ روز با کمترین ρ:")
    for d, v in daily["rho"].nsmallest(5).items():
        print(f"  {d.date()}  ρ={v:.4f}")

    header("پرحجم‌ترین ترکیب‌های (سلف، وعده)", 2)
    for r, m in top_volume_groups(df, 6):
        s = group_daily_series(df, r, m)
        print(f"  {r} × {m}: n={len(s)} میانگین={s.mean():.4f} std={s.std():.4f}")
    return daily


def run_stl(daily: pd.DataFrame) -> None:
    header("۴.۳.۲ تجزیه‌ی STL (period=7)")
    y = daily["rho_interp"].dropna()
    res = STL(y, period=7, robust=True).fit()
    kv("قدرت فصلی هفتگی (Wang-Smith-Hyndman)", f"{seasonal_strength(res):.4f}")
    kv("قدرت روند", f"{trend_strength(res):.4f}")
    kv("انحراف معیار جزء فصلی", f"{res.seasonal.std():.4f}")
    kv("انحراف معیار باقیمانده", f"{res.resid.std():.4f}")
    print("\nالگوی فصلی متوسط به تفکیک روز هفته (از جزء seasonal):")
    seas = pd.DataFrame({"seasonal": res.seasonal, "dow": res.seasonal.index.dayofweek})
    # pandas: Monday=0 ... Sunday=6 → تبدیل به قرارداد شمسی شنبه=0
    seas["dow_fa"] = (seas["dow"] + 2) % 7
    names = {0: "شنبه", 1: "یکشنبه", 2: "دوشنبه", 3: "سه‌شنبه", 4: "چهارشنبه", 5: "پنجشنبه", 6: "جمعه"}
    agg = seas.groupby("dow_fa")["seasonal"].mean().rename(index=names)
    print(agg.round(5).to_string())

    header("باقیمانده‌های بزرگ STL (پرت زمانی، ورودی بند ۴.۵)", 2)
    z = (res.resid - res.resid.mean()) / res.resid.std()
    big = z[abs(z) > 3].sort_values()
    print(f"تعداد |z|>3: {len(big)}")
    for d, v in big.items():
        print(f"  {d.date()}  z={v:+.2f}  ρ={daily.loc[d, 'rho']:.4f}"
              f"{'  (درون‌یابی‌شده)' if daily.loc[d, 'is_interpolated'] else ''}")


def run_acf(daily: pd.DataFrame) -> None:
    header("۴.۳.۳ خودهمبستگی (ACF/PACF تا lag 30)")
    y = daily["rho_interp"].dropna()
    a = acf(y, nlags=30, fft=False)
    p = pacf(y, nlags=30)
    ci = 1.96 / np.sqrt(len(y))
    kv("مرز معناداری ±1.96/√n", f"±{ci:.4f}")
    print("\nlag :   ACF     PACF   (* = خارج از مرز)")
    for k in range(1, 31):
        mark = "*" if abs(a[k]) > ci else " "
        markp = "*" if abs(p[k]) > ci else " "
        print(f"{k:3d} : {a[k]:+.4f}{mark}  {p[k]:+.4f}{markp}")
    print("\nقله‌های هفتگی مورد انتظار:")
    for k in [7, 14, 21, 28]:
        kv(f"  ACF(lag {k})", f"{a[k]:+.4f} {'✅ معنادار' if abs(a[k]) > ci else '❌ غیرمعنادار'}")

    from statsmodels.stats.diagnostic import acorr_ljungbox
    lb = acorr_ljungbox(y, lags=[7, 14, 21, 30], return_df=True)
    print("\nLjung-Box:")
    print(lb.to_string())


def run_stationarity(daily: pd.DataFrame) -> None:
    header("۴.۳.۴ ایستایی (ADF + KPSS)")
    y = daily["rho_interp"].dropna()
    adf = adfuller(y, autolag="AIC")
    print(f"ADF (H0: ریشه‌ی واحد / ناایستا): stat={adf[0]:.4f} p={adf[1]:.4g} "
          f"→ {'ایستا (H0 رد شد)' if adf[1] < 0.05 else 'ناایستا'}")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kp = kpss(y, regression="c", nlags="auto")
    print(f"KPSS (H0: ایستا): stat={kp[0]:.4f} p={kp[1]:.4g} "
          f"→ {'ناایستا (H0 رد شد)' if kp[1] < 0.05 else 'ایستا'}")
    print("\nنتیجه‌ی ترکیبی: " + (
        "هر دو آزمون ایستایی را تأیید می‌کنند → نیازی به تفاضل‌گیری نیست"
        if adf[1] < 0.05 and kp[1] >= 0.05 else
        "آزمون‌ها هم‌راستا نیستند → سری احتمالاً trend-stationary یا دارای شکست ساختاری است"))


def run_changepoints(daily: pd.DataFrame, df: pd.DataFrame) -> None:
    header("۴.۳.۵ نقاط شکست و تغییر رژیم")
    y = daily["rho_interp"].dropna()
    try:
        import ruptures as rpt
        for pen in [0.5, 1.0, 2.0]:
            algo = rpt.Pelt(model="rbf", min_size=10).fit(y.values)
            bkps = algo.predict(pen=pen)
            dates = [y.index[min(b, len(y) - 1)].date() for b in bkps[:-1]]
            print(f"PELT (rbf, pen={pen}): {len(bkps) - 1} نقطه‌ی شکست → {dates}")
    except ImportError:
        print("[ruptures نصب نیست — فقط CUSUM]")

    c = cusum(y)
    kv("بیشینه‌ی |CUSUM|", f"{c.abs().max():.4f} در {c.abs().idxmax().date()}")

    print("\nمیانگین ρ وزنی در بلوک‌های تقویمی (بررسی دستی رژیم):")
    df = df.copy()
    df["block"] = pd.cut(df["date_gregorian"],
                         bins=pd.to_datetime(["2023-11-01", "2023-12-22", "2024-01-21",
                                              "2024-02-20", "2024-03-20", "2024-04-20", "2024-05-25"]),
                         labels=["آذر", "دی", "بهمن", "اسفند", "فروردین(رمضان)", "اردیبهشت-خرداد"])
    print(df.groupby("block", observed=True).apply(
        lambda x: pd.Series({"n": len(x), "rho_w": x["NoRecv"].sum() / x["Res"].sum(),
                             "Res_total": x["Res"].sum()}), include_groups=False).round(4).to_string())


def run_figures(daily: pd.DataFrame, df: pd.DataFrame) -> None:
    viz_setup()
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    axes[0].plot(daily.index, daily["rho"], lw=1.2, color="#4C72B0")
    axes[0].scatter(daily.index[daily["is_interpolated"]], daily["rho_interp"][daily["is_interpolated"]],
                    s=8, color="#C44E52", label=fa("روز بدون سرو (درون‌یابی)"))
    axes[0].set_title(fa("نرخ عدم‌دریافت روزانه‌ی کل دانشگاه"))
    axes[0].set_ylabel(fa("نرخ عدم‌دریافت")); axes[0].legend(); axes[0].grid(alpha=.3)

    y = daily["rho_interp"].dropna()
    res = STL(y, period=7, robust=True).fit()
    axes[1].plot(res.trend.index, res.trend, color="#55A868")
    axes[1].set_title(fa("جزء روند (STL)")); axes[1].grid(alpha=.3)
    axes[2].plot(res.resid.index, res.resid, color="#8172B2", lw=.9)
    axes[2].axhline(0, color="k", lw=.6)
    axes[2].set_title(fa("باقیمانده‌ی STL — قله‌ها همان روزهای پرت هستند"))
    axes[2].set_xlabel(fa("تاریخ")); axes[2].grid(alpha=.3)
    fig.tight_layout()
    print("\n" + str(save_fig(fig, "4.3_daily_series_stl", FIGURES_DIR)))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    a = acf(y, nlags=30, fft=False)
    ci = 1.96 / np.sqrt(len(y))
    ax.bar(range(31), a, color=["#C44E52" if k % 7 == 0 and k > 0 else "#4C72B0" for k in range(31)])
    ax.axhline(ci, ls="--", color="gray"); ax.axhline(-ci, ls="--", color="gray")
    ax.set_title(fa("خودهمبستگی نرخ عدم‌دریافت — قله‌های هفتگی (قرمز) در تأخیر ۷، ۱۴، ۲۱، ۲۸"))
    ax.set_xlabel(fa("تأخیر (روز)")); ax.set_ylabel(fa("ضریب خودهمبستگی"))
    fig.tight_layout()
    print(save_fig(fig, "4.3_acf", FIGURES_DIR))
    plt.close(fig)


def main() -> None:
    setup()
    df = load_dataset()
    daily = run_series(df)
    run_stl(daily)
    run_acf(daily)
    run_stationarity(daily)
    run_changepoints(daily, df)
    run_figures(daily, df)


if __name__ == "__main__":
    main()
