"""بند ۸.۲ WBS فاز ۸ — تحلیل باقیمانده روی پیش‌بینی‌های Test قفل‌شده (بند ۸.۱).

قهرمان رسمی پروژه (`lightgbm_quantile`، ردیف ۴۰/۴۶ decision_log) کانون این تحلیل است —
باقیمانده = $\\rho - \\hat\\rho_\\tau$ (منفی یعنی مدل بیش‌برآورد کرده). داده از
`data/interim/phase8_holdout_predictions.parquet` (ساخته‌ی بند ۸.۱) خوانده می‌شود، نه
دوباره از fit — بازبرازش فقط یک‌بار در ۸.۱ مجاز بود.

اجرا: ``python -m src.models.run_phase8_residuals``
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf

from src.config import FIGURES_DIR, REPORTS_DIR
from src.eda_lib.figio import save_fig
from src.viz_fa import fa
from src.viz_fa import setup as viz_setup

PRED_PATH = REPORTS_DIR.parent / "data" / "interim" / "phase8_holdout_predictions.parquet"
OUT_DIR = REPORTS_DIR / "phase8"
FIG_DIR = FIGURES_DIR / "phase8"
CHAMPION = "lightgbm_quantile"
DOW_NAMES = ["شنبه", "یکشنبه", "دوشنبه", "سه‌شنبه", "چهارشنبه", "پنجشنبه", "جمعه"]


def load() -> pd.DataFrame:
    df = pd.read_parquet(PRED_PATH).sort_values("date_gregorian").reset_index(drop=True)
    df["resid"] = df["rho"] - df[f"pred__{CHAMPION}"]
    df["dow"] = (pd.to_datetime(df["date_gregorian"]).dt.dayofweek + 2) % 7
    df["dow_name"] = df["dow"].map(dict(enumerate(DOW_NAMES)))
    return df


def fig_hist_qq(df: pd.DataFrame) -> tuple[str, str]:
    r = df["resid"].to_numpy()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(r, bins=40, color="#4C72B0", edgecolor="white")
    ax.axvline(0.0, color="black", linewidth=1, linestyle="--")
    ax.set_title(fa(f"توزیع باقیمانده — {CHAMPION} روی Test قفل‌شده"))
    ax.set_xlabel(fa("باقیمانده ($\\rho - \\hat\\rho$)"))
    ax.set_ylabel(fa("فراوانی"))
    p1 = save_fig(fig, "8.2_residual_hist.png", FIG_DIR)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    stats.probplot(r, dist="norm", plot=ax)
    ax.set_title(fa("Q-Q Plot باقیمانده در برابر نرمال"))
    p2 = save_fig(fig, "8.2_residual_qq.png", FIG_DIR)
    plt.close(fig)
    return str(p1.relative_to(REPORTS_DIR.parent)), str(p2.relative_to(REPORTS_DIR.parent))


def fig_resid_vs(df: pd.DataFrame) -> dict[str, str]:
    paths = {}

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df[f"pred__{CHAMPION}"], df["resid"], s=8, alpha=0.4, color="#4C72B0")
    ax.axhline(0.0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel(fa("مقدار پیش‌بینی‌شده ($\\hat\\rho$)"))
    ax.set_ylabel(fa("باقیمانده"))
    ax.set_title(fa("باقیمانده در برابر پیش‌بینی"))
    p = save_fig(fig, "8.2_resid_vs_pred.png", FIG_DIR)
    plt.close(fig)
    paths["pred"] = str(p.relative_to(REPORTS_DIR.parent))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df["Res"], df["resid"], s=8, alpha=0.4, color="#DD8452")
    ax.axhline(0.0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel(fa("Res (حجم رزرو)"))
    ax.set_ylabel(fa("باقیمانده"))
    ax.set_title(fa("باقیمانده در برابر حجم رزرو"))
    p = save_fig(fig, "8.2_resid_vs_res.png", FIG_DIR)
    plt.close(fig)
    paths["res"] = str(p.relative_to(REPORTS_DIR.parent))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(df["date_gregorian"], df["resid"], s=8, alpha=0.4, color="#55A868")
    ax.axhline(0.0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel(fa("تاریخ"))
    ax.set_ylabel(fa("باقیمانده"))
    ax.set_title(fa("باقیمانده در طول زمان — پنجره‌ی Test"))
    fig.autofmt_xdate()
    p = save_fig(fig, "8.2_resid_vs_date.png", FIG_DIR)
    plt.close(fig)
    paths["date"] = str(p.relative_to(REPORTS_DIR.parent))

    fig, ax = plt.subplots(figsize=(7, 4))
    order = df.groupby("dow", observed=True)["resid"].median().sort_index().index
    data = [df.loc[df["dow"] == d, "resid"].to_numpy() for d in order]
    labels = [fa(DOW_NAMES[d]) for d in order]
    ax.boxplot(data, tick_labels=labels)
    ax.axhline(0.0, color="black", linewidth=1, linestyle="--")
    ax.set_ylabel(fa("باقیمانده"))
    ax.set_title(fa("باقیمانده به تفکیک روز هفته"))
    p = save_fig(fig, "8.2_resid_vs_dow.png", FIG_DIR)
    plt.close(fig)
    paths["dow"] = str(p.relative_to(REPORTS_DIR.parent))

    top_rest = df["RestaurantName"].value_counts().head(12).index
    fig, ax = plt.subplots(figsize=(9, 4))
    data = [df.loc[df["RestaurantName"] == r, "resid"].to_numpy() for r in top_rest]
    ax.boxplot(data, tick_labels=[fa(str(r)) for r in top_rest])
    ax.axhline(0.0, color="black", linewidth=1, linestyle="--")
    ax.set_ylabel(fa("باقیمانده"))
    ax.set_title(fa("باقیمانده به تفکیک سلف (۱۲ پرتکرار)"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    p = save_fig(fig, "8.2_resid_vs_restaurant.png", FIG_DIR)
    plt.close(fig)
    paths["restaurant"] = str(p.relative_to(REPORTS_DIR.parent))

    return paths


def acf_ljungbox(df: pd.DataFrame, lags: int = 10) -> tuple[pd.DataFrame, dict]:
    """ACF + Ljung-Box روی سری **روزانه** باقیمانده (میانگین روی سلف×وعده) — نه ردیفی،
    چون ACF زمانی روی رکوردهای هم‌روز مستقل نیست (ICC(روز)=۰.۲۲۵، F10)."""
    daily = df.groupby("date_gregorian", observed=True)["resid"].mean().sort_index()
    n_lags = min(lags, len(daily) // 2 - 1)
    acf_vals = acf(daily.to_numpy(), nlags=n_lags, fft=False)
    lb = acorr_ljungbox(daily.to_numpy(), lags=[n_lags], return_df=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.stem(range(len(acf_vals)), acf_vals)
    ci = 1.96 / np.sqrt(len(daily))
    ax.axhline(ci, color="red", linestyle="--", linewidth=1)
    ax.axhline(-ci, color="red", linestyle="--", linewidth=1)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xlabel(fa("وقفه (روز)"))
    ax.set_ylabel(fa("خودهمبستگی"))
    ax.set_title(fa("ACF باقیمانده‌ی روزانه (میانگین روی سلف×وعده)"))
    save_fig(fig, "8.2_resid_acf.png", FIG_DIR)
    plt.close(fig)

    return daily, {
        "n_days": len(daily), "n_lags": n_lags,
        "lb_stat": float(lb["lb_stat"].iloc[0]), "lb_pvalue": float(lb["lb_pvalue"].iloc[0]),
        "white_noise": bool(lb["lb_pvalue"].iloc[0] > 0.05),
    }


def render_report(df: pd.DataFrame, hist_paths, resid_paths, lb: dict) -> str:
    r = df["resid"].to_numpy()
    skew, kurt = float(stats.skew(r)), float(stats.kurtosis(r))
    _, sw_p = stats.shapiro(r if len(r) <= 5000 else np.random.default_rng(42).choice(r, 5000, replace=False))

    verdict = ("سفید (بدون خودهمبستگی معنادار)" if lb["white_noise"]
               else "**رد شد — خودهمبستگی باقی مانده، یعنی الگوی زمانی جامانده در فیچرها**")

    lines = [
        "# بند ۸.۲ — تحلیل باقیمانده روی Test قفل‌شده",
        "",
        f"> مدل: `{CHAMPION}` (قهرمان رسمی فاز ۷). داده از `reports/phase8/8.1_*` (بند ۸.۱، بدون بازبرازش تازه).",
        "",
        "## توزیع باقیمانده",
        "",
        f"- میانگین: {r.mean():+.5f} · انحراف معیار: {r.std():.5f} · چولگی: {skew:+.3f} · کشیدگی: {kurt:+.3f}",
        f"- آزمون نرمالیتی Shapiro-Wilk: p={sw_p:.2e} ({'رد نرمالیتی' if sw_p < 0.05 else 'رد نمی‌شود'})",
        f"- هیستوگرام: `{hist_paths[0]}` · Q-Q Plot: `{hist_paths[1]}`",
        "",
        "## باقیمانده در برابر ابعاد دیگر",
        "",
        f"- در برابر پیش‌بینی: `{resid_paths['pred']}`",
        f"- در برابر Res: `{resid_paths['res']}`",
        f"- در برابر تاریخ: `{resid_paths['date']}`",
        f"- به تفکیک روز هفته: `{resid_paths['dow']}`",
        f"- به تفکیک سلف (۱۲ پرتکرار): `{resid_paths['restaurant']}`",
        "",
        "## ACF + Ljung-Box (سری روزانه)",
        "",
        f"- تعداد روز: {lb['n_days']} · وقفه: {lb['n_lags']} · آماره‌ی Ljung-Box: {lb['lb_stat']:.3f} · p={lb['lb_pvalue']:.4f}",
        f"- نتیجه: {verdict}",
        f"- نمودار: `reports/figures/phase8/8.2_resid_acf.png`",
    ]
    return "\n".join(lines)


def main() -> None:
    viz_setup()
    df = load()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    hist_paths = fig_hist_qq(df)
    resid_paths = fig_resid_vs(df)
    daily, lb = acf_ljungbox(df)

    report = render_report(df, hist_paths, resid_paths, lb)
    (OUT_DIR / "8.2_residual_analysis.md").write_text(report + "\n")
    daily.to_csv(OUT_DIR / "8.2_daily_residual_series.csv")
    print(report)
    print(f"\nذخیره شد در {OUT_DIR}/8.2_residual_analysis.md")


if __name__ == "__main__":
    main()
