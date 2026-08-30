"""بازتولید شکل (۴-۳) — هیستوگرام و ACF باقیمانده. final_report/img/residual_hist.png
و .../resid_acf.png symlink به خروجی این اسکریپت‌اند — reports/figures/ منبع حقیقت
تصاویر گزارش است (tools/report_figures/README.md).

منبع: data/interim/phase8_holdout_predictions.parquet (همان بند ۸.۱، بدون بازبرازش).
برخلاف src/models/run_phase8_residuals.py (که برای reports/figures/phase8/ عنوان‌دار
رسم می‌کند)، این نسخه بی‌عنوان است چون کپشن لاتک همان اطلاعات را می‌دهد.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf

from src import viz_fa  # noqa: F401  — فونت فارسی و patch متن
from src.viz_fa import fa

PRED_PATH = "data/interim/phase8_holdout_predictions.parquet"
OUT = "reports/figures/phase8"
CHAMPION = "lightgbm_quantile"


def load() -> pd.DataFrame:
    df = pd.read_parquet(PRED_PATH).sort_values("date_gregorian").reset_index(drop=True)
    df["resid"] = df["rho"] - df[f"pred__{CHAMPION}"]
    return df


def main() -> None:
    df = load()
    r = df["resid"].to_numpy()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(r, bins=40, color="#4C72B0", edgecolor="white")
    ax.axvline(0.0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel(fa("باقیمانده (ρ − ρ̂)"))
    ax.set_ylabel(fa("فراوانی"))
    fig.tight_layout()
    fig.savefig(f"{OUT}/8.2_residual_hist_fa.png", dpi=200)
    plt.close(fig)

    daily = df.groupby("date_gregorian", observed=True)["resid"].mean().sort_index()
    n_lags = min(10, len(daily) // 2 - 1)
    acf_vals = acf(daily.to_numpy(), nlags=n_lags, fft=False)
    ci = 1.96 / np.sqrt(len(daily))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.stem(range(len(acf_vals)), acf_vals)
    ax.axhline(ci, color="red", linestyle="--", linewidth=1)
    ax.axhline(-ci, color="red", linestyle="--", linewidth=1)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xlabel(fa("وقفه (روز)"))
    ax.set_ylabel(fa("خودهمبستگی"))
    fig.tight_layout()
    fig.savefig(f"{OUT}/8.2_resid_acf_fa.png", dpi=200)
    plt.close(fig)
    print("saved")


if __name__ == "__main__":
    main()
