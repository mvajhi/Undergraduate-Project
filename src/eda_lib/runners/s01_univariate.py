"""بند ۴.۱ — تحلیل تک‌متغیره (بازاجرا روی `dataset_v2`، دور ۱).

نسبت به اجرای قبلی روی v1 دو چیز اضافه شده: (الف) بُعد `city` که تازه ساخته شده،
(ب) تفکیک `RestaurantType` (خوابگاهی/دانشگاهی) در آماره‌های توصیفی، طبق درخواست
ذی‌نفع برای ورود صریح تحلیل خوابگاه/غیرخوابگاه.

اجرا: `python -m src.eda_lib.runners.s01_univariate`
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FIGURES_DIR
from src.eda_lib.runners._common import boot_ci, header, kv, load_dataset, pct, setup
from src.eda_lib.univariate_helpers import (
    compare_transforms,
    descriptive_stats,
    fit_candidate_distributions,
    kde_peak_count,
    sarle_bimodality_coefficient,
    save_fig,
    zero_one_inflation,
)
from src.viz_fa import fa
from src.viz_fa import setup as viz_setup


def run_target_variable(df: pd.DataFrame) -> None:
    header("۴.۱.۱ متغیر هدف ρ")
    rho = df["rho"].to_numpy()

    s = descriptive_stats(rho)
    for k in ["n", "mean", "median", "mode", "std", "iqr", "skewness", "kurtosis_excess"]:
        kv(k, f"{s[k]:.4f}" if isinstance(s[k], float) else s[k])
    kv("mode_count", s["mode_count"])
    print("\nصدک‌ها:")
    for p, v in s["percentiles"].items():
        kv(f"  p{p}", f"{v:.4f}")

    m, lo, hi = boot_ci(rho)
    kv("\nمیانگین ρ (bootstrap 95% CI)", f"{m:.4f} [{lo:.4f}, {hi:.4f}]")
    # ρ وزنی حجم — عدد عملیاتی واقعی (کل عدم‌دریافت / کل رزرو)
    rho_w = df["NoRecv"].sum() / df["Res"].sum()
    kv("ρ وزنی حجم (ΣNoRecv/ΣRes)", f"{rho_w:.4f}")

    inf = zero_one_inflation(rho)
    print()
    for k, v in inf.items():
        kv(k, v)

    print()
    kv("ضریب دومُدی Sarle", f"{sarle_bimodality_coefficient(rho):.4f} (آستانه ۰.۵۵۵)")
    n_peaks, peaks = kde_peak_count(rho)
    kv("تعداد قله‌های KDE", n_peaks)

    print("\nبرازش توزیع‌های کاندید (نقاط داخلی، بدون ۰ و ۱):")
    print(fit_candidate_distributions(rho, interior_only=True).to_string(index=False))


def run_counts(df: pd.DataFrame) -> None:
    header("۴.۱.۲ متغیرهای شمارشی")
    for col in ["Res", "Recv", "NoRecv"]:
        s = descriptive_stats(df[col].to_numpy())
        print(f"\n{col}: mean={s['mean']:.1f} median={s['median']:.1f} std={s['std']:.1f} "
              f"skew={s['skewness']:.2f} kurt={s['kurtosis_excess']:.2f} "
              f"min={df[col].min():.0f} max={df[col].max():.0f}")
        print(compare_transforms(df[col].to_numpy()).to_string(index=False))

    header("توزیع اندازه‌ی سلف‌ها", level=2)
    size = (df.groupby(["RestaurantName", "RestaurantType", "city"])
              .agg(n=("rho", "size"), res_mean=("Res", "mean"), res_sum=("Res", "sum"),
                   rho_mean=("rho", "mean"))
              .sort_values("res_sum", ascending=False).reset_index())
    size["res_share"] = size["res_sum"] / size["res_sum"].sum()
    size["cum_share"] = size["res_share"].cumsum()
    print(size.round(4).to_string(index=False))
    kv("\nنسبت بزرگ‌ترین به کوچک‌ترین (میانگین Res)", f"{size['res_mean'].max() / size['res_mean'].min():.1f}x")
    n80 = int((size["cum_share"] <= 0.80).sum()) + 1
    kv("تعداد سلف تا ۸۰٪ حجم", f"{n80} از {len(size)} ({n80 / len(size):.1%})")


def run_categoricals(df: pd.DataFrame) -> None:
    header("۴.۱.۳ متغیرهای دسته‌ای")
    for col in ["Meal", "RestaurantType", "FoodType", "city", "dow_name", "ym"]:
        vc = df[col].value_counts(dropna=False)
        print(f"\n--- {col} ---")
        out = pd.DataFrame({"n": vc, "share": (vc / len(df))})
        out["rho_mean"] = df.groupby(col)["rho"].mean()
        out["rho_w"] = df.groupby(col).apply(lambda x: x["NoRecv"].sum() / x["Res"].sum(), include_groups=False)
        print(out.round(4).to_string())

    header("تقاطع کلیدی: وعده × نوع سلف (خوابگاهی/دانشگاهی)", level=2)
    ct = pd.crosstab(df["RestaurantType"], df["Meal"])
    print(ct.to_string())
    print("\nسهم سطری:")
    print((ct.div(ct.sum(axis=1), axis=0)).round(4).to_string())
    print("\nρ وزنی در هر خانه:")
    piv = df.pivot_table(index="RestaurantType", columns="Meal",
                         values=["NoRecv", "Res"], aggfunc="sum")
    print((piv["NoRecv"] / piv["Res"]).round(4).to_string())


def run_figures(df: pd.DataFrame) -> None:
    header("ذخیره‌ی نمودارها")
    viz_setup()
    rho = df["rho"].to_numpy()

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes[0, 0].hist(rho, bins=80, color="#4C72B0", edgecolor="white")
    axes[0, 0].set_title(fa("توزیع نرخ عدم‌دریافت ρ — چوله به راست با تمرکز روی صفر"))
    axes[0, 0].set_xlabel(fa("نرخ عدم‌دریافت")); axes[0, 0].set_ylabel(fa("فراوانی"))

    xs = np.sort(rho)
    axes[0, 1].plot(xs, np.arange(1, len(xs) + 1) / len(xs), color="#C44E52")
    axes[0, 1].set_title(fa("تابع توزیع تجمعی تجربی (ECDF)"))
    axes[0, 1].set_xlabel(fa("نرخ عدم‌دریافت")); axes[0, 1].set_ylabel(fa("نسبت تجمعی"))
    axes[0, 1].grid(alpha=.3)

    order = df.groupby("city")["rho"].median().sort_values().index
    axes[1, 0].violinplot([df.loc[df["city"] == c, "rho"].values for c in order], showmedians=True)
    axes[1, 0].set_xticks(range(1, len(order) + 1))
    axes[1, 0].set_xticklabels([fa(c) for c in order], rotation=20)
    axes[1, 0].set_title(fa("نرخ عدم‌دریافت به تفکیک شهر — تهران به‌وضوح بالاتر"))
    axes[1, 0].set_ylabel(fa("نرخ عدم‌دریافت"))

    for rt, color in [("khabgah", "#DD8452"), ("daneshgah", "#4C72B0")]:
        sub = df.loc[df["RestaurantType"] == rt, "rho"]
        axes[1, 1].hist(sub, bins=60, alpha=.55, density=True, label=fa("خوابگاهی" if rt == "khabgah" else "دانشگاهی"), color=color)
    axes[1, 1].legend(); axes[1, 1].set_title(fa("خوابگاهی در برابر دانشگاهی"))
    axes[1, 1].set_xlabel(fa("نرخ عدم‌دریافت")); axes[1, 1].set_ylabel(fa("چگالی"))

    fig.tight_layout()
    print(save_fig(fig, "4.1_rho_overview_v2", FIGURES_DIR))
    plt.close(fig)


def main() -> None:
    setup()
    df = load_dataset()
    header(f"داده: dataset_v2 — {len(df)} ردیف، {df['RestaurantName'].nunique()} سلف، "
           f"{df['DateReserve'].nunique()} روز، {df['city'].nunique()} شهر")
    run_target_variable(df)
    run_counts(df)
    run_categoricals(df)
    run_figures(df)


if __name__ == "__main__":
    main()
