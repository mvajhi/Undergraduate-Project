"""بند ۴.۴ — همبستگی و وابستگی (ماتریس‌ها، MI، dCor، VIF، CCF، همبستگی جزئی)، دور ۱.

⚠️ **قاعده‌ی برش اطلاعاتی در این بند:** `Recv`, `NoRecv` و `card_ratio` از خروجی *همان
وعده* مشتق می‌شوند و طبق بند ۴.۲ سند مسئله **هرگز فیچر نمی‌شوند**. اینجا فقط برای
فهم ساختار داده نگه داشته شده‌اند و در جدول‌ها با برچسب «ممنوع (نشتی)» علامت می‌خورند
تا کسی به اشتباه از همبستگی بالایشان نتیجه‌ی فیچر نگیرد.

اجرا: `python -m src.eda_lib.runners.s04_correlation`
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform

from src.config import FIGURES_DIR
from src.eda_lib.correlation_helpers import (
    distance_correlation,
    manual_ccf,
    mutual_info_target,
    save_fig,
    vif_table,
)
from src.eda_lib.runners._common import CALENDAR_PATH, header, kv, load_dataset_with_weather, setup
from src.viz_fa import fa
from src.viz_fa import setup as viz_setup

LEAKY = {"Recv", "NoRecv", "card_ratio", "rho"}

CANDIDATE_NUMERIC = [
    "Res", "gender_ratio", "DayOfWeek",
    "HolidayInWeekCount", "HolidayInPrevWeekCount", "HolidayInNextWeekCount",
    "NextHoliday_1", "NextHoliday_2", "PreviousHoliday_1", "PreviousHoliday_2",
    "days_to_next_holiday", "days_since_last_holiday", "week_of_semester", "days_to_exam_start",
    "temp_mean", "temp_min", "temp_max", "precipitation_sum", "snowfall_sum",
    "relative_humidity_mean", "wind_speed_max", "aqi_us_max", "pm2_5_mean",
    "distance_km_to_tehran_campus",
]


def run_matrices(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    header("۴.۴.۱ همبستگی با متغیر هدف — سه معیار خطی/رتبه‌ای + دو معیار غیرخطی")
    y = df["rho"]
    rows = []
    for c in cols:
        x = df[c]
        m = ~(x.isna() | y.isna())
        if m.sum() < 30 or x[m].nunique() < 3:
            continue
        pear = stats.pearsonr(x[m], y[m])
        spear = stats.spearmanr(x[m], y[m])
        kend = stats.kendalltau(x[m], y[m])
        dcor = distance_correlation(x[m].values, y[m].values)
        rows.append({"متغیر": c, "Pearson": pear[0], "Spearman": spear[0],
                     "Kendall": kend[0], "dCor": dcor, "p_spearman": spear[1]})
    res = pd.DataFrame(rows)
    mi = mutual_info_target(df[cols].fillna(df[cols].median()), y)
    res["MI"] = res["متغیر"].map(mi)
    res["|Spearman|"] = res["Spearman"].abs()
    res = res.sort_values("|Spearman|", ascending=False)
    print(res.drop(columns="|Spearman|").round(4).to_string(index=False))

    print("\nمتغیرهایی که MI بالا ولی Spearman پایین دارند (نشانه‌ی رابطه‌ی غیریکنوا):")
    cand = res[(res["MI"] > res["MI"].median()) & (res["Spearman"].abs() < 0.05)]
    print(cand[["متغیر", "Spearman", "MI", "dCor"]].round(4).to_string(index=False)
          if len(cand) else "  (موردی یافت نشد)")
    return res


def run_leakage_contrast(df: pd.DataFrame) -> None:
    header("مقایسه‌ی هشداردهنده: همبستگی ستون‌های ممنوع (نشتی) با هدف", 2)
    for c in ["NoRecv", "Recv", "card_ratio"]:
        r = stats.spearmanr(df[c], df["rho"], nan_policy="omit")[0]
        print(f"  {c:12s} Spearman با ρ = {r:+.4f}   ← ممنوع (خروجی همان وعده)")
    print("  این اعداد فقط برای نشان‌دادن اندازه‌ی وسوسه‌اند؛ هیچ‌کدام فیچر نمی‌شوند.")


def run_collinearity(df: pd.DataFrame, cols: list[str]) -> None:
    header("۴.۴.۲ هم‌خطی — VIF و خوشه‌بندی فیچرهای همبسته")
    X = df[cols].fillna(df[cols].median())
    X = X.loc[:, X.std() > 0]
    v = vif_table(X).sort_values("VIF", ascending=False)
    print(v.round(2).to_string(index=False))
    high = v[v["VIF"] > 10]
    kv("\nتعداد فیچر با VIF>10", f"{len(high)} از {len(v)}")

    corr = X.corr(method="spearman")
    dist = 1 - corr.abs()
    np.fill_diagonal(dist.values, 0.0)
    Z = linkage(squareform(dist.values, checks=False), method="average")
    for t in [0.2, 0.3]:
        labels = fcluster(Z, t=t, criterion="distance")
        clusters = pd.Series(labels, index=corr.columns).groupby(lambda i: labels[list(corr.columns).index(i)])
        groups = {}
        for col, lab in zip(corr.columns, labels):
            groups.setdefault(lab, []).append(col)
        multi = {k: v for k, v in groups.items() if len(v) > 1}
        print(f"\nخوشه‌بندی در آستانه‌ی 1-|corr| < {t}: {len(groups)} خوشه، "
              f"{len(multi)} خوشه‌ی چندعضوی")
        for k, v2 in multi.items():
            print(f"  خوشه {k}: {v2}")

    viz_setup()
    fig, ax = plt.subplots(figsize=(13, 6))
    dendrogram(Z, labels=[fa(c) for c in corr.columns], ax=ax, leaf_rotation=90)
    ax.set_title(fa("خوشه‌بندی سلسله‌مراتبی فیچرها بر پایه‌ی ۱ منهای قدر مطلق همبستگی"))
    ax.set_ylabel(fa("فاصله"))
    fig.tight_layout()
    print("\n" + str(save_fig(fig, "4.4_feature_dendrogram", FIGURES_DIR)))
    plt.close(fig)


def run_ccf_partial(df: pd.DataFrame) -> None:
    header("۴.۴.۳ همبستگی متقابل با تأخیر (CCF) و همبستگی جزئی")
    print("CCF بین ρ روزانه و متغیرهای هواشناسی — **فقط تهران** (تا اثر بین‌شهری وارد نشود):")
    teh = df[df.city == "تهران"]
    daily = (teh.groupby("date_gregorian")
                .agg(rho=("rho", "mean"), temp_min=("temp_min", "first"),
                     aqi_us_max=("aqi_us_max", "first"),
                     precipitation_sum=("precipitation_sum", "first")))
    daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max(), freq="D"))
    for col in ["temp_min", "aqi_us_max", "precipitation_sum"]:
        c = manual_ccf(daily[col], daily["rho"], max_lag=5)
        best = c.reindex(c["pearson_r"].abs().sort_values(ascending=False).index).iloc[0]
        print(f"\n  {col}: بیشترین |همبستگی| در تأخیر {int(best['lag'])} = {best["pearson_r"]:+.4f}")
        print("   " + c.round(3).to_string(index=False).replace("\n", "\n   "))

    header("همبستگی جزئی: آیا اثر AQI پس از کنترل دما باقی می‌ماند؟", 2)
    sub = daily.dropna(subset=["rho", "aqi_us_max", "temp_min"])
    r_aqi = stats.spearmanr(sub["aqi_us_max"], sub["rho"])
    r_temp = stats.spearmanr(sub["temp_min"], sub["rho"])
    r_at = stats.spearmanr(sub["aqi_us_max"], sub["temp_min"])
    print(f"  (فقط تهران، n={len(sub)} روز)")
    print(f"  ρ(AQI, نرخ)   = {r_aqi[0]:+.4f} (p={r_aqi[1]:.3g})")
    print(f"  ρ(دما, نرخ)   = {r_temp[0]:+.4f} (p={r_temp[1]:.3g})")
    print(f"  ρ(AQI, دما)   = {r_at[0]:+.4f} (p={r_at[1]:.3g})  ← هم‌خطی زمستان تهران")
    denom = np.sqrt((1 - r_aqi[0] ** 2) * (1 - r_at[0] ** 2))
    partial_aqi = (r_aqi[0] - r_temp[0] * r_at[0]) / np.sqrt((1 - r_temp[0] ** 2) * (1 - r_at[0] ** 2))
    partial_temp = (r_temp[0] - r_aqi[0] * r_at[0]) / np.sqrt((1 - r_aqi[0] ** 2) * (1 - r_at[0] ** 2))
    print(f"\n  همبستگی جزئی AQI با نرخ، با کنترل دما = {partial_aqi:+.4f}")
    print(f"  همبستگی جزئی دما با نرخ، با کنترل AQI = {partial_temp:+.4f}")


def main() -> None:
    setup()
    df = load_dataset_with_weather()
    cal = pd.read_csv(CALENDAR_PATH, parse_dates=["date_gregorian"])
    df = df.merge(cal[["date_gregorian", "days_to_next_holiday", "days_since_last_holiday",
                       "week_of_semester", "days_to_exam_start"]],
                  on="date_gregorian", how="left", validate="many_to_one")
    cols = [c for c in CANDIDATE_NUMERIC if c in df.columns and c not in LEAKY]
    header(f"n={len(df)} · {len(cols)} فیچر عددی کاندید (ستون‌های نشتی کنار گذاشته شدند)")
    run_matrices(df, cols)
    run_leakage_contrast(df)
    run_collinearity(df, cols)
    run_ccf_partial(df)


if __name__ == "__main__":
    main()
