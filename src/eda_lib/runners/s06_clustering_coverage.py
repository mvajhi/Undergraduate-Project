"""بند ۴.۶ (خوشه‌بندی) و ۴.۷ (توازن/پوشش/کفایت داده) — دور ۱ روی `dataset_v2`.

خروجی تصمیمی این اسکریپت، بند ۴.۶.۴ است: **مدل Global، per-cluster، یا per-restaurant؟**
هشدار WBS اینجا جدی گرفته می‌شود — با ۳۰ سلف، خوشه‌بندی الگوریتمی می‌تواند بی‌ثبات
باشد، پس علاوه بر امتیازهای کیفیت، **پایداری خوشه‌ها در برابر bootstrap** هم سنجیده
می‌شود؛ خوشه‌بندی‌ای که با حذف چند سلف عوض شود، مبنای تصمیم معماری مدل نمی‌شود.

اجرا: `python -m src.eda_lib.runners.s06_clustering_coverage`
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.config import FIGURES_DIR, RANDOM_SEED
from src.eda_lib.clustering_helpers import (
    cluster_quality_scan,
    coverage_matrix,
    food_feature_matrix,
    relative_week_index,
    restaurant_feature_matrix,
    series_length_distribution,
)
from src.eda_lib.figio import save_fig
from src.eda_lib.runners._common import header, kv, load_dataset, pct, setup
from src.viz_fa import fa
from src.viz_fa import setup as viz_setup


def run_restaurant_clustering(df: pd.DataFrame) -> pd.DataFrame:
    header("۴.۶.۱ خوشه‌بندی سلف‌ها")
    F = restaurant_feature_matrix(df)
    print(f"ماتریس ویژگی: {F.shape[0]} سلف × {F.shape[1]} ویژگی")
    print(f"ویژگی‌ها: {list(F.columns)}")
    Fn = F.select_dtypes(include=[np.number]).fillna(0.0)
    X = StandardScaler().fit_transform(Fn)

    print("\nامتیاز کیفیت خوشه‌بندی (K-Means):")
    q = cluster_quality_scan(X, range(2, 9))
    print(q.round(4).to_string(index=False))
    best_sil = q.loc[q["silhouette"].idxmax()]
    kv("\nبهترین k بر اساس Silhouette", f"k={int(best_sil['k'])} (sil={best_sil['silhouette']:.3f})")
    print("⚠️ Silhouette زیر ۰.۲۵ یعنی ساختار خوشه‌ای ضعیف/مصنوعی است."
          if best_sil["silhouette"] < 0.25 else "ساختار خوشه‌ای قابل‌قبول است.")

    Z = linkage(X, method="ward")
    for k in [2, 3, 4]:
        lab = fcluster(Z, t=k, criterion="maxclust")
        grp = pd.Series(lab, index=Fn.index)
        print(f"\nWard با k={k}:")
        summ = F.assign(cl=grp).groupby("cl").agg(
            n=("mean_rho", "size"), mean_rho=("mean_rho", "mean"), std_rho=("std_rho", "mean"))
        print(summ.round(4).to_string())
        for c in sorted(grp.unique()):
            print(f"  خوشه {c}: {list(grp[grp == c].index)}")

    header("پایداری خوشه‌ها (bootstrap با حذف تصادفی ۲۰٪ سلف‌ها)", 2)
    rng = np.random.default_rng(RANDOM_SEED)
    names = list(Fn.index)
    base = pd.Series(fcluster(Z, t=3, criterion="maxclust"), index=names)
    agree = []
    for _ in range(200):
        keep = rng.choice(len(names), size=int(len(names) * 0.8), replace=False)
        sub_names = [names[i] for i in keep]
        Xs = StandardScaler().fit_transform(Fn.loc[sub_names])
        lab = pd.Series(fcluster(linkage(Xs, method="ward"), t=3, criterion="maxclust"), index=sub_names)
        # نرخ توافق زوجی: چند درصد جفت‌سلف‌ها در هر دو خوشه‌بندی هم‌خوشه/غیرهم‌خوشه‌اند
        pairs = [(a, b) for i, a in enumerate(sub_names) for b in sub_names[i + 1:]]
        same_base = np.array([base[a] == base[b] for a, b in pairs])
        same_new = np.array([lab[a] == lab[b] for a, b in pairs])
        agree.append((same_base == same_new).mean())
    kv("نرخ توافق زوجی (میانگین ± std)", f"{np.mean(agree):.3f} ± {np.std(agree):.3f}")
    print("(۱.۰ = خوشه‌بندی کاملاً پایدار؛ زیر ~۰.۸ یعنی تصمیم per-cluster شکننده است)")

    viz_setup()
    fig, ax = plt.subplots(figsize=(13, 6))
    dendrogram(Z, labels=[fa(n) for n in Fn.index], ax=ax, leaf_rotation=90)
    ax.set_title(fa("خوشه‌بندی سلسله‌مراتبی سلف‌ها بر اساس پروفایل رفتاری (Ward)"))
    fig.tight_layout()
    print(save_fig(fig, "4.6_restaurant_dendrogram_v2", FIGURES_DIR))
    plt.close(fig)
    return F.assign(cluster3=base)


def run_food_clustering(df: pd.DataFrame) -> None:
    header("۴.۶.۲ خوشه‌بندی غذاها (شاخص محبوبیت)")
    G = food_feature_matrix(df)
    print(f"ماتریس ویژگی: {G.shape[0]} غذا × {G.shape[1]} ویژگی")
    Gn = G.select_dtypes(include=[np.number]).fillna(0.0)
    X = StandardScaler().fit_transform(Gn)
    q = cluster_quality_scan(X, range(2, 7))
    print(q.round(4).to_string(index=False))
    k = int(q.loc[q["silhouette"].idxmax(), "k"])
    lab = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=20).fit_predict(X)
    G = G.assign(cl=lab)
    print(f"\nخلاصه‌ی {k} خوشه‌ی غذا:")
    print(G.groupby("cl").agg(n=("mean_rho", "size"), **{
        c: (c, "mean") for c in Gn.columns[:4]}).round(3).to_string())
    print("\nبالاترین و پایین‌ترین نرخ عدم‌دریافت در سطح غذا:")
    s = G.sort_values("mean_rho")
    print("  کمترین:", ", ".join(f"{i} ({v:.3f})" for i, v in s["mean_rho"].head(5).items()))
    print("  بیشترین:", ", ".join(f"{i} ({v:.3f})" for i, v in s["mean_rho"].tail(5).items()))
    print("\n⚠️ هشدار نشتی: این شاخص از کل بازه ساخته شده. اگر در فاز ۵ فیچر شود، "
          "باید فقط از تاریخچه‌ی تا لحظه‌ی برش (expanding) محاسبه شود.")


def run_coverage(df: pd.DataFrame) -> None:
    header("۴.۷ توازن، پوشش و کفایت داده")
    df = df.copy()
    df["week"] = relative_week_index(df["date_gregorian"])
    cov = coverage_matrix(df, group_col="RestaurantName")
    kv("ابعاد ماتریس پوشش (سلف × هفته)", f"{cov.shape[0]} × {cov.shape[1]}")
    holes = (cov == 0).sum().sum()
    kv("خانه‌های خالی", f"{holes} از {cov.size} ({pct(holes / cov.size)})")
    per_r = (cov == 0).sum(axis=1).sort_values(ascending=False)
    print("\nسلف‌هایی با بیشترین هفته‌ی بدون داده:")
    print(per_r.head(8).to_string())
    per_w = (cov == 0).sum(axis=0)
    print("\nهفته‌هایی با بیشترین سلف غایب:")
    print(per_w.sort_values(ascending=False).head(6).to_string())

    header("طول سری‌های (وعده، سلف، غذا)", 2)
    sl = series_length_distribution(df)["n_obs"]
    print(sl.describe().round(1).to_string())
    for thr in [10, 30, 50]:
        n = int((sl < thr).sum())
        print(f"  سری‌های با کمتر از {thr} نقطه: {n} از {len(sl)} ({n / len(sl):.1%})")

    header("سری‌های (سلف، وعده) — واحد واقعی مدل‌سازی", 2)
    sl2 = df.groupby(["RestaurantName", "Meal"]).size().sort_values()
    print(f"تعداد سری: {len(sl2)} · میانه طول: {sl2.median():.0f} · "
          f"کمترین: {sl2.min()} · بیشترین: {sl2.max()}")
    print(f"سری‌های زیر ۳۰ نقطه: {int((sl2 < 30).sum())} ({(sl2 < 30).mean():.1%})")
    print(sl2.head(8).to_string())

    header("کفایت نمونه در دُم بالای ρ", 2)
    for q in [0.90, 0.95, 0.99]:
        thr = df["rho"].quantile(q)
        sub = df[df["rho"] >= thr]
        print(f"  بالای صدک {int(q * 100)} (ρ≥{thr:.3f}): n={len(sub)} · "
              f"{sub['RestaurantName'].nunique()} سلف · میانه Res={sub['Res'].median():.0f} · "
              f"سهم Res<50: {(sub['Res'] < 50).mean():.1%}")
    kv("\nسهم کل رکوردها با Res<10", pct((df["Res"] < 10).mean()))
    kv("سهم کل رکوردها با Res<30", pct((df["Res"] < 30).mean()))


def main() -> None:
    setup()
    df = load_dataset()
    run_restaurant_clustering(df)
    run_food_clustering(df)
    run_coverage(df)


if __name__ == "__main__":
    main()
