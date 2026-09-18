"""بند ۹-۷ — خوشه‌بندی چندنمایی رفتاری: اسکن $k$ و مقایسه‌ی چهار الگوریتم.

اجرا: python -m src.eda_lib.runners.r9e_clustering
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import DATA_INTERIM, set_global_seed
from src.eda_lib import figs
from src.eda_lib.person_clustering import (
    RANDOM_STATE, VIEWS, build_view_matrix, fit_algorithms, k_scan,
)
from src.eda_lib.runners._common_r9 import (
    N_MIN_MAIN, N_MIN_ROBUST, at_threshold, header, load_profile, md_table, setup,
)

LABELS_PATH = DATA_INTERIM / "person_cluster_labels_v4.parquet"
K_RANGE = range(2, 13)


def p971_kscan(P: pd.DataFrame) -> pd.DataFrame:
    """۹-۷-۴ اسکن $k$ روی هر شش نما با چهار معیار."""
    header("۹-۷-۴ اسکن $k$ روی شش نما", 2)
    out = []
    for v in VIEWS:
        X, idx, cols = build_view_matrix(P, v)
        s = k_scan(X, K_RANGE)
        s.insert(0, "نما", v)
        s["n"] = len(idx)
        out.append(s)
        best = s.loc[s.silhouette.idxmax()]
        print(f"  {v:22s} n={len(idx):6d} PCA={X.shape[1]:2d} → "
              f"بیشینه‌ی سیلوئت در k={int(best.k)} ({best.silhouette:.4f})")
    t = pd.concat(out, ignore_index=True)

    fig, axes = figs.new(2, 3, w=4.7, h=3.6)
    for ax, v in zip(axes.ravel(), VIEWS):
        s = t[t["نما"] == v]
        ax.plot(s.k, s.silhouette, "o-", color=figs.C_MAIN, label="سیلوئت")
        ax.set_title(v, fontsize=10)
        ax.set_xlabel("k")
        ax.set_ylabel("سیلوئت")
        ax2 = ax.twinx()
        ax2.plot(s.k, s.davies_bouldin, "s--", color=figs.C_ALT, label="دیویس-بولدین")
        ax2.set_ylabel("دیویس-بولدین", fontsize=8)
        ax2.grid(False)
        b = s.loc[s.silhouette.idxmax()]
        ax.axvline(b.k, color=figs.C_OK, ls=":", lw=1.2)
        figs.annotate(ax, f"بیشینه k={int(b.k)}\nسیلوئت={b.silhouette:.3f}", fontsize=7)
    figs.save(fig, "r9e_01_k_scan", t,
              caption="سیلوئت در همه‌ی نماها زیر ۰.۴ می‌ماند — ساختار پیوسته است نه جزیره‌ای")
    return t


def p973_algorithms(P: pd.DataFrame, k_by_view: dict[str, int]) -> pd.DataFrame:
    """۹-۷-۳ چهار الگوریتم روی هر نما و توافقشان (ARI)."""
    header("۹-۷-۳ توافق چهار الگوریتم روی هر نما", 2)
    from sklearn.metrics import adjusted_rand_score

    rows, store = [], {}
    for v, k in k_by_view.items():
        X, idx, cols = build_view_matrix(P, v)
        labs = fit_algorithms(X, k)
        store[v] = (idx, labs)
        names = list(labs)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                m = (labs[a] >= 0) & (labs[b] >= 0)   # نقاط بی‌خوشه‌ی HDBSCAN کنار
                rows.append({"نما": v, "k": k, "الگوریتم ۱": a, "الگوریتم ۲": b,
                             "ARI": adjusted_rand_score(labs[a][m], labs[b][m]),
                             "سهم مقایسه‌شده": float(m.mean())})
        hd = labs["HDBSCAN"]
        print(f"  {v:22s} k={k} · HDBSCAN: {len(set(hd[hd >= 0]))} خوشه، "
              f"{(hd < 0).mean():.1%} بی‌خوشه")
    t = pd.DataFrame(rows)
    print()
    print(md_table(t.groupby("نما", observed=True)["ARI"]
                   .agg(["min", "median", "max"]).round(3).reset_index()))

    fig, axes = figs.new(2, 3, w=4.7, h=3.8)
    for ax, v in zip(axes.ravel(), VIEWS):
        s = t[t["نما"] == v]
        pairs = [f"{a[:7]}↔{b[:7]}" for a, b in zip(s["الگوریتم ۱"], s["الگوریتم ۲"])]
        ax.barh(range(len(s)), s.ARI, color=figs.C_MAIN)
        ax.set_yticks(range(len(s)), pairs, fontsize=6)
        ax.set_xlabel("ARI")
        ax.set_xlim(0, 1)
        ax.set_title(v, fontsize=9)
        ax.axvline(0.5, color=figs.C_ALT, ls="--", lw=1)
    figs.save(fig, "r9e_02_algorithm_agreement", t,
              caption="ARI پایین بین الگوریتم‌ها یعنی ساختار به روش وابسته است، نه به داده")

    lab_df = pd.DataFrame({"PersonId": P.PersonId})
    for v, (idx, labs) in store.items():
        s = pd.Series(labs["K-Means"], index=idx.to_numpy(), name=v)
        lab_df[v] = lab_df.PersonId.map(s)
    lab_df.to_parquet(LABELS_PATH, index=False)
    print(f"\n  برچسب‌ها ذخیره شد: {LABELS_PATH.name} ({lab_df.shape})")
    return t


def p972_robustness(P_main: pd.DataFrame, P_rob: pd.DataFrame,
                    k_by_view: dict[str, int]) -> pd.DataFrame:
    """۹-۷-۱ آیا خوشه‌بندی به آستانه حساس است؟ (قاعده‌ی الف)."""
    header(f"۹-۷-۱ مقاومت به آستانه: n_min={N_MIN_MAIN} در برابر {N_MIN_ROBUST}", 2)
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    rows = []
    for v, k in k_by_view.items():
        Xa, ia, _ = build_view_matrix(P_main, v)
        Xb, ib, _ = build_view_matrix(P_rob, v)
        la = pd.Series(KMeans(n_clusters=k, random_state=RANDOM_STATE,
                              n_init=10).fit_predict(Xa), index=ia.to_numpy())
        lb = pd.Series(KMeans(n_clusters=k, random_state=RANDOM_STATE,
                              n_init=10).fit_predict(Xb), index=ib.to_numpy())
        both = la.index.intersection(lb.index)
        rows.append({"نما": v, "k": k, "افراد مشترک": len(both),
                     "ARI دو آستانه": adjusted_rand_score(la[both], lb[both])})
    t = pd.DataFrame(rows)
    print(md_table(t.round(4)))
    return t


def main() -> None:
    set_global_seed()
    setup()
    P0 = load_profile()
    P = at_threshold(P0, N_MIN_MAIN)
    header(f"۹-۷ خوشه‌بندی چندنمایی (آستانه‌ی اصلی n_min={N_MIN_MAIN}، n={len(P):,})")

    scan = p971_kscan(P)
    # انتخاب k: بیشینه‌ی سیلوئت، ولی با قید «کوچک‌ترین خوشه ≥۳٪» تا خوشه‌ی
    # چندنفره‌ای که فقط پرت است، k را نبرد.
    k_by_view = {}
    for v in VIEWS:
        s = scan[(scan["نما"] == v) & (scan.min_cluster_share >= 0.03)]
        s = s if len(s) else scan[scan["نما"] == v]
        k_by_view[v] = int(s.loc[s.silhouette.idxmax(), "k"])
    print("\nk انتخاب‌شده به‌ازای هر نما (بیشینه‌ی سیلوئت با قید خوشه‌ی ≥۳٪):")
    print(md_table(pd.DataFrame({"نما": list(k_by_view), "k": list(k_by_view.values())})))
    (DATA_INTERIM / "person_cluster_k_v4.json").write_text(
        json.dumps(k_by_view, ensure_ascii=False, indent=2), encoding="utf-8")

    p973_algorithms(P, k_by_view)
    p972_robustness(P, at_threshold(P0, N_MIN_ROBUST), k_by_view)
    header("پایان بند ۹-۷")


if __name__ == "__main__":
    main()
