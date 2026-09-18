"""بند ۹-۸ — اعتبارسنجی و پایداری خوشه‌ها: **دروازه‌ی تفسیر**.

اجرا: python -m src.eda_lib.runners.r9f_validation

## ⚠️ تله‌ای که این بسته باید از آن دوری کند

برچسب‌های بند ۹-۷ از **کل پنجره** ساخته شده‌اند و طبق قاعده‌ی ت فقط توصیفی‌اند.
استفاده از همان برچسب‌ها برای پیش‌بینی نرخ سال $t+1$ نشتی است: برچسب خودش
داده‌ی سال $t+1$ را دیده. پس بندهای ۹-۸-۴ و ۹-۸-۶ برچسب را **از نو و فقط روی
سال $t$** می‌سازند. این تفاوت، تفاوت بین «۰.۲۵ اعتبار پیش‌بین» و صفر است.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import DATA_INTERIM, set_global_seed
from src.eda_lib import figs
from src.eda_lib.person_clustering import (
    RANDOM_STATE, VIEWS, bootstrap_stability, build_view_matrix,
)
from src.eda_lib.person_profile_v4 import build_person_year_v4
from src.eda_lib.runners._common_r9 import (
    N_MIN_MAIN, at_threshold, header, load_profile, md_table, setup,
)

LABELS_PATH = DATA_INTERIM / "person_cluster_labels_v4.parquet"
K_PATH = DATA_INTERIM / "person_cluster_k_v4.json"

#: ستون‌های سال‌به‌سالِ نمای ن۱ — تنها فیچرهایی که از پنل فرد×سال ساختنی‌اند.
YEAR_COLS = ["n_res", "n_waste", "rho_eb", "n_active_days"]

#: دروازه‌ی بند ۹-۸-۶ — سقف $R^2$ یک ستون حجم مصرف (بند ۹-۶-۵).
R2_GATE_VOLUME = 0.0929
R2_GATE_DEMO = 0.0901


def _year_matrix(Y: pd.DataFrame, yb: str, n_min: int = N_MIN_MAIN):
    """ماتریس خوشه‌بندی از داده‌ی **فقط همان سال**."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import QuantileTransformer, StandardScaler

    s = Y[(Y.year_block == yb) & (Y.n_res >= n_min)]
    X = s[YEAR_COLS].to_numpy(dtype="float64")
    X = QuantileTransformer(output_distribution="normal", n_quantiles=1000,
                            random_state=RANDOM_STATE).fit_transform(X)
    X = StandardScaler().fit_transform(X)
    return PCA(n_components=0.90, random_state=RANDOM_STATE).fit_transform(X), s


def p981_stability(P: pd.DataFrame, k_by_view: dict[str, int]) -> pd.DataFrame:
    """۹-۸-۱ پایداری بوت‌استرپ هر خوشه (قاعده‌ی ج)."""
    header("۹-۸-۱ پایداری بوت‌استرپ هر خوشه (۱۰۰ تکرار)", 2)
    out = []
    for v, k in k_by_view.items():
        X, idx, _ = build_view_matrix(P, v)
        t = bootstrap_stability(X, k, n_boot=100)
        t.insert(0, "نما", v)
        out.append(t)
        n_stable = int((t["ژاکارِ میانه"] > 0.75).sum())
        print(f"  {v:22s} k={k} → {n_stable}/{k} خوشه پایدار · "
              f"میانه‌ی کل {t['ژاکارِ میانه'].median():.3f}")
    t = pd.concat(out, ignore_index=True)

    fig, ax = figs.new(w=8.2, h=5.4)
    y = np.arange(len(t))
    colors = {"پایدار": figs.C_OK, "با احتیاط": figs.C_ACCENT, "ناپایدار": figs.C_ALT}
    ax.barh(y, t["ژاکارِ میانه"], color=[colors[s] for s in t["وضعیت"]])
    ax.set_yticks(y, [f"{r['نما'][:12]}·خ{r['خوشه']}" for _, r in t.iterrows()], fontsize=6)
    for thr, c in ((0.60, figs.C_ALT), (0.75, figs.C_OK)):
        ax.axvline(thr, color=c, ls="--", lw=1.2)
    ax.set_xlabel("ژاکارِ میانه‌ی بوت‌استرپ")
    ax.set_title("قاعده‌ی ج: زیر ۰.۶ تفسیر نمی‌شود")
    ax.set_xlim(0, 1)
    figs.save(fig, "r9f_01_bootstrap_stability", t,
              caption="کدام خوشه واقعاً وجود دارد و کدام مصنوع یک اجرای K-Means است")
    return t


def p983_view_agreement(k_by_view: dict[str, int]) -> pd.DataFrame:
    """۹-۸-۳ آیا شش نما چیز متفاوتی می‌بینند یا یک چیز را شش بار؟"""
    header("۹-۸-۳ توافق بین نماها (ARI)", 2)
    from sklearn.metrics import adjusted_rand_score

    L = pd.read_parquet(LABELS_PATH)
    views = [v for v in VIEWS if v in L.columns]
    m = pd.DataFrame(index=views, columns=views, dtype="float64")
    for a in views:
        for b in views:
            ok = L[a].notna() & L[b].notna()
            m.loc[a, b] = adjusted_rand_score(L.loc[ok, a], L.loc[ok, b])
    print(md_table(m.reset_index().rename(columns={"index": "نما"}).round(3)))

    fig, ax = figs.new(w=6.6, h=5.8)
    im = ax.imshow(m.to_numpy(dtype="float64"), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(views)), views, fontsize=7, rotation=35, ha="right")
    ax.set_yticks(range(len(views)), views, fontsize=7)
    for i in range(len(views)):
        for j in range(len(views)):
            v = m.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if v > 0.5 else "black")
    ax.set_title("ARI بین شش نما — افزونگی یا استقلال؟")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046)
    figs.save(fig, "r9f_02_view_agreement", m.reset_index(),
              caption="ARI پایین یعنی هر نما ساختار متفاوتی می‌بیند — چندنمایی بودن توجیه دارد")
    return m


def p984_temporal(n_min: int = N_MIN_MAIN) -> pd.DataFrame:
    """⭐ ۹-۸-۴ پایداری زمانی: بخش‌بندی **صفت** است یا **حالت**؟"""
    header("۹-۸-۴ پایداری زمانی بخش‌بندی (خوشه‌بندی مستقل هر سال)", 2)
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    Y = build_person_year_v4()
    years = list(Y.year_block.cat.categories) if hasattr(Y.year_block, "cat") \
        else sorted(Y.year_block.unique())
    # ⚠️ دو k آزمون می‌شود: k=۲ که بند ۹-۷ انتخابش کرد، و k=۴ که دانه‌ریزیِ
    # قابل‌استفاده برای سیاست است. پرسش «صفت یا حالت» نباید به یک k وابسته بماند.
    rows = []
    for k in (2, 4):
        labs = {}
        for yb in years:
            X, s = _year_matrix(Y, yb, n_min)
            labs[yb] = pd.Series(KMeans(n_clusters=k, random_state=RANDOM_STATE,
                                        n_init=10).fit_predict(X),
                                 index=s.PersonId.to_numpy())
        for y0, y1 in zip(years[:-1], years[1:]):
            both = labs[y0].index.intersection(labs[y1].index)
            rows.append({"k": k, "جفت سال": f"{y0} → {y1}", "افراد مشترک": len(both),
                         "ARI": adjusted_rand_score(labs[y0][both], labs[y1][both]),
                         "سهم هم‌خوشه (پس از تطبیق)": _matched_agreement(
                             labs[y0][both].to_numpy(), labs[y1][both].to_numpy(), k)})
    t = pd.DataFrame(rows)
    print(md_table(t.round(4)))
    print("\nمرجع: ARI=۰ یعنی مستقل از تصادف · ARI=۱ یعنی بخش‌بندی یکسان")
    print("⚠️ خوشه‌بندی هر سال **مستقل** است — برچسب سال t هیچ داده‌ای از t+1 ندیده.")

    fig, ax = figs.new(w=7.4, h=4.4)
    piv = t.pivot(index="جفت سال", columns="k", values="ARI")
    x = np.arange(len(piv))
    for i, kk in enumerate(piv.columns):
        ax.bar(x + (i - 0.5) * 0.35, piv[kk], 0.35, color=figs.PALETTE[i],
               label=f"k={kk}")
    ax.set_xticks(x, piv.index, fontsize=8)
    ax.set_ylabel("ARI بین بخش‌بندی دو سال")
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color=figs.C_ALT, ls="--", lw=1.2)
    ax.set_title("آیا فرد سال بعد در همان بخش می‌ماند؟")
    ax.legend(fontsize=8)
    figs.save(fig, "r9f_03_temporal_stability", t,
              caption="بخش‌بندی صفت است یا حالت — و آیا سیاست سالانه معنا دارد")
    return t


def _matched_agreement(a: np.ndarray, b: np.ndarray, k: int) -> float:
    """سهم افرادی که پس از بهترین تطبیق برچسب‌ها، در همان خوشه می‌مانند."""
    from scipy.optimize import linear_sum_assignment

    M = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            M[i, j] = np.sum((a == i) & (b == j))
    r, c = linear_sum_assignment(-M)
    return float(M[r, c].sum() / len(a))


def p985_demographic_ari(P: pd.DataFrame) -> pd.DataFrame:
    """۹-۸-۵ آیا خوشه‌بندی فقط جمعیت‌شناسی را دوباره کشف کرده؟ (روش F231)."""
    header("۹-۸-۵ ARI خوشه‌ها با رده‌بندی‌های جمعیتی", 2)
    from sklearn.metrics import adjusted_rand_score

    L = pd.read_parquet(LABELS_PATH).merge(
        P[["PersonId", "demo_college", "demo_degree", "demo_session",
           "demo_is_dorm", "demo_gender"]], on="PersonId", how="left")
    demos = {"demo_college": "دانشکده", "demo_degree": "مقطع", "demo_session": "دوره",
             "demo_is_dorm": "خوابگاه", "demo_gender": "جنسیت"}
    rows = []
    for v in [c for c in VIEWS if c in L.columns]:
        for d, dl in demos.items():
            ok = L[v].notna() & L[d].notna()
            rows.append({"نما": v, "رده‌بندی جمعیتی": dl,
                         "ARI": adjusted_rand_score(L.loc[ok, v], L.loc[ok, d].astype(str))})
    t = pd.DataFrame(rows)
    piv = t.pivot(index="نما", columns="رده‌بندی جمعیتی", values="ARI")
    print(md_table(piv.reset_index().round(4)))
    print(f"\nبیشینه‌ی کل ARI = {t.ARI.max():.4f} (لنگر F231 برای `venue_type`: ۰.۲۴)")

    fig, ax = figs.new(w=7.4, h=4.6)
    im = ax.imshow(piv.to_numpy(dtype="float64"), cmap="Blues", vmin=0, vmax=0.3)
    ax.set_xticks(range(piv.shape[1]), piv.columns, fontsize=8)
    ax.set_yticks(range(piv.shape[0]), piv.index, fontsize=7)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.iloc[i, j]:.3f}", ha="center", va="center", fontsize=7)
    ax.set_title("خوشه‌ی رفتاری در برابر رده‌بندی جمعیتی")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046)
    figs.save(fig, "r9f_04_demographic_ari", t,
              caption="ARI نزدیک صفر یعنی رفتار چیزی می‌گوید که جمعیت‌شناسی نمی‌گوید")
    return t


def p986_predictive(n_min: int = N_MIN_MAIN) -> pd.DataFrame:
    """⭐ ۹-۸-۶ سخت‌گیرانه‌ترین آزمون: آیا برچسب خوشه اطلاعات **تازه** دارد؟

    هدف: نرخ EB فرد در سال $t+1$. برچسب خوشه **فقط از سال $t$** ساخته می‌شود.
    ارزیابی روی افراد کنارگذاشته‌شده، نه همان افرادی که مدل دیده.
    """
    header("۹-۸-۶ اعتبار پیش‌بین برچسب خوشه (روی افراد کنارگذاشته‌شده)", 2)
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder

    Y = build_person_year_v4()
    P = load_profile()[["PersonId", "demo_college", "demo_degree", "demo_session",
                        "demo_is_dorm", "demo_gender"]]
    years = list(Y.year_block.cat.categories) if hasattr(Y.year_block, "cat") \
        else sorted(Y.year_block.unique())

    rows = []
    for y0, y1 in zip(years[:-1], years[1:]):
        X0, s0 = _year_matrix(Y, y0, n_min)
        lab = pd.Series(KMeans(n_clusters=4, random_state=RANDOM_STATE,
                               n_init=10).fit_predict(X0), index=s0.PersonId.to_numpy())
        nxt = Y[(Y.year_block == y1) & (Y.n_res >= n_min)].set_index("PersonId")
        # ⚠️ `intersection` نام اندیس را می‌اندازد وقتی دو طرف هم‌نام نباشند، و
        # بعد `.loc[both]` همان نامِ تهی را به قاب تحمیل می‌کند ⇒ `reset_index`
        # ستون `index` می‌سازد نه `PersonId`. نام‌گذاری صریح جلویش را می‌گیرد.
        both = lab.index.intersection(nxt.index).rename("PersonId")
        d = pd.DataFrame({"PersonId": both, "cluster": lab[both].to_numpy(),
                          "y": nxt.loc[both, "rho_eb"].to_numpy()})
        prev = (s0.set_index("PersonId").loc[both, ["n_res", "rho_eb"]]
                .rename(columns={"rho_eb": "rho_prev"})
                .rename_axis("PersonId").reset_index())
        d = d.merge(prev, on="PersonId")
        d = d.merge(P, on="PersonId", how="left")
        tr, te = train_test_split(d, test_size=0.3, random_state=RANDOM_STATE)

        def fit(cols_num, cols_cat, use_cluster):
            def design(df, enc=None, fit_enc=False):
                parts = [np.asarray(df[cols_num], dtype="float64")] if cols_num else []
                if cols_num and "n_res" in cols_num:
                    parts[0] = parts[0].copy()
                    parts[0][:, cols_num.index("n_res")] = np.log1p(
                        parts[0][:, cols_num.index("n_res")])
                cats = list(cols_cat) + (["cluster"] if use_cluster else [])
                if cats:
                    A = df[cats].astype(str)
                    if fit_enc:
                        enc = OneHotEncoder(handle_unknown="ignore", min_frequency=20,
                                            sparse_output=False).fit(A)
                    parts.append(enc.transform(A))
                return (np.hstack(parts) if parts else np.zeros((len(df), 0))), enc
            Xtr, enc = design(tr, fit_enc=True)
            Xte, _ = design(te, enc=enc)
            m = LinearRegression().fit(Xtr, tr.y)
            return float(m.score(Xte, te.y))

        demo_cols = ["demo_college", "demo_degree", "demo_session", "demo_is_dorm",
                     "demo_gender"]
        specs = [
            ("جمعیت‌شناسی تنها", [], demo_cols, False),
            ("حجم مصرف تنها", ["n_res"], [], False),
            ("نرخ سال قبل تنها", ["rho_prev"], [], False),
            ("جمعیت + حجم + نرخ", ["n_res", "rho_prev"], demo_cols, False),
            ("⇧ + برچسب خوشه", ["n_res", "rho_prev"], demo_cols, True),
            ("برچسب خوشه تنها", [], [], True),
        ]
        base = None
        for lbl, cn, cc, uc in specs:
            r2 = fit(cn, cc, uc)
            if lbl == "جمعیت + حجم + نرخ":
                base = r2
            rows.append({"جفت سال": f"{y0} → {y1}", "مشخصه": lbl, "n آموزش": len(tr),
                         "n آزمون": len(te), "R² آزمون": r2,
                         "ΔR² نسبت به پایه": (r2 - base) if base is not None
                         and lbl == "⇧ + برچسب خوشه" else np.nan})
    t = pd.DataFrame(rows)
    print(md_table(t.round(5)))
    delta = t["ΔR² نسبت به پایه"].dropna()
    print(f"\n⭐ دروازه‌ی ۹-۸-۶: ΔR² افزوده‌ی برچسب خوشه = "
          f"{delta.mean():+.5f} (میانگین دو جفت‌سال)")
    print(f"   نتیجه: {'برچسب اطلاعات تازه دارد' if delta.mean() > 0.002 else 'برچسب اطلاعات تازه ندارد ⇒ نتیجه‌ی منفی ثبت می‌شود'}")

    fig, ax = figs.new(w=8.0, h=4.6)
    piv = t.pivot_table(index="مشخصه", columns="جفت سال", values="R² آزمون")
    piv = piv.reindex([s[0] for s in specs])
    x = np.arange(len(piv))
    for i, c in enumerate(piv.columns):
        ax.bar(x + (i - (len(piv.columns) - 1) / 2) * 0.35, piv[c], 0.35,
               color=figs.PALETTE[i], label=c)
    ax.set_xticks(x, piv.index, fontsize=7, rotation=18, ha="right")
    ax.set_ylabel("$R^2$ روی افراد کنارگذاشته‌شده")
    ax.set_title("آیا برچسب خوشه چیزی به تاریخچه و جمعیت‌شناسی اضافه می‌کند؟")
    ax.legend(fontsize=8)
    figs.save(fig, "r9f_05_predictive_validity", t,
              caption="سخت‌گیرانه‌ترین آزمون دور — برچسب فقط از سال t، ارزیابی روی افراد ندیده")
    return t


def p987_null(P: pd.DataFrame, k_by_view: dict[str, int]) -> pd.DataFrame:
    """۹-۸-۷ آزمون پوچی: همان لوله روی داده‌ی ستون‌درهم‌ریخته."""
    header("۹-۸-۷ آزمون پوچی (ستون‌ها مستقلاً درهم ریخته)", 2)
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    for v, k in k_by_view.items():
        X, idx, _ = build_view_matrix(P, v)
        samp = rng.choice(len(X), size=min(10000, len(X)), replace=False)
        obs = silhouette_score(X[samp], KMeans(n_clusters=k, random_state=RANDOM_STATE,
                                               n_init=10).fit_predict(X)[samp])
        nulls = []
        for r in range(10):
            Xs = np.column_stack([rng.permutation(X[:, j]) for j in range(X.shape[1])])
            nulls.append(silhouette_score(
                Xs[samp], KMeans(n_clusters=k, random_state=RANDOM_STATE,
                                 n_init=3).fit_predict(Xs)[samp]))
        rows.append({"نما": v, "k": k, "سیلوئت مشاهده‌شده": obs,
                     "میانگین پوچ": float(np.mean(nulls)),
                     "انحراف پوچ": float(np.std(nulls)),
                     "z": (obs - np.mean(nulls)) / (np.std(nulls) + 1e-12)})
    t = pd.DataFrame(rows)
    print(md_table(t.round(4)))

    fig, ax = figs.new(w=7.6, h=4.4)
    x = np.arange(len(t))
    ax.bar(x - 0.2, t["سیلوئت مشاهده‌شده"], 0.4, color=figs.C_MAIN, label="مشاهده‌شده")
    ax.bar(x + 0.2, t["میانگین پوچ"], 0.4, yerr=t["انحراف پوچ"], color=figs.C_GREY,
           label="پوچ (درهم‌ریخته)", capsize=3)
    ax.set_xticks(x, t["نما"], fontsize=7, rotation=18, ha="right")
    ax.set_ylabel("سیلوئت")
    ax.set_title("ساختار واقعی است یا هر داده‌ای خوشه می‌دهد؟")
    ax.legend(fontsize=8)
    figs.save(fig, "r9f_06_null_test", t,
              caption="فاصله‌ی ستون مشاهده‌شده از ستون پوچ، اندازه‌ی ساختار واقعی است")
    return t


def main() -> None:
    set_global_seed()
    setup()
    P = at_threshold(load_profile(), N_MIN_MAIN)
    k_by_view = json.loads(K_PATH.read_text(encoding="utf-8"))
    header(f"۹-۸ اعتبارسنجی و پایداری — دروازه‌ی تفسیر (n={len(P):,})")
    p981_stability(P, k_by_view)
    p983_view_agreement(k_by_view)
    p984_temporal()
    p985_demographic_ari(load_profile())
    p986_predictive()
    p987_null(P, k_by_view)
    header("پایان بند ۹-۸")


if __name__ == "__main__":
    main()
