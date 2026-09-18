"""بند ۹-۶ — پروفایل جمعیتی و دم توزیع (بازآزمون F75–F79 و سقف F86).

اجرا: python -m src.eda_lib.runners.r9d_demographic
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import set_global_seed
from src.eda_lib import figs
from src.eda_lib.runners._common_r9 import (
    N_MIN_MAIN, WORST_DEFS, at_threshold, header, load_profile, md_table, setup,
)

#: ستون‌های جمعیتی و برچسبشان. `demo_field` (۸۵۹ سطح) فقط در بند ۹-۶-۵ می‌آید.
DEMO_COLS = {
    "demo_gender": "جنسیت",
    "demo_degree": "مقطع",
    "demo_session": "دوره",
    "demo_is_dorm": "ساکن خوابگاه",
    "demo_college": "دانشکده",
}

#: گروه‌بندی `EducationSession` مطابق F50/F75 — «شبانه/نوبت دوم» یک رده است.
SESSION_GROUPS = {"شبانه(نوبت دوم)": "شبانه/نوبت دوم", "نوبت دوم": "شبانه/نوبت دوم"}

V2_ANCHORS = {"enrich_night": 1.53, "enrich_dorm": 0.60, "enrich_female": 1.05,
              "enrich_male": 0.96, "r2_demo": 0.0697}


def _collapse_session(s: pd.Series) -> pd.Series:
    return s.astype(str).map(lambda v: SESSION_GROUPS.get(v, v))


def _enrichment(S: pd.DataFrame, col: str, flag: np.ndarray,
                n_boot: int = 400, seed: int = 42) -> pd.DataFrame:
    """نسبت غنی‌شدگی هر رده در دهک بدترین، با CI بوت‌استرپی سطح فرد.

    غنی‌شدگی = سهم رده در دهک ÷ سهم رده در کل. مقدار ۱ یعنی بی‌اثر.
    """
    rng = np.random.default_rng(seed)
    v = S[col].astype(str).to_numpy()
    cats = pd.unique(v)
    base = pd.Series(v).value_counts(normalize=True)
    obs = pd.Series(v[flag]).value_counts(normalize=True)
    boots = {c: [] for c in cats}
    idx = np.arange(len(S))
    for _ in range(n_boot):
        b = rng.choice(idx, size=idx.size, replace=True)
        bb = pd.Series(v[b]).value_counts(normalize=True)
        ob = pd.Series(v[b][flag[b]]).value_counts(normalize=True)
        for c in cats:
            if c in bb and bb[c] > 0:
                boots[c].append(ob.get(c, 0.0) / bb[c])
    rows = []
    for c in cats:
        arr = np.array(boots[c]) if boots[c] else np.array([np.nan])
        lo, hi = np.nanquantile(arr, [0.025, 0.975])
        rows.append({"رده": c, "افراد": int((v == c).sum()),
                     "سهم کل": float(base.get(c, 0.0)),
                     "سهم در دهک": float(obs.get(c, 0.0)),
                     "غنی‌شدگی": float(obs.get(c, 0.0) / base[c]) if base.get(c, 0) else np.nan,
                     "CI پایین": float(lo), "CI بالا": float(hi)})
    return pd.DataFrame(rows).sort_values("غنی‌شدگی", ascending=False)


def p961_enrichment(P: pd.DataFrame, n_min: int = N_MIN_MAIN) -> pd.DataFrame:
    """۹-۶-۱ غنی‌شدگی دهک هر ویژگی جمعیتی (بازآزمون F75)."""
    header(f"۹-۶-۱ غنی‌شدگی دهک بدترین (آستانه n_min={n_min})", 2)
    S = at_threshold(P, n_min).copy()
    S["demo_session"] = _collapse_session(S.demo_session)
    S["demo_is_dorm"] = np.where(S.demo_is_dorm, "خوابگاهی", "غیرخوابگاهی")

    out = []
    for defn, dlbl in WORST_DEFS.items():
        k = int(round(0.10 * len(S)))
        flag = np.zeros(len(S), dtype=bool)
        flag[np.argsort(-S[defn].to_numpy())[:k]] = True
        for col, clbl in DEMO_COLS.items():
            if col == "demo_college":
                continue
            e = _enrichment(S, col, flag)
            e.insert(0, "ویژگی", clbl)
            e.insert(0, "تعریف", dlbl)
            out.append(e)
    t = pd.concat(out, ignore_index=True)
    # ⚠️ قاعده‌ی ب در عمل: F75 غنی‌شدگی را با `flag="worst_eb"` حساب کرده
    # (`r5a_worst_decile.q37`)، پس مقایسه‌ی لنگر **باید** روی همان تعریف باشد.
    # مقایسه با «هدررفت مطلق» — که تعریف سیاستی است — وارونگی مصنوعی می‌سازد،
    # چون خوابگاهی‌ها پرمصرف‌ترند و در هر تعریف مطلقی بالا می‌آیند.
    for dlbl in (WORST_DEFS["rho_eb"], WORST_DEFS["n_waste"]):
        tag = "لنگرِ F75" if dlbl == WORST_DEFS["rho_eb"] else "تعریف سیاستی"
        print(f"\n▸ تعریف «{dlbl}» ({tag}):")
        print(md_table(t[t["تعریف"] == dlbl].drop(columns=["تعریف"]).round(4)))
    print(f"\nلنگر v2 (F75، تعریف نرخ EB): شبانه/نوبت‌دوم {V2_ANCHORS['enrich_night']} · "
          f"خوابگاهی {V2_ANCHORS['enrich_dorm']} · زن {V2_ANCHORS['enrich_female']} · "
          f"مرد {V2_ANCHORS['enrich_male']}")

    fig, axes = figs.new(1, 3, w=5.0, h=4.6)
    for ax, (defn, dlbl) in zip(axes.ravel(), WORST_DEFS.items()):
        sub = t[t["تعریف"] == dlbl].sort_values("غنی‌شدگی")
        sub = sub[sub["افراد"] >= 200]
        y = np.arange(len(sub))
        ax.errorbar(sub["غنی‌شدگی"], y,
                    xerr=[sub["غنی‌شدگی"] - sub["CI پایین"], sub["CI بالا"] - sub["غنی‌شدگی"]],
                    fmt="o", color=figs.C_MAIN, capsize=3, ms=5)
        ax.axvline(1.0, color=figs.C_ALT, ls="--", lw=1.3)
        ax.set_yticks(y, sub["رده"], fontsize=8)
        ax.set_xlabel("نسبت غنی‌شدگی")
        ax.set_title(dlbl, fontsize=10)
    figs.save(fig, "r9d_01_decile_enrichment", t,
              caption="کدام ویژگی در دم توزیع غنی می‌شود — و اینکه پاسخ به تعریف «بدترین» وابسته است")
    return t


def p962_dorm_interaction(P: pd.DataFrame, n_min: int = N_MIN_MAIN) -> pd.DataFrame:
    """۹-۶-۲/۹-۶-۳ آیا ریسک جمعیتی پدیده‌ای غیرخوابگاهی است؟ (بازآزمون F77/F78)."""
    header(f"۹-۶-۲ برهم‌کنش جمعیت‌شناسی با سکونت خوابگاه (آستانه n_min={n_min})", 2)
    import statsmodels.formula.api as smf

    S = at_threshold(P, n_min).copy()
    S["session_g"] = _collapse_session(S.demo_session)
    S["is_night"] = (S.session_g == "شبانه/نوبت دوم").astype(int)
    S["is_grad"] = S.demo_degree.astype(str).str.contains("ارشد|دكتري|دکتری|دكترا", na=False).astype(int)
    S["is_male"] = (S.demo_gender.astype(str).str.strip() == "مرد").astype(int)
    S["is_dorm"] = S.demo_is_dorm.astype(int)

    rows = []
    for sub_lbl, sub in (("کل", S), ("خوابگاهی", S[S.is_dorm == 1]),
                         ("غیرخوابگاهی", S[S.is_dorm == 0])):
        m = smf.ols("rho_eb ~ is_night + is_grad + is_male", data=sub).fit(cov_type="HC3")
        for var in ("is_night", "is_grad", "is_male"):
            rows.append({"زیرجامعه": sub_lbl, "متغیر": var, "n": int(m.nobs),
                         "ضریب": float(m.params[var]), "p": float(m.pvalues[var])})
    t = pd.DataFrame(rows)
    print(md_table(t.round(5)))

    full = smf.ols("rho_eb ~ (is_night + is_grad + is_male) * is_dorm", data=S).fit(cov_type="HC3")
    print("\nمدل کامل با برهم‌کنش (HC3):")
    inter = pd.DataFrame({"جمله": full.params.index, "ضریب": full.params.values,
                          "p": full.pvalues.values})
    print(md_table(inter[inter["جمله"].str.contains(":")].round(5)))

    fig, ax = figs.new(w=7.6, h=4.4)
    piv = t.pivot(index="متغیر", columns="زیرجامعه", values="ضریب")
    x = np.arange(len(piv))
    for i, c in enumerate(["خوابگاهی", "غیرخوابگاهی"]):
        ax.bar(x + (i - 0.5) * 0.35, piv[c], 0.35,
               color=[figs.C_MAIN, figs.C_ALT][i], label=c)
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x, piv.index)
    ax.set_ylabel("ضریب روی نرخ منقبض‌شده")
    ax.set_title("همان ویژگی، دو رژیم — بازآزمون F77")
    ax.legend()
    figs.save(fig, "r9d_02_dorm_interaction", t,
              caption="آیا اثر جمعیتی داخل و بیرون خوابگاه یکی است؟")
    return t


def p964_college_ranking(P: pd.DataFrame, n_min: int = N_MIN_MAIN) -> pd.DataFrame:
    """۹-۶-۴ رتبه‌بندی دانشکده‌ها بین دو زیرجامعه (بازآزمون F79)."""
    header(f"۹-۶-۴ پایداری رتبه‌ی دانشکده بین خوابگاهی و غیرخوابگاهی (n_min={n_min})", 2)
    from scipy import stats as st

    S = at_threshold(P, n_min)
    g = (S.groupby([S.demo_college.astype(str), S.demo_is_dorm], observed=True)
          .agg(n=("rho_eb", "size"), rho=("rho_eb", "mean")).reset_index())
    g.columns = ["college", "is_dorm", "n", "rho"]
    w = g.pivot(index="college", columns="is_dorm", values="rho")
    n = g.pivot(index="college", columns="is_dorm", values="n")
    both = w.dropna()
    both = both[(n.loc[both.index] >= 50).all(axis=1)]
    sp = st.spearmanr(both[True], both[False])
    pe = st.pearsonr(both[True], both[False])
    diff = both[True] - both[False]
    print(f"دانشکده‌های مشترک با ≥۵۰ نفر در هر دو زیرجامعه: {len(both)}")
    print(f"اسپیرمن = {sp.statistic:.4f} (p={sp.pvalue:.2e}) · پیرسون = {pe.statistic:.4f}")
    print(f"دانشکده‌هایی که خوابگاهی‌شان نرخ کمتری دارد: {int((diff < 0).sum())} از {len(both)}")
    print("لنگر v2 (F79): اسپیرمن +۰.۵۹۷ · ۲۰ از ۲۰ اختلاف منفی")

    t = pd.DataFrame({"دانشکده": both.index, "خوابگاهی": both[True].values,
                      "غیرخوابگاهی": both[False].values, "اختلاف": diff.values})
    fig, ax = figs.new(w=6.4, h=5.6)
    ax.scatter(both[False], both[True], s=28, color=figs.C_MAIN, alpha=0.8)
    lim = [min(both.min()) * 0.95, max(both.max()) * 1.05]
    ax.plot(lim, lim, color=figs.C_GREY, ls=":", lw=1.4)
    ax.set_xlabel("نرخ EB — غیرخوابگاهی")
    ax.set_ylabel("نرخ EB — خوابگاهی")
    ax.set_title("هر نقطه یک دانشکده")
    figs.annotate(ax, f"اسپیرمن={sp.statistic:.3f}\n"
                      f"{int((diff < 0).sum())}/{len(both)} زیر قطر")
    figs.save(fig, "r9d_03_college_ranking", t.round(5),
              caption="نقطه‌ی زیر قطر = خوابگاهی‌های آن دانشکده کم‌نرخ‌ترند")
    return t


def p965_r2_ceiling(P: pd.DataFrame, n_min: int = N_MIN_MAIN) -> pd.DataFrame:
    """⭐ ۹-۶-۵ سقف توضیح‌دهندگی کل جمعیت‌شناسی (لنگر F86 = ۰.۰۶۹۷)."""
    header(f"۹-۶-۵ سقف $R^2$ جمعیت‌شناسی روی نرخ EB (آستانه n_min={n_min})", 2)
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import OneHotEncoder

    S = at_threshold(P, n_min).copy()
    S["session_g"] = _collapse_session(S.demo_session)
    y = S.rho_eb.to_numpy()

    specs = {
        "جنسیت": ["demo_gender"],
        "مقطع": ["demo_degree"],
        "دوره": ["session_g"],
        "خوابگاه": ["demo_is_dorm"],
        "دانشکده": ["demo_college"],
        "چهار ویژگی پایه": ["demo_gender", "demo_degree", "session_g", "demo_is_dorm"],
        "پایه + دانشکده": ["demo_gender", "demo_degree", "session_g", "demo_is_dorm",
                           "demo_college"],
        "پایه + دانشکده + رشته": ["demo_gender", "demo_degree", "session_g", "demo_is_dorm",
                                  "demo_college", "demo_field"],
        "حجم مصرف تنها (یک ستون)": ["__n_res"],
    }
    rows = []
    for lbl, cols in specs.items():
        if cols == ["__n_res"]:
            X = np.log1p(S.n_res.to_numpy()).reshape(-1, 1)
            k = 1
        else:
            enc = OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=True)
            X = enc.fit_transform(S[cols].astype(str))
            k = X.shape[1]
        m = LinearRegression().fit(X, y)
        r2 = float(m.score(X, y))
        n = len(y)
        rows.append({"مشخصه": lbl, "پارامتر": k, "R²": r2,
                     "R² تعدیل‌شده": 1 - (1 - r2) * (n - 1) / (n - k - 1)})
    t = pd.DataFrame(rows)
    print(md_table(t.round(5)))
    print(f"\nلنگر v2 (F86): R² جمعیت‌شناسی روی نرخ EB = {V2_ANCHORS['r2_demo']}")

    fig, ax = figs.new(w=7.8, h=4.6)
    o = t.sort_values("R²")
    c = [figs.C_ACCENT if "حجم" in s else figs.C_MAIN for s in o["مشخصه"]]
    ax.barh(range(len(o)), o["R²"], color=c)
    ax.set_yticks(range(len(o)), o["مشخصه"], fontsize=8)
    ax.axvline(V2_ANCHORS["r2_demo"], color=figs.C_ALT, ls="--", lw=1.3)
    ax.set_xlabel("$R^2$ روی نرخ منقبض‌شده‌ی فرد")
    ax.set_title("سقف کل جمعیت‌شناسی — و آنچه یک ستون از آن می‌گیرد")
    figs.annotate(ax, f"خط‌چین: لنگر v2 ({V2_ANCHORS['r2_demo']})", loc="lower right")
    figs.save(fig, "r9d_04_r2_ceiling", t,
              caption="سقف جمعیت‌شناسی — عدد مرجعی که خوشه‌بندی بند ۹-۸ باید از آن عبور کند")
    return t


def main() -> None:
    set_global_seed()
    setup()
    P = load_profile()
    header("۹-۶ پروفایل جمعیتی و دم توزیع")
    p961_enrichment(P, n_min=5)    # آستانه‌ی خود دور ۵ (MIN_RES_PROFILE)
    p961_enrichment(P)             # آستانه‌ی اصلی دور ۹
    p962_dorm_interaction(P)
    p964_college_ranking(P)
    p965_r2_ceiling(P)
    header("پایان بند ۹-۶")


if __name__ == "__main__":
    main()
