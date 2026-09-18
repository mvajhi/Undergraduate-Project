"""بند ۸-۶ — اطلس موجودیت و مکان (دور ۸).

اجرا: python -m src.eda_lib.runners.r8c_entity_atlas
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.config import set_global_seed
from src.eda_lib import figs
from src.eda_lib.runners._common_v4 import academic_year_order, header, load_features, md_table

MEAL_FA = {"lunch": "ناهار", "dinner": "شام"}


def _rate(g):
    return float(g.NoRecv.sum() / g.Res.sum()) if g.Res.sum() else np.nan


# ---------------------------------------------------------------------------
# ۸-۶-۱ پروفایل ۷۱ سلف
# ---------------------------------------------------------------------------

def p861_profile(f: pd.DataFrame) -> pd.DataFrame:
    header("۸-۶-۱ پروفایل ۷۱ سلف — نرخ، حجم، پوشش، نوسان", 2)
    g = (f.groupby("restaurant", observed=True)
         .apply(lambda x: pd.Series(dict(
             venue_type=x.venue_type.iloc[0], campus=x.campus.iloc[0],
             days=x.date_gregorian.nunique(), cells=len(x), res=x.Res.sum(),
             waste=x.NoRecv.sum(), rate=_rate(x),
             rate_sd_daily=x.groupby("date_gregorian", observed=True)
                            .apply(lambda d: _rate(d), include_groups=False).std(),
             dishes=x.dish.nunique(), zero_share=float((x.rho == 0).mean()))),
                include_groups=False).reset_index())
    g["share_waste"] = g.waste / g.waste.sum()
    g["share_res"] = g.res / g.res.sum()
    g = g.sort_values("res", ascending=False).reset_index(drop=True)
    overall = _rate(f)

    fig, ax = figs.new(1, 2, w=7.4, h=6.2)
    top = g.head(28)
    y = np.arange(len(top))
    colors = {vt: figs.PALETTE[i % len(figs.PALETTE)] for i, vt in enumerate(sorted(g.venue_type.unique()))}
    ax[0].barh(y, top.rate, color=[colors[v] for v in top.venue_type], height=0.74)
    ax[0].set_yticks(y); ax[0].set_yticklabels(top.restaurant, fontsize=7)
    ax[0].invert_yaxis(); ax[0].axvline(overall, color="black", lw=1.2, ls="--")
    ax[0].set_xlabel("نرخ وزنی"); ax[0].set_title("۲۸ سلف پرحجم — نرخ (خط‌چین: کل)")
    handles = [__import__("matplotlib").patches.Patch(color=c, label=v) for v, c in colors.items()]
    ax[0].legend(handles=handles, fontsize=7, loc="lower right")

    ax[1].scatter(g.res, g.rate, s=np.sqrt(g.cells) * 3.2,
                  c=[colors[v] for v in g.venue_type], alpha=0.8, edgecolor="white", lw=0.5)
    ax[1].set_xscale("log")
    ax[1].axhline(overall, color="black", lw=1.2, ls="--")
    ax[1].set_xlabel("حجم رزرو سلف (مقیاس لگاریتمی)"); ax[1].set_ylabel("نرخ وزنی")
    ax[1].set_title("نرخ در برابر حجم — اندازه‌ی نقطه = تعداد سلول")
    r_sp = stats.spearmanr(g.res, g.rate)
    top5 = g.nlargest(5, "share_waste")
    figs.annotate(ax[1], f"همبستگی رتبه‌ای حجم↔نرخ {r_sp.statistic:+.2f} (p={r_sp.pvalue:.2g})\n"
                         f"۵ سلف پرهدررفت = {top5.share_waste.sum():.1%} کل هدررفت\n"
                         f"با {top5.share_res.sum():.1%} حجم")
    figs.save(fig, "r8c_01_venue_profile", g)
    print(md_table(g.head(30)))
    print(f"\nکل سلف: {len(g)} · نرخ کل {overall:.4f} · "
          f"دامنه‌ی نرخ سلف‌ها {g.rate.min():.3f} تا {g.rate.max():.3f}")
    return g


# ---------------------------------------------------------------------------
# ۸-۶-۲ آیا venue_type رده‌بندی درستی است؟
# ---------------------------------------------------------------------------

def p862_taxonomy(f: pd.DataFrame, prof: pd.DataFrame) -> pd.DataFrame:
    header("۸-۶-۲ آیا `venue_type` رده‌بندی درستی است؟ — خوشه‌بندی مستقل", 2)
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import adjusted_rand_score
    from sklearn.preprocessing import StandardScaler

    d = f.copy()
    d["dow"] = d.day_of_week
    # امضای رفتاری هر سلف: نرخ به تفکیک روز هفته و وعده + سهم وعده + اندازه
    sig = d.pivot_table(index="restaurant", columns=["meal", "dow"],
                        values="NoRecv", aggfunc="sum")
    den = d.pivot_table(index="restaurant", columns=["meal", "dow"], values="Res", aggfunc="sum")
    sig = (sig / den.where(den >= 200))
    sig = sig.dropna(thresh=int(sig.shape[1] * 0.5))
    sig = sig.fillna(sig.mean())
    meta = prof.set_index("restaurant").loc[sig.index]
    X = np.column_stack([StandardScaler().fit_transform(sig.values),
                         StandardScaler().fit_transform(
                             np.log1p(meta[["res", "cells", "dishes"]].values))])
    rows = []
    for k in range(2, 9):
        lab = AgglomerativeClustering(n_clusters=k).fit_predict(X)
        rows.append(dict(k=k, ari_vs_venue_type=adjusted_rand_score(meta.venue_type, lab),
                         ari_vs_campus=adjusted_rand_score(meta.campus, lab)))
    out = pd.DataFrame(rows)
    k6 = AgglomerativeClustering(n_clusters=6).fit_predict(X)
    ct = pd.crosstab(meta.venue_type, k6)

    fig, ax = figs.new(1, 2, w=6.6, h=4.4)
    ax[0].plot(out.k, out.ari_vs_venue_type, "o-", lw=2, color=figs.C_MAIN, label="در برابر `venue_type`")
    ax[0].plot(out.k, out.ari_vs_campus, "s--", lw=2, color=figs.C_ALT, label="در برابر پردیس")
    ax[0].set_xlabel("تعداد خوشه"); ax[0].set_ylabel("ARI با رده‌بندی موجود")
    ax[0].legend(loc="upper right"); ax[0].set_title("خوشه‌بندی رفتاری چقدر رده‌بندی موجود را بازتولید می‌کند؟")
    ax[0].set_ylim(-0.05, 1.0)
    figs.annotate(ax[0], f"بیشینه‌ی ARI با نوع سلف {out.ari_vs_venue_type.max():.2f}\n"
                         f"‏۱.۰ = انطباق کامل، ۰ = تصادفی", loc="upper left", fontsize=8)

    im = ax[1].imshow(ct.values, aspect="auto", cmap="Blues")
    ax[1].set_xticks(range(ct.shape[1])); ax[1].set_xticklabels([f"خوشه {i}" for i in ct.columns], fontsize=8)
    ax[1].set_yticks(range(ct.shape[0])); ax[1].set_yticklabels(ct.index, fontsize=8)
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            if ct.values[i, j]:
                ax[1].text(j, i, ct.values[i, j], ha="center", va="center", fontsize=9,
                           color="white" if ct.values[i, j] > ct.values.max() * 0.5 else "black")
    ax[1].grid(False); ax[1].set_title("نوع سلف در برابر خوشه‌ی رفتاری (k=۶)")
    figs.save(fig, "r8c_02_venue_taxonomy", out)
    print(md_table(out)); print(); print(ct.to_string())
    return out


# ---------------------------------------------------------------------------
# ۸-۶-۳ اثر ثابت یا تصادفی؟
# ---------------------------------------------------------------------------

def p863_shrinkage(f: pd.DataFrame, prof: pd.DataFrame) -> pd.DataFrame:
    header("۸-۶-۳ اثر سلف — ثابت یا تصادفی؟ (لنگر: F155 واگرایی +۷۸٪)", 2)
    g = prof.copy()
    mu = _rate(f)
    # تخمین تجربی-بیز: واریانس بین‌سلفی در برابر واریانس نمونه‌گیری درون‌سلفی
    g["var_sampling"] = g.rate * (1 - g.rate) / g.res
    tau2 = max(float(g.rate.var() - g.var_sampling.mean()), 1e-9)
    g["w"] = tau2 / (tau2 + g.var_sampling)
    g["rate_shrunk"] = g.w * g.rate + (1 - g.w) * mu
    g["shrink_pct"] = 1 - g.w

    # پایداری بین سال‌ها: آیا رتبه‌ی سلف تکرار می‌شود؟
    by = (f.groupby(["restaurant", "academic_year"], observed=True)
          .apply(lambda x: pd.Series(dict(rate=_rate(x), res=x.Res.sum())), include_groups=False)
          .reset_index())
    piv = by[by.res >= 5000].pivot_table(index="restaurant", columns="academic_year", values="rate")
    years = [y for y in academic_year_order() if y in piv.columns]
    pair_rows = []
    for a in years:
        for b in years:
            if a < b:
                both = piv[[a, b]].dropna()
                if len(both) >= 8:
                    pair_rows.append(dict(year_a=a, year_b=b, n=len(both),
                                          pearson=stats.pearsonr(both[a], both[b]).statistic,
                                          spearman=stats.spearmanr(both[a], both[b]).statistic))
    pairs = pd.DataFrame(pair_rows)

    fig, ax = figs.new(1, 2, w=6.6, h=4.3)
    ax[0].scatter(g.rate, g.rate_shrunk, s=np.sqrt(g.res) / 8, color=figs.C_MAIN, alpha=0.8)
    lim = [0, g.rate.max() * 1.05]
    ax[0].plot(lim, lim, color="black", lw=1, ls=":")
    ax[0].axhline(mu, color=figs.C_ALT, lw=1.2, ls="--")
    ax[0].set_xlabel("نرخ خام سلف"); ax[0].set_ylabel("نرخ پس از انقباض تجربی-بیز")
    ax[0].set_title("انقباض: کدام سلف‌ها اطلاعات کافی ندارند؟")
    figs.annotate(ax[0], f"τ² بین‌سلفی {tau2:.2e}\nمیانه‌ی انقباض {g.shrink_pct.median():.1%}\n"
                         f"بیشینه {g.shrink_pct.max():.1%}")

    if len(years) >= 2:
        a, b = years[0], years[1]
        both = piv[[a, b]].dropna()
        ax[1].scatter(both[a], both[b], s=48, color=figs.C_OK, alpha=0.85, edgecolor="white")
        lim2 = [min(both.min()) * 0.9, max(both.max()) * 1.05]
        ax[1].plot(lim2, lim2, color="black", lw=1, ls=":")
        ax[1].set_xlabel(f"نرخ سلف در {a}"); ax[1].set_ylabel(f"نرخ سلف در {b}")
        ax[1].set_title("آیا رتبه‌ی سلف بین سال‌ها می‌ماند؟")
        r = stats.pearsonr(both[a], both[b])
        figs.annotate(ax[1], f"n={len(both)} سلف\nپیرسون {r.statistic:+.2f}\n"
                             f"اسپیرمن {stats.spearmanr(both[a], both[b]).statistic:+.2f}", loc="upper left")
    figs.save(fig, "r8c_03_venue_shrinkage", g[["restaurant", "rate", "rate_shrunk", "shrink_pct", "res"]])
    print(md_table(pairs))
    print(f"\nτ² = {tau2:.3e} · میانه‌ی انقباض {g.shrink_pct.median():.3f} · "
          f"سلف‌های با انقباض >۵۰٪: {(g.shrink_pct > 0.5).sum()}")
    return pairs


# ---------------------------------------------------------------------------
# ۸-۶-۴ هم‌خطی لانه‌ای مکان
# ---------------------------------------------------------------------------

def p864_nesting(f: pd.DataFrame) -> pd.DataFrame:
    header("۸-۶-۴ هم‌خطی لانه‌ای شهر / پردیس / سلف (لنگر: F146، F206)", 2)
    d = f[["city", "campus", "restaurant", "venue_type"]].drop_duplicates()
    rows = []
    for a, b in [("city", "campus"), ("campus", "city"), ("campus", "restaurant"),
                 ("venue_type", "campus"), ("city", "restaurant")]:
        ct = pd.crosstab(d[a], d[b])
        # V کرامر و «آیا a تابعی از b است؟»
        chi2 = stats.chi2_contingency(ct).statistic
        n = ct.values.sum()
        v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
        functional = float((ct.gt(0).sum(axis=0) == 1).mean())  # هر مقدار b دقیقاً یک a دارد؟
        rows.append(dict(a=a, b=b, n_a=d[a].nunique(), n_b=d[b].nunique(),
                         cramers_v=v, b_determines_a=functional))
    out = pd.DataFrame(rows)

    # واریانس توضیح‌داده‌شده‌ی لانه‌ای روی نرخ سلول
    lvl = []
    y = f.rho.to_numpy()
    tot = np.var(y)
    for col in ["city", "campus", "venue_type", "restaurant"]:
        grp = f.groupby(col, observed=True).rho.transform("mean").to_numpy()
        lvl.append(dict(level=col, n_groups=f[col].nunique(),
                        r2=float(1 - np.var(y - grp) / tot)))
    lvl = pd.DataFrame(lvl)

    fig, ax = figs.new(1, 2, w=6.6, h=4.2)
    x = np.arange(len(out))
    ax[0].bar(x, out.cramers_v, color=figs.C_MAIN)
    ax[0].set_xticks(x); ax[0].set_xticklabels([f"{r.a}\n↔ {r.b}" for r in out.itertuples()], fontsize=8)
    ax[0].set_ylabel("V کرامر"); ax[0].set_ylim(0, 1.05)
    ax[0].axhline(1, color=figs.C_ALT, lw=1.2, ls="--")
    ax[0].set_title("هم‌خطی محورهای مکان — ۱ یعنی اطلاعات تکراری")
    figs.annotate(ax[0], "V=۱ ⇒ ستون دوم ستون اول را کامل تعیین می‌کند", loc="lower left", fontsize=8)

    figs.rtl_barh(ax[1], lvl.level.tolist(), lvl.r2.tolist(), color=figs.C_OK, fmt="{:.3f}", pad=0.004)
    ax[1].set_xlabel("سهم واریانس نرخ سلول")
    ax[1].set_title("هر لایه‌ی مکان چقدر توضیح می‌دهد؟")
    figs.save(fig, "r8c_04_location_nesting", pd.concat([out, lvl], ignore_index=True))
    print(md_table(out)); print(); print(md_table(lvl))
    return out


# ---------------------------------------------------------------------------
# ۸-۶-۵ ورود و خروج موجودیت‌ها
# ---------------------------------------------------------------------------

def p865_coverage(f: pd.DataFrame, prof: pd.DataFrame) -> pd.DataFrame:
    header("۸-۶-۵ ورود و خروج سلف‌ها در سه سال", 2)
    d = f.copy()
    d["month"] = d.date_gregorian.dt.to_period("M").dt.to_timestamp()
    piv = d.pivot_table(index="restaurant", columns="month", values="Res", aggfunc="sum")
    order = prof.set_index("restaurant").res.reindex(piv.index).sort_values(ascending=False).index
    piv = piv.loc[order]
    present = piv.notna()

    fig, ax = figs.new(1, 2, w=7.2, h=6.0)
    ax[0].imshow(present.values, aspect="auto", cmap="Greys", interpolation="nearest")
    ax[0].set_yticks(range(0, len(piv), 3)); ax[0].set_yticklabels(piv.index[::3], fontsize=6)
    xt = range(0, piv.shape[1], 4)
    ax[0].set_xticks(list(xt)); ax[0].set_xticklabels([piv.columns[i].strftime("%Y-%m") for i in xt],
                                                      rotation=60, fontsize=7)
    ax[0].grid(False); ax[0].set_title("نقشه‌ی حضور سلف × ماه")

    n_active = present.sum(axis=0)
    ax[1].plot(piv.columns, n_active.values, "o-", lw=2, color=figs.C_MAIN)
    ax[1].set_ylabel("تعداد سلف فعال"); ax[1].tick_params(axis="x", rotation=60)
    ax[1].set_title("تعداد سلف فعال در هر ماه")
    life = pd.DataFrame(dict(
        restaurant=piv.index,
        first=[piv.columns[np.argmax(present.loc[r].values)] for r in piv.index],
        last=[piv.columns[len(piv.columns) - 1 - np.argmax(present.loc[r].values[::-1])] for r in piv.index],
        months_active=present.sum(axis=1).values))
    always = int((life.months_active == piv.shape[1]).sum())
    figs.annotate(ax[1], f"سلف حاضر در همه‌ی {piv.shape[1]} ماه: {always}\n"
                         f"میانه‌ی ماه فعال: {int(life.months_active.median())}\n"
                         f"کمینه/بیشینه‌ی فعال: {n_active.min()}/{n_active.max()}", loc="lower left")
    figs.save(fig, "r8c_05_entity_coverage", life)
    print(md_table(life.sort_values("months_active").head(15)))
    return life


# ---------------------------------------------------------------------------
# ۸-۶-۶ سلفی که مسئله برایش تعریف نیست
# ---------------------------------------------------------------------------

def p866_degenerate(f: pd.DataFrame, prof: pd.DataFrame) -> pd.DataFrame:
    header("۸-۶-۶ سلف‌هایی که مسئله برایشان تعریف نشده (لنگر: F151)", 2)
    g = prof.copy()
    big = f[f.Res >= 30]
    zb = (big.groupby("restaurant", observed=True)
          .apply(lambda x: pd.Series(dict(zero_share_big=float((x.rho == 0).mean()), n_big=len(x))),
                 include_groups=False).reset_index())
    g = g.merge(zb, on="restaurant", how="left")
    flag = g[(g.zero_share_big > 0.5) & (g.n_big >= 30)].sort_values("zero_share_big", ascending=False)

    fig, ax = figs.new(1, 2, w=6.6, h=4.3)
    ax[0].scatter(g.n_big.fillna(0), g.zero_share_big.fillna(0), s=np.sqrt(g.res) / 8,
                  color=figs.C_MAIN, alpha=0.75)
    ax[0].axhline(0.5, color=figs.C_ALT, lw=1.2, ls="--")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("تعداد سلول بزرگ (≥۳۰ رزرو)")
    ax[0].set_ylabel("سهم سلول‌های بزرگ با نرخ صفر")
    ax[0].set_title("سلف با نرخ عمدتاً صفر — مسئله آنجا تعریف نیست")
    figs.annotate(ax[0], f"سلف‌های بالای خط: {len(flag)}\n"
                         f"سهمشان از حجم {flag.share_res.sum():.2%}\n"
                         f"سهمشان از هدررفت {flag.share_waste.sum():.2%}")
    topz = g.nlargest(12, "zero_share_big")
    figs.rtl_barh(ax[1], topz.restaurant.tolist(), topz.zero_share_big.fillna(0).tolist(),
                  color=figs.C_WARN)
    ax[1].tick_params(axis="y", labelsize=7)
    ax[1].set_xlabel("سهم سلول‌های بزرگ با نرخ صفر")
    ax[1].set_title("۱۲ سلف با بیشترین صفر ساختاری")
    figs.save(fig, "r8c_06_degenerate_venues", g[["restaurant", "venue_type", "res", "rate",
                                                  "zero_share_big", "n_big", "share_waste"]])
    print(md_table(flag[["restaurant", "venue_type", "res", "rate", "zero_share_big", "n_big"]]))
    return flag


def main() -> None:
    set_global_seed()
    figs.setup()
    f = load_features()
    header(f"دور ۸ · اطلس موجودیت — {f.restaurant.nunique()} سلف، {f.campus.nunique()} پردیس")
    prof = p861_profile(f)
    p862_taxonomy(f, prof)
    p863_shrinkage(f, prof)
    p864_nesting(f)
    p865_coverage(f, prof)
    p866_degenerate(f, prof)


if __name__ == "__main__":
    main()
