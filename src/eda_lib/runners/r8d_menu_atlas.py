"""بند ۸-۷ — اطلس منو و غذا (دور ۸).

محور غذا در هیچ دوری موضوع اصلی یک بسته نبوده. F159/F162 گفتند اثر غذا
**درون‌سلفی** است نه سراسری — ادعایی که فقط با یک آماره آزموده شد.

اجرا: python -m src.eda_lib.runners.r8d_menu_atlas
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.config import set_global_seed
from src.eda_lib import figs
from src.eda_lib.runners._common_v4 import header, load_features, md_table

MEAL_FA = {"lunch": "ناهار", "dinner": "شام"}


def _rate(g):
    return float(g.NoRecv.sum() / g.Res.sum()) if g.Res.sum() else np.nan


# ---------------------------------------------------------------------------
# ۸-۷-۱ توزیع نرخ به تفکیک غذا
# ---------------------------------------------------------------------------

def p871_dish_forest(f: pd.DataFrame) -> pd.DataFrame:
    header("۸-۷-۱ نرخ به تفکیک غذا — با بازه‌ی اطمینان بلوکی", 2)
    g = (f.groupby("dish", observed=True)
         .apply(lambda x: pd.Series(dict(
             food_group=x.food_group.iloc[0], n_cells=len(x), res=x.Res.sum(),
             venues=x.restaurant.nunique(), days=x.date_gregorian.nunique(),
             rate=_rate(x))), include_groups=False).reset_index())
    overall = _rate(f)
    big = g[g.res >= 20000].sort_values("rate").reset_index(drop=True)

    # CI بلوکی دوبعدی برای ۱۲ غذای پرحجم و ۱۲ کم‌نرخ‌ترین/پرنرخ‌ترین
    sel = pd.concat([big.head(9), big.tail(9)]).drop_duplicates("dish")
    cis = []
    for dish in sel.dish:
        sub = f[f.dish == dish]
        lo, hi = figs.block_bootstrap_rate_ci(sub, n_boot=400)
        cis.append(dict(dish=dish, lo=lo, hi=hi))
    sel = sel.merge(pd.DataFrame(cis), on="dish")

    fig, ax = figs.new(1, 2, w=6.8, h=5.4)
    y = np.arange(len(sel))
    ax[0].errorbar(sel.rate, y, xerr=[sel.rate - sel.lo, sel.hi - sel.rate],
                   fmt="o", color=figs.C_MAIN, ecolor=figs.C_GREY, capsize=3, ms=5)
    ax[0].set_yticks(y); ax[0].set_yticklabels(sel.dish, fontsize=7); ax[0].invert_yaxis()
    ax[0].axvline(overall, color=figs.C_ALT, lw=1.3, ls="--")
    ax[0].set_xlabel("نرخ وزنی (بازه‌ی بلوکی ۹۵٪)")
    ax[0].set_title("غذاهای پرحجم — کم‌نرخ‌ترین و پرنرخ‌ترین")
    n_sig = int(((sel.lo > overall) | (sel.hi < overall)).sum())
    figs.annotate(ax[0], f"{n_sig} از {len(sel)} غذا بازه‌شان\nنرخ کل ({overall:.1%}) را در بر نمی‌گیرد",
                  loc="lower right", fontsize=8)

    gg = (f.groupby("food_group", observed=True)
          .apply(lambda x: pd.Series(dict(rate=_rate(x), res=x.Res.sum(), dishes=x.dish.nunique())),
                 include_groups=False).reset_index().sort_values("rate", ascending=False))
    figs.rtl_barh(ax[1], gg.food_group.tolist(), gg.rate.tolist(), color=figs.C_OK)
    ax[1].axvline(overall, color=figs.C_ALT, lw=1.3, ls="--")
    ax[1].set_xlabel("نرخ وزنی"); ax[1].set_title("گروه غذا — دامنه‌ی بسیار باریک‌تر")
    figs.annotate(ax[1], f"دامنه‌ی غذا {big.rate.min():.1%}–{big.rate.max():.1%}\n"
                         f"دامنه‌ی گروه {gg.rate.min():.1%}–{gg.rate.max():.1%}", loc="lower right", fontsize=8)
    figs.save(fig, "r8d_01_dish_forest", g)
    print(md_table(big.head(12))); print("..."); print(md_table(big.tail(12)))
    return g


# ---------------------------------------------------------------------------
# ۸-۷-۲ اثر غذا: درون‌سلفی یا سراسری؟
# ---------------------------------------------------------------------------

def p872_within_venue(f: pd.DataFrame) -> pd.DataFrame:
    header("۸-۷-۲ اثر غذا درون‌سلفی در برابر سراسری (لنگر: F159/F162)", 2)
    d = f.copy()
    y = d.rho.to_numpy()
    tot = np.var(y)

    def r2(cols):
        grp = d.groupby(cols, observed=True).rho.transform("mean").to_numpy()
        return float(1 - np.var(y - grp) / tot)

    rows = [
        dict(model="غذا (سراسری)", r2=r2(["dish"]), k=d.dish.nunique()),
        dict(model="گروه غذا", r2=r2(["food_group"]), k=d.food_group.nunique()),
        dict(model="سلف", r2=r2(["restaurant"]), k=d.restaurant.nunique()),
        dict(model="سلف + غذا (جمعی)", r2=np.nan, k=np.nan),
        dict(model="سلف×غذا (درون‌سلفی)", r2=r2(["restaurant", "dish"]),
             k=d.groupby(["restaurant", "dish"], observed=True).ngroups),
        dict(model="سلف×وعده", r2=r2(["restaurant", "meal"]),
             k=d.groupby(["restaurant", "meal"], observed=True).ngroups),
        dict(model="روز (شوک مشترک)", r2=r2(["date_gregorian"]), k=d.date_gregorian.nunique()),
    ]
    # مدل جمعی سلف + غذا با حداقل مربعات روی متغیر مجازی (بدون برهم‌کنش)
    ven = d.groupby("restaurant", observed=True).rho.transform("mean")
    dish_dev = d.rho - d.groupby("dish", observed=True).rho.transform("mean")
    add = ven - dish_dev.groupby(d.dish).transform("mean") * 0 + (
        d.groupby("dish", observed=True).rho.transform("mean") - d.rho.mean())
    rows[3]["r2"] = float(1 - np.var(y - add) / tot)
    rows[3]["k"] = d.restaurant.nunique() + d.dish.nunique()
    out = pd.DataFrame(rows)
    out["r2_per_param"] = out.r2 / out.k

    # آیا رتبه‌ی غذا درون سلف‌های مختلف یکسان است؟
    big = d[d.groupby(["restaurant", "dish"], observed=True).rho.transform("size") >= 20]
    piv = (big.groupby(["restaurant", "dish"], observed=True)
           .apply(lambda x: _rate(x), include_groups=False).unstack())
    piv = piv.dropna(thresh=12, axis=0).dropna(thresh=8, axis=1)
    cm = piv.T.corr(method="spearman", min_periods=8)
    cors = cm.to_numpy()[np.triu_indices_from(cm, k=1)]
    cors = cors[np.isfinite(cors)]

    fig, ax = figs.new(1, 2, w=6.6, h=4.3)
    o = out.dropna().sort_values("r2")
    figs.rtl_barh(ax[0], o.model.tolist(), o.r2.tolist(), color=figs.C_MAIN, fmt="{:.3f}", pad=0.004)
    ax[0].set_xlabel("سهم واریانس نرخ سلول")
    ax[0].set_title("کدام برش، واریانس را می‌گیرد؟")
    figs.annotate(ax[0], "«روز» لنگر است: سقف هر برش ایستا", loc="lower right", fontsize=8)

    ax[1].hist(cors, bins=30, color=figs.C_OK, alpha=0.9)
    ax[1].axvline(0, color="black", lw=1.2)
    ax[1].axvline(float(np.median(cors)), color=figs.C_ALT, lw=2, ls="--")
    ax[1].set_xlabel("همبستگی رتبه‌ای پروفایل غذا بین دو سلف")
    ax[1].set_ylabel("تعداد جفت سلف")
    ax[1].set_title("آیا «غذای پرنرخ» در همه‌ی سلف‌ها یکی است؟")
    figs.annotate(ax[1], f"{len(cors):,} جفت سلف\nمیانه {np.median(cors):+.2f}\n"
                         f"سهم جفت‌های با همبستگی >۰.۳: {np.mean(cors > 0.3):.0%}", loc="upper left")
    figs.save(fig, "r8d_02_dish_within_venue", out)
    print(md_table(out))
    print(f"\nجفت سلف مقایسه‌شده: {len(cors):,} · میانه‌ی همبستگی {np.median(cors):+.4f} · "
          f"سهم > ۰.۳: {np.mean(cors > 0.3):.3f}")
    return out


# ---------------------------------------------------------------------------
# ۸-۷-۳ تنوع منو
# ---------------------------------------------------------------------------

def p873_variety(f: pd.DataFrame) -> pd.DataFrame:
    header("۸-۷-۳ تنوع منو — آنتروپی روزانه و نرخ", 2)
    d = f.copy()
    # آنتروپی منوی هر (روز، وعده، سلف) بر پایه‌ی سهم رزرو هر غذا
    def ent(x):
        p = x.Res / x.Res.sum()
        return float(-(p * np.log(p + 1e-12)).sum())
    g = (d.groupby(["date_gregorian", "meal", "restaurant"], observed=True)
         .apply(lambda x: pd.Series(dict(n_dish=len(x), entropy=ent(x), res=x.Res.sum(),
                                         rate=_rate(x))), include_groups=False).reset_index())
    g = g[g.res >= 30]
    g["n_dish_c"] = g.n_dish.clip(upper=6)
    by_n = (g.groupby("n_dish_c", observed=True)
            .apply(lambda x: pd.Series(dict(rate=float((x.rate * x.res).sum() / x.res.sum()),
                                            n=len(x), res=x.res.sum())), include_groups=False).reset_index())
    g["ent_q"] = pd.qcut(g.entropy, 8, labels=False, duplicates="drop")
    by_e = (g.groupby("ent_q", observed=True)
            .apply(lambda x: pd.Series(dict(entropy=x.entropy.mean(),
                                            rate=float((x.rate * x.res).sum() / x.res.sum()),
                                            n=len(x))), include_groups=False).reset_index())

    fig, ax = figs.new(1, 2, w=6.6, h=4.2)
    ax[0].bar(by_n.n_dish_c, by_n.rate, color=figs.C_MAIN)
    ax[0].set_xlabel("تعداد غذای منو در آن وعده (≥۶ یکی شده)")
    ax[0].set_ylabel("نرخ وزنی"); ax[0].set_title("منوی متنوع‌تر ⇐ نرخ بالاتر؟")
    sp = stats.spearmanr(g.n_dish, g.rate)
    figs.annotate(ax[0], f"همبستگی رتبه‌ای تعداد↔نرخ {sp.statistic:+.3f}\n(p={sp.pvalue:.1e})")
    ax[1].plot(by_e.entropy, by_e.rate, "o-", lw=2, color=figs.C_ALT)
    ax[1].set_xlabel("آنتروپی منو (هشتک)"); ax[1].set_ylabel("نرخ وزنی")
    ax[1].set_title("تنوع پیوسته‌ی منو")
    se = stats.spearmanr(g.entropy, g.rate)
    figs.annotate(ax[1], f"همبستگی رتبه‌ای آنتروپی↔نرخ {se.statistic:+.3f}")
    figs.save(fig, "r8d_03_menu_variety", pd.concat([by_n.assign(kind="n_dish"),
                                                     by_e.assign(kind="entropy")], ignore_index=True))
    print(md_table(by_n)); print(); print(md_table(by_e))
    return by_n


# ---------------------------------------------------------------------------
# ۸-۷-۴ تکرار غذا
# ---------------------------------------------------------------------------

def p874_repetition(f: pd.DataFrame) -> pd.DataFrame:
    header("۸-۷-۴ تکرار غذا — فاصله از آخرین سرو همان غذا در همان سلف", 2)
    d = f[["date_gregorian", "meal", "restaurant", "dish", "Res", "NoRecv", "rho"]].copy()
    d = d.sort_values(["restaurant", "dish", "date_gregorian"])
    d["gap"] = (d.groupby(["restaurant", "dish"], observed=True).date_gregorian
                .diff().dt.days)
    dd = d[d.gap.notna()].copy()
    dd["gap_b"] = pd.cut(dd.gap, [0, 1, 2, 3, 6, 9, 13, 20, 30, 10**5],
                         labels=["۱", "۲", "۳", "۴–۶", "۷–۹", "۱۰–۱۳", "۱۴–۲۰", "۲۱–۳۰", ">۳۰"])
    g = (dd.groupby("gap_b", observed=True)
         .apply(lambda x: pd.Series(dict(n=len(x), res=x.Res.sum(), rate=_rate(x))),
                include_groups=False).reset_index())
    base = _rate(f)

    fig, ax = figs.new(1, 2, w=6.6, h=4.2)
    x = np.arange(len(g))
    ax[0].bar(x, g.rate, color=figs.C_MAIN)
    ax[0].axhline(base, color=figs.C_ALT, lw=1.3, ls="--")
    ax[0].set_xticks(x); ax[0].set_xticklabels(g.gap_b, rotation=45, ha="right")
    ax[0].set_xlabel("فاصله از سرو قبلی همان غذا در همان سلف (روز)")
    ax[0].set_ylabel("نرخ وزنی"); ax[0].set_title("آیا تکرار زودهنگام نرخ را بالا می‌برد؟")
    figs.annotate(ax[0], f"نرخ کل {base:.2%}\nکمینه {g.rate.min():.2%} · بیشینه {g.rate.max():.2%}")

    # کنترل تقویم: همان تحلیل درون هر روز هفته و وعده
    dd["dow"] = dd.date_gregorian.dt.dayofweek
    dev = (dd.assign(cell_mean=dd.groupby(["restaurant", "dish"], observed=True).rho.transform("mean"))
           .assign(resid=lambda x: x.rho - x.cell_mean))
    gr = (dev.groupby("gap_b", observed=True)
          .apply(lambda x: pd.Series(dict(resid_mean=float(np.average(x.resid, weights=x.Res)),
                                          n=len(x))), include_groups=False).reset_index())
    ax[1].bar(np.arange(len(gr)), gr.resid_mean, color=figs.C_OK)
    ax[1].axhline(0, color="black", lw=1)
    ax[1].set_xticks(np.arange(len(gr))); ax[1].set_xticklabels(gr.gap_b, rotation=45, ha="right")
    ax[1].set_xlabel("فاصله از سرو قبلی (روز)")
    ax[1].set_ylabel("انحراف نرخ از میانگین همان سلف×غذا")
    ax[1].set_title("پس از حذف اثر ثابت سلف×غذا")
    figs.annotate(ax[1], "اگر اینجا صاف شود،\nاثر اولی مصنوعِ ترکیب بود", loc="upper left", fontsize=8)
    figs.save(fig, "r8d_04_dish_repetition", g.merge(gr[["gap_b", "resid_mean"]], on="gap_b"))
    print(md_table(g)); print(); print(md_table(gr))
    return g


# ---------------------------------------------------------------------------
# ۸-۷-۵ و ۸-۷-۶ برهم‌کنش‌ها
# ---------------------------------------------------------------------------

def p876_interactions(f: pd.DataFrame) -> pd.DataFrame:
    header("۸-۷-۵ و ۸-۷-۶ برهم‌کنش غذا×وعده و غذا×فصل", 2)
    d = f.copy()
    top = d.groupby("dish", observed=True).Res.sum().nlargest(14).index
    sub = d[d.dish.isin(top)].copy()
    sub["season"] = pd.cut(sub.month_jalali, [0, 3, 6, 9, 12],
                           labels=["بهار", "تابستان", "پاییز", "زمستان"])

    def piv(frame, col):
        num = frame.pivot_table(index="dish", columns=col, values="NoRecv", aggfunc="sum")
        den = frame.pivot_table(index="dish", columns=col, values="Res", aggfunc="sum")
        return num / den.where(den >= 2000)

    m1 = piv(sub, "meal").rename(columns=MEAL_FA)
    m2 = piv(sub, "season")
    order = m1.mean(axis=1).sort_values().index
    m1, m2 = m1.loc[order], m2.reindex(order)

    fig, ax = figs.new(1, 2, w=6.0, h=5.4)
    for a, m, t in [(ax[0], m1, "غذا × وعده"), (ax[1], m2, "غذا × فصل")]:
        im = a.imshow(m.values, aspect="auto", cmap="RdYlBu_r", vmin=0.03, vmax=0.16)
        a.set_xticks(range(m.shape[1])); a.set_xticklabels(m.columns, rotation=45, ha="right", fontsize=9)
        a.set_yticks(range(m.shape[0])); a.set_yticklabels(m.index, fontsize=7)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                v = m.values[i, j]
                if np.isfinite(v):
                    a.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=7)
        a.grid(False); a.set_title(t)
    fig.colorbar(im, ax=ax, shrink=0.6, label="نرخ وزنی")
    out = pd.concat([m1.stack(future_stack=True).rename("rate").reset_index().assign(axis="meal"),
                     m2.stack(future_stack=True).rename("rate").reset_index().assign(axis="season")],
                    ignore_index=True)
    figs.save(fig, "r8d_05_dish_interactions", out)
    # آیا برهم‌کنش واقعی است؟ واریانس اثر متقابل در برابر اثرهای اصلی
    for m, nm in [(m1, "وعده"), (m2, "فصل")]:
        mm = m.dropna(how="any")
        if mm.shape[1] >= 2 and len(mm) >= 4:
            grand = mm.values.mean()
            row_e = mm.mean(axis=1).values[:, None] - grand
            col_e = mm.mean(axis=0).values[None, :] - grand
            inter = mm.values - grand - row_e - col_e
            print(f"{nm}: واریانس اثر غذا {np.var(row_e):.2e} · اثر {nm} {np.var(col_e):.2e} · "
                  f"برهم‌کنش {np.var(inter):.2e} · سهم برهم‌کنش "
                  f"{np.var(inter)/(np.var(row_e)+np.var(col_e)+np.var(inter)):.1%}")
    return out


def main() -> None:
    set_global_seed()
    figs.setup()
    f = load_features()
    header(f"دور ۸ · اطلس منو — {f.dish.nunique()} غذا، {f.food_group.nunique()} گروه")
    p871_dish_forest(f)
    p872_within_venue(f)
    p873_variety(f)
    p874_repetition(f)
    p876_interactions(f)


if __name__ == "__main__":
    main()
