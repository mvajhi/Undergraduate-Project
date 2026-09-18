"""بند ۸-۹ — ساختار وابستگی ۱۰۱ ستون ماتریس ویژگی v4 (دور ۸).

ماتریس ویژگی v4 ساخته شد ولی هرگز **به‌عنوان یک ماتریس** کاوش نشد.

اجرا: python -m src.eda_lib.runners.r8f_dependency
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster import hierarchy

from src.config import set_global_seed
from src.eda_lib import figs
from src.eda_lib.runners._common_v4 import header, load_features, md_table

TARGET = "rho"
# ستون‌هایی که **خروجی** روز هدف‌اند و هرگز فیچر نیستند — مبنای ممیزی نشت
OUTCOME_COLS = {"rho", "NoRecv", "Recv", "n_reservations", "n_people", "portions_per_person"}
ID_COLS = {"date_gregorian", "meal", "restaurant", "dish", "academic_year", "campus", "city",
           "venue_type", "gender", "food_group"}


def _numeric_frame(f: pd.DataFrame) -> pd.DataFrame:
    d = f.copy()
    for c in d.columns:
        if d[c].dtype == bool:
            d[c] = d[c].astype(np.int8)
    num = d.select_dtypes(include=[np.number])
    return num


# ---------------------------------------------------------------------------
# ۸-۹-۱ و ۸-۹-۳ ارتباط هر ستون با هدف، و ستون‌های مرده
# ---------------------------------------------------------------------------

def p891_association(f: pd.DataFrame) -> pd.DataFrame:
    header("۸-۹-۱ / ۸-۹-۳ ارتباط هر ستون با هدف، و ستون‌های مرده", 2)
    from sklearn.feature_selection import mutual_info_regression

    num = _numeric_frame(f)
    y = num[TARGET].to_numpy()
    feats = [c for c in num.columns if c not in OUTCOME_COLS]
    rows = []
    for c in feats:
        v = num[c]
        nun = int(v.nunique(dropna=True))
        nan_share = float(v.isna().mean())
        rows.append(dict(column=c, n_unique=nun, nan_share=nan_share,
                         pearson=float(v.corr(pd.Series(y), method="pearson")) if nun > 1 else np.nan,
                         spearman=float(v.corr(pd.Series(y), method="spearman")) if nun > 1 else np.nan))
    out = pd.DataFrame(rows)

    # اطلاعات متقابل روی نمونه‌ی ۱۵ هزارتایی — kNN روی ۶۴ هزار ردیف × ۹۰ ستون گران است
    rng = np.random.default_rng(42)
    idx = rng.choice(len(num), size=min(15000, len(num)), replace=False)
    live = out[(out.n_unique > 1)].column.tolist()
    X = num.loc[num.index[idx], live].fillna(num[live].median())
    mi = mutual_info_regression(X, y[idx], random_state=42)
    out = out.merge(pd.DataFrame(dict(column=live, mutual_info=mi)), on="column", how="left")
    out["abs_spearman"] = out.spearman.abs()
    out = out.sort_values("mutual_info", ascending=False).reset_index(drop=True)

    dead = out[(out.n_unique <= 1) | (out.nan_share > 0.5)]
    fig, ax = figs.new(1, 2, w=7.0, h=5.6)
    top = out.head(24).iloc[::-1]
    yy = np.arange(len(top))
    ax[0].barh(yy, top.mutual_info, color=figs.C_MAIN, height=0.74)
    ax[0].set_yticks(yy); ax[0].set_yticklabels(top.column, fontsize=7)
    ax[0].set_xlabel("اطلاعات متقابل با هدف"); ax[0].set_title("۲۴ ستون پراطلاع")
    figs.annotate(ax[0], f"ستون زنده {int((out.n_unique>1).sum())} از {len(out)}\n"
                         f"ستون مرده {len(dead)}", loc="lower right", fontsize=8)

    ax[1].scatter(out.abs_spearman, out.mutual_info, s=26, color=figs.C_ALT, alpha=0.8)
    for _, r in out.head(8).iterrows():
        ax[1].annotate(r.column, (r.abs_spearman, r.mutual_info), fontsize=6,
                       xytext=(3, 3), textcoords="offset points")
    ax[1].set_xlabel("|همبستگی اسپیرمن| با هدف"); ax[1].set_ylabel("اطلاعات متقابل")
    ax[1].set_title("رابطه‌ی یکنواخت در برابر رابطه‌ی هر شکل")
    both = out[["abs_spearman", "mutual_info"]].dropna()
    r_rank = stats.spearmanr(both.abs_spearman, both.mutual_info).statistic
    nonmono = out[(out.abs_spearman < 0.05) & (out.mutual_info > out.mutual_info.quantile(0.75))]
    figs.annotate(ax[1], f"همبستگی رتبه‌ای دو معیار {r_rank:+.2f}\n"
                         f"{len(nonmono)} ستون پراطلاع ولی غیریکنواخت", loc="upper left", fontsize=8)
    figs.save(fig, "r8f_01_feature_association", out)
    print(md_table(out.head(25)))
    print(f"\nستون مرده (ثابت یا >۵۰٪ تهی): {len(dead)} → {dead.column.tolist()}")
    if len(nonmono):
        print(f"ستون‌های پراطلاع با همبستگی خطی ناچیز: {nonmono.column.tolist()}")
    return out


# ---------------------------------------------------------------------------
# ۸-۹-۲ هم‌خطی و خوشه‌ی ستونی
# ---------------------------------------------------------------------------

def p892_collinearity(f: pd.DataFrame, assoc: pd.DataFrame) -> pd.DataFrame:
    header("۸-۹-۲ هم‌خطی — خوشه‌های ستونی و VIF", 2)
    num = _numeric_frame(f)
    live = assoc[(assoc.n_unique > 1) & (assoc.nan_share <= 0.5)].column.tolist()
    X = num[live].fillna(num[live].median())
    corr = X.corr(method="spearman").fillna(0)
    dist = 1 - corr.abs()
    Z = hierarchy.linkage(dist.to_numpy()[np.triu_indices_from(dist, k=1)], method="average")
    clusters = hierarchy.fcluster(Z, t=0.3, criterion="distance")
    cl = pd.DataFrame(dict(column=live, cluster=clusters))
    cl = cl.merge(assoc[["column", "mutual_info"]], on="column", how="left")
    reps = (cl.sort_values("mutual_info", ascending=False)
            .groupby("cluster", as_index=False).first().sort_values("cluster"))

    fig, ax = figs.new(1, 2, w=7.2, h=6.0)
    order = hierarchy.leaves_list(Z)
    cm = corr.to_numpy()[np.ix_(order, order)]
    im = ax[0].imshow(cm, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax[0].set_xticks([]); ax[0].set_yticks(range(0, len(order), 3))
    ax[0].set_yticklabels([live[i] for i in order[::3]], fontsize=5.5)
    ax[0].grid(False); ax[0].set_title("ماتریس همبستگی مرتب‌شده با خوشه‌بندی")
    fig.colorbar(im, ax=ax[0], shrink=0.6)

    sizes = cl.groupby("cluster").size().sort_values(ascending=False)
    big = sizes[sizes >= 2]
    labels = [f"{reps.set_index('cluster').column.get(c, '?')[:28]} (+{s-1})" for c, s in big.items()]
    figs.rtl_barh(ax[1], labels, big.tolist(), color=figs.C_OK, fmt="{:.0f}", pad=0.2)
    ax[1].tick_params(axis="y", labelsize=7)
    ax[1].set_xlabel("تعداد ستون در خوشه")
    ax[1].set_title("خوشه‌های هم‌خط (فاصله‌ی ۰.۳ ≡ |ρ|≥۰.۷)")
    figs.annotate(ax[1], f"{len(live)} ستون زنده → {cl.cluster.nunique()} خوشه\n"
                         f"بزرگ‌ترین خوشه {sizes.max()} ستون", loc="lower right", fontsize=8)
    figs.save(fig, "r8f_02_collinearity", cl.sort_values(["cluster", "mutual_info"],
                                                        ascending=[True, False]))
    print(md_table(reps.head(30)))
    print(f"\nستون زنده {len(live)} → {cl.cluster.nunique()} خوشه مستقل "
          f"(کاهش بودجه‌ی ستون {1 - cl.cluster.nunique()/len(live):.0%})")
    return cl


# ---------------------------------------------------------------------------
# ۸-۹-۴ ممیزی نشت دیداری
# ---------------------------------------------------------------------------

def p894_leakage(f: pd.DataFrame, assoc: pd.DataFrame) -> pd.DataFrame:
    header("۸-۹-۴ ممیزی نشت — هر ستون با |ρ|>۰.۵ دستی بررسی می‌شود", 2)
    sus = assoc[assoc.abs_spearman > 0.5].copy().sort_values("abs_spearman", ascending=False)
    num = _numeric_frame(f)
    fig, ax = figs.new(1, 2, w=6.6, h=4.3)
    ax[0].hist(assoc.abs_spearman.dropna(), bins=40, color=figs.C_MAIN)
    ax[0].axvline(0.5, color=figs.C_ALT, lw=1.6, ls="--")
    ax[0].axvline(0.7, color=figs.C_WARN, lw=1.6, ls=":")
    ax[0].set_xlabel("|همبستگی اسپیرمن با هدف|"); ax[0].set_ylabel("تعداد ستون")
    ax[0].set_title("توزیع ارتباط ستون‌ها با هدف")
    figs.annotate(ax[0], f"ستون با |ρ|>۰.۵: {len(sus)}\n"
                         f"ستون با |ρ|>۰.۷: {int((assoc.abs_spearman>0.7).sum())}\n"
                         f"بیشینه {assoc.abs_spearman.max():.3f}")
    strongest = assoc.nlargest(1, "abs_spearman").iloc[0]
    ax[1].scatter(num[strongest.column], num[TARGET], s=3, alpha=0.06,
                  color=figs.C_GREY, rasterized=True)
    ax[1].set_xlabel(strongest.column); ax[1].set_ylabel("نرخ عدم‌دریافت")
    ax[1].set_title(f"قوی‌ترین ستون: ρ={strongest.spearman:+.3f}")
    figs.annotate(ax[1], "نشت وقتی است که رابطه **مکانیکی** باشد،\nنه وقتی قوی باشد",
                  loc="upper left", fontsize=8)
    figs.save(fig, "r8f_03_leakage_audit", sus)
    print(md_table(sus[["column", "spearman", "pearson", "mutual_info"]]))
    print("\nهیچ ستونی خروجی روز هدف نیست — فهرست خروجی‌ها از ماتریس کنار گذاشته شده: "
          + ", ".join(sorted(OUTCOME_COLS)))
    return sus


# ---------------------------------------------------------------------------
# ۸-۹-۵ پایداری رتبه‌ی اهمیت بین سه سال
# ---------------------------------------------------------------------------

def p895_stability(f: pd.DataFrame, assoc: pd.DataFrame) -> pd.DataFrame:
    header("۸-۹-۵ پایداری رتبه‌ی اهمیت ستون‌ها بین سه سال", 2)
    from sklearn.feature_selection import mutual_info_regression
    num = _numeric_frame(f)
    live = assoc[(assoc.n_unique > 1) & (assoc.nan_share <= 0.5)].column.tolist()
    rng = np.random.default_rng(7)
    res = {}
    # ⚠️ ۲٬۰۵۴ سلول (۳.۲٪) روی ۳۶ روز سرویس هیچ سال تحصیلی ندارند — خارج از پوشش
    # تقویم آموزشی. هر تحلیل «به تفکیک سال» بی‌صدا آن‌ها را می‌اندازد.
    n_na = int(f.academic_year.isna().sum())
    print(f"  ⚠️ سلول بدون سال تحصیلی: {n_na:,} ({n_na/len(f):.1%}) روی "
          f"{f[f.academic_year.isna()].date_gregorian.nunique()} روز سرویس — از این تحلیل کنار گذاشته شدند")
    for y in sorted(x for x in f.academic_year.unique() if isinstance(x, str)):
        m = (f.academic_year == y).to_numpy()
        sub = num.loc[m, live + [TARGET]]
        if len(sub) < 3000:
            continue
        idx = rng.choice(len(sub), size=min(12000, len(sub)), replace=False)
        Xs = sub.iloc[idx][live].fillna(sub[live].median())
        res[y] = pd.Series(mutual_info_regression(Xs, sub.iloc[idx][TARGET], random_state=7),
                           index=live)
    piv = pd.DataFrame(res)
    ranks = piv.rank(ascending=False)
    years = list(piv.columns)
    pair = []
    for i in range(len(years)):
        for j in range(i + 1, len(years)):
            pair.append(dict(year_a=years[i], year_b=years[j],
                             spearman=float(stats.spearmanr(ranks[years[i]], ranks[years[j]]).statistic),
                             top10_overlap=len(set(ranks.nsmallest(10, years[i]).index)
                                               & set(ranks.nsmallest(10, years[j]).index)) / 10))
    out = pd.DataFrame(pair)

    fig, ax = figs.new(1, 2, w=6.6, h=4.6)
    top = piv.loc[piv.mean(axis=1).nlargest(16).index]
    for i, y in enumerate(years):
        ax[0].plot(top[y].values, np.arange(len(top)), "o", ms=7, color=figs.PALETTE[i], label=y, alpha=0.85)
    ax[0].set_yticks(np.arange(len(top))); ax[0].set_yticklabels(top.index, fontsize=7)
    ax[0].invert_yaxis(); ax[0].set_xlabel("اطلاعات متقابل"); ax[0].legend(fontsize=8, loc="lower right")
    ax[0].set_title("۱۶ ستون برتر — سه سال روی هم")

    x = np.arange(len(out))
    ax[1].bar(x - 0.2, out.spearman, width=0.4, color=figs.C_MAIN, label="همبستگی رتبه‌ای کل")
    ax[1].bar(x + 0.2, out.top10_overlap, width=0.4, color=figs.C_OK, label="اشتراک ۱۰ برتر")
    ax[1].set_xticks(x); ax[1].set_xticklabels([f"{a[-4:]}↔{b[-4:]}" for a, b in zip(out.year_a, out.year_b)])
    ax[1].set_ylim(0, 1.05); ax[1].legend(fontsize=8, loc="lower right")
    ax[1].set_title("پایداری اهمیت بین سال‌ها")
    figs.annotate(ax[1], "پایداری پایین ⇒ انتخاب فیچر\nروی یک سال تعمیم نمی‌یابد",
                  loc="upper left", fontsize=8)
    figs.save(fig, "r8f_04_importance_stability", piv.reset_index(names="column"))
    print(md_table(out))
    print(md_table(top.reset_index(names="column")))
    return out


def main() -> None:
    set_global_seed()
    figs.setup()
    f = load_features()
    header(f"دور ۸ · نقشه‌ی وابستگی — {f.shape[1]} ستون، {len(f):,} ردیف")
    assoc = p891_association(f)
    p892_collinearity(f, assoc)
    p894_leakage(f, assoc)
    p895_stability(f, assoc)


if __name__ == "__main__":
    main()
