"""بند ۸-۴ (اطلس هدف) و ۸-۳-۳ (رانش سه‌ساله) — دور ۸.

اجرا: python -m src.eda_lib.runners.r8a_target_atlas
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.config import set_global_seed
from src.eda_lib import figs
from src.eda_lib.runners._common_v4 import academic_year_order, header, load_features, md_table

MEAL_FA = {"lunch": "ناهار", "dinner": "شام"}


def _levels(f: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """سه سطح تجمیع از همان شبکه‌ی L1 — با نرخ **وزنی**، نه میانگین نرخ‌ها."""
    l1 = f[["date_gregorian", "meal", "restaurant", "dish", "Res", "NoRecv"]].copy()
    l2 = (l1.groupby(["date_gregorian", "meal", "restaurant"], observed=True)[["Res", "NoRecv"]]
          .sum().reset_index())
    l3 = (l1.groupby(["date_gregorian", "meal"], observed=True)[["Res", "NoRecv"]]
          .sum().reset_index())
    for d in (l1, l2, l3):
        d["rho"] = d.NoRecv / d.Res
    return {"L1 سلول": l1, "L2 سلف×وعده": l2, "L3 روز×وعده": l3}


# ---------------------------------------------------------------------------
# ۸-۴-۱ شکل توزیع در سه سطح
# ---------------------------------------------------------------------------

def p841_shape(levels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    header("۸-۴-۱ شکل توزیع هدف در سه سطح تجمیع", 2)
    fig, axes = figs.new(2, 3, w=5.0, h=3.6)
    rows = []
    for j, (name, d) in enumerate(levels.items()):
        r = d.rho.to_numpy()
        ax = axes[0, j]
        ax.hist(r, bins=60, color=figs.C_MAIN, alpha=0.85)
        ax.axvline(np.median(r), color=figs.C_ALT, ls="--", lw=1.5)
        ax.set_title(name)
        ax.set_xlabel("نرخ عدم‌دریافت")
        ax.set_ylabel("تعداد رکورد" if j == 0 else "")
        ax.set_xlim(0, min(1.0, np.quantile(r, 0.999) * 1.1))
        figs.annotate(ax, f"n={len(r):,}\nمیانه={np.median(r):.3f}\nصفر={np.mean(r == 0):.1%}")

        # QQ در برابر نرمال: انحراف از خط، شکست مدل گاوسی را نشان می‌دهد
        ax2 = axes[1, j]
        q = np.linspace(0.005, 0.995, 200)
        emp = np.quantile(r, q)
        norm_q = stats.norm.ppf(q, loc=r.mean(), scale=r.std())
        # بتا با گشتاور منطبق، روی مقادیر اکیداً داخل (0,1)
        rr = np.clip(r, 1e-4, 1 - 1e-4)
        m, v = rr.mean(), rr.var()
        k = max(m * (1 - m) / v - 1, 1e-6)
        beta_q = stats.beta.ppf(q, m * k, (1 - m) * k)
        ax2.plot(emp, norm_q, lw=1.8, color=figs.C_ALT, label="نرمال")
        ax2.plot(emp, beta_q, lw=1.8, color=figs.C_OK, label="بتا")
        lim = [0, float(np.quantile(r, 0.995))]
        ax2.plot(lim, lim, color="black", lw=1, ls=":")
        ax2.set_xlim(lim); ax2.set_ylim(lim)
        ax2.set_xlabel("چندک تجربی")
        ax2.set_ylabel("چندک نظری" if j == 0 else "")
        if j == 0:
            ax2.legend(loc="upper left", fontsize=9)
        mad_n = float(np.mean(np.abs(norm_q - emp)))
        mad_b = float(np.mean(np.abs(beta_q - emp)))
        figs.annotate(ax2, f"خطای نرمال {mad_n:.4f}\nخطای بتا {mad_b:.4f}", loc="lower right")
        rows.append(dict(level=name, n=len(r), mean=r.mean(), median=float(np.median(r)),
                         std=r.std(), zero_share=float(np.mean(r == 0)),
                         skew=float(stats.skew(r)), kurt=float(stats.kurtosis(r)),
                         mad_normal=mad_n, mad_beta=mad_b))
    fig.suptitle("توزیع هدف در سه سطح — بالا هیستوگرام، پایین QQ در برابر نرمال و بتا", y=1.01)
    out = pd.DataFrame(rows)
    figs.save(fig, "r8a_01_target_shape_by_level", out)
    print(md_table(out))
    return out


# ---------------------------------------------------------------------------
# ۸-۴-۲ جرم صفر: ساختاری یا تصادفی؟
# ---------------------------------------------------------------------------

def p842_zeros(l1: pd.DataFrame) -> pd.DataFrame:
    header("۸-۴-۲ جرم صفر — ساختاری یا نمونه‌گیری؟", 2)
    d = l1.copy()
    d["res_bin"] = pd.cut(d.Res, [0, 5, 10, 25, 50, 100, 250, 500, 10**6],
                          labels=["۱–۵", "۶–۱۰", "۱۱–۲۵", "۲۶–۵۰", "۵۱–۱۰۰", "۱۰۱–۲۵۰", "۲۵۱–۵۰۰", ">۵۰۰"])
    p_glob = d.NoRecv.sum() / d.Res.sum()
    g = d.groupby("res_bin", observed=True).apply(
        lambda x: pd.Series(dict(
            n=len(x),
            obs_zero=float((x.rho == 0).mean()),
            # اگر صفرها فقط نوفه‌ی دوجمله‌ای بودند: P(0) = (1-p)^Res با p نرخ وزنی گروه
            exp_zero=float(np.mean((1 - x.NoRecv.sum() / x.Res.sum()) ** x.Res)),
            rate=float(x.NoRecv.sum() / x.Res.sum()),
        )), include_groups=False).reset_index()
    g["excess"] = g.obs_zero - g.exp_zero

    fig, ax = figs.new(1, 2, w=6.4, h=4.2)
    x = np.arange(len(g))
    ax[0].bar(x - 0.2, g.obs_zero, width=0.4, color=figs.C_MAIN, label="مشاهده‌شده")
    ax[0].bar(x + 0.2, g.exp_zero, width=0.4, color=figs.C_GREY, label="انتظار دوجمله‌ای")
    ax[0].set_xticks(x); ax[0].set_xticklabels(g.res_bin, rotation=45, ha="right")
    ax[0].set_xlabel("تعداد رزرو سلول"); ax[0].set_ylabel("سهم سلول‌های با نرخ صفر")
    ax[0].legend(loc="upper right")
    ax[0].set_title("صفرها کجا بیش از انتظارند؟")

    ax[1].plot(x, g.excess, "o-", color=figs.C_ALT, lw=2)
    ax[1].axhline(0, color="black", lw=1)
    ax[1].set_xticks(x); ax[1].set_xticklabels(g.res_bin, rotation=45, ha="right")
    ax[1].set_xlabel("تعداد رزرو سلول"); ax[1].set_ylabel("صفر مازاد (مشاهده − انتظار)")
    ax[1].set_title("صفر مازاد — نشانه‌ی جرم ساختاری")
    big = g[g.n > 100]
    figs.annotate(ax[1], f"نرخ وزنی کل {p_glob:.2%}\nبیشینه‌ی مازاد {big.excess.max():.1%}")
    figs.save(fig, "r8a_02_zero_mass", g)
    print(md_table(g))
    return g


# ---------------------------------------------------------------------------
# ۸-۴-۳ و ۸-۴-۴ رابطه با Res و ساختار واریانس
# ---------------------------------------------------------------------------

def p843_res(l1: pd.DataFrame) -> pd.DataFrame:
    header("۸-۴-۳ و ۸-۴-۴ رابطه‌ی نرخ و واریانس با اندازه‌ی سلول", 2)
    d = l1.copy()
    d["q"] = pd.qcut(d.Res, 20, labels=False, duplicates="drop")
    g = d.groupby("q", observed=True).apply(lambda x: pd.Series(dict(
        res_median=float(x.Res.median()),
        n=len(x),
        rate_w=float(x.NoRecv.sum() / x.Res.sum()),
        rho_mean=float(x.rho.mean()),
        rho_var=float(x.rho.var()),
    )), include_groups=False).reset_index()
    # واریانس مورد انتظار دوجمله‌ای در هر گروه: p(1-p)/Res
    g["var_binom"] = g.rate_w * (1 - g.rate_w) / g.res_median
    g["phi"] = g.rho_var / g.var_binom  # ضریب بیش‌پراکندگی

    fig, ax = figs.new(1, 2, w=6.4, h=4.2)
    ax[0].plot(g.res_median, g.rate_w, "o-", color=figs.C_MAIN, lw=2, label="نرخ وزنی")
    ax[0].plot(g.res_median, g.rho_mean, "s--", color=figs.C_ALT, lw=1.6, label="میانگین ساده‌ی نرخ")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("اندازه‌ی سلول (رزرو، مقیاس لگاریتمی)")
    ax[0].set_ylabel("نرخ عدم‌دریافت")
    ax[0].legend(loc="upper right")
    ax[0].set_title("نرخ در برابر اندازه — دو تعریف، دو شکل")
    figs.annotate(ax[0], f"دامنه‌ی نرخ وزنی\n{g.rate_w.min():.1%} تا {g.rate_w.max():.1%}", loc="lower left")

    ax[1].plot(g.res_median, g.rho_var, "o-", color=figs.C_MAIN, lw=2, label="واریانس مشاهده‌شده")
    ax[1].plot(g.res_median, g.var_binom, "--", color=figs.C_GREY, lw=2, label="واریانس دوجمله‌ای")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("اندازه‌ی سلول (رزرو)")
    ax[1].set_ylabel("واریانس نرخ")
    ax[1].legend(loc="upper right")
    ax[1].set_title("قیف ناهم‌واریانسی و بیش‌پراکندگی")
    figs.annotate(ax[1], f"φ میانه {g.phi.median():.1f}×\nφ در سلول‌های بزرگ {g.phi.iloc[-1]:.1f}×", loc="lower left")
    figs.save(fig, "r8a_03_rate_variance_vs_res", g)
    print(md_table(g))
    return g


# ---------------------------------------------------------------------------
# ۸-۴-۵ تبدیل هدف
# ---------------------------------------------------------------------------

def p845_transform(l1: pd.DataFrame) -> pd.DataFrame:
    header("۸-۴-۵ کدام تبدیل هدف؟", 2)
    d = l1[l1.Res >= 20].copy()
    eps = 0.5 / d.Res
    t = {
        "خام": d.rho.to_numpy(),
        "logit": np.log((d.rho + eps) / (1 - d.rho + eps)).to_numpy(),
        "arcsine": np.arcsin(np.sqrt(d.rho)).to_numpy(),
        "ریشه‌ی دوم": np.sqrt(d.rho).to_numpy(),
    }
    rows = []
    fig, axes = figs.new(1, 4, w=3.7, h=3.5)
    for ax, (name, v) in zip(axes, t.items()):
        v = v[np.isfinite(v)]
        ax.hist(v, bins=60, color=figs.C_MAIN, alpha=0.85)
        ax.set_title(name); ax.set_xlabel("مقدار")
        sk = float(stats.skew(v))
        # پایداری واریانس: پراکندگی واریانسِ درون‌گروهی بین دهک‌های اندازه
        tmp = pd.DataFrame({"v": v, "Res": d.Res.to_numpy()[np.isfinite(t[name])]})
        tmp["q"] = pd.qcut(tmp.Res, 10, labels=False, duplicates="drop")
        vv = tmp.groupby("q", observed=True).v.var()
        ratio = float(vv.max() / vv.min()) if vv.min() > 0 else np.inf
        figs.annotate(ax, f"چولگی {sk:+.2f}\nنسبت واریانس {ratio:.1f}×")
        rows.append(dict(transform=name, skew=sk, kurtosis=float(stats.kurtosis(v)),
                         var_ratio_decile=ratio))
    fig.suptitle("چهار تبدیل هدف روی سلول‌های ≥۲۰ رزرو — چولگی و پایداری واریانس", y=1.02)
    out = pd.DataFrame(rows)
    figs.save(fig, "r8a_04_target_transform", out)
    print(md_table(out))
    return out


# ---------------------------------------------------------------------------
# ۸-۴-۶ دُم بالا
# ---------------------------------------------------------------------------

def p846_tail(f: pd.DataFrame) -> pd.DataFrame:
    header("۸-۴-۶ دُم بالا — چه کسانی در ۱۰٪ بالای نرخ‌اند؟", 2)
    d = f.copy()
    thr = d.rho.quantile(0.90)
    d["is_tail"] = d.rho >= thr
    rows = []
    for col, fa in [("meal", "وعده"), ("venue_type", "نوع سلف"), ("academic_year", "سال تحصیلی"),
                    ("food_group", "گروه غذا")]:
        base = d.groupby(col, observed=True).apply(
            lambda x: pd.Series(dict(share_cells=len(x) / len(d),
                                     share_tail=x.is_tail.sum() / d.is_tail.sum(),
                                     share_waste=x.NoRecv.sum() / d.NoRecv.sum(),
                                     rate=x.NoRecv.sum() / x.Res.sum())), include_groups=False)
        base["lift"] = base.share_tail / base.share_cells
        base = base.reset_index().rename(columns={col: "value"})
        base.insert(0, "axis", fa)
        rows.append(base)
    out = pd.concat(rows, ignore_index=True)

    tail_cells = d[d.is_tail]
    fig, ax = figs.new(1, 2, w=6.6, h=4.6)
    sub = out[out.axis == "نوع سلف"].sort_values("lift", ascending=False)
    figs.rtl_barh(ax[0], sub.value.tolist(), sub.lift.tolist(), color=figs.C_ALT, fmt="{:.2f}×", pad=0.03)
    ax[0].axvline(1, color="black", lw=1, ls="--")
    ax[0].set_xlabel("نسبت حضور در دُم بالا (۱ = سهم طبیعی)")
    ax[0].set_title("چه نوع سلفی در دُم بالا بیش‌نمایی دارد؟")

    ax[1].scatter(d.Res, d.rho, s=3, alpha=0.06, color=figs.C_GREY, rasterized=True)
    ax[1].scatter(tail_cells.Res, tail_cells.rho, s=4, alpha=0.18, color=figs.C_ALT, rasterized=True)
    ax[1].set_xscale("log")
    ax[1].axhline(thr, color="black", lw=1.2, ls="--")
    ax[1].set_xlabel("اندازه‌ی سلول (رزرو)"); ax[1].set_ylabel("نرخ عدم‌دریافت")
    ax[1].set_title("دُم بالا در فضای اندازه×نرخ")
    share_waste = tail_cells.NoRecv.sum() / d.NoRecv.sum()
    share_res = tail_cells.Res.sum() / d.Res.sum()
    figs.annotate(ax[1], f"آستانه‌ی دهک ۱۰ = {thr:.1%}\nسهم دُم از هدررفت {share_waste:.1%}\n"
                         f"سهم دُم از حجم {share_res:.1%}")
    figs.save(fig, "r8a_05_upper_tail", out)
    print(md_table(out))
    print(f"\nآستانه‌ی دهک ۱۰: {thr:.4f} · سهم دُم از کل هدررفت: {share_waste:.2%} · "
          f"سهم دُم از کل حجم: {share_res:.2%}")
    return out


# ---------------------------------------------------------------------------
# ۸-۳-۳ رانش سه‌ساله — سه روش مستقل
# ---------------------------------------------------------------------------

def p833_drift(f: pd.DataFrame) -> pd.DataFrame:
    header("۸-۳-۳ رانش سه‌ساله‌ی هدف — سه روش مستقل", 2)
    d = f.copy()

    # روش ۱ — نرخ وزنی سال‌به‌سال (لنگر: اعداد بند ۸-۰)
    m1 = d.groupby("academic_year", observed=True).apply(
        lambda x: pd.Series(dict(days=x.date_gregorian.nunique(), cells=len(x),
                                 res=x.Res.sum(), rate=x.NoRecv.sum() / x.Res.sum())),
        include_groups=False).reindex(academic_year_order()).reset_index()

    # روش ۲ — روند روی سری روزانه‌ی وزنی (رگرسیون با خطای بلوکی ماهانه)
    daily = (d.groupby("date_gregorian", observed=True)[["Res", "NoRecv"]].sum().reset_index())
    daily["rate"] = daily.NoRecv / daily.Res
    daily["t"] = (daily.date_gregorian - daily.date_gregorian.min()).dt.days
    sl, ic, r_, p_, se = stats.linregress(daily.t, daily.rate)
    slope_per_year = sl * 365

    # روش ۳ — درون‌سلفی: فقط سلف‌هایی که در هر سه سال حضور دارند (کنترل ترکیب)
    present = d.groupby("restaurant", observed=True).academic_year.nunique()
    stable = present[present == 3].index
    ds = d[d.restaurant.isin(stable)]
    m3 = ds.groupby("academic_year", observed=True).apply(
        lambda x: pd.Series(dict(rate=x.NoRecv.sum() / x.Res.sum(), res=x.Res.sum())),
        include_groups=False).reindex(academic_year_order()).reset_index()

    # تجزیه‌ی رانش: تغییر ترکیب در برابر تغییر رفتار (شاپلی دو‌جزئی ساده)
    first, last = academic_year_order()[0], academic_year_order()[-1]
    w = d.groupby(["academic_year", "restaurant"], observed=True).apply(
        lambda x: pd.Series(dict(res=x.Res.sum(), rate=x.NoRecv.sum() / x.Res.sum())),
        include_groups=False).reset_index()
    a = w[w.academic_year == first].set_index("restaurant")
    b = w[w.academic_year == last].set_index("restaurant")
    common = a.index.intersection(b.index)
    wa = a.loc[common, "res"] / a.loc[common, "res"].sum()
    wb = b.loc[common, "res"] / b.loc[common, "res"].sum()
    ra, rb = a.loc[common, "rate"], b.loc[common, "rate"]
    total = float((wb * rb).sum() - (wa * ra).sum())
    behaviour = float((wa * (rb - ra)).sum())
    mix = float(((wb - wa) * ra).sum())
    inter = total - behaviour - mix

    fig, ax = figs.new(1, 2, w=6.6, h=4.3)
    monthly = d.copy()
    monthly["ym"] = monthly.date_gregorian.dt.to_period("M").dt.to_timestamp()
    mm = monthly.groupby("ym", observed=True).apply(
        lambda x: pd.Series(dict(rate=x.NoRecv.sum() / x.Res.sum(), res=x.Res.sum())),
        include_groups=False).reset_index()
    mm = mm[mm.res > 20000]
    ax[0].plot(mm.ym, mm.rate, "o-", color=figs.C_MAIN, lw=1.6, ms=4)
    tt = (mm.ym - daily.date_gregorian.min()).dt.days
    ax[0].plot(mm.ym, ic + sl * tt, color=figs.C_ALT, lw=2, ls="--")
    for y, row in m1.set_index("academic_year").iterrows():
        pass
    ax[0].set_ylabel("نرخ وزنی ماهانه"); ax[0].set_xlabel("زمان")
    ax[0].set_title("سری ماهانه‌ی نرخ و خط روند")
    ax[0].tick_params(axis="x", rotation=30)
    figs.annotate(ax[0], f"شیب {slope_per_year:+.3%} در سال\np = {p_:.2g}", loc="upper left")

    x = np.arange(3)
    ax[1].bar(x - 0.2, m1.rate, width=0.4, color=figs.C_MAIN, label="همه‌ی سلف‌ها")
    ax[1].bar(x + 0.2, m3.rate, width=0.4, color=figs.C_OK, label=f"فقط {len(stable)} سلف پایدار")
    ax[1].set_xticks(x); ax[1].set_xticklabels(academic_year_order())
    ax[1].set_ylabel("نرخ وزنی"); ax[1].set_title("رانش با و بدون کنترل ترکیب موجودیت")
    ax[1].legend(loc="upper left")
    ax[1].set_ylim(0.060, max(m1.rate.max(), m3.rate.max()) * 1.12)
    figs.annotate(ax[1], f"تجزیه‌ی {first}→{last}\nکل {total:+.3%}\n"
                         f"رفتار {behaviour:+.3%}\nترکیب {mix:+.3%}", loc="lower left")
    out = pd.DataFrame([
        dict(method="۱ نرخ وزنی سال‌به‌سال", value=float(m1.rate.iloc[-1] - m1.rate.iloc[0]), unit="Δ نرخ ۳ سال"),
        dict(method="۲ روند سری روزانه", value=float(slope_per_year), unit="Δ نرخ در سال"),
        dict(method="۳ درون‌سلفی (سلف پایدار)", value=float(m3.rate.iloc[-1] - m3.rate.iloc[0]), unit="Δ نرخ ۳ سال"),
        dict(method="تجزیه — سهم رفتار", value=behaviour, unit="Δ نرخ"),
        dict(method="تجزیه — سهم ترکیب", value=mix, unit="Δ نرخ"),
        dict(method="تجزیه — برهم‌کنش", value=inter, unit="Δ نرخ"),
    ])
    figs.save(fig, "r8a_06_target_drift", out)
    print(md_table(m1)); print(); print(md_table(m3)); print()
    print(md_table(out, float_fmt="{:.5f}"))
    print(f"\nسلف‌های حاضر در هر سه سال: {len(stable)} از {d.restaurant.nunique()}")
    return out


def main() -> None:
    set_global_seed()
    figs.setup()
    f = load_features()
    header(f"دور ۸ · اطلس هدف — {len(f):,} سلول، {f.date_gregorian.nunique()} روز")
    levels = _levels(f)
    p841_shape(levels)
    p842_zeros(levels["L1 سلول"])
    p843_res(levels["L1 سلول"])
    p845_transform(levels["L1 سلول"])
    p846_tail(f)
    p833_drift(f)


if __name__ == "__main__":
    main()
