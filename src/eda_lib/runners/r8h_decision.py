"""بند ۸-۱۱ — کاوش تصمیم‌محور: پول کجاست؟ (دور ۸).

همه‌ی کاوش‌های قبلی روی **نرخ** بودند. تصمیم پروژه روی **پرس** است، و ۶۳٬۹۳۱
سلول سهم بسیار نابرابری از هدررفت دارند.

اجرا: python -m src.eda_lib.runners.r8h_decision
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import set_global_seed
from src.eda_lib import figs
from src.eda_lib.runners._common_v4 import DOW_FA, header, load_features, md_table

MEAL_FA = {"lunch": "ناهار", "dinner": "شام"}


def _rate(g):
    return float(g.NoRecv.sum() / g.Res.sum()) if g.Res.sum() else np.nan


def _gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return np.nan
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


# ---------------------------------------------------------------------------
# ۸-۱۱-۱ لورنتس هدررفت
# ---------------------------------------------------------------------------

def p8111_lorenz(f: pd.DataFrame) -> pd.DataFrame:
    header("۸-۱۱-۱ تمرکز هدررفت — منحنی لورنتس روی سلول‌ها", 2)
    rows = []
    fig, ax = figs.new(1, 2, w=6.6, h=4.4)
    units = [("سلول (روز×وعده×سلف×غذا)", f.NoRecv.to_numpy()),
             ("سلف×وعده", f.groupby(["restaurant", "meal"], observed=True).NoRecv.sum().to_numpy()),
             ("روز×وعده", f.groupby(["date_gregorian", "meal"], observed=True).NoRecv.sum().to_numpy()),
             ("سلف", f.groupby("restaurant", observed=True).NoRecv.sum().to_numpy())]
    for i, (name, w) in enumerate(units):
        w = np.sort(w)[::-1]
        cum = np.cumsum(w) / w.sum()
        x = np.arange(1, len(w) + 1) / len(w)
        ax[0].plot(x, cum, lw=2.2, color=figs.PALETTE[i], label=name)
        rows.append(dict(unit=name, n=len(w), gini=_gini(w),
                         top10_share=float(cum[max(int(0.10 * len(w)) - 1, 0)]),
                         top20_share=float(cum[max(int(0.20 * len(w)) - 1, 0)]),
                         top50_share=float(cum[max(int(0.50 * len(w)) - 1, 0)])))
    ax[0].plot([0, 1], [0, 1], color="black", lw=1, ls=":")
    ax[0].set_xlabel("سهم واحدها (از پرهدررفت به کم‌هدررفت)")
    ax[0].set_ylabel("سهم تجمعی پرس هدررفته")
    ax[0].legend(loc="lower right", fontsize=8)
    ax[0].set_title("تمرکز هدررفت در هر سطح تجمیع")
    out = pd.DataFrame(rows)
    figs.annotate(ax[0], f"۱۰٪ سلول پرهدررفت = {out.top10_share.iloc[0]:.0%} کل هدررفت",
                  loc="upper left", fontsize=9)

    # منحنی پوشش-ارزش: اگر فقط k٪ سلول‌ها مدل بگیرند
    w = np.sort(f.NoRecv.to_numpy())[::-1]
    cum = np.cumsum(w) / w.sum()
    ks = np.array([0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.75, 1.0])
    vals = [cum[max(int(k * len(w)) - 1, 0)] for k in ks]
    ax[1].plot(ks * 100, np.array(vals) * 100, "o-", lw=2.4, color=figs.C_ALT)
    ax[1].plot([0, 100], [0, 100], color="black", lw=1, ls=":")
    ax[1].set_xlabel("درصد سلول‌هایی که مدل می‌گیرند (پرهدررفت‌ترین‌ها)")
    ax[1].set_ylabel("درصد هدررفت پوشش‌داده‌شده")
    ax[1].set_title("منحنی پوشش–ارزش: بودجه‌ی مدل‌سازی کجا خرج شود؟")
    k20 = vals[4]
    figs.annotate(ax[1], f"۲۰٪ سلول‌ها ⇒ {k20:.0%} هدررفت\n"
                         f"۵۰٪ سلول‌ها ⇒ {vals[7]:.0%} هدررفت", loc="lower right")
    figs.save(fig, "r8h_01_waste_lorenz", out)
    print(md_table(out))
    print(pd.DataFrame(dict(coverage=ks, value=vals)).to_string(index=False))
    return out


# ---------------------------------------------------------------------------
# ۸-۱۱-۲ نقشه‌ی پول
# ---------------------------------------------------------------------------

def p8112_money_map(f: pd.DataFrame) -> pd.DataFrame:
    header("۸-۱۱-۲ نقشه‌ی پول — پرس هدررفته به تفکیک محور تصمیم", 2)
    d = f.copy()
    d["dow_fa"] = d.day_of_week.map(DOW_FA)
    frames = []
    for col, fa in [("venue_type", "نوع سلف"), ("meal", "وعده"), ("dow_fa", "روز هفته"),
                    ("food_group", "گروه غذا"), ("academic_year", "سال تحصیلی")]:
        g = (d.groupby(col, observed=True)
             .apply(lambda x: pd.Series(dict(waste=x.NoRecv.sum(), res=x.Res.sum(),
                                             rate=_rate(x), cells=len(x))),
                    include_groups=False).reset_index().rename(columns={col: "value"}))
        g["share_waste"] = g.waste / d.NoRecv.sum()
        g["share_res"] = g.res / d.Res.sum()
        g["excess"] = g.share_waste - g.share_res
        g.insert(0, "axis", fa)
        frames.append(g)
    out = pd.concat(frames, ignore_index=True)

    fig, ax = figs.new(1, 2, w=7.0, h=4.8)
    sub = out[out.axis.isin(["نوع سلف", "گروه غذا"])].sort_values("excess")
    y = np.arange(len(sub))
    colors = [figs.C_ALT if e > 0 else figs.C_MAIN for e in sub.excess]
    ax[0].barh(y, sub.excess * 100, color=colors, height=0.72)
    ax[0].set_yticks(y); ax[0].set_yticklabels([f"{a} · {v}" for a, v in zip(sub.axis, sub.value)],
                                               fontsize=7)
    ax[0].axvline(0, color="black", lw=1)
    ax[0].set_xlabel("سهم از هدررفت منهای سهم از حجم (واحد درصد)")
    ax[0].set_title("کجا بیش از سهمش هدر می‌دهد؟")
    figs.annotate(ax[0], "مثبت = هدررفت متمرکز، منفی = بهتر از متوسط",
                  loc="lower right", fontsize=8)

    num = d.pivot_table(index="dow_fa", columns="meal", values="NoRecv", aggfunc="sum")
    num = num.reindex([DOW_FA[i] for i in range(7) if DOW_FA[i] in num.index])
    num = num.rename(columns=MEAL_FA)
    im = ax[1].imshow(num.values / d.NoRecv.sum() * 100, aspect="auto", cmap="YlOrRd")
    ax[1].set_xticks(range(num.shape[1])); ax[1].set_xticklabels(num.columns)
    ax[1].set_yticks(range(num.shape[0])); ax[1].set_yticklabels(num.index, fontsize=9)
    for i in range(num.shape[0]):
        for j in range(num.shape[1]):
            v = num.values[i, j] / d.NoRecv.sum() * 100
            if np.isfinite(v):
                ax[1].text(j, i, f"{v:.1f}٪", ha="center", va="center", fontsize=9)
    ax[1].grid(False); ax[1].set_title("سهم هر خانه‌ی روز×وعده از کل هدررفت")
    fig.colorbar(im, ax=ax[1], shrink=0.7, label="٪ از کل پرس هدررفته")
    figs.save(fig, "r8h_02_money_map", out)
    print(md_table(out))
    print(f"\nکل پرس هدررفته در پنجره: {int(d.NoRecv.sum()):,} از {int(d.Res.sum()):,} رزرو")
    return out


# ---------------------------------------------------------------------------
# ۸-۱۱-۳ نامتقارنی ریسک
# ---------------------------------------------------------------------------

def p8113_asymmetry(f: pd.DataFrame) -> pd.DataFrame:
    header("۸-۱۱-۳ نامتقارنی ریسک — دورریز در برابر کمبود (لنگر: F164)", 2)
    d = f.copy()
    rows = []
    for tau in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
        # سیاست: بپز Res × (1 − q_τ) که q_τ چندک τ‌ام نرخ تاریخی همان سلف×وعده است
        q = (d.groupby(["restaurant", "meal"], observed=True).rho
             .transform(lambda s: s.quantile(tau)))
        cook = d.Res * (1 - q)
        served = d.Res * (1 - d.rho)
        over = np.maximum(cook - served, 0)     # پخت اضافه ⇒ دورریز
        under = np.maximum(served - cook, 0)    # پخت کم ⇒ کمبود
        rows.append(dict(tau=tau, waste_portions=float(over.sum()),
                         shortage_portions=float(under.sum()),
                         shortage_rate=float(under.sum() / served.sum()),
                         waste_rate=float(over.sum() / served.sum()),
                         cells_short=float((under > 0).mean())))
    out = pd.DataFrame(rows)

    # وضعیت واقعی امروز: پخت = تعداد رزرو (سیاست فعلی ضمنی)
    served = d.Res * (1 - d.rho)
    today_waste = float((d.Res - served).sum())

    fig, ax = figs.new(1, 2, w=6.6, h=4.3)
    ax[0].plot(out.tau, out.waste_portions / 1e3, "o-", lw=2.2, color=figs.C_ALT, label="دورریز")
    ax[0].plot(out.tau, out.shortage_portions / 1e3, "s-", lw=2.2, color=figs.C_MAIN, label="کمبود")
    ax[0].axhline(today_waste / 1e3, color=figs.C_GREY, lw=1.5, ls="--")
    ax[0].set_xlabel("τ سیاست (چندک نرخ تاریخی سلف×وعده)")
    ax[0].set_ylabel("هزار پرس در کل پنجره")
    ax[0].legend(loc="center right"); ax[0].set_title("مبادله‌ی دورریز و کمبود")
    figs.annotate(ax[0], f"خط‌چین: سیاست فعلی (پخت = رزرو)\n{today_waste/1e3:.0f} هزار پرس دورریز",
                  loc="upper right", fontsize=8)

    ax[1].plot(out.waste_portions / 1e3, out.shortage_portions / 1e3, "o-", lw=2.2,
               color=figs.C_OK)
    for _, r in out.iterrows():
        ax[1].annotate(f"τ={r.tau:g}", (r.waste_portions / 1e3, r.shortage_portions / 1e3),
                       fontsize=8, xytext=(5, 4), textcoords="offset points")
    ax[1].set_xlabel("دورریز (هزار پرس)"); ax[1].set_ylabel("کمبود (هزار پرس)")
    ax[1].set_title("مرز کارا — انتخاب τ یک تصمیم هزینه‌ای است")
    figs.save(fig, "r8h_03_risk_asymmetry", out)
    print(md_table(out, float_fmt="{:.4f}"))
    print(f"\nپخت = رزرو (سیاست ضمنی فعلی): دورریز {today_waste:,.0f} پرس · کمبود صفر")
    return out


# ---------------------------------------------------------------------------
# ۸-۱۱-۴ آیا τ بهینه در همه‌ی بخش‌ها یکی است؟
# ---------------------------------------------------------------------------

def p8114_tau_by_segment(f: pd.DataFrame) -> pd.DataFrame:
    header("۸-۱۱-۴ τ بهینه به تفکیک بخش — آیا یک عدد برای همه درست است؟", 2)
    d = f.copy()
    taus = np.arange(0.02, 0.51, 0.02)
    rows = []
    for name, grp in [("کل", d)] + [(f"نوع سلف: {k}", g) for k, g in d.groupby("venue_type", observed=True)] \
                     + [(f"وعده: {MEAL_FA[k]}", g) for k, g in d.groupby("meal", observed=True)]:
        if grp.Res.sum() < 50000:
            continue
        served = grp.Res * (1 - grp.rho)
        for tau in taus:
            q = grp.groupby(["restaurant", "meal"], observed=True).rho.transform(
                lambda s: s.quantile(tau))
            cook = grp.Res * (1 - q)
            over = np.maximum(cook - served, 0).sum()
            under = np.maximum(served - cook, 0).sum()
            # هزینه با نسبت Co/Cu = ۱ به ۴ (کمبود چهار برابر دورریز گران‌تر)
            for ratio, lbl in [(1.0, "۱:۱"), (4.0, "۱:۴")]:
                rows.append(dict(segment=name, tau=tau, cost=float(over + ratio * under),
                                 ratio=lbl, served=float(served.sum())))
    out = pd.DataFrame(rows)
    best = (out.loc[out.groupby(["segment", "ratio"]).cost.idxmin()]
            .sort_values(["ratio", "tau"]).reset_index(drop=True))

    fig, ax = figs.new(1, 2, w=6.8, h=4.4)
    for i, (ratio, a) in enumerate(zip(["۱:۱", "۱:۴"], ax)):
        sub = out[out.ratio == ratio]
        for j, seg in enumerate(sub.segment.unique()):
            s = sub[sub.segment == seg]
            a.plot(s.tau, s.cost / s.served.iloc[0], lw=2 if seg == "کل" else 1.4,
                   color="black" if seg == "کل" else figs.PALETTE[j % len(figs.PALETTE)],
                   label=seg, alpha=1.0 if seg == "کل" else 0.85)
        a.set_xlabel("τ"); a.set_ylabel("هزینه به‌ازای هر پرس سرو‌شده")
        a.set_title(f"نسبت هزینه‌ی دورریز به کمبود {ratio}")
        if i == 0:
            a.legend(fontsize=7, loc="upper right")
        bb = best[best.ratio == ratio]
        figs.annotate(a, "τ بهینه: " + " · ".join(f"{r.tau:.2f}" for _, r in bb.iterrows()),
                      loc="lower left", fontsize=7)
    figs.save(fig, "r8h_04_tau_by_segment", best)
    print(md_table(best[["segment", "ratio", "tau", "cost"]], float_fmt="{:.3f}"))
    return best


def main() -> None:
    set_global_seed()
    figs.setup()
    f = load_features()
    header(f"دور ۸ · اطلس تصمیم — {int(f.NoRecv.sum()):,} پرس هدررفته")
    p8111_lorenz(f)
    p8112_money_map(f)
    p8113_asymmetry(f)
    p8114_tau_by_segment(f)


if __name__ == "__main__":
    main()
