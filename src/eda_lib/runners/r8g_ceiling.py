"""بند ۸-۱۰ — سقف پیش‌بینی‌پذیری (دور ۸).

پروژه ۱۶۹ پیکربندی مدل اجرا کرده و هیچ‌وقت نپرسیده «بهترین ممکن چقدر است؟».
بدون آن عدد، «مدل خوب است» و «مسئله آسان است» قابل‌تفکیک نیستند.

اجرا: python -m src.eda_lib.runners.r8g_ceiling
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import set_global_seed
from src.eda_lib import figs
from src.eda_lib.runners._common_v4 import header, load_features, md_table


def _rate(g):
    return float(g.NoRecv.sum() / g.Res.sum()) if g.Res.sum() else np.nan


# ---------------------------------------------------------------------------
# ۸-۱۰-۱ واریانس تقلیل‌ناپذیر
# ---------------------------------------------------------------------------

def p8101_irreducible(f: pd.DataFrame) -> pd.DataFrame:
    header("۸-۱۰-۱ چقدر از واریانس نرخ فقط نوفه‌ی نمونه‌گیری است؟", 2)
    rows = []
    for thr in [1, 5, 10, 20, 30, 50, 100, 200]:
        d = f[f.Res >= thr]
        if len(d) < 200:
            continue
        p = d.rho
        var_obs = float(p.var())
        # نوفه‌ی دوجمله‌ای مورد انتظار در هر سلول: p(1-p)/Res، میانگین روی سلول‌ها
        var_noise = float((p * (1 - p) / d.Res).mean())
        rows.append(dict(min_res=thr, n_cells=len(d), var_observed=var_obs,
                         var_binomial_noise=var_noise,
                         signal_share=max(0.0, 1 - var_noise / var_obs),
                         share_of_volume=float(d.Res.sum() / f.Res.sum())))
    out = pd.DataFrame(rows)

    fig, ax = figs.new(1, 2, w=6.6, h=4.2)
    ax[0].plot(out.min_res, out.signal_share, "o-", lw=2.2, color=figs.C_MAIN)
    ax[0].set_xscale("log"); ax[0].set_ylim(0, 1.02)
    ax[0].set_xlabel("آستانه‌ی حداقل رزرو سلول")
    ax[0].set_ylabel("سهم واریانس که سیگنال است")
    ax[0].set_title("سقف بالای هر مدل روی شبکه‌ی سلول")
    figs.annotate(ax[0], f"روی کل شبکه {out.signal_share.iloc[0]:.0%} سیگنال است\n"
                         f"روی سلول‌های ≥۱۰۰ رزرو {out[out.min_res==100].signal_share.iloc[0]:.0%}",
                  loc="lower right")
    ax[1].plot(out.min_res, out.var_observed, "o-", lw=2, color=figs.C_MAIN, label="واریانس مشاهده‌شده")
    ax[1].plot(out.min_res, out.var_binomial_noise, "s--", lw=2, color=figs.C_ALT, label="نوفه‌ی دوجمله‌ای")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("آستانه‌ی حداقل رزرو"); ax[1].set_ylabel("واریانس نرخ")
    ax[1].legend(loc="upper right"); ax[1].set_title("سیگنال و نوفه، هر دو با اندازه کوچک می‌شوند")
    figs.save(fig, "r8g_01_irreducible_noise", out)
    print(md_table(out))
    return out


# ---------------------------------------------------------------------------
# ۸-۱۰-۲ و ۸-۱۰-۳ سقف اوراکل و سهم هر لایه
# ---------------------------------------------------------------------------

def p8102_oracle(f: pd.DataFrame) -> pd.DataFrame:
    header("۸-۱۰-۲ / ۸-۱۰-۳ سقف اوراکل و سهم هر لایه‌ی اطلاعاتی", 2)
    d = f.copy()
    y = d.rho.to_numpy()
    tot = float(np.var(y))

    def r2(cols):
        g = d.groupby(list(cols), observed=True).rho.transform("mean").to_numpy()
        return float(1 - np.var(y - g) / tot)

    layers = [
        ("میانگین کل (خط پایه)", [], 1),
        ("وعده", ["meal"], 2),
        ("سلف", ["restaurant"], d.restaurant.nunique()),
        ("سلف × وعده", ["restaurant", "meal"], d.groupby(["restaurant", "meal"], observed=True).ngroups),
        ("سلف × وعده × روزهفته", ["restaurant", "meal", "day_of_week"],
         d.groupby(["restaurant", "meal", "day_of_week"], observed=True).ngroups),
        ("سلول (سلف×وعده×غذا)", ["restaurant", "meal", "dish"],
         d.groupby(["restaurant", "meal", "dish"], observed=True).ngroups),
        ("⚠️ اوراکل روز (غیرقابل‌اجرا)", ["date_gregorian", "meal"],
         d.groupby(["date_gregorian", "meal"], observed=True).ngroups),
        ("⚠️ اوراکل روز × سلف", ["date_gregorian", "meal", "restaurant"],
         d.groupby(["date_gregorian", "meal", "restaurant"], observed=True).ngroups),
    ]
    rows = [dict(layer=n, r2=r2(c) if c else 0.0, n_params=k) for n, c, k in layers]
    out = pd.DataFrame(rows)

    # حساسیت سهم «اثر روز» به آستانه‌ی اندازه‌ی سلول — چرا F59/F199 عددهای مختلف دادند
    sens = []
    for thr in [1, 10, 30, 100, 200]:
        dd = f[f.Res >= thr]
        yy = dd.rho.to_numpy(); tt = float(np.var(yy))
        g = dd.groupby(["date_gregorian", "meal"], observed=True).rho.transform("mean").to_numpy()
        sens.append(dict(min_res=thr, n_cells=len(dd), day_r2=float(1 - np.var(yy - g) / tt)))
    sens = pd.DataFrame(sens)

    fig, ax = figs.new(1, 2, w=6.8, h=4.6)
    o = out.sort_values("r2")
    colors = [figs.C_WARN if "اوراکل" in n else figs.C_MAIN for n in o.layer]
    figs.rtl_barh(ax[0], o.layer.tolist(), o.r2.tolist(), fmt="{:.3f}", pad=0.006)
    for bar, c in zip(ax[0].patches, colors):
        bar.set_color(c)
    ax[0].tick_params(axis="y", labelsize=8)
    ax[0].set_xlabel("سهم واریانس نرخ سلول")
    ax[0].set_title("نردبان اطلاعات — نارنجی = دانش غیرقابل‌دسترس")
    prac = o[~o.layer.str.contains("اوراکل")].r2.max()
    orac = o[o.layer.str.contains("اوراکل")].r2.max()
    figs.annotate(ax[0], f"سقف ایستای در دسترس {prac:.3f}\nسقف اوراکل {orac:.3f}\n"
                         f"فاصله {orac - prac:.3f}", loc="lower right", fontsize=8)

    ax[1].plot(sens.min_res, sens.day_r2, "o-", lw=2.2, color=figs.C_ALT)
    ax[1].set_xscale("log"); ax[1].set_ylim(0, 0.9)
    ax[1].axhline(0.832, color=figs.C_GREY, lw=1.4, ls="--")
    ax[1].text(1.1, 0.845, "عدد ثبت‌شده‌ی F59 روی v2: ۰.۸۳۲", fontsize=8, color="#555555")
    ax[1].set_xlabel("آستانه‌ی حداقل رزرو سلول")
    ax[1].set_ylabel("سهم «اثر روز×وعده» از واریانس")
    ax[1].set_title("چرا این عدد در هر گزارش فرق داشت")
    figs.annotate(ax[1], f"روی کل شبکه {sens.day_r2.iloc[0]:.3f}\n"
                         f"روی ≥۲۰۰ رزرو {sens.day_r2.iloc[-1]:.3f}\n"
                         f"⇒ عدد بدون آستانه بی‌معناست", loc="upper left", fontsize=8)
    figs.save(fig, "r8g_02_oracle_ladder", pd.concat([out, sens], ignore_index=True))
    print(md_table(out)); print(); print(md_table(sens))
    return out


# ---------------------------------------------------------------------------
# ۸-۱۰-۴ منحنی یادگیری روی طول پنجره
# ---------------------------------------------------------------------------

def p8104_learning(f: pd.DataFrame) -> pd.DataFrame:
    """⚠️ این بند عمداً روی **سه پنجره‌ی آزمون و دو seed** اجرا می‌شود.

    نسخه‌ی تک‌پنجره‌ای یک منحنی کاملاً صاف داد و نتیجه‌اش «داده‌ی بیشتر هیچ کمکی
    نمی‌کند» خوانده می‌شد. با سه پنجره معلوم شد بخشی از آن صافی، نوسان یک پنجره
    بوده: بهبود ۶۰→۲۵۰ روز واقعی است (~۶٪) و بعد از ۲۵۰ روز اشباع می‌شود.
    """
    header("۸-۱۰-۴ آیا ۵۹۳ روز کافی است؟ منحنی یادگیری روی طول پنجره", 2)
    import lightgbm as lgb

    d = f.sort_values("date_gregorian").copy()
    feat = ["log_res", "day_of_week", "meal_seq", "venue_hist_rho", "venue_dow_hist_rho",
            "cell_hist_rho", "dish_hist_rho", "global_hist_rho", "venue_last_rho",
            "day_shock_lag1", "semester_progress", "is_exam_period", "is_holiday_any",
            "is_ramadan", "days_since_semester_start", "composition_mean",
            "composition_high_risk_share", "venue_hist_res_mean", "res_vs_venue_hist"]
    feat = [c for c in feat if c in d.columns]
    days = np.sort(d.date_gregorian.unique())

    def pinball(y, p, w, tau=0.10):
        e = y - p
        return float(np.average(np.maximum(tau * e, (tau - 1) * e), weights=w))

    rows = []
    for fold, end in enumerate([len(days) - 180, len(days) - 90, len(days)]):
        test_days = days[end - 90:end]
        te = d[d.date_gregorian.isin(test_days)]
        Xte, yte, wte = te[feat].astype(float), te.rho, te.Res
        q = float(np.quantile(d[d.date_gregorian.isin(days[:end - 90])].rho, 0.10))
        rows.append(dict(fold=fold, train_days=0, seed=0, pinball_t010=pinball(yte, q, wte),
                         mae=float(np.average(np.abs(yte - q), weights=wte)), kind="لنگر ثابت"))
        for n_train in [60, 120, 180, 250, 330, 420]:
            tr_days = days[:end - 90][-n_train:]
            if len(tr_days) < n_train:
                continue
            tr = d[d.date_gregorian.isin(tr_days)]
            for seed in (42, 7):
                m = lgb.LGBMRegressor(objective="quantile", alpha=0.10, n_estimators=300,
                                      learning_rate=0.06, num_leaves=31, min_child_samples=40,
                                      verbose=-1, random_state=seed)
                m.fit(tr[feat].astype(float), tr.rho, sample_weight=tr.Res)
                pr = np.clip(m.predict(Xte), 0, 1)
                rows.append(dict(fold=fold, train_days=n_train, seed=seed,
                                 pinball_t010=pinball(yte, pr, wte),
                                 mae=float(np.average(np.abs(yte - pr), weights=wte)), kind="مدل"))
    out = pd.DataFrame(rows)
    mdl = out[out.kind == "مدل"]
    agg = mdl.groupby("train_days").pinball_t010.agg(["mean", "min", "max", "count"]).reset_index()
    anchor = float(out[out.kind == "لنگر ثابت"].pinball_t010.mean())

    fig, ax = figs.new(1, 2, w=6.6, h=4.3)
    ax[0].fill_between(agg.train_days, agg["min"], agg["max"], color=figs.C_MAIN, alpha=0.18)
    ax[0].plot(agg.train_days, agg["mean"], "o-", lw=2.4, color=figs.C_MAIN)
    ax[0].axhline(anchor, color=figs.C_ALT, lw=1.5, ls="--")
    ax[0].set_xlabel("تعداد روز آموزش")
    ax[0].set_ylabel("زیان پین‌بال τ=۰.۱۰ (وزن رزرو)")
    ax[0].set_title("میانگین سه پنجره‌ی آزمون × دو seed")
    a60 = float(agg.loc[agg.train_days == 60, "mean"].iloc[0])
    a250 = float(agg.loc[agg.train_days == 250, "mean"].iloc[0])
    amax = float(agg["mean"].iloc[-1])
    figs.annotate(ax[0], f"خط‌چین: کوانتایل ثابت ({anchor:.4f})\n"
                         f"۶۰→۲۵۰ روز: {(a60-a250)/a60:+.1%}\n"
                         f"۲۵۰→{int(agg.train_days.iloc[-1])} روز: {(a250-amax)/a250:+.1%}")

    for fo, c in zip(sorted(mdl.fold.unique()), figs.PALETTE):
        s = mdl[mdl.fold == fo].groupby("train_days").pinball_t010.mean()
        ax[1].plot(s.index, s.values / s.iloc[0], "o-", lw=2, color=c, label=f"پنجره‌ی آزمون {fo+1}")
    ax[1].axhline(1, color="black", lw=1, ls=":")
    ax[1].set_xlabel("تعداد روز آموزش"); ax[1].set_ylabel("زیان نسبت به آموزش ۶۰ روزه")
    ax[1].legend(loc="upper right", fontsize=8)
    ax[1].set_title("هر پنجره جدا — صافی مصنوع یک پنجره نیست")
    figs.save(fig, "r8g_03_learning_curve", out)
    print(md_table(agg, float_fmt="{:.6f}"))
    print(f"\nلنگر کوانتایل ثابت (میانگین سه پنجره): {anchor:.6f}")
    print(md_table(mdl.pivot_table(index="train_days", columns="fold",
                                   values="pinball_t010").reset_index(), float_fmt="{:.6f}"))
    return out


def main() -> None:
    set_global_seed()
    figs.setup()
    f = load_features()
    header(f"دور ۸ · سقف پیش‌بینی‌پذیری — {len(f):,} سلول")
    p8101_irreducible(f)
    p8102_oracle(f)
    p8104_learning(f)


if __name__ == "__main__":
    main()
