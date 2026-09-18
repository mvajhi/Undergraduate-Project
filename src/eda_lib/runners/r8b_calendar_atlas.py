"""بند ۸-۵ — اطلس تقویم و فصل (دور ۸).

اجرا: python -m src.eda_lib.runners.r8b_calendar_atlas
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.config import set_global_seed
from src.eda_lib import figs
from src.eda_lib.runners._common_v4 import DOW_FA, academic_year_order, header, load_features, md_table

MONTH_FA = ["فروردین", "اردیبهشت", "خرداد", "تیر", "مرداد", "شهریور",
            "مهر", "آبان", "آذر", "دی", "بهمن", "اسفند"]
MEAL_FA = {"lunch": "ناهار", "dinner": "شام"}


def _rate(g: pd.DataFrame) -> float:
    return float(g.NoRecv.sum() / g.Res.sum()) if g.Res.sum() else np.nan


def _pivot_rate(df, index, columns):
    num = df.pivot_table(index=index, columns=columns, values="NoRecv", aggfunc="sum")
    den = df.pivot_table(index=index, columns=columns, values="Res", aggfunc="sum")
    return num / den.where(den >= 500)


def _heat(ax, mat, *, vmin, vmax, cmap="RdYlBu_r", fmt="{:.1%}", fs=8):
    im = ax.imshow(mat.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(mat.shape[1])); ax.set_xticklabels(mat.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(mat.shape[0])); ax.set_yticklabels(mat.index, fontsize=9)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=fs,
                        color="black" if abs(v - (vmin + vmax) / 2) < (vmax - vmin) * 0.3 else "white")
    ax.grid(False)
    return im


# ---------------------------------------------------------------------------
# ۸-۵-۱ نقشه‌ی گرمایی روز هفته × ماه شمسی، سه سال
# ---------------------------------------------------------------------------

def p851_dow_month(f: pd.DataFrame) -> pd.DataFrame:
    header("۸-۵-۱ نقشه‌ی روز هفته × ماه شمسی در سه سال", 2)
    d = f.copy()
    d["dow_fa"] = d.day_of_week.map(DOW_FA)
    d["month_fa"] = d.month_jalali.map(lambda m: MONTH_FA[m - 1])
    years = academic_year_order()
    mats = {}
    for y in years:
        m = _pivot_rate(d[d.academic_year == y], "dow_fa", "month_fa")
        m = m.reindex(index=[DOW_FA[i] for i in range(7) if DOW_FA[i] in m.index],
                      columns=[c for c in MONTH_FA if c in m.columns])
        mats[y] = m
    vmin = min(float(np.nanmin(m.values)) for m in mats.values())
    vmax = max(float(np.nanmax(m.values)) for m in mats.values())
    fig, axes = figs.new(3, 1, w=9.0, h=3.2)
    for ax, y in zip(axes, years):
        im = _heat(ax, mats[y], vmin=vmin, vmax=vmax, fs=7)
        ax.set_title(f"سال تحصیلی {y}")
    fig.colorbar(im, ax=axes, shrink=0.6, label="نرخ عدم‌دریافت")
    fig.suptitle("نرخ به تفکیک روز هفته و ماه شمسی — خانه‌های کم‌حجم (<۵۰۰ رزرو) خالی‌اند")
    out = pd.concat({y: m.stack(future_stack=True) for y, m in mats.items()}).reset_index()
    out.columns = ["academic_year", "dow", "month", "rate"]
    figs.save(fig, "r8b_01_dow_month_heatmap", out)
    # پایداری الگو بین سال‌ها
    flat = out.pivot_table(index=["dow", "month"], columns="academic_year", values="rate")
    cors = flat.corr(method="spearman")
    print(md_table(cors.reset_index()))
    return out


# ---------------------------------------------------------------------------
# ۸-۵-۲ پروفایل هفته‌ی نیمسال
# ---------------------------------------------------------------------------

def p852_week_of_semester(f: pd.DataFrame) -> pd.DataFrame:
    header("۸-۵-۲ پروفایل هفته‌ی نیمسال × نیمسال × سال", 2)
    d = f[f.week_of_semester.notna() & f.semester.notna()].copy()
    d["week_of_semester"] = d.week_of_semester.astype(int)
    d = d[d.week_of_semester.between(0, 20)]
    g = (d.groupby(["academic_year", "semester", "week_of_semester"], observed=True)
         .apply(lambda x: pd.Series(dict(rate=_rate(x), res=x.Res.sum(), n=len(x))), include_groups=False)
         .reset_index())
    g = g[g.res >= 2000]
    overall = _rate(d)

    fig, axes = figs.new(1, 2, w=6.8, h=4.4, sharey=True)
    for ax, sem in zip(axes, [1.0, 2.0]):
        sub = g[g.semester == sem]
        for i, y in enumerate(academic_year_order()):
            s = sub[sub.academic_year == y].sort_values("week_of_semester")
            if len(s) < 3:
                continue
            ax.plot(s.week_of_semester, s.rate, "o-", lw=2, ms=4,
                    color=figs.PALETTE[i], label=y)
        ax.axhline(overall, color="black", lw=1, ls=":")
        ax.set_xlabel("هفته‌ی نیمسال"); ax.set_title(f"نیمسال {int(sem)}")
        if sem == 1.0:
            ax.set_ylabel("نرخ وزنی"); ax.legend(loc="upper left", fontsize=9)
    fig.suptitle("پروفایل درون‌نیمسال — آیا شکل بین سال‌ها تکرار می‌شود؟")
    figs.save(fig, "r8b_02_week_of_semester", g)

    # ⭐ آزمون تکرارپذیری: همبستگی پروفایل سال‌به‌سال (لنگر: F176/F187)
    rows = []
    for sem in [1.0, 2.0]:
        piv = g[g.semester == sem].pivot_table(index="week_of_semester",
                                               columns="academic_year", values="rate")
        for a in piv.columns:
            for b in piv.columns:
                if a < b:
                    both = piv[[a, b]].dropna()
                    if len(both) >= 5:
                        r = stats.spearmanr(both[a], both[b]).statistic
                        rows.append(dict(semester=int(sem), year_a=a, year_b=b,
                                         n_weeks=len(both), spearman=r))
    out = pd.DataFrame(rows)
    print(md_table(out))
    return out


# ---------------------------------------------------------------------------
# ۸-۵-۳ شکل کامل اثر رمضان
# ---------------------------------------------------------------------------

def p853_ramadan(f: pd.DataFrame) -> pd.DataFrame:
    """⚠️ این بند دو ادعای ثبت‌شده‌ی پروژه را تصحیح می‌کند.

    ۱. جدول بند ۱۴-۰ می‌گوید پنجره‌ی v4 «۳ رمضان کامل» دارد. تقویم ۹۰ روز رمضان
       علامت زده، ولی **شبکه‌ی سرویس ۳۳ روز** دارد و رمضان سال ۱۴۰۴-۱۴۰۵ اصلاً
       داخل پنجره نیست.
    ۲. «شکل اثر رمضان» (F180) از منحنی‌ای خوانده شده که هر نقطه‌اش **یک روز**
       است، و دو رمضان موجود فقط در روزهای ۱ تا ۷ هم‌پوشانی دارند: روزهای ۸–۱۶
       فقط از ۱۴۰۳-۱۴۰۴ می‌آیند، روزهای ۲۳–۳۰ فقط از ۱۴۰۲-۱۴۰۳، و روزهای ۱۷–۲۲
       در هیچ سالی سرویس ندارند. آن منحنی، دو سال دوخته‌شده به هم است نه یک شکل
       تکرارشده.
    """
    header("۸-۵-۳ رمضان — تصحیح پوشش، و شکلی که تکرار نشده", 2)
    d = f.copy()
    base = {m: _rate(d[(~d.is_ramadan) & (d.meal == m)]) for m in ["lunch", "dinner"]}
    vol_base = {m: d[(~d.is_ramadan) & (d.meal == m)].groupby("date_gregorian").Res.sum().mean()
                for m in ["lunch", "dinner"]}
    r = d[d.is_ramadan].copy()
    r["day"] = r.days_into_ramadan.astype(int)

    cov = (r.groupby(["academic_year", "meal", "day"], observed=True)
           .apply(lambda x: pd.Series(dict(
               service_days=x.date_gregorian.nunique(), res=x.Res.sum(), rate=_rate(x))),
                  include_groups=False).reset_index())
    cov["vol_ratio"] = cov.apply(lambda x: (x.res / x.service_days) / vol_base[x.meal], axis=1)
    cov["lift"] = cov.apply(lambda x: x.rate / base[x.meal], axis=1)

    fig, ax = figs.new(1, 3, w=4.7, h=4.3)
    years = [y for y in academic_year_order() if y in cov.academic_year.unique()]

    # قاب ۱ — نقشه‌ی پوشش: کدام روز رمضان از کدام سال آمده
    grid = np.zeros((len(years), 30))
    for i, y in enumerate(years):
        for dd in cov[(cov.academic_year == y) & (cov.meal == "dinner")].day:
            grid[i, int(dd) - 1] = 1
    ax[0].imshow(grid, aspect="auto", cmap="Greys", vmin=0, vmax=1.4)
    ax[0].set_yticks(range(len(years))); ax[0].set_yticklabels(years, fontsize=9)
    ax[0].set_xticks(range(0, 30, 2)); ax[0].set_xticklabels(range(1, 31, 2), fontsize=8)
    ax[0].set_xlabel("روز از آغاز رمضان"); ax[0].grid(False)
    ax[0].set_title("پوشش واقعی — هر خانه یک روز سرویس")
    figs.annotate(ax[0], "هم‌پوشانی دو سال:\nفقط روزهای ۱ تا ۷\nروزهای ۱۷–۲۲: بدون داده",
                  loc="lower left", fontsize=8)

    # قاب ۲ — فروپاشی حجم
    for i, meal in enumerate(["lunch", "dinner"]):
        s = (cov[cov.meal == meal].groupby("day", observed=True)
             .apply(lambda x: (x.res.sum() / x.service_days.sum()) / vol_base[meal],
                    include_groups=False))
        ax[1].plot(s.index, s.values, "o-", lw=2, ms=4.5, color=figs.PALETTE[i], label=MEAL_FA[meal])
    ax[1].axhline(1, color=figs.C_GREY, lw=1.2, ls="--")
    ax[1].set_xlabel("روز از آغاز رمضان"); ax[1].set_ylabel("حجم رزرو نسبت به روز عادی")
    ax[1].legend(loc="upper right"); ax[1].set_title("رمضان اول حجم را می‌زند")
    ld = int(cov[cov.meal == "lunch"].service_days.sum())
    dd_ = int(cov[cov.meal == "dinner"].service_days.sum())
    figs.annotate(ax[1], f"ناهار در کل رمضان‌ها {ld} روز سرویس\nشام {dd_} روز",
                  loc="upper left", fontsize=8)

    # قاب ۳ — نرخ شام، با رنگ سال مبدأ
    for i, y in enumerate(years):
        s = cov[(cov.academic_year == y) & (cov.meal == "dinner")].sort_values("day")
        ax[2].plot(s.day, s.lift, "o-", lw=2, ms=5, color=figs.PALETTE[i], label=y)
        for _, row in s.iterrows():
            if row.res < 1500:
                ax[2].plot(row.day, row.lift, "o", ms=9, mfc="none", mec=figs.C_WARN, mew=1.6)
    ax[2].axhline(1, color=figs.C_GREY, lw=1.2, ls="--")
    ax[2].set_xlabel("روز از آغاز رمضان"); ax[2].set_ylabel("نسبت نرخ شام به روز عادی")
    ax[2].legend(loc="upper left", fontsize=8); ax[2].set_title("نرخ شام — رنگ = سال مبدأ")
    figs.annotate(ax[2], "حلقه‌ی روشن: <۱۵۰۰ رزرو\nهر نقطه از یک روز می‌آید",
                  loc="upper right", fontsize=8)

    fig.suptitle("رمضان: آنچه داده واقعاً دارد، و آنچه منحنی نشان می‌داد")
    figs.save(fig, "r8b_03_ramadan_shape", cov)
    print(md_table(cov))
    print(f"\nروز رمضانِ دارای سرویس در شبکه: {r.date_gregorian.nunique()} از ۹۰ روز تقویمی · "
          f"به تفکیک سال: {r.groupby('academic_year').date_gregorian.nunique().to_dict()}")
    print(f"نرخ کل رمضان (شام): {_rate(r[r.meal=='dinner']):.4f} · پایه‌ی شام: {base['dinner']:.4f} · "
          f"نسبت: {_rate(r[r.meal=='dinner'])/base['dinner']:.3f}×")
    return cov


# ---------------------------------------------------------------------------
# ۸-۵-۴ نامتقارنی پیش/پس تعطیلات
# ---------------------------------------------------------------------------

def p854_holiday(f: pd.DataFrame) -> pd.DataFrame:
    header("۸-۵-۴ نامتقارنی پیش و پس از تعطیلات، و طول بلوک", 2)
    d = f.copy()
    # فاصله‌ی علامت‌دار تا نزدیک‌ترین تعطیلی: منفی = پیش، مثبت = پس
    d["offset"] = np.where(d.days_to_next_holiday <= d.days_since_last_holiday,
                           -d.days_to_next_holiday, d.days_since_last_holiday)
    d = d[d.offset.between(-5, 5)]
    g = (d.groupby(["offset", "meal"], observed=True)
         .apply(lambda x: pd.Series(dict(rate=_rate(x), res=x.Res.sum(), n=len(x))),
                include_groups=False).reset_index())
    base = _rate(f[f.days_to_next_holiday > 5])

    blk = f[f.holiday_block_length > 0].copy()
    blk["blk"] = blk.holiday_block_length.clip(upper=8).astype(int)
    gb = (blk.groupby("blk", observed=True)
          .apply(lambda x: pd.Series(dict(rate=_rate(x), res=x.Res.sum(), n=len(x))),
                 include_groups=False).reset_index())

    fig, ax = figs.new(1, 2, w=6.6, h=4.3)
    for i, meal in enumerate(["lunch", "dinner"]):
        s = g[g.meal == meal].sort_values("offset")
        ax[0].plot(s.offset, s.rate, "o-", lw=2, ms=5, color=figs.PALETTE[i], label=MEAL_FA[meal])
    ax[0].axhline(base, color="black", lw=1, ls=":")
    ax[0].axvline(0, color=figs.C_GREY, lw=1.2, ls="--")
    ax[0].set_xlabel("فاصله تا تعطیلی (منفی = پیش از تعطیلی)")
    ax[0].set_ylabel("نرخ وزنی"); ax[0].legend(loc="upper left")
    ax[0].set_title("پروفایل رویدادمحور تعطیلات")
    pre = g[g.offset.between(-2, -1)]; post = g[g.offset.between(1, 2)]
    figs.annotate(ax[0], f"پایه {base:.2%}\nدو روز پیش {_wm(pre):.2%}\nدو روز پس {_wm(post):.2%}")

    ax[1].bar(gb.blk, gb.rate, color=figs.C_MAIN)
    ax[1].axhline(base, color="black", lw=1, ls=":")
    ax[1].set_xlabel("طول بلوک تعطیلی (روز، ≥۸ یکی شده)")
    ax[1].set_ylabel("نرخ وزنی روزهای مجاور")
    ax[1].set_title("آیا طول بلوک اثر یکنواخت دارد؟")
    rho_lin = stats.spearmanr(gb.blk, gb.rate).statistic
    figs.annotate(ax[1], f"همبستگی رتبه‌ای طول↔نرخ {rho_lin:+.2f}")
    out = pd.concat([g.assign(kind="offset"), gb.rename(columns={"blk": "offset"}).assign(kind="block")],
                    ignore_index=True)
    figs.save(fig, "r8b_04_holiday_profile", out)
    print(md_table(g)); print(); print(md_table(gb))
    return out


def _wm(g: pd.DataFrame) -> float:
    return float((g.rate * g.res).sum() / g.res.sum())


# ---------------------------------------------------------------------------
# ۸-۵-۵ روز هفته × وعده × نوع سلف
# ---------------------------------------------------------------------------

def p855_dow_meal_venue(f: pd.DataFrame) -> pd.DataFrame:
    header("۸-۵-۵ روز هفته × وعده × نوع سلف", 2)
    d = f.copy()
    d["dow_fa"] = d.day_of_week.map(DOW_FA)
    fig, axes = figs.new(1, 2, w=6.6, h=4.6)
    frames = []
    allv = _pivot_rate(d, "venue_type", "dow_fa")
    for ax, meal in zip(axes, ["lunch", "dinner"]):
        m = _pivot_rate(d[d.meal == meal], "venue_type", "dow_fa")
        m = m.reindex(columns=[DOW_FA[i] for i in range(7) if DOW_FA[i] in m.columns])
        m = m.loc[allv.mean(axis=1).sort_values(ascending=False).index.intersection(m.index)]
        im = _heat(ax, m, vmin=0.02, vmax=0.16, fs=7)
        ax.set_title(MEAL_FA[meal])
        frames.append(m.stack(future_stack=True).rename("rate").reset_index().assign(meal=meal))
    fig.colorbar(im, ax=axes, shrink=0.7, label="نرخ وزنی")
    fig.suptitle("نرخ به تفکیک نوع سلف و روز هفته — دو وعده، دو الگوی متفاوت")
    out = pd.concat(frames, ignore_index=True)
    figs.save(fig, "r8b_05_dow_venue_meal", out)
    print(md_table(out.pivot_table(index="venue_type", columns="meal", values="rate").reset_index()))
    return out


# ---------------------------------------------------------------------------
# ۸-۵-۶ تجزیه‌ی چنددوره‌ای
# ---------------------------------------------------------------------------

def p856_mstl(f: pd.DataFrame) -> pd.DataFrame:
    header("۸-۵-۶ تجزیه‌ی چنددوره‌ای MSTL و سهم واریانس", 2)
    from statsmodels.tsa.seasonal import MSTL
    rows = []
    fig, axes = figs.new(2, 1, w=9.4, h=3.0, sharex=True)
    for ax, meal in zip(axes, ["lunch", "dinner"]):
        s = (f[f.meal == meal].groupby("date_gregorian", observed=True)[["Res", "NoRecv"]]
             .sum())
        ser = (s.NoRecv / s.Res).asfreq("D")
        filled = ser.interpolate(limit_direction="both")
        res = MSTL(filled, periods=(7, 182)).fit()
        # ⚠️ سهم واریانس فقط روی روزهای **مشاهده‌شده** محاسبه می‌شود. شکاف سرویس در
        # این داده فصلی است (تابستان و تعطیلات)، و درون‌یابی‌اش یک مؤلفه‌ی نیم‌سالی
        # ساختگی می‌سازد — همان دام «عددی که غلط است و خطا نمی‌دهد» (بخش ۸ سند دور ۷).
        obs = ser.notna().reindex(filled.index, fill_value=False)
        var_t = np.var(res.trend[obs]); var_s7 = np.var(res.seasonal["seasonal_7"][obs])
        var_s182 = np.var(res.seasonal["seasonal_182"][obs]); var_r = np.var(res.resid[obs])
        tot = var_t + var_s7 + var_s182 + var_r
        rows.append(dict(meal=meal, n_days=int(ser.notna().sum()), filled=int(filled.notna().sum() - ser.notna().sum()),
                         share_trend=var_t / tot, share_weekly=var_s7 / tot,
                         share_halfyear=var_s182 / tot, share_resid=var_r / tot))
        ax.plot(ser.index, ser.values, color=figs.C_GREY, lw=0.8, alpha=0.6, label="نرخ روزانه")
        ax.plot(res.trend.index, res.trend.values, color=figs.C_ALT, lw=2.4, label="روند")
        ax.plot(res.trend.index, (res.trend + res.seasonal["seasonal_182"]).values,
                color=figs.C_MAIN, lw=1.4, alpha=0.9, label="روند + فصلی نیم‌سالی")
        ax.set_ylabel(MEAL_FA[meal]); ax.set_ylim(0, min(0.3, float(np.nanquantile(ser, 0.999))))
        if meal == "lunch":
            ax.legend(loc="upper left", ncol=3, fontsize=9)
        figs.annotate(ax, f"هفتگی {var_s7/tot:.1%} · نیم‌سالی {var_s182/tot:.1%} · "
                          f"روند {var_t/tot:.1%} · باقی‌مانده {var_r/tot:.1%}", loc="upper right", fontsize=8)
    fig.suptitle("تجزیه‌ی چنددوره‌ای سری روزانه — سهم هر مؤلفه از واریانس")
    out = pd.DataFrame(rows)
    figs.save(fig, "r8b_06_mstl", out)
    print(md_table(out))
    return out


# ---------------------------------------------------------------------------
# ۸-۵-۷ انتقال‌پذیری فصلی
# ---------------------------------------------------------------------------

def p857_transfer(f: pd.DataFrame) -> pd.DataFrame:
    header("۸-۵-۷ انتقال‌پذیری فصلی بین سال‌ها", 2)
    d = f.copy()
    d["doy"] = d.date_gregorian.dt.dayofyear
    d["jm"] = d.month_jalali
    prof = (d.groupby(["academic_year", "jm"], observed=True)
            .apply(lambda x: pd.Series(dict(rate=_rate(x), res=x.Res.sum())),
                   include_groups=False).reset_index())
    prof = prof[prof.res >= 20000]
    piv = prof.pivot_table(index="jm", columns="academic_year", values="rate")

    rows = []
    years = [y for y in academic_year_order() if y in piv.columns]
    for a in years:
        for b in years:
            if a < b:
                both = piv[[a, b]].dropna()
                if len(both) >= 4:
                    r = stats.pearsonr(both[a], both[b])
                    # خطای انتقال: اگر پروفایل سال a را روی سال b به‌کار ببریم
                    mae_transfer = float(np.mean(np.abs(both[a] - both[b])))
                    mae_const = float(np.mean(np.abs(both[b] - both[b].mean())))
                    rows.append(dict(year_a=a, year_b=b, n_months=len(both),
                                     pearson=r.statistic, p=r.pvalue,
                                     mae_transfer=mae_transfer, mae_constant=mae_const,
                                     gain=(mae_const - mae_transfer) / mae_const))
    out = pd.DataFrame(rows)

    fig, ax = figs.new(1, 2, w=6.6, h=4.3)
    for i, y in enumerate(years):
        s = piv[y].dropna()
        ax[0].plot([MONTH_FA[m - 1] for m in s.index], s.values, "o-", lw=2, color=figs.PALETTE[i], label=y)
    ax[0].set_ylabel("نرخ وزنی ماهانه"); ax[0].legend(loc="upper left", fontsize=9)
    ax[0].tick_params(axis="x", rotation=60)
    ax[0].set_title("پروفایل ماه شمسی در سه سال")

    x = np.arange(len(out))
    ax[1].bar(x - 0.2, out.mae_constant, width=0.4, color=figs.C_GREY, label="خطای پیش‌بینی ثابت")
    ax[1].bar(x + 0.2, out.mae_transfer, width=0.4, color=figs.C_MAIN, label="خطای انتقال پروفایل سال قبل")
    ax[1].set_xticks(x); ax[1].set_xticklabels([f"{a[-4:]}→{b[-4:]}" for a, b in zip(out.year_a, out.year_b)])
    ax[1].set_ylabel("MAE نرخ ماهانه"); ax[1].legend(loc="upper right", fontsize=9)
    ax[1].set_title("آیا پروفایل سال قبل کمک می‌کند؟")
    figs.annotate(ax[1], "بهره‌ی مثبت = پروفایل منتقل می‌شود\n"
                         + " · ".join(f"{g:+.0%}" for g in out.gain), loc="upper left", fontsize=8)
    figs.save(fig, "r8b_07_seasonal_transfer", out)
    print(md_table(out))
    return out


# ---------------------------------------------------------------------------
# ۸-۵-۸ نوروز، تابستان و بین‌نیمسال
# ---------------------------------------------------------------------------

def p858_regimes(f: pd.DataFrame) -> pd.DataFrame:
    header("۸-۵-۸ رژیم‌های تقویمی — نوروز، تابستان، بین‌نیمسال", 2)
    d = f.copy()
    d["regime"] = np.select(
        [d.is_nowruz_block, d.is_summer_break, d.is_inter_semester_break,
         d.is_exam_period, d.is_ramadan],
        ["نوروز", "تابستان", "بین‌نیمسال", "امتحانات", "رمضان"], default="عادی")
    g = (d.groupby("regime", observed=True)
         .apply(lambda x: pd.Series(dict(days=x.date_gregorian.nunique(), cells=len(x),
                                         res=x.Res.sum(), rate=_rate(x),
                                         waste=x.NoRecv.sum(),
                                         res_per_day=x.Res.sum() / x.date_gregorian.nunique())),
                include_groups=False).reset_index().sort_values("rate", ascending=False))
    g["share_waste"] = g.waste / g.waste.sum()
    g["share_res"] = g.res / g.res.sum()
    base = float(g.loc[g.regime == "عادی", "rate"].iloc[0])
    g["lift"] = g.rate / base

    fig, ax = figs.new(1, 2, w=6.6, h=4.3)
    figs.rtl_barh(ax[0], g.regime.tolist(), g.rate.tolist(), color=figs.C_MAIN)
    ax[0].axvline(base, color="black", lw=1.2, ls="--")
    ax[0].set_xlabel("نرخ وزنی"); ax[0].set_title("نرخ به تفکیک رژیم تقویمی")

    ax[1].scatter(g.res_per_day, g.rate, s=np.sqrt(g.res) / 6, color=figs.C_ALT, alpha=0.8)
    for _, r in g.iterrows():
        ax[1].annotate(r.regime, (r.res_per_day, r.rate), fontsize=9,
                       xytext=(4, 4), textcoords="offset points")
    ax[1].set_xlabel("حجم رزرو در هر روز سرویس"); ax[1].set_ylabel("نرخ وزنی")
    ax[1].set_title("رژیم کم‌حجم ≠ رژیم پرنرخ")
    figs.annotate(ax[1], "اندازه‌ی نقطه = حجم کل رژیم", loc="upper right", fontsize=8)
    figs.save(fig, "r8b_08_calendar_regimes", g)
    print(md_table(g))
    return g


def main() -> None:
    set_global_seed()
    figs.setup()
    f = load_features()
    header(f"دور ۸ · اطلس تقویم و فصل — {len(f):,} سلول، {f.date_gregorian.nunique()} روز")
    p851_dow_month(f)
    p852_week_of_semester(f)
    p853_ramadan(f)
    p854_holiday(f)
    p855_dow_meal_venue(f)
    p856_mstl(f)
    p857_transfer(f)
    p858_regimes(f)


if __name__ == "__main__":
    main()
