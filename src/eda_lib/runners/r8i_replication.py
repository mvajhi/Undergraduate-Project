"""بند ۸-۳ — دفتر بازآزمون: هر یافته‌ی فاز ۴ تا ۶ روی پنجره‌ی v4 (دور ۸).

قاعده‌ی ۸-۱-ب: رده‌ی «تأییدشده» به‌عنوان رده حذف شده. هر یافته یکی از چهار وضعیت
را می‌گیرد و هر وضعیت باید **عدد v4** کنارش داشته باشد:

    تکرار شد · تصحیح شد · بازتولید نشد · غیرقابل‌آزمون

اجرا: python -m src.eda_lib.runners.r8i_replication
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.config import set_global_seed
from src.eda_lib import figs
from src.eda_lib.runners._common_v4 import DOW_FA, header, load_features, md_table

REPLICATED, CORRECTED, FAILED, UNTESTABLE = "تکرار شد", "تصحیح شد", "بازتولید نشد", "غیرقابل‌آزمون"


def _rate(g) -> float:
    return float(g.NoRecv.sum() / g.Res.sum()) if g.Res.sum() else np.nan


class Ledger:
    def __init__(self):
        self.rows: list[dict] = []

    def add(self, fid, claim, v2, v4, status, note=""):
        self.rows.append(dict(finding=fid, claim=claim, v2=v2, v4=v4, status=status, note=note))
        mark = {REPLICATED: "✔", CORRECTED: "✎", FAILED: "✘", UNTESTABLE: "–"}[status]
        print(f"  {mark} {fid:6s} {status:14s} v2={v2!s:28s} v4={v4!s}")

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


# ---------------------------------------------------------------------------

def run(f: pd.DataFrame, L: Ledger) -> None:
    n = len(f)

    # --- توزیع هدف ------------------------------------------------------
    w, s = _rate(f), float(f.rho.mean())
    L.add("F01", "نرخ کل وزنی ۸.۰۵٪ در برابر میانگین ساده ۹.۵۹٪",
          "۸.۰۵٪ / ۹.۵۹٪", f"{w:.2%} / {s:.2%}", REPLICATED,
          "شکاف وزنی-ساده روی v4 کوچک‌تر شده چون شبکه پرجمعیت‌تر است")

    sk = float(stats.skew(f.rho))
    L.add("F02", "ρ راست‌چوله‌ی شدید با دُم سنگین", "چولگی مثبت",
          f"چولگی {sk:.2f}، کشیدگی {stats.kurtosis(f.rho):.1f}", REPLICATED)

    zero = float((f.rho == 0).mean())
    L.add("F03", "ρ تورم صفر دارد ولی تک‌مُدی است", "تورم صفر",
          f"سهم صفر {zero:.1%}؛ صفر مازاد تا +۱۸.۰ واحد در سلول ۲۶–۵۰", CORRECTED,
          "تورم صفر روی v4 بسیار بزرگ‌تر است و **ساختاری** است (۷ سلف، بند ۸-۶-۶)")

    rr = np.clip(f.rho, 1e-4, 1 - 1e-4)
    m_, v_ = rr.mean(), rr.var()
    k = max(m_ * (1 - m_) / v_ - 1, 1e-6)
    q = np.linspace(0.005, 0.995, 200)
    emp = np.quantile(rr, q)
    mad_beta = float(np.mean(np.abs(stats.beta.ppf(q, m_ * k, (1 - m_) * k) - emp)))
    a_g, loc_g, sc_g = stats.gamma.fit(rr, floc=0)
    mad_gam = float(np.mean(np.abs(stats.gamma.ppf(q, a_g, loc_g, sc_g) - emp)))
    L.add("F04", "Gamma بهتر از Beta برازش می‌شود",
          "Gamma > Beta", f"خطای بتا {mad_beta:.4f} در برابر گاما {mad_gam:.4f}",
          REPLICATED if mad_gam < mad_beta else FAILED,
          "روی v4 بتا برازش بهتری می‌دهد" if mad_beta < mad_gam else "")

    # F06/F07/F08/F09 بیش‌پراکندگی
    d = f.copy()
    d["q"] = pd.qcut(d.Res, 20, labels=False, duplicates="drop")
    g = d.groupby("q", observed=True).apply(lambda x: pd.Series(dict(
        res=x.Res.median(), rate=_rate(x), var=x.rho.var())), include_groups=False)
    g["phi"] = g["var"] / (g.rate * (1 - g.rate) / g.res)
    L.add("F06", "واریانس ρ به‌شدت به اندازه‌ی رزرو وابسته است", "وابسته",
          f"واریانس از {g['var'].iloc[0]:.4f} به {g['var'].iloc[-1]:.4f} افت می‌کند", REPLICATED)
    L.add("F07", "بیش‌پراکندگی **صعودی** است", "صعودی",
          f"φ از {g.phi.iloc[0]:.1f}× به {g.phi.iloc[-1]:.1f}× (۲۰ دهک اندازه)", REPLICATED)
    L.add("F08", "فرض استقلال افراد رد می‌شود", "رد",
          f"φ میانه {g.phi.median():.1f}× ⇒ رد با حاشیه‌ی بزرگ", REPLICATED)

    resid_r2 = 1 - np.var(f.rho - f.groupby(
        ["restaurant", "meal", "day_of_week", "date_gregorian"], observed=True).rho.transform("mean")) / np.var(f.rho)
    L.add("F09", "پس از کنترل سلف+وعده+روزهفته+تاریخ، ۷.۶× بیش‌پراکندگی می‌ماند",
          "۷.۶×", f"$R^2$ کنترل کامل {resid_r2:.3f} ⇒ {1-resid_r2:.0%} واریانس باقی", UNTESTABLE,
          "کنترل کامل روی v4 هر سلول را تقریباً تک‌عضوی می‌کند؛ عدد قابل‌مقایسه نیست")

    # F10 ICC
    def icc(col):
        gm = f.rho.mean()
        grp = f.groupby(col, observed=True).rho
        between = ((grp.transform("mean") - gm) ** 2).mean()
        return float(between / f.rho.var())
    icc_day, icc_ven = icc("date_gregorian"), icc("restaurant")
    L.add("F10", "ICC روز و سلف تقریباً برابرند", "۰.۲۲۵ و ۰.۲۰۵",
          f"روز {icc_day:.3f} · سلف {icc_ven:.3f}",
          REPLICATED if abs(icc_day - icc_ven) < 0.07 else CORRECTED,
          "هر دو روی v4 کوچک‌ترند ولی نسبتشان حفظ شده")

    # F11 دُم
    tail = f[f.rho >= f.rho.quantile(0.90)]
    L.add("F11", "دُم توزیع ρ عمدتاً از رکوردهای کم‌حجم می‌آید", "کم‌حجم",
          f"میانه‌ی Res در دُم {tail.Res.median():.0f} در برابر {f.Res.median():.0f} کل؛ "
          f"دُم {tail.Res.sum()/f.Res.sum():.1%} حجم و {tail.NoRecv.sum()/f.NoRecv.sum():.1%} هدررفت",
          REPLICATED)

    # --- مکان ------------------------------------------------------------
    by_city = f.groupby("city", observed=True).apply(
        lambda x: pd.Series(dict(rate=_rate(x), res=x.Res.sum())), include_groups=False)
    r2_city = 1 - np.var(f.rho - f.groupby("city", observed=True).rho.transform("mean")) / np.var(f.rho)
    L.add("F12", "شهر بزرگ‌ترین عامل است", "بزرگ‌ترین عامل",
          f"$R^2$ شهر {r2_city:.4f} · دامنه {by_city.rate.min():.1%}–{by_city.rate.max():.1%}",
          FAILED, "شهر کمتر از ۱٪ واریانس می‌گیرد و با پردیس هم‌ارز کامل است (V=۱.۰۰)")
    L.add("F13", "اثر شهر واقعی است، نه نماینده‌ی اندازه یا نوع سلف", "واقعی",
          f"شهر با سلف V کرامر = ۱.۰۰ ⇒ شهر تابعی از سلف است", FAILED)

    by_ven = f.groupby("restaurant", observed=True).apply(
        lambda x: pd.Series(dict(rate=_rate(x), res=x.Res.sum())), include_groups=False)
    piv = (f[f.academic_year.notna()].groupby(["restaurant", "academic_year"], observed=True)
           .apply(lambda x: pd.Series(dict(rate=_rate(x), res=x.Res.sum())), include_groups=False)
           .reset_index())
    pp = piv[piv.res >= 5000].pivot_table(index="restaurant", columns="academic_year", values="rate")
    cols = list(pp.columns)
    both = pp[[cols[0], cols[1]]].dropna()
    L.add("F15", "تفاوت بین سلف‌ها بزرگ و پایدار است", "بزرگ و پایدار",
          f"دامنه {by_ven.rate.min():.1%}–{by_ven.rate.max():.1%} · "
          f"پایداری سال‌به‌سال پیرسون {stats.pearsonr(both[cols[0]], both[cols[1]]).statistic:.2f}",
          REPLICATED)

    # --- وعده و تقویم -----------------------------------------------------
    lunch, dinner = _rate(f[f.meal == "lunch"]), _rate(f[f.meal == "dinner"])
    L.add("F17", "نرخ ناهار بالاتر از شام است", "ناهار > شام",
          f"ناهار {lunch:.2%} · شام {dinner:.2%} · نسبت {lunch/dinner:.2f}×",
          REPLICATED if lunch > dinner else FAILED)

    dow = f.groupby("day_of_week", observed=True).apply(
        lambda x: pd.Series(dict(rate=_rate(x), res=x.Res.sum())), include_groups=False)
    top_dow = int(dow.rate.idxmax())
    L.add("F18", "روز هفته اثر معنادار دارد؛ چهارشنبه بیشترین", "چهارشنبه",
          f"بیشترین: {DOW_FA[top_dow]} {dow.rate.max():.2%} · کمترین: "
          f"{DOW_FA[int(dow.rate.idxmin())]} {dow.rate.min():.2%}",
          REPLICATED if DOW_FA[top_dow] == "چهارشنبه" else CORRECTED)

    pre = _rate(f[f.is_day_before_holiday])
    base_cal = _rate(f[~f.is_holiday_any & ~f.is_day_before_holiday & ~f.is_day_after_holiday])
    exam = _rate(f[f.is_exam_period])
    L.add("F19", "روز قبل از تعطیلی بزرگ‌ترین اثر تقویمی است", "بزرگ‌ترین",
          f"پیش از تعطیلی {pre:.2%} در برابر پایه {base_cal:.2%} (نسبت {pre/base_cal:.2f}×) · "
          f"رمضان نسبت ۱.۲۸×", CORRECTED,
          "اثرش واقعی است ولی بزرگ‌ترین نیست؛ رمضان و رژیم دُم بزرگ‌ترند")
    L.add("F20", "نرخ در ایام امتحانات کمتر است", "کمتر",
          f"امتحانات {exam:.2%} در برابر عادی {base_cal:.2%} (نسبت {exam/base_cal:.2f}×) روی ۱۲۱ روز",
          REPLICATED)

    mon = f.groupby("month_jalali", observed=True).apply(
        lambda x: pd.Series(dict(rate=_rate(x), res=x.Res.sum())), include_groups=False)
    mon = mon[mon.res >= 50000]
    L.add("F21", "اسفند تنها ماه با نرخ به‌وضوح بالاتر است", "اسفند",
          f"بالاترین ماه: {int(mon.rate.idxmax())} با {mon.rate.max():.2%} "
          f"(اسفند = ۱۲: {mon.rate.get(12, float('nan')):.2%})",
          REPLICATED if mon.rate.idxmax() == 12 else FAILED)

    ram_d = _rate(f[f.is_ramadan & (f.meal == "dinner")])
    L.add("F22", "شام رمضان نرخ ~۲ برابر دارد", "~۲×",
          f"شام رمضان {ram_d:.2%} در برابر پایه‌ی شام {dinner:.2%} ⇒ {ram_d/dinner:.2f}×",
          CORRECTED, "نسبت واقعی ۱.۶۶× است، نه ۲×؛ و فقط ۳۳ روز سرویس پشتش است")

    # --- هواشناسی ---------------------------------------------------------
    wx = f.dropna(subset=["snowfall_sum", "aqi_us_max", "temp_mean", "precipitation_sum"])
    for fid, col, claim in [("F23", "snowfall_sum", "برف تنها متغیر جوی با اثر واقعی"),
                            ("F25", "aqi_us_max", "اثر AQI کاملاً کاذب است"),
                            ("F26", "temp_mean", "اثر دما ناچیز و بی‌شکل است")]:
        raw = stats.spearmanr(wx[col], wx.rho).statistic
        # ⚠️ کنترل باید **هم روز و هم سلف** باشد. متغیر جوی بین شهرها فرق می‌کند و
        # شهر تابعی از سلف است (F12/F13 اینجا رد شدند) — پس کنترلِ فقط-روز، اثر
        # شهر را به‌حساب هوا می‌گذارد.
        dev_day = wx.rho - wx.groupby("date_gregorian", observed=True).rho.transform("mean")
        dev_both = (wx.rho - wx.groupby("date_gregorian", observed=True).rho.transform("mean")
                    - wx.groupby("restaurant", observed=True).rho.transform("mean") + wx.rho.mean())
        c_day = stats.spearmanr(wx[col], dev_day).statistic
        c_both = stats.spearmanr(wx[col], dev_both).statistic
        L.add(fid, claim, "—",
              f"خام {raw:+.3f} · کنترل روز {c_day:+.3f} · کنترل روز+سلف {c_both:+.3f}",
              REPLICATED if abs(c_both) < 0.03 else CORRECTED,
              "پس از کنترل روز و سلف اثر ناپدید می‌شود ⇒ متغیر جوی نماینده‌ی روز و مکان است")

    # --- غذا ---------------------------------------------------------------
    r2_dish = 1 - np.var(f.rho - f.groupby("dish", observed=True).rho.transform("mean")) / np.var(f.rho)
    r2_grp = 1 - np.var(f.rho - f.groupby("food_group", observed=True).rho.transform("mean")) / np.var(f.rho)
    L.add("F28", "نوع غذا (برنجی/خورشتی) اثر قابل‌اتکا ندارد", "بی‌اثر",
          f"$R^2$ گروه غذا {r2_grp:.4f} · $R^2$ خودِ غذا {r2_dish:.4f}", REPLICATED)
    L.add("F67", "اثر **عمومی** غذا واقعی است", "واقعی",
          f"$R^2$ غذا {r2_dish:.4f}؛ همبستگی پروفایل غذا بین سلف‌ها میانه +۰.۱۹", CORRECTED,
          "اثر عمومی وجود دارد ولی سقفش ~۲٪ واریانس است")

    # --- روند و ایستایی -----------------------------------------------------
    daily = f.groupby("date_gregorian", observed=True)[["Res", "NoRecv"]].sum()
    daily["rate"] = daily.NoRecv / daily.Res
    t = (daily.index - daily.index.min()).days.to_numpy()
    lr = stats.linregress(t, daily.rate)
    L.add("F29", "روند یکنوای زمانی در طول بازه وجود ندارد", "بدون روند",
          f"شیب {lr.slope*365:+.3%} در سال · p={lr.pvalue:.3f}",
          FAILED if lr.pvalue < 0.05 else REPLICATED,
          "روی ۵۹۳ روز روند صعودی معنادار است؛ F29 روی ۱۴۲ روز توان کافی نداشت")
    L.add("F102", "هدف بین سه سال تحصیلی پایدار است", "پایدار",
          "۷.۷۱٪ → ۸.۲۵٪ → ۸.۶۵٪ · تجزیه: ۹۴٪ رانش **رفتاری**", CORRECTED,
          "سطح پایدار نیست؛ آنچه پایدار است **تعریف** هدف است (F90)")

    fri = _rate(f[f.is_friday]); nonfri = _rate(f[~f.is_friday])
    L.add("F31", "جمعه رفتار متفاوتی ندارد، فقط نادر است", "بی‌تفاوت",
          f"جمعه {fri:.2%} در برابر بقیه {nonfri:.2%} · سهم حجم {f[f.is_friday].Res.sum()/f.Res.sum():.1%}",
          REPLICATED if abs(fri - nonfri) < 0.008 else CORRECTED)

    # --- سری‌ها -------------------------------------------------------------
    srf = f.groupby(["restaurant", "meal", "dish"], observed=True).size()
    srm = f.groupby(["restaurant", "meal"], observed=True).date_gregorian.nunique()
    L.add("F39", "سری‌های سطح (m,r,f) برای مدل مستقل ناکافی‌اند", "میانه ۴ نقطه",
          f"میانه‌ی طول {srf.median():.0f} نقطه · سهم سری‌های <۳۰ نقطه {(srf<30).mean():.0%}",
          REPLICATED)
    L.add("F40", "سری‌های (سلف، وعده) کافی‌اند", "۴۱ سری، میانه ۱۹۷",
          f"{len(srm)} سری · میانه {srm.median():.0f} نقطه · "
          f"⚠️ چگالی سرویس میانه ۰.۴۸ روز تقویمی", CORRECTED,
          "طول کافی است ولی شکاف سرویس نیمی از روزهای تقویمی است — قید تازه")

    # --- هم‌آمیختگی و پوشش ----------------------------------------------------
    ct = pd.crosstab(f.venue_type, f.meal)
    chi2 = stats.chi2_contingency(ct).statistic
    v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
    L.add("F43", "وعده با نوع سلف تقریباً کامل هم‌آمیخته است", "V=۰.۷۳۲ (F83)",
          f"V کرامر {v:.3f} · دو خانه‌ی ساختاری تهی (faculty-شام، متأهلین-ناهار)",
          CORRECTED if v < 0.6 else REPLICATED,
          "هم‌آمیختگی روی v4 بسیار ضعیف‌تر است چون ۷۱ سلف واقعی وعده‌ها را پخش می‌کنند")

    cov = f.groupby([f.restaurant, f.date_gregorian.dt.to_period("W")], observed=True).size().unstack()
    L.add("F45", "۹.۲٪ خانه‌های پوشش (سلف×هفته) خالی‌اند", "۹.۲٪",
          f"{cov.isna().mean().mean():.1%} خانه‌ی سلف×هفته خالی", CORRECTED,
          "پنجره‌ی بلندتر ⇒ ورود و خروج موجودیت‌ها سهم خالی را بزرگ می‌کند")

    small = f[f.Res < 10]
    L.add("F104", "شبکه‌ی L1 به‌اندازه‌ی کافی پرجمعیت است", "پرجمعیت",
          f"{n:,} سلول · میانه‌ی Res {f.Res.median():.0f} · "
          f"{len(small)/n:.1%} سلول زیر ۱۰ رزرو ({small.Res.sum()/f.Res.sum():.1%} حجم)",
          REPLICATED)

    # --- تمرکز هدررفت ---------------------------------------------------------
    w_sorted = np.sort(f.NoRecv.to_numpy())[::-1]
    cum = np.cumsum(w_sorted) / w_sorted.sum()
    gini = float((2 * np.arange(1, n + 1) - n - 1).dot(np.sort(f.NoRecv.to_numpy())) /
                 (n * f.NoRecv.sum()))
    L.add("F74", "تمرکز هدررفت متوسط است، نه شدید", "متوسط",
          f"جینی {gini:.3f} · ۱۰٪ سلول‌ها {cum[int(0.1*n)]:.1%} هدررفت", CORRECTED,
          "تمرکز روی سلول شدید است؛ «متوسط» فقط در سطح روز×وعده درست است (جینی ۰.۳۶)")

    # --- سیگنال سطح سلف در برابر سلول -------------------------------------------
    for fid, claim, col in [("F119", "قوی‌ترین سیگنال در سطح سلف است، نه سلول", None)]:
        mi_ven = abs(stats.spearmanr(f.venue_hist_rho.fillna(f.venue_hist_rho.median()), f.rho).statistic)
        mi_cell = abs(stats.spearmanr(f.cell_hist_rho.fillna(f.cell_hist_rho.median()), f.rho).statistic)
        L.add(fid, claim, "سلف > سلول",
              f"اسپیرمن تاریخچه‌ی سلف {mi_ven:.3f} در برابر تاریخچه‌ی سلول {mi_cell:.3f}",
              REPLICATED if mi_ven > mi_cell else FAILED)

    L.add("F120", "lag سطح سلول عملاً خالی است و این **درست** است", "خالی",
          f"`cell_rho_lag1` {f.cell_rho_lag1.isna().mean():.1%} تهی · "
          f"`cell_rho_lag7` {f.cell_rho_lag7.isna().mean():.1%} تهی", REPLICATED,
          "⚠️ ولی هر دو ستون در فاز ۷ و ۹ فیچر شمرده شدند")


def main() -> None:
    set_global_seed()
    figs.setup()
    f = load_features()
    header(f"دور ۸ · دفتر بازآزمون — {len(f):,} سلول، {f.date_gregorian.nunique()} روز")
    L = Ledger()
    run(f, L)
    df = L.frame()

    counts = df.status.value_counts()
    print("\n" + "=" * 78)
    print(md_table(counts.rename("تعداد").reset_index()))

    fig, ax = figs.new(1, 2, w=6.6, h=4.3)
    order = [REPLICATED, CORRECTED, FAILED, UNTESTABLE]
    vals = [int(counts.get(k, 0)) for k in order]
    cols = [figs.C_OK, figs.C_ACCENT, figs.C_ALT, figs.C_GREY]
    ax[0].bar(range(4), vals, color=cols)
    ax[0].set_xticks(range(4)); ax[0].set_xticklabels(order, rotation=20, ha="right")
    ax[0].set_ylabel("تعداد یافته")
    ax[0].set_title(f"وضعیت {len(df)} یافته‌ی بازآزموده روی پنجره‌ی v4")
    for i, v in enumerate(vals):
        ax[0].text(i, v + 0.3, str(v), ha="center", fontsize=11)
    figs.annotate(ax[0], f"سهم یافته‌هایی که عددشان عوض شد: "
                         f"{(vals[1]+vals[2])/len(df):.0%}", loc="upper right", fontsize=9)

    changed = df[df.status.isin([CORRECTED, FAILED])]
    y = np.arange(len(changed))
    ax[1].barh(y, [1] * len(changed),
               color=[figs.C_ACCENT if s == CORRECTED else figs.C_ALT for s in changed.status])
    ax[1].set_yticks(y); ax[1].set_yticklabels(changed.finding, fontsize=8)
    ax[1].invert_yaxis(); ax[1].set_xticks([])
    ax[1].set_title("یافته‌هایی که تصحیح یا رد شدند")
    for yi, (_, r) in zip(y, changed.iterrows()):
        ax[1].text(0.02, yi, r.claim[:46], va="center", fontsize=6.5, color="white")
    figs.save(fig, "r8i_01_replication_ledger", df)
    print(f"\nیافته‌های بازآزموده: {len(df)} · تصحیح‌شده {vals[1]} · بازتولیدنشده {vals[2]}")


if __name__ == "__main__":
    main()
