"""بند ۹-۱۰ — کوهورت، چرخه‌ی عمر و بقا.

اجرا: python -m src.eda_lib.runners.r9h_lifecycle

## ⚠️ دو سانسور که این بسته را می‌سازند یا خراب می‌کنند (قاعده‌ی ث)

- **سانسور چپ:** فردی که در ماه اول داده دیده می‌شود احتمالاً پیش از آن هم فعال
  بوده. تِنورش کوتاه‌تر از واقع ثبت می‌شود. F202 سهمش را ۴۰.۸٪ داده.
- **سانسور راست:** فردی که در ماه آخر داده فعال است هنوز خارج نشده. تحلیل بقا
  باید او را «سانسورشده» بشمارد، نه «خارج‌شده».

و یک تصحیح دامنه‌ای: **سال ورود دانشجو در داده نیست.** `StudentNumber` دو رقم
اولش `CollegeCode` است و ارقام سوم و چهارمش کد گروه آموزشی — نه سال. پس «کوهورت»
اینجا فقط **نخستین حضور در داده** است و این تعریف خودش سانسور چپ دارد.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as st

from src.config import set_global_seed
from src.eda_lib import figs
from src.eda_lib.person_profile_v4 import build_person_rows_v4
from src.eda_lib.runners._common_r9 import (
    N_MIN_MAIN, at_threshold, header, load_profile, md_table, setup,
)


def p9101_survival(P: pd.DataFrame) -> pd.DataFrame:
    """۹-۱۰-۱ منحنی بقا با سانسور راست صریح (کاپلان-مایر دستی)."""
    header("۹-۱۰-۱ بقا در سیستم — کاپلان-مایر با سانسور راست", 2)
    S = at_threshold(P, N_MIN_MAIN)
    dur = S.tenure_days.to_numpy(dtype="float64")
    event = (~S.right_censored.to_numpy()).astype(int)   # ۱ = خارج شد، ۰ = سانسور

    order = np.argsort(dur)
    d, e = dur[order], event[order]
    times = np.unique(d)
    n_at_risk, surv, rows = len(d), 1.0, []
    for t in times:
        at_t = d == t
        n_i = n_at_risk
        d_i = int(e[at_t].sum())
        if n_i > 0 and d_i > 0:
            surv *= (1 - d_i / n_i)
        rows.append({"روز": float(t), "در معرض": n_i, "خروج": d_i, "بقا": surv})
        n_at_risk -= int(at_t.sum())
    km = pd.DataFrame(rows)

    print(f"افراد: {len(S):,} · خارج‌شده (رویداد): {int(event.sum()):,} "
          f"({event.mean():.1%}) · سانسور راست: {int((1 - event).sum()):,}")
    for q in (0.75, 0.50, 0.25):
        sub = km[km.بقا <= q]
        med = float(sub.روز.iloc[0]) if len(sub) else np.nan
        print(f"  زمان تا بقای {q:.0%}: {med:.0f} روز" if not np.isnan(med)
              else f"  زمان تا بقای {q:.0%}: نرسید (سانسور)")
    print("  لنگر F204: ۸۱.۳٪ خروج (۳۷٬۸۱۳ از ۴۶٬۵۲۳) — **بدون** تصحیح سانسور")

    fig, ax = figs.new(1, 2, w=5.6, h=4.2)
    ax[0].step(km.روز, km.بقا, where="post", color=figs.C_MAIN, lw=2)
    ax[0].set_xlabel("تِنور (روز از نخستین حضور)")
    ax[0].set_ylabel("سهم باقی‌مانده در سیستم")
    ax[0].set_title("منحنی بقا (کاپلان-مایر)")
    ax[0].set_ylim(0, 1)

    for lbl, mask, c in (("سانسور چپ", S.left_censored, figs.C_ALT),
                         ("بدون سانسور چپ", ~S.left_censored, figs.C_OK)):
        sub = S[mask]
        if len(sub) < 100:
            continue
        ax[1].hist(sub.tenure_days, bins=50, alpha=0.55, color=c, label=f"{lbl} (n={len(sub):,})")
    ax[1].set_xlabel("تِنور (روز)")
    ax[1].set_ylabel("تعداد فرد")
    ax[1].set_title("سانسور چپ تِنور را کوتاه نشان می‌دهد")
    ax[1].legend(fontsize=8)
    figs.save(fig, "r9h_01_survival", km.iloc[::50],
              caption="بدون تفکیک «خارج‌شده» از «سانسورشده»، نرخ خروج بیش‌برآورد می‌شود")
    return km


def p9102_exit_selective(P: pd.DataFrame) -> pd.DataFrame:
    """۹-۱۰-۲ بازآزمون F204 — آیا خروج انتخابی است؟"""
    header("۹-۱۰-۲ خروج انتخابی (بازآزمون F204)", 2)
    S = at_threshold(P, 1)
    g = S.groupby(S.right_censored.map({True: "ماندگار (سانسور راست)",
                                        False: "خارج‌شده"}), observed=True)
    t = g.agg(افراد=("n_res", "size"), میانه_رزرو=("n_res", "median"),
              نرخ_EB=("rho_eb", "mean"), میانه_تنور=("tenure_days", "median")).reset_index()
    print(md_table(t.round(4)))
    ex = S[~S.right_censored]
    st_ = S[S.right_censored]
    print(f"\nسهم خارج‌شده: {(~S.right_censored).mean():.4f} (لنگر F204: ۰.۸۱۳)")
    print(f"نسبت نرخ خارج‌شده به ماندگار: {ex.rho_eb.mean() / st_.rho_eb.mean():.3f}× "
          f"(لنگر F204: ۰.۱۳۴۷ در برابر ۰.۰۸۵۸ ⇒ ۱.۵۷×)")
    mw = st.mannwhitneyu(ex.rho_eb, st_.rho_eb)
    print(f"Mann-Whitney p={mw.pvalue:.3e}")
    return t


def p9103_tenure_profile(rows: pd.DataFrame) -> pd.DataFrame:
    """۹-۱۰-۳ نرخ بر حسب موقعیت در چرخه‌ی عمر فرد (ابتدا، میانه، انتها)."""
    header("۹-۱۰-۳ نرخ در طول چرخه‌ی عمر فرد", 2)
    d = rows[["PersonId", "meal_seq", "dont_receive"]].sort_values(["PersonId", "meal_seq"])
    n = d.groupby("PersonId", observed=True).PersonId.transform("size")
    d = d[n >= 50].copy()
    d["pos"] = (d.groupby("PersonId", observed=True).cumcount()
                / (d.groupby("PersonId", observed=True).PersonId.transform("size") - 1))
    d["band"] = pd.cut(d.pos, bins=np.linspace(0, 1, 11), include_lowest=True)
    t = d.groupby("band", observed=True).dont_receive.agg(["mean", "size"]).reset_index()
    t["band"] = t["band"].astype(str)
    t.columns = ["دهک موقعیت", "نرخ", "رزرو"]
    print(md_table(t.round(5)))
    print(f"  افراد (≥۵۰ رزرو): {d.PersonId.nunique():,}")
    print("  ⚠️ دهک آخر شامل افرادی است که **رها کرده‌اند** — افزایش نرخ آنجا")
    print("     می‌تواند «خداحافظی» باشد، نه تغییر تدریجی رفتار.")

    fig, ax = figs.new(w=7.4, h=4.2)
    x = np.arange(len(t))
    ax.plot(x, t.نرخ * 100, "o-", color=figs.C_MAIN, lw=2)
    ax.axhline(t.نرخ.mean() * 100, color=figs.C_GREY, ls=":", lw=1.2)
    ax.set_xticks(x, [f"{i + 1}" for i in x])
    ax.set_xlabel("دهک موقعیت در چرخه‌ی عمر فرد")
    ax.set_ylabel("نرخ عدم‌دریافت (٪)")
    ax.set_title("نرخ از ابتدا تا انتهای حضور فرد")
    figs.save(fig, "r9h_02_tenure_profile", t,
              caption="آیا فرد در پایان حضورش بی‌قیدتر می‌شود؟")
    return t


def p9104_cohort(P: pd.DataFrame) -> pd.DataFrame:
    """۹-۱۰-۴ کوهورت نخستین‌حضور — با اعلام صریح سانسور چپ."""
    header("۹-۱۰-۴ کوهورت نخستین حضور در داده", 2)
    S = at_threshold(P, N_MIN_MAIN).copy()
    S["cohort"] = pd.to_datetime(S.first_seen).dt.to_period("Q").astype(str)
    t = (S.groupby("cohort", observed=True)
         .agg(افراد=("n_res", "size"), نرخ_EB=("rho_eb", "mean"),
              میانه_رزرو=("n_res", "median"), میانه_تنور=("tenure_days", "median"),
              سهم_سانسور_چپ=("left_censored", "mean")).reset_index())
    t = t[t.افراد >= 200]
    print(md_table(t.round(4)))
    print("\n⚠️ کوهورت نخستین ربع **کاملاً سانسور چپ** است — آن ردیف کوهورت نیست،")
    print("   «همه‌ی کسانی که پیش از شروع داده فعال بودند» است.")

    clean = t[t.سهم_سانسور_چپ < 0.5]
    if len(clean) >= 3:
        sp = st.spearmanr(range(len(clean)), clean.نرخ_EB)
        print(f"روند نرخ بین کوهورت‌های بدون سانسور چپ: اسپیرمن={sp.statistic:.3f} "
              f"(p={sp.pvalue:.3f}) روی {len(clean)} کوهورت")

    fig, ax = figs.new(w=8.0, h=4.4)
    x = np.arange(len(t))
    c = [figs.C_ALT if v >= 0.5 else figs.C_MAIN for v in t.سهم_سانسور_چپ]
    ax.bar(x, t.نرخ_EB, color=c)
    ax.set_xticks(x, t.cohort, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("نرخ EB میانگین")
    ax.set_title("کوهورت نخستین حضور (نارنجی = عمدتاً سانسور چپ، قابل‌تفسیر نیست)")
    figs.save(fig, "r9h_03_cohort", t,
              caption="سال ورود در داده نیست؛ این نمای جایگزین است و سانسور چپ دارد")
    return t


def main() -> None:
    set_global_seed()
    setup()
    P = load_profile()
    rows = build_person_rows_v4(columns=["PersonId", "meal_seq", "dont_receive"])
    header("۹-۱۰ کوهورت، چرخه‌ی عمر و بقا")
    p9101_survival(P)
    p9102_exit_selective(P)
    p9103_tenure_profile(rows)
    p9104_cohort(P)
    header("پایان بند ۹-۱۰")


if __name__ == "__main__":
    main()
