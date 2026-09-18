"""بند ۹-۱۱ — محورهای کاوش‌نشده: رضایت، فراموشی کارت، قیمت، و تحرک مکانی.

اجرا: python -m src.eda_lib.runners.r9i_unexplored

سه ستون خام این بسته در `doc/data_dictionary_individual.md` «فیچر بالقوه‌ی فاز ۵»
علامت خورده بودند و فاز ۵ تمام شد بی‌آنکه ساخته شوند: `Comment` (۳۷۴٬۸۷۳ رأی
غیرپیش‌فرض)، `Reception`/`ReceptionType` (۹.۴٪ ردیف‌ها)، و `Price`.
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


def p9111_satisfaction_level(rows: pd.DataFrame) -> pd.DataFrame:
    """۹-۱۱-۱ آیا نارضایتی با عدم‌دریافت رابطه دارد؟ (سطح رزرو)."""
    header("۹-۱۱-۱ نرخ عدم‌دریافت به تفکیک رأی همان رزرو", 2)
    t = (rows.groupby("Comment", observed=True)
         .agg(رزرو=("dont_receive", "size"), نرخ=("dont_receive", "mean"))
         .reset_index().sort_values("رزرو", ascending=False))
    t["سهم رزرو"] = t.رزرو / t.رزرو.sum()
    print(md_table(t.round(5)))
    print("\n⚠️ این جدول **هم‌زمانی** است، نه پیش‌بینی: رأی پس از سرو ثبت می‌شود.")
    print("   کسی که غذا را نگرفته عملاً نمی‌تواند رأی بدهد ⇒ رابطه‌ی مکانیکی.")
    return t


def p9112_satisfaction_leading(rows: pd.DataFrame) -> pd.DataFrame:
    """⭐ ۹-۱۱-۲ آیا نارضایتی **پیشرو** است؟ رأی «بد» امروز، غیبت فردا؟

    ⚠️ قاعده‌ی برش: رأی هر رزرو پس از دریافت ثبت می‌شود. پس فقط رأیِ فرصت‌های
    **پیشین** مجاز است و پیش‌بینی روی فرصت **بعدی** همان فرد سنجیده می‌شود.
    """
    header("۹-۱۱-۲ رأی فرصت قبلی در برابر عدم‌دریافت فرصت بعدی", 2)
    d = rows[["PersonId", "meal_seq", "dont_receive", "Comment"]].sort_values(
        ["PersonId", "meal_seq"])
    n = d.groupby("PersonId", observed=True).PersonId.transform("size")
    d = d[n >= N_MIN_MAIN].copy()
    d["prev_comment"] = d.groupby("PersonId", observed=True)["Comment"].shift()
    d["prev_dont"] = d.groupby("PersonId", observed=True)["dont_receive"].shift()
    d = d.dropna(subset=["prev_comment", "prev_dont"])
    # فقط فرصت‌هایی که قبلی‌شان **دریافت شده** بوده — وگرنه رأی معنا ندارد
    d = d[~d.prev_dont.astype(bool)]

    base = d.dont_receive.mean()
    t = (d.groupby("prev_comment", observed=True)
         .agg(n=("dont_receive", "size"), نرخ_بعدی=("dont_receive", "mean"))
         .reset_index())
    t["نسبت به پایه"] = t.نرخ_بعدی / base
    t = t[t.n >= 500].sort_values("نرخ_بعدی", ascending=False)
    print(f"نرخ پایه (پس از یک دریافت موفق): {base:.5f}")
    print(md_table(t.round(5)))

    bad = d[d.prev_comment.astype(str) == "بد"].dont_receive
    ok = d[d.prev_comment.astype(str) == "خوب"].dont_receive
    if len(bad) > 100 and len(ok) > 100:
        z = st.chi2_contingency([[bad.sum(), len(bad) - bad.sum()],
                                 [ok.sum(), len(ok) - ok.sum()]])
        print(f"\nآزمون «بد» در برابر «خوب»: chi2={z.statistic:.2f} · p={z.pvalue:.3e} · "
              f"نسبت نرخ = {bad.mean() / ok.mean():.3f}")

    fig, ax = figs.new(w=7.2, h=4.2)
    x = np.arange(len(t))
    ax.bar(x, t.نرخ_بعدی * 100, color=figs.C_MAIN)
    ax.axhline(base * 100, color=figs.C_ALT, ls="--", lw=1.4, label="نرخ پایه")
    ax.set_xticks(x, t.prev_comment.astype(str), fontsize=9)
    ax.set_xlabel("رأی فرصتِ پیشین")
    ax.set_ylabel("نرخ عدم‌دریافت فرصت بعدی (٪)")
    ax.set_title("آیا نارضایتی هشدار زودهنگام است؟")
    ax.legend(fontsize=8)
    figs.save(fig, "r9i_01_satisfaction_leading", t,
              caption="تنها سیگنال بالقوه‌ی «هشدار زودهنگام» در کل داده")
    return t


def p9113_forgot_card(P: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """۹-۱۱-۳ آیا «فراموش‌کار» تیپ مستقلی است؟"""
    header("۹-۱۱-۳ فراموشی کارت در برابر عدم‌دریافت", 2)
    S = at_threshold(P, N_MIN_MAIN)
    sp = st.spearmanr(S.forgot_card_rate, S.rho_eb)
    print(f"اسپیرمن(نرخ فراموشی، نرخ عدم‌دریافت EB) = {sp.statistic:.4f} (p={sp.pvalue:.2e})")
    q = pd.qcut(S.forgot_card_rate.rank(method="first"), 4,
                labels=["Q1 کم‌فراموش", "Q2", "Q3", "Q4 پرفراموش"])
    t = S.groupby(q, observed=True).agg(
        افراد=("rho_eb", "size"), نرخ_EB=("rho_eb", "mean"),
        نرخ_فراموشی=("forgot_card_rate", "mean"),
        رزرو=("n_res", "median")).reset_index()
    print(md_table(t.round(5)))

    # آیا فراموشی درون فرد با عدم‌دریافت هم‌زمان است؟
    d = rows[["PersonId", "forgot_card", "dont_receive"]]
    ct = pd.crosstab(d.forgot_card, d.dont_receive, normalize="index")
    print("\nسطح رزرو — نرخ عدم‌دریافت مشروط به فراموشی:")
    print(md_table(ct.reset_index().round(5)))

    fig, ax = figs.new(w=7.2, h=4.2)
    ax.plot(range(len(t)), t.نرخ_EB, "o-", color=figs.C_MAIN)
    ax.set_xticks(range(len(t)), t.iloc[:, 0].astype(str), fontsize=8)
    ax.set_ylabel("نرخ عدم‌دریافت EB")
    ax.set_title(f"چارک نرخ فراموشی کارت (اسپیرمن={sp.statistic:.3f})")
    figs.save(fig, "r9i_02_forgot_card", t,
              caption="آیا فراموشی کارت تیپ رفتاری جدایی است یا همان بی‌نظمی؟")
    return t


def p9114_price(P: pd.DataFrame) -> pd.DataFrame:
    """۹-۱۱-۴ آیا رده‌ی قیمت رفتار را جدا می‌کند؟"""
    header("۹-۱۱-۴ قیمت میانه‌ی فرد در برابر نرخ", 2)
    S = at_threshold(P, N_MIN_MAIN).dropna(subset=["demo_price_median"])
    q = pd.qcut(S.demo_price_median.rank(method="first"), 5,
                labels=[f"P{i}" for i in range(1, 6)])
    t = S.groupby(q, observed=True).agg(
        افراد=("rho_eb", "size"), قیمت_میانه=("demo_price_median", "median"),
        نرخ_EB=("rho_eb", "mean"), رزرو_میانه=("n_res", "median"),
        سهم_خوابگاهی=("demo_is_dorm", "mean")).reset_index()
    print(md_table(t.round(4)))
    sp = st.spearmanr(S.demo_price_median, S.rho_eb)
    print(f"اسپیرمن(قیمت میانه، نرخ EB) = {sp.statistic:.4f} (p={sp.pvalue:.2e})")
    print(f"دامنه‌ی نرخ بین پنجک‌ها: {t.نرخ_EB.max() - t.نرخ_EB.min():.4f} واحد")
    return t


def p9115_mobility(P: pd.DataFrame) -> pd.DataFrame:
    """۹-۱۱-۵ تحرک مکانی: آیا جابه‌جایی بین سلف‌ها با هدررفت رابطه دارد؟"""
    header("۹-۱۱-۵ تحرک بین سلف‌ها", 2)
    S = at_threshold(P, N_MIN_MAIN).dropna(subset=["venue_switch_rate"])
    q = pd.qcut(S.venue_switch_rate.rank(method="first"), 5,
                labels=["V1 وفادار", "V2", "V3", "V4", "V5 متحرک"])
    t = S.groupby(q, observed=True).agg(
        افراد=("rho_eb", "size"), نرخ_جابجایی=("venue_switch_rate", "mean"),
        نرخ_EB=("rho_eb", "mean"), سلف_یکتا=("n_venues", "median"),
        رزرو_میانه=("n_res", "median")).reset_index()
    print(md_table(t.round(4)))
    sp = st.spearmanr(S.venue_switch_rate, S.rho_eb)
    print(f"اسپیرمن خام(نرخ جابه‌جایی، نرخ EB) = {sp.statistic:.4f} (p={sp.pvalue:.2e})")

    # ⚠️ تحرک مکانیکاً به حجم گره خورده: کسی با ۷۲ رزرو نمی‌تواند بین سه سلف
    # بچرخد. رابطه‌ی خام بالا ممکن است تماماً همان زنجیره‌ی «کم‌حجمی ← نرخ بالا»
    # باشد که F81 هشدارش را داده. پس همان اسپیرمن **داخل** هر پنجک حجم.
    S = S.assign(vol_q=pd.qcut(S.n_res.rank(method="first"), 5,
                               labels=[f"حجم Q{i}" for i in range(1, 6)]))
    strat = []
    for vq, g in S.groupby("vol_q", observed=True):
        s2 = st.spearmanr(g.venue_switch_rate, g.rho_eb)
        strat.append({"پنجک حجم": str(vq), "افراد": len(g),
                      "میانه رزرو": float(g.n_res.median()),
                      "اسپیرمن درون‌پنجک": float(s2.statistic),
                      "p": float(s2.pvalue)})
    ts = pd.DataFrame(strat)
    print("\n▸ همان رابطه، **داخل** هر پنجک حجم:")
    print(md_table(ts.round(4)))
    print(f"  میانگین اسپیرمن درون‌پنجک: {ts['اسپیرمن درون‌پنجک'].mean():+.4f} "
          f"در برابر خام {sp.statistic:+.4f}")

    fig, ax = figs.new(1, 2, w=5.4, h=4.0)
    ax[0].plot(range(len(t)), t.نرخ_EB, "o-", color=figs.C_MAIN)
    ax[0].set_xticks(range(len(t)), t.iloc[:, 0].astype(str), fontsize=7)
    ax[0].set_ylabel("نرخ عدم‌دریافت EB")
    ax[0].set_title("تحرک مکانی")
    ax[1].plot(range(len(t)), t.رزرو_میانه, "s-", color=figs.C_ALT)
    ax[1].set_xticks(range(len(t)), t.iloc[:, 0].astype(str), fontsize=7)
    ax[1].set_ylabel("میانه‌ی تعداد رزرو")
    ax[1].set_title("و همبسته‌ی پنهانش: حجم")
    figs.save(fig, "r9i_03_mobility", t,
              caption="هر رابطه‌ی ظاهری تحرک با نرخ باید در برابر حجم کنترل شود")
    return t


def main() -> None:
    set_global_seed()
    setup()
    P = load_profile()
    rows = build_person_rows_v4(columns=["PersonId", "meal_seq", "dont_receive",
                                         "Comment", "forgot_card"])
    header("۹-۱۱ محورهای کاوش‌نشده")
    p9111_satisfaction_level(rows)
    p9112_satisfaction_leading(rows)
    p9113_forgot_card(P, rows)
    p9114_price(P)
    p9115_mobility(P)
    header("پایان بند ۹-۱۱")


if __name__ == "__main__":
    main()
