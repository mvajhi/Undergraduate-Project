"""بند ۹-۳ — ماتریس پروفایل فرد: پوشش آستانه، انقباض، و نقشه‌ی تهی‌بودن.

اجرا: python -m src.eda_lib.runners.r9a_person_profile
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import DATA_INTERIM
from src.eda_lib import figs
from src.eda_lib.person_profile_v4 import eb_shrink
from src.eda_lib.runners._common_r9 import (
    N_MIN_MAIN, THRESHOLDS, coverage_row, header, load_profile, md_table, setup,
)

#: بلوک‌های بند ۹-۳-۲ برای نقشه‌ی تهی‌بودن — ترتیب همان جدول WBS.
BLOCKS = {
    "ب۱ حجم و شدت": ["n_res", "n_portions", "n_active_days", "n_active_months",
                     "span_days", "res_per_active_month", "res_per_active_day", "density"],
    "ب۲ هدررفت": ["n_waste", "rho_raw", "rho_eb", "waste_share", "rho_lunch",
                  "rho_dinner", "lunch_share", "rho_slope_seq"],
    "ب۳ ریتم زمانی": ["dow_entropy", "gap_mean", "gap_cv", "burstiness"],
    "ب۴ توالی": ["spell_n", "spell_len_mean", "spell_len_max",
                 "waste_in_spells_share", "lag1_autocorr"],
    "ب۵ سبد مکانی": ["n_venues", "venue_hhi", "venue_top_share", "dorm_venue_share",
                     "venue_switch_rate"],
    "ب۶ سبد منو": ["n_dishes", "dish_hhi", "dish_top_share", "dish_entropy",
                   "mainstream_index"],
    "ب۷ چرخه‌ی عمر": ["tenure_days", "days_since_last", "n_gaps_gt14", "max_gap",
                      "volume_trend"],
    "ب۸ پاسخ به شوک": ["rho_on_bad_days", "rho_on_good_days", "shock_gap", "shock_slope"],
    "ب۹ رضایت و فراموشی": ["vote_rate", "vote_good_share", "vote_bad_share",
                           "vote_abstain_share", "forgot_card_rate"],
}


def p934_thresholds(P: pd.DataFrame) -> pd.DataFrame:
    """۹-۳-۴ جدول آستانه — مرجع قاعده‌ی الف برای کل دور."""
    header("۹-۳-۴ جدول آستانه‌ی حداقل رزرو", 2)
    t = pd.DataFrame([coverage_row(P, n) for n in THRESHOLDS])
    print(md_table(t.round(4)))

    fig, ax = figs.new(1, 2, w=5.6, h=4.0)
    ax[0].plot(t.n_min, t["سهم افراد"] * 100, "o-", color=figs.C_GREY, label="افراد")
    ax[0].plot(t.n_min, t["سهم رزرو"] * 100, "s-", color=figs.C_MAIN, label="رزرو")
    ax[0].plot(t.n_min, t["سهم هدررفت"] * 100, "^-", color=figs.C_ALT, label="هدررفت")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("آستانه‌ی حداقل رزرو")
    ax[0].set_ylabel("درصد باقی‌مانده")
    ax[0].set_title("هزینه‌ی آستانه: چه چیزی حذف می‌شود؟")
    ax[0].legend()
    ax[0].axvline(N_MIN_MAIN, color=figs.C_OK, ls="--", lw=1.2)
    figs.annotate(ax[0], f"آستانه‌ی اصلی {N_MIN_MAIN}:\n"
                         f"{t.loc[t.n_min == N_MIN_MAIN, 'سهم افراد'].iloc[0]:.1%} افراد، "
                         f"{t.loc[t.n_min == N_MIN_MAIN, 'سهم هدررفت'].iloc[0]:.1%} هدررفت",
                  loc="lower left")

    ax[1].plot(t.n_min, t["نرخ وزنی"] * 100, "o-", color=figs.C_ALT)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("آستانه‌ی حداقل رزرو")
    ax[1].set_ylabel("نرخ وزنی عدم‌دریافت (٪)")
    ax[1].set_title("نرخ با آستانه پایین می‌رود")
    figs.annotate(ax[1], "کم‌رزروها پرنرخ‌ترند\n(سازوکار F64)")
    figs.save(fig, "r9a_01_threshold_coverage", t,
              caption="آستانه‌ی ۲۰ رزرو ۱۸.۹٪ افراد را حذف می‌کند ولی فقط ۱.۹٪ هدررفت را")
    return t


def p933_shrinkage(P: pd.DataFrame) -> dict:
    """۹-۳-۳ پیشین بتا-دوجمله‌ای و اثر انقباض (قاعده‌ی پ)."""
    header("۹-۳-۳ انقباض تجربی-بیز", 2)
    meta = json.loads((DATA_INTERIM / "person_profile_v4_meta.json").read_text(encoding="utf-8"))
    a, b = meta["eb_alpha"], meta["eb_beta"]
    print(f"α = {a:.4f} · β = {b:.4f} · قدرت پیشین = {a + b:.2f} رزرو "
          f"(لنگر v2 دور ۵: ۹.۷۴)")
    print(f"میانگین پیشین = {a / (a + b):.4f} · نرخ وزنی کل = "
          f"{P.n_waste.sum() / P.n_res.sum():.4f}")

    bins = [1, 5, 10, 20, 50, 100, 250, 1000, 10 ** 9]
    P = P.assign(_bin=pd.cut(P.n_res, bins=bins, right=False))
    g = P.groupby("_bin", observed=True)
    tab = pd.DataFrame({
        "افراد": g.size(),
        "میانه rho_raw": g.rho_raw.median(),
        "میانه rho_eb": g.rho_eb.median(),
        "انحراف rho_raw": g.rho_raw.std(),
        "انحراف rho_eb": g.rho_eb.std(),
    }).reset_index()
    tab["کاهش پراکندگی"] = 1 - tab["انحراف rho_eb"] / tab["انحراف rho_raw"]
    tab["_bin"] = tab["_bin"].astype(str)
    print(md_table(tab.round(4)))

    fig, ax = figs.new(1, 2, w=5.6, h=4.0)
    s = P.sample(min(12000, len(P)), random_state=42)
    ax[0].scatter(s.n_res, s.rho_raw, s=3, alpha=0.18, color=figs.C_GREY, label="خام")
    ax[0].scatter(s.n_res, s.rho_eb, s=3, alpha=0.30, color=figs.C_MAIN, label="منقبض‌شده")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("تعداد رزرو فرد")
    ax[0].set_ylabel("نرخ عدم‌دریافت")
    ax[0].set_title("انقباض کجا کار می‌کند")
    ax[0].legend(markerscale=4)
    figs.annotate(ax[0], f"قدرت پیشین {a + b:.1f} رزرو")

    n_grid = np.unique(np.round(np.logspace(0, 3, 60)).astype(int))
    for k_frac, c, lbl in ((0.0, figs.C_OK, "فرد بی‌هدررفت"),
                           (0.5, figs.C_ALT, "فرد با نرخ خام ۰.۵"),
                           (1.0, figs.C_WARN, "فرد با نرخ خام ۱.۰")):
        ax[1].plot(n_grid, eb_shrink(n_grid * k_frac, n_grid, a, b), color=c, label=lbl)
    ax[1].axhline(a / (a + b), color=figs.C_GREY, ls=":", lw=1.2)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("تعداد رزرو فرد")
    ax[1].set_ylabel("نرخ منقبض‌شده")
    ax[1].set_title("سرنوشت سه فرد فرضی")
    ax[1].legend(fontsize=8)
    figs.save(fig, "r9a_02_eb_shrinkage", tab,
              caption=f"قدرت پیشین {a + b:.2f} رزرو — فرد با کمتر از این، بیشتر پیشین است تا داده")
    return {"alpha": a, "beta": b, "strength": a + b}


def p932_nullity(P: pd.DataFrame) -> pd.DataFrame:
    """نقشه‌ی تهی‌بودن — درس F220: فیچری که تهی است، بی‌صدا تهی است."""
    header("۹-۳-۲ نقشه‌ی تهی‌بودن ماتریس پروفایل", 2)
    rows = []
    for blk, cols in BLOCKS.items():
        for c in cols:
            if c in P.columns:
                rows.append({"بلوک": blk, "ستون": c,
                             "سهم تهی": float(P[c].isna().mean())})
    t = pd.DataFrame(rows).sort_values("سهم تهی", ascending=False)
    bad = t[t["سهم تهی"] > 0.05]
    print(f"ستون‌های با >۵٪ تهی‌بودن: {len(bad)} از {len(t)}")
    print(md_table(bad.round(4)))

    fig, ax = figs.new(w=8.4, h=6.4)
    order = t.sort_values(["بلوک", "سهم تهی"])
    colors = {b: figs.PALETTE[i % len(figs.PALETTE)] for i, b in enumerate(BLOCKS)}
    ax.barh(range(len(order)), order["سهم تهی"] * 100,
            color=[colors[b] for b in order["بلوک"]])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order["ستون"], fontsize=7)
    ax.set_xlabel("سهم تهی (٪)")
    ax.set_title("کدام فیچر پروفایل واقعاً مقدار دارد؟")
    ax.axvline(5, color=figs.C_ALT, ls="--", lw=1.2)
    handles = [__import__("matplotlib").patches.Patch(color=v, label=k) for k, v in colors.items()]
    ax.legend(handles=handles, fontsize=7, loc="lower right")
    figs.save(fig, "r9a_03_nullity_map", t,
              caption="نیمی از افراد هرگز شام نمی‌گیرند ⇒ rho_dinner برای ۵۰٪ تعریف نشده")
    return t


def p932_distributions(P: pd.DataFrame) -> pd.DataFrame:
    """شکل توزیع فیچرهای کلیدی — پایه‌ی تصمیم تبدیل در بند ۹-۷-۱."""
    header("۹-۳-۲ شکل توزیع فیچرهای کلیدی", 2)
    keys = [("n_res", "تعداد رزرو", True), ("rho_eb", "نرخ منقبض‌شده", False),
            ("tenure_days", "تِنور (روز)", False), ("burstiness", "انفجاری‌بودن", False),
            ("venue_hhi", "تمرکز سلف (HHI)", False),
            ("mainstream_index", "همراهی با جمع", False)]
    fig, axes = figs.new(2, 3, w=4.6, h=3.4)
    rows = []
    for ax, (col, lbl, logx) in zip(axes.ravel(), keys):
        v = P[col].dropna().to_numpy()
        ax.hist(np.log10(v[v > 0]) if logx else v, bins=60, color=figs.C_MAIN, alpha=0.85)
        ax.set_title(lbl)
        ax.set_xlabel("log۱۰" if logx else "")
        ax.set_ylabel("تعداد فرد")
        from scipy import stats as st
        # نسبت دهک‌های انتهایی: سنجه‌ی دامنه‌ی دینامیکی که چولگی به‌تنهایی نمی‌دهد
        q10, q90 = np.quantile(v, [0.1, 0.9])
        rows.append({"ستون": col, "میانه": float(np.median(v)),
                     "چولگی": float(st.skew(v)), "کشیدگی": float(st.kurtosis(v)),
                     "q۱۰": float(q10), "q۹۰": float(q90),
                     "نسبت q۹۰/q۱۰": float(q90 / q10) if q10 > 0 else np.nan})
        figs.annotate(ax, f"میانه={np.median(v):.3g}\nچولگی={st.skew(v):.2f}")
    t = pd.DataFrame(rows)
    print(md_table(t.round(3)))
    sk = t.loc[t["ستون"] == "n_res", "چولگی"].iloc[0]
    rr = t.loc[t["ستون"] == "n_res", "نسبت q۹۰/q۱۰"].iloc[0]
    figs.save(fig, "r9a_04_feature_distributions", t,
              caption=f"n_res: چولگی {sk:.2f} ولی نسبت q۹۰/q۱۰ برابر {rr:.0f} ⇒ "
                      "تبدیل رتبه‌ای پیش از هر روش فاصله‌محور اجباری است")
    return t


def main() -> None:
    setup()
    P = load_profile()
    header("۹-۳ ماتریس پروفایل فرد — پنجره‌ی v4")
    print(f"افراد: {len(P):,} · ستون: {P.shape[1]} · رزرو: {int(P.n_res.sum()):,} · "
          f"هدررفت: {int(P.n_waste.sum()):,} · نرخ کل: {P.n_waste.sum() / P.n_res.sum():.4f}")
    p934_thresholds(P)
    p933_shrinkage(P)
    p932_nullity(P)
    p932_distributions(P)
    header("پایان بند ۹-۳")


if __name__ == "__main__":
    main()
