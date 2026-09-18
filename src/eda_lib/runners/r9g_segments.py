"""بند ۹-۹ — بخش‌بندی سیاستی روی محورهای تصمیم، و پروفایل هر بخش.

اجرا: python -m src.eda_lib.runners.r9g_segments

## چرا «برش اعلام‌شده» و نه «خوشه‌ی کشف‌شده»

بند ۹-۷ نشان داد فضای رفتاری پیوسته است: هر شش نما $k=۲$ می‌دهند، سیلوئت نمای
جامع ۰.۲۶۵ است، و در نماهای «اندازه» چهار الگوریتم توافق ندارند (ARI میانه
۰.۳۵–۰.۴۴). در چنین فضایی، خوشه‌ی کشف‌شده **برش دلخواهی** است که وانمود می‌کند
مرزی را یافته.

پس بخش‌بندی این بند از دو محوری ساخته می‌شود که **متغیر تصمیم پروژه مستقیماً
رویشان می‌نشیند**: هدررفت هر فرد دقیقاً $\\rho \\times n$ است. برش روی سه‌یک‌های
هر محور، نُه بخش می‌دهد که هر کدام تعریف صریح، بازتولیدپذیر و قابل‌توضیح برای
ذی‌نفع دارند — و هیچ ادعای «کشف» با خود حمل نمی‌کنند.

نماهای «سبک» بند ۹-۷ (که ساختار پایدار داشتند) به‌عنوان **توصیف‌گر** هر بخش
می‌آیند، نه تعریفش.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import set_global_seed
from src.eda_lib import figs
from src.eda_lib.person_profile_v4 import build_person_year_v4
from src.eda_lib.runners._common_r9 import (
    N_MIN_MAIN, at_threshold, header, load_profile, md_table, setup,
)

VOL_LABELS = ["کم‌مصرف", "میان‌مصرف", "پرمصرف"]
RATE_LABELS = ["کم‌نرخ", "میان‌نرخ", "پرنرخ"]


def assign_grid(df: pd.DataFrame, vol_col: str = "n_res",
                rate_col: str = "rho_eb") -> pd.DataFrame:
    """برش سه‌یکی روی دو محور تصمیم. مرزها از همین جامعه می‌آیند و گزارش می‌شوند."""
    out = df.copy(deep=False)
    out["vol_tier"] = pd.qcut(out[vol_col].rank(method="first"), 3, labels=VOL_LABELS)
    out["rate_tier"] = pd.qcut(out[rate_col].rank(method="first"), 3, labels=RATE_LABELS)
    out["segment"] = (out.vol_tier.astype(str) + " × " + out.rate_tier.astype(str))
    return out


def p991_segment_cards(P: pd.DataFrame) -> pd.DataFrame:
    """۹-۹-۱ کارت پروفایل هر بخش — اندازه، سهم هدررفت، و مشخصه‌های متمایزکننده."""
    header(f"۹-۹-۱ نُه بخش روی محورهای تصمیم (آستانه n_min={N_MIN_MAIN})", 2)
    S = assign_grid(at_threshold(P, N_MIN_MAIN))
    tot_w, tot_r = S.n_waste.sum(), S.n_res.sum()

    g = S.groupby(["vol_tier", "rate_tier"], observed=True)
    t = pd.DataFrame({
        "افراد": g.size(),
        "سهم افراد": g.size() / len(S),
        "سهم رزرو": g.n_res.sum() / tot_r,
        "سهم هدررفت": g.n_waste.sum() / tot_w,
        "میانه رزرو": g.n_res.median(),
        "میانه نرخ EB": g.rho_eb.median(),
        "هدررفت سرانه": g.n_waste.mean(),
    }).reset_index()
    t["بازده هدف‌گیری"] = t["سهم هدررفت"] / t["سهم افراد"]
    t = t.sort_values("سهم هدررفت", ascending=False)
    print(md_table(t.round(4)))
    top = t.iloc[0]
    print(f"\n▸ پرهدررفت‌ترین بخش: «{top.vol_tier} × {top.rate_tier}» — "
          f"{top['سهم افراد']:.1%} افراد، {top['سهم هدررفت']:.1%} هدررفت "
          f"(بازده {top['بازده هدف‌گیری']:.2f}×)")

    fig, axes = figs.new(1, 2, w=5.6, h=4.6)
    piv = t.pivot(index="rate_tier", columns="vol_tier", values="سهم هدررفت")
    piv = piv.reindex(index=RATE_LABELS[::-1], columns=VOL_LABELS)
    im = axes[0].imshow(piv.to_numpy(dtype="float64") * 100, cmap="Reds")
    axes[0].set_xticks(range(3), VOL_LABELS, fontsize=9)
    axes[0].set_yticks(range(3), RATE_LABELS[::-1], fontsize=9)
    for i in range(3):
        for j in range(3):
            v = piv.iloc[i, j] * 100
            axes[0].text(j, i, f"{v:.1f}٪", ha="center", va="center", fontsize=11,
                         color="white" if v > piv.to_numpy().max() * 55 else "black")
    axes[0].set_title("سهم هر بخش از کل هدررفت")
    axes[0].grid(False)
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    pe = t.pivot(index="rate_tier", columns="vol_tier", values="بازده هدف‌گیری")
    pe = pe.reindex(index=RATE_LABELS[::-1], columns=VOL_LABELS)
    im2 = axes[1].imshow(pe.to_numpy(dtype="float64"), cmap="Blues")
    axes[1].set_xticks(range(3), VOL_LABELS, fontsize=9)
    axes[1].set_yticks(range(3), RATE_LABELS[::-1], fontsize=9)
    for i in range(3):
        for j in range(3):
            axes[1].text(j, i, f"{pe.iloc[i, j]:.2f}×", ha="center", va="center",
                         fontsize=11)
    axes[1].set_title("بازده هدف‌گیری (سهم هدررفت ÷ سهم افراد)")
    axes[1].grid(False)
    fig.colorbar(im2, ax=axes[1], fraction=0.046)
    figs.save(fig, "r9g_01_segment_grid", t,
              caption="نُه بخش با تعریف صریح — هر خانه می‌گوید چند درصد افراد چند درصد هدررفت می‌سازند")
    return t


def p992_segment_traits(P: pd.DataFrame) -> pd.DataFrame:
    """۹-۹-۲ مشخصه‌های رفتاری و جمعیتی هر بخش (توصیفی، نه علّی)."""
    header("۹-۹-۲ مشخصه‌های متمایزکننده‌ی هر بخش", 2)
    S = assign_grid(at_threshold(P, N_MIN_MAIN))
    traits = ["dow_entropy", "burstiness", "venue_hhi", "venue_switch_rate",
              "dish_entropy", "mainstream_index", "shock_gap", "tenure_days",
              "volume_trend", "vote_rate", "forgot_card_rate", "lunch_share",
              "waste_in_spells_share"]
    g = S.groupby("segment", observed=True)
    m = g[traits].mean()
    z = (m - S[traits].mean()) / S[traits].std()   # انحراف از میانگین کل، برحسب SD
    demo = pd.DataFrame({
        "سهم خوابگاهی": g.demo_is_dorm.mean(),
        "سهم مرد": g.demo_gender.apply(lambda s: float((s.astype(str).str.strip() == "مرد").mean())),
        "سهم شبانه": g.demo_session.apply(
            lambda s: float(s.astype(str).str.contains("نوبت دوم").mean())),
    })
    out = z.round(3).join(demo.round(3))
    print("انحراف هر مشخصه از میانگین کل، برحسب انحراف معیار (|z|>۰.۳ پررنگ است):")
    print(md_table(out.reset_index().round(3)))

    fig, ax = figs.new(w=8.6, h=5.2)
    order = [f"{v} × {r}" for v in VOL_LABELS for r in RATE_LABELS]
    zz = z.reindex(order)
    im = ax.imshow(zz.to_numpy(dtype="float64"), cmap="RdBu_r", vmin=-1.0, vmax=1.0,
                   aspect="auto")
    ax.set_xticks(range(len(traits)), traits, fontsize=6, rotation=40, ha="right")
    ax.set_yticks(range(len(order)), order, fontsize=7)
    ax.set_title("امضای رفتاری هر بخش (انحراف از میانگین کل، برحسب SD)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.03)
    figs.save(fig, "r9g_02_segment_traits", out.reset_index(),
              caption="بخش‌ها روی محور اندازه تعریف شده‌اند؛ این شکل می‌گوید در محور سبک چه فرقی دارند")
    return out


def p994_segment_flow(n_min: int = N_MIN_MAIN) -> pd.DataFrame:
    """۹-۹-۴ جریان بین بخش‌ها: فرد سال بعد کجا می‌رود؟"""
    header("۹-۹-۴ ماتریس گذار بین نُه بخش", 2)
    Y = build_person_year_v4()
    years = list(Y.year_block.cat.categories) if hasattr(Y.year_block, "cat") \
        else sorted(Y.year_block.unique())
    order = [f"{v} × {r}" for v in VOL_LABELS for r in RATE_LABELS]

    rows, mats = [], {}
    for y0, y1 in zip(years[:-1], years[1:]):
        a = assign_grid(Y[(Y.year_block == y0) & (Y.n_res >= n_min)]).set_index("PersonId")
        b = assign_grid(Y[(Y.year_block == y1) & (Y.n_res >= n_min)]).set_index("PersonId")
        both = a.index.intersection(b.index)
        m = pd.crosstab(a.loc[both, "segment"], b.loc[both, "segment"], normalize="index")
        m = m.reindex(index=order, columns=order).fillna(0.0)
        mats[f"{y0} → {y1}"] = m
        rows.append({"جفت سال": f"{y0} → {y1}", "افراد مشترک": len(both),
                     "ماندن در همان بخش": float(np.diag(m.to_numpy()).mean()),
                     "ماندن در «پرمصرف × پرنرخ»":
                         float(m.loc["پرمصرف × پرنرخ", "پرمصرف × پرنرخ"]),
                     "ماندن در همان سه‌یکِ حجم": _tier_persistence(a, b, both, "vol_tier"),
                     "ماندن در همان سه‌یکِ نرخ": _tier_persistence(a, b, both, "rate_tier")})
    t = pd.DataFrame(rows)
    print(md_table(t.round(4)))
    print("\nمرجع: تصادفی کامل ⇒ ۰.۱۱۱ برای بخش، ۰.۳۳۳ برای سه‌یک")

    fig, axes = figs.new(1, len(mats), w=5.2, h=4.8, squeeze=False)
    for ax, (lbl, m) in zip(axes.ravel(), mats.items()):
        im = ax.imshow(m.to_numpy(dtype="float64"), cmap="Blues", vmin=0, vmax=0.6)
        ax.set_xticks(range(9), order, fontsize=5, rotation=55, ha="right")
        ax.set_yticks(range(9), order, fontsize=5)
        ax.set_title(lbl, fontsize=9)
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046)
    figs.save(fig, "r9g_03_segment_flow", t,
              caption="پایداری بخش سال‌به‌سال — پیش‌شرط هر سیاست سالانه")
    return t


def _tier_persistence(a, b, both, col) -> float:
    return float((a.loc[both, col].astype(str) == b.loc[both, col].astype(str)).mean())


def main() -> None:
    set_global_seed()
    setup()
    P = load_profile()
    header("۹-۹ بخش‌بندی سیاستی و پروفایل بخش‌ها")
    p991_segment_cards(P)
    p992_segment_traits(P)
    p994_segment_flow()
    header("پایان بند ۹-۹")


if __name__ == "__main__":
    main()
