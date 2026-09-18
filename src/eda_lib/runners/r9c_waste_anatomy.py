"""بند ۹-۵ — آناتومی هدررفت فردی روی پنجره‌ی v4 (بازاجرای F73/F74 و پایداری دهک).

اجرا: python -m src.eda_lib.runners.r9c_waste_anatomy
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import set_global_seed
from src.eda_lib import figs
from src.eda_lib.person_profile_v4 import build_person_year_v4
from src.eda_lib.runners._common_r9 import (
    N_MIN_MAIN, THRESHOLDS, WORST_DEFS, at_threshold, gini, header, jaccard,
    load_profile, lorenz, md_table, setup, top_share,
)

#: لنگرهای دور ۵ (پنجره‌ی قفل‌شده v2) برای ستون «عدد v2» دفتر بازآزمون.
V2_ANCHORS = {"gini_waste": 0.562, "gini_res": 0.463, "top10": 0.381, "top20": 0.577,
              "top1": 0.073, "zero_share": 0.136, "jaccard_raw_abs": 0.122}


def _boot_gini(x: np.ndarray, n_boot: int = 500, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = [gini(rng.choice(x, size=x.size, replace=True)) for _ in range(n_boot)]
    return tuple(np.quantile(vals, [0.025, 0.975]))


def p951_concentration(P: pd.DataFrame) -> pd.DataFrame:
    """۹-۵-۱ تمرکز هدررفت: لورنتس و جینی (بازآزمون F74)."""
    header("۹-۵-۱ تمرکز هدررفت — لورنتس و جینی", 2)
    S = at_threshold(P, 1)
    rows = []
    for col, lbl in (("n_waste", "هدررفت مطلق"), ("n_res", "رزرو")):
        x = S[col].to_numpy(dtype="float64")
        g = gini(x)
        lo, hi = _boot_gini(x)
        rows.append({"کمیت": lbl, "جینی": g, "CI پایین": lo, "CI بالا": hi,
                     "سهم ۱٪ برتر": top_share(S, col, 0.01, col),
                     "سهم ۱۰٪ برتر": top_share(S, col, 0.10, col),
                     "سهم ۲۰٪ برتر": top_share(S, col, 0.20, col)})
    t = pd.DataFrame(rows)
    print(md_table(t.round(4)))
    zero = float((S.n_waste == 0).mean())
    print(f"\nافراد با صفر هدررفت: {zero:.4f} (لنگر v2: {V2_ANCHORS['zero_share']})")
    print(f"لنگر v2 — جینی هدررفت {V2_ANCHORS['gini_waste']} · جینی رزرو {V2_ANCHORS['gini_res']} · "
          f"۱۰٪ برتر {V2_ANCHORS['top10']}")

    fig, ax = figs.new(1, 2, w=5.6, h=4.2)
    for col, lbl, c in (("n_waste", "هدررفت", figs.C_ALT), ("n_res", "رزرو", figs.C_MAIN)):
        pop, cum = lorenz(S[col].to_numpy(dtype="float64"))
        ax[0].plot(pop * 100, cum * 100, color=c, lw=2,
                   label=f"{lbl} (جینی={gini(S[col].to_numpy(dtype='float64')):.3f})")
    ax[0].plot([0, 100], [0, 100], color=figs.C_GREY, ls=":", lw=1.2, label="برابری کامل")
    ax[0].axvline(10, color=figs.C_OK, ls="--", lw=1.1)
    ax[0].set_xlabel("درصد افراد (از بدترین)")
    ax[0].set_ylabel("درصد تجمعی")
    ax[0].set_title("منحنی تمرکز — که چقدر در دست چه کسی است")
    ax[0].legend(fontsize=8, loc="lower right")
    t10 = top_share(S, "n_waste", 0.10)
    figs.annotate(ax[0], f"۱۰٪ بدترین = {t10:.1%} هدررفت\n(لنگر v2: {V2_ANCHORS['top10']:.1%})",
                  loc="upper left")

    fr = np.linspace(0.01, 1.0, 60)
    for col, lbl, c in (("n_waste", "هدف‌گیری بر هدررفت مطلق", figs.C_ALT),
                        ("rho_eb", "هدف‌گیری بر نرخ منقبض‌شده", figs.C_MAIN),
                        ("rho_raw", "هدف‌گیری بر نرخ خام", figs.C_WARN)):
        ax[1].plot(fr * 100, [top_share(S, col, f) * 100 for f in fr], color=c, lw=1.8, label=lbl)
    ax[1].plot([0, 100], [0, 100], color=figs.C_GREY, ls=":", lw=1.2, label="تصادفی")
    ax[1].set_xlabel("درصد افراد هدف‌گیری‌شده")
    ax[1].set_ylabel("درصد هدررفت پوشش‌داده‌شده")
    ax[1].set_title("سه تعریف، سه منحنی پوشش")
    ax[1].legend(fontsize=8, loc="lower right")
    figs.save(fig, "r9c_01_lorenz_concentration", t,
              caption=f"۱۰٪ بدترین {t10:.1%} هدررفت دارند — جینی هدررفت "
                      f"{gini(S.n_waste.to_numpy(dtype='float64')):.3f}")
    return t


def p952_three_definitions(P: pd.DataFrame, n_min: int = 1) -> pd.DataFrame:
    """۹-۵-۲ سه تعریف «بدترین ۱۰٪» چقدر هم‌پوشانی دارند؟ (بازآزمون F73)."""
    header(f"۹-۵-۲ سه تعریف «بدترین ۱۰٪» (آستانه n_min={n_min})", 2)
    S = at_threshold(P, n_min)
    k = int(round(0.10 * len(S)))
    lists = {c: set(S.nlargest(k, c).PersonId) for c in WORST_DEFS}

    jm = pd.DataFrame(index=list(WORST_DEFS.values()), columns=list(WORST_DEFS.values()),
                      dtype="float64")
    for a, la in WORST_DEFS.items():
        for b, lb in WORST_DEFS.items():
            jm.loc[la, lb] = jaccard(lists[a], lists[b])
    print("ماتریس ژاکارِ سه تعریف:")
    print(md_table(jm.reset_index().rename(columns={"index": "تعریف"}).round(3)))

    rows = []
    for c, lbl in WORST_DEFS.items():
        sub = S[S.PersonId.isin(lists[c])]
        rows.append({"تعریف": lbl, "افراد": len(sub),
                     "سهم از کل هدررفت": sub.n_waste.sum() / S.n_waste.sum(),
                     "میانه رزرو": sub.n_res.median(),
                     "میانه نرخ خام": sub.rho_raw.median(),
                     "میانه نرخ EB": sub.rho_eb.median(),
                     "سهم ≤۵ رزرو": float((sub.n_res <= 5).mean())})
    t = pd.DataFrame(rows)
    print("\nپروفایل هر فهرست:")
    print(md_table(t.round(4)))
    print(f"\nلنگر v2 (F73): ژاکارِ نرخ‌خام∩هدررفت‌مطلق = {V2_ANCHORS['jaccard_raw_abs']} · "
          "سهم از هدررفت: نرخ خام ۱۶.۲٪ · EB ۲۳.۷٪ · مطلق ۳۸.۱٪")

    fig, ax = figs.new(1, 2, w=5.6, h=4.2)
    im = ax[0].imshow(jm.to_numpy(dtype="float64"), cmap="Blues", vmin=0, vmax=1)
    ax[0].set_xticks(range(3), jm.columns, fontsize=8, rotation=20)
    ax[0].set_yticks(range(3), jm.index, fontsize=8)
    for i in range(3):
        for j in range(3):
            v = jm.iloc[i, j]
            ax[0].text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=10,
                       color="white" if v > 0.5 else "black")
    ax[0].set_title("ژاکارِ: سه فهرست چقدر یکی‌اند؟")
    ax[0].grid(False)
    fig.colorbar(im, ax=ax[0], fraction=0.046)

    x = np.arange(len(t))
    ax[1].bar(x - 0.2, t["سهم از کل هدررفت"] * 100, 0.4, color=figs.C_ALT, label="سهم از هدررفت (٪)")
    ax2 = ax[1].twinx()
    ax2.bar(x + 0.2, t["میانه رزرو"], 0.4, color=figs.C_MAIN, label="میانه رزرو")
    ax2.set_yscale("log")
    ax[1].set_xticks(x, t["تعریف"], fontsize=8, rotation=12)
    ax[1].set_ylabel("سهم از کل هدررفت (٪)")
    ax2.set_ylabel("میانه‌ی تعداد رزرو (لگاریتمی)")
    ax[1].set_title("سه فهرست، سه جامعه")
    ax2.grid(False)
    figs.save(fig, "r9c_02_three_definitions", t,
              caption=f"ژاکارِ نرخ خام با هدررفت مطلق = {jm.iloc[0, 2]:.3f} — "
                      "سه تعریف سه جامعه‌ی تقریباً مجزا می‌دهند")
    return t


def p953_decile_persistence(n_min: int = N_MIN_MAIN) -> pd.DataFrame:
    """⭐ ۹-۵-۳ آیا پرهدررفت‌بودن سال بعد هم ادامه دارد؟ (هرگز محاسبه نشده)."""
    header(f"۹-۵-۳ پایداری دهک بین سال‌ها (آستانه n_min={n_min} در **هر دو** سال)", 2)
    Y = build_person_year_v4()
    years = list(Y.year_block.cat.categories) if hasattr(Y.year_block, "cat") \
        else sorted(Y.year_block.unique())

    out_rows, mats = [], {}
    for y0, y1 in zip(years[:-1], years[1:]):
        a = Y[(Y.year_block == y0) & (Y.n_res >= n_min)].set_index("PersonId")
        b = Y[(Y.year_block == y1) & (Y.n_res >= n_min)].set_index("PersonId")
        both = a.index.intersection(b.index)
        if len(both) < 200:
            continue
        for col, lbl in (("rho_eb", "نرخ منقبض‌شده"), ("n_waste", "هدررفت مطلق")):
            d0 = pd.qcut(a.loc[both, col].rank(method="first"), 10, labels=False)
            d1 = pd.qcut(b.loc[both, col].rank(method="first"), 10, labels=False)
            m = pd.crosstab(d0, d1, normalize="index")
            mats[(y0, y1, col)] = m
            spearman = float(pd.Series(d0).corr(pd.Series(d1), method="spearman"))
            out_rows.append({
                "جفت سال": f"{y0} → {y1}", "تعریف": lbl, "افراد مشترک": len(both),
                "ماندن در دهک ۱۰": float(m.iloc[9, 9]),
                "ماندن در دهک ۱۰ یا ۹": float(m.iloc[9, 8:].sum()),
                "سقوط به نیمه‌ی پایین": float(m.iloc[9, :5].sum()),
                "اسپیرمن دهک": spearman,
            })
    t = pd.DataFrame(out_rows)
    print(md_table(t.round(4)))
    print("\nمرجع مقایسه: تصادفی کامل ⇒ ماندن در دهک ۱۰ = ۰.۱۰۰ · پایداری کامل ⇒ ۱.۰۰۰")
    print("F46/F193 همبستگی **نرخ** را ۰.۵۰–۰.۶۳ گزارش کرده‌اند؛ این جدول همبستگی "
          "**عضویت در فهرست** را می‌سنجد و آن دو یکی نیستند.")

    keys = [k for k in mats if k[2] == "rho_eb"]
    fig, axes = figs.new(1, max(len(keys), 1), w=5.0, h=4.4, squeeze=False)
    for ax, k in zip(axes.ravel(), keys):
        m = mats[k]
        im = ax.imshow(m.to_numpy(), cmap="Blues", vmin=0, vmax=float(m.to_numpy().max()))
        ax.set_xlabel(f"دهک در {k[1]}")
        ax.set_ylabel(f"دهک در {k[0]}")
        ax.set_title(f"گذار دهک (نرخ EB)\n{k[0]} → {k[1]}", fontsize=10)
        ax.set_xticks(range(10), [str(i + 1) for i in range(10)], fontsize=7)
        ax.set_yticks(range(10), [str(i + 1) for i in range(10)], fontsize=7)
        ax.grid(False)
        ax.text(9, 9, f"{m.iloc[9, 9]:.2f}", ha="center", va="center", fontsize=9,
                color="white" if m.iloc[9, 9] > m.to_numpy().max() * 0.6 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    figs.save(fig, "r9c_03_decile_transition", t,
              caption="احتمال ماندن در بدترین دهک سال بعد — عددی که هر سیاست هدف‌گیری سالانه به آن وابسته است")
    return t


def p954_threshold_sensitivity(P: pd.DataFrame) -> pd.DataFrame:
    """۹-۵-۴ سهم ۱۰٪ برتر با آستانه چقدر جابه‌جا می‌شود؟ (قاعده‌ی الف)."""
    header("۹-۵-۴ حساسیت سهم ۱۰٪ برتر به آستانه", 2)
    rows = []
    for n in THRESHOLDS:
        S = at_threshold(P, n)
        r = {"n_min": n, "افراد": len(S)}
        for c, lbl in WORST_DEFS.items():
            r[lbl] = top_share(S, c, 0.10)
        r["جینی هدررفت"] = gini(S.n_waste.to_numpy(dtype="float64"))
        rows.append(r)
    t = pd.DataFrame(rows)
    print(md_table(t.round(4)))

    fig, ax = figs.new(w=7.4, h=4.4)
    for c, lbl in WORST_DEFS.items():
        ax.plot(t.n_min, t[lbl] * 100, "o-", label=lbl)
    ax.plot(t.n_min, t["جینی هدررفت"] * 100, "s--", color=figs.C_GREY, label="جینی هدررفت ×۱۰۰")
    ax.set_xscale("log")
    ax.set_xlabel("آستانه‌ی حداقل رزرو")
    ax.set_ylabel("سهم هدررفتِ ۱۰٪ برتر (٪)")
    ax.set_title("همان آماره، شش آستانه — قاعده‌ی الف دور ۹")
    ax.legend(fontsize=8)
    rng = t[list(WORST_DEFS.values())].max().max() - t[list(WORST_DEFS.values())].min().min()
    figs.annotate(ax, f"دامنه‌ی جابه‌جایی: {rng:.1%} واحد")
    figs.save(fig, "r9c_04_threshold_sensitivity", t,
              caption="هیچ «سهم ۱۰٪ برتر» بدون ذکر آستانه قابل‌نقل نیست")
    return t


def p955_gini_by_year(n_min: int = N_MIN_MAIN) -> pd.DataFrame:
    """۹-۵-۵ آیا تمرکز فردی بین سال‌ها تغییر کرده؟"""
    header(f"۹-۵-۵ تمرکز هدررفت به تفکیک سال (آستانه n_min={n_min})", 2)
    Y = build_person_year_v4()
    rows = []
    for yb, sub in Y[Y.n_res >= n_min].groupby("year_block", observed=True):
        x = sub.n_waste.to_numpy(dtype="float64")
        lo, hi = _boot_gini(x, n_boot=300)
        rows.append({"سال": str(yb), "افراد": len(sub),
                     "نرخ وزنی": sub.n_waste.sum() / sub.n_res.sum(),
                     "جینی هدررفت": gini(x), "CI پایین": lo, "CI بالا": hi,
                     "سهم ۱۰٪ برتر": top_share(sub, "n_waste", 0.10)})
    t = pd.DataFrame(rows)
    print(md_table(t.round(4)))

    fig, ax = figs.new(w=7.0, h=4.2)
    x = np.arange(len(t))
    ax.errorbar(x, t["جینی هدررفت"], yerr=[t["جینی هدررفت"] - t["CI پایین"],
                                           t["CI بالا"] - t["جینی هدررفت"]],
                fmt="o-", color=figs.C_MAIN, capsize=4, label="جینی هدررفت")
    ax.plot(x, t["سهم ۱۰٪ برتر"], "s--", color=figs.C_ALT, label="سهم ۱۰٪ برتر")
    ax.set_xticks(x, t["سال"])
    ax.set_ylabel("مقدار")
    ax.set_title("تمرکز هدررفت در سه سال")
    ax.legend(fontsize=9)
    figs.save(fig, "r9c_05_gini_by_year", t,
              caption="آیا نابرابری هدررفت در حال تغییر است یا ساختاری ثابت دارد؟")
    return t


def main() -> None:
    set_global_seed()
    setup()
    P = load_profile()
    header("۹-۵ آناتومی هدررفت فردی — پنجره‌ی v4")
    print(f"افراد {len(P):,} · رزرو {int(P.n_res.sum()):,} · هدررفت {int(P.n_waste.sum()):,}")
    p951_concentration(P)
    p952_three_definitions(P, n_min=1)
    p952_three_definitions(P, n_min=N_MIN_MAIN)
    p953_decile_persistence()
    p954_threshold_sensitivity(P)
    p955_gini_by_year()
    header("پایان بند ۹-۵")


if __name__ == "__main__":
    main()
