"""بند ۸.۵ WBS فاز ۸ — شبیه‌سازی عملیاتی (Backtest کسب‌وکاری) روی Test قفل‌شده.

سناریو: «اگر قهرمان رسمی (`lightgbm_quantile`) در کل پنجره‌ی Test اجرا می‌شد، چه
اتفاقی می‌افتاد؟» روی شبکه‌ی کامل τ (`TAU_GRID`، بند ۶.۵) — هایپرپارامتر همان بهترین
S2 (تنظیم‌شده روی τ=۰.۲۰) طبق یافته‌ی ۲۴ فاز ۷ که تعمیم بدون بازتنظیم را تأیید کرد.

**واحد ریالی:** فقط $C_o$≈۱۲۰ هزار تومان/پرس (بند ۲-۲ project-definition.md) — عددی
مستقل تخمین‌زده‌شده. $C_u$ (هزینه‌ی کمبود) هنوز تصمیم سیاستی تثبیت‌نشده است (همان سند)،
پس این نمودار عمداً محور X را **نرخ** کمبود می‌گذارد نه هزینه‌ی ریالی‌اش — دقیقاً طبق
مشخصات بند ۸.۵ WBS.

⚠️ این محاسبه روی یک پنجره‌ی ۲۵روزه است، **نه سالانه** — تعمیم سالانه با عدم‌قطعیت
بوت‌استرپ کار بند ۱۰.۴ (فاز ۱۰) است.

اجرا: ``python -m src.models.run_phase8_business_backtest``
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.baselines import TAU_GRID, operational_metrics
from src.config import FIGURES_DIR, REPORTS_DIR
from src.eda_lib.figio import save_fig
from src.models.card_writer import load_s2_result
from src.models.run_phase8_holdout_eval import load_holdout
from src.viz_fa import fa
from src.viz_fa import setup as viz_setup

OUT_DIR = REPORTS_DIR / "phase8"
FIG_DIR = FIGURES_DIR / "phase8"
CHAMPION = "lightgbm_quantile"
COST_PER_PORTION_TOMAN = 120_000


def main() -> None:
    viz_setup()
    train, test, fold = load_holdout()
    result = load_s2_result(CHAMPION, "F02")
    hp = result["best_hyperparams"]
    import importlib
    fn = importlib.import_module("src.models.families.f02_tree").MODELS[CHAMPION]

    rows = []
    for tau in TAU_GRID:
        pred = np.clip(np.asarray(fn(train, test, tau, **hp), dtype=float), 0.0, 1.0)
        m = operational_metrics(test, pred, tau)
        b0 = operational_metrics(test, np.zeros(len(test)), tau)  # رهگیری مرجع B0 (پخت کامل رزرو)
        portions_saved = b0["surplus_portions"] - m["surplus_portions"]
        rial_saved = portions_saved * COST_PER_PORTION_TOMAN
        n_shortage_cells = int(round(m["shortage_rate"] * m["n"]))
        rows.append({
            "tau": tau, "pinball": m["pinball"], "shortage_rate": m["shortage_rate"],
            "n_shortage_cells": n_shortage_cells, "shortage_portions": m["shortage_portions"],
            "max_shortage_cell": None,  # پایین‌تر پر می‌شود
            "surplus_portions": m["surplus_portions"], "portions_saved_vs_B0": portions_saved,
            "rial_saved_toman": rial_saved, "waste_reduction_pct": m["waste_reduction_pct"],
            "coverage": m["coverage"], "coverage_gap": m["coverage_gap"],
        })

    df = pd.DataFrame(rows)

    for i, tau in enumerate(TAU_GRID):
        pred = np.clip(np.asarray(fn(train, test, tau, **hp), dtype=float), 0.0, 1.0)
        cook = np.ceil(test["Res"].to_numpy() * (1.0 - pred))
        shortfall = np.maximum(test["Recv"].to_numpy() - cook, 0.0)
        df.loc[i, "max_shortage_cell"] = float(shortfall.max())

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df["shortage_rate"] * 100, df["rial_saved_toman"] / 1e9, marker="o", color="#4C72B0")
    for _, r in df.iterrows():
        ax.annotate(f"τ={r['tau']:.2f}", (r["shortage_rate"] * 100, r["rial_saved_toman"] / 1e9),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel(fa("نرخ کمبود (٪)"))
    ax.set_ylabel(fa("صرفه‌جویی ریالی (میلیارد تومان، پنجره‌ی ۲۵ روزه)"))
    ax.set_title(fa(f"منحنی trade-off — {CHAMPION} روی Test قفل‌شده"))
    ax.grid(alpha=0.3)
    save_fig(fig, "8.5_tradeoff_curve.png", FIG_DIR)
    plt.close(fig)

    op_tau = 0.20
    op_row = df.loc[np.isclose(df["tau"], op_tau)].iloc[0]

    lines = [
        "# بند ۸.۵ — شبیه‌سازی عملیاتی (Backtest کسب‌وکاری) روی Test قفل‌شده",
        "",
        f"> مدل: `{CHAMPION}`. پنجره: {fold.test_start.date()} تا {fold.test_end.date()} (۲۵ روز، ۱٬۵۲۶ ردیف). "
        f"$C_o$={COST_PER_PORTION_TOMAN:,} تومان/پرس (project-definition.md بند ۲-۲). "
        "$C_u$ تثبیت‌نشده — محور X عمداً نرخ کمبود است، نه هزینه‌ی ریالی‌اش.",
        "",
        "| τ | pinball | نرخ کمبود | تعداد سلول کمبود | بیشترین کمبود (پرس) | پرس صرفه‌جویی‌شده | صرفه‌جویی (میلیارد تومان) | هدررفت‌کاهی | پوشش | شکاف پوشش |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['tau']:.2f} | {r['pinball']:.5f} | {r['shortage_rate']:.1%} | {int(r['n_shortage_cells'])} | "
            f"{r['max_shortage_cell']:.0f} | {r['portions_saved_vs_B0']:.0f} | {r['rial_saved_toman']/1e9:.2f} | "
            f"{r['waste_reduction_pct']:.1%} | {r['coverage']:.1%} | {r['coverage_gap']:+.1%} |"
        )
    lines += [
        "",
        f"**نقطه‌ی عملیاتی پروژه (τ=0.20):** نرخ کمبود {op_row['shortage_rate']:.1%}، "
        f"{op_row['portions_saved_vs_B0']:.0f} پرس صرفه‌جویی‌شده (~{op_row['rial_saved_toman']/1e9:.2f} میلیارد "
        "تومان) نسبت به B0 (پخت کامل رزرو) در همین ۲۵ روز.",
        "",
        "⚠️ این اعداد برای یک پنجره‌ی ۲۵روزه‌ی خاص‌اند (شامل رژیم غیرعادی بازگشت از رمضان + "
        "سوگواری ملی، ردیف ۴۹ decision_log) — **تعمیم سالانه با فاصله‌ی اطمینان بوت‌استرپ "
        "کار بند ۱۰.۴ است، نه اینجا.**",
        "",
        f"نمودار trade-off: `reports/figures/phase8/8.5_tradeoff_curve.png`",
    ]

    report = "\n".join(lines)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "8.5_business_backtest.md").write_text(report + "\n")
    df.to_csv(OUT_DIR / "8.5_tradeoff_table.csv", index=False)
    print(report)
    print(f"\nذخیره شد در {OUT_DIR}/8.5_business_backtest.md")


if __name__ == "__main__":
    main()
