"""بند ۸.۸ WBS فاز ۸ — کالیبراسیون کوانتایل روی Test قفل‌شده.

نمودار قابلیت اطمینان (τ∈{۰.۰۵..۰.۹۵}) برای قهرمان رسمی (`lightgbm_quantile`، بدون
کالیبراسیون) روی همان مرز holdout بند ۸.۱. بند ۸.۳/۸.۱ از قبل نشان داده بودند این
پنجره در τ=۰.۲۰ شکاف پوشش بزرگ دارد (+۱۳.۶٪) چون بر خلاف foldهای رسمی فاز ۷، ACI
رویش اعمال نشده بود — اینجا دقیقاً همان ACI (`src/models/conformal.py`، یافته‌ی ۲۲
فاز ۷) روی نقطه‌ی عملیاتی اعمال و پیش/پس مقایسه می‌شود.

اجرا: ``python -m src.models.run_phase8_calibration``
"""

import importlib

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.baselines import operational_metrics, pinball_loss
from src.config import FIGURES_DIR, REPORTS_DIR
from src.eda_lib.figio import save_fig
from src.models.axes import TUNING_TAU
from src.models.card_writer import load_s2_result
from src.models.conformal import aci_predict
from src.models.run_phase8_holdout_eval import load_holdout
from src.viz_fa import fa
from src.viz_fa import setup as viz_setup

OUT_DIR = REPORTS_DIR / "phase8"
FIG_DIR = FIGURES_DIR / "phase8"
CHAMPION = "lightgbm_quantile"
RELIABILITY_TAUS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
OP_TAU = TUNING_TAU


def main() -> None:
    viz_setup()
    train, test, fold = load_holdout()
    result = load_s2_result(CHAMPION, "F02")
    hp = result["best_hyperparams"]
    fn = importlib.import_module("src.models.families.f02_tree").MODELS[CHAMPION]

    rows = []
    for tau in RELIABILITY_TAUS:
        pred = np.clip(np.asarray(fn(train, test, tau, **hp), dtype=float), 0.0, 1.0)
        coverage = float((test["rho"].to_numpy() <= pred).mean())
        rows.append({"tau": tau, "coverage": coverage, "gap": coverage - tau})
    rel_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label=fa("خط ایده‌آل"))
    ax.plot(rel_df["tau"], rel_df["coverage"], marker="o", color="#4C72B0", label=fa("قهرمان (بدون کالیبراسیون)"))
    ax.set_xlabel(fa("τ اسمی"))
    ax.set_ylabel(fa("پوشش تجربی (نسبت مشاهدات زیر پیش‌بینی)"))
    ax.set_title(fa(f"نمودار قابلیت اطمینان — {CHAMPION} روی Test قفل‌شده"))
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    save_fig(fig, "8.8_reliability_diagram.png", FIG_DIR)
    plt.close(fig)

    max_gap_row = rel_df.loc[rel_df["gap"].abs().idxmax()]

    # ACI روی نقطه‌ی عملیاتی τ=0.20
    pred_before = np.clip(np.asarray(fn(train, test, OP_TAU, **hp), dtype=float), 0.0, 1.0)
    m_before = operational_metrics(test, pred_before, OP_TAU)

    pred_after, corr_path = aci_predict(fn, train, test, OP_TAU, hp)
    m_after = operational_metrics(test, pred_after, OP_TAU)

    lines = [
        "# بند ۸.۸ — کالیبراسیون کوانتایل روی Test قفل‌شده",
        "",
        f"> مدل: `{CHAMPION}`. داده: پنجره‌ی Test بند ۸.۱.",
        "",
        "## نمودار قابلیت اطمینان (بدون کالیبراسیون)",
        "",
        "| τ اسمی | پوشش تجربی | شکاف |",
        "|---|---|---|",
    ]
    for _, r in rel_df.iterrows():
        lines.append(f"| {r['tau']:.2f} | {r['coverage']:.1%} | {r['gap']:+.1%} |")
    lines += [
        "",
        f"بیشترین شکاف: τ={max_gap_row['tau']:.2f} (شکاف {max_gap_row['gap']:+.1%}) — "
        "**سیستماتیک بیش‌پوشش‌دهنده** (مدل محافظه‌کارتر از حد اسمی، سازگار با بند ۸.۱/۸.۳).",
        f"نمودار: `reports/figures/phase8/8.8_reliability_diagram.png`",
        "",
        "## بازکالیبراسیون با ACI (نقطه‌ی عملیاتی τ=0.20)",
        "",
        "| حالت | پوشش | شکاف پوشش | pinball |",
        "|---|---|---|---|",
        f"| قبل از ACI | {m_before['coverage']:.1%} | {m_before['coverage_gap']:+.1%} | {m_before['pinball']:.5f} |",
        f"| بعد از ACI | {m_after['coverage']:.1%} | {m_after['coverage_gap']:+.1%} | {m_after['pinball']:.5f} |",
        "",
        f"تصحیح اولیه (از کالیبراسیون CQR روی برش proper/calib آموزش): {corr_path[0]:+.4f}؛ "
        f"تصحیح نهایی پس از ۲۵ به‌روزرسانی روزانه: {corr_path[-1]:+.4f}.",
        "",
        f"**نتیجه:** ACI شکاف پوشش را از {m_before['coverage_gap']:+.1%} به "
        f"{m_after['coverage_gap']:+.1%} {'برد' if abs(m_after['coverage_gap']) < abs(m_before['coverage_gap']) else 'نبرد'} "
        f"و pinball را {'بهتر کرد' if m_after['pinball'] < m_before['pinball'] else 'بدتر کرد'} "
        f"({m_before['pinball']:.5f}→{m_after['pinball']:.5f}) — سازگار با یافته‌ی ۲۲ فاز ۷ "
        "(ACI لایه‌ی کالیبراسیون نهایی پروژه).",
    ]

    report = "\n".join(lines)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "8.8_calibration.md").write_text(report + "\n")
    rel_df.to_csv(OUT_DIR / "8.8_reliability_table.csv", index=False)
    print(report)
    print(f"\nذخیره شد در {OUT_DIR}/8.8_calibration.md")


if __name__ == "__main__":
    main()
