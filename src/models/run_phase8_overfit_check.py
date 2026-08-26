"""بند ۸.۹ WBS فاز ۸ — بررسی Overfit/Underfit روی Test قفل‌شده.

قهرمان رسمی (`lightgbm_quantile`). دو بخش:

1. **شکاف train-test** — همان الگوی `src/models/fit_diagnosis.py` (گام ۱۱ کارت مدل)
   ولی روی مرز holdout به‌جای ۵ fold رسمی — نسبت pinball(test)/pinball(train).
2. **منحنی یادگیری** — بازبرازش روی پیشوندهای زمانی فزاینده‌ی train (۱۰٪ تا ۱۰۰٪ روزها،
   بدون شکستن ترتیب زمانی)، pinball روی همان زیرمجموعه‌ی train و روی کل Test.

اجرا: ``python -m src.models.run_phase8_overfit_check``
"""

import importlib

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FIGURES_DIR, REPORTS_DIR
from src.cv import DATE_COL
from src.eda_lib.figio import save_fig
from src.models.axes import TUNING_TAU
from src.models.card_writer import load_s2_result
from src.models.fit_diagnosis import render_step11, train_test_gap
from src.models.run_phase8_holdout_eval import load_holdout
from src.viz_fa import fa
from src.viz_fa import setup as viz_setup

OUT_DIR = REPORTS_DIR / "phase8"
FIG_DIR = FIGURES_DIR / "phase8"
CHAMPION = "lightgbm_quantile"
TAU = TUNING_TAU
FRACTIONS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]


def main() -> None:
    viz_setup()
    train, test, fold = load_holdout()
    result = load_s2_result(CHAMPION, "F02")
    hp = result["best_hyperparams"]
    fn = importlib.import_module("src.models.families.f02_tree").MODELS[CHAMPION]

    gap_rows = train_test_gap(fn, [(train, test)], TAU, hp)
    gap_report = render_step11(gap_rows)

    from src.baselines import operational_metrics
    uniq_days = np.sort(train[DATE_COL].unique())
    curve_rows = []
    for frac in FRACTIONS:
        n_days = max(10, int(round(len(uniq_days) * frac)))
        cutoff = uniq_days[n_days - 1]
        sub_train = train[train[DATE_COL] <= cutoff]
        pred_tr = np.clip(np.asarray(fn(sub_train, sub_train, TAU, **hp), dtype=float), 0.0, 1.0)
        pred_te = np.clip(np.asarray(fn(sub_train, test, TAU, **hp), dtype=float), 0.0, 1.0)
        pb_tr = operational_metrics(sub_train, pred_tr, TAU)["pinball"]
        pb_te = operational_metrics(test, pred_te, TAU)["pinball"]
        curve_rows.append({"frac": frac, "n_days": n_days, "n_rows": len(sub_train),
                           "pinball_train": pb_tr, "pinball_test": pb_te})
    curve_df = pd.DataFrame(curve_rows)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(curve_df["n_rows"], curve_df["pinball_train"], marker="o", label=fa("pinball — آموزش"))
    ax.plot(curve_df["n_rows"], curve_df["pinball_test"], marker="s", label=fa("pinball — Test قفل‌شده"))
    ax.set_xlabel(fa("تعداد ردیف آموزش"))
    ax.set_ylabel(fa("pinball loss"))
    ax.set_title(fa(f"منحنی یادگیری — {CHAMPION}"))
    ax.legend()
    ax.grid(alpha=0.3)
    save_fig(fig, "8.9_learning_curve.png", FIG_DIR)
    plt.close(fig)

    b3_pinball = float(pd.read_csv(OUT_DIR / "8.1_holdout_metrics.csv", index_col=0)
                       .loc["B3_empirical_quantile", "pinball"])

    final_gap = curve_df.iloc[-1]
    gap_ratio = final_gap["pinball_test"] / final_gap["pinball_train"]
    plateaued = abs(curve_df["pinball_test"].iloc[-1] - curve_df["pinball_test"].iloc[-3]) < 0.0005
    # مرجع «بد» نه یک آستانه‌ی دلبخواه بلکه B3 (مرجع رسمی پروژه) است — بند 7.9.2/۸.۱
    if gap_ratio > 1.20:
        verdict = f"Overfit قابل‌توجه (نسبت {gap_ratio:.3f} از آستانه‌ی بند 7.6.3 عبور کرد)"
    elif final_gap["pinball_test"] > b3_pinball:
        verdict = (f"Underfit احتمالی — pinball test ({final_gap['pinball_test']:.5f}) "
                   f"بدتر از مرجع B3 ({b3_pinball:.5f}) است")
    else:
        verdict = (f"بدون نشانه‌ی over/underfit — نسبت test/train ({gap_ratio:.3f}) زیر آستانه‌ی "
                   f"۱.۲۰ و pinball test بهتر از مرجع B3 ({b3_pinball:.5f}) است")

    lines = [
        "# بند ۸.۹ — بررسی Overfit/Underfit روی Test قفل‌شده",
        "",
        f"> مدل: `{CHAMPION}`. τ={TAU}.",
        "",
        "## شکاف train-test (الگوی گام ۱۱ کارت مدل)",
        "",
        "⚠️ متن زیر از `fit_diagnosis.py` نقل شده و به «۵ fold رسمی» اشاره می‌کند — اینجا "
        "روی **یک** fold (مرز holdout بند ۸.۱) اجرا شده، نه ۵ fold فاز ۷.",
        "",
        gap_report,
        "",
        "## منحنی یادگیری (پیشوند زمانی فزاینده)",
        "",
        "| کسر | n روز | n ردیف | pinball train | pinball test |",
        "|---|---|---|---|---|",
    ]
    for _, r in curve_df.iterrows():
        lines.append(f"| {r['frac']:.0%} | {int(r['n_days'])} | {int(r['n_rows'])} | "
                      f"{r['pinball_train']:.5f} | {r['pinball_test']:.5f} |")
    lines += [
        "",
        f"نسبت نهایی test/train (۱۰۰٪ داده): {gap_ratio:.3f}. منحنی test "
        f"{'به‌نظر تخت رسیده (بهبود <۰.۰۰۰۵ در ۲ گام آخر)' if plateaued else 'هنوز در حال بهبود با داده‌ی بیشتر است'}.",
        f"**داوری:** {verdict}.",
        "",
        f"نمودار: `reports/figures/phase8/8.9_learning_curve.png`",
    ]

    report = "\n".join(lines)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "8.9_overfit_check.md").write_text(report + "\n")
    curve_df.to_csv(OUT_DIR / "8.9_learning_curve.csv", index=False)
    print(report)
    print(f"\nذخیره شد در {OUT_DIR}/8.9_overfit_check.md")


if __name__ == "__main__":
    main()
