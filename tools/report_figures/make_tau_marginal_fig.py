"""بازتولید شکل بند ۳-۲-۶ — نرخ مبادله‌ی حاشیه‌ای در هر گام τ.
`final_report/img/tau_marginal.png` symlink به خروجی این اسکریپت است — منبع حقیقت
تصاویر گزارش `reports/figures/` است (tools/report_figures/README.md).

منبع اعداد (هیچ عددی اینجا محاسبه نمی‌شود): جدول «تحلیل حاشیه‌ای» در
`reports/tau_sensitivity.md` (تولید `src/run_protocol.py`، روی خط پایه‌ی B3 و
walk-forward ۵ بخشی) — همان اعدادی که ردیف ۳۴ `decision_log` به آن استناد می‌کند.
"""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from src import viz_fa  # noqa: F401  — فونت فارسی و patch متن
from src.viz_fa import fa

SRC = Path("reports/tau_sensitivity.md")
OUT = Path("reports/figures/6.10_tau_marginal_fa.png")
COST_RATIO = 4.0  # برآورد نسبت هزینه‌ی کمبود به مازاد (ردیف ۳۴ decision_log)

ROW = re.compile(
    r"\|\s*([\d.]+)\s*→\s*([\d.]+)\s*\|[^|]*\|[^|]*\|\s*\*\*([\d.]+)×\*\*\s*\|"
)


def read_steps() -> list[tuple[str, str, float]]:
    rows = ROW.findall(SRC.read_text(encoding="utf-8"))
    if len(rows) < 5:
        raise SystemExit(f"جدول تحلیل حاشیه‌ای در {SRC} یافت نشد (ردیف: {len(rows)})")
    return [(a, b, float(r)) for a, b, r in rows]


def main() -> None:
    steps = read_steps()
    labels = [f"{a}→{b}" for a, b, _ in steps]
    rates = [r for _, _, r in steps]

    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    colors = ["#2F4F6F" if r >= COST_RATIO else "#C3CDD8" for r in rates]
    ax.bar(range(len(rates)), rates, color=colors, width=0.6)

    for i, r in enumerate(rates):
        ax.text(i, r * 1.06, f"{r:.1f}×", ha="center", va="bottom", fontsize=9)

    ax.axhline(COST_RATIO, color="black", linestyle="--", linewidth=1)
    ax.text(
        len(rates) - 0.45,
        COST_RATIO * 1.12,
        fa("نسبت هزینه‌ی کمبود به مازاد"),
        ha="right",
        va="bottom",
        fontsize=9,
    )

    ax.set_yscale("log")
    ax.minorticks_off()
    ax.set_ylim(1.5, 60)
    ax.set_yticks([2, 4, 10, 30])
    ax.set_yticklabels(["2×", "4×", "10×", "30×"])
    ax.set_xticks(range(len(rates)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel(fa("گام کوانتایل"))
    ax.set_ylabel(fa("پرس مازاد صرفه‌جویی‌شده\nبه‌ازای هر پرس کمبود"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linewidth=0.4, alpha=0.35)
    ax.set_axisbelow(True)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200)
    plt.close(fig)
    print(f"saved {OUT} — rates={rates}")


if __name__ == "__main__":
    main()
