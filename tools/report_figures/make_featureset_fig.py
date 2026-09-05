"""بازتولید شکل بند ۳-۲-۳ — $R^2$ خارج‌نمونه‌ی شش مجموعه‌ی ویژگی در برابر تعداد ویژگی.
`final_report/img/feature_sets.png` symlink به خروجی این اسکریپت است — منبع حقیقت
تصاویر گزارش `reports/figures/` است (tools/report_figures/README.md).

منبع اعداد (هیچ عددی اینجا محاسبه نمی‌شود):
  - $R^2$: لایه ۳ ممیزی نشت در `doc/leakage_audit.md` (تولید `src/features/audit.py`،
    تقسیم زمانی منفرد ۷۵/۲۵ با HistGradientBoosting).
  - تعداد ویژگی: `data/processed/feature_sets_v1.json` (تولید `src/features/build.py`).
"""

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from src import viz_fa  # noqa: F401  — فونت فارسی و patch متن
from src.viz_fa import fa

AUDIT_PATH = Path("doc/leakage_audit.md")
SETS_PATH = Path("data/processed/feature_sets_v1.json")
OUT = Path("reports/figures/5.12_feature_sets_fa.png")

# برچسب فارسی هر مجموعه — ترتیب نمایش با تعداد ویژگی صعودی می‌شود
LABELS = {
    "FS_baseline": "خط پایه",
    "FS_calendar": "تقویمی",
    "FS_lag": "وقفه‌ای",
    "FS_day": "عامل روز",
    "FS_full_A": "کامل تجمیعی",
    "FS_bridge": "کامل + ترکیب",
}


def read_r2() -> dict[str, float]:
    """جدول لایه ۳ ممیزی نشت: | `FS_x` | +0.1394 | ✅ |"""
    text = AUDIT_PATH.read_text(encoding="utf-8")
    found = dict(re.findall(r"\|\s*`(FS_\w+)`\s*\|\s*([+-]?\d+\.\d+)\s*\|", text))
    missing = set(LABELS) - set(found)
    if missing:
        raise SystemExit(f"در {AUDIT_PATH} یافت نشد: {sorted(missing)}")
    return {k: float(v) for k, v in found.items()}


def main() -> None:
    r2 = read_r2()
    counts = {k: len(v) for k, v in json.loads(SETS_PATH.read_text()).items()}

    keys = sorted(LABELS, key=lambda k: counts[k])
    n = [counts[k] for k in keys]
    y = [r2[k] for k in keys]
    best = max(keys, key=lambda k: r2[k])

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    colors = ["#2F4F6F" if k == best else "#A8B7C6" for k in keys]
    bars = ax.bar(range(len(keys)), y, color=colors, width=0.62)

    for i, (bar, k) in enumerate(zip(bars, keys)):
        ax.text(
            i,
            bar.get_height() + 0.004,
            f"{r2[k]:+.3f}".replace("-", "−"),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([fa(f"{LABELS[k]}\n({counts[k]})") for k in keys], fontsize=9)
    ax.set_ylabel("Out-of-sample $R^2$")
    ax.set_xlabel(fa("مجموعه‌ی ویژگی (تعداد ویژگی)"))
    ax.set_ylim(0, max(y) * 1.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linewidth=0.4, alpha=0.35)
    ax.set_axisbelow(True)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200)
    plt.close(fig)
    print(f"saved {OUT} — n={n}")


if __name__ == "__main__":
    main()
