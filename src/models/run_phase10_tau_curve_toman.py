"""نسخه‌ی مکمل شکل بند ۱۰.۷: روی همان محور عمودی چپِ کاهش هدررفت، برچسب هر تیک هم درصد
هم صرفه‌جویی سالانه‌ی معادل (میلیارد تومان) را نشان می‌دهد — بدون افزودن محور جداگانه
(دو محور هم‌جهت روی هم استاندارد نیست). تیک‌ها دقیقاً روی ۷ نقطه‌ی داده‌ی شبکه‌ی τ
گذاشته شده‌اند، پس مقدار میلیارد‌تومانی هر برچسب همان عدد محاسبه‌شده‌ی واقعی است، نه
درون‌یابی‌شده. فرمول عیناً بند ۱۰.۴ (annual_toman = میانگین روزانه‌ی پرس صرفه‌جویی‌شده
× ۷ × ACADEMIC_WEEKS_PER_YEAR × COST_PER_PORTION_TOMAN)، بدون بازآموزش مدل: از خروجی
ذخیره‌شده‌ی ``reports/phase10/10.7_tau_curve.csv`` می‌خواند.

اجرا: ``python -m src.models.run_phase10_tau_curve_toman``
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import viz_fa  # noqa: F401  — فونت فارسی
from src.config import REPORTS_DIR
from src.models.run_phase10_annual_savings import ACADEMIC_WEEKS_PER_YEAR, COST_PER_PORTION_TOMAN
from src.models.run_phase10_tau_curve import OPERATING_TAU

IN_CSV = REPORTS_DIR / "phase10" / "10.7_tau_curve.csv"
FIG_PATH = str(REPORTS_DIR / "figures" / "phase10" / "10.7_tau_tradeoff_toman_fa.png")


def main() -> None:
    d = pd.read_csv(IN_CSV)
    d["annual_toman_billion"] = (
        d["portions_saved"] / d["n_days"] * 7 * ACADEMIC_WEEKS_PER_YEAR * COST_PER_PORTION_TOMAN / 1e9
    )

    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    ax2 = ax.twinx()
    l1, = ax.plot(d["tau"], d["waste_reduction_pct"], "o-", color="#3b6ea5", lw=2, ms=6,
                  label="کاهش هدررفت")
    l2, = ax2.plot(d["tau"], 100 * d["shortage_rate"], "s-", color="#b8443c", lw=2, ms=6,
                   label="نرخ کمبود واقعی")
    l3, = ax2.plot(d["tau"], 100 * d["tau"], "--", color="0.5", lw=1.3,
                   label="نرخ کمبود مورد انتظار")

    ax.axvline(OPERATING_TAU, ls=":", color="#2e8b57", lw=1.4)
    ax.annotate("نقطه‌ی عملیاتی", xy=(OPERATING_TAU + 0.005, ax.get_ylim()[0] + 1.2),
                fontsize=9, color="#2e8b57")
    ax.set_xlabel("سطح کوانتایل: بالاتر یعنی پخت کمتر")
    ax.set_ylabel("کاهش هدررفت (درصد) / صرفه‌جویی سالانه (میلیارد تومان)", color="#3b6ea5")
    ax2.set_ylabel("نرخ کمبود (درصد)", color="#b8443c")
    ax2.tick_params(axis="y", labelcolor="#b8443c")
    ax.grid(alpha=0.25)
    ax.legend(handles=[l1, l2, l3], loc="lower right", fontsize=9, framealpha=0.92)

    # تیک‌های محور چپ را دقیقاً روی ۷ نقطه‌ی داده می‌گذاریم، پس برچسب هر تیک هم درصد
    # هم صرفه‌جویی سالانه‌ی معادل همان نقطه است — عدد واقعی، نه درون‌یابی‌شده.
    ax.set_yticks(d["waste_reduction_pct"])
    ax.set_yticklabels([f"{p:.0f}% ({t:.0f}B)" for p, t in zip(d["waste_reduction_pct"], d["annual_toman_billion"])],
                        fontsize=7.5)
    ax.tick_params(axis="y", labelcolor="#3b6ea5")

    ax3 = ax.twiny()
    ax3.set_xlim(ax.get_xlim())
    ax3.set_xticks(list(d["tau"]))
    ax3.set_xticklabels([f"{(1 - t) / t:.3g}×" for t in d["tau"]], fontsize=9)
    ax3.set_xlabel("نسبت هزینه‌ی کمبود به هزینه‌ی مازاد", labelpad=8)

    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=200)
    print(f"شکل: {FIG_PATH}")
    print(d[["tau", "waste_reduction_pct", "annual_toman_billion"]].to_string(index=False))


if __name__ == "__main__":
    main()
