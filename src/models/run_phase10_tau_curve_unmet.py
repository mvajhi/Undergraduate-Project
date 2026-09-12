"""نسخه‌ی دیگر شکل بند ۱۰.۷: به‌جای «نرخ کمبود» (که فقط رخداد را می‌شمارد)، محور راست
«کسر تقاضای برآورده‌نشده» را نشان می‌دهد — مکمل Fill Rate (سطح خدمت نوع دوم)، که
*اندازه*‌ی کمبود را هم وزن می‌دهد (همان تعریف بند ۱۰.۴: unmet_rate = مجموع پرس کمبود
÷ مجموع پرس دریافت‌شده). محور چپ عیناً مثل ``run_phase10_tau_curve_toman.py`` است.

نرخ کمبود رخدادی و کسر تقاضای برآورده‌نشده دو مقیاس کاملاً متفاوت دارند (بند ۱۰.۴:
۱۵.۴٪ در برابر ۰.۲۸٪ در τ=۰.۲۰)، پس نمی‌شود هر دو را روی یک محور راست گذاشت — این شکل
جایگزین شکل قبلی است، نه رویش.

**بازآموزش مدل دارد** (بر خلاف ``run_phase10_tau_curve_toman.py``): چون
``reports/phase10/10.7_tau_curve.csv`` ستون Recv/shortage_portions ندارد، این اسکریپت
باید OOF قهرمان را روی هر ۷ نقطه‌ی شبکه‌ی τ از نو بسازد (عیناً همان `oof_predictions`
بند ۱۰.۳/۱۰.۴). خروجی در ``reports/phase10/10.7_tau_curve_unmet.csv`` ذخیره می‌شود تا
اجرای بعدی نیازی به بازآموزش نداشته باشد.

اجرا: ``python -m src.models.run_phase10_tau_curve_unmet``
"""

from pathlib import Path

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import viz_fa  # noqa: F401  — فونت فارسی
from src.config import REPORTS_DIR, set_global_seed
from src.models.run_phase10_annual_savings import (
    ACADEMIC_WEEKS_PER_YEAR, COST_PER_PORTION_TOMAN, oof_predictions,
)
from src.models.run_phase10_tau_curve import OPERATING_TAU, TAU_GRID

OUT_CSV = REPORTS_DIR / "phase10" / "10.7_tau_curve_unmet.csv"
FIG_PATH = str(REPORTS_DIR / "figures" / "phase10" / "10.7_tau_tradeoff_unmet_fa.png")


def curve() -> pd.DataFrame:
    rows = []
    for tau in TAU_GRID:
        oof = oof_predictions(tau)
        saved = float(oof["portions_saved"].sum())
        base = float(oof["surplus_b0"].sum())
        n_days = oof["DateOnly"].nunique() if "DateOnly" in oof else oof.iloc[:, 0].nunique()
        unmet_rate = float(oof["shortage_portions"].sum() / oof["Recv"].sum())
        rows.append({
            "tau": tau,
            "waste_reduction_pct": 100.0 * saved / base,
            "unmet_demand_pct": 100.0 * unmet_rate,
            "portions_saved": saved,
            "n_days": n_days,
        })
        print(f"  τ={tau:.2f}  کاهش هدررفت={rows[-1]['waste_reduction_pct']:.1f}%  "
              f"تقاضای برآورده‌نشده={rows[-1]['unmet_demand_pct']:.3f}%")
    return pd.DataFrame(rows)


def plot(d: pd.DataFrame) -> None:
    d["annual_toman_billion"] = (
        d["portions_saved"] / d["n_days"] * 7 * ACADEMIC_WEEKS_PER_YEAR * COST_PER_PORTION_TOMAN / 1e9
    )

    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    ax2 = ax.twinx()
    l1, = ax.plot(d["tau"], d["waste_reduction_pct"], "o-", color="#3b6ea5", lw=2, ms=6,
                  label="کاهش هدررفت")
    l2, = ax2.plot(d["tau"], d["unmet_demand_pct"], "s-", color="#b8443c", lw=2, ms=6,
                   label="تقاضای برآورده‌نشده")

    ax.axvline(OPERATING_TAU, ls=":", color="#2e8b57", lw=1.4)
    ax.annotate("نقطه‌ی عملیاتی", xy=(OPERATING_TAU + 0.005, ax.get_ylim()[0] + 1.2),
                fontsize=9, color="#2e8b57")
    ax.set_xlabel("سطح کوانتایل: بالاتر یعنی پخت کمتر")
    ax.set_ylabel("کاهش هدررفت (درصد) / صرفه‌جویی سالانه (میلیارد تومان)", color="#3b6ea5")
    ax2.set_ylabel("تقاضای برآورده‌نشده (درصد پرس)", color="#b8443c")
    ax2.tick_params(axis="y", labelcolor="#b8443c")
    ax.grid(alpha=0.25)
    ax.legend(handles=[l1, l2], loc="lower right", fontsize=9, framealpha=0.92)

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
    Path(FIG_PATH).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=200)
    print(f"شکل: {FIG_PATH}")


def main() -> None:
    set_global_seed()
    if OUT_CSV.exists():
        d = pd.read_csv(OUT_CSV)
    else:
        d = curve()
        OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        d.to_csv(OUT_CSV, index=False)
    plot(d)
    print(d.to_string(index=False))


if __name__ == "__main__":
    main()
