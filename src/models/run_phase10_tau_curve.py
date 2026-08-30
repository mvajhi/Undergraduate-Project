"""بند ۱۰.۷ — منحنی معاوضه‌ی کمبود/هدررفت قهرمان روی شبکه‌ی τ (شکل فصل ۴ گزارش نهایی).

**چرا این بند اضافه شد.** گزارش نهایی برای بند «تحلیل کسب‌وکاری» به یک منحنی معاوضه
نیاز داشت، ولی هیچ‌کدام از دو منحنی موجود مخزن با جدول همان بند هم‌پایه نبودند:
`reports/figures/8.5_tradeoff_curve.png` روی پنجره‌ی Test (رژیم غیرعادی ردیف ۴۹) است و
`reports/figures/report_16_tau_tradeoff.png` از تحلیل حساسیت فاز ۶ روی خط پایه می‌آید
(نرخ کمبود ۹.۷٪ در τ=۰.۲۰ در برابر ۱۵.۴٪ سناریوهای بند ۱۰.۳). گذاشتن هرکدام کنار
جدول سناریوها یک تناقض عددی قابل‌مشاهده می‌ساخت.

**پایه‌ی داده اینجا عیناً همان بند ۱۰.۳/۱۰.۴ است:** OOF قهرمان روی هر ۵ fold رسمی، با
همان `oof_predictions` — فقط روی شبکه‌ی چگال‌تر τ اجرا می‌شود تا شکل منحنی دیده شود.
پس سه نقطه‌ی سناریوهای بند ۱۰.۳ باید دقیقاً روی این منحنی بیفتند.

اجرا: ``python -m src.models.run_phase10_tau_curve``
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import viz_fa  # noqa: F401  — فونت فارسی
from src.config import REPORTS_DIR, set_global_seed
from src.models.run_phase10_annual_savings import oof_predictions

OUT_DIR = REPORTS_DIR / "phase10"
# final_report/img/tau_tradeoff.png symlink به این است — reports/figures/ منبع حقیقت
# تصاویر گزارش است (tools/report_figures/README.md).
FIG_PATH = str(REPORTS_DIR / "figures" / "phase10" / "10.7_tau_tradeoff_fa.png")
TAU_GRID = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
OPERATING_TAU = 0.20


def curve() -> pd.DataFrame:
    rows = []
    for tau in TAU_GRID:
        oof = oof_predictions(tau)
        saved = float(oof["portions_saved"].sum())
        base = float(oof["surplus_b0"].sum())
        n_days = oof["DateOnly"].nunique() if "DateOnly" in oof else oof.iloc[:, 0].nunique()
        rows.append({
            "tau": tau,
            "shortage_rate": float(oof["shortage"].mean()),
            "waste_reduction_pct": 100.0 * saved / base,
            "portions_saved": saved,
            "n_rows": len(oof),
            "n_days": n_days,
        })
        print(f"  τ={tau:.2f}  کمبود={rows[-1]['shortage_rate']:.1%}  "
              f"کاهش هدررفت={rows[-1]['waste_reduction_pct']:.1f}%")
    return pd.DataFrame(rows)


def plot(d: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
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
    ax.set_ylabel("کاهش هدررفت (درصد)", color="#3b6ea5")
    ax2.set_ylabel("نرخ کمبود (درصد)", color="#b8443c")
    ax.tick_params(axis="y", labelcolor="#3b6ea5")
    ax2.tick_params(axis="y", labelcolor="#b8443c")
    ax.grid(alpha=0.25)
    ax.legend(handles=[l1, l2, l3], loc="lower right", fontsize=9, framealpha=0.92)

    # محور بالا: نسبت هزینه‌ی کمبود به مازاد که هر τ از آن می‌آید — τ* = C_o/(C_u+C_o)
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
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    d = curve()
    d.to_csv(OUT_DIR / "10.7_tau_curve.csv", index=False)
    plot(d)

    lines = ["# بند ۱۰.۷ — منحنی معاوضه‌ی کمبود/هدررفت قهرمان روی شبکه‌ی τ", "",
             "> پایه: OOF قهرمان روی هر ۵ fold رسمی — همان پایه‌ی بند ۱۰.۳/۱۰.۴، فقط شبکه‌ی τ چگال‌تر.",
             "> این سند برای شکل بند ۴-۵-۵ گزارش نهایی ساخته شد (دلیل در docstring ماژول).", "",
             "| τ | نرخ کمبود واقعی | کاهش هدررفت |", "|---|---|---|"]
    for _, r in d.iterrows():
        lines.append(f"| {r['tau']:.2f} | {r['shortage_rate']:.1%} | {r['waste_reduction_pct']:.1f}% |")
    (OUT_DIR / "10.7_tau_curve.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("گزارش: reports/phase10/10.7_tau_curve.md")


if __name__ == "__main__":
    main()
