"""بازتولید دو شکل گزارش نهایی با برچسب‌های سالم (τ/Δ به‌صورت mathtext، بدون عنوان)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from src import viz_fa  # noqa: F401  — فونت فارسی و patch متن

# final_report/img/reliability.png و .../perm_importance.png symlink به این دو هستند —
# reports/figures/ منبع حقیقت تصاویر گزارش است (tools/report_figures/README.md).
OUT_RELIABILITY = "reports/figures/phase8/8.8_reliability_fa.png"
OUT_PERM_IMPORTANCE = "reports/figures/phase9/9.1_perm_importance_fa.png"

# ---- شکل کالیبراسیون (بند ۴-۴-۳) ----
d = pd.read_csv("reports/phase8/8.8_reliability_table.csv")
fig, ax = plt.subplots(figsize=(6.0, 4.4))
ax.plot([0, 1], [0, 1], ls="--", c="0.45", lw=1.2, label="خط ایده‌آل")
ax.plot(d["tau"], d["coverage"], "o-", lw=1.8, ms=5, color="#3b6ea5", label="پوشش مشاهده‌شده")
ax.axvline(0.20, ls=":", c="#2e8b57", lw=1.3)
ax.annotate("نقطه‌ی عملیاتی", xy=(0.205, 0.03), fontsize=9, color="#2e8b57")
ax.set_xlabel("کوانتایل اسمی")
ax.set_ylabel("پوشش تجربی")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(OUT_RELIABILITY, dpi=200); plt.close(fig)

# ---- شکل اهمیت ویژگی (بند ۴-۵-۲) ----
f = pd.read_csv("reports/phase9/9.1_feature_importance.csv", index_col=0)
f = f.sort_values("perm_importance_mean", ascending=False).head(8).iloc[::-1]
fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.barh(range(len(f)), f["perm_importance_mean"], xerr=f["perm_importance_std"],
        color="#c8794a", ecolor="0.25", capsize=2.5, height=0.68)
ax.set_yticks(range(len(f)))
ax.set_yticklabels(f.index, fontsize=9)
ax.set_xlabel(r"افزایش زیان Pinball پس از به‌هم‌ریختن ویژگی")
ax.grid(axis="x", alpha=0.25)
fig.tight_layout(); fig.savefig(OUT_PERM_IMPORTANCE, dpi=200); plt.close(fig)
print("saved")
