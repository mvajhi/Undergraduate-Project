"""بند ۹.۳ WBS فاز ۹ — اثرات نهایی (PDP · ICE · ALE) روی قهرمان، مرز holdout بند ۸.۱.

**چرا دستی پیاده شد، نه با `sklearn.inspection`:** فیچرست قهرمان ستون‌های `category`
خام دارد (رمزگذاری بومی LightGBM، بند 7.5.3) و مسیر داخلی sklearn هنگام جایگزینی
شبکه‌ی مقادیر، dtype را می‌شکند (`train and valid dataset categorical_feature do not
match`). پیاده‌سازی مستقیم، ستون‌های دسته‌ای را دست‌نخورده نگه می‌دارد.

**سه سنجه، سه کاربرد متفاوت:**

- **PDP** میانگین اثر حاشیه‌ای — ⚠️ در حضور هم‌خطی گمراه‌کننده است (نقاط غیرواقعی
  می‌سازد؛ مثلاً `log_res` بزرگ با `log_daily_total_res` کوچک).
- **ICE** همان محاسبه ولی بدون میانگین‌گیری — **ناهمگنی** را نشان می‌دهد: اگر منحنی‌های
  فردی شکل‌های متفاوت داشته باشند، PDP میانگینِ دو رفتار متضاد است و بی‌معنا.
- **ALE** ✅ **مرجع اصلی این بند** — فقط از نقاط واقعی همان بازه استفاده می‌کند، پس به
  هم‌خطی مقاوم است (بند ۹.۳ WBS صریحاً آن را بر PDP ترجیح می‌دهد). یافته‌ی ۹ فاز ۷
  خوشه‌ی هم‌خط VIF ۲۰۰۰-۱۰۰۰۰ را در همین فیچرست مستند کرده — پس این ترجیح اینجا
  نظری نیست، مصداق دارد.

فیچرهای دسته‌ای (`RestaurantName`, `dow_x_city` — دو فیچر برتر بند ۹.۱/۹.۲) ALE/PDP
یک‌بعدی استاندارد ندارند (ترتیب ندارند)، پس برایشان **میانگین پیش‌بینی به‌ازای هر سطح**
گزارش می‌شود که معادل قابل‌دفاع همان مفهوم است.

اجرا: ``python -m src.models.run_phase9_effects``
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FIGURES_DIR, REPORTS_DIR, set_global_seed
from src.eda_lib.figio import save_fig
from src.models.axes import TUNING_TAU
from src.models.phase9_common import fit_champion
from src.viz_fa import fa
from src.viz_fa import setup as viz_setup

OUT_DIR = REPORTS_DIR / "phase9"
FIG_DIR = FIGURES_DIR / "phase9"
TAU = TUNING_TAU

#: فیچرهای عددی برتر طبق بند ۹.۱ (permutation) و ۹.۲ (SHAP) — هر دو رتبه‌بندی توافق دارند
NUMERIC_FEATURES = ["log_res", "rho_cell_lag1", "day_shock_lag1",
                    "cell_dow_expanding_rate", "res_vs_history"]
CATEGORICAL_FEATURES = ["RestaurantName", "dow_x_city"]
N_GRID = 25
N_ICE_LINES = 80
N_ALE_BINS = 20


def pdp_ice(model, X: pd.DataFrame, feature: str, n_grid: int = N_GRID
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """برمی‌گرداند (grid، منحنی PDP، ماتریس ICE با شکل (n_rows, n_grid))."""
    grid = np.quantile(X[feature].dropna().to_numpy(), np.linspace(0.02, 0.98, n_grid))
    grid = np.unique(grid)
    ice = np.empty((len(X), len(grid)), dtype=float)
    for j, v in enumerate(grid):
        Xmod = X.copy()
        Xmod[feature] = v
        ice[:, j] = model.predict(Xmod)
    return grid, ice.mean(axis=0), ice


def ale(model, X: pd.DataFrame, feature: str, n_bins: int = N_ALE_BINS
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ALE مرتبه‌اول (Apley & Zhu) — اثر **محلی** انباشته، فقط روی نقاط واقعی هر بازه.

    برمی‌گرداند (لبه‌های بازه، ALE مرکز‌شده، تعداد نقطه در هر بازه).
    """
    vals = X[feature].to_numpy(dtype=float)
    edges = np.unique(np.quantile(vals[~np.isnan(vals)], np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return edges, np.zeros(len(edges)), np.zeros(max(len(edges) - 1, 0))

    # هر نقطه به بازه‌ی خودش نسبت داده می‌شود (بازه‌ی ۱ تا len(edges)-1)
    bin_idx = np.clip(np.searchsorted(edges, vals, side="left"), 1, len(edges) - 1)

    local_effects = np.zeros(len(edges) - 1)
    counts = np.zeros(len(edges) - 1)
    for k in range(1, len(edges)):
        mask = bin_idx == k
        counts[k - 1] = mask.sum()
        if not mask.any():
            continue
        X_lo, X_hi = X[mask].copy(), X[mask].copy()
        X_lo[feature] = edges[k - 1]
        X_hi[feature] = edges[k]
        local_effects[k - 1] = float(np.mean(model.predict(X_hi) - model.predict(X_lo)))

    ale_uncentered = np.concatenate([[0.0], np.cumsum(local_effects)])
    # مرکزکردن با وزن فراوانی، تا ALE=0 در «نقطه‌ی میانگین داده» باشد
    mid = (ale_uncentered[:-1] + ale_uncentered[1:]) / 2.0
    weight = counts.sum()
    center = float((mid * counts).sum() / weight) if weight > 0 else 0.0
    return edges, ale_uncentered - center, counts


def categorical_effect(model, X: pd.DataFrame, feature: str) -> pd.DataFrame:
    """معادل PDP برای فیچر دسته‌ای بدون ترتیب: میانگین پیش‌بینی وقتی **همه‌ی** ردیف‌ها
    روی هر سطح گذاشته شوند (اثر حاشیه‌ای واقعی، نه صرفاً میانگین گروه مشاهده‌شده)."""
    rows = []
    for level in X[feature].cat.categories:
        Xmod = X.copy()
        Xmod[feature] = pd.Categorical([level] * len(X), categories=X[feature].cat.categories)
        rows.append({"level": str(level), "mean_prediction": float(model.predict(Xmod).mean()),
                     "n_observed": int((X[feature] == level).sum())})
    return pd.DataFrame(rows).sort_values("mean_prediction", ascending=False).reset_index(drop=True)


def main() -> None:
    viz_setup()
    set_global_seed()
    model, Xtr, Xte, train, test, cols = fit_champion(TAU)
    rng = np.random.default_rng(42)

    numeric_summaries = []
    for feat in NUMERIC_FEATURES:
        grid, pdp_curve, ice_mat = pdp_ice(model, Xte, feat)
        edges, ale_vals, counts = ale(model, Xte, feat)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        sel = rng.choice(len(ice_mat), size=min(N_ICE_LINES, len(ice_mat)), replace=False)
        for i in sel:
            axes[0].plot(grid, ice_mat[i], color="gray", alpha=0.15, linewidth=0.7)
        axes[0].plot(grid, pdp_curve, color="#C44E52", linewidth=2.5, label=fa("PDP (میانگین)"))
        axes[0].set_xlabel(fa(feat))
        axes[0].set_ylabel(fa("پیش‌بینی ρ̂"))
        axes[0].set_title(fa(f"PDP + ICE — {feat}"))
        axes[0].legend()

        axes[1].plot(edges, ale_vals, marker="o", color="#4C72B0", markersize=3)
        axes[1].axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        axes[1].set_xlabel(fa(feat))
        axes[1].set_ylabel(fa("ALE (اثر محلی انباشته)"))
        axes[1].set_title(fa(f"ALE — {feat} ✅ مقاوم به هم‌خطی"))
        fig.tight_layout()
        save_fig(fig, f"9.3_effects_{feat}.png", FIG_DIR)
        plt.close(fig)

        # ناهمگنی: پراکندگی دامنه‌ی منحنی‌های ICE فردی
        ice_ranges = ice_mat.max(axis=1) - ice_mat.min(axis=1)
        numeric_summaries.append({
            "feature": feat,
            "pdp_range": float(pdp_curve.max() - pdp_curve.min()),
            "ale_range": float(ale_vals.max() - ale_vals.min()),
            "ale_direction": "صعودی" if ale_vals[-1] > ale_vals[0] else "نزولی",
            "ice_range_median": float(np.median(ice_ranges)),
            "ice_range_p90": float(np.percentile(ice_ranges, 90)),
            "heterogeneity_ratio": float(np.percentile(ice_ranges, 90) /
                                          (pdp_curve.max() - pdp_curve.min()))
            if pdp_curve.max() > pdp_curve.min() else float("nan"),
        })

    num_df = pd.DataFrame(numeric_summaries)

    cat_tables = {}
    for feat in CATEGORICAL_FEATURES:
        tbl = categorical_effect(model, Xte, feat)
        cat_tables[feat] = tbl
        top = tbl.head(12)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.barh(range(len(top)), top["mean_prediction"].to_numpy()[::-1], color="#55A868")
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels([fa(s) for s in top["level"][::-1]], fontsize=8)
        ax.set_xlabel(fa("میانگین پیش‌بینی ρ̂"))
        ax.set_title(fa(f"اثر حاشیه‌ای سطوح — {feat} (۱۲ بالا)"))
        fig.tight_layout()
        save_fig(fig, f"9.3_categorical_{feat}.png", FIG_DIR)
        plt.close(fig)

    lines = [
        "# بند ۹.۳ — اثرات نهایی (PDP · ICE · ALE) روی قهرمان (`lightgbm_quantile`)",
        "",
        f"> τ={TAU}. داده: مرز holdout بند ۸.۱ (Test، {len(Xte):,} ردیف). "
        "✅ **ALE مرجع اصلی است، نه PDP** — بند ۹.۳ WBS و یافته‌ی ۹ فاز ۷ (خوشه‌ی هم‌خط "
        "VIF ۲۰۰۰-۱۰۰۰۰ در همین فیچرست).",
        "",
        "## فیچرهای عددی",
        "",
        "| فیچر | دامنه‌ی PDP | دامنه‌ی ALE | جهت ALE | میانه‌ی دامنه‌ی ICE | نسبت ناهمگنی (ICE p90 ÷ دامنه‌ی PDP) |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in num_df.iterrows():
        lines.append(f"| `{r['feature']}` | {r['pdp_range']:.5f} | {r['ale_range']:.5f} | "
                     f"{r['ale_direction']} | {r['ice_range_median']:.5f} | {r['heterogeneity_ratio']:.2f} |")
    lines += [
        "",
        "**خواندن ستون آخر:** نسبت ≈۱ یعنی منحنی‌های فردی تقریباً هم‌شکل‌اند و PDP نماینده‌ی "
        "خوبی است؛ نسبت **بسیار بزرگ‌تر از ۱** یعنی ناهمگنی جدی — PDP میانگینِ رفتارهای "
        "متفاوت است و به‌تنهایی گمراه‌کننده.",
        "",
        "نمودارها (هر فیچر: PDP+ICE در چپ، ALE در راست): "
        + " · ".join(f"`reports/figures/phase9/9.3_effects_{f}.png`" for f in NUMERIC_FEATURES),
        "",
        "## فیچرهای دسته‌ای (اثر حاشیه‌ای سطوح)",
        "",
        "⚠️ ALE/PDP یک‌بعدی استاندارد برای فیچر دسته‌ای **بدون ترتیب** تعریف‌شده نیست. "
        "به‌جایش میانگین پیش‌بینی وقتی همه‌ی ردیف‌های Test روی هر سطح گذاشته شوند گزارش می‌شود.",
        "",
    ]
    for feat, tbl in cat_tables.items():
        hi, lo = tbl.iloc[0], tbl.iloc[-1]
        lines += [
            f"### `{feat}` ({len(tbl)} سطح)",
            "",
            f"- بیشترین اثر: **{hi['level']}** (ρ̂ میانگین={hi['mean_prediction']:.4f}، "
            f"{hi['n_observed']} مشاهده در Test)",
            f"- کمترین اثر: **{lo['level']}** (ρ̂ میانگین={lo['mean_prediction']:.4f}، "
            f"{lo['n_observed']} مشاهده در Test)",
            f"- دامنه‌ی کل: {hi['mean_prediction'] - lo['mean_prediction']:.4f}",
            f"- نمودار: `reports/figures/phase9/9.3_categorical_{feat}.png`",
            "",
        ]

    report = "\n".join(lines)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "9.3_effects.md").write_text(report + "\n")
    num_df.to_csv(OUT_DIR / "9.3_numeric_effects.csv", index=False)
    for feat, tbl in cat_tables.items():
        tbl.to_csv(OUT_DIR / f"9.3_categorical_{feat}.csv", index=False)
    print(report)
    print(f"\nذخیره شد در {OUT_DIR}/9.3_effects.md")


if __name__ == "__main__":
    main()
