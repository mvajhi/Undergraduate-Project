"""بند ۹.۱ WBS فاز ۹ — اهمیت ویژگی روی قهرمان (`lightgbm_quantile`)، مرز holdout بند ۸.۱.

اهمیت درختی (gain/split) **سوگیری به کاردینالیتی بالا دارد** (بند بالای WBS) — به همین
دلیل Permutation Importance روی Test قفل‌شده هم محاسبه و دو رتبه‌بندی مقایسه می‌شوند.

اجرا: ``python -m src.models.run_phase9_feature_importance``
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.baselines import pinball_loss
from src.config import FIGURES_DIR, REPORTS_DIR, set_global_seed
from src.eda_lib.figio import save_fig
from src.models.axes import TUNING_TAU
from src.models.phase9_common import fit_champion
from src.viz_fa import fa
from src.viz_fa import setup as viz_setup

OUT_DIR = REPORTS_DIR / "phase9"
FIG_DIR = FIGURES_DIR / "phase9"
TAU = TUNING_TAU
N_REPEATS = 10
TOP_N = 15


def permutation_importance(model, Xte: pd.DataFrame, y: np.ndarray, tau: float,
                           n_repeats: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base_pred = model.predict(Xte)
    base_loss = pinball_loss(y, base_pred, tau).mean()

    rows = []
    for col in Xte.columns:
        deltas = []
        for i in range(n_repeats):
            Xperm = Xte.copy()
            shuffled = Xperm[col].sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1))).to_numpy()
            Xperm[col] = pd.Categorical(shuffled, categories=Xte[col].cat.categories) \
                if hasattr(Xte[col], "cat") else shuffled
            pred = model.predict(Xperm)
            loss = pinball_loss(y, pred, tau).mean()
            deltas.append(loss - base_loss)
        rows.append({"feature": col, "perm_importance_mean": float(np.mean(deltas)),
                     "perm_importance_std": float(np.std(deltas))})
    return pd.DataFrame(rows).sort_values("perm_importance_mean", ascending=False).reset_index(drop=True)


def main() -> None:
    viz_setup()
    set_global_seed()
    model, Xtr, Xte, train, test, cols = fit_champion(TAU)

    gain = pd.Series(model.booster_.feature_importance(importance_type="gain"), index=cols, name="gain")
    split = pd.Series(model.booster_.feature_importance(importance_type="split"), index=cols, name="split")
    tree_df = pd.concat([gain, split], axis=1).sort_values("gain", ascending=False)
    tree_df["gain_rank"] = tree_df["gain"].rank(ascending=False)
    tree_df["split_rank"] = tree_df["split"].rank(ascending=False)

    perm_df = permutation_importance(model, Xte, test["rho"].to_numpy(), TAU, N_REPEATS)
    perm_df["perm_rank"] = perm_df["perm_importance_mean"].rank(ascending=False)

    merged = tree_df.join(perm_df.set_index("feature"))
    rho_gain_perm, p_gain_perm = spearmanr(merged["gain_rank"], merged["perm_rank"])

    # نمودار: ۱۵ فیچر برتر هر دو روش
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    top_gain = tree_df.head(TOP_N)
    axes[0].barh(range(len(top_gain)), top_gain["gain"].to_numpy()[::-1], color="#4C72B0")
    axes[0].set_yticks(range(len(top_gain)))
    axes[0].set_yticklabels([fa(f) for f in top_gain.index[::-1]], fontsize=8)
    axes[0].set_title(fa("اهمیت درختی (gain) — ۱۵ برتر"))

    top_perm = perm_df.head(TOP_N)
    axes[1].barh(range(len(top_perm)), top_perm["perm_importance_mean"].to_numpy()[::-1],
                xerr=top_perm["perm_importance_std"].to_numpy()[::-1], color="#DD8452")
    axes[1].set_yticks(range(len(top_perm)))
    axes[1].set_yticklabels([fa(f) for f in top_perm["feature"][::-1]], fontsize=8)
    axes[1].set_title(fa("Permutation Importance (Δpinball) — ۱۵ برتر"))
    fig.tight_layout()
    save_fig(fig, "9.1_feature_importance.png", FIG_DIR)
    plt.close(fig)

    overlap = len(set(tree_df.head(TOP_N).index) & set(perm_df.head(TOP_N)["feature"]))

    lines = [
        "# بند ۹.۱ — اهمیت ویژگی روی قهرمان (`lightgbm_quantile`)",
        "",
        f"> τ={TAU}. داده: مرز holdout بند ۸.۱. اهمیت درختی سوگیری به کاردینالیتی بالا دارد "
        "(`RestaurantName`, `FoodType` دسته‌ای‌اند) — پس Permutation Importance مرجع اصلی است.",
        "",
        f"## ۱۵ فیچر برتر — اهمیت درختی (gain)",
        "",
        "| رتبه | فیچر | gain | split |",
        "|---|---|---|---|",
    ]
    for i, (feat, r) in enumerate(tree_df.head(TOP_N).iterrows(), 1):
        lines.append(f"| {i} | `{feat}` | {r['gain']:.1f} | {int(r['split'])} |")
    lines += [
        "",
        f"## ۱۵ فیچر برتر — Permutation Importance (Δpinball روی Test، میانگین {N_REPEATS} تکرار)",
        "",
        "| رتبه | فیچر | Δpinball | std |",
        "|---|---|---|---|",
    ]
    for i, r in perm_df.head(TOP_N).iterrows():
        lines.append(f"| {i+1} | `{r['feature']}` | {r['perm_importance_mean']:+.5f} | {r['perm_importance_std']:.5f} |")
    lines += [
        "",
        f"## مقایسه‌ی دو رتبه‌بندی",
        "",
        f"همبستگی رتبه‌ای Spearman (کل {len(cols)} فیچر): ρ={rho_gain_perm:.3f} (p={p_gain_perm:.2e}). "
        f"همپوشانی ۱۵ فیچر برتر: {overlap}/{TOP_N}.",
        "",
        f"نمودار: `reports/figures/phase9/9.1_feature_importance.png`",
    ]

    report = "\n".join(lines)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "9.1_feature_importance.md").write_text(report + "\n")
    merged.to_csv(OUT_DIR / "9.1_feature_importance.csv")
    print(report)
    print(f"\nذخیره شد در {OUT_DIR}/9.1_feature_importance.md")


if __name__ == "__main__":
    main()
