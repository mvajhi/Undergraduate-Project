"""بند ۸.۶ WBS فاز ۸ — پایداری و حساسیت، روی Test قفل‌شده (بند ۸.۱).

قهرمان رسمی (`lightgbm_quantile`). چهار زیربند:

1. **پایداری بین fold** — از فاز ۷ نقل می‌شود (بند 7.6.3، `reports/models/lightgbm_quantile.md`)،
   نه بازمحاسبه — Test یک پنجره‌ی واحد است، fold ندارد که رویش std بگیریم.
2. **حساسیت seed** — ۵ seed، بازبرازش واقعی روی همان مرز holdout (feature_fraction=0.92
   در هایپرپارامتر قهرمان یعنی seed واقعاً اثر دارد، برخلاف مدل‌های بدون subsampling).
3. **آزمون تنش** — حذف فیچرهای هواشناسی (`temp_min`, `precip_type`, `is_snow_day`، هر
   سه در فیچرست ۶۲ستونی قهرمان حاضرند) و بازبرازش، شبیه‌سازی خرابی/قطع فید هوا.
4. **حساسیت τ** — به بند ۸.۵ ارجاع داده می‌شود (`8.5_tradeoff_table.csv`)، دوباره
   محاسبه نمی‌شود.

اجرا: ``python -m src.models.run_phase8_stability``
"""

import importlib

import numpy as np
import pandas as pd

from src.baselines import operational_metrics, pinball_loss
from src.config import REPORTS_DIR
from src.models.axes import TUNING_TAU
from src.models.card_writer import load_s2_result
from src.models.families import common
from src.models.run_phase8_holdout_eval import load_holdout

OUT_DIR = REPORTS_DIR / "phase8"
CHAMPION = "lightgbm_quantile"
TAU = TUNING_TAU
SEEDS = [42, 1, 7, 123, 2024]
WEATHER_COLS = ["temp_min", "precip_type", "is_snow_day"]


def _raw_categorical_design(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]
                            ) -> tuple[pd.DataFrame, pd.DataFrame]:
    tr, te = train[cols].copy(), test[cols].copy()
    for c in cols:
        if tr[c].dtype == object:
            tr[c] = tr[c].astype("category")
            te[c] = pd.Categorical(te[c], categories=tr[c].cat.categories)
    return tr, te


def _fit_lightgbm(train, test, tau, cols, seed, hp):
    import lightgbm as lgb
    Xtr, Xte = _raw_categorical_design(train, test, cols)
    model = lgb.LGBMRegressor(objective="quantile", alpha=tau, num_leaves=hp["num_leaves"],
                              learning_rate=hp["learning_rate"], min_child_samples=hp["min_child_samples"],
                              feature_fraction=hp["feature_fraction"], n_estimators=hp["n_estimators"],
                              verbosity=-1, random_state=seed)
    model.fit(Xtr, train["rho"])
    return np.clip(model.predict(Xte), 0.0, 1.0)


def seed_sensitivity(train, test, cols, hp) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        pred = _fit_lightgbm(train, test, TAU, cols, seed, hp)
        m = operational_metrics(test, pred, TAU)
        rows.append({"seed": seed, "pinball": m["pinball"], "shortage_rate": m["shortage_rate"],
                     "coverage": m["coverage"]})
    return pd.DataFrame(rows)


def stress_test(train, test, cols, hp) -> dict:
    reduced_cols = [c for c in cols if c not in WEATHER_COLS]
    pred_full = _fit_lightgbm(train, test, TAU, cols, 42, hp)
    pred_reduced = _fit_lightgbm(train, test, TAU, reduced_cols, 42, hp)
    m_full = operational_metrics(test, pred_full, TAU)
    m_reduced = operational_metrics(test, pred_reduced, TAU)

    from src.cv import DATE_COL, diebold_mariano
    dm_stat, dm_p = diebold_mariano(
        pinball_loss(test["rho"].to_numpy(), pred_reduced, TAU),
        pinball_loss(test["rho"].to_numpy(), pred_full, TAU))
    return {"full": m_full, "reduced": m_reduced, "n_dropped": len(WEATHER_COLS),
            "dm_stat": dm_stat, "dm_p": dm_p}


def main() -> None:
    train, test, fold = load_holdout()
    result = load_s2_result(CHAMPION, "F02")
    hp = result["best_hyperparams"]
    mod = importlib.import_module("src.models.families.f02_tree")
    cols = mod._feature_cols_s2()

    seeds_df = seed_sensitivity(train, test, cols, hp)
    stress = stress_test(train, test, cols, hp)

    fold_stability_note = (
        "طبق `reports/models/lightgbm_quantile.md` (بند 7.6.3، جدول پایداری): "
        "بهترین trial S2 در **۰ از ۵ fold** رسمی بین ۱۰ trial برتر تکرار نشد — قهرمان "
        "با DM-test مستقیم (یافته‌ی ۱۵) انتخاب شد، نه با معیار پایداری fold. این بدهی "
        "مستند از فاز ۷ است، نه یافته‌ی جدید اینجا."
    )

    lines = [
        "# بند ۸.۶ — پایداری و حساسیت روی Test قفل‌شده",
        "",
        f"> مدل: `{CHAMPION}`. τ={TAU}. داده: پنجره‌ی Test بند ۸.۱.",
        "",
        "## پایداری بین fold (نقل از فاز ۷)",
        "",
        fold_stability_note,
        "",
        "## حساسیت seed (۵ seed، بازبرازش واقعی)",
        "",
        "| seed | pinball | نرخ کمبود | پوشش |",
        "|---|---|---|---|",
    ]
    for _, r in seeds_df.iterrows():
        lines.append(f"| {int(r['seed'])} | {r['pinball']:.5f} | {r['shortage_rate']:.1%} | {r['coverage']:.1%} |")
    pb_range = seeds_df["pinball"].max() - seeds_df["pinball"].min()
    lines += [
        "",
        f"دامنه‌ی pinball بین ۵ seed: {pb_range:.5f} (میانگین={seeds_df['pinball'].mean():.5f}، "
        f"std={seeds_df['pinball'].std():.5f})",
        "",
        "## آزمون تنش — حذف فیچرهای هواشناسی (شبیه‌سازی قطع فید هوا)",
        "",
        f"حذف‌شده ({stress['n_dropped']} ستون): `{'`, `'.join(WEATHER_COLS)}`",
        "",
        "| حالت | pinball | نرخ کمبود | پوشش |",
        "|---|---|---|---|",
        f"| کامل | {stress['full']['pinball']:.5f} | {stress['full']['shortage_rate']:.1%} | {stress['full']['coverage']:.1%} |",
        f"| بدون هوا | {stress['reduced']['pinball']:.5f} | {stress['reduced']['shortage_rate']:.1%} | {stress['reduced']['coverage']:.1%} |",
        "",
        f"Δpinball={stress['reduced']['pinball']-stress['full']['pinball']:+.5f}، "
        f"DM آماره={stress['dm_stat']:.3f}، p={stress['dm_p']:.4f} — "
        f"{'افت معنادار' if stress['dm_p'] < 0.05 and stress['reduced']['pinball'] > stress['full']['pinball'] else 'افت معنادار نیست'}.",
        "",
        "## حساسیت τ",
        "",
        "به بند ۸.۵ ارجاع داده می‌شود (`reports/phase8/8.5_tradeoff_table.csv`) — "
        "همان بازبرازش روی همین Test، دوباره محاسبه نشد.",
    ]

    report = "\n".join(lines)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "8.6_stability_sensitivity.md").write_text(report + "\n")
    seeds_df.to_csv(OUT_DIR / "8.6_seed_sensitivity.csv", index=False)
    print(report)
    print(f"\nذخیره شد در {OUT_DIR}/8.6_stability_sensitivity.md")


if __name__ == "__main__":
    main()
