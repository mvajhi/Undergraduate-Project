"""اسپرینت D — ترکیب کوانتایلی قهرمانان (بند 7.21، خ۱۲) روی خروجی **کالیبره‌شده‌ی
ACI** (یافته‌ی ۲۲، `doc/progress/07-*.md`).

## بند 7.21.2 — چرا میانگین‌گیری ساده اینجا درست است

هشدار WBS: «میانگین کوانتایل ≠ کوانتایل میانگین» برای وقتی است که چند **کوانتایل
مختلف** یا **توزیع کامل** ترکیب می‌شوند (Vincentization واقعی: میانگین تابع کوانتایل
روی چند τ). اینجا هر ۴ مدل فقط **یک** τ=۰.۲۰ می‌دهند — میانگین‌گیری ساده‌ی
$\\hat\\rho_{\\tau}$ چند برآوردگر مستقل از همان τ، دقیقاً همان چیزی است که Vincentization
در حالت تک‌τ به آن تحویل می‌شود؛ مشکلی که هشدار درباره‌اش است (پهن/باریک‌شدن کاذب
توزیع) وقتی رخ می‌دهد که τهای مختلف را جابه‌جا کنیم، نه اینجا.

اجرا: ``python -m src.models.run_ensemble_champions``
"""

import importlib

import numpy as np
import pandas as pd

from src.baselines import b3_empirical_quantile, pinball_loss
from src.cv import DATE_COL, load_cv_folds
from src.features.build import FEATURES_A_PATH
from src.models import conformal
from src.models.axes import TUNING_TAU
from src.models.card_writer import _FAMILY_MODULES, load_s2_result
from src.models.run_cqr_champions import CHAMPIONS


def _official_folds() -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    df = pd.read_parquet(FEATURES_A_PATH).sort_values(DATE_COL).reset_index(drop=True)
    fold_meta, _ = load_cv_folds()
    return [(df.loc[m1], df.loc[m2]) for f in fold_meta for m1, m2 in [f.masks(df[DATE_COL])]]


def champion_aci_predictions(folds: list, tau: float) -> dict[str, np.ndarray]:
    """پیش‌بینی ACI-کالیبره‌شده‌ی هر قهرمان روی هر ۵ fold، به ترتیب ردیف یکسان
    (چون همه از همان ``folds`` با همان ترتیب concatenate می‌شوند)."""
    preds = {}
    for family, model_id in CHAMPIONS:
        result = load_s2_result(model_id, family)
        mod = importlib.import_module(_FAMILY_MODULES[family])
        fit_fn = mod.MODELS[model_id]
        oof = conformal.oof_aci_predictions(fit_fn, folds, tau, result["best_hyperparams"])
        preds[model_id] = oof["pred_q"].to_numpy()
    return preds


def evaluate_ensembles(folds: list, tau: float) -> pd.DataFrame:
    preds = champion_aci_predictions(folds, tau)
    actual = np.concatenate([te["rho"].to_numpy() for _, te in folds])
    res = np.concatenate([te["Res"].to_numpy() for _, te in folds])
    b3_pred = np.concatenate([b3_empirical_quantile(tr, te, tau) for tr, te in folds])
    pb_b3 = float(pinball_loss(actual, b3_pred, tau).mean())

    rows = []
    for name, pred in preds.items():
        rows.append({"combo": name, "pinball": float(pinball_loss(actual, pred, tau).mean()),
                    "coverage": float((actual <= pred).mean())})

    model_ids = list(preds)
    stacked = np.column_stack([preds[m] for m in model_ids])
    mean_all = np.clip(stacked.mean(axis=1), 0.0, 1.0)
    rows.append({"combo": "میانگین هر ۴", "pinball": float(pinball_loss(actual, mean_all, tau).mean()),
                "coverage": float((actual <= mean_all).mean())})

    # میانگین ۳ برتر (بر اساس pinball تکی از reports/phase7/S2_tuning) — بند 7.21.1 عضو ۹ الگو
    ranked = sorted(model_ids, key=lambda m: pinball_loss(actual, preds[m], tau).mean())
    top3 = ranked[:3]
    mean_top3 = np.clip(np.column_stack([preds[m] for m in top3]).mean(axis=1), 0.0, 1.0)
    rows.append({"combo": f"میانگین ۳ برتر ({', '.join(top3)})",
                "pinball": float(pinball_loss(actual, mean_top3, tau).mean()),
                "coverage": float((actual <= mean_top3).mean())})

    df = pd.DataFrame(rows)
    df["gap"] = df["coverage"] - tau
    df["pinball_vs_B3"] = df["pinball"] - pb_b3
    return df, pb_b3


def render_report(df: pd.DataFrame, pb_b3: float, tau: float) -> str:
    lines = [
        "# اسپرینت D — ترکیب کوانتایلی قهرمانان (خ۱۲، بند 7.21)",
        "",
        f"> τ={tau}. ورودی: پیش‌بینی **ACI-کالیبره‌شده** هر ۴ قهرمان (یافته‌ی ۲۲) روی هر ۵ "
        f"fold رسمی. مرجع B3: pinball={pb_b3:.5f}.",
        "",
        "| ترکیب | pinball | Δ نسبت به B3 | پوشش | شکاف از τ |",
        "|---|---|---|---|---|",
    ]
    for _, r in df.sort_values("pinball").iterrows():
        mark = " 🎯" if r["pinball_vs_B3"] < 0 else ""
        lines.append(f"| `{r['combo']}`{mark} | {r['pinball']:.5f} | {r['pinball_vs_B3']:+.5f} | "
                    f"{r['coverage']:.4f} | {r['gap']:+.4f} |")

    best = df.loc[df["pinball"].idxmin()]
    lines += ["", f"**بهترین ترکیب: `{best['combo']}`** (pinball={best['pinball']:.5f})."]
    return "\n".join(lines)


def main() -> None:
    from src.config import REPORTS_DIR, set_global_seed

    set_global_seed()
    folds = _official_folds()
    df, pb_b3 = evaluate_ensembles(folds, TUNING_TAU)
    report = render_report(df, pb_b3, TUNING_TAU)
    out = REPORTS_DIR / "phase7"
    out.mkdir(parents=True, exist_ok=True)
    (out / "ensemble_champions.md").write_text(report + "\n")
    df.to_json(out / "ensemble_champions.json", orient="records", indent=2, force_ascii=False)
    print(report)
    print(f"\nذخیره شد در {out / 'ensemble_champions.md'}")


if __name__ == "__main__":
    main()
