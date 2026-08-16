"""اسپرینت A، بند ۱ سند تصمیم ۳۷ — «آزمون بند ۶.۶ روی نتایج موجود خ۱ اجرا شد».

یافته‌ی ۱۱ (`doc/progress/07-*.md`) نشان داد سه «برنده»ی S2 خ۱ (نسبت به B3) margin
ناچیز و پایداری بسیار پایین دارند — این ماژول ادعای برد را با آزمون Diebold-Mariano
(بند ۶.۶، `src/cv.py::diebold_mariano`) روی زیان **هر ردیف** (نه میانگین fold) واقعاً
می‌آزماید. همچنین B6 (تعریف‌شده در بند ۶.۵ ولی هرگز روی foldهای رسمی فاز ۷ اجرا نشده)
را به‌عنوان مرجع دوم اضافه می‌کند.

⚠️ برخلاف مقایسه‌ی pinball میانگین (که فقط ۵ عدد fold-level دارد)، DM-test روی
بردار زیان تک‌تک ردیف‌ها کار می‌کند — دقیقاً چیزی که بند ۶.۶ برای «معنادار بودن»
می‌خواهد، نه فقط «بزرگ‌تر بودن».
"""

import json

import numpy as np
import pandas as pd

from src.baselines import b3_empirical_quantile, b6_day_factor, pinball_loss, quantile_adjust
from src.cv import diebold_mariano
from src.models.axes import TUNING_TAU
from src.models.calibration import oof_predictions

#: سه «برنده»ی S2 خ۱ که ادعای برد روی B3 دارند (یافته‌ی ۷/۱۱ doc/progress)
F01_WINNERS = ("l1_quantile_regression", "elasticnet", "lasso")


def _load_official_folds() -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    from src.cv import DATE_COL, load_cv_folds
    from src.features.build import FEATURES_A_PATH

    df = pd.read_parquet(FEATURES_A_PATH).sort_values(DATE_COL).reset_index(drop=True)
    fold_meta, _ = load_cv_folds()
    folds = []
    for f in fold_meta:
        tr_mask, te_mask = f.masks(df[DATE_COL])
        folds.append((df.loc[tr_mask], df.loc[te_mask]))
    return folds


def baseline_row_losses(folds: list, tau: float, name: str) -> np.ndarray:
    """زیان pinball هر ردیف out-of-fold برای B3 یا B6 روی foldهای رسمی."""
    fn = {"B3": b3_empirical_quantile, "B6": b6_day_factor}[name]
    mean_targeting = name == "B6"
    parts = []
    for tr, te in folds:
        rho_hat = fn(tr, te, tau)
        if mean_targeting:
            rho_hat = quantile_adjust(rho_hat, tr, tau, fn)
        parts.append(pinball_loss(te["rho"].to_numpy(), rho_hat, tau))
    return np.concatenate(parts)


def model_row_losses(model_id: str, family: str, folds: list, tau: float) -> np.ndarray:
    """زیان pinball هر ردیف out-of-fold یک مدل F01، با بهترین هایپرپارامتر S2 (بازبرازش واقعی)."""
    import importlib

    from src.models.card_writer import _FAMILY_MODULES, load_s2_result

    result = load_s2_result(model_id, family)
    mod = importlib.import_module(_FAMILY_MODULES[family])
    fit_fn = mod.MODELS[model_id]
    oof = oof_predictions(fit_fn, folds, tau, result["best_hyperparams"])
    return pinball_loss(oof["actual"].to_numpy(), oof["pred_q"].to_numpy(), tau)


def run_significance(family: str, model_ids: tuple[str, ...], tau: float = TUNING_TAU) -> pd.DataFrame:
    """برای هر مدل: DM-test در برابر B3 **و** B6 روی همان ۵ fold رسمی — عمومی، برای هر
    خانواده (نه فقط خ۱)."""
    folds = _load_official_folds()
    losses_b3 = baseline_row_losses(folds, tau, "B3")
    losses_b6 = baseline_row_losses(folds, tau, "B6")

    rows = []
    for mid in model_ids:
        losses_m = model_row_losses(mid, family, folds, tau)
        for ref_name, losses_ref in (("B3", losses_b3), ("B6", losses_b6)):
            dm, p = diebold_mariano(losses_m, losses_ref)
            rows.append({
                "model": mid, "reference": ref_name,
                "pinball_model": float(losses_m.mean()),
                "pinball_reference": float(losses_ref.mean()),
                "delta_pinball": float(losses_m.mean() - losses_ref.mean()),
                "dm_stat": dm, "p_value": p,
                "significant_at_0.05": bool(np.isfinite(p) and p < 0.05 and losses_m.mean() < losses_ref.mean()),
            })
    return pd.DataFrame(rows)


def run_f01_significance(tau: float = TUNING_TAU) -> pd.DataFrame:
    return run_significance("F01", F01_WINNERS, tau)


def render_report(df: pd.DataFrame, tau: float, family: str = "F01") -> str:
    lines = [
        f"# آزمون معناداری S2 {family} — Diebold-Mariano در برابر B3 و B6",
        "",
        f"> اسپرینت A/C، بند ۱ سند تصمیم `doc/decisions/37-phase7-rescope.md`. τ={tau}، هر ۵ "
        "fold رسمی (`cv_folds.json`)، زیان **هر ردیف** (نه میانگین fold) — بند ۶.۶.",
        "",
        f"⚠️ **وزن‌دهی pinball اینجا با `S2_tuning_{family}.md` فرق دارد و عمدی است.** "
        "جدول‌های S2 میانگین pinball را با **وزن برابر برای هر fold** گزارش می‌کنند — "
        "fold۲ فقط ۱۸۵ ردیف دارد (بازه‌ی شکاف رمضان) ولی وزن کامل می‌گیرد و میانگین را بالا "
        "می‌کشد. آزمون DM اما به یک بردار زیان **تک‌ردیفی** نیاز دارد، پس اینجا همه‌ی "
        "۳٬۸۸۰ ردیف با وزن برابر تجمیع شده‌اند — عدد B3/B6 پایین‌تر از S2_tuning است، "
        "نه خطا، فقط سؤال متفاوت («میانگین هر ردیف» به‌جای «میانگین هر fold»).",
        "",
        "| مدل | مرجع | pinball مدل | pinball مرجع | Δ | آماره‌ی DM | p-value | معنادار (۰.۰۵)؟ |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        sig = "✅ بله" if r["significant_at_0.05"] else "❌ خیر"
        lines.append(
            f"| `{r['model']}` | {r['reference']} | {r['pinball_model']:.5f} | "
            f"{r['pinball_reference']:.5f} | {r['delta_pinball']:+.5f} | {r['dm_stat']:.3f} | "
            f"{r['p_value']:.4f} | {sig} |"
        )
    n_sig = int(df["significant_at_0.05"].sum())
    vs_b3 = df[df["reference"] == "B3"]
    vs_b6 = df[df["reference"] == "B6"]
    lines += [
        "",
        f"**نتیجه: {n_sig} از {len(df)} مقایسه معنادار (p<۰.۰۵) به‌نفع مدل بود.**",
        "",
        f"- در برابر **B3**: {int(vs_b3['significant_at_0.05'].sum())}/{len(vs_b3)} معنادار.",
        f"- در برابر **B6**: {int(vs_b6['significant_at_0.05'].sum())}/{len(vs_b6)} معنادار "
        f"(B6 خودش pinball={vs_b6['pinball_reference'].iloc[0]:.5f} در برابر B3="
        f"{vs_b3['pinball_reference'].iloc[0]:.5f} دارد — قبل از تفسیر برد در برابر B6، "
        "این عدد را با B3 مقایسه کنید).",
    ]
    return "\n".join(lines)


def main(family: str = "F01", model_ids: tuple[str, ...] = F01_WINNERS) -> None:
    from src.config import REPORTS_DIR, set_global_seed

    set_global_seed()
    df = run_significance(family, model_ids)
    report = render_report(df, TUNING_TAU, family)
    out_dir = REPORTS_DIR / "phase7"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"dm_test_{family}.md").write_text(report + "\n")
    df.to_json(out_dir / f"dm_test_{family}.json", orient="records", indent=2, force_ascii=False)
    print(report)
    print(f"\nذخیره شد در {out_dir / f'dm_test_{family}.md'}")


if __name__ == "__main__":
    main()
