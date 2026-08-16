"""اسپرینت D — حساسیت τ روی قهرمان نهایی (بند 7.21، چک‌لیست خط ۱۶۰).

قهرمان نهایی = ``lightgbm_quantile`` (F02) — طبق یافته‌ی ۲۳ تنها مدلی که در pinball از
ترکیب‌ها هم بهتر بود. هایپرپارامترهای آن فقط روی τ=۰.۲۰ تنظیم شده‌اند (``S2_tuning_F02.json``؛
دقیقاً همان مشکلی که به‌عنوان اشتباه اول در `doc/decisions/37-phase7-rescope.md` مستند شده:
«τ تنظیم جدا از تصمیم پروژه»). این اسکریپت **بدون تنظیم مجدد**، همان هایپرپارامترها را روی
کل شبکه‌ی ``TAU_GRID`` امتحان می‌کند تا معلوم شود آیا برتری نسبت به B3 در سایر τها هم پایدار
می‌ماند یا مختص τ=۰.۲۰ است — و کالیبراسیون ACI (یافته‌ی ۲۲) را هم در هر τ دوباره اعمال می‌کند
چون تصحیح ACI خودش تابعی از τ است.

اجرا: ``python -m src.models.tau_sensitivity``
"""

import numpy as np
import pandas as pd

from src.baselines import b3_empirical_quantile, pinball_loss
from src.cv import DATE_COL, load_cv_folds
from src.features.build import FEATURES_A_PATH
from src.models import conformal
from src.models.axes import TAU_GRID
from src.models.card_writer import load_s2_result
from src.models.families.f02_tree import MODELS

FINAL_CHAMPION_MODEL_ID = "lightgbm_quantile"
FINAL_CHAMPION_FAMILY = "F02"


def _official_folds() -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    df = pd.read_parquet(FEATURES_A_PATH).sort_values(DATE_COL).reset_index(drop=True)
    fold_meta, _ = load_cv_folds()
    return [(df.loc[m1], df.loc[m2]) for f in fold_meta for m1, m2 in [f.masks(df[DATE_COL])]]


def evaluate_tau_grid(folds: list, hyperparams: dict) -> pd.DataFrame:
    fit_fn = MODELS[FINAL_CHAMPION_MODEL_ID]
    rows = []
    for tau in TAU_GRID:
        oof = conformal.oof_aci_predictions(fit_fn, folds, tau, hyperparams)
        actual = oof["actual"].to_numpy()
        pred = oof["pred_q"].to_numpy()
        b3_pred = np.concatenate([np.asarray(b3_empirical_quantile(tr, te, tau)) for tr, te in folds])
        pb = float(pinball_loss(actual, pred, tau).mean())
        pb_b3 = float(pinball_loss(actual, b3_pred, tau).mean())
        coverage = float((actual <= pred).mean())
        rows.append({
            "tau": tau,
            "pinball": pb,
            "pinball_B3": pb_b3,
            "delta_vs_B3": pb - pb_b3,
            "beats_B3": pb < pb_b3,
            "coverage": coverage,
            "gap": coverage - tau,
        })
    return pd.DataFrame(rows)


def render_report(df: pd.DataFrame, hyperparams: dict) -> str:
    lines = [
        "# اسپرینت D — حساسیت τ روی قهرمان نهایی",
        "",
        f"> قهرمان: `{FINAL_CHAMPION_MODEL_ID}` ({FINAL_CHAMPION_FAMILY}), هایپرپارامتر ثابت "
        f"(تنظیم‌شده فقط روی τ=۰.۲۰): `{hyperparams}`. کالیبراسیون ACI در هر τ جداگانه اعمال شد.",
        "",
        "| τ | pinball | pinball B3 | Δ نسبت به B3 | پوشش | شکاف از τ |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        mark = " ✅" if r["beats_B3"] else " ❌"
        lines.append(f"| {r['tau']:.2f} | {r['pinball']:.5f} | {r['pinball_B3']:.5f} | "
                    f"{r['delta_vs_B3']:+.5f}{mark} | {r['coverage']:.4f} | {r['gap']:+.4f} |")

    n_beats = int(df["beats_B3"].sum())
    lines += ["", f"**برتری نسبت به B3 در {n_beats} از {len(df)} τ حفظ شد.**"]
    if n_beats == len(df):
        lines.append("هایپرپارامترهای تنظیم‌شده روی τ=۰.۲۰ به کل شبکه‌ی τ تعمیم می‌یابند — "
                    "نیازی به تنظیم جداگانه‌ی هر τ نیست.")
    else:
        missed = df.loc[~df["beats_B3"], "tau"].tolist()
        lines.append(f"در τ={missed} برتری از دست رفت — هایپرپارامتر τ=۰.۲۰ برای این نقاط "
                    "بهینه نیست؛ اگر تصمیم عملیاتی به این τها هم نیاز داشت، تنظیم جداگانه لازم است.")
    return "\n".join(lines)


def main() -> None:
    from src.config import REPORTS_DIR, set_global_seed

    set_global_seed()
    folds = _official_folds()
    result = load_s2_result(FINAL_CHAMPION_MODEL_ID, FINAL_CHAMPION_FAMILY)
    hyperparams = result["best_hyperparams"]
    df = evaluate_tau_grid(folds, hyperparams)
    report = render_report(df, hyperparams)
    out = REPORTS_DIR / "phase7"
    out.mkdir(parents=True, exist_ok=True)
    (out / "tau_sensitivity_champion.md").write_text(report + "\n")
    df.to_json(out / "tau_sensitivity_champion.json", orient="records", indent=2, force_ascii=False)
    print(report)
    print(f"\nذخیره شد در {out / 'tau_sensitivity_champion.md'}")


if __name__ == "__main__":
    main()
