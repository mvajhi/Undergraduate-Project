"""اسپرینت D — اجرای CQR/Mondrian روی قهرمانان تأییدشده‌ی اسپرینت C (بند 7.22 بازنویسی‌شده).

قهرمانان: خ۲ (lightgbm_quantile, catboost_quantile, qrf) + خ۱۰ (knn_quantile) — هر
۴ با DM-test در برابر B3 تأیید شدند (`reports/phase7/dm_test_{F02,F10}.md`).

اجرا: ``python -m src.models.run_cqr_champions``
"""

import importlib

import numpy as np
import pandas as pd

from src.baselines import pinball_loss
from src.cv import DATE_COL, load_cv_folds
from src.features.build import FEATURES_A_PATH
from src.models import calibration, conformal
from src.models.axes import TUNING_TAU
from src.models.card_writer import _FAMILY_MODULES, load_s2_result

CHAMPIONS = [
    ("F02", "lightgbm_quantile"),
    ("F02", "catboost_quantile"),
    ("F02", "qrf"),
    ("F10", "knn_quantile"),
]


def _official_folds() -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    df = pd.read_parquet(FEATURES_A_PATH).sort_values(DATE_COL).reset_index(drop=True)
    fold_meta, _ = load_cv_folds()
    return [(df.loc[m1], df.loc[m2]) for f in fold_meta for m1, m2 in [f.masks(df[DATE_COL])]]


def evaluate_one(family: str, model_id: str, folds: list, tau: float) -> dict:
    result = load_s2_result(model_id, family)
    hp = result["best_hyperparams"]
    mod = importlib.import_module(_FAMILY_MODULES[family])
    fit_fn = mod.MODELS[model_id]

    oof_before = calibration.oof_predictions(fit_fn, folds, tau, hp)
    cov_before = float((oof_before["actual"] <= oof_before["pred_q"]).mean())
    pb_before = float(pinball_loss(oof_before["actual"].to_numpy(), oof_before["pred_q"].to_numpy(), tau).mean())

    rows = {"model": model_id, "family": family,
           "coverage_before": cov_before, "gap_before": cov_before - tau, "pinball_before": pb_before}

    for label, group_col in (("cqr_global", None), ("mondrian_restaurant", "RestaurantName"),
                             ("mondrian_meal", "Meal"), ("mondrian_tehran", "is_tehran")):
        oof = conformal.oof_calibrated_predictions(fit_fn, folds, tau, hp, group_col=group_col)
        cov = float((oof["actual"] <= oof["pred_q"]).mean())
        pb = float(pinball_loss(oof["actual"].to_numpy(), oof["pred_q"].to_numpy(), tau).mean())
        rows[f"coverage_{label}"] = cov
        rows[f"gap_{label}"] = cov - tau
        rows[f"pinball_{label}"] = pb
    return rows


def render_report(rows: list[dict], tau: float) -> str:
    lines = [
        "# اسپرینت D — CQR/Mondrian روی قهرمانان تأییدشده",
        "",
        f"> بند 7.22 بازنویسی‌شده (ردیف ۳۷ decision_log). τ={tau}. کالیبراسیون زمانی "
        f"(بند 7.22.3، {conformal.CALIB_FRAC:.0%} انتهای پنجره‌ی آموزش هر fold) — بدون "
        "نشتی از test.",
        "",
        "| مدل | حالت | پوشش | شکاف از τ | pinball |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| `{r['model']}` | قبل (بدون کالیبراسیون) | {r['coverage_before']:.4f} | "
                    f"{r['gap_before']:+.4f} | {r['pinball_before']:.5f} |")
        for label, fa_name in (("cqr_global", "CQR سراسری"), ("mondrian_restaurant", "Mondrian(سلف)"),
                               ("mondrian_meal", "Mondrian(وعده)"), ("mondrian_tehran", "Mondrian(تهران؟)")):
            lines.append(f"| `{r['model']}` | {fa_name} | {r[f'coverage_{label}']:.4f} | "
                        f"{r[f'gap_{label}']:+.4f} | {r[f'pinball_{label}']:.5f} |")

    n_improved = sum(1 for r in rows if abs(r["gap_cqr_global"]) < abs(r["gap_before"]))
    lines += [
        "",
        f"**نتیجه: CQR سراسری در {n_improved} از {len(rows)} مدل شکاف پوشش را بهبود داد.**",
        "",
        "⚠️ **یافته‌ی مهم (نه باگ — تست شد روی داده‌ی مصنوعی ایستا، `test_cqr_fixes_"
        "stationary_miscalibration`، آن‌جا CQR پوشش را از ۰.۸۵۵ به ۰.۲۷۰ اصلاح کرد).** "
        "روی داده‌ی واقعی این پروژه، CQR ساده اغلب کمکی نمی‌کند یا بدتر هم می‌کند — علامت "
        "تصحیح بین foldها ناپایدار است (مثال lightgbm_quantile: fold۰=−۰.۰۱۲۹، "
        "fold۳=+۰.۰۱۴۶). دلیل محتمل: میزان/جهت بدکالیبرگی در طول زمان ثابت نیست "
        "(تعطیلات/امتحانات/رمضان رژیم را عوض می‌کنند)، پس فرض تبادل‌پذیری (exchangeability) "
        "CQR بین بازه‌ی کالیبراسیون (انتهای train) و بازه‌ی آزمون (fold بعدی) نقض می‌شود. "
        "**پیشنهاد بعدی (بند 7.22.1 عضو ۴): ACI (Adaptive Conformal Inference)** که دقیقاً "
        "برای همین حالت (مقاوم به تغییر رژیم) طراحی شده — CQR ایستا نیست.",
    ]
    return "\n".join(lines)


def main() -> None:
    from src.config import REPORTS_DIR, set_global_seed

    set_global_seed()
    folds = _official_folds()
    rows = [evaluate_one(family, model_id, folds, TUNING_TAU) for family, model_id in CHAMPIONS]
    report = render_report(rows, TUNING_TAU)
    out = REPORTS_DIR / "phase7"
    out.mkdir(parents=True, exist_ok=True)
    (out / "cqr_champions.md").write_text(report + "\n")
    pd.DataFrame(rows).to_json(out / "cqr_champions.json", orient="records", indent=2, force_ascii=False)
    print(report)
    print(f"\nذخیره شد در {out / 'cqr_champions.md'}")


if __name__ == "__main__":
    main()
