"""بند 7.23.1 — قاعده‌ی ضدتقاطع، روی اعضای مسیر Q1 خ۱ که هر τ را **جدا** برازش می‌کنند.

فقط `QUANTREG_MODEL_IDS` (بند 7.5.3): `quantile_regression`، `l1_quantile_regression`،
`composite_quantile_regression`، `expectile_regression`. بقیه‌ی اعضای خ۱ مسیر Q2/Q3
دارند (کوانتایل از توزیع پارامتری یا باقیمانده) و ذاتاً تک‌τ اجرا می‌شوند — قاعده‌ی
ضدتقاطع فقط برای Q1 معنا دارد (بند 7.23.1: «برای مسیر Q1 با چند τ جداگانه»).

⚠️ **روی fold۰ تنها** اجرا می‌شود، نه هر ۵ fold — این یک آزمون **ساختاری/مکانیکی**
است (آیا مدل به‌ازای همان هایپرپارامتر، پیش‌بینی‌های ناسازگار بین τها می‌دهد؟)، نه یک
معیار عملکردی که به استحکام CV نیاز داشته باشد؛ همان سطح هزینه‌ای که S0 برای آزمون‌های
مکانیکی می‌پذیرد (بند 7.3.1). هایپرپارامتر هر مدل از S2 (تنظیم‌شده فقط روی τ=۰.۲۰،
بند بالای `tau_sensitivity.py`) بدون تنظیم مجدد روی کل `TAU_GRID` اعمال می‌شود.

اجرا: ``python -m src.models.quantile_crossing``
"""

import numpy as np
import pandas as pd

from src.baselines import pinball_loss
from src.cv import DATE_COL, load_cv_folds
from src.features.build import FEATURES_A_PATH
from src.models.axes import TAU_GRID
from src.models.card_writer import load_s2_result
from src.models.families.f01_linear import MODELS, QUANTREG_MODEL_IDS

FAMILY = "F01"
#: بند 7.23.1: «اگر >۱٪» — آستانه‌ی رسمی
VIOLATION_THRESHOLD = 0.01


def _fold0() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet(FEATURES_A_PATH).sort_values(DATE_COL).reset_index(drop=True)
    fold_meta, _ = load_cv_folds()
    tr_mask, te_mask = fold_meta[0].masks(df[DATE_COL])
    return df.loc[tr_mask], df.loc[te_mask]


#: 🐛 **باگ کشف‌شده هنگام نوشتن این ماژول.** `ModelS2Result.best_hyperparams` برای
#: `composite_quantile_regression` مقدار `trial.params` خام Optuna را نگه می‌دارد
#: (`{"taus_grid": "wide"}`) نه دیکشنری واقعی‌ای که `_space_composite_quantile_
#: regression` به تابع مدل می‌دهد (`{"taus": (0.02,…,0.25)}`) — چون فضای جستجو یک
#: پارامتر دسته‌ای (`taus_grid`) را به یک تاپل نگاشت می‌کند و فقط مقدار خام ثبت
#: می‌شود. نتیجه: هر بازبرازشی که مستقیم `**best_hyperparams` را به تابع مدل بدهد
#: (`card_writer` گام ۱۱/۱۳، `model_comparison.py`، این ماژول) بی‌سروصدا به شبکه‌ی
#: پیش‌فرض تابع (`taus=(0.05,0.10,0.15,0.20)`) سقوط می‌کند، نه شبکه‌ی واقعاً برنده‌ی
#: S2 (`wide`). خودِ S2 درست بود (هر trial با شکل صحیح صدا زده شد)؛ فقط ذخیره‌سازی
#: برای بازتولید بعدی ناقص است. رفعش نیازمند تغییر `s2_runner.py` (زیرساخت مشترک،
#: خ۱ منجمد است — ردیف ۱۲ decision_log) است، پس اینجا فقط **دور زده** می‌شود، نه
#: رفع ریشه‌ای — یافته‌ی گزارش‌شده در `doc/decision_log.md`.
_COMPOSITE_TAUS_GRIDS = {"narrow": (0.05, 0.10, 0.15, 0.20),
                         "wide": (0.02, 0.05, 0.10, 0.15, 0.20, 0.25)}


def _resolve_hp(model_id: str, hp: dict) -> dict:
    if model_id == "composite_quantile_regression" and "taus_grid" in hp:
        return {"taus": _COMPOSITE_TAUS_GRIDS[hp["taus_grid"]]}
    return hp


def predict_tau_grid(model_id: str, train: pd.DataFrame, test: pd.DataFrame,
                     hp: dict) -> pd.DataFrame:
    """پیش‌بینی همان مدل با همان هایپرپارامتر روی هر τ شبکه — یک ستون به‌ازای هر τ."""
    fit_fn = MODELS[model_id]
    hp = _resolve_hp(model_id, hp)
    preds = {tau: np.asarray(fit_fn(train, test, tau, **hp), dtype=float) for tau in TAU_GRID}
    return pd.DataFrame(preds, index=test.index)


def crossing_violations(pred_grid: pd.DataFrame, taus: tuple[float, ...] = TAU_GRID,
                        eps: float = 1e-9) -> tuple[pd.Series, dict]:
    """بند 7.23.1: به‌ازای هر جفت τ **مجاور**، ٪ ردیف‌هایی که پیش‌بینی τ کوچک‌تر از
    پیش‌بینی τ بزرگ‌تر بیشتر شده — به‌علاوه‌ی نرخ کلی (هر ردیف با حداقل یک نقض)."""
    any_violation = pd.Series(False, index=pred_grid.index)
    per_pair = {}
    for lo, hi in zip(taus[:-1], taus[1:]):
        bad = pred_grid[lo] > pred_grid[hi] + eps
        per_pair[f"{lo}->{hi}"] = float(bad.mean())
        any_violation |= bad
    return any_violation, per_pair


def isotonic_fix(pred_grid: pd.DataFrame, taus: tuple[float, ...] = TAU_GRID) -> pd.DataFrame:
    """گزینه‌ی «ب» بند 7.23.1: مرتب‌سازی پس‌پردازشی (isotonic) — هر ردیف را در طول
    محور τ به‌ترتیب صعودی برمی‌گرداند (``np.sort``، ساده‌ترین شکل isotonic برای یک
    شبکه‌ی گسسته‌ی τ ثابت — معادل PAVA وقتی فقط باید غیرنزولی شود، نه وزن‌دار)."""
    sorted_vals = np.sort(pred_grid[list(taus)].to_numpy(), axis=1)
    return pd.DataFrame(sorted_vals, columns=list(taus), index=pred_grid.index)


def evaluate_model(model_id: str, train: pd.DataFrame, test: pd.DataFrame) -> dict:
    hp = load_s2_result(model_id, FAMILY)["best_hyperparams"]
    pred_grid = predict_tau_grid(model_id, train, test, hp)
    any_violation, per_pair = crossing_violations(pred_grid)
    violation_rate = float(any_violation.mean())

    actual = test["rho"].to_numpy()
    pb_before = float(pinball_loss(actual, pred_grid[0.20].to_numpy(), 0.20).mean())

    result = {
        "model_id": model_id, "violation_rate": violation_rate, "per_pair": per_pair,
        "exceeds_threshold": violation_rate > VIOLATION_THRESHOLD,
        "pinball_before_at_0.20": pb_before, "fix_applied": False,
        "pinball_after_at_0.20": None, "violation_rate_after": None,
    }
    if violation_rate > VIOLATION_THRESHOLD:
        fixed = isotonic_fix(pred_grid)
        any_after, _ = crossing_violations(fixed)
        result["fix_applied"] = True
        result["violation_rate_after"] = float(any_after.mean())
        result["pinball_after_at_0.20"] = float(pinball_loss(actual, fixed[0.20].to_numpy(), 0.20).mean())
    return result


def render_report(results: list[dict]) -> str:
    lines = [
        "# آزمون ضدتقاطع کوانتایل — بند 7.23.1",
        "",
        f"> روی fold۰ تنها ({len(TAU_GRID)} نقطه‌ی `TAU_GRID`، هایپرپارامتر S2 هر مدل بدون "
        "تنظیم مجدد روی τ دیگر — بند بالای ماژول). فقط اعضای مسیر Q1 خ۱ که هر τ را جدا "
        f"برازش می‌کنند: {', '.join(sorted(QUANTREG_MODEL_IDS))}.",
        "",
        "| مدل | نرخ نقض (هر جفت مجاور) | آستانه‌ی ۱٪ رد شد؟ | اصلاح اعمال شد؟ | نرخ پس از اصلاح | pinball@۰.۲۰ قبل | pinball@۰.۲۰ بعد |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        exceeds = "🔴 بله" if r["exceeds_threshold"] else "✅ خیر"
        fixed = "بله (isotonic)" if r["fix_applied"] else "—"
        rate_after = f"{r['violation_rate_after']:.2%}" if r["violation_rate_after"] is not None else "—"
        pb_after = f"{r['pinball_after_at_0.20']:.5f}" if r["pinball_after_at_0.20"] is not None else "—"
        lines.append(
            f"| `{r['model_id']}` | {r['violation_rate']:.2%} | {exceeds} | {fixed} | "
            f"{rate_after} | {r['pinball_before_at_0.20']:.5f} | {pb_after} |"
        )

    lines += ["", "## تفکیک هر جفت τ مجاور", ""]
    for r in results:
        lines.append(f"**`{r['model_id']}`:** " + " · ".join(
            f"{k}: {v:.2%}" for k, v in r["per_pair"].items()))

    n_exceed = sum(1 for r in results if r["exceeds_threshold"])
    lines += [
        "",
        f"**{n_exceed} از {len(results)} مدل از آستانه‌ی ۱٪ (بند 7.23.1) عبور کردند.**",
    ]
    return "\n".join(lines)


def main() -> None:
    from src.config import REPORTS_DIR, set_global_seed

    set_global_seed()
    train, test = _fold0()
    results = [evaluate_model(mid, train, test) for mid in sorted(QUANTREG_MODEL_IDS)]
    report = render_report(results)
    out = REPORTS_DIR / "phase7"
    out.mkdir(parents=True, exist_ok=True)
    (out / "quantile_crossing_F01.md").write_text(report + "\n")
    pd.DataFrame(results).to_json(out / "quantile_crossing_F01.json", orient="records",
                                  indent=2, force_ascii=False)
    print(report)
    print(f"\nذخیره شد در {out / 'quantile_crossing_F01.md'}")


if __name__ == "__main__":
    main()
