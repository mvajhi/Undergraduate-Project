"""بند ۸.۷ WBS فاز ۸ — مقایسه‌ی آماری قهرمان‌ها با هر دو خط پایه (B2 و B3) روی Test.

بند ۸.۱ (`8.1_holdout_significance.csv`) از قبل DM-test + بوت‌استرپ بلوکی دوبعدی هر
۴ نامزد را **در برابر B3** محاسبه کرده — اینجا فقط نسخه‌ی **در برابر B2** اضافه می‌شود
(بند ۸.۷ صریحاً هر دو را می‌خواهد) و گزاره‌ی نهایی به شکل دقیق درخواستی WBS ساخته
می‌شود: «مدل X با اطمینان ۹۵٪، pinball loss را نسبت به B{2,3} به میزان Δ [CI] کاهش
می‌دهد.» چیزی دوباره fit نمی‌شود — از `data/interim/phase8_holdout_predictions.parquet`
(بند ۸.۱) خوانده می‌شود.

اجرا: ``python -m src.models.run_phase8_significance_vs_baselines``
"""

import pandas as pd

from src.baselines import pinball_loss
from src.config import REPORTS_DIR
from src.cv import DATE_COL, block_bootstrap_2d, diebold_mariano
from src.models.axes import TUNING_TAU

OUT_DIR = REPORTS_DIR / "phase8"
PRED_PATH = REPORTS_DIR.parent / "data" / "interim" / "phase8_holdout_predictions.parquet"
TAU = TUNING_TAU
CANDIDATES = ["lightgbm_quantile", "catboost_quantile", "bhm_varying_dispersion"]


def vs_baseline(df: pd.DataFrame, model_col: str, base_col: str) -> dict:
    rho = df["rho"].to_numpy()
    pred = df[model_col].to_numpy()
    base = df[base_col].to_numpy()
    dm_stat, dm_p = diebold_mariano(pinball_loss(rho, pred, TAU), pinball_loss(rho, base, TAU))

    merged = df.copy()
    merged["rho"] = rho
    merged["pred_q"] = pred
    merged["base_pred_q"] = base

    def _delta(sample: pd.DataFrame) -> float:
        m = pinball_loss(sample["rho"].to_numpy(), sample["pred_q"].to_numpy(), TAU).mean()
        b = pinball_loss(sample["rho"].to_numpy(), sample["base_pred_q"].to_numpy(), TAU).mean()
        return m - b

    delta, lo, hi = block_bootstrap_2d(merged, _delta, day_col=DATE_COL,
                                       unit_col="RestaurantName", n_boot=1000, seed=42)
    return {"dm_stat": dm_stat, "dm_p": dm_p, "delta": delta, "ci_lo": lo, "ci_hi": hi}


def statement(model: str, baseline: str, r: dict) -> str:
    if r["ci_hi"] < 0:
        return (f"**`{model}` با اطمینان ۹۵٪، pinball loss را نسبت به {baseline} به میزان "
                f"{abs(r['delta']):.5f} [CI: {abs(r['ci_hi']):.5f}, {abs(r['ci_lo']):.5f}] کاهش می‌دهد.**")
    if r["ci_lo"] > 0:
        return (f"**`{model}` با اطمینان ۹۵٪، pinball loss را نسبت به {baseline} به میزان "
                f"{r['delta']:.5f} [CI: {r['ci_lo']:.5f}, {r['ci_hi']:.5f}] افزایش می‌دهد (بدتر).**")
    return (f"`{model}` تفاوت معناداری با {baseline} ندارد (Δ={r['delta']:+.5f}، "
            f"CI=[{r['ci_lo']:+.5f}, {r['ci_hi']:+.5f}] شامل صفر).")


def main() -> None:
    df = pd.read_parquet(PRED_PATH)
    b3_sig = pd.read_csv(OUT_DIR / "8.1_holdout_significance.csv", index_col=0)

    rows_b2 = {}
    for model in CANDIDATES:
        rows_b2[model] = vs_baseline(df, f"pred__{model}", "pred__B2_group_shrunk")
    b2_df = pd.DataFrame(rows_b2).T

    lines = [
        "# بند ۸.۷ — مقایسه‌ی آماری با B2 و B3 روی Test قفل‌شده",
        "",
        f"> τ={TAU}. DM-test + بوت‌استرپ بلوکی دوبعدی (روز×سلف، ۱۰۰۰ تکرار، `src/cv.py::block_bootstrap_2d`). "
        "مقایسه‌ی در برابر B3 از بند ۸.۱ نقل شده (بازمحاسبه نشد).",
        "",
        "## در برابر B3 (نقل از بند ۸.۱)",
        "",
    ]
    for model in CANDIDATES:
        r = b3_sig.loc[model].to_dict()
        r = {"delta": r["delta_vs_B3"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
             "dm_stat": r["dm_stat"], "dm_p": r["dm_p"]}
        lines.append("- " + statement(model, "B3", r))
    lines += ["", "## در برابر B2 (محاسبه‌ی تازه)", ""]
    for model in CANDIDATES:
        lines.append("- " + statement(model, "B2", rows_b2[model]))

    report = "\n".join(lines)
    (OUT_DIR / "8.7_significance_vs_baselines.md").write_text(report + "\n")
    b2_df.to_csv(OUT_DIR / "8.7_significance_vs_B2.csv")
    print(report)
    print(f"\nذخیره شد در {OUT_DIR}/8.7_significance_vs_baselines.md")


if __name__ == "__main__":
    main()
