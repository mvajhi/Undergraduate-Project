"""بند ۸.۱ WBS فاز ۸ — ارزیابی نهایی روی پنجره‌ی Test قفل‌شده (فقط یک‌بار).

⚠️ **نقطه‌ی بی‌بازگشت.** این اسکریپت `src/cv.py::holdout_split` را برای اولین‌بار در کل
پروژه فراخوانی می‌کند. پس از اجرای آن، مدل/فیچر/هایپرپارامتر دیگر قابل تغییر نیست
(ماتریس وابستگی WBS). فهرست نامزد نهایی طبق ردیف ۴۸ `doc/decision_log.md`:

- `lightgbm_quantile` (قهرمان مطلق فاز ۷)
- `bhm_varying_dispersion` (نزدیک‌ترین رقیب — ⚠️ چک‌لیست همگرایی 7.17.3 پاس نشده بود)
- `catboost_quantile` (سومین برنده‌ی معنادار)
- خط پایه‌ها: B2، B3 (بند 8.7 WBS)

## چرا `bhm_varying_dispersion` اینجا **بازبرازش واقعی محلی** می‌شود (نه ادغام JSON)

بر خلاف `model_comparison.py` (که قهرمانان GPU را از JSON آماده می‌خواند چون محیط محلی
عمداً torch/gpytorch/numpyro ندارد)، اینجا `numpyro` **موقتاً روی CPU نصب شد**
(`requirements.lock` به‌روزرسانی شد) چون:

1. هیچ posterior ذخیره‌شده‌ای دقیقاً روی مرزِ آموزش/آزمونِ Test قفل‌شده برازش نشده —
   نزدیک‌ترین (`fold4`، train تا ۲۰۲۴-۰۴-۳۰) به داخل خودِ پنجره‌ی Test (۲۰۲۴-۰۴-۲۳ تا
   ۲۰۲۴-۰۵-۱۹) نشت می‌کند؛ استفاده از آن دقیقاً همان چیزی است که «نقطه‌ی بی‌بازگشت»
   قرار است از آن جلوگیری کند.
2. این یک برازش **تکی** با هایپرپارامتر قهرمانِ از‌پیش‌تعیین‌شده است (نه جست‌وجوی
   چندصدتایی S1/S2) — روی CPU با `num_warmup=500, num_samples=600, num_chains=2`
   حدود ۱۰-۱۵ دقیقه به‌ازای هر seed طول می‌کشد (اسموک‌تست محلی)، هزینه‌ای یک‌باره
   و قابل قبول برای دقیق‌ترین عدد ممکن روی مهم‌ترین ارزیابی پروژه.

هایپرپارامتر از `reports/gpu/champion_F08_bhm_varying_dispersion.json` (بازبرازش S3
واقعی GPU روی ۵ fold رسمی) خوانده شده، **دقیقاً همان دو seed** (۴۲، ۱۲۳۴) که champion
GPU استفاده کرد — میانگین پیش‌بینی دو seed گزارش می‌شود (سازگار با نحوه‌ی گزارش
S3 در `F08_bayesian_L1.md`).

اجرا: ``python -m src.models.run_phase8_holdout_eval``
"""

import importlib
import json
import time

import numpy as np
import pandas as pd

from src.baselines import b2_group_shrunk, b3_empirical_quantile, operational_metrics, pinball_loss, quantile_adjust
from src.config import REPORTS_DIR, set_global_seed
from src.cv import DATE_COL, block_bootstrap_2d, diebold_mariano, effective_sample_size, holdout_split
from src.features.build import FEATURES_A_PATH
from src.models.axes import TUNING_TAU
from src.models.card_writer import load_s2_result
from src.models.families.f08_bayesian import _fit_generic
from src.models.families.f08_bayesian import _predict as _bayes_predict

TAU = TUNING_TAU
DAY_ICC = 0.225
BAYES_SEEDS = (42, 1234)
GPU_CHAMPION_PATH = REPORTS_DIR / "gpu" / "champion_F08_bhm_varying_dispersion.json"
OUT_DIR = REPORTS_DIR / "phase8"
PRED_PATH = FEATURES_A_PATH.parent.parent / "interim" / "phase8_holdout_predictions.parquet"


def _bayes_hp() -> dict:
    champ = json.loads(GPU_CHAMPION_PATH.read_text())
    hp = dict(champ["hyperparams"])
    hp.pop("num_chains", None)
    return {**hp, "num_chains": 2}


def load_holdout() -> tuple[pd.DataFrame, pd.DataFrame, object]:
    df = pd.read_parquet(FEATURES_A_PATH).sort_values(DATE_COL).reset_index(drop=True)
    tr_mask, te_mask, f = holdout_split(df)
    return df.loc[tr_mask].reset_index(drop=True), df.loc[te_mask].reset_index(drop=True), f


def _predict_tree(model_id: str, train: pd.DataFrame, test: pd.DataFrame, tau: float
                  ) -> tuple[np.ndarray, float, dict]:
    result = load_s2_result(model_id, "F02")
    mod = importlib.import_module("src.models.families.f02_tree")
    fn = mod.MODELS[model_id]
    t0 = time.time()
    pred = np.clip(np.asarray(fn(train, test, tau, **result["best_hyperparams"]), dtype=float), 0.0, 1.0)
    return pred, time.time() - t0, result["best_hyperparams"]


def _predict_bhm(train: pd.DataFrame, test: pd.DataFrame, tau: float) -> tuple[np.ndarray, float, dict]:
    hp = _bayes_hp()
    preds, diags = [], []
    t0 = time.time()
    for seed in BAYES_SEEDS:
        model = _fit_generic(train, tau, use_day=True, varying_dispersion=True, seed=seed, **hp)
        diags.append(model.diagnostics)
        preds.append(_bayes_predict(model, test, tau))
    seconds = time.time() - t0
    pred = np.clip(np.mean(preds, axis=0), 0.0, 1.0)
    return pred, seconds, {"hyperparams": hp, "seeds": BAYES_SEEDS, "diagnostics": diags}


def _predict_b2(train: pd.DataFrame, test: pd.DataFrame, tau: float) -> tuple[np.ndarray, float, dict]:
    t0 = time.time()
    rho_hat = b2_group_shrunk(train, test, tau)
    pred = quantile_adjust(rho_hat, train, tau, b2_group_shrunk)
    return np.asarray(pred, dtype=float), time.time() - t0, {}


def _predict_b3(train: pd.DataFrame, test: pd.DataFrame, tau: float) -> tuple[np.ndarray, float, dict]:
    t0 = time.time()
    pred = np.asarray(b3_empirical_quantile(train, test, tau), dtype=float)
    return pred, time.time() - t0, {}


CANDIDATES = {
    "lightgbm_quantile": lambda tr, te, tau: _predict_tree("lightgbm_quantile", tr, te, tau),
    "catboost_quantile": lambda tr, te, tau: _predict_tree("catboost_quantile", tr, te, tau),
    "bhm_varying_dispersion": _predict_bhm,
    "B2_group_shrunk": _predict_b2,
    "B3_empirical_quantile": _predict_b3,
}


def render_report(metrics_df: pd.DataFrame, sig_df: pd.DataFrame, fold, n_eff: float, n_raw: int) -> str:
    lines = [
        "# بند ۸.۱ — ارزیابی نهایی روی Test قفل‌شده (دروازه‌ی M5)",
        "",
        f"> **پنجره‌ی Test** (فقط یک‌بار باز شد): {fold.test_start.date()} تا {fold.test_end.date()}؛ "
        f"آموزش: {fold.train_start.date()} تا {fold.train_end.date()}. "
        f"τ={TAU}. نامزدها طبق ردیف ۴۸ decision_log.",
        "",
        f"⚠️ **n_eff هشدار:** اندازه‌ی خام {n_raw:,} ردیف، اندازه‌ی مؤثر (ICC(روز)={DAY_ICC}) فقط **{n_eff:,.0f}**.",
        "",
        "| مدل | pinball نرخ | pinball پرس | RMSE نرخ | MAE نرخ | R² | پوشش | شکاف پوشش | نرخ کمبود | هدررفت‌کاهی | ثانیه |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for model_id, r in metrics_df.iterrows():
        lines.append(
            f"| `{model_id}` | {r['pinball']:.5f} | {r['pinball_portions']:.3f} | {r['RMSE_rho']:.5f} | "
            f"{r['MAE_rho']:.5f} | {r['R2_rho']:.3f} | {r['coverage']:.1%} | {r['coverage_gap']:+.1%} | "
            f"{r['shortage_rate']:.1%} | {r['waste_reduction_pct']:.1%} | {r['fit_seconds']:.1f} |"
        )
    lines += [
        "",
        "## مقایسه‌ی آماری در برابر B3 (DM-test + بوت‌استرپ بلوکی دوبعدی)",
        "",
        "| مدل | Δ نسبت به B3 | CI ۹۵٪ | DM آماره | DM p | برد معنادار؟ |",
        "|---|---|---|---|---|---|",
    ]
    for model_id, r in sig_df.iterrows():
        sig = "✅" if r["ci_hi"] < 0 else "—"
        lines.append(
            f"| `{model_id}` | {r['delta_vs_B3']:+.5f} | [{r['ci_lo']:+.5f}, {r['ci_hi']:+.5f}] | "
            f"{r['dm_stat']:.3f} | {r['dm_p']:.4f} | {sig} |"
        )
    champion = metrics_df["pinball"].idxmin()
    lines += [
        "",
        f"**قهرمان روی Test قفل‌شده (بر اساس pinball نرخ): `{champion}`.**",
    ]
    return "\n".join(lines)


def main() -> None:
    set_global_seed()
    train, test, fold = load_holdout()
    print(f"Holdout: {fold}")
    print(f"train={len(train)} test={len(test)}")

    preds_frame = test[[DATE_COL, "RestaurantName", "Meal", "Res", "Recv", "rho", "is_tehran"]].copy()
    metric_rows, meta = {}, {}
    for model_id, fn in CANDIDATES.items():
        print(f"--- {model_id} ---")
        pred, seconds, info = fn(train, test, TAU)
        preds_frame[f"pred__{model_id}"] = pred
        m = operational_metrics(test, pred, TAU)
        m["fit_seconds"] = seconds
        metric_rows[model_id] = m
        meta[model_id] = info
        print(f"{model_id}: pinball={m['pinball']:.5f} coverage={m['coverage']:.3f} seconds={seconds:.1f}")

    metrics_df = pd.DataFrame(metric_rows).T

    b3_pred = preds_frame["pred__B3_empirical_quantile"].to_numpy()
    rho = test["rho"].to_numpy()
    sig_rows = {}
    for model_id in CANDIDATES:
        if model_id == "B3_empirical_quantile":
            continue
        pred = preds_frame[f"pred__{model_id}"].to_numpy()
        dm_stat, dm_p = diebold_mariano(
            pinball_loss(rho, pred, TAU), pinball_loss(rho, b3_pred, TAU))

        merged = preds_frame.copy()
        merged["rho"] = rho
        merged["pred_q"] = pred
        merged["b3_pred_q"] = b3_pred

        def _delta(sample: pd.DataFrame) -> float:
            m = pinball_loss(sample["rho"].to_numpy(), sample["pred_q"].to_numpy(), TAU).mean()
            b = pinball_loss(sample["rho"].to_numpy(), sample["b3_pred_q"].to_numpy(), TAU).mean()
            return m - b

        delta, lo, hi = block_bootstrap_2d(merged, _delta, day_col=DATE_COL,
                                           unit_col="RestaurantName", n_boot=1000, seed=42)
        sig_rows[model_id] = {"dm_stat": dm_stat, "dm_p": dm_p, "delta_vs_B3": delta, "ci_lo": lo, "ci_hi": hi}

    sig_df = pd.DataFrame(sig_rows).T

    cluster_sizes = preds_frame.groupby(DATE_COL, observed=True).size().to_numpy()
    n_eff = effective_sample_size(len(preds_frame), cluster_sizes, icc=DAY_ICC)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(OUT_DIR / "8.1_holdout_metrics.csv")
    sig_df.to_csv(OUT_DIR / "8.1_holdout_significance.csv")
    (OUT_DIR / "8.1_bhm_hyperparams.json").write_text(
        json.dumps(meta["bhm_varying_dispersion"], ensure_ascii=False, indent=2, default=str) + "\n")
    PRED_PATH.parent.mkdir(parents=True, exist_ok=True)
    preds_frame.to_parquet(PRED_PATH)

    report = render_report(metrics_df, sig_df, fold, n_eff, len(preds_frame))
    (OUT_DIR / "8.1_final_holdout_evaluation.md").write_text(report + "\n")
    print(report)
    print(f"\nذخیره شد در {OUT_DIR}/8.1_*.{{md,csv}} و {PRED_PATH}")


if __name__ == "__main__":
    main()
