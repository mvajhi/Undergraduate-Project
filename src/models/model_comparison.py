"""بند 7.25 — جدول مقایسه‌ی نهایی و پروتکل داوری (دروازه‌ی M4).

فقط مدل‌هایی که بودجه‌ی کامل S2 گرفتند (فهرست کوتاه ردیف ۳۷ decision_log، ۲۱ مدل از ۵
خانواده) اینجا با آزمون کامل بند 7.25.1 (بوت‌استرپ بلوکی دوبعدی + n_eff) داوری می‌شوند.
مدل‌هایی که زودتر (S0/S1/پیمایش محور) کنار رفتند اینجا دوباره آزموده نمی‌شوند — دقیقاً
همان اصلی که ردیف ۳۷ decision_log برای جلوگیری از «بودجه‌ی کور به کاردینالیتی» وضع کرد؛
شاهد کنارگذاشتن آن‌ها در `S0_feasibility.md` / `S1_screening_*.md` / `axis_screening.md`
است، نه اینجا.

بوت‌استرپ بلوکی دوبعدی و n_eff از زیرساخت فاز ۶ (`src/cv.py::block_bootstrap_2d`,
``effective_sample_size``) استفاده می‌کنند — بازپیاده‌سازی نشدند.

## ⚠️ قهرمانان GPU — ادغام از JSON آماده، نه بازبرازش

چهار قهرمان GPU (بند 7.8، `reports/gpu/champion_*.json`) این‌جا **بازبرازش نمی‌شوند**
— محیط محلی این پروژه عمداً CPU-فقط است و torch/gpytorch/numpyro روی آن نصب نیست، پس
`fit_fn` این خانواده‌ها اصلاً import نمی‌شود. به‌جایش اعداد آماده‌ی همان JSON که خودِ
نوت‌بوک GPU (با بازبرازش واقعی روی هر ۵ fold رسمی، `gpu_runner.finalize_champion`)
تولید کرده، مستقیم خوانده می‌شوند.

**دو پایه‌ی متفاوت محاسبه، عمداً برچسب‌گذاری‌شده (ستون ``metric_basis``):**

- `pinball_rate`، `coverage`، `coverage_gap` — هر دو گروه **ردیفی** (روی تمام سطرهای
  OOF concatenate‌شده) محاسبه شده‌اند؛ **مستقیماً قابل‌قیاس‌اند**.
- `pinball_portions`/`shortage_rate`/`waste_reduction_pct`/`RMSE_rho`/`MAE_rho`/`R2_rho`
  برای CPU **ردیفی** است ولی برای GPU **میانگین fold** (`gpu_runner.aggregate_fold_metrics`
  — همان تفاوت وزن‌دهی‌ای که `src/models/significance.py` بین `S2_tuning_*.md` و
  `dm_test_*.md` هشدار می‌دهد: fold۲ فقط ۱۸۵ ردیف دارد ولی وزن کامل fold می‌گیرد).
  **این ستون‌ها بین دو گروه مستقیماً قابل‌قیاس نیستند** — مقایسه‌ی رتبه‌ای فقط با
  `pinball_rate` معتبر است.
- `beats_B3_significant` برای CPU از بوت‌استرپ بلوکی دوبعدی (`block_bootstrap_2d`) و
  برای GPU از آزمون Diebold-Mariano تک‌ردیفی (`src/cv.py::diebold_mariano`، همان
  آزمونی که `dm_test_*.md`های CPU هم استفاده کرده‌اند) می‌آید — ستون
  ``significance_method`` می‌گوید کدام. برای GPU فاصله‌ی اطمینان بوت‌استرپ محاسبه
  نشده (نیازمند OOF خام است که فقط داخل session GPU موجود بود، نه در JSON ذخیره‌شده).

اجرا: ``python -m src.models.model_comparison``
"""

import glob
import importlib
import json

import numpy as np
import pandas as pd

from src.baselines import b3_empirical_quantile, operational_metrics, pinball_loss
from src.config import REPORTS_DIR
from src.cv import DATE_COL, block_bootstrap_2d, effective_sample_size, load_cv_folds
from src.features.build import FEATURES_A_PATH
from src.models.axes import TUNING_TAU
from src.models.card_writer import _FAMILY_MODULES, load_s2_result

#: بند 7.9.1 (اسپرینت A، یافته‌ی ۱۰): ICC(روز)=۰.۲۲۵ ≈ ICC(سلف)=۰.۲۰۵ (F10) — روز
#: به‌عنوان محور غالب برای تصحیح n_eff انتخاب شد (همان مقدار `run_protocol.py`).
DAY_ICC = 0.225

FAMILIES = ("F01", "F02", "F09", "F10", "F11")

GPU_REPORTS_DIR = REPORTS_DIR / "gpu"


def _load_roster() -> list[tuple[str, str]]:
    roster = []
    for family in FAMILIES:
        path = REPORTS_DIR / "phase7" / f"S2_tuning_{family}.json"
        payload = json.loads(path.read_text())
        roster.extend((family, mid) for mid in payload if not mid.startswith("_"))
    return roster


def _official_folds() -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    df = pd.read_parquet(FEATURES_A_PATH).sort_values(DATE_COL).reset_index(drop=True)
    fold_meta, _ = load_cv_folds()
    return [(df.loc[m1], df.loc[m2]) for f in fold_meta for m1, m2 in [f.masks(df[DATE_COL])]]


def _oof_frame(fit_fn, folds: list, tau: float, hyperparams: dict) -> pd.DataFrame:
    """پیش‌بینی out-of-fold + همه‌ی ستون‌های لازم برای operational_metrics و بوت‌استرپ."""
    parts = []
    for tr, te in folds:
        pred = np.clip(np.asarray(fit_fn(tr, te, tau, **hyperparams), dtype=float), 0.0, 1.0)
        part = te[[DATE_COL, "RestaurantName", "Res", "Recv", "rho", "Meal", "is_tehran"]].copy()
        part["pred_q"] = pred
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def build_comparison_table(tau: float = TUNING_TAU, n_boot: int = 1000, seed: int = 42
                           ) -> tuple[pd.DataFrame, dict, float]:
    folds = _official_folds()
    b3_oof = _oof_frame(lambda tr, te, t, **_: b3_empirical_quantile(tr, te, t), folds, tau, {})
    b3_metrics = operational_metrics(b3_oof, b3_oof["pred_q"].to_numpy(), tau)
    b3_pinball = float(pinball_loss(b3_oof["rho"].to_numpy(), b3_oof["pred_q"].to_numpy(), tau).mean())

    rows = []
    for family, model_id in _load_roster():
        result = load_s2_result(model_id, family)
        hp = dict(result["best_hyperparams"])
        mod = importlib.import_module(_FAMILY_MODULES[family])
        fit_fn = mod.MODELS[model_id]
        oof = _oof_frame(fit_fn, folds, tau, hp)
        metrics = operational_metrics(oof, oof["pred_q"].to_numpy(), tau)
        pinball_rate = float(pinball_loss(oof["rho"].to_numpy(), oof["pred_q"].to_numpy(), tau).mean())

        merged = oof.copy()
        merged["b3_pred_q"] = b3_oof["pred_q"].to_numpy()  # همان ترتیب ردیف (از همان folds)

        def _delta_stat(sample: pd.DataFrame, _tau: float = tau) -> float:
            m = pinball_loss(sample["rho"].to_numpy(), sample["pred_q"].to_numpy(), _tau).mean()
            b = pinball_loss(sample["rho"].to_numpy(), sample["b3_pred_q"].to_numpy(), _tau).mean()
            return m - b

        delta_point, delta_lo, delta_hi = block_bootstrap_2d(
            merged, _delta_stat, day_col=DATE_COL, unit_col="RestaurantName", n_boot=n_boot, seed=seed)

        cluster_sizes = merged.groupby(DATE_COL, observed=True).size().to_numpy()
        n_eff = effective_sample_size(len(merged), cluster_sizes, icc=DAY_ICC)

        rows.append({
            "family": family, "model_id": model_id, "level": "L1", "target": "rho",
            "status": "تکمیل‌شده",
            "pinball_rate": pinball_rate,
            "pinball_portions": metrics["pinball_portions"],
            "shortage_rate": metrics["shortage_rate"],
            "waste_reduction_pct": metrics["waste_reduction_pct"],
            "coverage": metrics["coverage"], "coverage_gap": metrics["coverage_gap"],
            "RMSE_rho": metrics["RMSE_rho"], "MAE_rho": metrics["MAE_rho"], "R2_rho": metrics["R2_rho"],
            "delta_vs_B3": delta_point, "delta_vs_B3_ci_lo": delta_lo, "delta_vs_B3_ci_hi": delta_hi,
            "beats_B3_significant": bool(delta_hi < 0),
            "n_raw": len(merged), "n_eff": n_eff,
            "fit_seconds": result["seconds"], "n_trials": result["n_trials"],
            "converged": result["converged"],
        })

    df = pd.DataFrame(rows).sort_values("pinball_rate").reset_index(drop=True)
    df["compute"] = "local"
    df["metric_basis"] = "row_level_oof"
    df["significance_method"] = "block_bootstrap_2d"
    df["dm_p_value"] = np.nan
    return df, b3_metrics, b3_pinball


def _gpu_champion_files() -> list[str]:
    return sorted(glob.glob(str(GPU_REPORTS_DIR / "champion_*.json")))


def load_gpu_champion_rows(n_raw: int, n_eff: float) -> pd.DataFrame:
    """قهرمانان GPU را از ``reports/gpu/champion_*.json`` می‌خواند (بدون بازبرازش —
    بند «قهرمانان GPU» بالای ماژول). ``n_raw``/``n_eff`` از همان محاسبه‌ی CPU پاس داده
    می‌شود چون سطح ارزیابی (L1، ۳۸۸۰ ردیف OOF) برای هر پنج fold رسمی مشترک است، حتی
    برای مدل‌هایی که سطح **آموزش**شان L4/L5 است (بند 7.1.2 قاعده‌ی مقایسه‌پذیری)."""
    rows = []
    for path in _gpu_champion_files():
        champ = json.loads(open(path, encoding="utf-8").read())
        family, model_id = champ["family"], champ["model_id"]
        s2_path = GPU_REPORTS_DIR / f"S2_tuning_{family}_{model_id}.json"
        s2 = json.loads(s2_path.read_text()) if s2_path.exists() else {}
        m = champ["metrics"]  # ⚠️ میانگین fold، نه ردیفی — بند «قهرمانان GPU»
        dm = champ["dm_vs_b3"]
        rows.append({
            "family": family, "model_id": model_id, "level": champ["level"], "target": "rho",
            "status": "تکمیل‌شده (GPU)",
            "pinball_rate": champ["pinball_row_mean"],          # ردیفی — قابل‌قیاس مستقیم
            "pinball_portions": m["pinball_portions"],           # ⚠️ میانگین fold
            "shortage_rate": m["shortage_rate"],                 # ⚠️ میانگین fold
            "waste_reduction_pct": m["waste_reduction_pct"],     # ⚠️ میانگین fold
            "coverage": champ["coverage"], "coverage_gap": champ["coverage_gap"],  # ردیفی
            "RMSE_rho": m["RMSE_rho"], "MAE_rho": m["MAE_rho"], "R2_rho": m["R2_rho"],  # ⚠️ میانگین fold
            "delta_vs_B3": dm["delta_pinball"], "delta_vs_B3_ci_lo": np.nan, "delta_vs_B3_ci_hi": np.nan,
            "beats_B3_significant": bool(dm["significant_at_0.05"]),
            "n_raw": n_raw, "n_eff": n_eff,
            "fit_seconds": s2.get("seconds", champ.get("fit_seconds", float("nan"))),
            "n_trials": s2.get("n_trials_done"), "converged": s2.get("converged"),
            "compute": "colab/kaggle", "metric_basis": "fold_mean_refit",
            "significance_method": "diebold_mariano", "dm_p_value": dm["p_value"],
        })
    return pd.DataFrame(rows)


def build_full_comparison_table(tau: float = TUNING_TAU, n_boot: int = 1000, seed: int = 42
                                ) -> tuple[pd.DataFrame, dict, float]:
    """جدول CPU (بازبرازش کامل) + قهرمانان GPU (از JSON آماده) — یک رتبه‌بندی واحد،
    با ستون‌های ``compute``/``metric_basis``/``significance_method`` برای ردیابی اینکه
    کدام عدد از کجا و با کدام روش آمده (بند «قهرمانان GPU» بالای ماژول)."""
    cpu_df, b3_metrics, b3_pinball = build_comparison_table(tau, n_boot, seed)
    gpu_df = load_gpu_champion_rows(int(cpu_df["n_raw"].iloc[0]), float(cpu_df["n_eff"].iloc[0]))
    full = pd.concat([cpu_df, gpu_df], ignore_index=True).sort_values("pinball_rate").reset_index(drop=True)
    return full, b3_metrics, b3_pinball


def render_report(df: pd.DataFrame, b3_metrics: dict, b3_pinball: float, tau: float) -> str:
    """⚠️ برای سازگاری با فراخوان‌های قدیمی‌تر (و تست واحد ``test_model_comparison_
    render_report_counts_significant_wins`` که ``compute``/``dm_p_value`` را نمی‌دهد)،
    ستون‌های مخصوص GPU اگر موجود نباشند با مقدار پیش‌فرض CPU/local پر می‌شوند — یعنی
    یک ``DataFrame`` فقط-CPU دقیقاً همان گزارش قدیمی را می‌دهد."""
    df = df.copy()
    if "compute" not in df.columns:
        df["compute"] = "local"
    if "dm_p_value" not in df.columns:
        df["dm_p_value"] = np.nan

    n_eff_ratio = df["n_raw"].iloc[0] / df["n_eff"].iloc[0]
    n_gpu = int((df["compute"] != "local").sum())
    lines = [
        "# جدول مقایسه‌ی نهایی — بند 7.25، دروازه‌ی M4",
        "",
        f"> τ={tau}. مرجع B3: pinball(نرخ)={b3_pinball:.5f}، هدررفت‌کاهی={b3_metrics['waste_reduction_pct']:.1%}، "
        f"نرخ کمبود={b3_metrics['shortage_rate']:.1%}، پوشش={b3_metrics['coverage']:.1%}. "
        f"بوت‌استرپ بلوکی دوبعدی (روز×سلف، ۱۰۰۰ تکرار، `src/cv.py::block_bootstrap_2d`) روی Δ=pinball(مدل)−pinball(B3)؛ "
        f"n_eff با ICC(روز)={DAY_ICC} (F10) — ضریب تورم واریانس **{n_eff_ratio:.1f}×**.",
        "",
        "⚠️ فقط مدل‌هایی که بودجه‌ی کامل S2 گرفتند اینجا هستند: ۲۱ مدل CPU (فهرست کوتاه "
        f"ردیف ۳۷ decision_log) + **{n_gpu} قهرمان GPU** (بند 7.8، `⚡` در ستون منبع). "
        "مدل‌های کنارگذاشته‌شده در S0/S1/پیمایش محور با شاهد جداگانه در همان مرحله ثبت شده‌اند، نه اینجا.",
        "",
        "⚠️⚠️ **ستون‌های `هدررفت‌کاهی`/`نرخ کمبود`/`R²` برای ردیف‌های `⚡` روی پایه‌ی متفاوتی "
        "محاسبه شده‌اند** (میانگین fold، نه میانگین ردیفی — `metric_basis` در CSV) و "
        "**مستقیماً با ردیف‌های `🖥️` قابل‌قیاس نیستند**. تنها ستون‌هایی که بین هر دو گروه "
        "مستقیماً قابل‌قیاس‌اند: `pinball نرخ`، `پوشش`، و `Δ نسبت به B3`. جزئیات کامل در "
        "docstring بالای `src/models/model_comparison.py`.",
        "",
        "| منبع | مدل | خانواده | pinball نرخ | pinball پرس | Δ نسبت به B3 [CI ۹۵٪ / روش] | برد معنادار؟ | هدررفت‌کاهی | نرخ کمبود | پوشش | R² | ساعت-هسته | pinball/ساعت |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        hours = r["fit_seconds"] / 3600 if pd.notna(r["fit_seconds"]) else float("nan")
        pb_per_hour = r["pinball_rate"] / hours if hours and hours > 1e-6 else float("nan")
        sig = "✅" if r["beats_B3_significant"] else "—"
        is_gpu = r["compute"] != "local"
        src = "⚡" if is_gpu else "🖥️"
        ci = (f"DM p={r['dm_p_value']:.4f}" if is_gpu
              else f"[{r['delta_vs_B3_ci_lo']:+.5f}, {r['delta_vs_B3_ci_hi']:+.5f}]")
        lines.append(
            f"| {src} | `{r['model_id']}` | {r['family']} | {r['pinball_rate']:.5f} | {r['pinball_portions']:.3f} | "
            f"{r['delta_vs_B3']:+.5f} [{ci}] | {sig} | "
            f"{r['waste_reduction_pct']:.1%} | {r['shortage_rate']:.1%} | {r['coverage']:.1%} | "
            f"{r['R2_rho']:.3f} | {hours:.3f} | {pb_per_hour:.5f} |"
        )

    n_sig = int(df["beats_B3_significant"].sum())
    winners = df[df["beats_B3_significant"]]["model_id"].tolist()
    top = df.iloc[0]
    lines += [
        "",
        f"**قهرمان مطلق (رتبه‌بندی با `pinball نرخ`، تنها ستونی که بدون قید و شرط برای هر "
        f"مدل روی این جدول معتبر است): `{top['model_id']}` ({'⚡GPU' if top['compute'] != 'local' else '🖥️CPU'}، "
        f"{top['pinball_rate']:.5f}).**",
        "",
        f"**{n_sig} از {len(df)} مدل به‌طور معنادار از B3 بهتر بودند "
        f"(CPU: CI بوت‌استرپ کاملاً منفی؛ GPU: Diebold-Mariano p<۰.۰۵ **و** جهت برد): "
        f"{', '.join(winners) if winners else '—'}.**",
        "",
        f"⚠️ **n_eff هشدار (بند A7/۶.۶):** اندازه‌ی خام هر مدل {int(df['n_raw'].iloc[0]):,} ردیف است ولی "
        f"اندازه‌ی مؤثر (با احتساب خوشه‌بندی روزانه) فقط **{df['n_eff'].iloc[0]:,.0f}** — هر مقایسه‌ای که این را "
        "نادیده بگیرد، اطمینانش کاذب است. n_eff برای ردیف‌های GPU هم همین عدد است، چون سطح "
        "ارزیابی (نه لزوماً سطح آموزش) برای همه یکی است — بند 7.1.2.",
    ]
    return "\n".join(lines)


def main() -> None:
    from src.config import set_global_seed

    set_global_seed()
    df, b3_metrics, b3_pinball = build_full_comparison_table()
    report = render_report(df, b3_metrics, b3_pinball, TUNING_TAU)
    out = REPORTS_DIR / "phase7"
    out.mkdir(parents=True, exist_ok=True)
    (out / "model_comparison.md").write_text(report + "\n")
    df.to_csv(out / "model_comparison.csv", index=False)
    print(report)
    print(f"\nذخیره شد در {out / 'model_comparison.md'} و {out / 'model_comparison.csv'}")


if __name__ == "__main__":
    main()
