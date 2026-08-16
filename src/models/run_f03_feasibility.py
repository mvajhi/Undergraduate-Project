"""بند 7.12.5 — معیار پایان خانواده‌ی ۳ (سری‌زمانی L3)، سنجش امکان‌سنجی روی هر ۵ fold رسمی.

⚠️ این ارزیابی **مستقل از `model_comparison.md`** است (سطح/واحد متفاوت — بند بالای
`f03_timeseries.py`). مرجع اینجا **مدل صفر** (پیش‌بینی day_shock=۰، یعنی «هیچ شوکی
انتظار نمی‌رود») است، نه B3 — چون day_shock به‌تعریف حول صفر نوسان می‌کند.

اجرا: ``python -m src.models.run_f03_feasibility``
"""

import time
import warnings

import numpy as np
import pandas as pd

import src.models.families.f03_timeseries as f03
from src.baselines import pinball_loss
from src.cv import load_cv_folds
from src.features.l3_series import build_l3_series


def _official_l3_folds(series: pd.DataFrame) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    fold_meta, _ = load_cv_folds()
    return [(series.loc[m1], series.loc[m2]) for f in fold_meta for m1, m2 in [f.masks(series["date_gregorian"])]]


def run_feasibility(tau: float = 0.2) -> pd.DataFrame:
    series = build_l3_series()
    rows = []
    for meal, s in series.items():
        folds = _official_l3_folds(s)
        for model_id, fit_fn in f03.MODELS.items():
            preds, actuals, seconds, n_fail = [], [], 0.0, 0
            for tr, te in folds:
                t0 = time.time()
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        pred = fit_fn(tr, te, tau)
                except Exception:
                    n_fail += 1
                    continue
                seconds += time.time() - t0
                mask = te["day_shock"].notna().to_numpy()
                preds.append(np.asarray(pred)[mask])
                actuals.append(te["day_shock"].to_numpy()[mask])
            if not preds:
                rows.append({"meal": meal, "model_id": model_id, "status": "❌ شکست در همه‌ی fold‌ها",
                            "pinball": float("nan"), "pinball_zero": float("nan"), "n_eval": 0, "seconds": seconds})
                continue
            actual = np.concatenate(actuals)
            pred = np.concatenate(preds)
            pb = float(pinball_loss(actual, pred, tau).mean())
            pb_zero = float(pinball_loss(actual, np.zeros_like(actual), tau).mean())
            rows.append({
                "meal": meal, "model_id": model_id, "status": "✅" if n_fail == 0 else f"⚠️ {n_fail} fold شکست",
                "pinball": pb, "pinball_zero": pb_zero, "beats_zero": pb < pb_zero,
                "n_eval": len(actual), "seconds": seconds,
            })
    return pd.DataFrame(rows)


def render_report(df: pd.DataFrame, tau: float) -> str:
    lines = [
        "# امکان‌سنجی خ۳ (سری‌زمانی L3) — بند 7.12.5",
        "",
        f"> τ={tau}. هر ۵ fold رسمی، جدا برای ناهار/شام (F33). مرجع: **مدل صفر** "
        "(day_shock=۰) — نه B3، چون واحد متفاوت است (بالای `f03_timeseries.py`).",
        "",
        "| وعده | مدل | وضعیت | pinball | pinball صفر | برد بر صفر؟ | n ارزیابی | ثانیه |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        beats = "✅" if r.get("beats_zero", False) else "—"
        lines.append(f"| {r['meal']} | `{r['model_id']}` | {r['status']} | {r['pinball']:.5f} | "
                    f"{r['pinball_zero']:.5f} | {beats} | {r['n_eval']} | {r['seconds']:.2f} |")

    n_beat = int(df.get("beats_zero", pd.Series(dtype=bool)).sum())
    lines += ["", f"**{n_beat} از {len(df)} ترکیب (وعده×مدل) از مدل صفر بهتر بودند.**",
             "",
             "⚠️ **یادآوری دامنه:** این جدول سطح L3 (`day_shock`) را می‌سنجد، نه نرخ سلولی. "
             "آشتی به سطح L1 برای مقایسه‌ی مستقیم با B3 نیازمند رویکرد H2/H3 (بند 7.24) "
             "است که خارج از فهرست کوتاه فعلی می‌ماند."]
    return "\n".join(lines)


def main() -> None:
    from src.config import REPORTS_DIR, set_global_seed

    set_global_seed()
    df = run_feasibility()
    report = render_report(df, 0.2)
    out = REPORTS_DIR / "phase7"
    out.mkdir(parents=True, exist_ok=True)
    (out / "F03_feasibility.md").write_text(report + "\n")
    df.to_json(out / "F03_feasibility.json", orient="records", indent=2, force_ascii=False)
    print(report)
    print(f"\nذخیره شد در {out / 'F03_feasibility.md'}")


if __name__ == "__main__":
    main()
