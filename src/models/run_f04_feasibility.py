"""بند 7.13.4 — معیار پایان خانواده‌ی ۴ (DFM سطح L4)، سنجش امکان‌سنجی روی هر ۵ fold رسمی.

⚠️ مستقل از `model_comparison.md` (سطح/دانه‌بندی متفاوت — بالای `f04_multivariate.py`).
مرجع اینجا **کوانتایل تجربی خودِ هر سری** (بدون در نظر گرفتن عامل مشترک) است — سؤالی که
این جدول جواب می‌دهد: «آیا مدل‌سازی عامل مشترک (F60) روی کوانتایل تک‌سری‌ای اضافه‌ارزش دارد؟»

اجرا: ``python -m src.models.run_f04_feasibility``
"""

import time
import warnings

import numpy as np
import pandas as pd

from src.baselines import pinball_loss
from src.cv import DATE_COL, load_cv_folds
from src.features.l4_series import build_l4_panel
from src.models.families.f04_multivariate import fit_predict_dfm


def _official_panel_folds(panel: pd.DataFrame) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    fold_meta, _ = load_cv_folds()
    dates = panel.reset_index()[DATE_COL]
    dates.index = panel.index
    return [(panel.loc[m1], panel.loc[m2]) for f in fold_meta for m1, m2 in [f.masks(dates)]]


def _naive_series_quantile(train_panel: pd.DataFrame, test_panel: pd.DataFrame, tau: float) -> pd.DataFrame:
    """مرجع: کوانتایل تجربی τ خودِ هر سری روی دوره‌ی آموزش، بدون هیچ ساختار مشترک."""
    q = train_panel.quantile(tau)
    return pd.DataFrame(np.tile(q.to_numpy(), (len(test_panel), 1)),
                        index=test_panel.index, columns=test_panel.columns)


def run_feasibility(tau: float = 0.2, k_factors: int = 1) -> pd.DataFrame:
    panel = build_l4_panel()
    folds = _official_panel_folds(panel)

    rows = []
    for i, (train, test) in enumerate(folds):
        actual_long = test.stack()
        if actual_long.empty:
            continue

        naive = _naive_series_quantile(train, test, tau)
        naive_long = naive.stack().reindex(actual_long.index)
        pb_naive = float(pinball_loss(actual_long.to_numpy(), naive_long.to_numpy(), tau).mean())

        t0 = time.time()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dfm_pred = fit_predict_dfm(train, test, tau, k_factors=k_factors)
            seconds = time.time() - t0
            dfm_long = dfm_pred.stack().reindex(actual_long.index)
            pb_dfm = float(pinball_loss(actual_long.to_numpy(), dfm_long.to_numpy(), tau).mean())
            status = "✅"
        except Exception as exc:  # DFM می‌تواند عددی ناهمگرا شود — شکست خودش داده است
            seconds = time.time() - t0
            pb_dfm = float("nan")
            status = f"❌ {type(exc).__name__}"

        rows.append({
            "fold": i, "n_series": train.shape[1], "n_train_days": len(train), "n_test_days": len(test),
            "status": status, "pinball_dfm": pb_dfm, "pinball_naive": pb_naive,
            "beats_naive": (pb_dfm < pb_naive) if np.isfinite(pb_dfm) else False,
            "seconds": seconds, "n_eval": len(actual_long.dropna()),
        })
    return pd.DataFrame(rows)


def render_report(df: pd.DataFrame, tau: float, k_factors: int) -> str:
    lines = [
        "# امکان‌سنجی خ۴ (DFM سطح L4) — بند 7.13.4",
        "",
        f"> τ={tau}، k_factors={k_factors}. هر ۵ fold رسمی، ۴۱ سری $(m,r)$ هم‌زمان. مرجع: "
        "کوانتایل تجربی خودِ هر سری (بدون عامل مشترک).",
        "",
        "| fold | n_train روز | n_test روز | وضعیت | pinball DFM | pinball تک‌سری | برد؟ | ثانیه |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        beats = "✅" if r["beats_naive"] else "—"
        pb_dfm = f"{r['pinball_dfm']:.5f}" if np.isfinite(r["pinball_dfm"]) else "—"
        lines.append(f"| {r['fold']} | {r['n_train_days']} | {r['n_test_days']} | {r['status']} | "
                    f"{pb_dfm} | {r['pinball_naive']:.5f} | {beats} | {r['seconds']:.1f} |")

    n_ok = int((df["status"] == "✅").sum())
    n_beat = int(df["beats_naive"].sum())
    lines += ["", f"**DFM در {n_ok} از {len(df)} fold با موفقیت برازش شد؛ در {n_beat} از آن‌ها از مرجع تک‌سری بهتر بود.**",
             "",
             "⚠️ **یادآوری دامنه:** سطح L4 (میانگین روی FoodName)، نه سلولی — مستقیماً با "
             "`model_comparison.md` قابل‌جمع‌بندی نیست (بند 7.24: آشتی سطوح، کار جداگانه)."]
    return "\n".join(lines)


def main() -> None:
    from src.config import REPORTS_DIR, set_global_seed

    set_global_seed()
    df = run_feasibility()
    report = render_report(df, 0.2, 1)
    out = REPORTS_DIR / "phase7"
    out.mkdir(parents=True, exist_ok=True)
    (out / "F04_feasibility.md").write_text(report + "\n")
    df.to_json(out / "F04_feasibility.json", orient="records", indent=2, force_ascii=False)
    print(report)
    print(f"\nذخیره شد در {out / 'F04_feasibility.md'}")


if __name__ == "__main__":
    main()
