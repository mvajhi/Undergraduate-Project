"""بند 7.4 گام ۱۱ (تشخیص برازش) — یکی از چهار گام غیرقابل‌حذف کارت مدل.

سنجه‌ی اصلی: **شکاف pinball بین train و test** روی هر ۵ fold رسمی، با بهترین
هایپرپارامتر S2 — بازبرازش واقعی، نه شبیه‌سازی (دقیقاً همان الگوی
``src/models/calibration.py``). ``fit_predict_*`` امضای یکسانی دارد (بند 7.1.1)،
پس ``fn(tr, tr, tau, **hp)`` پیش‌بینی درون‌نمونه و ``fn(tr, te, tau, **hp)``
پیش‌بینی برون‌نمونه می‌دهد — بدون نیاز به دسترسی به داخل هیچ مدلی.

⚠️ **محدودیت صادقانه.** بند 7.4 «مقایسه با سقف واقع‌بینانه‌ی $R^2\\approx0.4$–$0.5$»
(بند ۵.۱۳) را هم می‌خواهد؛ آن سقف برای پیش‌بینی **میانگین** تعریف شده، ولی
``fit_predict_*`` فقط کوانتایل خروجی می‌دهد (بند 7.1.1)، نه پیش‌بینی نقطه‌ای زیرین.
این ماژول به‌جایش نسبت pinball تست/train را گزارش می‌کند — سنجه‌ی هم‌ارز بیش‌برازش
در واحد خودِ معیار پروژه، بدون فرض اضافی روی مدل‌های زیرین.
"""

from typing import Callable

import numpy as np
import pandas as pd

from src.baselines import operational_metrics

#: نسبت pinball تست/train بالاتر از این آستانه، پرچم بیش‌برازش می‌گیرد.
_OVERFIT_RATIO_THRESHOLD = 1.20


def train_test_gap(fit_fn: Callable, folds: list[tuple[pd.DataFrame, pd.DataFrame]],
                   tau: float, hyperparams: dict) -> list[dict]:
    rows = []
    for i, (tr, te) in enumerate(folds):
        pred_tr = np.asarray(fit_fn(tr, tr, tau, **hyperparams), dtype=float)
        pred_te = np.asarray(fit_fn(tr, te, tau, **hyperparams), dtype=float)
        pb_tr = operational_metrics(tr, pred_tr, tau)["pinball"]
        pb_te = operational_metrics(te, pred_te, tau)["pinball"]
        rows.append({"fold": i, "n_train": len(tr), "n_test": len(te),
                    "pinball_train": pb_tr, "pinball_test": pb_te,
                    "ratio": pb_te / pb_tr if pb_tr > 0 else float("inf")})
    return rows


def n_design_columns(design_fn: Callable, train: pd.DataFrame, test: pd.DataFrame) -> int:
    Xtr, _ = design_fn(train, test)
    return int(Xtr.shape[1])


def render_step11(rows: list[dict], n_cols: int | None = None) -> str:
    mean_ratio = float(np.mean([r["ratio"] for r in rows if np.isfinite(r["ratio"])]))
    verdict = "⚠️ نشانه‌ی بیش‌برازش" if mean_ratio > _OVERFIT_RATIO_THRESHOLD else "✅ بدون نشانه‌ی بیش‌برازش قابل‌توجه"

    lines = [
        "شکاف pinball@τ بین train و test روی هر ۵ fold رسمی، با بهترین هایپرپارامتر S2 "
        "(بازبرازش واقعی؛ الگوی `src/models/fit_diagnosis.py`، مشابه گام ۱۳).",
        "",
        f"**میانگین نسبت pinball(test)/pinball(train) روی ۵ fold: {mean_ratio:.3f}** — {verdict} "
        f"(آستانه: {_OVERFIT_RATIO_THRESHOLD}).",
        "",
        "| fold | n_train | n_test | pinball train | pinball test | نسبت test/train |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['fold']} | {r['n_train']:,} | {r['n_test']:,} | "
                     f"{r['pinball_train']:.5f} | {r['pinball_test']:.5f} | {r['ratio']:.3f} |")
    lines += ["", "⚠️ **محدودیت سقف $R^2$ (بند ۵.۱۳):** `fit_predict_*` فقط کوانتایل خروجی می‌دهد، "
             "نه پیش‌بینی میانگین زیرین (بند 7.1.1)، پس مقایسه‌ی مستقیم با سقف $R^2\\approx0.4$–$0.5$ "
             "اینجا ممکن نیست — نسبت pinball بالا معادل هم‌ارز آن در واحد معیار پروژه است."]
    if n_cols is not None:
        lines += ["", f"تعداد ستون ماتریس طراحی (پس از یک‌هات‌کردن، **کران بالای** تعداد پارامتر مؤثر — "
                 f"برای مدل‌های منظم‌شده با انتخاب فیچر ضمنی مثل Lasso، تعداد مؤثر واقعی کمتر است): "
                 f"**{n_cols}**."]
    return "\n".join(lines)
