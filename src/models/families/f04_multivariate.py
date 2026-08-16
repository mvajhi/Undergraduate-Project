"""بند 7.13 سند فاز ۷ — خانواده‌ی ۴: سری‌زمانی چندمتغیره/پنلی (F04).

⚠️ **سطح L4** (پنل wide، ۴۱ سری $(m,r)$؛ `src/features/l4_series.py`)، نه L1. طبق فهرست
کوتاه اسپرینت C (`doc/decisions/37-phase7-rescope.md` بند ۵: «خ۴ چندمتغیره: DFM») فقط
یک عضو ساخته شده — **Dynamic Factor Model** (`statsmodels.DynamicFactorMQ`، عضو ⭐۶ بند
7.13.1)، نه VAR کامل (بند 7.13.3: ۴۱ سری ⇒ ۱٬۷۲۲ پارامتر، غیرقابل‌برآورد).

هدف خودِ نرخ $\\rho$ است (نه انحراف مثل خ۳) — خروجی به [۰,۱] کلیپ می‌شود.

## چرا مقایسه‌ی مستقیم با B3 اینجا نیست

همان دلیل خ۳: خروجی سطح $(d,m,r)$ (میانگین روی غذا) است، نه سطح سلولی $(d,m,r,f)$
`model_comparison.md`. مرجع اینجا **کوانتایل تجربی خودِ هر سری** (نادیده‌گرفتن عامل
مشترک) است — دقیقاً همان چیزی که DFM باید بر آن اضافه‌ارزش نشان دهد.
"""

import numpy as np
import pandas as pd


def fit_predict_dfm(train_panel: pd.DataFrame, test_panel: pd.DataFrame, tau: float,
                    k_factors: int = 1, factor_order: int = 1, **hp) -> pd.DataFrame:
    """DFM روی پنل wide — برمی‌گرداند DataFrame هم‌شکل ``test_panel`` با کوانتایل $\\tau$.

    ⚠️ قرارداد این تابع با بقیه‌ی خانواده‌ها فرق دارد (پنل، نه ردیف‌های سلولی) — عمداً،
    چون DFM ذاتاً چندمتغیره است. فراخوان (`run_f04_feasibility.py`) این تفاوت را مستند و
    مدیریت می‌کند.
    """
    from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ

    model = DynamicFactorMQ(train_panel, factors=k_factors, factor_orders=factor_order,
                            idiosyncratic_ar1=True)
    res = model.fit(disp=False, maxiter=100)

    resid = train_panel - res.predict()
    offsets = resid.apply(lambda col: np.nanquantile(col.dropna(), tau) if col.notna().sum() >= 5
                          else np.nanquantile(resid.to_numpy(), tau))

    fc = res.get_forecast(steps=len(test_panel))
    mu = fc.predicted_mean
    mu.index = test_panel.index
    out = (mu + offsets).clip(lower=0.0, upper=1.0)
    return out[train_panel.columns]


MODELS = {"dfm": fit_predict_dfm}
