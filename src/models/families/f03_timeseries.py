"""بند 7.12 سند فاز ۷ — خانواده‌ی ۳: سری‌زمانی کلاسیک تک‌متغیره (F03).

⚠️ **این خانواده سطح L3 دارد، نه L1** — قرارداد یکسان `fit_predict_*(train, test, tau,
**hp)` هنوز برقرار است، ولی ``train``/``test`` اینجا برش‌های زمانی سری L3
(`src/features/l3_series.py::build_l3_series`) هستند، نه ردیف‌های سلول. هدف
``day_shock`` است (انحراف از میانگین، نه نرخ) — **هرگز به [۰,۱] کلیپ نمی‌شود**.

طبق فهرست کوتاه اسپرینت C (`doc/decisions/37-phase7-rescope.md` بند ۵): فقط دو عضو از
۱۴ عضو بند 7.12.2 ساخته شده‌اند — **SARIMAX** (با رگرسور تقویمی) و **Theta**.

## چرا مقایسه‌ی مستقیم با B3 اینجا نیست

خروجی این ماژول نرخ سلولی $\\rho$ نیست، ``day_shock`` سطح $(d,m)$ است — یک واحد متفاوت.
تطبیق آن به سطح L1 نیازمند رویکرد آشتی H2/H3 (بند 7.24 WBS) است که در فهرست کوتاه فعلی
اسپرینت C نیست؛ ارزیابی این خانواده روی معیار **خودِ سطح L3** انجام می‌شود (بند 7.12.5:
تشخیص باقیمانده، مقایسه‌ی مدیریت شکاف)، نه روی جدول مقایسه‌ی L1 (`model_comparison.md`).
"""

import numpy as np
import optuna
import pandas as pd

from src.models.registry import ModelSpec, register
from src.models.spaces import register_space

FAMILY = "F03"
LEVEL = "L3"

EXOG_COLS = ["is_holiday_any", "is_day_before_holiday", "is_exam_period",
            "is_final_exam_period", "is_nowruz_block"]


def _resid_quantile_offset(resid: np.ndarray, tau: float) -> float:
    resid = resid[np.isfinite(resid)]
    return float(np.quantile(resid, tau)) if len(resid) else 0.0


def fit_predict_sarimax_calendar(train: pd.DataFrame, test: pd.DataFrame, tau: float,
                                 p: int = 1, q: int = 1, P: int = 1, Q: int = 1,
                                 **hp) -> np.ndarray:
    """SARIMAX($p$,۰,$q$)($P$,۰,$Q$)$_7$ با رگرسور تقویمی — عضو ⭐ ۶ بند 7.12.2.

    $d=0$ ثابت (F36: سری ایستاست). فیلتر کالمن `statsmodels` روی ``NaN``های ``day_shock``
    (روزهای بدون سرویس، F35) بومی کار می‌کند — بدون حذف یا درون‌یابی.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    model = SARIMAX(train["day_shock"], exog=train[EXOG_COLS].astype(float),
                    order=(p, 0, q), seasonal_order=(P, 0, Q, 7),
                    enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    offset = _resid_quantile_offset(res.resid.to_numpy(dtype=float), tau)
    fc = res.get_forecast(steps=len(test), exog=test[EXOG_COLS].astype(float))
    return fc.predicted_mean.to_numpy() + offset


def fit_predict_theta(train: pd.DataFrame, test: pd.DataFrame, tau: float,
                      period: int = 7, **hp) -> np.ndarray:
    """Theta (Assimakopoulos & Nikolopoulos 2000) — عضو ۹ بند 7.12.2، محک M3/M4.

    ⚠️ **محدودیت مستندشده:** برخلاف SARIMAX، Theta فضای‌حالت نیست و ``NaN`` داخلی را
    بومی نمی‌پذیرد — اینجا فقط روی روزهای **سرویس‌داده‌شده** (``dropna``) برازش می‌شود،
    یعنی ساختار شکاف رمضان/جمعه را نمی‌بیند. این خودش یک یافته‌ی مقایسه‌ای است (بند
    7.12.5: «مقایسه‌ی مدیریت شکاف: کالمن در برابر...»), نه باگ.
    """
    from statsmodels.tsa.forecasting.theta import ThetaModel

    y = train["day_shock"].dropna()
    y.index = pd.RangeIndex(len(y))
    model = ThetaModel(y, period=period, deseasonalize=len(y) >= 2 * period)
    res = model.fit()
    # ThetaModel معادل resid مستقیم ندارد — برای آفست کوانتایل از انحراف نسبت به
    # میانگین کل سری استفاده می‌شود (تقریب محافظه‌کارانه، نه رگرسیون کامل)
    naive_resid = y.to_numpy() - y.to_numpy().mean()
    offset = _resid_quantile_offset(naive_resid, tau)
    fc = res.forecast(steps=len(test))
    return fc.to_numpy() + offset


MODELS = {"sarimax_calendar": fit_predict_sarimax_calendar, "theta": fit_predict_theta}

register(ModelSpec(model_id="sarimax_calendar", family=FAMILY, levels=(LEVEL,), quantile_route="Q3",
                   algorithm="statsmodels.SARIMAX + رگرسور تقویمی + کوانتایل تجربی باقیمانده"))
register(ModelSpec(model_id="theta", family=FAMILY, levels=(LEVEL,), quantile_route="Q3",
                   algorithm="statsmodels.ThetaModel (M3/M4) + کوانتایل تجربی باقیمانده"))

@register_space("sarimax_calendar", version=1, n_hyperparams=4, cardinality=3 * 3 * 2 * 2)
def _space_sarimax_calendar(trial: optuna.Trial) -> dict:
    return {
        "p": trial.suggest_int("p", 0, 2), "q": trial.suggest_int("q", 0, 2),
        "P": trial.suggest_int("P", 0, 1), "Q": trial.suggest_int("Q", 0, 1),
    }


@register_space("theta", version=1, n_hyperparams=1, cardinality=1)
def _space_theta(trial: optuna.Trial) -> dict:
    return {"period": trial.suggest_categorical("period", [7])}
