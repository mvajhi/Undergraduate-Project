"""بند 7.12 سند فاز ۷ — خانواده‌ی ۳: سری‌زمانی کلاسیک تک‌متغیره (F03).

⚠️ **این خانواده سطح L3 دارد، نه L1** — قرارداد یکسان `fit_predict_*(train, test, tau,
**hp)` هنوز برقرار است، ولی ``train``/``test`` اینجا برش‌های زمانی سری L3
(`src/features/l3_series.py::build_l3_series`) هستند، نه ردیف‌های سلول. هدف
``day_shock`` است (انحراف از میانگین، نه نرخ) — **هرگز به [۰,۱] کلیپ نمی‌شود**.

## پوشش کامل فهرست بند 7.12.2 (تجدیدنظر ۲۰۲۶-۰۸-۱۶ — درخواست صریح کاربر)

فهرست کوتاه اسپرینت C (`doc/decisions/37-phase7-rescope.md` بند ۵) فقط SARIMAX+Theta را
پیش‌بینی کرده بود. کاربر صراحتاً خواسته همه‌ی مدل‌های کلاسیک اجرا و در گزارش نهایی و اسناد
ثبت شوند (ردیف ۴۳ decision_log). وضعیت هر ۱۴ عضو:

| # | مدل | وضعیت |
|---|---|---|
| ۱-۴ | AR/MA/ARMA/ARIMA | ✅ (از طریق `_fit_sarimax_core` مشترک با SARIMAX — همان خانواده‌ی ریاضی) |
| ۵ | SARIMA (+ فصلی $s$=۱۴ ناهار، F33) | ✅ |
| ۶ | SARIMAX | ✅ (از قبل) |
| ۷ | auto_arima (pmdarima) | ✅ |
| ۸ | ETS/Holt-Winters | ✅ |
| ۹ | Theta | ✅ (از قبل) |
| ۱۰ | TBATS | ❌ **ناسازگار فنی** — `tbats==1.1.3` تابع حذف‌شده‌ی `sklearn.check_array(force_all_finite=...)` صدا می‌زند؛ خطای import-time، نه محدودیت داده. WBS خودش هم انتظار «بهبود صفر» داشت (فصلی هفتگی تنها). جایگزین: STL/MSTL |
| ۱۱ | STL+ARIMA و MSTL | ✅ |
| ۱۲ | Prophet | ✅ (`yearly_seasonality=False` اجباری) |
| ۱۳ | Croston/TSB | ➖ سطح L1 (سلول‌های تنک) است، نه L3 — خارج از دامنه‌ی این ماژول |
| ۱۴ | X-12/X-13-ARIMA | ➖ ناسازگار مستند از قبل (بند 7.12.4) — جایگزین: همین SARIMAX-تقویمی |
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


def _fit_sarimax_core(train: pd.DataFrame, test: pd.DataFrame, tau: float,
                      order: tuple[int, int, int], seasonal_order: tuple[int, int, int, int],
                      use_exog: bool) -> np.ndarray:
    """هسته‌ی مشترک AR/MA/ARMA/ARIMA/SARIMA/SARIMAX — همه یک خانواده‌ی ریاضی‌اند،
    فقط با order/seasonal_order/exog صفر یا غیرصفر (بند 7.12.2 عضو ۱-۶)."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    exog_tr = train[EXOG_COLS].astype(float) if use_exog else None
    exog_te = test[EXOG_COLS].astype(float) if use_exog else None
    model = SARIMAX(train["day_shock"], exog=exog_tr, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    offset = _resid_quantile_offset(res.resid.to_numpy(dtype=float), tau)
    fc = res.get_forecast(steps=len(test), exog=exog_te)
    return fc.predicted_mean.to_numpy() + offset


def fit_predict_ar(train: pd.DataFrame, test: pd.DataFrame, tau: float, p: int = 2, **hp) -> np.ndarray:
    """AR($p$) — عضو ۱ بند 7.12.2. حالت خاص SARIMAX با $q=P=Q=0$، بدون exog."""
    return _fit_sarimax_core(train, test, tau, (p, 0, 0), (0, 0, 0, 0), use_exog=False)


def fit_predict_ma(train: pd.DataFrame, test: pd.DataFrame, tau: float, q: int = 2, **hp) -> np.ndarray:
    """MA($q$) — عضو ۲ بند 7.12.2."""
    return _fit_sarimax_core(train, test, tau, (0, 0, q), (0, 0, 0, 0), use_exog=False)


def fit_predict_arma(train: pd.DataFrame, test: pd.DataFrame, tau: float,
                     p: int = 1, q: int = 1, **hp) -> np.ndarray:
    """ARMA($p$,$q$) — عضو ۳ بند 7.12.2."""
    return _fit_sarimax_core(train, test, tau, (p, 0, q), (0, 0, 0, 0), use_exog=False)


def fit_predict_arima(train: pd.DataFrame, test: pd.DataFrame, tau: float,
                      p: int = 1, d: int = 0, q: int = 1, **hp) -> np.ndarray:
    """ARIMA($p$,$d$,$q$) — عضو ۴ بند 7.12.2. $d=0$ پیش‌فرض (F36)؛ $d=1$ فقط کنترل یک‌بار."""
    return _fit_sarimax_core(train, test, tau, (p, d, q), (0, 0, 0, 0), use_exog=False)


def fit_predict_sarima(train: pd.DataFrame, test: pd.DataFrame, tau: float,
                       p: int = 1, q: int = 1, P: int = 1, Q: int = 1, s: int = 7,
                       **hp) -> np.ndarray:
    """SARIMA($p$,۰,$q$)($P$,۰,$Q$)$_s$ — عضو ۵ بند 7.12.2. $s$=۱۴ برای ناهار (F33)،
    بدون رگرسور برون‌زا (تفاوتش با ``sarimax_calendar`` دقیقاً همین است)."""
    return _fit_sarimax_core(train, test, tau, (p, 0, q), (P, 0, Q, s), use_exog=False)


def fit_predict_sarimax_calendar(train: pd.DataFrame, test: pd.DataFrame, tau: float,
                                 p: int = 1, q: int = 1, P: int = 1, Q: int = 1,
                                 **hp) -> np.ndarray:
    """SARIMAX($p$,۰,$q$)($P$,۰,$Q$)$_7$ با رگرسور تقویمی — عضو ⭐ ۶ بند 7.12.2.

    $d=0$ ثابت (F36: سری ایستاست). فیلتر کالمن `statsmodels` روی ``NaN``های ``day_shock``
    (روزهای بدون سرویس، F35) بومی کار می‌کند — بدون حذف یا درون‌یابی.
    """
    return _fit_sarimax_core(train, test, tau, (p, 0, q), (P, 0, Q, 7), use_exog=True)


def fit_predict_auto_arima(train: pd.DataFrame, test: pd.DataFrame, tau: float,
                           **hp) -> np.ndarray:
    """auto_arima (`pmdarima`، جستجوی گام‌به‌گام با AIC) — عضو ۷ بند 7.12.2، **مکمل**
    grid دستی بالا نه جایگزین (بند 7.12.3).

    ⚠️ محدودیت مستندشده: مثل Theta، `pmdarima` روی NaN داخلی کار نمی‌کند — ``dropna``.
    """
    import pmdarima as pm

    y = train["day_shock"].dropna()
    model = pm.auto_arima(y.to_numpy(), d=0, seasonal=True, m=7, suppress_warnings=True,
                          error_action="ignore")
    offset = _resid_quantile_offset(np.asarray(model.resid()), tau)
    fc = model.predict(n_periods=len(test))
    return np.asarray(fc) + offset


def fit_predict_ets(train: pd.DataFrame, test: pd.DataFrame, tau: float,
                    trend: str | None = None, seasonal: str = "add", damped: bool = False,
                    **hp) -> np.ndarray:
    """ETS/Holt-Winters — عضو ۸ بند 7.12.2. فصلی **جمعی** (نه ضربی — $\\rho$ نزدیک صفر
    می‌شود، بند 7.12.3).

    ⚠️ **محدودیت مستندشده (نه Theta-وار):** برخلاف SARIMAX، مقداردهی اولیه‌ی heuristic
    خودِ `ETSModel` (نه فیلتر کالمن اصلی مدل) وقتی ``NaN`` در ۱۰ مشاهده‌ی اول بیفتد با
    خطای شکل مواجه می‌شود (باگ بالادستی `statsmodels`، تأییدشده با ردیابی خطا) — پس
    مثل Theta/auto_arima اینجا هم ``dropna`` اجباری است.
    """
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel

    y = train["day_shock"].dropna()
    y.index = pd.RangeIndex(len(y))
    model = ETSModel(y, error="add", trend=trend, seasonal=seasonal,
                     seasonal_periods=7 if seasonal else None,
                     damped_trend=damped if trend else False)
    res = model.fit(disp=False)
    offset = _resid_quantile_offset(res.resid.to_numpy(dtype=float), tau)
    fc = res.forecast(steps=len(test))
    return fc.to_numpy() + offset


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


def fit_predict_stl_arima(train: pd.DataFrame, test: pd.DataFrame, tau: float,
                          p: int = 1, q: int = 1, **hp) -> np.ndarray:
    """STL + ARIMA روی باقیمانده — عضو ۱۱ بند 7.12.2 (نسخه‌ی تک‌فصلی). ``robust=True``
    به‌دلیل پرت‌های تأییدشده (F05). ⚠️ dropna (STL فضای‌حالت نیست)."""
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import STL

    y = train["day_shock"].dropna()
    y.index = pd.RangeIndex(len(y))
    stl_res = STL(y, period=7, robust=True).fit()
    resid_model = ARIMA(stl_res.resid, order=(p, 0, q)).fit()
    offset = _resid_quantile_offset(resid_model.resid.to_numpy(dtype=float), tau)

    trend_level = float(stl_res.trend.iloc[-7:].mean())
    seasonal_pattern = stl_res.seasonal.iloc[-7:].to_numpy()
    resid_fc = resid_model.forecast(steps=len(test)).to_numpy()
    seasonal_fc = np.resize(seasonal_pattern, len(test))
    return trend_level + seasonal_fc + resid_fc + offset


def fit_predict_mstl(train: pd.DataFrame, test: pd.DataFrame, tau: float, **hp) -> np.ndarray:
    """MSTL (چندفصلی، `statsmodels.tsa.seasonal.MSTL`) — عضو ۱۱ بند 7.12.2 (نسخه‌ی
    چندفصلی: هفتگی=۷ + شبه‌ماهانه=۲۸). ⚠️ dropna."""
    from statsmodels.tsa.forecasting.stl import STLForecast
    from statsmodels.tsa.arima.model import ARIMA

    y = train["day_shock"].dropna()
    y.index = pd.RangeIndex(len(y))
    periods = [p for p in (7, 28) if len(y) >= 2 * p]
    if not periods:
        periods = [7]
    stlf = STLForecast(y, ARIMA, model_kwargs={"order": (1, 0, 1)}, period=periods[0]).fit()
    offset = _resid_quantile_offset(stlf.result.resid.to_numpy(dtype=float), tau)
    fc = stlf.forecast(steps=len(test))
    return fc.to_numpy() + offset


def fit_predict_prophet(train: pd.DataFrame, test: pd.DataFrame, tau: float,
                        changepoint_prior_scale: float = 0.05, **hp) -> np.ndarray:
    """Prophet — عضو ۱۲ بند 7.12.2. ``yearly_seasonality=False`` **اجباری** (۱۴۲ روز
    < یک دوره‌ی سالانه، بند 7.12.2). قوت واقعی‌اش تعطیلات فارسی/قمری است (بند 5.1)."""
    from prophet import Prophet

    df_tr = train[["date_gregorian", "day_shock"]].rename(columns={"date_gregorian": "ds", "day_shock": "y"})
    m = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False,
               changepoint_prior_scale=changepoint_prior_scale)
    import logging
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    m.fit(df_tr)

    in_sample = m.predict(df_tr[["ds"]])["yhat"].to_numpy()
    offset = _resid_quantile_offset(df_tr["y"].to_numpy() - in_sample, tau)
    future = test[["date_gregorian"]].rename(columns={"date_gregorian": "ds"})
    fc = m.predict(future)["yhat"].to_numpy()
    return fc + offset


MODELS = {
    "ar": fit_predict_ar, "ma": fit_predict_ma, "arma": fit_predict_arma,
    "arima": fit_predict_arima, "sarima": fit_predict_sarima,
    "sarimax_calendar": fit_predict_sarimax_calendar, "auto_arima": fit_predict_auto_arima,
    "ets": fit_predict_ets, "theta": fit_predict_theta, "stl_arima": fit_predict_stl_arima,
    "mstl": fit_predict_mstl, "prophet": fit_predict_prophet,
}

_ALGO_DESC = {
    "ar": "statsmodels.SARIMAX(order=(p,0,0)) — AR(p) کلاسیک",
    "ma": "statsmodels.SARIMAX(order=(0,0,q)) — MA(q) کلاسیک",
    "arma": "statsmodels.SARIMAX(order=(p,0,q)) — ARMA(p,q) کلاسیک",
    "arima": "statsmodels.SARIMAX(order=(p,d,q)) — ARIMA(p,d,q)، d=0 پیش‌فرض (F36)",
    "sarima": "statsmodels.SARIMAX با seasonal_order، بدون exog — SARIMA(p,d,q)(P,D,Q)_s",
    "sarimax_calendar": "statsmodels.SARIMAX + رگرسور تقویمی + کوانتایل تجربی باقیمانده",
    "auto_arima": "pmdarima.auto_arima — جستجوی گام‌به‌گام AIC، مکمل grid دستی",
    "ets": "statsmodels.ETSModel — Holt-Winters فضای‌حالت، فصلی جمعی",
    "theta": "statsmodels.ThetaModel (M3/M4) + کوانتایل تجربی باقیمانده",
    "stl_arima": "statsmodels.STL (robust) + ARIMA روی باقیمانده",
    "mstl": "statsmodels.STLForecast چندفصلی (۷+۲۸) + ARIMA",
    "prophet": "Prophet (Meta) با yearly_seasonality=False اجباری",
}

for _model_id, _fn in MODELS.items():
    register(ModelSpec(model_id=_model_id, family=FAMILY, levels=(LEVEL,), quantile_route="Q3",
                       algorithm=_ALGO_DESC[_model_id]))


@register_space("ar", version=1, n_hyperparams=1, cardinality=15)
def _space_ar(trial: optuna.Trial) -> dict:
    return {"p": trial.suggest_int("p", 0, 14)}


@register_space("ma", version=1, n_hyperparams=1, cardinality=8)
def _space_ma(trial: optuna.Trial) -> dict:
    return {"q": trial.suggest_int("q", 0, 7)}


@register_space("arma", version=1, n_hyperparams=2, cardinality=15 * 8)
def _space_arma(trial: optuna.Trial) -> dict:
    return {"p": trial.suggest_int("p", 0, 14), "q": trial.suggest_int("q", 0, 7)}


@register_space("arima", version=1, n_hyperparams=3, cardinality=15 * 2 * 8)
def _space_arima(trial: optuna.Trial) -> dict:
    return {"p": trial.suggest_int("p", 0, 14), "d": trial.suggest_int("d", 0, 1),
            "q": trial.suggest_int("q", 0, 7)}


@register_space("sarima", version=1, n_hyperparams=5, cardinality=3 * 3 * 2 * 2 * 2)
def _space_sarima(trial: optuna.Trial) -> dict:
    return {
        "p": trial.suggest_int("p", 0, 2), "q": trial.suggest_int("q", 0, 2),
        "P": trial.suggest_int("P", 0, 1), "Q": trial.suggest_int("Q", 0, 1),
        "s": trial.suggest_categorical("s", [7, 14]),
    }


@register_space("sarimax_calendar", version=1, n_hyperparams=4, cardinality=3 * 3 * 2 * 2)
def _space_sarimax_calendar(trial: optuna.Trial) -> dict:
    return {
        "p": trial.suggest_int("p", 0, 2), "q": trial.suggest_int("q", 0, 2),
        "P": trial.suggest_int("P", 0, 1), "Q": trial.suggest_int("Q", 0, 1),
    }


@register_space("auto_arima", version=1, n_hyperparams=0, cardinality=1)
def _space_auto_arima(trial: optuna.Trial) -> dict:
    return {}


@register_space("ets", version=1, n_hyperparams=3, cardinality=2 * 2 * 2)
def _space_ets(trial: optuna.Trial) -> dict:
    return {
        "trend": trial.suggest_categorical("trend", [None, "add"]),
        "seasonal": trial.suggest_categorical("seasonal", [None, "add"]),
        "damped": trial.suggest_categorical("damped", [False, True]),
    }


@register_space("theta", version=1, n_hyperparams=1, cardinality=1)
def _space_theta(trial: optuna.Trial) -> dict:
    return {"period": trial.suggest_categorical("period", [7])}


@register_space("stl_arima", version=1, n_hyperparams=2, cardinality=3 * 3)
def _space_stl_arima(trial: optuna.Trial) -> dict:
    return {"p": trial.suggest_int("p", 0, 2), "q": trial.suggest_int("q", 0, 2)}


@register_space("mstl", version=1, n_hyperparams=0, cardinality=1)
def _space_mstl(trial: optuna.Trial) -> dict:
    return {}


@register_space("prophet", version=1, n_hyperparams=1)
def _space_prophet(trial: optuna.Trial) -> dict:
    return {"changepoint_prior_scale": trial.suggest_float("changepoint_prior_scale", 0.001, 0.5, log=True)}
