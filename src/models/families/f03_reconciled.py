"""آشتی سطح L3→L1 (بند 7.24، رویکرد شبیه H2) — قهرمان خ۳ (`ma`, یافته‌ی ۲۸) با
میانگین تاریخی هر سلف ترکیب می‌شود تا کوانتایل سطح **سلولی** بدهد.

⚠️ **این ماژول قرارداد یکسان L1 دارد** (`fit_predict_*(train, test, tau, **hp)` با
``train``/``test`` = ردیف‌های سلول، نه سری L3) — برخلاف `f03_timeseries.py` — دقیقاً
برای همین مستقیماً با `model_comparison.py`/`mandatory_cuts.py` سازگار است.

## چرا این بازتست معماری دومرحله‌ای متفاوت از یافته‌ی ۱۴ است

اسپرینت A (`axis_screening.py::_two_stage_predict`) «عامل روز» را با یک رگرسیون
خطی/LGBM سطح‌روز تخمین زد، نه یک مدل زمانی واقعی — و نتیجه «بی‌اثر» گرفت. اینجا
stage1 واقعاً `MA(2)` (برنده‌ی مستندشده‌ی خ۳، یافته‌ی ۲۸) است.

## معماری سه‌جزئی (نه دوجزئی مثل یافته‌ی ۱۴)

$\\hat\\rho_{d,m,r}(\\tau) = \\bar\\rho_{m,r}^{\\text{train}} + \\widehat{\\text{shock}}_{d,m} +
\\text{offset}_\\tau$

۱. میانگین تاریخی سلف-وعده (فقط از train). ۲. پیش‌بینی نقطه‌ای MA(2) روی سری
day_shock (نه کوانتایل — آفست یک‌بار در پایان اعمال می‌شود، نه سه‌بار روی هم انباشته).
۳. یک آفست کوانتایل تجربی τ روی باقیمانده‌ی **ترکیبی** (نه سه آفست جداگانه که هم
نمی‌شوند) — دقیقاً برای اجتناب از مشکل «کوانتایل مجموع ≠ مجموع کوانتایل‌ها» که در
یافته‌ی ۱۴ به‌عنوان محدودیت مستند شده بود.
"""

import numpy as np
import pandas as pd

from src.features.aggregate_features import build_day_factor
from src.models.registry import ModelSpec, register

FAMILY = "F03"
LEVEL = "L1"  # ⚠️ خروجی سلولی است (بالا)، برخلاف بقیه‌ی خ۳ که L3 دارند

MA_ORDER = 2


def _forecast_day_shock(cell_train: pd.DataFrame, test_dates: pd.DataFrame) -> pd.Series:
    """MA(2) روی سری day_shock هر وعده (ساخته‌شده فقط از ``cell_train``)، پیش‌بینی
    **نقطه‌ای** (بدون آفست کوانتایل) برای هر (تاریخ, وعده) در ``test_dates``."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    cell = cell_train.rename(columns={"rho": "rho_cell"})
    day = build_day_factor(cell)[["date_gregorian", "Meal", "day_shock"]]

    preds = []
    for meal in test_dates["Meal"].unique():
        te_dates_meal = pd.to_datetime(test_dates.loc[test_dates["Meal"] == meal, "date_gregorian"].unique())
        sub = day[day["Meal"] == meal].sort_values("date_gregorian")
        full_range = pd.date_range(sub["date_gregorian"].min(), max(sub["date_gregorian"].max(), te_dates_meal.max()), freq="D")
        y = sub.set_index("date_gregorian")["day_shock"].reindex(full_range)

        n_train_days = (full_range <= sub["date_gregorian"].max()).sum()
        y_fit = y.iloc[:n_train_days]
        y_fit = y_fit.asfreq("D")
        steps = len(full_range) - n_train_days
        if steps <= 0:
            continue

        model = SARIMAX(y_fit, order=(0, 0, MA_ORDER), enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(disp=False)
        # ⚠️ ``get_forecast(steps=...)`` در این نسخه‌ی statsmodels گاه‌گاهی با خطای
        # «end after start» شکست می‌خورد (باگ ناپایدار در استنتاج ایندکس تاریخی داخلی؛
        # تکرارپذیر نبود با ورودی یکسان). موقعیت صریح عددی (start/end) پایدار است.
        fc = res.get_prediction(start=len(y_fit), end=len(y_fit) + steps - 1).predicted_mean
        fc.index = full_range[n_train_days:]

        wanted = fc.reindex(te_dates_meal)
        preds.append(pd.DataFrame({"date_gregorian": wanted.index, "Meal": meal, "day_shock_hat": wanted.to_numpy()}))

    return pd.concat(preds, ignore_index=True) if preds else pd.DataFrame(
        columns=["date_gregorian", "Meal", "day_shock_hat"])


def fit_predict_l3_reconciled_ma(train: pd.DataFrame, test: pd.DataFrame, tau: float, **hp) -> np.ndarray:
    restaurant_mean = train.groupby(["RestaurantName", "Meal"], observed=True)["rho"].mean()

    day_shock_hat = _forecast_day_shock(train, test[["date_gregorian", "Meal"]].drop_duplicates())
    shock_map = day_shock_hat.set_index(["date_gregorian", "Meal"])["day_shock_hat"]

    te = test.copy()
    te["date_gregorian"] = pd.to_datetime(te["date_gregorian"])
    te_base = te.set_index(["RestaurantName", "Meal"]).index.map(restaurant_mean).to_numpy(dtype=float)
    te_shock = te.set_index(["date_gregorian", "Meal"]).index.map(shock_map).to_numpy(dtype=float)
    te_shock = np.nan_to_num(te_shock, nan=0.0)  # سری خیلی کوتاه/شکاف کامل ⇒ بدون شوک اضافه (لنگر=میانگین سلف)
    point_pred_test = te_base + te_shock

    tr_base = train.set_index(["RestaurantName", "Meal"]).index.map(restaurant_mean).to_numpy(dtype=float)
    day_shock_train = _forecast_in_sample_day_shock(train)
    tr_shock = train.set_index(["date_gregorian", "Meal"]).index.map(day_shock_train).to_numpy(dtype=float)
    tr_shock = np.nan_to_num(tr_shock, nan=0.0)
    point_pred_train = tr_base + tr_shock

    combined_resid = train["rho"].to_numpy() - point_pred_train
    offset = float(np.quantile(combined_resid[np.isfinite(combined_resid)], tau))

    return np.clip(point_pred_test + offset, 0.0, 1.0)


def _forecast_in_sample_day_shock(cell_train: pd.DataFrame) -> pd.Series:
    """مقدار **مشاهده‌شده‌ی** day_shock روی خودِ train (نه پیش‌بینی) — برای محاسبه‌ی
    باقیمانده‌ی ترکیبی سازگار با نحوه‌ی ساخت offset."""
    cell = cell_train.rename(columns={"rho": "rho_cell"})
    day = build_day_factor(cell)[["date_gregorian", "Meal", "day_shock"]]
    day["date_gregorian"] = pd.to_datetime(day["date_gregorian"])
    return day.set_index(["date_gregorian", "Meal"])["day_shock"]


MODELS = {"l3_reconciled_ma": fit_predict_l3_reconciled_ma}

register(ModelSpec(model_id="l3_reconciled_ma", family=FAMILY, levels=(LEVEL,), quantile_route="Q3",
                   algorithm="MA(2) روی day_shock + میانگین تاریخی سلف + یک آفست کوانتایل ترکیبی"))
