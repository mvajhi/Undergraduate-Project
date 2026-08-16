"""بند 7.22 — لایه‌ی کالیبراسیون متعامد: CQR (Conformal Quantile Regression، Romano,
Patterson & Candès 2019) + Mondrian/Group-Conditional Conformal (بند 7.22.1 عضو ۵،
«⭐⭐ مهم‌ترین عضو» طبق خودِ WBS).

🔄 طبق بند 7.22 بازنویسی‌شده (ردیف ۳۷ decision_log)، این لایه یک اسپرینت زودهنگام
است، نه آخرین گام — روی قهرمانان تأییدشده‌ی اسپرینت C اعمال می‌شود، نه منتظر پایان
همه‌ی خانواده‌ها می‌ماند.

## چرا این با ``calibration.py`` فرق دارد

`calibration.py` فقط **می‌سنجد** پوشش تجربی چقدر است (تشخیصی، گام ۱۳ کارت مدل).
این ماژول واقعاً **تصحیح می‌کند** — یک لایه‌ی پس‌پردازش که روی خروجی هر مدل دیگری
می‌نشیند (بند 7.22 «جایگاه نظری»)، دقیقاً مثل `baselines.quantile_adjust` با یک
تفاوت حیاتی: تصحیح روی یک **زیرمجموعه‌ی کالیبراسیونِ ندیده** (نه باقیمانده‌ی خودِ
train) برآورد می‌شود — این همان چیزی است که ضمانت پوشش نمونه‌محدود (finite-sample)
CQR را می‌دهد، نه فقط یک آفست تجربی دلبخواه.

## سه تله‌ی بند 7.22.3 که اینجا رعایت شده

۱. **مجموعه‌ی کالیبراسیون زمانی است** (`_time_split`، نه split تصادفی) — چون
   ICC(روز)=۰.۲۲۵ (F10)، split تصادفی نشتی مؤثر می‌سازد.
۲. **گروه خیلی ریز ⇒ سقوط امن به تصحیح سراسری** (`_MIN_GROUP_CALIB`).
۳. اندازه‌ی کالیبراسیون ۲۰٪ انتهای پنجره‌ی آموزش (داخل بازه‌ی ۱۰-۳۰٪ مصوب).
"""

import numpy as np
import pandas as pd

#: بند 7.22.2 — اندازه‌ی مجموعه‌ی کالیبراسیون (۱۰-۳۰٪ انتهای پنجره‌ی آموزش)
CALIB_FRAC = 0.20
#: زیر این تعداد نمونه‌ی کالیبراسیون در یک گروه، تصحیح گروهی بی‌ثبات است
_MIN_GROUP_CALIB = 20


def _time_split(train: pd.DataFrame, date_col: str, calib_frac: float = CALIB_FRAC
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """بند 7.22.3 تله‌ی ۱: کالیبراسیون باید زمانی باشد، نه split تصادفی."""
    dates = sorted(train[date_col].unique())
    split = max(1, int(len(dates) * (1 - calib_frac)))
    proper_dates, calib_dates = set(dates[:split]), set(dates[split:])
    proper = train[train[date_col].isin(proper_dates)]
    calib = train[train[date_col].isin(calib_dates)]
    return proper, calib


def cqr_predict(fit_fn, train: pd.DataFrame, test: pd.DataFrame, tau: float, hyperparams: dict,
                date_col: str = "date_gregorian", group_col: str | None = None
                ) -> tuple[np.ndarray, dict]:
    """پیش‌بینی کالیبره‌شده با CQR. ``group_col=None`` ⇒ سراسری؛ وگرنه Mondrian
    (تصحیح جدا به‌ازای هر مقدار ``group_col``، بند 7.22.1 عضو ۵).

    برمی‌گرداند: (پیش‌بینی کالیبره‌شده، دیکشنری تصحیح هر گروه — برای گزارش/بازرسی).
    """
    proper, calib = _time_split(train, date_col)
    if len(calib) < _MIN_GROUP_CALIB or len(proper) < _MIN_GROUP_CALIB:
        pred_test = np.asarray(fit_fn(train, test, tau, **hyperparams), dtype=float)
        return np.clip(pred_test, 0.0, 1.0), {"_global": 0.0, "_note": "کالیبراسیون ناکافی — تصحیح صفر"}

    pred_calib = np.asarray(fit_fn(proper, calib, tau, **hyperparams), dtype=float)
    pred_test = np.asarray(fit_fn(proper, test, tau, **hyperparams), dtype=float)
    resid_calib = calib["rho"].to_numpy() - pred_calib
    global_correction = float(np.quantile(resid_calib, tau))

    if group_col is None:
        return np.clip(pred_test + global_correction, 0.0, 1.0), {"_global": global_correction}

    corrections: dict = {"_global": global_correction}
    calib_groups = calib[group_col].to_numpy()
    test_groups = test[group_col].to_numpy()
    out = pred_test.copy()
    for g in pd.unique(test_groups):
        mask_c = calib_groups == g
        # بند 7.22.3 تله‌ی ۲: گروه خیلی ریز در کالیبراسیون ⇒ سقوط امن به تصحیح سراسری
        corr = float(np.quantile(resid_calib[mask_c], tau)) if mask_c.sum() >= _MIN_GROUP_CALIB else global_correction
        corrections[str(g)] = corr
        out[test_groups == g] = pred_test[test_groups == g] + corr
    return np.clip(out, 0.0, 1.0), corrections


def oof_calibrated_predictions(fit_fn, folds: list, tau: float, hyperparams: dict,
                               group_col: str | None = None) -> pd.DataFrame:
    """CQR روی هر ۵ fold رسمی — برای هر fold، کالیبراسیون از **همان train** آن fold
    گرفته می‌شود (نه نشتی از test)، سپس روی test همان fold اعمال می‌شود."""
    parts = []
    for tr, te in folds:
        pred_cal, _ = cqr_predict(fit_fn, tr, te, tau, hyperparams, group_col=group_col)
        parts.append(pd.DataFrame({
            "actual": te["rho"].to_numpy(), "pred_q": pred_cal,
            "RestaurantName": te["RestaurantName"].to_numpy(), "Meal": te["Meal"].to_numpy(),
            "Res": te["Res"].to_numpy(), "is_tehran": te["is_tehran"].to_numpy(),
        }))
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# ACI — Adaptive Conformal Inference (Gibbs & Candès 2021)، بند 7.22.1 عضو ۴
# ---------------------------------------------------------------------------
#
# یافته‌ی ۲۰ (`doc/progress/07-*.md`): CQR ایستا روی هر ۴ قهرمان بدتر شد چون علامت
# تصحیح بین foldها ناپایدار است — رژیم زمانی عوض می‌شود (تعطیلات/امتحانات/رمضان).
# ACI دقیقاً برای همین طراحی شده: به‌جای یک تصحیح ثابت برآوردشده از یک بازه‌ی
# کالیبراسیونِ گذشته، تصحیح را **روز به روز، آنلاین** به‌روزرسانی می‌کند — اگر پوشش
# اخیر کمتر از τ بوده (خطای زیاد)، تصحیح بالا می‌رود؛ اگر بیشتر بوده، پایین می‌آید.
# مدل زیرین فقط **یک‌بار** برازش می‌شود (روی proper-train)؛ آنچه آنلاین به‌روز
# می‌شود فقط آفست است — هزینه‌ی محاسباتی عملاً برابر CQR است.

#: نرخ یادگیری ACI — بند 7.22.1: بدون منبع اختصاصی پروژه، مقدار مرجع مقالات (γ≈۰.۰۵)
ACI_GAMMA = 0.05


def aci_predict(fit_fn, train: pd.DataFrame, test: pd.DataFrame, tau: float, hyperparams: dict,
                date_col: str = "date_gregorian", gamma: float = ACI_GAMMA
                ) -> tuple[np.ndarray, list[float]]:
    """ACI — تصحیح اولیه از CQR (بند بالا)، سپس **روز به روز** به‌روزرسانی آنلاین:
    ``correction += gamma * (tau - miscoverage_rate_روز)``. برمی‌گرداند: (پیش‌بینی
    کالیبره‌شده، مسیر تصحیح در طول زمان — برای بازرسی/گزارش)."""
    proper, calib = _time_split(train, date_col)
    if len(calib) < _MIN_GROUP_CALIB or len(proper) < _MIN_GROUP_CALIB:
        pred_test = np.asarray(fit_fn(train, test, tau, **hyperparams), dtype=float)
        return np.clip(pred_test, 0.0, 1.0), [0.0]

    pred_calib = np.asarray(fit_fn(proper, calib, tau, **hyperparams), dtype=float)
    correction = float(np.quantile(calib["rho"].to_numpy() - pred_calib, tau))

    # مدل فقط یک‌بار روی کل test برازش/پیش‌بینی می‌شود — آنلاین‌بودن فقط در تصحیح است
    pred_test_base = np.asarray(fit_fn(proper, test, tau, **hyperparams), dtype=float)

    out = np.empty(len(test), dtype=float)
    test_dates = test[date_col].to_numpy()
    actual = test["rho"].to_numpy()
    path = [correction]
    for day in pd.unique(test_dates):
        mask = test_dates == day
        pred_day = np.clip(pred_test_base[mask] + correction, 0.0, 1.0)
        out[mask] = pred_day
        # ⚠️ باید «نرخ پوشش» باشد (actual <= pred)، نه «نرخ نقض» — چون هدف P(actual<=pred)=τ
        # است، نه ۱−τ. علامت اشتباه (نرخ نقض) باعث واگرایی نامحدود تصحیح می‌شد (رگرسیون تست).
        coverage_rate = float((actual[mask] <= pred_day).mean())
        correction += gamma * (tau - coverage_rate)
        path.append(correction)
    return out, path


def oof_aci_predictions(fit_fn, folds: list, tau: float, hyperparams: dict,
                        date_col: str = "date_gregorian", gamma: float = ACI_GAMMA) -> pd.DataFrame:
    """ACI روی هر ۵ fold رسمی — تصحیح هر fold از صفر شروع می‌شود (بدون نشتی بین fold)."""
    parts = []
    for tr, te in folds:
        pred_cal, _ = aci_predict(fit_fn, tr, te, tau, hyperparams, date_col=date_col, gamma=gamma)
        parts.append(pd.DataFrame({
            "actual": te["rho"].to_numpy(), "pred_q": pred_cal,
            "RestaurantName": te["RestaurantName"].to_numpy(), "Meal": te["Meal"].to_numpy(),
            "Res": te["Res"].to_numpy(), "is_tehran": te["is_tehran"].to_numpy(),
        }))
    return pd.concat(parts, ignore_index=True)
