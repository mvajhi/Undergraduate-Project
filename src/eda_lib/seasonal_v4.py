"""تجزیه‌ی فصلی گسسته‌ی سری‌های v4 (بندهای ۱۴-۵-۱ تا ۱۴-۵-۷).

## چرا رگرسیون هارمونیک و نه مستقیماً MSTL

F35 ثبت کرده که گمشدگی روزها **MCAR نیست**: تعطیلات، تابستان و پنجره‌ی شوک شکاف
می‌سازند، و درون‌یابی الگوی هفتگی را تحریف می‌کند. WBS صریحاً **درون‌یابی خطی را
ممنوع** کرده.

MSTL ورودی هم‌فاصله می‌خواهد. دو راه معمول هر دو خراب‌اند:
- درون‌یابی شکاف‌ها ⇒ همان چیزی که ممنوع شده.
- شاخص‌گذاری با «روز سرویس» به‌جای تاریخ ⇒ شکاف ۳ ماهه‌ی تابستان را به یک گام
  تبدیل می‌کند و مؤلفه‌ی سالانه را بی‌معنا می‌کند.

رگرسیون هارمونیک روی **روزهای مشاهده‌شده** هر دو مشکل را ندارد: شکاف صرفاً یعنی آن
سطر در ماتریس طرح نیست. قیمتش این است که شکل فصلی را به چند هارمونیک محدود می‌کند —
که با ۳ سال داده محدودیت درستی است، نه سخت‌گیری بی‌دلیل.

MSTL به‌عنوان **اعتبارسنجی متقاطع** روی بلندترین قطعه‌های بی‌شکاف اجرا می‌شود، نه
به‌عنوان روش اصلی.

## معیار مقایسه

هر مؤلفه با **سهم واریانس افزایشی** گزارش می‌شود (پس از ورود مؤلفه‌های قبلی)، نه
حاشیه‌ای — چون مؤلفه‌ها متعامد نیستند و گزارش حاشیه‌ای مجموعشان از ۱۰۰٪ بیشتر می‌شود.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def fourier_terms(pos: np.ndarray, k: int, prefix: str) -> pd.DataFrame:
    """$k$ جفت جمله‌ی سینوسی/کسینوسی روی موقعیت نرمال‌شده‌ی $[0,1)$."""
    out = {}
    for j in range(1, k + 1):
        out[f"{prefix}_sin{j}"] = np.sin(2 * np.pi * j * pos)
        out[f"{prefix}_cos{j}"] = np.cos(2 * np.pi * j * pos)
    return pd.DataFrame(out)


def _ols_r2(X: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> tuple[float, np.ndarray]:
    """$R^2$ و باقی‌مانده‌ی رگرسیون وزن‌دار با شبه‌وارون (پایدار در برابر هم‌خطی)."""
    if w is None:
        w = np.ones(len(y))
    sw = np.sqrt(w)
    Xw, yw = X * sw[:, None], y * sw
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    pred = X @ beta
    resid = y - pred
    ybar = np.average(y, weights=w)
    ss_res = float(np.sum(w * resid ** 2))
    ss_tot = float(np.sum(w * (y - ybar) ** 2))
    return 1 - ss_res / ss_tot, resid


def incremental_variance(y: np.ndarray, blocks: dict[str, np.ndarray],
                         w: np.ndarray | None = None) -> pd.DataFrame:
    """سهم واریانس **افزایشی** هر بلوک، به ترتیب ورود.

    هر بلوک یک ماتریس ستون‌هاست. ترتیب اهمیت دارد و باید در سند اعلام شود.
    """
    n = len(y)
    X = np.ones((n, 1))
    prev, rows = 0.0, []
    for name, B in blocks.items():
        X = np.hstack([X, B])
        r2, _ = _ols_r2(X, y, w)
        rows.append({"مؤلفه": name, "ستون": B.shape[1], "R² تجمعی": r2, "سهم افزایشی": r2 - prev})
        prev = r2
    rows.append({"مؤلفه": "باقی‌مانده", "ستون": 0, "R² تجمعی": prev, "سهم افزایشی": 1 - prev})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# موقعیت در سه تقویم
# ---------------------------------------------------------------------------

def calendar_positions(f: pd.DataFrame) -> pd.DataFrame:
    """موقعیت نرمال‌شده‌ی هر روز در **سه** تقویم — ورودی بند ۱۴-۵-۲.

    - **میلادی:** روزِ سال میلادی / ۳۶۵
    - **شمسی:** روزِ سال شمسی از ۱ فروردین / ۳۶۵
    - **تحصیلی:** روز از آغاز سال تحصیلی (نخستین روز سرویسِ آن سال) / طول آن سال
    """
    out = pd.DataFrame(index=f.index)
    d = f["date_gregorian"]
    out["pos_greg"] = (d.dt.dayofyear - 1) / 365.0

    # روز سال شمسی از ماه و روزِ شمسی (۶ ماه ۳۱ روزه، ۵ ماه ۳۰ روزه، اسفند ۲۹/۳۰)
    cum = np.array([0, 31, 62, 93, 124, 155, 186, 216, 246, 276, 306, 336])
    m = f["month_jalali"].to_numpy().astype(int)
    day = f["day_of_month_jalali"].to_numpy().astype(int)
    out["pos_jalali"] = (cum[m - 1] + day - 1) / 365.0

    # ⚠️ ۳۶ روز سرویس (۳.۱٪ پرس) هیچ سال تحصیلی‌ای ندارند — تعطیلات تابستان و
    # بین دو نیمسال. حذفشان اریبی انتخاب می‌سازد (نرخشان ۰.۰۷۴ در برابر ۰.۰۸۱)،
    # پس به سال تحصیلی **قبلی** الحاق می‌شوند: از نظر معنایی «دُمِ» همان سال‌اند.
    order = d.argsort()
    ay = f["academic_year"].copy()
    ay.iloc[order] = ay.iloc[order].ffill().bfill()
    start = d.groupby(ay).transform("min")
    end = d.groupby(ay).transform("max")
    span = (end - start).dt.days.clip(lower=1)
    out["pos_acad"] = ((d - start).dt.days / span).clip(0, 0.9999)
    out["academic_year_filled"] = ay.to_numpy()
    return out
