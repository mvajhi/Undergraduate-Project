"""بارگذاری مشترک و ابزار چاپ دور ۹ (سطح فرد، پنجره‌ی v4).

تفاوت با `_common_v4.py`: آنجا مرجع شبکه‌ی سلول L1 است؛ اینجا **ماتریس پروفایل
فرد** (`person_profile_v4.parquet`، ۴۶٬۵۲۳ ردیف × ۷۲ ستون).

## قاعده‌ی الف دور ۹ در کد

هیچ تابعی از این ماژول آماره‌ی سطح فردی برنمی‌گرداند بدون اینکه آستانه‌ی
$n_{\\min}$ی که رویش حساب شده صریح باشد. `at_threshold()` تنها راه اعمال آستانه
است تا هیچ‌جا آستانه‌ی ضمنی جا نماند.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import set_global_seed  # noqa: F401  — اجراکننده‌ها از اینجا می‌گیرند
from src.eda_lib import figs
from src.eda_lib.person_profile_v4 import build_person_profile_v4

#: آستانه‌های مرجع دور ۹ (جدول بند ۹-۳-۴).
THRESHOLDS = (1, 5, 10, 20, 50, 100)

#: آستانه‌ی اصلی تحلیل‌های خوشه‌بندی و آستانه‌ی مقاومت (بند ۹-۷-۱).
N_MIN_MAIN = 20
N_MIN_ROBUST = 50

#: سه تعریف «بدترین» (قاعده‌ی ب). کلید = نام ستون، مقدار = برچسب فارسی.
WORST_DEFS = {
    "rho_raw": "نرخ خام",
    "rho_eb": "نرخ منقبض‌شده (EB)",
    "n_waste": "هدررفت مطلق",
}


def load_profile() -> pd.DataFrame:
    return build_person_profile_v4()


def at_threshold(P: pd.DataFrame, n_min: int) -> pd.DataFrame:
    """زیرمجموعه‌ی افراد با دست‌کم `n_min` رزرو — تنها راه مجاز اعمال آستانه."""
    return P[P.n_res >= n_min]


def coverage_row(P: pd.DataFrame, n_min: int) -> dict:
    """سهم افراد، رزرو و هدررفتِ باقی‌مانده در آستانه‌ی `n_min`."""
    s = at_threshold(P, n_min)
    return {"n_min": n_min,
            "افراد": len(s),
            "سهم افراد": len(s) / len(P),
            "سهم رزرو": s.n_res.sum() / P.n_res.sum(),
            "سهم هدررفت": s.n_waste.sum() / P.n_waste.sum(),
            "نرخ وزنی": s.n_waste.sum() / s.n_res.sum()}


def gini(x: np.ndarray) -> float:
    """ضریب جینی با تعریف نیم‌میانگینِ اختلاف مطلق (پایدار برای مقادیر صفر زیاد)."""
    x = np.sort(np.asarray(x, dtype="float64"))
    n = x.size
    if n == 0 or x.sum() == 0:
        return float("nan")
    idx = np.arange(1, n + 1)
    return float((2 * idx - n - 1).dot(x) / (n * x.sum()))


def lorenz(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """منحنی لورنتس: سهم تجمعی جمعیت (از بدترین) در برابر سهم تجمعی مقدار."""
    x = np.sort(np.asarray(x, dtype="float64"))[::-1]
    cum = np.cumsum(x) / x.sum()
    pop = np.arange(1, x.size + 1) / x.size
    return pop, cum


def top_share(P: pd.DataFrame, rank_col: str, frac: float,
              value_col: str = "n_waste") -> float:
    """سهم `value_col` که در اختیار بالاترین `frac` افراد بر اساس `rank_col` است."""
    k = max(1, int(round(frac * len(P))))
    top = P.nlargest(k, rank_col)
    return float(top[value_col].sum() / P[value_col].sum())


def jaccard(a: set, b: set) -> float:
    u = len(a | b)
    return len(a & b) / u if u else float("nan")


def header(title: str, level: int = 1) -> None:
    bar = {1: "═", 2: "─", 3: "·"}.get(level, "·")
    print(f"\n{bar * 78}\n{title}\n{bar * 78}", flush=True)


def md_table(df: pd.DataFrame, float_fmt: str = "{:.4f}") -> str:
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "—" if pd.isna(v) else float_fmt.format(v))
    head = "| " + " | ".join(str(c) for c in d.columns) + " |"
    sep = "|" + "|".join(["---"] * len(d.columns)) + "|"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def setup() -> None:
    """استایل شکل و مقصد دور ۹."""
    figs.use_round("round9")
    figs.setup()
