"""بارگذاری مشترک داده‌ی فاز ۴ و ابزار چاپ.

همه‌ی اجراکننده‌های فاز ۴ داده را از اینجا می‌گیرند تا نسخه‌ی داده (v2 پس از اصلاحیه‌ی
ردیف‌های ۲۱-۲۳ `decision_log`) در یک نقطه تعریف شود، نه پراکنده در هر اسکریپت.
"""

import numpy as np
import pandas as pd

from src.config import DATA_EXTERNAL, DATA_PROCESSED, FIGURES_DIR, set_global_seed

DATASET_PATH = DATA_PROCESSED / "dataset_v2.csv"
PERSON_DIM_PATH = DATA_PROCESSED / "person_dim_v3.csv"
PERSON_FACT_PATH = DATA_PROCESSED / "person_reservation_fact_v3.csv"
WEATHER_BY_CITY_PATH = DATA_EXTERNAL / "weather_aqi_by_city.csv"
CALENDAR_PATH = DATA_EXTERNAL / "calendar_tehran.csv"
EVENTS_PATH = DATA_EXTERNAL / "events.csv"

DOW_FA = {0: "شنبه", 1: "یکشنبه", 2: "دوشنبه", 3: "سه‌شنبه", 4: "چهارشنبه", 5: "پنجشنبه", 6: "جمعه"}


def load_dataset() -> pd.DataFrame:
    """دیتاست تجمیعی (d,m,r,f) نسخه‌ی ۲ + ستون‌های تقویمی مشتق."""
    df = pd.read_csv(DATASET_PATH, parse_dates=["date_gregorian"])
    df["dow_name"] = df["DayOfWeek"].map(DOW_FA)
    df["jmonth"] = df["DateReserve"].str.slice(5, 7).astype(int)
    df["jyear"] = df["DateReserve"].str.slice(0, 4).astype(int)
    df["ym"] = df["DateReserve"].str.slice(0, 7)
    return df


def load_weather() -> pd.DataFrame:
    """هواشناسی/AQI روزانه به تفکیک شهر — کلید الحاق `(city, date_gregorian)`."""
    w = pd.read_csv(WEATHER_BY_CITY_PATH, parse_dates=["date_gregorian"])
    return w


def load_dataset_with_weather() -> pd.DataFrame:
    """دیتاست + هواشناسی شهر **درست** هر سلف (نه تهران برای همه — ردیف ۲۱ decision_log)."""
    df = load_dataset()
    w = load_weather().drop(columns=["province", "date_jalali"], errors="ignore")
    out = df.merge(w, on=["city", "date_gregorian"], how="left", validate="many_to_one")
    missing = int(out["temp_mean"].isna().sum())
    if missing:
        print(f"[warn] {missing} rows without weather match")
    return out


def load_person_data(fact_cols: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(fact, dim) سطح فرد نسخه‌ی ۲."""
    fact = pd.read_csv(PERSON_FACT_PATH, usecols=fact_cols, parse_dates=["date_gregorian"])
    dim = pd.read_csv(PERSON_DIM_PATH)
    return fact, dim


# ---------------------------------------------------------------------------
# چاپ
# ---------------------------------------------------------------------------

def header(title: str, level: int = 1) -> None:
    bar = "=" if level == 1 else "-"
    print(f"\n{bar * 78}\n{title}\n{bar * 78}")


def kv(label: str, value, width: int = 42) -> None:
    print(f"{label:.<{width}} {value}")


def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def boot_ci(x, stat=np.mean, n_boot: int = 2000, alpha: float = 0.05, seed: int = 42):
    """فاصله‌ی اطمینان bootstrap (percentile) برای یک آماره."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    stats = np.array([stat(x[i]) for i in idx])
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(stat(x)), float(lo), float(hi)


def setup() -> None:
    import warnings

    set_global_seed()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)
    # فونت فارسی (Vazirmatn) حرف یونانی ρ ندارد؛ در عنوان نمودارها به‌جای نماد،
    # واژه‌ی فارسی «نرخ عدم‌دریافت» به کار می‌رود، پس این هشدار فقط نویز است.
    warnings.filterwarnings("ignore", message=".*Glyph.*missing from font.*")
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
