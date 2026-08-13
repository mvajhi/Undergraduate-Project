"""توابع کمکی تحلیل سری زمانی (بند ۴.۳ WBS) — استفاده در notebooks/04_03_time_series_structure.ipynb.

این ماژول جدید است (نه ویرایش فایل موجود) طبق دستور پروژه برای جلوگیری از تداخل
با سایر subagent هایی که هم‌زمان روی src/ کار می‌کنند.

نکته‌ی کلیدی درباره‌ی گمشدگی روزها: ~۴۰ از ۱۸۲ روز تقویمی هیچ سرو ندارند (بیشتر
جمعه‌ها، تعطیلات، بلوک نوروز، و بازه‌ی بین‌ترم). برای STL/ACF/PACF که به فاصله‌ی
منظم روزانه (period=7) نیاز دارند، `daily_university_series` این روزها را با
درون‌یابی خطی پر می‌کند و پرچم `is_interpolated` را نگه می‌دارد تا هیچ نموداری
گمشدگی واقعی را پنهان نکند.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def daily_university_series(df: pd.DataFrame) -> pd.DataFrame:
    """سری روزانه‌ی تجمیعی کل دانشگاه: rho_day = Sum(NoRecv) / Sum(Res) در آن روز.

    روی کل بازه‌ی تقویمی (min..max تاریخ سرو) reindex می‌شود، نه فقط روزهای سرو‌شده،
    تا فاصله‌گذاری هفتگی (period=7 در STL/ACF) درست بماند. ستون‌های خروجی:
    - ``rho``: مقدار واقعی (NaN در روزهای بدون سرو)
    - ``is_interpolated``: True یعنی آن روز سرو نداشته و مقدارش درون‌یابی شده
    - ``rho_interp``: ``rho`` با درون‌یابی خطی زمانی برای روزهای NaN (دو-طرفه)
    """
    daily = (
        df.groupby("date_gregorian")
        .agg(Res=("Res", "sum"), Recv=("Recv", "sum"), NoRecv=("NoRecv", "sum"))
        .reset_index()
    )
    daily["date_gregorian"] = pd.to_datetime(daily["date_gregorian"])
    daily = daily.sort_values("date_gregorian").set_index("date_gregorian")

    full_range = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_range)
    daily.index.name = "date_gregorian"

    daily["rho"] = daily["NoRecv"] / daily["Res"]
    daily["is_interpolated"] = daily["rho"].isna()
    daily["rho_interp"] = daily["rho"].interpolate(method="time", limit_direction="both")
    return daily


def group_daily_series(df: pd.DataFrame, restaurant: str, meal: str) -> pd.Series:
    """سری روزانه‌ی rho وزن‌دار برای یک ترکیب (سلف, وعده) — بدون reindex/پرکردن.

    عمداً گمشدگی واقعی را نگه می‌دارد (مثلاً توقف کامل ناهار در رمضان) چون در
    نمودارهای small-multiples دیدن این شکاف‌ها خودش یافته است، نه چیزی که باید پنهان شود.
    """
    sub = df[(df["RestaurantName"] == restaurant) & (df["Meal"] == meal)]
    g = sub.groupby("date_gregorian").agg(Res=("Res", "sum"), NoRecv=("NoRecv", "sum"))
    g.index = pd.to_datetime(g.index)
    g = g.sort_index()
    return (g["NoRecv"] / g["Res"]).rename("rho")


def top_volume_groups(df: pd.DataFrame, n: int = 6) -> list[tuple[str, str]]:
    """پرحجم‌ترین n ترکیب (سلف, وعده) بر اساس مجموع Res — برای small multiples."""
    vol = df.groupby(["RestaurantName", "Meal"])["Res"].sum().sort_values(ascending=False)
    return list(vol.head(n).index)


def seasonal_strength(stl_result) -> float:
    """قدرت فصلی Wang-Smith-Hyndman: 1 - Var(remainder) / Var(seasonal + remainder).

    به بازه‌ی [0, 1] کلیپ می‌شود چون با سری کوتاه/پرنویز نسبت خام می‌تواند کمی
    بیرون از این بازه بیفتد.
    """
    resid = np.asarray(stl_result.resid)
    seasonal = np.asarray(stl_result.seasonal)
    denom = np.nanvar(seasonal + resid)
    if denom == 0:
        return float("nan")
    return float(np.clip(1 - np.nanvar(resid) / denom, 0, 1))


def trend_strength(stl_result) -> float:
    """قدرت روند Wang-Smith-Hyndman: 1 - Var(remainder) / Var(trend + remainder)."""
    resid = np.asarray(stl_result.resid)
    trend = np.asarray(stl_result.trend)
    denom = np.nanvar(trend + resid)
    if denom == 0:
        return float("nan")
    return float(np.clip(1 - np.nanvar(resid) / denom, 0, 1))


def cusum(series: pd.Series) -> pd.Series:
    """CUSUM ساده: مجموع تجمعی انحراف از میانگین سری — برای تشخیص بصری تغییر رژیم."""
    x = series.dropna()
    return (x - x.mean()).cumsum()


def nearby_events(date, events: pd.DataFrame, window_days: int = 3) -> pd.DataFrame:
    """ردیف‌های ``events`` که بازه‌ی [date_start, date_end] آن‌ها در window_days روزی ``date`` باشد.

    برای متقاطع‌کردن نقاط شکست کشف‌شده (بند ۴.۳.۵) با رویدادهای شناخته‌شده استفاده می‌شود.
    """
    date = pd.Timestamp(date)
    start = pd.to_datetime(events["date_start"])
    end = pd.to_datetime(events["date_end"])
    mask = (start - pd.Timedelta(days=window_days) <= date) & (
        end + pd.Timedelta(days=window_days) >= date
    )
    return events[mask]


def nearby_calendar_flags(date, calendar: pd.DataFrame, window_days: int = 3) -> pd.DataFrame:
    """ردیف‌های ``calendar_tehran.csv`` در window_days روزیِ ``date`` با حداقل یک پرچم فعال.

    پرچم‌های بررسی‌شده: تعطیلی، بلوک نوروز، بازه‌ی بین‌ترم، دوره‌ی افزودن‌وحذف،
    میان‌ترم، پایان‌ترم — برای تفسیر نقاط شکست در بند ۴.۳.۵.
    """
    date = pd.Timestamp(date)
    cal = calendar.copy()
    cal["date_gregorian"] = pd.to_datetime(cal["date_gregorian"])
    mask = (cal["date_gregorian"] >= date - pd.Timedelta(days=window_days)) & (
        cal["date_gregorian"] <= date + pd.Timedelta(days=window_days)
    )
    window = cal[mask]
    flag_cols = [
        "is_holiday_any",
        "is_nowruz_block",
        "is_inter_semester_break",
        "is_add_drop_period",
        "is_midterm_period",
        "is_final_exam_period",
    ]
    any_flag = window[flag_cols].any(axis=1)
    return window[any_flag]


def save_fig(fig, name: str, figures_dir: Path | str, dpi: int = 150) -> Path:
    """ذخیره‌ی شکل matplotlib با نام یکتا (پیشوند بند WBS) در reports/figures/."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / name
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    return out_path
