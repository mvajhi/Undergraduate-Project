"""Module for fetching, processing, and saving weather and air quality data for Tehran.

Data sources:
- Open-Meteo Historical Weather API (https://archive-api.open-meteo.com/v1/archive)
- Open-Meteo Air Quality Historical API (https://air-quality-api.open-meteo.com/v1/air-quality)
"""

import logging
from pathlib import Path
import urllib.request
import json
import pandas as pd
import jdatetime

from src.config import DATA_EXTERNAL, set_global_seed

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Coordinates for University of Tehran Central Campus
TEHRAN_LAT = 35.705
TEHRAN_LON = 51.396


def gregorian_to_jalali(gy: int, gm: int, gd: int) -> str:
    """Convert Gregorian date (year, month, day) to Jalali string (YYYY-MM-DD)."""
    jd = jdatetime.date.fromgregorian(year=gy, month=gm, day=gd)
    return f"{jd.year:04d}-{jd.month:02d}-{jd.day:02d}"


def fetch_open_meteo_weather(
    start_date: str = "2023-11-01",
    end_date: str = "2024-06-30",
    lat: float = TEHRAN_LAT,
    lon: float = TEHRAN_LON,
) -> pd.DataFrame:
    """Fetch hourly historical weather from Open-Meteo Archive API."""
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}&"
        f"start_date={start_date}&end_date={end_date}&"
        f"hourly=temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,rain,snowfall,wind_speed_10m&"
        f"timezone=Asia%2FTehran"
    )
    logger.info(f"Fetching weather data from Open-Meteo: {start_date} to {end_date}")
    req = urllib.request.Request(url, headers={"User-Agent": "food-demand-forecasting/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    hourly = data["hourly"]
    df_weather = pd.DataFrame(hourly)
    df_weather["time"] = pd.to_datetime(df_weather["time"])
    return df_weather


def fetch_open_meteo_aqi(
    start_date: str = "2023-11-01",
    end_date: str = "2024-06-30",
    lat: float = TEHRAN_LAT,
    lon: float = TEHRAN_LON,
) -> pd.DataFrame:
    """Fetch hourly historical air quality from Open-Meteo Air Quality API."""
    url = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality?"
        f"latitude={lat}&longitude={lon}&"
        f"start_date={start_date}&end_date={end_date}&"
        f"hourly=pm10,pm2_5,us_aqi&"
        f"timezone=Asia%2FTehran"
    )
    logger.info(f"Fetching AQI data from Open-Meteo: {start_date} to {end_date}")
    req = urllib.request.Request(url, headers={"User-Agent": "food-demand-forecasting/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    hourly = data["hourly"]
    df_aqi = pd.DataFrame(hourly)
    df_aqi["time"] = pd.to_datetime(df_aqi["time"])
    return df_aqi


def process_daily_weather_aqi(
    start_date: str = "2023-11-01",
    end_date: str = "2024-06-30",
    output_path: Path = DATA_EXTERNAL / "weather_aqi_tehran.csv",
) -> pd.DataFrame:
    """Fetch hourly data, aggregate to daily resolution, add Jalali dates, and save CSV."""
    df_weather = fetch_open_meteo_weather(start_date, end_date)
    df_aqi = fetch_open_meteo_aqi(start_date, end_date)

    # Merge on timestamp
    df_merged = pd.merge(df_weather, df_aqi, on="time", how="outer")
    df_merged["date"] = df_merged["time"].dt.strftime("%Y-%m-%d")

    # Daily Aggregations
    daily = df_merged.groupby("date").agg(
        temp_max=("temperature_2m", "max"),
        temp_min=("temperature_2m", "min"),
        temp_mean=("temperature_2m", "mean"),
        feels_like_max=("apparent_temperature", "max"),
        feels_like_min=("apparent_temperature", "min"),
        precipitation_sum=("precipitation", "sum"),
        rain_sum=("rain", "sum"),
        snowfall_sum=("snowfall", "sum"),
        wind_speed_max=("wind_speed_10m", "max"),
        relative_humidity_mean=("relative_humidity_2m", "mean"),
        pm2_5_mean=("pm2_5", "mean"),
        pm2_5_max=("pm2_5", "max"),
        pm10_mean=("pm10", "mean"),
        pm10_max=("pm10", "max"),
        aqi_us_mean=("us_aqi", "mean"),
        aqi_us_max=("us_aqi", "max"),
    ).reset_index()

    # Round numeric columns for cleanliness
    float_cols = daily.select_dtypes(include=["float64"]).columns
    daily[float_cols] = daily[float_cols].round(2)

    # Add Jalali date
    jalali_dates = []
    for d_str in daily["date"]:
        dt = pd.to_datetime(d_str)
        jalali_dates.append(gregorian_to_jalali(dt.year, dt.month, dt.day))
    daily["date_jalali"] = jalali_dates

    # Rename date to date_gregorian for clarity
    daily = daily.rename(columns={"date": "date_gregorian"})

    # Reorder columns
    cols_order = [
        "date_gregorian",
        "date_jalali",
        "temp_max",
        "temp_min",
        "temp_mean",
        "feels_like_max",
        "feels_like_min",
        "precipitation_sum",
        "rain_sum",
        "snowfall_sum",
        "wind_speed_max",
        "relative_humidity_mean",
        "pm2_5_mean",
        "pm2_5_max",
        "pm10_mean",
        "pm10_max",
        "aqi_us_mean",
        "aqi_us_max",
    ]
    daily = daily[cols_order]

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output_path, index=False)
    logger.info(f"Saved weather & AQI dataset ({len(daily)} rows) to {output_path}")

    return daily


def process_daily_weather_aqi_by_city(
    start_date: str = "2023-11-01",
    end_date: str = "2024-06-30",
    output_path: Path = DATA_EXTERNAL / "weather_aqi_by_city.csv",
) -> pd.DataFrame:
    """هواشناسی/AQI روزانه را برای **هر شهرِ حاضر در `campus_geo.csv`** می‌گیرد (اصلاحیه‌ی فاز ۴).

    تا پیش از این، فقط تهران گرفته می‌شد و به همه‌ی سلف‌ها الحاق می‌گشت — که برای پنج
    پردیس خارج از تهران (کرج، قم، پاکدشت، رضوانشهر، فومن) نادرست بود (ردیف ۲۱
    `doc/decision_log.md`). خروجی یک فایل بلند با کلید `(city, date_gregorian)` است.
    """
    from src.data.campus_geo import load_campus_geo

    geo = load_campus_geo()
    cities = geo[["city", "province", "lat", "lon"]].drop_duplicates("city").sort_values("city")

    frames = []
    for row in cities.itertuples(index=False):
        logger.info(f"--- {row.city} ({row.lat}, {row.lon}) ---")
        df_w = fetch_open_meteo_weather(start_date, end_date, lat=row.lat, lon=row.lon)
        df_a = fetch_open_meteo_aqi(start_date, end_date, lat=row.lat, lon=row.lon)
        merged = pd.merge(df_w, df_a, on="time", how="outer")
        merged["date"] = merged["time"].dt.strftime("%Y-%m-%d")
        daily = _aggregate_daily(merged)
        daily.insert(0, "province", row.province)
        daily.insert(0, "city", row.city)
        frames.append(daily)

    out = pd.concat(frames, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    logger.info(f"Saved per-city weather & AQI ({len(out)} rows, {out['city'].nunique()} cities) to {output_path}")
    return out


def _aggregate_daily(df_merged: pd.DataFrame) -> pd.DataFrame:
    """تجمیع ساعتی → روزانه (همان تعریف `process_daily_weather_aqi`، مشترک بین تک‌شهر و چندشهر)."""
    daily = df_merged.groupby("date").agg(
        temp_max=("temperature_2m", "max"),
        temp_min=("temperature_2m", "min"),
        temp_mean=("temperature_2m", "mean"),
        feels_like_max=("apparent_temperature", "max"),
        feels_like_min=("apparent_temperature", "min"),
        precipitation_sum=("precipitation", "sum"),
        rain_sum=("rain", "sum"),
        snowfall_sum=("snowfall", "sum"),
        wind_speed_max=("wind_speed_10m", "max"),
        relative_humidity_mean=("relative_humidity_2m", "mean"),
        pm2_5_mean=("pm2_5", "mean"),
        pm2_5_max=("pm2_5", "max"),
        pm10_mean=("pm10", "mean"),
        pm10_max=("pm10", "max"),
        aqi_us_mean=("us_aqi", "mean"),
        aqi_us_max=("us_aqi", "max"),
    ).reset_index()
    float_cols = daily.select_dtypes(include=["float64"]).columns
    daily[float_cols] = daily[float_cols].round(2)
    daily["date_jalali"] = [
        gregorian_to_jalali(dt.year, dt.month, dt.day) for dt in pd.to_datetime(daily["date"])
    ]
    return daily.rename(columns={"date": "date_gregorian"})


if __name__ == "__main__":
    import sys

    set_global_seed()
    if "--by-city" in sys.argv:
        df_result = process_daily_weather_aqi_by_city()
        print(df_result.groupby("city")[["temp_mean", "precipitation_sum", "aqi_us_mean"]].mean().round(2).to_string())
    else:
        df_result = process_daily_weather_aqi()
        print("Dataset Sample:")
        print(df_result.head())
        print("\nSummary Stats:")
        print(df_result.describe().T[["mean", "min", "max"]])
