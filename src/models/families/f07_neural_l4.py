"""بند 7.16.1 گروه ۷-ب (عضو ۲۳) — **NeuralProphet** روی L4 (۴۱ سری، سلف×وعده).

⚠️ **بیرون فهرست کوتاه رسمی اسپرینت C است.** `doc/decisions/37-phase7-rescope.md`
بند ۵ برای F07 سقف «حداکثر ۲ مدل» گذاشته بود (MLP جدولی+embedding و یک مدل L5) —
NeuralProphet در آن فهرست نبود. افزودنش درخواست صریح کاربر بود (۲۰۲۶-۰۸-۱۶)، **نه**
بازنگری در تصمیم ۳۷؛ WBS خودش این مدل را «⭐ پل بین خ۳ و خ۷» می‌خواند چون معماری‌اش
(AR-Net + رگرسور آینده + پیشین‌های تعطیلات) دقیقاً بین `f03_timeseries.py` (کلاسیک)
و `f07_neural.py` (جدولی) می‌نشیند.

## چرا L4 و نه L1

بند 7.16 (جدول انتظار) صریح می‌گوید: «L4 (۴۱ سری × ۱۹۷) با آموزش **سراسری** (نه هر
سری جدا) شانس دارد». NeuralProphet این را با پارامتر `trend_global_local="global"` +
`season_global_local="global"` بومی پشتیبانی می‌کند: یک مدل، یک برازش، هر ۴۱ سری با
ستون `ID`. این هم‌راستا با F59/F60 است (شوک روزانه‌ی مشترک + هم‌حرکتی سلف‌ها).

## سطح آموزش در برابر سطح ارزیابی (بند 7.1.2)

مدل روی پنل L4 (میانگین `rho` هر $(d,m,r)$، سری = $(m,r)$) یاد می‌گیرد، ولی ارزیابی
—مثل پل L5 (`f07_neural_l5.py`)— روی همان ردیف‌های **L1** انجام می‌شود که بقیه‌ی
خانواده‌ها با آن سنجیده شدند: `load_l4_bridge()` دقیقاً همان `LevelData` سطح L1 را
برمی‌گرداند و فقط برچسب `level` را به `"L4"` عوض می‌کند — چون منبع داده (و هش
دروازه‌ی انصاف A1) هر دو یکی است، فقط واحد یادگیری فرق دارد.

## پیش‌بینی: چند-گام **کور** (blind)، نه بازخوراندن مقدار واقعی

طول افق آزمون بین foldها فرق می‌کند (تا ~۲۹ روز در fold آخر). به‌جای بازخوراندن
مقدار واقعی $\\rho$ روز قبل به مدل (که هم پیچیده است و هم مرز خوانایی «چه‌کسی چه‌زمانی
می‌داند» را تار می‌کند)، پیش‌بینی دقیقاً مثل SARIMAX/ETS/Theta در `f03_timeseries.py`
**کور و چندگامه** است: `NeuralProphet.predict()` روی `make_future_dataframe(periods=N)`
صدا زده می‌شود و مدل خودش (با `n_lags>0`) به‌صورت بازگشتی از پیش‌بینی‌های خودش برای
گام‌های بعدی استفاده می‌کند — نه از $\\rho$ واقعی آینده.

## رگرسور آینده: از تقویم خام، نه از `features_A_v1.parquet`

`features_A_v1.parquet` فقط روزهای **سرویس‌داده‌شده** را دارد (بدون جمعه‌های تعطیل و
شکاف ۲۹روزه‌ی رمضان)، ولی پنل L4 با فرکانس روزانه‌ی **کامل** کار می‌کند (مثل L3،
`l4_series.py`). پس رگرسورهای آینده از `data/external/calendar_tehran.csv` خوانده
می‌شوند (پوشش کامل ۲۴۳ روز، بدون NaN) — دقیقاً همان دلیل و همان منبعی که
`f03_timeseries.py::CALENDAR_EXOG` استفاده می‌کند.

## ⚠️ هشدار صداقت — این ماژول محلی قابل‌اجرا نیست

نه `torch` و نه `neuralprophet` روی محیط CPU محلی نصب‌اند (بند ۷.۸ — این پروژه
عمداً محلی CPU-فقط است). **اولین آزمون واقعی این کد سلول R0 نوت‌بوک GPU05 است** —
دقیقاً همان کاری که S0/R0 برای کل فاز ۷ طراحی شده تا انجام دهد (یافته‌های ۰ و ۴:
هر باگ واقعی خ۱ همین‌جا کشف شد، نه در تحلیل کدِ بدون اجرا). نام‌گذاری ستون کوانتایل
خروجی `NeuralProphet` (`f"yhat1 {q*100}%"`) با یک تطبیق فازی (`_pick_quantile_column`)
خوانده می‌شود، نه رشته‌ی هاردکد، دقیقاً برای مقاومت در برابر تفاوت نسخه.
"""

import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from src.cv import DATE_COL
from src.models.axes import TAU_GRID
from src.models.gpu_runner import FamilyFitter, models_from_fitters
from src.models.registry import ModelSpec, register
from src.models.spaces import register_space

FAMILY = "F07"
LEVEL = "L4"
FEATURE_SET = "FS_F07_l4_v1"
SEP = "__"

#: همان `f03_timeseries.py::CALENDAR_EXOG` — پوشش کامل ۲۴۳ روز، بدون NaN، مستقل از
#: سرویس‌دهی (بند بالای ماژول: «چرا L4 و نه L1»)
CALENDAR_REGRESSORS = ["is_holiday_any", "is_day_before_holiday", "is_exam_period",
                       "is_final_exam_period", "is_nowruz_block"]

#: τهایی که هم‌زمان تخمین زده می‌شوند — NeuralProphet خودش میانه (۰.۵) را اضافه می‌کند
TRAIN_TAUS: tuple[float, ...] = TAU_GRID


@lru_cache(maxsize=1)
def _calendar_frame() -> pd.DataFrame:
    from src.eda_lib.runners._common import CALENDAR_PATH

    cal = pd.read_csv(CALENDAR_PATH, usecols=[DATE_COL, *CALENDAR_REGRESSORS])
    cal[DATE_COL] = pd.to_datetime(cal[DATE_COL])
    for c in CALENDAR_REGRESSORS:
        cal[c] = cal[c].astype(float)
    return cal


def load_l4_bridge():
    """`LevelData` **همان L1** با برچسب سطح `"L4"` — بند بالای ماژول «سطح آموزش در
    برابر سطح ارزیابی». منبع داده و هر دو هش دروازه‌ی انصاف با L1 یکسان می‌مانند."""
    from dataclasses import replace

    from src.models.gpu_runner import load_l1

    return replace(load_l1(), level=LEVEL)


def _series_id(restaurant: pd.Series, meal: pd.Series) -> pd.Series:
    return restaurant.astype(str) + SEP + meal.astype(str)


def _long_panel(train_l1: pd.DataFrame) -> pd.DataFrame:
    """(ds, ID, y) + رگرسورهای تقویمی — یک ردیف به‌ازای هر $(d,m,r)$ در پنجره‌ی train."""
    g = (train_l1.groupby([DATE_COL, "RestaurantName", "Meal"], observed=True)["rho"]
        .mean().reset_index())
    g["ID"] = _series_id(g["RestaurantName"], g["Meal"])
    g = g.rename(columns={DATE_COL: "ds", "rho": "y"})
    g = g.merge(_calendar_frame(), left_on="ds", right_on=DATE_COL, how="left")
    for c in CALENDAR_REGRESSORS:
        g[c] = g[c].fillna(0.0)
    return g[["ds", "ID", "y", *CALENDAR_REGRESSORS]]


def _future_regressor_frame(ids: list[str], test_dates: np.ndarray) -> pd.DataFrame:
    """(ds, ID, رگرسورها) برای هر (سری × تاریخ آزمون) — بدون `y`، چون پیش‌بینی کور است."""
    dates = pd.to_datetime(pd.unique(test_dates))
    cal = _calendar_frame().rename(columns={DATE_COL: "ds"})
    grid = pd.MultiIndex.from_product([dates, ids], names=["ds", "ID"]).to_frame(index=False)
    grid = grid.merge(cal, on="ds", how="left")
    for c in CALENDAR_REGRESSORS:
        grid[c] = grid[c].fillna(0.0)
    return grid


def _pick_quantile_column(columns: list[str], tau: float) -> str:
    """ستون کوانتایل خروجی NeuralProphet را با تطبیق فازی پیدا می‌کند (نه رشته‌ی
    هاردکد) — بند «هشدار صداقت» بالای ماژول: مقاوم در برابر تفاوت جزئی نسخه."""
    best, best_gap = None, float("inf")
    for c in columns:
        m = re.match(r"yhat1\s+([\d.]+)%$", c)
        if not m:
            continue
        gap = abs(float(m.group(1)) - tau * 100.0)
        if gap < best_gap:
            best, best_gap = c, gap
    return best or "yhat1"


class NeuralProphetModel:
    def __init__(self, net, train_panel: pd.DataFrame, series_ids: list[str],
                 quantiles: list[float], config: dict):
        self.net = net
        self.train_panel = train_panel
        self.series_ids = series_ids
        self.quantiles = quantiles
        self.config = config

    def forecast_series(self, test_dates: np.ndarray) -> pd.DataFrame:
        n_periods = len(pd.unique(test_dates))
        future_reg = _future_regressor_frame(self.series_ids, test_dates)
        future = self.net.make_future_dataframe(
            self.train_panel, periods=n_periods, n_historic_predictions=0,
            regressors_df=future_reg)
        return self.net.predict(future)


def fit_l4(train: pd.DataFrame, tau: float, *, seed: int = 42, **hp) -> NeuralProphetModel:
    from neuralprophet import NeuralProphet, set_log_level

    set_log_level("ERROR")
    panel = _long_panel(train)
    series_ids = sorted(pd.unique(panel["ID"]))
    quantiles = sorted(set(float(q) for q in TRAIN_TAUS) | {tau})

    n_lags = int(hp.get("n_lags", 7))
    n_hidden = int(hp.get("num_hidden_layers", 0))
    net = NeuralProphet(
        n_forecasts=1, n_lags=n_lags,
        num_hidden_layers=n_hidden,
        d_hidden=int(hp.get("d_hidden", 16)) if n_hidden > 0 else None,
        ar_reg=float(hp.get("ar_reg", 0.0)),
        seasonality_reg=float(hp.get("seasonality_reg", 0.0)),
        learning_rate=hp.get("learning_rate"),
        epochs=int(hp.get("epochs", 60)),
        batch_size=int(hp.get("batch_size", 64)),
        weekly_seasonality=True, yearly_seasonality=False, daily_seasonality=False,
        trend_global_local="global", season_global_local="global",
        quantiles=quantiles, normalize="soft",
    )
    for c in CALENDAR_REGRESSORS:
        net.add_future_regressor(c)
    net.fit(panel, freq="D", progress="none")

    config = {"n_lags": n_lags, "num_hidden_layers": n_hidden,
             "d_hidden": int(hp.get("d_hidden", 16)) if n_hidden > 0 else None,
             "ar_reg": float(hp.get("ar_reg", 0.0)),
             "seasonality_reg": float(hp.get("seasonality_reg", 0.0)),
             "epochs": int(hp.get("epochs", 60)), "batch_size": int(hp.get("batch_size", 64)),
             "quantiles": quantiles, "n_series": len(series_ids), "n_train_rows": len(panel)}
    return NeuralProphetModel(net, panel, series_ids, quantiles, config)


def predict_l4(model: NeuralProphetModel, test: pd.DataFrame, tau: float) -> np.ndarray:
    test_dates = pd.to_datetime(test[DATE_COL]).to_numpy()
    forecast = model.forecast_series(test_dates)
    q_col = _pick_quantile_column(list(forecast.columns), tau)

    fc = forecast[["ds", "ID", q_col]].rename(columns={q_col: "rho_q"})
    fc[["RestaurantName", "Meal"]] = fc["ID"].str.split(SEP, n=1, expand=True)
    fc = fc.drop(columns="ID").rename(columns={"ds": DATE_COL})

    left = test[[DATE_COL, "RestaurantName", "Meal"]].copy()
    left["RestaurantName"] = left["RestaurantName"].astype(str)
    left["Meal"] = left["Meal"].astype(str)
    merged = left.merge(fc, on=[DATE_COL, "RestaurantName", "Meal"], how="left")

    out = merged["rho_q"].to_numpy(dtype=float)
    fallback = float(np.nanmedian(out)) if np.isfinite(out).any() else 0.0
    return np.clip(np.where(np.isfinite(out), out, fallback), 0.0, 1.0)


def _save(model: NeuralProphetModel, stem) -> list:
    import json

    from neuralprophet import save as np_save

    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    net_path = Path(f"{stem}.np")
    np_save(model.net, str(net_path))
    meta = Path(f"{stem}.json")
    meta.write_text(json.dumps({
        "config": model.config, "series_ids": model.series_ids, "quantiles": model.quantiles,
    }, ensure_ascii=False, indent=2) + "\n")
    panel_path = Path(f"{stem}__panel.parquet")
    model.train_panel.to_parquet(panel_path)
    return [net_path, meta, panel_path]


def _load(stem) -> NeuralProphetModel:
    import json

    from neuralprophet import load as np_load

    stem = Path(stem)
    net = np_load(str(Path(f"{stem}.np")))
    meta = json.loads(Path(f"{stem}.json").read_text())
    panel = pd.read_parquet(Path(f"{stem}__panel.parquet"))
    return NeuralProphetModel(net, panel, meta["series_ids"], meta["quantiles"], meta["config"])


FITTERS: dict[str, FamilyFitter] = {
    "neuralprophet_l4": FamilyFitter(
        "neuralprophet_l4", fit=fit_l4, predict=predict_l4, save=_save, load=_load,
        defaults={"n_lags": 7, "num_hidden_layers": 0, "d_hidden": 16, "ar_reg": 0.0,
                  "seasonality_reg": 0.0, "epochs": 60, "batch_size": 64}),
}

MODELS = models_from_fitters(FITTERS)

register(ModelSpec(model_id="neuralprophet_l4", family=FAMILY, levels=(LEVEL,),
                   quantile_route="Q1",
                   algorithm="neuralprophet.NeuralProphet(global trend+season, AR-Net)"))


@register_space("neuralprophet_l4", version=1, n_hyperparams=8)
def _space_neuralprophet(trial: optuna.Trial) -> dict:
    n_hidden = trial.suggest_int("num_hidden_layers", 0, 4)
    return {
        # ⭐ بند 7.16.2 ردیف NeuralProphet: n_lags ۰…۲۸ — باید ۱۴/۲۸ را بپوشاند (F33)
        "n_lags": trial.suggest_int("n_lags", 0, 28),
        "num_hidden_layers": n_hidden,
        "d_hidden": trial.suggest_int("d_hidden", 8, 64, log=True) if n_hidden > 0 else 16,
        "ar_reg": trial.suggest_float("ar_reg", 0.0, 10.0),
        "seasonality_reg": trial.suggest_float("seasonality_reg", 0.0, 10.0),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
        "epochs": trial.suggest_int("epochs", 20, 150),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
    }


QUANTREG_MODEL_IDS: frozenset[str] = frozenset()
TUNING_EXCLUDED: frozenset[str] = frozenset()
