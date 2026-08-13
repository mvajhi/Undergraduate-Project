"""Helper functions for WBS 4.10 (dedicated analysis of external variables).

Covers: loading + merging dataset_v1 with weather/calendar external tables,
controlling for day-of-week + restaurant (group-demeaning and OLS with
dummies), binscatter + LOWESS plotting, and the non-parametric group-
comparison tests (Mann-Whitney, Kruskal-Wallis + Dunn) used throughout
notebooks/04_07_external_variables.ipynb.

Read-only with respect to data/: nothing here writes to data/raw,
data/processed, or data/external. `is_ramadan` is computed on the fly from
date_gregorian (jdatetime) rather than persisted back into calendar_tehran.csv,
since that file is part of the read-only external-data set for this phase.
"""

from __future__ import annotations

import datetime
from typing import Iterable, Sequence

import jdatetime
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess

from src.config import DATA_EXTERNAL, DATA_PROCESSED

# Ramadan 1445 AH == 22 Esfand 1402 .. 21 Farvardin 1403 (Jalali), per the task brief.
_RAMADAN_START_JALALI = jdatetime.date(1402, 12, 22)
_RAMADAN_END_JALALI = jdatetime.date(1403, 1, 21)

CONTROL_COLS = ["DayOfWeek", "RestaurantName"]


def _to_jalali(date_gregorian: str) -> jdatetime.date:
    y, m, d = (int(p) for p in date_gregorian.split("-"))
    return jdatetime.date.fromgregorian(date=datetime.date(y, m, d))


def load_merged_dataset() -> pd.DataFrame:
    """Load dataset_v1 and left-join weather + calendar external tables on date_gregorian.

    Also derives a few analysis-only columns that do not exist upstream:
    - ``precip_cat``: 'snow' if snowfall_sum > 0, else 'rain' if rain_sum > 0, else 'dry'.
    - ``is_ramadan``: bool, True for rows whose date falls in Ramadan 1445 AH
      (1402-12-22 .. 1403-01-21 Jalali). Computed here, not read from any file.

    Read-only: does not touch data/raw, data/processed, or data/external.
    """
    df = pd.read_csv(DATA_PROCESSED / "dataset_v1.csv")
    weather = pd.read_csv(DATA_EXTERNAL / "weather_aqi_tehran.csv")
    calendar = pd.read_csv(DATA_EXTERNAL / "calendar_tehran.csv")

    merged = df.merge(weather, on="date_gregorian", how="left", suffixes=("", "_weather"))
    merged = merged.merge(calendar, on="date_gregorian", how="left", suffixes=("", "_cal"))

    merged["precip_cat"] = np.select(
        [merged["snowfall_sum"] > 0, merged["rain_sum"] > 0],
        ["snow", "rain"],
        default="dry",
    )
    merged["is_ramadan"] = merged["date_gregorian"].apply(
        lambda s: _RAMADAN_START_JALALI <= _to_jalali(s) <= _RAMADAN_END_JALALI
    )
    return merged


def add_group_residual(
    df: pd.DataFrame,
    target: str = "rho",
    group_cols: Sequence[str] = CONTROL_COLS,
    out_col: str = "rho_resid",
) -> pd.DataFrame:
    """Return a copy of df with a group-demeaned residual column.

    residual_i = target_i - mean(target | group_cols of row i)

    This is the "demeaning" control prescribed by WBS 4.10: it removes the
    restaurant-level and day-of-week-level baseline so that any remaining
    association between the residual and an external variable (temperature,
    AQI, ...) cannot be a spurious artifact of restaurant mix or weekday mix.
    """
    out = df.copy()
    group_mean = out.groupby(list(group_cols))[target].transform("mean")
    out[out_col] = out[target] - group_mean
    return out


def ols_quadratic_control(
    df: pd.DataFrame,
    var: str,
    degree: int = 2,
    target: str = "rho",
    control_cols: Sequence[str] = CONTROL_COLS,
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Fit target ~ C(control_cols) + var + var^2 [+ var^3] via OLS.

    Used to test whether an external variable's relationship with rho is
    linear or curved (U-shaped) *after* controlling for day-of-week and
    restaurant fixed effects, per WBS 4.10's explicit instruction.
    """
    work = df.copy()
    work = work.dropna(subset=[target, var, *control_cols])
    work["_var_c"] = work[var] - work[var].mean()  # center to reduce collinearity with var^2/var^3

    terms = ["_var_c"]
    for p in range(2, degree + 1):
        col = f"_var_c_p{p}"
        work[col] = work["_var_c"] ** p
        terms.append(col)

    control_terms = " + ".join(f"C(Q('{c}'))" for c in control_cols)
    formula = f"Q('{target}') ~ {control_terms} + " + " + ".join(terms)
    model = smf.ols(formula=formula, data=work).fit()
    return model


def binscatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    n_bins: int = 12,
    weight: str | None = None,
) -> pd.DataFrame:
    """Equal-frequency binscatter table: mean(x), mean(y), sem(y), n per bin.

    If `weight` is given, the bin means for y are weighted by that column
    (e.g. Res, so restaurants with tiny reservation counts do not dominate).
    """
    work = df.dropna(subset=[x, y]).copy()
    work["_bin"] = pd.qcut(work[x], n_bins, duplicates="drop")

    rows = []
    for b, grp in work.groupby("_bin", observed=True):
        if weight is not None:
            w = grp[weight].to_numpy()
            y_mean = np.average(grp[y], weights=w)
        else:
            y_mean = grp[y].mean()
        rows.append(
            {
                "bin": b,
                "x_mean": grp[x].mean(),
                "y_mean": y_mean,
                "y_sem": grp[y].sem(),
                "n": len(grp),
            }
        )
    return pd.DataFrame(rows).sort_values("x_mean").reset_index(drop=True)


def lowess_curve(x: pd.Series, y: pd.Series, frac: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    """Thin wrapper around statsmodels lowess, returns (x_sorted, y_smoothed)."""
    mask = x.notna() & y.notna()
    fit = lowess(y[mask], x[mask], frac=frac, return_sorted=True)
    return fit[:, 0], fit[:, 1]


def mannwhitney_report(
    a: pd.Series,
    b: pd.Series,
    label_a: str = "A",
    label_b: str = "B",
) -> dict:
    """Mann-Whitney U test between two samples + rank-biserial effect size.

    Uses scipy directly (not pingouin) to avoid a hard version-coupling
    dependency; rank-biserial correlation is computed from U the same way
    pingouin does: RBC = 1 - 2U / (n_a * n_b).
    """
    a = pd.Series(a).dropna()
    b = pd.Series(b).dropna()
    u_stat, p_val = stats.mannwhitneyu(a, b, alternative="two-sided")
    n_a, n_b = len(a), len(b)
    rbc = 1 - (2 * u_stat) / (n_a * n_b)
    return {
        "label_a": label_a,
        "label_b": label_b,
        "n_a": n_a,
        "n_b": n_b,
        "median_a": a.median(),
        "median_b": b.median(),
        "U": u_stat,
        "p_value": p_val,
        "rank_biserial": rbc,
    }


def kruskal_report(groups: dict[str, pd.Series]) -> dict:
    """Kruskal-Wallis H test across >=2 groups + epsilon-squared effect size."""
    samples = {k: pd.Series(v).dropna() for k, v in groups.items()}
    h_stat, p_val = stats.kruskal(*samples.values())
    n_total = sum(len(s) for s in samples.values())
    k = len(samples)
    epsilon_sq = (h_stat - k + 1) / (n_total - k) if n_total > k else float("nan")
    return {
        "groups": {k: len(v) for k, v in samples.items()},
        "medians": {k: v.median() for k, v in samples.items()},
        "H": h_stat,
        "p_value": p_val,
        "epsilon_squared": epsilon_sq,
    }


def holiday_offset_table(
    df: pd.DataFrame,
    resid_col: str = "rho_resid",
    offsets: Iterable[int] = (-3, -2, -1, 0, 1, 2),
) -> pd.DataFrame:
    """Mean (+/- 95% CI) of resid_col by day-offset from the nearest holiday block.

    offset < 0  -> days_to_next_holiday == -offset (approaching a holiday)
    offset == 0 -> is_holiday_any == True (the holiday day itself)
    offset > 0  -> days_since_last_holiday == offset (days after a holiday block)

    Expects df to already carry days_to_next_holiday, days_since_last_holiday,
    is_holiday_any (from calendar_tehran.csv) and resid_col (from
    add_group_residual).
    """
    rows = []
    for off in offsets:
        if off < 0:
            mask = df["days_to_next_holiday"] == -off
        elif off == 0:
            mask = df["is_holiday_any"] == True  # noqa: E712
        else:
            mask = df["days_since_last_holiday"] == off
        vals = df.loc[mask, resid_col].dropna()
        if len(vals) == 0:
            mean, sem, ci = np.nan, np.nan, np.nan
        else:
            mean = vals.mean()
            sem = vals.sem()
            ci = 1.96 * sem
        rows.append({"offset": off, "n": len(vals), "mean_resid": mean, "sem": sem, "ci95": ci})
    return pd.DataFrame(rows)
