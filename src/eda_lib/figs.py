"""زیرساخت شکل دور ۸ — قاعده‌ی ۸-۱-الف: هیچ بسته‌ای بدون شکل بسته نمی‌شود.

هر شکل دو خروجی دارد: تصویر در `reports/figures/round8/` و **داده‌ی پشتش** در
`reports/figures/round8/data/`. دومی اختیاری نیست — شکلی که داده‌اش بازتولیدپذیر
نباشد، ادعای بازبینی‌ناپذیر است.

چرا این ماژول وجود دارد: دور ۷ هیچ شکلی تولید نکرد و هر اجراکننده‌اش اگر می‌خواست
شکل بسازد باید فونت فارسی، اندازه، و مسیر را از نو تعریف می‌کرد. اینجا یک‌بار
تعریف می‌شود.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import viz_fa  # noqa: F401  — با import، فونت و شکل‌دهی فارسی فعال می‌شود
from src.config import FIGURES_DIR

ROUND8_DIR = FIGURES_DIR / "round8"
ROUND8_DATA = ROUND8_DIR / "data"

# پالت امن برای کوررنگی (Okabe–Ito) — ترتیب عمدی: اولی برای «مقدار اصلی»
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#666666"]
C_MAIN, C_ALT, C_OK, C_WARN, C_ACCENT = PALETTE[0], PALETTE[1], PALETTE[2], PALETTE[3], PALETTE[4]
C_GREY = "#8A8A8A"

_STYLE = {
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
}


def setup() -> None:
    """استایل مشترک دور ۸ را اعمال می‌کند. اجراکننده اول این را صدا می‌زند."""
    viz_fa.setup()
    plt.rcParams.update(_STYLE)
    ROUND8_DIR.mkdir(parents=True, exist_ok=True)
    ROUND8_DATA.mkdir(parents=True, exist_ok=True)


def new(nrows: int = 1, ncols: int = 1, w: float = 7.2, h: float = 4.3, **kw):
    """یک شکل با اندازه‌ی پیش‌فرض خوانا در گزارش فارسی.

    `layout="constrained"` پیش‌فرض است چون بدون آن، برچسب محور ردیف بالا زیر
    عنوان ردیف پایین می‌رود — خطایی که در شکل‌های چندقابی همیشه رخ می‌دهد و در
    تصویر نهایی دیده نمی‌شود مگر کسی نگاه کند.
    """
    kw.setdefault("layout", "constrained")
    return plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows), **kw)


def save(fig, name: str, data: pd.DataFrame | None = None, *, caption: str | None = None) -> Path:
    """شکل را ذخیره و داده‌ی پشتش را کنارش می‌نویسد.

    `name` بدون پسوند و بدون پیشوند مسیر — مثلاً `r8a_target_shape`.
    """
    setup_needed = not ROUND8_DIR.exists()
    if setup_needed:
        ROUND8_DIR.mkdir(parents=True, exist_ok=True)
        ROUND8_DATA.mkdir(parents=True, exist_ok=True)
    path = ROUND8_DIR / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    if data is not None:
        data.to_csv(ROUND8_DATA / f"{name}.csv", index=False)
    print(f"  ✔ شکل: {path.relative_to(Path.cwd()) if path.is_absolute() else path}"
          + (f"  ({len(data)} ردیف داده)" if data is not None else ""))
    if caption:
        print(f"    ↳ {caption}")
    return path


# ---------------------------------------------------------------------------
# ابزار آماری مشترک شکل‌ها
# ---------------------------------------------------------------------------

def block_bootstrap_ci(df: pd.DataFrame, value: str, stat, *,
                       day_col: str = "date_gregorian", venue_col: str = "restaurant",
                       n_boot: int = 400, alpha: float = 0.05, seed: int = 42):
    """بازه‌ی اطمینان بوت‌استرپ بلوکی دوبعدی (روز × سلف) — قاعده‌ی ۱۴-۱-ب.

    نمونه‌گیری مستقل سطری بازه را باریک‌تر از واقعیت می‌دهد چون ICC روز ۰.۲۲۵ و
    ICC سلف ۰.۲۰۵ است (F10). اینجا در هر تکرار، **روزها** و **سلف‌ها** جداگانه با
    جایگذاری نمونه‌گیری می‌شوند و تقاطعشان برداشته می‌شود.
    """
    rng = np.random.default_rng(seed)
    days = df[day_col].unique()
    venues = df[venue_col].unique()
    by_day = {d: g for d, g in df.groupby(day_col, observed=True)}
    out = []
    for _ in range(n_boot):
        ds = rng.choice(days, size=len(days), replace=True)
        vs = set(rng.choice(venues, size=len(venues), replace=True))
        parts = [by_day[d] for d in ds if d in by_day]
        if not parts:
            continue
        samp = pd.concat(parts, ignore_index=True)
        samp = samp[samp[venue_col].isin(vs)]
        if len(samp) < 5:
            continue
        try:
            out.append(stat(samp[value]) if value else stat(samp))
        except Exception:
            continue
    if not out:
        return (np.nan, np.nan)
    return tuple(np.nanquantile(out, [alpha / 2, 1 - alpha / 2]))


def weighted_rate(df: pd.DataFrame, num: str = "NoRecv", den: str = "Res") -> float:
    """نرخ وزنی — تنها تعریف معتبر «نرخ یک گروه» در این پروژه.

    میانگین ساده‌ی `rho` روی سلول‌ها، سلول ۳ رزروی را هم‌وزن سلول ۱۹۵۶ رزروی
    می‌کند و عددی می‌دهد که به هیچ تصمیم پختی ترجمه نمی‌شود.
    """
    d = df[den].sum()
    return float(df[num].sum() / d) if d else np.nan


def rtl_barh(ax, labels, values, *, color=C_MAIN, fmt="{:.1%}", pad=0.004):
    """میله‌ی افقی با برچسب فارسی — ترتیب از بالا به پایین، بزرگ‌ترین بالا."""
    y = np.arange(len(labels))
    ax.barh(y, values, color=color, height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    for yi, v in zip(y, values):
        ax.text(v + pad, yi, fmt.format(v), va="center", fontsize=9, color="#333333")
    ax.set_xlim(0, max(values) * 1.18)
    return ax


def annotate(ax, text: str, *, loc: str = "upper right", fontsize: int = 9):
    """یادداشت روی شکل — جای ادعای شکل، طبق قاعده‌ی ۸-۱-ث."""
    xy = {"upper right": (0.98, 0.96, "right", "top"),
          "upper left": (0.02, 0.96, "left", "top"),
          "lower right": (0.98, 0.04, "right", "bottom"),
          "lower left": (0.02, 0.04, "left", "bottom")}[loc]
    ax.text(xy[0], xy[1], text, transform=ax.transAxes, ha=xy[2], va=xy[3],
            fontsize=fontsize, color="#333333",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#CCCCCC", alpha=0.92))
    return ax
