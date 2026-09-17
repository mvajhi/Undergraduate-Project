"""دور ۶-الف — رده‌بندی موجودیت‌های `RestaurantName` و چرخه‌ی عمر سلف‌ها (snapshot v4).

اجرا: ``python -m src.eda_lib.runners.r6a_restaurant_taxonomy``

این دور به دو پرسشی پاسخ می‌دهد که **پیش از هر تجمیع و هر مدلی** باید تعیین تکلیف شوند:

۱. کدام موجودیت‌های ستون `RestaurantName` اصلاً «سلف» نیستند؟
۲. کدام سلف‌ها موقت/تعطیل‌شده‌اند و نباید مدل روی آن‌ها تنظیم شود؟

خروجی: `doc/eda/round6/R6a_restaurant_taxonomy.md` و `reports/figures/round6/`.
"""

from __future__ import annotations

import pandas as pd

from src.config import FIGURES_DIR, set_global_seed
from src.data.load_v4 import load_clean_v4
from src.eda_lib.figio import save_fig
from src.eda_lib.restaurant_taxonomy import (
    CORE_CLASS,
    add_restaurant_class,
    flag_suspect_restaurants,
    lifecycle,
)
from src.eda_lib.runners._common import header, kv, pct

ROUND_FIGURES = FIGURES_DIR / "round6"


def _report_classes(life: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    header("۱. رده‌بندی موجودیت‌ها")
    tbl = life.groupby("restaurant_class").agg(
        n_entities=("rows", "size"),
        rows=("rows", "sum"),
        people=("people", "median"),
        rho_median=("rho", "median"),
        price_median=("price_med", "median"),
        foods_median=("foods", "median"),
    )
    tbl["row_share"] = tbl["rows"] / tbl["rows"].sum()
    print(tbl.round(4).to_string())
    return tbl


def _report_core_lifecycle(life: pd.DataFrame) -> pd.DataFrame:
    header("۲. چرخه‌ی عمر سلف‌های core")
    core = life[life.restaurant_class == CORE_CLASS]
    kv("تعداد سلف core", len(core))
    kv("بازه‌ی نرخ عدم‌دریافت", f"{core.rho.min():.4f} تا {core.rho.max():.4f}")
    kv("نسبت بیشینه به کمینه", f"{core.rho.max() / max(core.rho.min(), 1e-9):.1f}×")
    suspects = flag_suspect_restaurants(life)
    core_suspects = suspects[suspects.restaurant_class == CORE_CLASS]
    kv("سلف core نیازمند تصمیم", f"{len(core_suspects)} از {len(core)}")
    if len(core_suspects):
        print()
        print(core_suspects[["rows", "days", "density", "rho", "first", "last", "dead_days", "flags"]].to_string())
    return core


def _report_mokammel(df: pd.DataFrame) -> None:
    header("۳. رستوران‌های مکمل (بخش خصوصی)")
    mk = df[df.restaurant_class == "mokammel"]
    kv("ردیف", f"{len(mk):,}")
    kv("شعبه", mk.RestaurantName.nunique())
    kv("افراد", f"{mk.PersonId.nunique():,}")
    kv("سهم افراد از کل", pct(mk.PersonId.nunique() / df.PersonId.nunique()))
    kv("بازه", f"{mk.date_gregorian.min().date()} تا {mk.date_gregorian.max().date()}")
    kv("نرخ عدم‌دریافت", pct(mk.dont_receive.mean()))
    print("\nگسترش ماهانه (شعبه‌ی فعال):")
    roll = mk.groupby(mk.date_gregorian.dt.to_period("M")).agg(
        rows=("PersonId", "size"), outlets=("RestaurantName", "nunique"), people=("PersonId", "nunique")
    )
    print(roll.to_string())


def _plot_rho_by_class(life: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    order = [CORE_CLASS, "mokammel", "guest", "howzeh"]
    data = [life.loc[life.restaurant_class == c, "rho"].dropna().values for c in order]
    labels = ["سلف واقعی", "مکمل خصوصی", "سهمیه‌ی مهمان", "دوره‌ی موقت"]
    ax.boxplot(data, tick_labels=labels, showmeans=True)
    ax.set_ylabel("نرخ عدم‌دریافت")
    ax.set_title("نرخ عدم‌دریافت به تفکیک رده‌ی موجودیت — چهار رده، چهار رژیم متفاوت")
    ax.grid(axis="y", alpha=0.3)
    save_fig(fig, "6a_rho_by_class", ROUND_FIGURES)
    plt.close(fig)


def main() -> None:
    set_global_seed()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 55)

    df = add_restaurant_class(load_clean_v4())
    life = lifecycle(df)

    _report_classes(life, df)
    _report_core_lifecycle(life)
    _report_mokammel(df)
    _plot_rho_by_class(life)

    header("خلاصه")
    core_rows = life.loc[life.restaurant_class == CORE_CLASS, "rows"].sum()
    kv("ردیف قابل‌استفاده برای هدف (core)", f"{core_rows:,}")
    kv("ردیف کنارگذاشته‌شده", f"{len(df) - core_rows:,} ({pct(1 - core_rows / len(df))})")


if __name__ == "__main__":
    main()
