"""بازتولید دو شکل بند ۳-۳-۳ — توزیع متغیر هدف و شوک مشترک روزانه.

`final_report/img/target_distribution.png` و `final_report/img/day_shock.png`
symlink به خروجی این اسکریپت‌اند؛ منبع حقیقت تصاویر گزارش `reports/figures/` است
(tools/report_figures/README.md).

نسخه‌ی `_fa` تفاوتش با نمودار خام EDA (`report_01_*`, `report_14_*`) فقط حذف عنوان
داخل شکل است: عنوان آن نمودارها یک جمله‌ی خبری کامل بود که در گزارش با کپشن لاتک
تکرار می‌شد.

منبع داده: `data/processed/dataset_v2.csv` (snapshot قفل‌شده‌ی سطح L1) و
`data/external/calendar_tehran.csv`. شوک روزانه دقیقاً طبق رابطه‌ی (۱) گزارش
محاسبه می‌شود: میانگین انحراف هر سلف از نرخ عادی خودش در آن روز-وعده.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import viz_fa  # noqa: F401  — فونت فارسی و patch متن
from src.viz_fa import fa

DATA = Path("data/processed/dataset_v2.csv")
CALENDAR = Path("data/external/calendar_tehran.csv")
PERSON_FEATURES = Path("data/processed/person_features_v1.parquet")
OUT_DIR = Path("reports/figures")

BLUE, RED, ORANGE, PURPLE = "#4C72B0", "#C44E52", "#DD8452", "#8172B2"


def fig_target(df: pd.DataFrame) -> Path:
    """توزیع نرخ عدم‌دریافت: هیستوگرام + تابع توزیع تجمعی."""
    rho = df["rho"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9))

    axes[0].hist(rho, bins=90, color=BLUE, edgecolor="white")
    axes[0].axvline(rho.median(), color=RED, ls="--",
                    label=fa(f"میانه {rho.median():.3f}"))
    axes[0].set_xlabel(fa("نرخ عدم‌دریافت"))
    axes[0].set_ylabel(fa("تعداد رکورد"))
    axes[0].legend()

    xs = np.sort(rho.values)
    zero_share = float((rho == 0).mean())
    axes[1].plot(xs, np.arange(1, len(xs) + 1) / len(xs), color=RED, lw=2)
    axes[1].axhline(zero_share, color="gray", ls=":",
                    label=fa(f"{zero_share:.1%} رکوردها نرخ صفر دارند"))
    axes[1].set_xlabel(fa("نرخ عدم‌دریافت"))
    axes[1].set_ylabel(fa("نسبت تجمعی"))
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out = OUT_DIR / "report_01_target_distribution_fa.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out} — median={rho.median():.3f}, zeros={zero_share:.2%}")
    return out


def build_day_shock(df: pd.DataFrame) -> pd.DataFrame:
    """رابطه‌ی (۱): میانگین انحراف سلف‌ها از نرخ عادی خودشان در هر روز-وعده."""
    cell = (df.groupby(["date_gregorian", "Meal", "RestaurantName"], observed=True)
              .agg(Res=("Res", "sum"), NoRecv=("NoRecv", "sum")).reset_index())
    cell["rate"] = cell["NoRecv"] / cell["Res"]
    cell["dev"] = cell["rate"] - cell.groupby(
        ["RestaurantName", "Meal"], observed=True)["rate"].transform("mean")
    day = (cell.groupby(["date_gregorian", "Meal"], observed=True)
               .agg(shock=("dev", "mean"), n_cells=("dev", "size")).reset_index())
    return day[day["n_cells"] >= 3].copy()


JALALI_MONTHS = {9: "آذر", 10: "دی", 11: "بهمن", 12: "اسفند",
                 1: "فروردین", 2: "اردیبهشت", 3: "خرداد"}


def fig_day_shock(day: pd.DataFrame) -> Path:
    cal = pd.read_csv(CALENDAR, parse_dates=["date_gregorian"])
    day = day.merge(cal[["date_gregorian", "is_day_before_holiday",
                         "month_jalali", "day_of_month_jalali"]],
                    on="date_gregorian", how="left")

    fig, ax = plt.subplots(figsize=(11.5, 3.9))
    for meal, color, label in [("lunch", ORANGE, "ناهار"), ("dinner", PURPLE, "شام")]:
        s = day[day["Meal"] == meal].sort_values("date_gregorian").copy()
        # شکاف بیش از سه روز (نوروز، رمضان) با خط وصل نشود
        gap = s["date_gregorian"].diff().dt.days > 3
        s.loc[gap, "shock"] = np.nan
        ax.plot(s["date_gregorian"], s["shock"] * 100, lw=1.4, color=color,
                label=fa(label), alpha=0.85)
    ax.axhline(0, color="k", lw=0.9, ls="--")
    pre = day[day["is_day_before_holiday"].fillna(False)]
    ax.scatter(pre["date_gregorian"], pre["shock"] * 100, s=22, color=RED,
               zorder=5, label=fa("روز پیش از تعطیلی"))

    firsts = (day.sort_values("date_gregorian")
                 .drop_duplicates("month_jalali", keep="first"))
    ax.set_xticks(firsts["date_gregorian"])
    ax.set_xticklabels([fa(JALALI_MONTHS[int(m)]) for m in firsts["month_jalali"]])
    ax.set_ylabel(fa("شوک روزانه (واحد درصد)"))
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out = OUT_DIR / "report_14_day_shock_fa.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out} — sd={day['shock'].std():.4f}, n={len(day)}")
    return out


# ---------------------------------------------------------------------------
# نسخه‌ی گزارش‌آماده‌ی نمودارهای کاوش داده
#
# این نمودارها در فاز ۴ ساخته شده‌اند و اعدادشان بازتولید نمی‌شود؛ تنها کاری که
# اینجا انجام می‌شود حذف نوار عنوان بالای شکل است، چون عنوان آن نمودارها یک
# جمله‌ی خبری کامل بود که در گزارش با کپشن لاتک تکرار می‌شد. عنوان هر پنل (که
# برچسب است نه تکرار کپشن) دست‌نخورده می‌ماند.
# ---------------------------------------------------------------------------

# نمودارهایی که فقط نوار عنوانشان برداشته می‌شود
TITLE_STRIP = [
    "report_02_city_effect",
    "report_03_aqi_spurious",
    "report_07_pre_holiday",
    "report_08_acf_by_meal",
    "report_09_daily_series_volume",
    "report_10_dorm_resident",
    "round4/r4_2_return_curve",
]


def strip_title(name: str) -> Path:
    """نوار عنوان بالای شکل را می‌برد و عنوان پنل‌ها را نگه می‌دارد."""
    from PIL import Image

    src = OUT_DIR / f"{name}.png"
    img = Image.open(src).convert("RGB")
    ink = np.asarray(img).min(axis=2) < 245
    rows = np.flatnonzero(ink.any(axis=1))
    bands = np.split(rows, np.flatnonzero(np.diff(rows) > 2) + 1)

    if len(bands[0]) > 60:
        # عنوان به بدنه چسبیده: نخستین سطر «پهن» همان چارچوب محور است
        wide = np.flatnonzero(ink.sum(axis=1) > 0.6 * img.width)
        top = int(wide[wide > bands[0][0] + 15][0])
    else:
        top = (int(bands[0][-1]) + int(bands[1][0])) // 2

    out = OUT_DIR / f"{name}_fa.png"
    img.crop((0, max(top, 0), img.width, img.height)).save(out)
    print(f"saved {out} — از سطر {max(top, 0)} به بعد")
    return out


def fig_lorenz() -> Path:
    """منحنی لورنتس تمرکز عدم‌دریافت در افراد (بازسازی‌شده: نسخه‌ی خام نویسه‌ی گمشده دارد)."""
    f = pd.read_parquet(PERSON_FEATURES, columns=["PersonId", "dont_receive"])
    per = f.groupby("PersonId")["dont_receive"].sum().sort_values().values
    cum = np.concatenate([[0.0], np.cumsum(per) / per.sum()]) * 100
    frac = np.linspace(0, 100, len(cum))
    gini = 1 - 2 * np.trapezoid(cum / 100, frac / 100)

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.plot([0, 100], [0, 100], "k--", lw=1.2, label=fa("توزیع کاملاً برابر"))
    ax.plot(frac, cum, color=RED, lw=2.2)
    ax.fill_between(frac, cum, frac, color=RED, alpha=0.12)
    for q in (80, 90):
        share = np.interp(q, frac, cum)
        ax.axvline(q, color="gray", ls=":", lw=0.9)
        ax.annotate(fa(f"{100 - q}٪ بدترین‌ها: {100 - share:.0f}٪ کل عدم‌دریافت"),
                    xy=(q, share), xytext=(q - 46, share + 12), fontsize=9,
                    arrowprops=dict(arrowstyle="->", color="gray", lw=0.9))
    ax.set_xlabel(fa("درصد تجمعی دانشجویان (از کم‌مصرف‌ترین)"))
    ax.set_ylabel(fa("درصد تجمعی موارد عدم‌دریافت"))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = OUT_DIR / "report_12_lorenz_fa.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out} — gini={gini:.3f}")
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA, parse_dates=["date_gregorian"])
    fig_target(df)
    fig_day_shock(build_day_shock(df))
    fig_lorenz()
    for name in TITLE_STRIP:
        strip_title(name)


if __name__ == "__main__":
    main()
