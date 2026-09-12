"""دو نمودار باکس «نتایج» پوستر — نسخه‌ی ۲ (`doc/poster/poster-results-box-spec.md`, بند ۴-۵).

نسخه‌ی قبلی این اسکریپت چهار نمودار می‌ساخت (لورنتس، غنی‌سازی، شهر، تقویم). طبق بند ۰ سند
نسخه‌ی ۲، نمودار لورنتس و نمودار شهر حذف شدند و آن دو بلوک در پوستر فقط‌متنی‌اند؛ این‌جا
فقط دو نمودار باقی‌مانده — غنی‌سازی و تقویم دوپنله — بازتولید می‌شوند.

هر دو فایل در **ابعاد نهایی فیزیکی خودشان** (نه ابعاد نمایشگر) رسم می‌شوند: بوم matplotlib
دقیقاً معادل اندازه‌ی چاپی (۱۴۰×۹۰ و ۲۵۶×۷۲ میلی‌متر) است تا هر فونت‌سایز نوشته‌شده در کد،
همان‌قدر پوینت فیزیکی روی پوستر چاپ‌شده باشد. به همین دلیل، مثل نسخه‌ی قبلی، از
``bbox_inches="tight"`` استفاده نمی‌شود — آن گزینه با برش حاشیه ابعاد نهایی (و نسبت واقعی
فونت به بوم) را به‌هم می‌زند؛ حاشیه‌ی سفید هر شکل به‌جایش با ``add_axes`` دستی به‌تقریب صفر
رسانده شده.

داده‌ی هر دو نمودار مستقیماً از خروجی رانرهای EDA گرفته شده (نه گمانه‌زنی):
- غنی‌سازی: ``python -m src.eda_lib.runners.r5a_worst_decile`` (Q37) + فراخوانی مستقیم
  ``enrichment_table(..., "CollegeName")`` برای ردیف «پردیس البرز» که در چاپ استاندارد Q37
  به ۱۲ ردیف بالا بریده می‌شود.
- تقویم (پنل راست): ``src.eda_lib.runners.s10_external.run_holiday_profile`` (باقیمانده‌ی ρ
  با کنترل سلف×وعده×روزهفته، d−۵ تا d−۱).
- تقویم (پنل چپ): ``src.eda_lib.runners.r4_disruption.q32_return_curve`` (شوک وزنی، روزهای
  بازگشت ۱ تا ۵ پس از شکاف سرویس ≥۴ روزه).

طبق تجربه‌ی ثبت‌شده‌ی `make_poster_eval_protocol.py`: فقط ``arabic_reshaper`` کافی است،
افزودن ``python-bidi`` متن را دوباره برمی‌گرداند و خراب می‌کند.

اجرا: ``PYTHONPATH=. .venv/bin/python tools/report_figures/make_poster_results_figures.py``
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import viz_fa  # noqa: F401 -- فونت فارسی، اثر جانبی روی import
from src.config import ROOT_DIR

OUT_DIR = ROOT_DIR / "poster" / "assets"

# کد رنگی پوستر (بند ۵ سند): قرمز بالای مرجع، آبی زیر مرجع، خاکستری بی‌اثر.
RED = "#B7443C"
BLUE = "#3B6E9E"
GRAY = "#9EA3AA"
AXIS = "#D8DADE"
TEXT = "#3A3A3A"

MM = 1 / 25.4

PERSIAN_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")


def fa_num(x, decimals=1, sign=False):
    s = f"{x:+.{decimals}f}" if sign else f"{x:.{decimals}f}"
    return s.replace("-", "−").translate(PERSIAN_DIGITS)


def _style_axis(ax, left=True, bottom=True):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side, on in (("left", left), ("bottom", bottom)):
        ax.spines[side].set_visible(on)
        if on:
            ax.spines[side].set_color(AXIS)
    ax.grid(False)


# ---------------------------------------------------------------------------
# نمودار ۱ — جنگلی، غنی‌شدگی در بدترین دهک (۱۰ ردیف)
# ---------------------------------------------------------------------------

W1, H1 = 140 * MM, 90 * MM

# نقطه/CI از Q37 (doc/eda/round5/R5a_worst_decile.md) و از فراخوانی مستقیم
# enrichment_table برای سه ردیفی که در چاپ استاندارد Q37 نیستند (کارشناسی‌ارشد‌ناپیوسته،
# روزانه، پردیس البرز) — طبق ⚠️ بند ۵ سند، هیچ عددی از روی تصویر یا حدس گرفته نشده.
# همه‌ی ۱۲ ردیف سند (بند ۵، نمودار ۱) با نام واقعی نگه داشته شده‌اند — بدون حذف یا
# ناشناس‌سازی؛ تنها رنگ هر ردیف طبق قاعده‌ی «بازه شامل ۱ ⇐ خاکستری» تنظیم شده.
ENRICHMENT_ROWS = [
    ("حقوق و علوم سیاسی", 1.57, 1.38, 1.78, RED),
    ("شبانه و نوبت دوم", 1.53, 1.40, 1.67, RED),
    ("دکتری عمومی", 1.51, 1.23, 1.84, RED),
    ("غیرساکن خوابگاه", 1.15, 1.11, 1.19, RED),
    ("زن", 1.05, 0.99, 1.10, GRAY),
    ("کارشناسی ارشد ناپیوسته", 1.04, 0.98, 1.10, GRAY),
    ("مرد", 0.96, 0.92, 1.01, GRAY),
    ("کارشناسی پیوسته", 0.95, 0.90, 1.00, GRAY),
    ("روزانه", 0.94, 0.91, 0.98, GRAY),
    ("شهریه‌پرداز", 0.86, 0.67, 1.09, GRAY),
    ("ساکن خوابگاه", 0.60, 0.54, 0.65, BLUE),
    ("پردیس البرز", 0.38, 0.16, 0.86, BLUE),
]


def fig_enrichment():
    fig = plt.figure(figsize=(W1, H1))
    ax = fig.add_axes([0.46, 0.24, 0.50, 0.63])

    n = len(ENRICHMENT_ROWS)
    ys = list(range(n, 0, -1))
    ax.axvline(1.0, color="#6a6a6a", ls="--", lw=1.4, zorder=1)
    ax.text(1.0, n + 0.75, viz_fa.fa("سهم منصفانه"), fontsize=15,
             ha="center", va="bottom", color="#4a4a4a")

    for y, (label, point, lo, hi, color) in zip(ys, ENRICHMENT_ROWS):
        ax.plot([lo, hi], [y, y], color=color, lw=2.5, zorder=2, solid_capstyle="round")
        ax.plot([point], [y], marker="o", ms=9.5, color=color, mec="white", mew=1.0, zorder=3)

    ax.set_ylim(0.25, n + 1.15)
    ax.set_yticks(ys)
    ax.set_yticklabels([viz_fa.fa(r[0]) for r in ENRICHMENT_ROWS], fontsize=14.5)
    ax.set_xlim(0.0, 2.0)
    ax.set_xticks([0.5, 1.0, 1.5])
    ax.set_xticklabels([fa_num(v, 1) for v in [0.5, 1.0, 1.5]], fontsize=15.5)
    ax.set_xlabel(viz_fa.fa("نسبت حضور (۱ = سهم منصفانه)"), fontsize=15, labelpad=6)
    ax.tick_params(color=AXIS, labelcolor=TEXT, length=3)
    ax.tick_params(axis="y", length=0)
    _style_axis(ax, left=False)
    return fig


# ---------------------------------------------------------------------------
# نمودار ۲ — تقویم، دو پنل در یک فایل (راست=قبل از تعطیلی، چپ=بازگشت پس از وقفه)
# ---------------------------------------------------------------------------

W2, H2 = 256 * MM, 72 * MM

# پنل راست — باقیمانده‌ی ρ×۱۰۰ بر حسب «روز تا تعطیلی بعدی» (F19، منبع S10 run_holiday_profile)
HOLIDAY_DAYS = [-5, -4, -3, -2, -1]
HOLIDAY_VAL = [1.73, -0.19, 1.64, 1.09, 3.14]
HOLIDAY_LO = [0.44, -0.84, 0.75, 0.14, 2.10]
HOLIDAY_HI = [3.01, 0.46, 2.54, 2.04, 4.18]

# پنل چپ — شوک وزنی×۱۰۰ در روزهای بازگشت پس از شکاف سرویس ≥۴ روزه (F70، منبع r4_disruption q32_return_curve)
RETURN_DAYS = [1, 2, 3, 4, 5]
RETURN_VAL = [1.53, 2.39, 0.83, 2.01, 0.20]
RETURN_LO = [0.30, 1.74, 0.17, 1.05, -0.83]
RETURN_HI = [3.81, 3.16, 1.57, 3.17, 1.45]

YMIN, YMAX = -1.5, 5.0


def _panel(ax, xs, vals, los, his, index_i, tick_labels, title):
    x = np.arange(len(xs))
    ax.axhline(0, color="#8a8a8a", ls="--", lw=1.2, zorder=1)
    ax.plot(x, vals, "-", color="#a8a8a8", lw=1.3, zorder=2)
    for i in range(len(xs)):
        is_idx = i == index_i
        color = RED if is_idx else GRAY
        ms = 11 if is_idx else 7
        yerr = [[vals[i] - los[i]], [his[i] - vals[i]]]
        ax.errorbar(x[i], vals[i], yerr=yerr, fmt="o", ms=ms, color=color,
                    ecolor=color, elinewidth=1.6, capsize=4, capthick=1.6, zorder=3)
    ax.annotate(fa_num(vals[index_i], 1, sign=True), xy=(x[index_i], his[index_i]),
                xytext=(0, 6), textcoords="offset points", ha="center",
                fontsize=15, fontweight="bold", color=RED)
    ax.set_xlim(-0.65, len(xs) - 0.35)
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=16)
    ax.set_ylim(YMIN, YMAX)
    ax.set_yticks([0, 2, 4])
    ax.set_title(viz_fa.fa(title), fontsize=19, fontweight="bold", pad=10)
    ax.tick_params(color=AXIS, labelcolor=TEXT, length=3)


def fig_calendar():
    fig = plt.figure(figsize=(W2, H2))

    gap = 8 * MM
    panel_w = (W2 - gap) / 2
    frac = panel_w / W2
    gap_frac = gap / W2
    margin = 0.04 * frac
    left_axis_margin = 9 * MM / W2  # جای برچسب‌های عددی محور عمودی، که حالا هر دو پنل دارند

    ax_b = fig.add_axes([left_axis_margin, 0.20, frac - left_axis_margin - margin, 0.62])  # چپ: بعد از وقفه
    ax_a = fig.add_axes([frac + gap_frac + margin, 0.20, frac - 2 * margin, 0.62])          # راست: قبل از تعطیلی

    _panel(ax_a, HOLIDAY_DAYS, HOLIDAY_VAL, HOLIDAY_LO, HOLIDAY_HI, index_i=4,
           tick_labels=[fa_num(d, 0) for d in HOLIDAY_DAYS], title="قبل از تعطیلی")
    ax_a.set_yticklabels([fa_num(v, 0) for v in [0, 2, 4]], fontsize=16)
    ax_a.set_ylabel(viz_fa.fa("واحد درصد"), fontsize=17)
    _style_axis(ax_a)

    _panel(ax_b, RETURN_DAYS, RETURN_VAL, RETURN_LO, RETURN_HI, index_i=1,
           tick_labels=[fa_num(d, 0) for d in RETURN_DAYS], title="بعد از وقفه")
    ax_b.set_yticklabels([fa_num(v, 0) for v in [0, 2, 4]], fontsize=16)
    ax_b.text(1.5, -0.20, viz_fa.fa("روزهای عادی"), fontsize=13,
              color="#6a6a6a", ha="center", va="top")
    _style_axis(ax_b)

    return fig


FIGURES = {
    "poster_res_enrichment": fig_enrichment,
    "poster_res_calendar": fig_calendar,
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, builder in FIGURES.items():
        fig = builder()
        fig.patch.set_facecolor("white")
        for suffix, dpi in ((".pdf", 300), (".png", 300)):
            out_path = OUT_DIR / f"{name}{suffix}"
            fig.savefig(out_path, dpi=dpi, facecolor="white")
            print(out_path)
        plt.close(fig)

    stale = ["poster_res_lorenz", "poster_res_city"]
    for name in stale:
        for suffix in (".pdf", ".png"):
            p = OUT_DIR / f"{name}{suffix}"
            if p.exists():
                p.unlink()
                print(f"removed {p}")


if __name__ == "__main__":
    main()
