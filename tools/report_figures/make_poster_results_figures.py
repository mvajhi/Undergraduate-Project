"""چهار نمودار باکس «نتایج» پوستر (`doc/poster/poster-results-box-spec.md`, بند ۴).

هر چهار فایل باید دقیقاً ۱۲۳ × ۸۵ میلی‌متر باشند (نمودار ۴ دو پنل ۵۸×۸۵ کنار هم با فاصله‌ی
۷ میلی‌متر) چون در `\\PosterImage` با width و height صریح (نه فقط width) جاگذاری می‌شوند و
چیدمان ردیفی باکس نتایج روی همین ابعاد ثابت حساب شده. به همین دلیل، برخلاف
`make_slides_figs.py`/`figio.save_fig`، اینجا از ``bbox_inches="tight"`` استفاده نمی‌شود —
آن گزینه ابعاد نهایی را با برش حاشیه به‌هم می‌زند؛ به‌جایش حاشیه‌ی داخلی هر شکل با
``subplots_adjust``/``add_axes`` دستی جمع شده تا حاشیه‌ی سفید عملاً صفر و ابعاد دقیق باشد.

طبق تجربه‌ی ثبت‌شده‌ی `make_poster_eval_protocol.py`: فقط ``arabic_reshaper`` کافی است،
افزودن ``python-bidi`` متن را دوباره برمی‌گرداند و خراب می‌کند (matplotlib این محیط خودش
ترتیب راست‌به‌چپ را هنگام رندر انجام می‌دهد).

اجرا: ``PYTHONPATH=. .venv/bin/python tools/report_figures/make_poster_results_figures.py``
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import viz_fa  # noqa: F401 -- فونت فارسی، اثر جانبی روی import
from src.config import ROOT_DIR

OUT_DIR = ROOT_DIR / "poster" / "assets"

RED = "#B7443C"      # هدررفت
AMBER = "#C8862B"    # کمبود
BLUE = "#3B6E9E"      # خروجی مدل / مقادیر مرجع
GRAY = "#B9BCC1"      # خنثی
AXIS = "#D8DADE"      # محورها

MM = 1 / 25.4
W, H = 123 * MM, 85 * MM  # ابعاد نهایی هر چهار فایل

PERSIAN_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")


def fa_num(x, decimals=1):
    s = f"{x:.{decimals}f}" if decimals else str(int(round(x)))
    return s.translate(PERSIAN_DIGITS)


def _style_axis(ax, left=True, bottom=True):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side, on in (("left", left), ("bottom", bottom)):
        ax.spines[side].set_visible(on)
        if on:
            ax.spines[side].set_color(AXIS)
    ax.grid(False)


# ---------------------------------------------------------------------------
# نمودار ۱ — منحنی لورنتس ساده‌شده
# ---------------------------------------------------------------------------

def fig_lorenz():
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0.15, 0.24, 0.83, 0.70])

    x = [i / 100 for i in range(0, 101)]
    p = 2.603  # از رابطه‌ی ۰.۹^p=۰.۷۶ (بدترین ۱۰٪ افراد = ۲۴٪ هدررفت، F73/F74)
    y = [100 * (xi ** p) for xi in x]
    xp = [xi * 100 for xi in x]

    ax.plot([0, 100], [0, 100], ls="--", lw=1.3, color="#9a9a9a", zorder=2)
    ax.fill_between(xp, y, [xi for xi in xp], color=BLUE, alpha=0.12, zorder=1)
    ax.plot(xp, y, lw=3.0, color=BLUE, zorder=3)

    x0, y0 = 90, 100 * (0.9 ** p)  # ≈۷۶
    ax.plot([x0, x0], [0, y0], ls=":", lw=1.4, color="#8a8a8a", zorder=2)
    ax.plot([x0], [y0], marker="o", ms=8, color=RED, mec="white", mew=1.2, zorder=4)
    ax.annotate(viz_fa.fa("۲۴٪"), xy=(x0, y0), xytext=(x0 - 27, y0 - 6),
                fontsize=26, fontweight="bold", color=RED, ha="left", va="center")

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xticks([0, 50, 90, 100])
    ax.set_xticklabels([fa_num(v, 0) for v in [0, 50, 90, 100]], fontsize=15)
    ax.set_yticks([0, 50, 100])
    ax.set_yticklabels([fa_num(v, 0) for v in [0, 50, 100]], fontsize=15)
    ax.set_xlabel(viz_fa.fa("درصد دانشجویان (از کم‌ریسک‌ترین)"), fontsize=18)
    ax.set_ylabel(viz_fa.fa("درصد هدررفت"), fontsize=18)
    ax.tick_params(color=AXIS, labelcolor="#2a2a2a", length=3)
    _style_axis(ax)
    return fig


# ---------------------------------------------------------------------------
# نمودار ۲ — جنگلی چهارردیفه (غنی‌شدگی دهک بدترین)
# ---------------------------------------------------------------------------

def fig_enrichment():
    # نقطه و CI: R5a_worst_decile.md (سکونت، مقطع/کارشناسی‌پیوسته به‌جای «دوره»ی مشابه، دوره)
    rows = [
        ("شبانه و نوبت دوم", 1.53, 1.40, 1.67),
        ("غیرساکن", 1.15, 1.11, 1.19),
        ("روزانه", 0.95, 0.90, 1.00),
        ("ساکن خوابگاه", 0.59, 0.53, 0.64),
    ]

    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0.47, 0.14, 0.49, 0.78])

    ys = list(range(len(rows), 0, -1))
    ax.axvline(1.0, color="#4a4a4a", ls="--", lw=1.3, zorder=1)
    ax.text(1.0, len(rows) + 0.55, viz_fa.fa("سهم منصفانه"), fontsize=14,
             ha="center", va="bottom", color="#4a4a4a")

    for y, (label, point, lo, hi) in zip(ys, rows):
        color = RED if point > 1 else BLUE
        ax.plot([lo, hi], [y, y], color=color, lw=2.2, zorder=2)
        ax.plot([point], [y], marker="o", ms=9, color=color, zorder=3)

    ax.set_ylim(0.4, len(rows) + 0.9)
    ax.set_yticks(ys)
    ax.set_yticklabels([viz_fa.fa(r[0]) for r in rows], fontsize=15)
    ax.set_xlim(0.35, 1.85)
    ax.set_xticks([0.5, 1.0, 1.5])
    ax.set_xticklabels([fa_num(v, 1) for v in [0.5, 1.0, 1.5]], fontsize=15)
    ax.set_xlabel(viz_fa.fa("نسبت حضور در بدترین دهک"), fontsize=18)
    ax.tick_params(color=AXIS, labelcolor="#2a2a2a", length=3)
    _style_axis(ax, left=False)
    return fig


# ---------------------------------------------------------------------------
# نمودار ۳ — میله‌ای افقی شش‌شهره
# ---------------------------------------------------------------------------

def fig_city():
    cities = [
        ("تهران", 8.7, True),
        ("کرج", 7.0, False),
        ("قم", 6.0, False),
        ("پاکدشت", 4.2, False),
        ("فومن", 3.1, False),
        ("رضوانشهر", 3.1, False),
    ]

    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0.30, 0.05, 0.62, 0.92])

    ys = list(range(len(cities), 0, -1))
    for y, (name, val, is_tehran) in zip(ys, cities):
        color = RED if is_tehran else GRAY
        ax.barh(y, val, color=color, height=0.62, zorder=2)
        ax.text(val + 0.25, y, viz_fa.fa(f"{fa_num(val, 1)}٪"), fontsize=18,
                 va="center", ha="left", color="#1a1a1a")

    ax.set_ylim(0.35, len(cities) + 0.65)
    ax.set_yticks(ys)
    ax.set_yticklabels([viz_fa.fa(c[0]) for c in cities], fontsize=18)
    ax.set_xlim(0, 11)
    ax.get_xaxis().set_visible(False)
    ax.spines["bottom"].set_visible(False)
    _style_axis(ax, left=False, bottom=False)
    return fig


# ---------------------------------------------------------------------------
# نمودار ۴ — تقویم، دو پنل کنار هم (راست=قبل از تعطیلی، چپ=بعد از وقفه)
# ---------------------------------------------------------------------------

def fig_calendar():
    fig = plt.figure(figsize=(W, H))

    gap = 7 * MM
    panel_w = (W - gap) / 2
    left_frac = panel_w / W
    gap_frac = gap / W
    margin = 0.04 * left_frac  # حاشیه‌ی کوچک هر پنل تا برچسب/میله به لبه‌ی شکل نچسبد

    ax_b = fig.add_axes([margin, 0.20, left_frac - 2 * margin, 0.64])
    ax_a = fig.add_axes([left_frac + gap_frac + margin, 0.20, left_frac - 2 * margin, 0.64])

    ymax = 4.0

    # پنل ۴الف — قبل از تعطیلی (برچسب دوخطی چون پنل ۵۸mm برای دو عبارت کنار هم تنگ است)
    a_labels = ["روز\nعادی", "روز قبل\nتعطیلی"]
    a_vals = [0.0, 3.1]
    a_colors = [GRAY, RED]
    ax_a.bar([0, 1], a_vals, color=a_colors, width=0.55, zorder=2)
    ax_a.text(1, 3.1 + 0.22, viz_fa.fa(f"+{fa_num(3.1,1)} واحد درصد"), fontsize=13,
               fontweight="bold", color=RED, ha="center", va="bottom")
    ax_a.set_xlim(-0.65, 1.65)
    ax_a.set_xticks([0, 1])
    ax_a.set_xticklabels([viz_fa.fa(t) for t in a_labels], fontsize=12)
    ax_a.set_ylim(0, ymax)
    ax_a.set_yticks([0, 1, 2, 3, 4])
    ax_a.set_yticklabels([fa_num(v, 0) for v in [0, 1, 2, 3, 4]], fontsize=14)
    ax_a.set_ylabel(viz_fa.fa("واحد درصد"), fontsize=15)
    ax_a.set_title(viz_fa.fa("قبل از تعطیلی"), fontsize=16, fontweight="bold", pad=8)
    ax_a.tick_params(color=AXIS, labelcolor="#2a2a2a", length=3)
    _style_axis(ax_a)

    # پنل ۴ب — چهار روز اول بازگشت
    b_vals = [1.5, 2.4, 0.8, 2.0, 0.2]
    b_colors = [GRAY, RED, GRAY, GRAY, GRAY]
    xs = list(range(5))
    ax_b.bar(xs, b_vals, color=b_colors, width=0.6, zorder=2)
    ax_b.axhline(0, color="#9a9a9a", lw=1.1, zorder=1)
    ax_b.text(4.75, 0.12, viz_fa.fa("روزهای عادی"), fontsize=13, color="#6a6a6a",
               ha="right", va="bottom")
    ax_b.set_xlim(-0.65, 4.65)
    ax_b.set_xticks(xs)
    ax_b.set_xticklabels([fa_num(v, 0) for v in [1, 2, 3, 4, 5]], fontsize=14)
    ax_b.set_ylim(0, ymax)
    ax_b.set_yticks([0, 1, 2, 3, 4])
    ax_b.set_yticklabels([])
    ax_b.set_title(viz_fa.fa("بعد از وقفه"), fontsize=16, fontweight="bold", pad=8)
    ax_b.tick_params(color=AXIS, labelcolor="#2a2a2a", length=3)
    _style_axis(ax_b, left=False)

    return fig


FIGURES = {
    "poster_res_lorenz": fig_lorenz,
    "poster_res_enrichment": fig_enrichment,
    "poster_res_city": fig_city,
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


if __name__ == "__main__":
    main()
