"""نمودارهای تازه‌ی ارائه به استاد راهنما (`doc/peresentation.md`).

۱۴ شکلی که آن سند «باید ساخته شود» علامت زده — همه از داده‌ی موجود مخزن (نه عدد
دستی) یا از فرمول صریح مسئله می‌آیند و همه در `reports/figures/slides/` می‌نشینند.
سه شکلی که سند می‌گفت نیاز به نسخه‌ی `_fa` دارند، بررسی و رد شدند: `4.8_variance_vs_res.png`
(جایگزین آماده‌ی report_04)، `round5/R5_q43_volume_vs_rate.png` و
`round5/R5_q43_three_rankings.png` از قبل برچسب فارسی کامل دارند — نیازی به کار اضافه نبود.

اجرا: ``PYTHONPATH=. .venv/bin/python tools/report_figures/make_slides_figs.py``
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle, Wedge

from src import viz_fa  # noqa: F401 -- فونت فارسی، اثر جانبی روی import
from src.config import FIGURES_DIR
from src.eda_lib.figio import save_fig

OUT_DIR = FIGURES_DIR / "slides"

RED = "#b8443c"      # هدررفت / کمبود / هزینه
BLUE = "#3b6ea5"      # صرفه‌جویی / نجات‌یافته
GREEN = "#2e8b57"     # نقطه‌ی عملیاتی / برجسته
GRAY = "#bdbdbd"
GRAY_DARK = "#7a7a7a"


def _finish(fig, name: str) -> None:
    print(save_fig(fig, name, OUT_DIR, dpi=200))
    plt.close(fig)


# ---------------------------------------------------------------------------
# اسلاید ۱ — وافل ۸ از ۱۰۰
# ---------------------------------------------------------------------------

def fig_01_waffle() -> None:
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    n_red = 8
    order = list(range(100))
    for i in order:
        row, col = divmod(i, 10)
        color = RED if i < n_red else "#e3e3e3"
        ax.add_patch(Rectangle((col, 9 - row), 0.86, 0.86, facecolor=color, edgecolor="white", lw=1.5))
    ax.set_xlim(-0.3, 10.3)
    ax.set_ylim(-0.3, 10.3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(5, -1.1, "۸.۰۵٪ رزروها تحویل گرفته نمی‌شوند", ha="center", va="top",
            fontsize=15, fontweight="bold", color=RED)
    fig.tight_layout()
    _finish(fig, "slides_01_waffle")


# ---------------------------------------------------------------------------
# اسلاید ۳ — منحنی زیان نامتقارن نیوزوندور
# ---------------------------------------------------------------------------

def fig_03_asymmetric_cost() -> None:
    D = 100.0
    Cu, Co = 4.0, 1.0
    Q = np.linspace(0, 1.8 * D, 400)
    cost = np.where(Q < D, Cu * (D - Q), Co * (Q - D))

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.plot(Q, cost, color=GRAY_DARK, lw=2.6)
    ax.axvline(D, color=GREEN, ls=":", lw=1.6)
    ax.annotate("تقاضای واقعی", xy=(D, ax.get_ylim()[1] * 0.02), xytext=(D + 8, cost.max() * 0.18),
                fontsize=10, color=GREEN)
    ax.fill_between(Q, cost, where=Q < D, color=RED, alpha=0.12)
    ax.fill_between(Q, cost, where=Q >= D, color=BLUE, alpha=0.12)
    ax.annotate("۴ برابر شیب‌دارتر", xy=(D * 0.55, Cu * D * 0.45),
                xytext=(D * 0.12, Cu * D * 0.78),
                fontsize=13, fontweight="bold", color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))
    ax.text(D * 0.4, -Cu * D * 0.10, "کمبود", ha="center", color=RED, fontsize=11)
    ax.text(D * 1.4, -Cu * D * 0.10, "مازاد", ha="center", color=BLUE, fontsize=11)
    ax.set_xlabel("مقدار پخت")
    ax.set_ylabel("هزینه (واحد نمادین)")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    _finish(fig, "slides_03_asymmetric_cost")


# ---------------------------------------------------------------------------
# اسلاید ۵ — دو تابع زیان Pinball
# ---------------------------------------------------------------------------

def fig_05_pinball() -> None:
    u = np.linspace(-3, 3, 400)

    def pinball(u, tau):
        return np.where(u >= 0, tau * u, (tau - 1) * u)

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.plot(u, pinball(u, 0.5), color=GRAY_DARK, lw=2.4, label="چندک ۰.۵۰ (متقارن)")
    ax.plot(u, pinball(u, 0.2), color=RED, lw=2.6, label="چندک ۰.۲۰ (کج‌شده)")
    ax.axvline(0, color="0.85", lw=1)
    ax.annotate("۴ برابر شیب‌دارتر", xy=(-1.6, pinball(np.array([-1.6]), 0.2)[0]),
                xytext=(-2.9, 1.9), fontsize=12, fontweight="bold", color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.3))
    ax.set_xlabel("خطا  (واقعی − پیش‌بینی)")
    ax.set_ylabel("زیان")
    ax.legend(fontsize=10, framealpha=0.9)
    ax.set_yticks([])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    _finish(fig, "slides_05_pinball")


# ---------------------------------------------------------------------------
# اسلاید ۶ — نگاشت نسبت هزینه به چندک هدف
# ---------------------------------------------------------------------------

def fig_06_tau_mapping() -> None:
    r = np.linspace(1, 12, 300)
    tau_correct = 1.0 / (r + 1.0)
    tau_wrong = r / (r + 1.0)

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.plot(r, tau_correct, color=BLUE, lw=2.6, label="فرمول درست — چندک نرخ عدم‌دریافت")
    ax.plot(r, tau_wrong, color="0.55", lw=2.0, ls="--", label="فرمول اشتباه — چندک تقاضا")
    r0, tau0 = 4.0, 0.20
    ax.scatter([r0], [tau0], color=GREEN, s=90, zorder=5)
    ax.annotate("۰.۲۰", xy=(r0, tau0), xytext=(r0 + 0.9, tau0 + 0.12),
                fontsize=14, fontweight="bold", color=GREEN,
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.4))
    ax.axvline(r0, color=GREEN, ls=":", lw=1.2, alpha=0.7)
    ax.set_xlabel("نسبت هزینه‌ی کمبود به مازاد")
    ax.set_ylabel("چندک هدف")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10, framealpha=0.9, loc="upper right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    _finish(fig, "slides_06_tau_mapping")


# ---------------------------------------------------------------------------
# اسلاید ۷ — خط زمانی قاعده‌ی برش
# ---------------------------------------------------------------------------

def fig_07_timeline() -> None:
    fig, ax = plt.subplots(figsize=(9.4, 3.6))
    ax.axhline(0, color="0.6", lw=1.6, zorder=1)

    events = [
        (0.0, "ناهار روز قبل", "#8a8a8a", "o"),
        (1.0, "شام روز قبل", RED, "s"),
        (3.2, "ناهار روز هدف", GREEN, "D"),
        (4.2, "شام روز هدف", GREEN, "D"),
    ]
    for x, label, color, marker in events:
        ax.scatter([x], [0], s=210, color=color, marker=marker, zorder=3, edgecolor="white", lw=1.2)
        ax.text(x, 0.34, label, ha="center", fontsize=11, color=color, fontweight="bold")

    cutoffs = [(2.0, "برش ناهار\nساعت پانزده روز قبل"), (3.7, "برش شام\nساعت بیست‌وسه روز قبل")]
    for x, label in cutoffs:
        ax.axvline(x, color="0.25", ls="--", lw=1.4, zorder=2)
        ax.text(x, -0.55, label, ha="center", va="top", fontsize=9.5, color="0.25")

    ax.axvspan(1.55, 1.95, color=RED, alpha=0.10, zorder=0)
    ax.annotate("در لحظه‌ی برش ناهار هنوز سرو نشده، پس ممنوعه",
                xy=(1.0, 0), xytext=(1.0, -1.25),
                ha="center", fontsize=10.5, color=RED, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.3))

    ax.set_xlim(-0.6, 4.9)
    ax.set_ylim(-1.7, 0.9)
    ax.axis("off")
    fig.tight_layout()
    _finish(fig, "slides_07_timeline")


# ---------------------------------------------------------------------------
# اسلاید ۸ — احتمال نگرفتن شام، مشروط به ناهار
# ---------------------------------------------------------------------------

def fig_08_dinner_given_lunch() -> None:
    labels = ["ناهار را گرفت", "ناهار را نگرفت"]
    vals = [5.01, 15.93]
    colors = [BLUE, RED]

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    bars = ax.bar(labels, vals, color=colors, width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.1f}٪", ha="center", fontsize=12)
    ax.annotate("", xy=(1, vals[1] + 1.6), xytext=(0, vals[0] + 1.6),
                arrowprops=dict(arrowstyle="-", color="0.3", lw=1.2,
                                 connectionstyle="arc3,rad=-0.25"))
    ax.text(0.5, max(vals) + 3.3, "۳.۲ برابر", ha="center", fontsize=15, fontweight="bold", color=RED)
    ax.set_ylabel("احتمال نگرفتن شام همان روز (٪)")
    ax.set_ylim(0, max(vals) + 6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    _finish(fig, "slides_08_dinner_given_lunch")


# ---------------------------------------------------------------------------
# اسلاید ۹ (پنل چپ) — دونات تجزیه‌ی واریانس
# ---------------------------------------------------------------------------

def fig_09_variance_donut() -> None:
    vals = [83, 10, 7]
    labels = ["شوک مشترک روز", "ترکیب رزروکنندگان", "نویز نمونه‌گیری"]
    colors = [RED, BLUE, GRAY]

    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    wedges, _ = ax.pie(vals, colors=colors, startangle=90, counterclock=False,
                        wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2))
    ax.text(0, 0.08, "۸۳٪", ha="center", va="center", fontsize=30, fontweight="bold", color=RED)
    ax.text(0, -0.14, "شوک مشترک روز", ha="center", va="center", fontsize=12, color=RED)

    for w, lab, v in zip(wedges, labels, vals):
        if v == 83:
            continue
        ang = np.deg2rad((w.theta1 + w.theta2) / 2)
        x, y = 1.28 * np.cos(ang), 1.28 * np.sin(ang)
        ax.annotate(f"{lab}  {v}٪", xy=(0.92 * np.cos(ang), 0.92 * np.sin(ang)), xytext=(x, y),
                    ha="center", va="center", fontsize=10.5,
                    arrowprops=dict(arrowstyle="-", color="0.4", lw=1))
    ax.set_aspect("equal")
    fig.tight_layout()
    _finish(fig, "slides_09_variance_donut")


# ---------------------------------------------------------------------------
# اسلاید ۱۵ — قیف غربال مدل
# ---------------------------------------------------------------------------

def fig_15_funnel() -> None:
    stages = ["خانواده‌ی مدل", "مدل آموزش‌دیده", "برد آماری معنادار", "قهرمان رسمی"]
    counts = [13, 25, 5, 1]
    colors = [GRAY, "#8fa8c4", BLUE, GREEN]

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    y = np.arange(len(stages))[::-1]
    max_c = max(counts)
    for yi, c, lab, col in zip(y, counts, stages, colors):
        w = 0.85 * c / max_c + 0.10
        ax.add_patch(FancyBboxPatch((-w / 2, yi - 0.30), w, 0.60,
                                     boxstyle="round,pad=0.02,rounding_size=0.04",
                                     facecolor=col, edgecolor="white"))
        ax.text(-0.62, yi, lab, ha="right", va="center", fontsize=11.5, color="0.2")
        ax.text(w / 2 + 0.06, yi, str(c), ha="left", va="center", fontsize=14,
                fontweight="bold", color=col)
    ax.set_xlim(-1.7, 1.15)
    ax.set_ylim(-0.7, max(y) + 0.7)
    ax.axis("off")
    fig.tight_layout()
    _finish(fig, "slides_15_funnel")


# ---------------------------------------------------------------------------
# اسلاید ۱۷ — نردبان عددی (روز / ترم / سال)
# ---------------------------------------------------------------------------

def fig_17_ladder() -> None:
    panels = [
        ("یک روز سرو", 1037, 737, 300, "۲۷۷ میلیون تومان"),
        ("یک ترم (۱۱۲ روز)", 116142, 82577, 33565, "۳۱.۰ میلیارد تومان"),
        ("یک سال تحصیلی (۲۲۴ روز)", 232285, 165155, 67130, "۶۱.۹ میلیارد تومان"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 5.4))
    for ax, (title, total, saved, remain, toman) in zip(axes, panels):
        ax.bar([0], [saved], color=BLUE, width=0.5, label="صرفه‌جویی")
        ax.bar([0], [remain], bottom=[saved], color=GRAY, width=0.5, label="هدررفت باقی‌مانده")
        ax.text(0, total * 1.03, f"{total:,}", ha="center", fontsize=10, color="0.3")
        ax.text(0, saved / 2, f"{saved:,}", ha="center", va="center", fontsize=10.5,
                color="white", fontweight="bold")
        ax.text(0, saved + remain / 2, f"{remain:,}", ha="center", va="center", fontsize=9.5, color="0.25")
        ax.set_title(title, fontsize=11.5)
        ax.set_xlim(-0.9, 0.9)
        ax.set_ylim(0, total * 1.18)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.text(0, -total * 0.10, toman, ha="center", fontsize=11, fontweight="bold", color=BLUE)
    axes[0].legend(loc="upper left", fontsize=9, framealpha=0.9, bbox_to_anchor=(-0.05, 1.05))
    fig.tight_layout()
    _finish(fig, "slides_17_ladder")


# ---------------------------------------------------------------------------
# اسلاید ۱۸ — نرخ هدررفت پیش و پس از مدل
# ---------------------------------------------------------------------------

def fig_18_waste_rate() -> None:
    labels = ["بدون مدل\n(پخت = رزرو کامل)", "با مدل\n(چندک ۰.۲۰)"]
    vals = [8.05, 2.33]
    colors = [RED, BLUE]

    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    bars = ax.bar(labels, vals, color=colors, width=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.2, f"{v:.2f}٪", ha="center", fontsize=12)
    ax.annotate("", xy=(1, vals[1] + 1.1), xytext=(0, vals[0] + 1.1),
                arrowprops=dict(arrowstyle="-", color="0.3", lw=1.2,
                                 connectionstyle="arc3,rad=-0.25"))
    ax.text(0.5, max(vals) + 2.0, "۷۱٪ کاهش", ha="center", fontsize=15, fontweight="bold", color=BLUE)
    ax.set_ylabel("نرخ هدررفت (٪ رزروها)")
    ax.set_ylim(0, max(vals) + 3.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    _finish(fig, "slides_18_waste_rate")


# ---------------------------------------------------------------------------
# اسلاید ۱۹ — نجات‌یافته در برابر کمبود
# ---------------------------------------------------------------------------

def fig_19_tradeoff_bars() -> None:
    labels = ["پرس هدررفتِ نجات‌یافته", "پرس کمبود"]
    vals = [165155, 6944]
    colors = [BLUE, RED]

    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    ax.barh(labels, vals, color=colors, height=0.5)
    for i, v in enumerate(vals):
        ax.text(v + 3000, i, f"{v:,}", va="center", fontsize=12)
    ax.text(vals[0] * 0.5, 1.05, "۲۴ به ۱", ha="center", fontsize=17, fontweight="bold", color=BLUE)
    ax.set_xlabel("پرس در سال")
    ax.set_xlim(0, vals[0] * 1.28)
    ax.invert_yaxis()
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    _finish(fig, "slides_19_tradeoff_bars")


# ---------------------------------------------------------------------------
# اسلاید ۲۰ — نمونه‌ی یک سلف: فنی امیرآباد
# ---------------------------------------------------------------------------

def fig_20_restaurant_example() -> None:
    labels = ["رزرو", "پخت پیشنهادی\n(چندک ۰.۲۰)"]
    vals = [756, 709]
    colors = [GRAY_DARK, BLUE]

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    bars = ax.bar(labels, vals, color=colors, width=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 4, f"{v}", ha="center", fontsize=13, fontweight="bold")
    ax.annotate("", xy=(1, 709), xytext=(1, 756), arrowprops=dict(arrowstyle="-", color=RED, lw=1.3))
    ax.annotate("۴۷ پرس کمتر\n۱۷.۴ میلیون تومان", xy=(1.28, 732), fontsize=10.5, color=RED,
                fontweight="bold", va="center")
    ax.set_ylim(650, 800)
    ax.set_ylabel("پرس در هر وعده (میانگین)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    _finish(fig, "slides_20_restaurant_example")


# ---------------------------------------------------------------------------
# اسلاید ۲۱ — پوشش تقویمی داده (آذر تا خرداد)
# ---------------------------------------------------------------------------

def fig_21_calendar_coverage() -> None:
    months = ["فروردین", "اردیبهشت", "خرداد", "تیر", "مرداد", "شهریور",
              "مهر", "آبان", "آذر", "دی", "بهمن", "اسفند"]
    covered = {"فروردین", "اردیبهشت", "خرداد", "آذر", "دی", "بهمن", "اسفند"}

    fig, ax = plt.subplots(figsize=(10.5, 2.6))
    for i, m in enumerate(months):
        color = BLUE if m in covered else "#dddddd"
        ax.add_patch(Rectangle((i, 0), 0.92, 1, facecolor=color, edgecolor="white", lw=2))
        txt_color = "white" if m in covered else "0.4"
        ax.text(i + 0.46, 0.5, m, ha="center", va="center", fontsize=10.5,
                color=txt_color, rotation=0)
    ax.set_xlim(-0.2, 12.2)
    ax.set_ylim(-0.3, 1.3)
    ax.axis("off")
    ax.text(6, 1.15, "۷ از ۱۲ ماه سال در داده دیده شده — تابستان و نوروز نه",
            ha="center", fontsize=11.5, color=BLUE, fontweight="bold")
    fig.tight_layout()
    _finish(fig, "slides_21_calendar_coverage")


# ---------------------------------------------------------------------------
# اسلاید ۲۲ — پروتکل پایلوت: جفت‌های همتاشده
# ---------------------------------------------------------------------------

def fig_22_pilot_pairs() -> None:
    pairs = [
        ("مدیریت", 329, "علوم", 357),
        ("کشاورزی", 486, "کوی", 556),
    ]
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    for row, (p_name, p_vol, c_name, c_vol) in enumerate(pairs):
        y = 1 - row
        for x, name, vol, color, tag in [
            (0.0, p_name, p_vol, BLUE, "پایلوت"),
            (1.6, c_name, c_vol, GRAY_DARK, "کنترل"),
        ]:
            ax.add_patch(FancyBboxPatch((x, y - 0.32), 1.15, 0.64,
                                         boxstyle="round,pad=0.02,rounding_size=0.06",
                                         facecolor=color, edgecolor="white"))
            ax.text(x + 0.575, y + 0.10, name, ha="center", va="center", fontsize=12,
                    color="white", fontweight="bold")
            ax.text(x + 0.575, y - 0.14, f"{tag} · میانگین {vol} رزرو", ha="center", va="center",
                    fontsize=8.5, color="white")
        ax.annotate("", xy=(1.6, y), xytext=(1.15, y),
                    arrowprops=dict(arrowstyle="-", color="0.4", lw=1.3, ls=(0, (3, 2))))
    ax.set_xlim(-0.2, 2.95)
    ax.set_ylim(-0.7, 1.7)
    ax.axis("off")
    fig.tight_layout()
    _finish(fig, "slides_22_pilot_pairs")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_01_waffle()
    fig_03_asymmetric_cost()
    fig_05_pinball()
    fig_06_tau_mapping()
    fig_07_timeline()
    fig_08_dinner_given_lunch()
    fig_09_variance_donut()
    fig_15_funnel()
    fig_17_ladder()
    fig_18_waste_rate()
    fig_19_tradeoff_bars()
    fig_20_restaurant_example()
    fig_21_calendar_coverage()
    fig_22_pilot_pairs()


if __name__ == "__main__":
    main()
