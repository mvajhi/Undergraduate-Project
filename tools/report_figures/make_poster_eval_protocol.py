"""نمودار پروتکل ارزیابی برای باکس «روش» پوستر (`doc/poster/poster-method-box-spec.md`, بند ۴).

پیش‌تر این شکل مستقیم در ابزاری بیرون از مخزن ساخته می‌شد و نتیجه‌اش (`poster_eval_protocol.png/.pdf`)
بدون اسکریپت منبع در `poster/assets/` نشسته بود و رندر فارسی‌اش باگ داشت (یک حرف اضافه‌ی
شکسته در «آموزش»). این اسکریپت هر دو فایل را از نو و به‌صورت تکرارپذیر می‌سازد — دقیقاً با همان
الگوی `tools/report_figures/make_slides_figs.py` (`src.viz_fa`، فقط `arabic_reshaper`): محیط
matplotlib این پروژه ترتیب راست‌به‌چپ را خودش هنگام رندر انجام می‌دهد، پس افزودن
`python-bidi` روی آن، متن را دوباره برمی‌گرداند و خراب می‌کند (تجربه‌شده هنگام ساخت این فایل).

اجرا: ``PYTHONPATH=. .venv/bin/python tools/report_figures/make_poster_eval_protocol.py``
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src import viz_fa  # noqa: F401 -- فونت فارسی، اثر جانبی روی import
from src.config import ROOT_DIR

OUT_DIR = ROOT_DIR / "poster" / "assets"

GRAY = "#B9BCC1"
BLUE = "#3B6E9E"
DARK = "#3A3A3A"

FS = 17  # اندازه‌ی برچسب‌های اصلی؛ ۱۳ کنتراست دیدش روی پوستر کم بود
FS_CAPTION = 15

PERSIAN_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")


def build_figure():

    n_rounds = 5
    test_len = 3.0
    train_lens = [8.0 + i * 2.5 for i in range(n_rounds)]  # ۸، ۱۰.۵، ۱۳، ۱۵.۵، ۱۸ — گسترش یکنواخت

    bh, gap = 0.68, 0.44
    row_step = bh + gap
    round_ys = [(n_rounds - 1 - i) * row_step for i in range(n_rounds)]
    holdout_y = -row_step * 1.65

    fig, ax = plt.subplots(figsize=(10.6, 2.75))

    # مبدأ آموزش ثابت
    ax.axvline(0, color="0.35", ls=":", lw=1.3, zorder=1)
    ax.text(0.3, round_ys[0] + bh / 2 + 0.34, "مبدأ آموزش ثابت",
            ha="right", va="bottom", fontsize=FS, color=DARK)

    for i in range(n_rounds):
        y = round_ys[i]
        train_len = train_lens[i]
        ax.add_patch(Rectangle((0, y - bh / 2), train_len, bh,
                                facecolor=GRAY, edgecolor="white", lw=1.0, zorder=3))
        ax.add_patch(Rectangle((train_len, y - bh / 2), test_len, bh,
                                facecolor=BLUE, edgecolor="white", lw=1.0, zorder=3))
        label = f"دور {str(i + 1).translate(PERSIAN_DIGITS)}"
        # ha="right": محور معکوس است، پس این متن را به‌جای برگشتن روی نوار، به چپ می‌راند
        ax.text(train_len + test_len + 0.4, y, label, ha="right", va="center",
                fontsize=FS, color=DARK)

    # جداکننده‌ی بصری بین دورهای اعتبارسنجی و پنجره‌ی قفل‌شده
    sep_y = (round_ys[-1] + holdout_y) / 2 + row_step * 0.18
    x_max = train_lens[-1] + test_len + 6.5
    ax.axhline(sep_y, color="0.8", lw=1.0, zorder=1, xmin=0.0, xmax=1.0)

    holdout_train_len = train_lens[-1] + test_len
    holdout_test_len = test_len
    ax.add_patch(Rectangle((0, holdout_y - bh / 2), holdout_train_len, bh,
                            facecolor=GRAY, edgecolor=DARK, lw=1.5, ls="--", zorder=3))
    ax.add_patch(Rectangle((holdout_train_len, holdout_y - bh / 2), holdout_test_len, bh,
                            facecolor=BLUE, alpha=0.30, edgecolor=DARK, lw=1.5, ls="--",
                            hatch="//", zorder=3))
    ax.text(holdout_train_len + holdout_test_len + 0.4, holdout_y, "پنجره‌ی قفل‌شده",
            ha="right", va="center", fontsize=FS, color=DARK, fontweight="bold")
    ax.text(holdout_train_len + holdout_test_len, holdout_y - bh / 2 - 0.30,
            "فقط یک‌بار باز شد، بعد از انتخاب مدل", ha="right", va="top",
            fontsize=FS_CAPTION, color=DARK, fontweight="bold")

    # راهنما — راست‌به‌چپ: اول «آموزش» (خاکستری)، بعد «آزمون» (آبی).
    # محور معکوس است: x کوچک‌تر یعنی سمت راستِ صفحه، پس گروه «آموزش» باید x کوچک‌تر بگیرد.
    legend_y = round_ys[0] + bh / 2 + 0.85
    swatch_text_gap = 1.4  # فاصله‌ی امن بین متن و نمونه‌رنگ؛ با بزرگ‌شدن فونت این هم بزرگ‌تر شد
    train_x = x_max - 9.0
    ax.add_patch(Rectangle((train_x - 1.5, legend_y - 0.13), 1.5, 0.30, facecolor=GRAY, edgecolor="none"))
    ax.text(train_x - 1.5 - swatch_text_gap, legend_y + 0.02, "آموزش", ha="left", va="center",
            fontsize=FS, color=DARK)
    test_x = x_max - 3.0
    ax.add_patch(Rectangle((test_x - 1.5, legend_y - 0.13), 1.5, 0.30, facecolor=BLUE, edgecolor="none"))
    ax.text(test_x - 1.5 - swatch_text_gap, legend_y + 0.02, "آزمون", ha="left", va="center",
            fontsize=FS, color=DARK)

    ax.set_xlim(x_max, -2.0)  # محور معکوس: مبدأ در راست، گذر زمان به چپ (خوانش فارسی)
    ax.set_ylim(holdout_y - bh / 2 - 0.75, legend_y + 0.30)
    ax.axis("off")
    fig.tight_layout()
    return fig


def main() -> None:
    fig = build_figure()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix, dpi in ((".pdf", 300), (".png", 300)):
        out_path = OUT_DIR / f"poster_eval_protocol{suffix}"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(out_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
