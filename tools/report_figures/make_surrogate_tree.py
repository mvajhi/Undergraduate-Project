"""رسم درخت جانشین عمق ۳ با برچسب‌های کوتاه فارسی و اندازه‌ی خوانا."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from src import viz_fa  # noqa: F401

NODE = "#dbe6f0"; LEAF = "#f3ddcb"

# (متن، x, y) — ساختار از reports/phase9/9.5_surrogate_tree.txt
internal = {
    "root": ("نرخ تاریخی سلول×روزهفته\n≤ 0.065", 0.500, 0.86),
    "L":    ("نرخ تاریخی سلول\n≤ 0.040",        0.250, 0.58),
    "R":    ("حجم رزرو (لگاریتم)\n≤ 2.441",      0.750, 0.58),
    "LL":   ("کل رزرو روز\n≤ 8.786",             0.125, 0.32),
    "LR":   ("شوک روز قبل\n≤ 0.010",             0.375, 0.32),
    "RL":   ("فصلی روز هفته\n≤ −0.391",          0.625, 0.32),
    "RR":   ("نرخ کوچک‌شده\n≤ 0.112",            0.875, 0.32),
}
# برگ‌ها: (درصد پخت از رزرو)
leaves = [96.4, 98.4, 95.3, 92.7, 98.0, 99.6, 92.8, 90.0]
edges = [("root","L"),("root","R"),("L","LL"),("L","LR"),("R","RL"),("R","RR")]

fig, ax = plt.subplots(figsize=(8.6, 4.6))
ax.set_xlim(0, 1); ax.set_ylim(0.02, 1); ax.axis("off")

def box(x, y, text, color, fs=8.5):
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, zorder=3,
            bbox=dict(boxstyle="round,pad=0.34", fc=color, ec="0.45", lw=0.8))

for a, b in edges:
    x1, y1 = internal[a][1], internal[a][2]
    x2, y2 = internal[b][1], internal[b][2]
    ax.plot([x1, x2], [y1 - 0.055, y2 + 0.055], c="0.5", lw=1.0, zorder=1)

for i, val in enumerate(leaves):
    lx = 0.0625 + i * 0.125
    parent = ["LL","LL","LR","LR","RL","RL","RR","RR"][i]
    px, py = internal[parent][1], internal[parent][2]
    ax.plot([px, lx], [py - 0.055, 0.10 + 0.045], c="0.5", lw=1.0, zorder=1)
    box(lx, 0.10, f"{val:.1f}٪", LEAF, fs=9)

for key, (txt, x, y) in internal.items():
    box(x, y, txt, NODE)

ax.text(0.34, 0.735, "بله", fontsize=8, color="0.35", ha="center")
ax.text(0.66, 0.735, "خیر", fontsize=8, color="0.35", ha="center")
ax.text(0.5, 0.005, "عدد هر برگ: درصدی از رزرو که باید پخته شود", fontsize=8.5,
        ha="center", color="0.3")
fig.tight_layout()
# final_report/img/surrogate_tree.png symlink به این است — reports/figures/ منبع
# حقیقت تصاویر گزارش است (tools/report_figures/README.md).
fig.savefig("reports/figures/phase9/9.5_surrogate_tree_fa.png", dpi=220, bbox_inches="tight")
print("saved")
