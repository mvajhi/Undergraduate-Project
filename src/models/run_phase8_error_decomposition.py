"""بند ۸.۳ WBS فاز ۸ — تجزیه‌ی خطا به تفکیک ابعاد، روی Test قفل‌شده (بند ۸.۱).

قهرمان رسمی (`lightgbm_quantile`) به تفکیک: سلف، وعده، روز هفته، چارک Res، نوع غذا،
نوع روز (عادی/قبل‌تعطیل/امتحان)، ماه‌به‌ماه. مدل B (سطح فرد) در فهرست نامزدهای بند ۸.۱
نیست (بند ۸.۱۰ جداگانه L1/L2 در برابر L5 را می‌سنجد) — پس ردیف‌های اختصاصی مدل B اینجا
اعمال نمی‌شوند.

داده از `data/interim/phase8_holdout_predictions.parquet` (بند ۸.۱) + ستون‌های
FoodType/is_holiday_any/... که در آن فایل نگه داشته نشده بودند، از
`src/models/run_phase8_holdout_eval.py::load_holdout` بازخوانی و **به همان ترتیب
ردیفی قطعی** (sort + mask یکسان، بدون بازبرازش) الحاق می‌شوند.

اجرا: ``python -m src.models.run_phase8_error_decomposition``
"""

import numpy as np
import pandas as pd

from src.baselines import pinball_loss
from src.config import REPORTS_DIR
from src.models.axes import TUNING_TAU
from src.models.run_phase8_holdout_eval import load_holdout

PRED_PATH = REPORTS_DIR.parent / "data" / "interim" / "phase8_holdout_predictions.parquet"
OUT_DIR = REPORTS_DIR / "phase8"
CHAMPION = "lightgbm_quantile"
TAU = TUNING_TAU

EXTRA_COLS = ["FoodType", "FoodName", "is_holiday_any", "is_day_before_holiday",
              "is_exam_period", "dow_name", "ym"]


def load() -> pd.DataFrame:
    preds = pd.read_parquet(PRED_PATH)
    _, test_full, _ = load_holdout()
    assert len(preds) == len(test_full), "ترتیب/طول ردیف پیش‌بینی و داده‌ی کامل باید یکی باشد"
    for c in EXTRA_COLS:
        preds[c] = test_full[c].to_numpy()
    preds["pred_q"] = preds[f"pred__{CHAMPION}"]
    preds["day_type"] = np.select(
        [preds["is_exam_period"].astype(bool), preds["is_day_before_holiday"].astype(bool),
         preds["is_holiday_any"].astype(bool)],
        ["امتحان", "قبل‌تعطیل", "تعطیل"], default="عادی")
    preds["res_quartile"] = pd.qcut(preds["Res"], 4, labels=["Q1(کم)", "Q2", "Q3", "Q4(زیاد)"])
    return preds


def _group_table(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(group_col, observed=True):
        pb = pinball_loss(g["rho"].to_numpy(), g["pred_q"].to_numpy(), TAU)
        cook = np.ceil(g["Res"].to_numpy() * (1.0 - np.clip(g["pred_q"].to_numpy(), 0, 1)))
        shortage = (cook < g["Recv"].to_numpy()).mean()
        coverage = (g["rho"].to_numpy() <= g["pred_q"].to_numpy()).mean()
        rows.append({group_col: key, "n": len(g), "pinball": pb.mean(),
                     "shortage_rate": shortage, "coverage": coverage,
                     "coverage_gap": coverage - TAU})
    return pd.DataFrame(rows).sort_values(group_col)


def render_table(title: str, dim_label: str, tbl: pd.DataFrame, col: str) -> list[str]:
    lines = [f"### {title}", "", f"| {dim_label} | n | pinball | نرخ کمبود | پوشش | شکاف پوشش |",
             "|---|---|---|---|---|---|"]
    for _, r in tbl.iterrows():
        lines.append(f"| {r[col]} | {int(r['n'])} | {r['pinball']:.5f} | "
                      f"{r['shortage_rate']:.1%} | {r['coverage']:.1%} | {r['coverage_gap']:+.1%} |")
    lines.append("")
    return lines


def main() -> None:
    df = load()
    dims = [
        ("سلف", "RestaurantName", "به تفکیک سلف"),
        ("وعده", "Meal", "به تفکیک وعده"),
        ("روز هفته", "dow_name", "به تفکیک روز هفته"),
        ("چارک Res", "res_quartile", "به تفکیک چارک حجم رزرو"),
        ("نوع غذا", "FoodType", "به تفکیک نوع غذا"),
        ("نوع روز", "day_type", "به تفکیک نوع روز (عادی/قبل‌تعطیل/تعطیل/امتحان)"),
        ("ماه", "ym", "ماه‌به‌ماه (بررسی drift)"),
    ]

    lines = [
        "# بند ۸.۳ — تجزیه‌ی خطا به تفکیک ابعاد روی Test قفل‌شده",
        "",
        f"> مدل: `{CHAMPION}` (قهرمان رسمی فاز ۷). داده: پنجره‌ی Test بند ۸.۱ (۱٬۵۲۶ ردیف، ۲۵ روز).",
        f"> τ={TAU}. مرجع کلی از بند ۸.۱: pinball=۰.۰۱۰۰۸، نرخ کمبود=۲۰.۲٪، پوشش=۳۳.۶٪.",
        "",
    ]
    tables = {}
    for dim_label, col, title in dims:
        tbl = _group_table(df, col)
        tables[col] = tbl
        lines += render_table(title, dim_label, tbl, col)

    worst_rest = tables["RestaurantName"].sort_values("pinball", ascending=False).head(3)
    best_rest = tables["RestaurantName"].sort_values("pinball", ascending=True).head(3)
    day_type_gap = tables["day_type"].set_index("day_type")["pinball"]
    normal_pb = day_type_gap.get("عادی", float("nan"))
    non_normal = day_type_gap.drop(labels=["عادی"], errors="ignore") - normal_pb
    worst_day_type = non_normal.idxmax() if len(non_normal) > 0 else None

    lines += [
        "## خلاصه",
        "",
        f"- بدترین ۳ سلف (pinball): {', '.join(worst_rest['RestaurantName'].astype(str))}",
        f"- بهترین ۳ سلف (pinball): {', '.join(best_rest['RestaurantName'].astype(str))}",
    ]
    if worst_day_type is not None:
        delta = day_type_gap[worst_day_type] - normal_pb
        verdict = "بدتر از" if delta > 0 else "بهتر یا هم‌تراز با"
        lines.append(
            f"- بیشترین فاصله‌ی نوع‌روز نسبت به «عادی» ({normal_pb:.5f}): `{worst_day_type}` "
            f"({day_type_gap[worst_day_type]:.5f}, Δ={delta:+.5f} — {verdict} عادی؛ نمونه‌ی کوچک، احتیاط لازم)")
    month_tbl = tables["ym"].sort_values("ym")
    if len(month_tbl) > 1:
        drift = month_tbl["pinball"].iloc[-1] - month_tbl["pinball"].iloc[0]
        lines.append(f"- روند ماه‌به‌ماه (اولین در برابر آخرین ماه پنجره‌ی Test): Δpinball={drift:+.5f} "
                     f"({'بدترشدن' if drift > 0 else 'بهترشدن یا ثابت'} در طول پنجره)")

    report = "\n".join(lines)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "8.3_error_decomposition.md").write_text(report + "\n")
    for col, tbl in tables.items():
        tbl.to_csv(OUT_DIR / f"8.3_by_{col}.csv", index=False)
    print(report)
    print(f"\nذخیره شد در {OUT_DIR}/8.3_error_decomposition.md")


if __name__ == "__main__":
    main()
