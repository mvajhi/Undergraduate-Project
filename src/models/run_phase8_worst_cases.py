"""بند ۸.۴ WBS فاز ۸ — تحلیل بدترین موارد روی Test قفل‌شده (بند ۸.۱).

قهرمان رسمی (`lightgbm_quantile`). دو جدول: ۲۰ پیش‌بینی با بیشترین pinball loss،
و جداگانه بدترین **کمبودها** (چون هزینه‌ی کمبود نامتقارن و بالاتر است، بند ۶.۷/۱۰.۲).

اجرا: ``python -m src.models.run_phase8_worst_cases``
"""

import numpy as np
import pandas as pd

from src.baselines import pinball_loss
from src.config import REPORTS_DIR
from src.models.axes import TUNING_TAU
from src.models.run_phase8_error_decomposition import load

OUT_DIR = REPORTS_DIR / "phase8"
CHAMPION = "lightgbm_quantile"
TAU = TUNING_TAU
DISPLAY_COLS = ["date_gregorian", "RestaurantName", "Meal", "FoodType", "dow_name",
                 "day_type", "Res", "Recv", "rho", "pred_q", "cook_qty", "shortfall", "pinball"]


def build(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pinball"] = pinball_loss(df["rho"].to_numpy(), df["pred_q"].to_numpy(), TAU)
    df["cook_qty"] = np.ceil(df["Res"].to_numpy() * (1.0 - np.clip(df["pred_q"].to_numpy(), 0.0, 1.0)))
    df["shortfall"] = np.maximum(df["Recv"].to_numpy() - df["cook_qty"].to_numpy(), 0.0)
    return df


def _fmt_table(tbl: pd.DataFrame) -> list[str]:
    lines = ["| تاریخ | سلف | وعده | نوع غذا | روز | نوع روز | Res | Recv | ρ | ρ̂ | پخت | کمبود | pinball |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in tbl.iterrows():
        lines.append(
            f"| {pd.Timestamp(r['date_gregorian']).date()} | {r['RestaurantName']} | {r['Meal']} | "
            f"{r['FoodType']} | {r['dow_name']} | {r['day_type']} | {r['Res']:.0f} | {r['Recv']:.0f} | "
            f"{r['rho']:.3f} | {r['pred_q']:.3f} | {r['cook_qty']:.0f} | {r['shortfall']:.0f} | {r['pinball']:.4f} |"
        )
    return lines


def _pattern_check(df: pd.DataFrame, worst: pd.DataFrame, col: str, label: str) -> str:
    base = df[col].value_counts(normalize=True)
    top = worst[col].value_counts(normalize=True)
    over = []
    for key, share in top.items():
        base_share = base.get(key, 0.0)
        if share >= 0.25 and share > 2 * base_share:
            over.append(f"{key} ({share:.0%} در برابر پایه‌ی {base_share:.0%})")
    return f"- {label}: " + ("، ".join(over) if over else "الگوی غالب دیده نشد")


def main() -> None:
    df = build(load())

    worst_pinball = df.sort_values("pinball", ascending=False).head(20)
    worst_shortfall = df[df["shortfall"] > 0].sort_values("shortfall", ascending=False).head(20)

    lines = [
        "# بند ۸.۴ — تحلیل بدترین موارد روی Test قفل‌شده",
        "",
        f"> مدل: `{CHAMPION}`. τ={TAU}. داده: پنجره‌ی Test بند ۸.۱ (۱٬۵۲۶ ردیف).",
        "",
        "## ۲۰ پیش‌بینی با بیشترین pinball loss",
        "",
    ]
    lines += _fmt_table(worst_pinball[DISPLAY_COLS])
    lines += [
        "",
        "### الگوی مشترک (بدترین ۲۰ در برابر پایه‌ی کل پنجره)",
        "",
        _pattern_check(df, worst_pinball, "RestaurantName", "سلف"),
        _pattern_check(df, worst_pinball, "Meal", "وعده"),
        _pattern_check(df, worst_pinball, "dow_name", "روز هفته"),
        _pattern_check(df, worst_pinball, "day_type", "نوع روز"),
        _pattern_check(df, worst_pinball, "FoodType", "نوع غذا"),
        "",
        "## ۲۰ بدترین کمبود (شکاف پخت − دریافت، جداگانه چون هزینه‌اش نامتقارن است)",
        "",
    ]
    lines += _fmt_table(worst_shortfall[DISPLAY_COLS])
    lines += [
        "",
        "### الگوی مشترک بدترین کمبودها",
        "",
        _pattern_check(df, worst_shortfall, "RestaurantName", "سلف"),
        _pattern_check(df, worst_shortfall, "Meal", "وعده"),
        _pattern_check(df, worst_shortfall, "dow_name", "روز هفته"),
        _pattern_check(df, worst_shortfall, "day_type", "نوع روز"),
        _pattern_check(df, worst_shortfall, "FoodType", "نوع غذا"),
        "",
        f"**همپوشانی:** {len(set(worst_pinball.index) & set(worst_shortfall.index))} از ۲۰ ردیف "
        "هم در بدترین pinball و هم در بدترین کمبود حاضرند.",
    ]

    report = "\n".join(lines)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "8.4_worst_cases.md").write_text(report + "\n")
    worst_pinball[DISPLAY_COLS].to_csv(OUT_DIR / "8.4_worst_pinball.csv", index=False)
    worst_shortfall[DISPLAY_COLS].to_csv(OUT_DIR / "8.4_worst_shortfall.csv", index=False)
    print(report)
    print(f"\nذخیره شد در {OUT_DIR}/8.4_worst_cases.md")


if __name__ == "__main__":
    main()
