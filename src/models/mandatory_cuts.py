"""بند 7.25.3 — برش‌های اجباری گزارش، روی برنده‌ی جدول 7.25.1 (دروازه‌ی M4، پروتکل
داوری قدم ۸): «برنده‌ی کلی ممکن است در یک برش مهم بازنده باشد».

از ۷ برش WBS، پنج‌تا اینجا روی L1 قابل‌محاسبه‌اند. دو مورد نیستند و **بی‌صدا حذف نشدند**:
- **cold-start در برابر باسابقه:** فقط سطح L5 (مدل B) معنا دارد؛ فاز ۷ فعلاً فقط L1 دارد.
- **پنجره‌ی انتهایی قفل‌شده:** طبق قاعده‌ی قرمز #۱ فاز ۷، فقط یک‌بار و در فاز ۸ لمس می‌شود.

اجرا: ``python -m src.models.mandatory_cuts``
"""

import importlib

import numpy as np
import pandas as pd

from src.baselines import b3_empirical_quantile, pinball_loss
from src.config import REPORTS_DIR
from src.cv import DATE_COL, load_cv_folds
from src.features.build import FEATURES_A_PATH
from src.models.axes import TUNING_TAU
from src.models.card_writer import _FAMILY_MODULES, load_s2_result

CHAMPION_FAMILY = "F02"
CHAMPION_MODEL = "lightgbm_quantile"

#: (برچسب فارسی، ستون) — بند 7.25.3. چارک Res جداگانه با qcut ساخته می‌شود.
CUT_COLUMNS = [
    ("ناهار در برابر شام", "Meal"),
    ("تهران در برابر غیرتهران", "is_tehran"),
    ("چارک Res", "res_quartile"),
    ("سلف دانشکده‌ای در برابر خوابگاهی", "is_khabgah"),
    ("روز عادی در برابر پیش‌ازتعطیلی", "is_day_before_holiday"),
    ("روز عادی در برابر رمضان", "is_ramadan"),
    ("روز عادی در برابر برف", "is_snow_day"),
]

_RAW_COLS = ["Meal", "is_tehran", "is_khabgah", "is_day_before_holiday", "is_ramadan",
            "is_snow_day", "Res", "rho"]


def _official_folds() -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    df = pd.read_parquet(FEATURES_A_PATH).sort_values(DATE_COL).reset_index(drop=True)
    fold_meta, _ = load_cv_folds()
    return [(df.loc[m1], df.loc[m2]) for f in fold_meta for m1, m2 in [f.masks(df[DATE_COL])]]


def build_cuts_table(tau: float = TUNING_TAU) -> pd.DataFrame:
    folds = _official_folds()
    result = load_s2_result(CHAMPION_MODEL, CHAMPION_FAMILY)
    hp = dict(result["best_hyperparams"])
    mod = importlib.import_module(_FAMILY_MODULES[CHAMPION_FAMILY])
    fit_fn = mod.MODELS[CHAMPION_MODEL]

    parts = []
    for tr, te in folds:
        pred_model = np.clip(np.asarray(fit_fn(tr, te, tau, **hp), dtype=float), 0.0, 1.0)
        pred_b3 = np.asarray(b3_empirical_quantile(tr, te, tau), dtype=float)
        part = te[_RAW_COLS].copy()
        part["pred_model"] = pred_model
        part["pred_b3"] = pred_b3
        parts.append(part)
    df = pd.concat(parts, ignore_index=True)
    df["res_quartile"] = pd.qcut(df["Res"], 4, labels=["Q1 (کوچک)", "Q2", "Q3", "Q4 (بزرگ)"],
                                 duplicates="drop")

    rows = []
    for label, col in CUT_COLUMNS:
        for seg, g in df.groupby(col, observed=True):
            pb_m = float(pinball_loss(g["rho"].to_numpy(), g["pred_model"].to_numpy(), tau).mean())
            pb_b3 = float(pinball_loss(g["rho"].to_numpy(), g["pred_b3"].to_numpy(), tau).mean())
            rows.append({
                "cut": label, "segment": str(seg), "n": len(g),
                "pinball_model": pb_m, "pinball_B3": pb_b3, "delta": pb_m - pb_b3,
                "champion_wins": pb_m < pb_b3,
            })
    return pd.DataFrame(rows)


def render_report(df: pd.DataFrame, tau: float) -> str:
    lines = [
        "# برش‌های اجباری — بند 7.25.3 (دروازه‌ی M4)",
        "",
        f"> برنده‌ی جدول 7.25.1: `{CHAMPION_MODEL}` ({CHAMPION_FAMILY}) در برابر B3، τ={tau}. "
        "هر ۵ fold رسمی، هایپرپارامتر S2 بدون تنظیم مجدد.",
        "",
        "| برش | بخش | n | pinball مدل | pinball B3 | Δ | برنده‌ی کلی اینجا هم برنده؟ |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        mark = "✅" if r["champion_wins"] else "❌ بازنده در این برش"
        lines.append(f"| {r['cut']} | {r['segment']} | {r['n']} | {r['pinball_model']:.5f} | "
                    f"{r['pinball_B3']:.5f} | {r['delta']:+.5f} | {mark} |")

    losers = df[~df["champion_wins"]]
    lines += ["", f"**در {len(losers)} از {len(df)} بخش، برنده‌ی کلی جدول اصلی را می‌بازد.**"]
    if len(losers):
        for _, r in losers.iterrows():
            lines.append(f"- `{r['cut']}` / بخش «{r['segment']}» (n={r['n']}): Δ={r['delta']:+.5f}")
    else:
        lines.append("در همه‌ی برش‌های قابل‌محاسبه، برنده‌ی جدول اصلی پایدار مانده است.")

    lines += [
        "",
        "⚠️ **دو برش خارج از محدوده‌ی این جدول (نه حذف بی‌صدا):**",
        "- cold-start در برابر باسابقه: فقط سطح L5 معنا دارد، فاز ۷ فعلاً فقط L1 دارد.",
        "- پنجره‌ی انتهایی قفل‌شده: طبق قاعده‌ی قرمز #۱ فاز ۷، فقط یک‌بار و در فاز ۸.",
    ]
    return "\n".join(lines)


def main() -> None:
    from src.config import set_global_seed

    set_global_seed()
    df = build_cuts_table()
    report = render_report(df, TUNING_TAU)
    out = REPORTS_DIR / "phase7"
    out.mkdir(parents=True, exist_ok=True)
    (out / "mandatory_cuts.md").write_text(report + "\n")
    df.to_csv(out / "mandatory_cuts.csv", index=False)
    print(report)
    print(f"\nذخیره شد در {out / 'mandatory_cuts.md'}")


if __name__ == "__main__":
    main()
