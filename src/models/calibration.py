"""بند 7.4 گام ۱۳ (کالیبراسیون و پوشش) — یکی از چهار گام غیرقابل‌حذف کارت مدل.

با بهترین هایپرپارامتر هر مدل (خروجی S2)، مدل به‌ازای هر ۵ fold دوباره برازش و
پیش‌بینی می‌شود. چون foldهای ``src/cv.py`` پنجره‌ی گسترشی‌اند و بازه‌های آزمونشان
هم‌پوشان نیستند، اتصال پیش‌بینی‌های out-of-fold هر ۵ fold یک نمونه‌ی out-of-sample
تقریباً کامل می‌سازد (به‌جز ۶۰ روز اول که فقط در نقش train ظاهر می‌شوند) — دقیقاً
همان چیزی که سنجش پوشش تجربی نیاز دارد؛ نه یک شبیه‌سازی، بازبرازش واقعی است.

بند 7.22.3: مجموعه‌ی سنجش پوشش باید **زمانی** باشد، نه تصادفی — دقیقاً همین‌جا با
استفاده از foldهای رسمی fold تضمین می‌شود.
"""

from typing import Callable

import numpy as np
import pandas as pd

#: بند ۱۴۱۰ WBS: چارک $Res$ یکی از چهار برش اجباری پوشش شرطی است.
RES_QUARTILE_LABELS = ["Q1(کم)", "Q2", "Q3", "Q4(زیاد)"]
#: زیر این تعداد رکورد در یک برش، پرچم‌گذاری میانگین را نامعتبر می‌دانیم (نویز نمونه‌ی کوچک).
_MIN_N_FOR_FLAG = 20
#: شکاف پوشش تجربی از τ اسمی بالاتر از این آستانه، پرچم ⚠️ می‌گیرد.
_GAP_FLAG_THRESHOLD = 0.05


def oof_predictions(fit_fn: Callable, folds: list[tuple[pd.DataFrame, pd.DataFrame]],
                    tau: float, hyperparams: dict) -> pd.DataFrame:
    """بازبرازش ``fit_fn`` (امضای یکسان ``fit_predict_*``، بند 7.1.1) روی هر fold با
    بهترین هایپرپارامتر S2، و تجمیع پیش‌بینی‌های out-of-fold + ستون‌های برش."""
    parts = []
    for tr, te in folds:
        pred = np.asarray(fit_fn(tr, te, tau, **hyperparams), dtype=float)
        parts.append(pd.DataFrame({
            "actual": te["rho"].to_numpy(),
            "pred_q": pred,
            "RestaurantName": te["RestaurantName"].to_numpy(),
            "Meal": te["Meal"].to_numpy(),
            "Res": te["Res"].to_numpy(),
            "is_tehran": te["is_tehran"].to_numpy(),
        }))
    df = pd.concat(parts, ignore_index=True)
    df["covered"] = (df["actual"] <= df["pred_q"]).astype(float)
    df["res_quartile"] = pd.qcut(df["Res"], 4, labels=RES_QUARTILE_LABELS, duplicates="drop")
    return df


def _coverage_row(label: str, sub: pd.DataFrame, tau: float) -> dict:
    n = len(sub)
    cov = float(sub["covered"].mean()) if n else float("nan")
    gap = cov - tau if n else float("nan")
    flag = "⚠️" if n >= _MIN_N_FOR_FLAG and np.isfinite(gap) and abs(gap) > _GAP_FLAG_THRESHOLD else ""
    return {"label": label, "n": n, "coverage": cov, "gap": gap, "flag": flag}


def coverage_by_cut(df: pd.DataFrame, cut_col: str, tau: float) -> list[dict]:
    return [_coverage_row(str(val), sub, tau) for val, sub in df.groupby(cut_col, observed=True)]


def render_step13(df: pd.DataFrame, tau: float) -> str:
    overall = _coverage_row("کلی", df, tau)
    cuts = [("Meal", "وعده"), ("RestaurantName", "سلف"),
           ("res_quartile", "چارک Res"), ("is_tehran", "تهران؟")]

    all_rows = []
    for col, label in cuts:
        for r in coverage_by_cut(df, col, tau):
            all_rows.append((label, r))
    worst_label, worst = max(all_rows, key=lambda lr: abs(lr[1]["gap"]) if np.isfinite(lr[1]["gap"]) else -1)

    lines = [
        f"پوشش تجربی @τ={tau} روی پیش‌بینی‌های out-of-fold تجمیع‌شده‌ی هر ۵ fold "
        "(بازبرازش واقعی با بهترین هایپرپارامتر S2 — نه شبیه‌سازی؛ چون foldها پنجره‌ی "
        "گسترشی‌اند و آزمون‌هایشان هم‌پوشان نیستند، اتصالشان یک نمونه‌ی out-of-sample "
        "تقریباً کامل می‌سازد، بند 7.22.3).",
        "",
        f"**پوشش کلی: {overall['coverage']:.4f}** (اسمی τ={tau}, شکاف={overall['gap']:+.4f}, "
        f"n={overall['n']:,})",
        "",
        f"🔴 **بدترین برش (بند ۱۴۱۰):** {worst_label}=`{worst['label']}` — پوشش="
        f"{worst['coverage']:.4f} (شکاف={worst['gap']:+.4f}, n={worst['n']:,}){worst['flag']}",
        "",
        "| برش | مقدار | n | پوشش تجربی | شکاف از τ | پرچم |",
        "|---|---|---|---|---|---|",
    ]
    for col, label in cuts:
        for r in coverage_by_cut(df, col, tau):
            lines.append(f"| {label} | `{r['label']}` | {r['n']:,} | {r['coverage']:.4f} | "
                        f"{r['gap']:+.4f} | {r['flag']} |")
    n_flagged = sum(1 for _, r in all_rows if r["flag"])
    lines += ["", f"⚠️ {n_flagged} از {len(all_rows)} ردیف برش، شکاف پوشش بیش از "
             f"{_GAP_FLAG_THRESHOLD:.0%} دارند."]
    return "\n".join(lines)
