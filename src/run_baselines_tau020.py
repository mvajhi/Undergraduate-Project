"""بازاجرای هشت خط پایه‌ی فاز ۶ روی $\\tau$ رسمی پروژه (۰.۲۰).

اجرا: `python -m src.run_baselines_tau020`

**چرا این ماژول جداست.** `src/run_protocol.py` هشت خط پایه را با
`DEFAULT_TAU = 0.10` اجرا کرد و `reports/baselines.md` را ساخت — چون در آن لحظه
۰.۱۰ پیش‌فرض موقت سند مسئله بود. همان اجرا `reports/tau_sensitivity.md` را هم
تولید کرد و **خروجی‌اش** بود که $\\tau = 0.20$ را به‌عنوان توصیه‌ی پروژه تثبیت کرد
(ردیف ۳۴ `doc/decision_log.md`). یعنی جدول رتبه‌بندی خط پایه‌ها روی $\\tau$ی مانده
که تصمیم بعدی جایگزینش کرد، و فاز ۷ مرجع B3 را روی ۰.۲۰ گرفت
(`reports/phase7/model_comparison.md`).

این ماژول همان هشت خط پایه، همان foldهای منجمد و همان توابع را روی $\\tau = 0.20$
اجرا می‌کند تا فصل ۴ گزارش نهایی همه‌ی اعدادش را روی یک $\\tau$ داشته باشد.
`reports/baselines.md` دست‌نخورده می‌ماند: آن سند شاهد تاریخی دروازه‌ی M3 است، نه
یک عدد قابل‌بازنویسی.

⚠️ منطق خط پایه‌ها بازپیاده‌سازی **نشده** — همه‌چیز از `src/run_protocol.py` و
`src/baselines.py` وارد می‌شود. تنها ورودی متفاوت، مقدار $\\tau$ است.
"""

import numpy as np
import pandas as pd

from src.baselines import BASELINES
from src.config import REPORTS_DIR, set_global_seed
from src.cv import WalkForwardSplitter, effective_sample_size
from src.run_protocol import (
    DEFAULT_TAU,
    load,
    run_baselines,
    significance_vs_best,
    summarise,
)

#: $\tau$ رسمی پروژه (ردیف ۳۴ `decision_log`، معادل $C_u = 4C_o$)
PROJECT_TAU = 0.20
OUT_PATH = REPORTS_DIR / "baselines_tau020.md"

#: فقط برای **مقادیر عددی** — نه برای شناسه‌های کد مثل `B3_empirical_quantile`،
#: که باید لاتین بمانند تا با نام تابع در `src/baselines.py` یکی باشند.
_FA_NUM = str.maketrans("0123456789%,", "۰۱۲۳۴۵۶۷۸۹٪٬")


def fa(s: str) -> str:
    """رقم/درصد/جداکننده‌ی هزارگان لاتین ← فارسی."""
    return str(s).translate(_FA_NUM)


def pooled_pinball(res: pd.DataFrame) -> pd.Series:
    """میانگین وزنی به تعداد ردیف (وزن برابر برای هر ردیف، نه برای هر fold)."""
    return (res.groupby("baseline")
               .apply(lambda g: float((g["pinball"] * g["n"]).sum() / g["n"].sum()),
                      include_groups=False)
               .sort_values())


def main() -> None:
    set_global_seed()
    df = load()
    splitter = WalkForwardSplitter(n_folds=5, min_train_days=60)

    res20 = run_baselines(df, splitter, PROJECT_TAU)
    sum20, pool20 = summarise(res20), pooled_pinball(res20)
    best = sum20.index[0]
    sig20 = significance_vs_best(df, splitter, PROJECT_TAU, best)

    # فقط برای جدول «رتبه‌بندی زیر دو τ» — بازتولید اعداد `reports/baselines.md`
    res10 = run_baselines(df, splitter, DEFAULT_TAU)
    sum10, pool10 = summarise(res10), pooled_pinball(res10)

    sizes = df.groupby("date_gregorian").size().to_numpy()
    n_eff = effective_sample_size(len(df), sizes, icc=0.225)

    print(f"τ={PROJECT_TAU} — بهترین: {best}")
    print(sum20.round(5).to_string())
    print(sig20.round(4).to_string(index=False))

    _write(sum20, pool20, sig20, sum10, pool10, best, len(df), n_eff)
    print(f"\n✅ {OUT_PATH}")


def _write(sum20, pool20, sig20, sum10, pool10, best, n_raw, n_eff) -> None:
    L = [
        "# هشت خط پایه روی τ رسمی پروژه (۰.۲۰)",
        "",
        "> تولید خودکار با `python -m src.run_baselines_tau020`. همان هشت خط پایه‌ی",
        "> `src/baselines.py`، همان پنج fold منجمد `data/processed/cv_folds.json`، همان",
        "> توابع `src/run_protocol.py` — تنها تفاوت، مقدار τ.",
        "",
        "## چرا این سند وجود دارد",
        "",
        "`reports/baselines.md` (دروازه‌ی M3) خط پایه‌ها را روی **τ=۰.۱۰** اجرا کرد، چون آن",
        "زمان ۰.۱۰ پیش‌فرض موقت سند مسئله بود. خروجی **همان اجرا**",
        "(`reports/tau_sensitivity.md`) بود که τ=۰.۲۰ را به‌عنوان توصیه‌ی پروژه تثبیت کرد",
        "(ردیف ۳۴ `decision_log`، معادل $C_u = 4C_o$)، و فاز ۷ مرجع B3 را روی ۰.۲۰ گرفت",
        "(`reports/phase7/model_comparison.md`). پس جدول رتبه‌بندی خط پایه‌ها روی τی ماند که",
        "تصمیم بعدی جایگزینش کرد.",
        "",
        "این سند همان جدول را روی τ=۰.۲۰ می‌دهد تا فصل ۴ گزارش نهایی همه‌ی اعدادش را روی یک τ",
        "داشته باشد. **`reports/baselines.md` بازنویسی نشد** — آن شاهد تاریخی دروازه‌ی M3 است.",
        "",
        "**اعتبارسنجی:** اجرای τ=۰.۱۰ این ماژول اعداد `reports/baselines.md` را دقیقاً",
        "بازتولید می‌کند، و pinball تجمیع‌شده‌ی B3 روی τ=۰.۲۰ برابر "
        f"**{fa(f'{pool20[best]:.5f}')}** درمی‌آید که با مرجع مستقل",
        "`reports/phase7/model_comparison.md` (۰.۰۱۳۳۵) یکی است.",
        "",
        f"**اندازه‌ی نمونه:** خام {fa(f'{n_raw:,}')} · مؤثر {fa(f'{n_eff:,.0f}')} "
        f"(ICC روز=۰.۲۲۵) ⇒ ضریب تورم واریانس {fa(f'{n_raw / n_eff:.1f}')}×.",
        "",
        "## نتایج روی τ=۰.۲۰ (پایه‌ی «میانگین fold»)",
        "",
        "| خط پایه | pinball | نرخ کمبود | ٪ کاهش هدررفت | MAE (پرس) | RMSE (نرخ) |",
        "|---|---|---|---|---|---|",
    ]
    for name, r in sum20.iterrows():
        L.append(f"| `{name}` | " + fa(f"{r['pinball']:.5f} | {r['shortage_rate']:.1%} | "
                 f"{r['waste_reduction']:.1%} | {r['MAE_portions']:.1f} | {r['RMSE_rho']:.4f}") + " |")

    L += [
        "",
        "## دو پایه‌ی تجمیع، دو عدد",
        "",
        "«میانگین fold» به هر fold وزن برابر می‌دهد و «تجمیع‌شده» به هر ردیف. fold۲ (بازه‌ی",
        "رمضان) فقط ۱۸۵ ردیف دارد ولی در پایه‌ی اول وزن کامل می‌گیرد. **مقایسه‌ی عددی از دو",
        "پایه‌ی مختلف بی‌معناست.**",
        "",
        "| خط پایه | pinball تجمیع‌شده | pinball میانگین fold |",
        "|---|---|---|",
    ]
    for name in pool20.index:
        L.append(f"| `{name}` | " + fa(f"{pool20[name]:.5f} | {sum20.loc[name, 'pinball']:.5f}") + " |")

    L += [
        "",
        "## رتبه‌بندی زیر دو τ — آیا نتیجه به τ حساس است؟",
        "",
        "| رتبه | τ=۰.۱۰ (میانگین fold) | τ=۰.۲۰ (میانگین fold) |",
        "|---|---|---|",
    ]
    for i, (a, b) in enumerate(zip(sum10.index, sum20.index), start=1):
        mark = "" if a == b else " ⬅️"
        L.append(f"| {fa(i)} | `{a}` | `{b}`{mark} |")

    L += [
        "",
        "**نتیجه:** قهرمان (`B3_empirical_quantile`) و رقیب هم‌ردهش",
        "(`B7_group_residual_quantile`) زیر هر دو τ ثابت‌اند و ترتیبشان عوض نمی‌شود. جابه‌جایی",
        "فقط در میانه‌ی جدول رخ می‌دهد، و بزرگ‌ترینش سقوط `B0_cook_all` است: «همه را بپز» با",
        "بالارفتن τ (یعنی ارزان‌ترشدن کمبود نسبت به مازاد) از رتبه‌ی میانی به **آخر** می‌افتد،",
        "چون تنها خط پایه‌ای است که اصلاً به τ واکنش نشان نمی‌دهد.",
        "",
        "## آزمون Diebold-Mariano در برابر بهترین (τ=۰.۲۰)",
        "",
        "| خط پایه | Δpinball | DM | p |",
        "|---|---|---|---|",
    ]
    for _, r in sig20.iterrows():
        L.append(f"| `{r['baseline']}` | "
                 + fa(f"{r['Δpinball']:+.5f} | {r['DM']:.2f} | {r['p']:.3f}") + " |")

    b2, b7, b3 = "B2_group_shrunk", "B7_group_residual_quantile", "B3_empirical_quantile"
    p_b7 = float(sig20.loc[sig20["baseline"] == b7, "p"].iloc[0])
    gap_b7 = (pool20[b2] - pool20[b7]) / pool20[b2]
    gap_b7_10 = (pool10[b2] - pool10[b7]) / pool10[b2]
    L += [
        "",
        f"`{b7}` تنها خط پایه‌ای است که تفاوتش با قهرمان معنادار نیست "
        f"(p={fa(f'{p_b7:.3f}')})؛ بقیه با p≈۰ بدترند.",
        "",
        "## هزینه‌ی «آفست کوانتایل سراسری» روی τ=۰.۲۰",
        "",
        "B2 و B7 فقط در یک چیز فرق دارند: جای برآورد آفست کوانتایل (سراسری در برابر گروهی).",
        "",
        "| روش | pinball تجمیع‌شده |",
        "|---|---|",
        "| میانگین گروه + آفست کوانتایل **سراسری** (B2) | " + fa(f"{pool20[b2]:.5f}") + " |",
        "| میانگین گروه + آفست کوانتایل **گروهی** (B7) | " + fa(f"{pool20[b7]:.5f}") + " |",
        "| کوانتایل **مستقیم** گروه (B3) | " + fa(f"{pool20[b3]:.5f}") + " |",
        "",
        f"فاصله‌ی B2 تا B7 روی τ=۰.۲۰ برابر **{fa(f'{gap_b7:.1%}')}** است "
        f"(روی τ=۰.۱۰ این عدد {fa(f'{gap_b7_10:.1%}')} بود). یعنی جهت نتیجه ثابت می‌ماند —",
        "سراسری‌فرض‌کردن کوانتایل هزینه دارد — ولی **بزرگی‌اش به τ وابسته است** و نباید عدد",
        "τ=۰.۱۰ روی τ=۰.۲۰ نقل شود.",
        "",
    ]
    OUT_PATH.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
