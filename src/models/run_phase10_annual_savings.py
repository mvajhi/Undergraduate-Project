"""بند ۱۰.۳/۱۰.۴ WBS فاز ۱۰ — سناریوهای سیاستی + برآورد صرفه‌جویی سالانه با عدم‌قطعیت.

**پایه‌ی داده:** OOF قهرمان روی هر ۵ fold رسمی (نه پنجره‌ی Test بند ۸.۱) — چون Test
یک رژیم غیرعادی مستند است (بازگشت از رمضان + سوگواری ملی، ردیف ۴۹ decision_log) و
تعمیم سالانه از یک نمونه‌ی غیرعادی گمراه‌کننده است؛ ۵ fold رسمی نماینده‌ی رفتار
«معمول» سال تحصیلی‌اند.

**آمار پایه:** میانگین پرس صرفه‌جویی‌شده **به‌ازای هر ردیف** (نه به‌ازای هر روز) با
بوت‌استرپ بلوکی دوبعدی — نه میانگین روزانه، چون آماره‌ی نسبتی «مجموع/تعداد‌روز‌یکتا»
زیر بوت‌استرپ بلوکی سوگیری رو به بالا دارد (تکرار یک روز در نمونه‌گیری، صورت‌کسر را
تکرار می‌کند ولی مخرج -تعداد روز یکتا- را نه). میانگین ردیفی این مشکل را ندارد و
دقیقاً همان الگویی است که برای Δpinball در سراسر فاز ۷/۸ استفاده شده.

**فرض تعمیم سالانه (صریح، طبق دستور بند ۱۰.۴):** سال تحصیلی ≈ ۳۲ هفته‌ی آموزشی (دو
نیم‌سال ۱۶هفته‌ای) — **نه ۵۲ هفته**، چون تعطیلات نوروز/تابستان الگوی سرو ندارند. این
یک فرض صریح است، نه واقعیت اندازه‌گیری‌شده؛ گزارش باید دقیقاً همین‌طور نقلش کند.

اجرا: ``python -m src.models.run_phase10_annual_savings``
"""

import importlib

import numpy as np
import pandas as pd

from src.config import REPORTS_DIR, set_global_seed
from src.cv import DATE_COL, block_bootstrap_2d, load_cv_folds
from src.features.build import FEATURES_A_PATH
from src.models.card_writer import load_s2_result

OUT_DIR = REPORTS_DIR / "phase10"
COST_PER_PORTION_TOMAN = 120_000
ACADEMIC_WEEKS_PER_YEAR = 32          # فرض صریح — دو نیم‌سال ۱۶هفته‌ای
SCENARIOS = {"محافظه‌کارانه": 0.02, "متعادل": 0.10, "تهاجمی": 0.20}


def oof_predictions(tau: float) -> pd.DataFrame:
    df = pd.read_parquet(FEATURES_A_PATH).sort_values(DATE_COL).reset_index(drop=True)
    fold_meta, _ = load_cv_folds()
    folds = [(df.loc[m1], df.loc[m2]) for f in fold_meta for m1, m2 in [f.masks(df[DATE_COL])]]
    result = load_s2_result("lightgbm_quantile", "F02")
    fn = importlib.import_module("src.models.families.f02_tree").MODELS["lightgbm_quantile"]

    parts = []
    for tr, te in folds:
        pred = np.clip(np.asarray(fn(tr, te, tau, **result["best_hyperparams"]), dtype=float), 0.0, 1.0)
        part = te[[DATE_COL, "Meal", "RestaurantName", "Res", "Recv"]].copy()
        part["pred_q"] = pred
        parts.append(part)
    oof = pd.concat(parts, ignore_index=True)
    oof["cook_qty"] = np.ceil(oof["Res"].to_numpy() * (1.0 - oof["pred_q"].to_numpy()))
    oof["shortage"] = oof["cook_qty"] < oof["Recv"]
    oof["shortage_portions"] = np.maximum(oof["Recv"] - oof["cook_qty"], 0.0)
    oof["surplus"] = np.maximum(oof["cook_qty"] - oof["Recv"], 0.0)
    oof["surplus_b0"] = np.maximum(oof["Res"] - oof["Recv"], 0.0)
    oof["portions_saved"] = oof["surplus_b0"] - oof["surplus"]
    return oof


def annualize(oof: pd.DataFrame, n_boot: int = 1000, seed: int = 42) -> dict:
    n_days = oof[DATE_COL].nunique()
    rows_per_day = len(oof) / n_days
    rows_per_year = rows_per_day * 7 * ACADEMIC_WEEKS_PER_YEAR

    def stat(sample: pd.DataFrame) -> float:
        return float(sample["portions_saved"].mean())

    point, lo, hi = block_bootstrap_2d(oof, stat, day_col=DATE_COL, unit_col="RestaurantName",
                                       n_boot=n_boot, seed=seed)
    # نرخ کمبود فقط *رخداد* را می‌شمارد: یک پرس کمبود و هزار پرس کمبود هر دو یک واحد.
    # دو معیار زیر اندازه‌ی کمبود را هم وارد می‌کنند — میانگین روزانه (عدد ملموس برای
    # مدیر سلف) و کسر تقاضای برآورده‌نشده = مکمل Fill Rate / سطح خدمت نوع دوم (بند ۶-۳
    # doc/model-evaluation-metrics.md).
    daily_shortage = oof.groupby(DATE_COL)["shortage_portions"].sum()
    # میانگین روزانه ناهار و شام را با هم جمع می‌کند و برای مدیر سلف گمراه‌کننده است:
    # تصمیم پخت وعده‌به‌وعده گرفته می‌شود. تجمیع (روز، وعده) روی مجموع همه‌ی سلف‌ها،
    # واحدی است که واقعاً با یک تصمیم پخت متناظر است.
    meal_shortage = oof.groupby([DATE_COL, "Meal"])["shortage_portions"].sum()
    unmet_rate = float(oof["shortage_portions"].sum() / oof["Recv"].sum())
    return {
        "shortage_rate": float(oof["shortage"].mean()),
        "shortage_portions_per_day": float(daily_shortage.mean()),
        "worst_day_shortage_portions": float(daily_shortage.max()),
        "shortage_portions_per_meal": float(meal_shortage.mean()),
        "median_shortage_portions_per_meal": float(meal_shortage.median()),
        "worst_meal_shortage_portions": float(meal_shortage.max()),
        "mean_shortage_depth": float(oof.loc[oof["shortage"], "shortage_portions"].mean()),
        "unmet_demand_rate": unmet_rate,
        "fill_rate": 1.0 - unmet_rate,
        "n_oof_days": n_days, "rows_per_day": rows_per_day, "rows_per_year_assumed": rows_per_year,
        "per_row_portions_saved": point, "per_row_ci_lo": lo, "per_row_ci_hi": hi,
        "annual_portions_saved": point * rows_per_year,
        "annual_portions_saved_lo": lo * rows_per_year,
        "annual_portions_saved_hi": hi * rows_per_year,
        "annual_toman": point * rows_per_year * COST_PER_PORTION_TOMAN,
        "annual_toman_lo": lo * rows_per_year * COST_PER_PORTION_TOMAN,
        "annual_toman_hi": hi * rows_per_year * COST_PER_PORTION_TOMAN,
    }


def main() -> None:
    set_global_seed()
    scenario_rows = []
    detail = {}
    for name, tau in SCENARIOS.items():
        oof = oof_predictions(tau)
        stats = annualize(oof)
        detail[name] = stats
        scenario_rows.append({"scenario": name, "tau": tau, **stats})
    scen_df = pd.DataFrame(scenario_rows)
    n_days = int(scen_df["n_oof_days"].iloc[0])

    # --- بند ۱۰.۳: جدول سیاستی ---
    lines_103 = [
        "# بند ۱۰.۳ — سناریوهای سیاستی",
        "",
        "> پایه: OOF قهرمان (`lightgbm_quantile`) روی هر ۵ fold رسمی (رفتار «معمول»، نه "
        "پنجره‌ی غیرعادی Test بند ۸.۱). نرخ کمبود = کسر سلول‌ها با کمبود واقعی (فقط "
        "رخداد). کمبود روزانه = مجموع پرس کمبود در یک روز سرو، میانگین‌گرفته روی "
        f"{n_days} روز. کسر تقاضای برآورده‌نشده = مکمل Fill Rate (سطح خدمت نوع دوم) = "
        "مجموع پرس کمبود ÷ مجموع پرس دریافت‌شده — برخلاف نرخ کمبود، *اندازه*‌ی کمبود را "
        "وزن می‌دهد. صرفه‌جویی با قیمت "
        f"{COST_PER_PORTION_TOMAN:,} تومان به‌ازای هر پرس.",
        "",
        "| سناریو | τ | نرخ کمبود (رخداد) | کمبود روزانه (پرس) | کسر تقاضای برآورده‌نشده |"
        " صرفه‌جویی سالانه انتظاری (میلیارد تومان) | CI ۹۵٪ |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in scen_df.iterrows():
        lines_103.append(
            f"| **{r['scenario']}** | {r['tau']:.2f} | {r['shortage_rate']:.1%} | "
            f"{r['shortage_portions_per_day']:.0f} | {r['unmet_demand_rate']:.2%} | "
            f"{r['annual_toman']/1e9:.2f} | [{r['annual_toman_lo']/1e9:.2f}, {r['annual_toman_hi']/1e9:.2f}] |"
        )
    lines_103 += [
        "",
        "⚠️ اعداد این جدول **نقطه‌ای‌اند برای مقایسه‌ی سریع سه سیاست** — فرضیات کامل تعمیم "
        "سالانه، حساسیت آن‌ها، و صداقت روش‌شناختی لازم در بند ۱۰.۴ می‌آید؛ این دو بند را "
        "جدا از هم نخوانید.",
        "",
        f"**توصیه‌ی این پروژه (سازگار با نقطه‌ی عملیاتی τ=۰.۲۰، ردیف ۳۴ decision_log):** "
        "سناریوی **تهاجمی** — نرخ کمبود انتظاری آن هنوز طبق تعریف کوانتایل ≈τ است و "
        "بیشترین صرفه‌جویی را می‌دهد؛ انتخاب نهایی به تصمیم سیاستی مدیریت سلف درباره‌ی "
        "$C_u/C_o$ بستگی دارد (بند ۱۰.۲).",
    ]
    (OUT_DIR / "10.3_policy_scenarios.md").write_text("\n".join(lines_103) + "\n")
    scen_df.to_csv(OUT_DIR / "10.3_policy_scenarios.csv", index=False)

    # --- بند ۱۰.۴: تعمیم سالانه با عدم‌قطعیت ---
    op = detail["تهاجمی"]  # τ=0.20 نقطه‌ی عملیاتی
    lines_104 = [
        "# بند ۱۰.۴ — برآورد صرفه‌جویی سالانه با عدم‌قطعیت",
        "",
        "## فرضیات صریح (طبق دستور صداقت روش‌شناختی بند ۱۰.۴ WBS)",
        "",
        "1. **پایه:** پیش‌بینی OOF قهرمان روی هر ۵ fold رسمی فاز ۷ (۳٬۸۸۰ ردیف، ۸۰ روز "
        "سرو) — نه پنجره‌ی Test بند ۸.۱ که رژیم غیرعادی مستند دارد.",
        f"2. **تعمیم زمانی:** سال تحصیلی = **{ACADEMIC_WEEKS_PER_YEAR} هفته‌ی آموزشی** "
        "(دو نیم‌سال ۱۶هفته‌ای) — تعطیلات نوروز/تابستان بیرون گذاشته شدند چون الگوی سرو "
        "متفاوتی دارند که در این داده مشاهده نشده.",
        f"3. **تعمیم حجم:** میانگین {op['rows_per_day']:.1f} ردیف $(m,r,f)$ در هر روز "
        "سرو، از داده‌ی مشاهده‌شده — فرض می‌شود همین نرخ در طول سال ثابت بماند (بدون "
        "رشد/کاهش ثبت‌نام).",
        "4. **قیمت ثابت:** $C_o$=۱۲۰٬۰۰۰ تومان/پرس در طول سال ثابت فرض شده (بدون تورم).",
        "5. **الگوی هفتگی تکرارشونده:** رفتار مشاهده‌شده در ۸۰ روز fold‌های رسمی نماینده‌ی "
        "کل سال تحصیلی فرض شده — این محدودیت اصلی است؛ داده فقط ۵ ماه (آذر تا خرداد) "
        "را پوشش می‌دهد.",
        "",
        "## برآورد (نقطه‌ی عملیاتی τ=0.20، بوت‌استرپ بلوکی دوبعدی روز×سلف، ۱۰۰۰ تکرار)",
        "",
        f"- پرس صرفه‌جویی‌شده به‌ازای هر ردیف: {op['per_row_portions_saved']:.3f} "
        f"[CI: {op['per_row_ci_lo']:.3f}, {op['per_row_ci_hi']:.3f}]",
        f"- ردیف در سال (فرضی): {op['rows_per_year_assumed']:,.0f}",
        f"- **پرس صرفه‌جویی‌شده در سال: {op['annual_portions_saved']:,.0f} "
        f"[{op['annual_portions_saved_lo']:,.0f}, {op['annual_portions_saved_hi']:,.0f}]**",
        f"- **صرفه‌جویی ریالی سالانه: {op['annual_toman']/1e9:.2f} میلیارد تومان "
        f"[CI ۹۵٪: {op['annual_toman_lo']/1e9:.2f}, {op['annual_toman_hi']/1e9:.2f}]** "
        f"(با {COST_PER_PORTION_TOMAN:,} تومان به‌ازای هر پرس)",
        f"- هزینه‌ی طرف مقابل، در سه مقیاس: عمق میانگین هر سلول کمبوددار "
        f"{op['mean_shortage_depth']:.1f} پرس؛ مجموع کمبود همه‌ی سلف‌ها **در یک وعده** "
        f"به‌طور میانگین {op['shortage_portions_per_meal']:.0f} پرس، میانه "
        f"{op['median_shortage_portions_per_meal']:.0f} پرس، بیشینه‌ی کل دوره "
        f"{op['worst_meal_shortage_portions']:.0f} پرس؛ و در کل دوره کسر تقاضای "
        f"برآورده‌نشده‌ی {op['unmet_demand_rate']:.2%} (Fill Rate = {op['fill_rate']:.2%}). "
        f"نرخ کمبود {op['shortage_rate']:.1%} سلول‌ها فقط *رخداد* را می‌شمارد. "
        f"⚠️ تجمیع روزانه ({op['shortage_portions_per_day']:.0f} پرس در یک روز سرو، "
        f"بدترین روز {op['worst_day_shortage_portions']:.0f} پرس) ناهار و شام را با هم "
        f"جمع می‌کند و با واحد تصمیم پخت متناظر نیست — در گزارش نهایی سطح وعده نقل شود",
        "",
        "## عبارت صادقانه (طبق الگوی دقیق بند ۱۰.۴ WBS)",
        "",
        f"> «بر اساس عملکرد قهرمان روی fold‌های رسمی اعتبارسنجی و با فرض تداوم الگوی "
        f"هفتگی در {ACADEMIC_WEEKS_PER_YEAR} هفته‌ی آموزشی سال، صرفه‌جویی سالانه در بازه‌ی "
        f"{op['annual_toman_lo']/1e9:.1f} تا {op['annual_toman_hi']/1e9:.1f} میلیارد تومان "
        "برآورد می‌شود؛ این برآورد به تابستان، تعطیلات، و رشد ثبت‌نام تعمیم‌پذیر نیست.»",
        "",
        "## مقایسه با پنجره‌ی Test (برای شفافیت وابستگی به رژیم)",
        "",
        "برای مقایسه، بند ۸.۵ روی همان τ=۰.۲۰ ولی پنجره‌ی Test (رژیم غیرعادی) عدد "
        "۲.۹۲ میلیارد تومان **در ۲۵ روز** داد (نه سالانه) — مقیاس این دو عدد قابل‌مقایسه "
        "نیست (پنجره‌ی متفاوت، بدون تعمیم زمانی)؛ اینجا صرفاً برای نشان‌دادن این‌که مبنای "
        "فولدهای رسمی و مبنای Test می‌توانند نتایج متفاوت بدهند نقل شد — دلیل دیگری که "
        "چرا بند ۱۰.۴ از fold‌های رسمی استفاده کرد، نه از Test.",
    ]
    (OUT_DIR / "10.4_annual_savings.md").write_text("\n".join(lines_104) + "\n")
    pd.DataFrame([op]).to_csv(OUT_DIR / "10.4_annual_estimate.csv", index=False)

    print("\n".join(lines_103))
    print()
    print("\n".join(lines_104))
    print(f"\nذخیره شد در {OUT_DIR}/10.3_policy_scenarios.md و {OUT_DIR}/10.4_annual_savings.md")


if __name__ == "__main__":
    main()
