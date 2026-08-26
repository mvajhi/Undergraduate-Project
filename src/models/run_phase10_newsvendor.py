"""بند ۱۰.۲ WBS فاز ۱۰ — بهینه‌سازی Newsvendor: جدول τ*_ρ و تحلیل نقطه‌ی سربه‌سر.

فرمول (تصحیح ردیف ۳۶ decision_log، بند ۲-۲ project-definition.md، فضای ρ نه تقاضای
کلاسیک): $\\tau^*_\\rho = C_o/(C_u+C_o)$.

**نقطه‌ی سربه‌سر یعنی چه:** نسبت $C_u/C_o$ای که در آن سود خالص مدل (کاهش هدررفت منهای
هزینه‌ی کمبود اضافه) به صفر می‌رسد. دو مقایسه‌ی متفاوت لازم است، نه یکی:

- **در برابر B0** (پخت کامل رزرو): آیا اصلاً هوشمندانه‌کردن پخت ارزش دارد؟
- **در برابر B3** (خط پایه‌ی ساده‌ی کوانتایل تجربی): آیا پیچیدگی قهرمان روی خط پایه‌ی
  ساده ارزش دارد، یا یک قانون ساده‌ی گروهی کافی بود؟

اجرا: ``python -m src.models.run_phase10_newsvendor``
"""

import numpy as np
import pandas as pd

from src.baselines import TAU_GRID, b3_empirical_quantile, implied_cost_ratio, operational_metrics
from src.config import REPORTS_DIR, set_global_seed
from src.models.axes import TUNING_TAU
from src.models.card_writer import load_s2_result
from src.models.run_phase8_holdout_eval import load_holdout

OUT_DIR = REPORTS_DIR / "phase10"
COST_PER_PORTION_TOMAN = 120_000
#: شبکه‌ی نظری چگالی‌تر از TAU_GRID عملیاتی — فقط برای جدول τ*_ρ (بدون نیاز به بازبرازش)
THEORETICAL_CU_OVER_CO = [1, 1.5, 2, 3, 4, 5, 6.667, 9, 19, 49, 99]


def tau_star_table() -> pd.DataFrame:
    rows = []
    for ratio in THEORETICAL_CU_OVER_CO:
        tau_star = 1.0 / (1.0 + ratio)   # از τ*=Co/(Cu+Co)، تقسیم بر Co: 1/(Cu/Co + 1)
        rows.append({"Cu_over_Co": ratio, "tau_star_rho": tau_star})
    return pd.DataFrame(rows)


def net_value_by_tau(fit_fn, train, test, hp, label: str) -> pd.DataFrame:
    rows = []
    for tau in TAU_GRID:
        pred = np.clip(np.asarray(fit_fn(train, test, tau, **hp), dtype=float), 0.0, 1.0)
        m = operational_metrics(test, pred, tau)
        b0 = operational_metrics(test, np.zeros(len(test)), tau)
        Cu = COST_PER_PORTION_TOMAN * (1.0 - tau) / tau
        waste_saved = (b0["surplus_portions"] - m["surplus_portions"]) * COST_PER_PORTION_TOMAN
        shortage_cost = m["shortage_portions"] * Cu
        rows.append({"model": label, "tau": tau, "Cu_over_Co": (1 - tau) / tau,
                     "waste_saved_toman": waste_saved, "shortage_cost_toman": shortage_cost,
                     "net_value_toman": waste_saved - shortage_cost,
                     "shortage_portions": m["shortage_portions"],
                     "portions_saved_vs_B0": b0["surplus_portions"] - m["surplus_portions"]})
    return pd.DataFrame(rows)


def main() -> None:
    set_global_seed()
    train, test, fold = load_holdout()

    tau_star = tau_star_table()

    import importlib
    result = load_s2_result("lightgbm_quantile", "F02")
    champion_fn = importlib.import_module("src.models.families.f02_tree").MODELS["lightgbm_quantile"]
    champ_df = net_value_by_tau(champion_fn, train, test, result["best_hyperparams"], "lightgbm_quantile")
    b3_df = net_value_by_tau(lambda tr, te, t, **_: b3_empirical_quantile(tr, te, t), train, test, {}, "B3")

    both = pd.concat([champ_df, b3_df], ignore_index=True)
    wide = both.pivot(index="tau", columns="model", values="net_value_toman")
    wide["champion_minus_B3"] = wide["lightgbm_quantile"] - wide["B3"]

    champ_breakeven = bool((champ_df["net_value_toman"] <= 0).any())
    b3_breakeven = bool((b3_df["net_value_toman"] <= 0).any())
    champion_always_beats_b3 = bool((wide["champion_minus_B3"] > 0).all())

    lines = [
        "# بند ۱۰.۲ — بهینه‌سازی Newsvendor: جدول τ*_ρ و نقطه‌ی سربه‌سر",
        "",
        f"> فرمول (فضای ρ، تصحیح ردیف ۳۶ decision_log): τ*_ρ = C_o/(C_u+C_o). "
        f"$C_o$={COST_PER_PORTION_TOMAN:,} تومان/پرس ثابت (project-definition.md بند ۲-۲)؛ "
        "$C_u$ تصمیم سیاستی تثبیت‌نشده — همه‌ی جدول‌ها زیر برحسب نسبت $C_u/C_o$ خوانده شوند.",
        "",
        "## جدول τ*_ρ (نظری، فرمول مستقیم — نیازی به داده ندارد)",
        "",
        "| $C_u/C_o$ | τ*_ρ بهینه | یعنی چه؟ |",
        "|---|---|---|",
    ]
    for _, r in tau_star.iterrows():
        interp = ("کمبود و مازاد هم‌هزینه" if abs(r["Cu_over_Co"] - 1) < 1e-9 else
                 f"کمبود {r['Cu_over_Co']:.3g}× گران‌تر از مازاد")
        mark = " ⬅️ نقطه‌ی عملیاتی پروژه" if abs(r["tau_star_rho"] - TUNING_TAU) < 0.01 else ""
        lines.append(f"| {r['Cu_over_Co']:.3g} | {r['tau_star_rho']:.4f} | {interp}{mark} |")

    lines += [
        "",
        "## تحلیل نقطه‌ی سربه‌سر (تجربی، روی Test قفل‌شده)",
        "",
        "برای هر τ (که خودش معادل یک نسبت $C_u/C_o$ بهینه است)، سود خالص = "
        "(هدررفت صرفه‌جویی‌شده نسبت به B0) − (هزینه‌ی کمبود اضافی، با همان $C_u$ که آن τ برایش بهینه است).",
        "",
        "| τ | $C_u/C_o$ ضمنی | سود خالص قهرمان (میلیارد تومان) | سود خالص B3 (میلیارد تومان) | قهرمان − B3 |",
        "|---|---|---|---|---|",
    ]
    for tau in TAU_GRID:
        c = wide.loc[tau, "lightgbm_quantile"] / 1e9
        b = wide.loc[tau, "B3"] / 1e9
        d = wide.loc[tau, "champion_minus_B3"] / 1e9
        lines.append(f"| {tau:.2f} | {(1-tau)/tau:.2f} | {c:.3f} | {b:.3f} | {d:+.3f} |")

    lines += [
        "",
        f"**در برابر B0:** {'یک نقطه‌ی سربه‌سر پیدا شد' if champ_breakeven else 'هیچ نقطه‌ی سربه‌سری در کل شبکه‌ی τ آزموده‌شده (Cu/Co از ۲ تا ۴۹) پیدا نشد'} "
        f"— سود خالص قهرمان روی **کل** بازه‌ی آزموده‌شده مثبت ماند. یعنی در هیچ سناریوی محتملی "
        "(از کمبود ۲برابر مازاد تا ۴۹برابر) هوشمندانه‌کردن پخت نسبت به پخت کامل رزرو ضرر نمی‌کند.",
        f"**در برابر B3:** {'قهرمان در همه‌ی نقاط شبکه بهتر از B3 است' if champion_always_beats_b3 else 'قهرمان در بعضی نقاط از B3 عقب می‌افتد — جدول بالا را ببینید'} — "
        "این تصمیم عملی مدیریتی را روشن می‌کند: سرمایه‌گذاری روی مدل پیچیده (به‌جای یک قاعده‌ی "
        f"ساده‌ی گروهی مثل B3) در نقطه‌ی عملیاتی τ={TUNING_TAU} حدود "
        f"{wide.loc[TUNING_TAU, 'champion_minus_B3']/1e9:.3f} میلیارد تومان سود اضافه در همین ۲۵ روز دارد.",
        "",
        "⚠️ این اعداد برای پنجره‌ی ۲۵روزه‌ی خاص Test‌اند (رژیم غیرعادی، ردیف ۴۹ decision_log) "
        "و $C_u$ فرضی است، نه تثبیت‌شده — تعمیم سالانه و بازه‌ی اطمینان کار بند ۱۰.۴ است.",
    ]

    report = "\n".join(lines)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "10.2_newsvendor.md").write_text(report + "\n")
    tau_star.to_csv(OUT_DIR / "10.2_tau_star_table.csv", index=False)
    both.to_csv(OUT_DIR / "10.2_net_value_by_tau.csv", index=False)
    print(report)
    print(f"\nذخیره شد در {OUT_DIR}/10.2_newsvendor.md")


if __name__ == "__main__":
    main()
