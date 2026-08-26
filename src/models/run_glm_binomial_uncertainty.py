"""بند 7.10.1 عضو ۱۴ — `glm_binomial` رد‌شده است (F07: بیش‌پراکندگی صعودی ۳.۷×→۱۵.۶×)
ولی «یک‌بار اجرا و گزارش می‌شود تا کم‌برآورد عدم‌قطعیت کمّی گردد» (`doc/WBS-food-demand-
forecasting.md`). S0 (`reports/phase7/S0_feasibility.md`) فقط pinball/زمان fold۰ را داد؛
این ماژول همان کمّی‌سازی صریح را می‌دهد که آن بند خواسته بود.

## تفاوت با F07 اصلی (`s08_variance_interaction.py`، فاز ۴)

F07 با نرخ **سراسری** $\\bar p$ (یک عدد برای کل داده) واریانس نظری دوجمله‌ای را حساب کرد.
اینجا واریانس نظری از $\\hat\\mu(x)$ **خودِ مدل** (پیش‌بینی OOF روی ۵ fold رسمی) می‌آید —
پس این آزمون مستقیماً می‌گوید «فاصله‌ی اطمینان خودِ `glm_binomial` چقدر کوچک‌تر از واقعیت
است»، نه فقط اینکه «داده به‌طور کلی بیش‌پراکنده است» (که فاز ۴ از قبل اثبات کرده بود).

## یافته‌ی دوم، غیرمنتظره: پوشش تجمیعی گمراه‌کننده است

پوشش سراسری `glm_binomial` (۲۳.۵٪ در برابر τ=۰.۲۰) نزدیک به‌نظر می‌رسد و اصلاً فاجعه‌بار
نیست — ولی این عدد خطای مدل میانگین (لینک لاجیت) و کم‌برآوردی واریانس را باهم قاطی
می‌کند و همدیگر را تا حدی خنثی می‌کنند. تفکیک به چارک Res پوشش واقعی را نشان می‌دهد:
سلول‌های کم‌حجم به‌شدت پوش‌بیش (over-covered)، سلول‌های پرحجم تقریباً درست. **نسبت
واریانس مشاهده‌شده/ضمنی** (که مستقیماً فقط جزء واریانس را می‌سنجد، نه میانگین) الگوی
واضح‌تر و یکنواخت‌تری می‌دهد و مستقیماً F07 را با شاهد سطح-مدل تأیید می‌کند.

اجرا: ``python -m src.models.run_glm_binomial_uncertainty``
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.baselines import operational_metrics
from src.cv import DATE_COL, load_cv_folds
from src.features.build import FEATURES_A_PATH
from src.models.axes import TUNING_TAU
from src.models.families.f01_linear import _add_const, _design

#: همان مرزبندی F07 اصلی (`s08_variance_interaction.py`) — برای مقایسه‌ی مستقیم
RES_BIN_EDGES = [0, 20, 50, 100, 200, 400, 800, 2000]
RES_BIN_LABELS = ["<20", "20-50", "50-100", "100-200", "200-400", "400-800", ">800"]


def _official_folds() -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    df = pd.read_parquet(FEATURES_A_PATH).sort_values(DATE_COL).reset_index(drop=True)
    fold_meta, _ = load_cv_folds()
    return [(df.loc[m1], df.loc[m2]) for f in fold_meta for m1, m2 in [f.masks(df[DATE_COL])]]


def oof_binomial_predictions(folds: list) -> pd.DataFrame:
    """برازش `sm.GLM(Binomial)` روی هر ۵ fold رسمی (بدون کوانتایل — فقط μ̂، تا
    واریانس ضمنی خودِ مدل به‌جای نرخ سراسری F07 محاسبه شود)."""
    parts = []
    for tr, te in folds:
        Xtr, Xte = _design(tr, te)
        Xtr_c, Xte_c = _add_const(Xtr, Xte)
        model = sm.GLM(tr["rho"].to_numpy(), Xtr_c, family=sm.families.Binomial(),
                       var_weights=tr["Res"].to_numpy()).fit()
        mu_te = np.clip(np.asarray(model.predict(Xte_c)), 1e-6, 1 - 1e-6)
        part = te[["Res", "Recv", "rho"]].copy()
        part["mu_hat"] = mu_te
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def variance_ratio_table(oof: pd.DataFrame) -> pd.DataFrame:
    d = oof.copy()
    d["res_bin"] = pd.cut(d["Res"], bins=RES_BIN_EDGES, labels=RES_BIN_LABELS)
    d["var_implied"] = d["mu_hat"] * (1 - d["mu_hat"]) / d["Res"]
    d["resid"] = d["rho"] - d["mu_hat"]

    rows = []
    for label, g in d.groupby("res_bin", observed=True):
        var_obs = float(g["resid"].var())
        var_imp = float(g["var_implied"].mean())
        rows.append({"res_bin": label, "n": len(g), "res_median": float(g["Res"].median()),
                    "var_observed": var_obs, "var_implied": var_imp,
                    "ratio": var_obs / var_imp if var_imp > 0 else float("nan")})
    return pd.DataFrame(rows)


def coverage_by_res_quartile(oof: pd.DataFrame, tau: float) -> pd.DataFrame:
    """پوشش (نه واریانس) به تفکیک چارک Res — نشان می‌دهد چرا پوشش سراسری به‌تنهایی
    گمراه‌کننده است (بند بالای ماژول)."""
    from src.models.families import common

    d = oof.copy()
    pred_q = common.binomial_normal_quantile(d["mu_hat"].to_numpy(), d["Res"].to_numpy(), tau)
    d["pred_q"] = np.clip(pred_q, 0.0, 1.0)
    d["res_q"] = pd.qcut(d["Res"], 4, labels=["Q1(کم)", "Q2", "Q3", "Q4(زیاد)"])
    rows = []
    for label, g in d.groupby("res_q", observed=True):
        cov = float((g["rho"] <= g["pred_q"]).mean())
        rows.append({"res_quartile": label, "n": len(g), "res_median": float(g["Res"].median()),
                    "coverage": cov, "gap": cov - tau})
    return pd.DataFrame(rows), d["pred_q"].to_numpy()


def render_report(ratio_tab: pd.DataFrame, cov_tab: pd.DataFrame, overall: dict, tau: float) -> str:
    lines = [
        "# کم‌برآورد عدم‌قطعیت `glm_binomial` — کمّی‌سازی صریح (پشتیبان F07)",
        "",
        f"> بند 7.10.1 عضو ۱۴. هر ۵ fold رسمی (OOF)، τ={tau}. مدل: `sm.GLM(Binomial, "
        "var_weights=Res)`، همان کد S0 (`f01_linear.py::fit_predict_glm_binomial`).",
        "",
        "## نسبت واریانس مشاهده‌شده به واریانس ضمنی مدل (به تفکیک اندازه‌ی رزرو)",
        "",
        "⭐ **همان آزمون F07، ولی با واریانس ضمنی خودِ `glm_binomial` (نه نرخ سراسری) —**",
        "شاهد سطح-مدل، نه فقط سطح-داده.",
        "",
        "| بسته‌ی Res | n | میانه‌ی Res | واریانس مشاهده‌شده (باقیمانده) | واریانس ضمنی مدل | نسبت |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in ratio_tab.iterrows():
        lines.append(f"| {r['res_bin']} | {int(r['n'])} | {r['res_median']:.0f} | "
                    f"{r['var_observed']:.6f} | {r['var_implied']:.6f} | **{r['ratio']:.2f}×** |")

    r_min, r_max = ratio_tab["ratio"].min(), ratio_tab["ratio"].max()
    lines += [
        "",
        f"**نسبت از {r_min:.1f}× (کوچک‌ترین بسته‌ی Res) تا {r_max:.1f}× (بزرگ‌ترین) صعود می‌کند** — "
        "همان الگوی صعودی F07 (۳.۷×→۱۵.۶×، با نرخ سراسری)، اینجا با دامنه‌ی وسیع‌تر و با "
        "واریانس ضمنی خودِ مدل. تأیید مستقیم: `glm_binomial` فاصله‌ی اطمینانش را به‌طور "
        "نظام‌مند کوچک‌تر از واقعیت می‌سازد، **بدتر برای سلف‌های پرحجم**.",
        "",
        "## پوشش تجمیعی در برابر پوشش شرطی — چرا عدد کلی گمراه‌کننده است",
        "",
        f"پوشش سراسری: **{overall['coverage']:.4f}** (شکاف از τ={tau}: {overall['coverage_gap']:+.4f}) — "
        "به‌نظر نزدیک می‌رسد، ولی خطای مدل میانگین (لینک لاجیت) و کم‌برآوردی واریانس را "
        "قاطی می‌کند. تفکیک به چارک Res:",
        "",
        "| چارک Res | n | میانه‌ی Res | پوشش | شکاف از τ |",
        "|---|---|---|---|---|",
    ]
    for _, r in cov_tab.iterrows():
        lines.append(f"| {r['res_quartile']} | {int(r['n'])} | {r['res_median']:.0f} | "
                    f"{r['coverage']:.4f} | {r['gap']:+.4f} |")

    worst = cov_tab.loc[cov_tab["gap"].abs().idxmax()]
    lines += [
        "",
        f"بدترین شکاف: **{worst['res_quartile']}** ({worst['gap']:+.4f}) — سلول‌های کم‌حجم "
        "به‌شدت پوش‌بیش می‌شوند (نه پوش‌کم، برخلاف انتظار ساده‌لوحانه‌ی «واریانس کوچک ⇐ "
        "شکاف منفی») چون تقسیم بر Res کوچک خودش واریانس ضمنی را بزرگ می‌کند؛ سلول‌های "
        "پرحجم که نسبت واریانس بدترین است (جدول بالا)، در پوشش خام تقریباً درست به‌نظر "
        "می‌رسند چون بازه‌شان همان‌قدر که کوچک است تصادفاً هم‌راستای خطای مدل میانگین "
        "می‌افتد — **پوشش تجمیعی/شرطی به‌تنهایی نمی‌تواند کم‌برآوردی واریانس را افشا "
        "کند؛ نسبت واریانس مستقیم (بخش بالا) شاهد قابل‌اعتمادتر است.**",
        "",
        "## نتیجه",
        "",
        "`glm_binomial` به دلیل مستندشده (F07) رد ماند: فرض دوجمله‌ای ساده — واریانس فقط "
        "از نمونه‌گیری تصادفی Recv/Res — نادرست است؛ بیش‌پراکندگی واقعی (تفاوت ذاتی بین "
        "رکوردها، نه فقط نویز نمونه‌گیری) وجود دارد و با اندازه‌ی رزرو تشدید می‌شود. "
        "Beta-Binomial/GLMM (خ۸، `bhm_*`) دقیقاً برای همین ساخته شدند — و برخلاف "
        "`glm_binomial`، $\\phi$ (دیسپرسیون) را صریح مدل می‌کنند (`bhm_varying_dispersion` "
        "حتی $\\phi$ را تابع $\\log Res$ می‌کند، پاسخ مستقیم به همین یافته).",
    ]
    return "\n".join(lines)


def main() -> None:
    from src.config import REPORTS_DIR, set_global_seed

    set_global_seed()
    folds = _official_folds()
    oof = oof_binomial_predictions(folds)
    tau = TUNING_TAU

    ratio_tab = variance_ratio_table(oof)
    cov_tab, pred_q = coverage_by_res_quartile(oof, tau)
    overall = operational_metrics(oof, pred_q, tau)

    report = render_report(ratio_tab, cov_tab, overall, tau)
    out = REPORTS_DIR / "phase7"
    out.mkdir(parents=True, exist_ok=True)
    (out / "glm_binomial_uncertainty.md").write_text(report + "\n")
    ratio_tab.to_json(out / "glm_binomial_uncertainty_variance_ratio.json", orient="records",
                      indent=2, force_ascii=False)
    cov_tab.to_json(out / "glm_binomial_uncertainty_coverage.json", orient="records",
                    indent=2, force_ascii=False)
    print(report)
    print(f"\nذخیره شد در {out / 'glm_binomial_uncertainty.md'}")


if __name__ == "__main__":
    main()
