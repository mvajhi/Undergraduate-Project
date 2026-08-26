"""بند ۹.۲ WBS فاز ۹ — تحلیل SHAP روی قهرمان (`lightgbm_quantile`)، مرز holdout بند ۸.۱.

TreeExplainer روی خروجی خام مدل (کوانتایل τ=۰.۲۰) — بدون فرض میانگین. چهار زیربخش:
Global (summary/bar)، Dependence (۵ فیچر برتر بند ۹.۱)، Local (waterfall چند مورد
جالب)، Interaction (تأیید تعامل‌های بند ۴.۹: `dow×restaurant_type`, `meal×restaurant_type`,
`dow×city`, `city×meal` — ردیف ۲۴ decision_log).

اجرا: ``python -m src.models.run_phase9_shap``
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.config import FIGURES_DIR, REPORTS_DIR, set_global_seed
from src.eda_lib.figio import save_fig
from src.models.axes import TUNING_TAU
from src.models.phase9_common import fit_champion
from src.viz_fa import fa
from src.viz_fa import setup as viz_setup

OUT_DIR = REPORTS_DIR / "phase9"
FIG_DIR = FIGURES_DIR / "phase9"
TAU = TUNING_TAU
TOP_N_DEPENDENCE = 5
INTERACTION_SAMPLE = 300
#: چهار تعامل تأییدشده‌ی بند ۴.۹ (ردیف ۲۴ decision_log) — نگاشت به نام ستون فیچرست
VALIDATED_INTERACTIONS = {
    "dow_x_type": ("dow", "RestaurantType", 56.2),
    "meal_x_type": ("Meal", "RestaurantType", 52.7),
    "dow_x_city": ("dow", "city", 20.7),
    "city_x_meal": ("city", "Meal", 19.4),
}


def _numeric_for_shap(X: pd.DataFrame) -> pd.DataFrame:
    """SHAP plot های استاندارد به کد عددی نیاز دارند، نه dtype category خام."""
    out = X.copy()
    for c in out.columns:
        if str(out[c].dtype) == "category":
            out[c] = out[c].cat.codes
    return out


def main() -> None:
    viz_setup()
    set_global_seed()
    model, Xtr, Xte, train, test, cols = fit_champion(TAU)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(Xte)
    Xte_num = _numeric_for_shap(Xte)

    mean_abs = pd.Series(np.abs(shap_values).mean(axis=0), index=cols).sort_values(ascending=False)

    # --- Global: beeswarm + bar ---
    fig = plt.figure(figsize=(9, 8))
    shap.summary_plot(shap_values, Xte_num, show=False, plot_size=None)
    save_fig(fig, "9.2_shap_beeswarm.png", FIG_DIR)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 7))
    shap.summary_plot(shap_values, Xte_num, plot_type="bar", show=False, plot_size=None)
    save_fig(fig, "9.2_shap_bar.png", FIG_DIR)
    plt.close(fig)

    # --- Dependence: ۵ فیچر برتر ---
    top5 = mean_abs.head(TOP_N_DEPENDENCE).index.tolist()
    dep_paths = []
    for feat in top5:
        fig = plt.figure(figsize=(6, 4))
        shap.dependence_plot(feat, shap_values, Xte_num, show=False)
        p = save_fig(plt.gcf(), f"9.2_dependence_{feat}.png", FIG_DIR)
        plt.close("all")
        dep_paths.append(p)

    # --- Local: waterfall ۳ مورد جالب ---
    interesting = {
        "بیشترین |SHAP| کل (بحرانی‌ترین پیش‌بینی)": int(np.argmax(np.abs(shap_values).sum(axis=1))),
        "میانه‌ی پیش‌بینی (روز معمولی)": int(np.argsort(model.predict(Xte))[len(Xte) // 2]),
    }
    holiday_idx = np.where(test["is_day_before_holiday"].to_numpy() == 1)[0]
    if len(holiday_idx):
        interesting["یک روز قبل‌تعطیل"] = int(holiday_idx[0])

    local_paths = {}
    exp = shap.Explanation(values=shap_values, base_values=np.full(len(Xte), explainer.expected_value),
                           data=Xte_num.to_numpy(), feature_names=cols)
    for label, idx in interesting.items():
        fig = plt.figure(figsize=(8, 6))
        shap.plots.waterfall(exp[idx], show=False)
        fname = f"9.2_waterfall_{idx}.png"
        p = save_fig(plt.gcf(), fname, FIG_DIR)
        plt.close("all")
        local_paths[label] = (idx, p)

    # --- Interaction values: تأیید تعامل‌های بند ۴.۹ ---
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(Xte), size=min(INTERACTION_SAMPLE, len(Xte)), replace=False)
    # shap_interaction_values نمی‌تواند ستون category خام (رشته‌ای) را cast به float کند —
    # کدهای عددی همان‌هایی‌اند که lightgbm خودش داخلی برای بیت‌ست دسته‌ای استفاده می‌کند
    Xte_sample = Xte_num.iloc[sample_idx]
    inter_vals = explainer.shap_interaction_values(Xte_sample)
    col_idx = {c: i for i, c in enumerate(cols)}

    inter_rows = []
    for engineered_col, (base_a, base_b, aic) in VALIDATED_INTERACTIONS.items():
        eng_importance = float(mean_abs.get(engineered_col, np.nan))
        base_a_imp = float(mean_abs.get(base_a, np.nan))
        base_b_imp = float(mean_abs.get(base_b, np.nan))
        inter_rows.append({"engineered_col": engineered_col, "base_pair": f"{base_a}×{base_b}",
                           "phase4_delta_aic": aic, "shap_importance_engineered_col": eng_importance,
                           "base_a_importance": base_a_imp, "base_b_importance": base_b_imp})
    inter_df = pd.DataFrame(inter_rows).sort_values("shap_importance_engineered_col", ascending=False)

    # قوی‌ترین تعامل‌هایی که مدل **واقعاً** استفاده می‌کند (بدون فرض قبلی از فاز ۴)
    offdiag = inter_vals.copy()
    for i in range(offdiag.shape[1]):
        offdiag[:, i, i] = 0.0
    mean_inter = np.abs(offdiag).mean(axis=0)
    top_pairs = sorted(
        ((cols[i], cols[j], float(mean_inter[i, j]))
         for i in range(len(cols)) for j in range(i + 1, len(cols))),
        key=lambda t: -t[2])[:8]

    lines = [
        "# بند ۹.۲ — تحلیل SHAP روی قهرمان (`lightgbm_quantile`)",
        "",
        f"> τ={TAU}. داده: مرز holdout بند ۸.۱ (Test، ۱٬۵۲۶ ردیف). TreeExplainer، خروجی خام کوانتایل.",
        "",
        "## Global",
        "",
        "beeswarm: `reports/figures/phase9/9.2_shap_beeswarm.png` · "
        "bar (میانگین |SHAP|): `reports/figures/phase9/9.2_shap_bar.png`",
        "",
        "| رتبه | فیچر | میانگین |SHAP| |",
        "|---|---|---|",
    ]
    for i, (feat, v) in enumerate(mean_abs.head(10).items(), 1):
        lines.append(f"| {i} | `{feat}` | {v:.5f} |")
    lines += ["", "## Dependence — ۵ فیچر برتر", ""]
    for feat, p in zip(top5, dep_paths):
        lines.append(f"- `{feat}`: `{p.relative_to(REPORTS_DIR.parent)}`")
    lines += ["", "## Local — waterfall", ""]
    for label, (idx, p) in local_paths.items():
        r = test.iloc[idx]
        lines.append(f"- **{label}** (ردیف {idx}، {r['RestaurantName']}، {r['Meal']}، "
                     f"{pd.Timestamp(r['date_gregorian']).date()}): `{p.relative_to(REPORTS_DIR.parent)}`")
    lines += [
        "",
        "## Interaction values — تأیید تعامل‌های بند ۴.۹ (ردیف ۲۴ decision_log)",
        "",
        f"محاسبه روی نمونه‌ی تصادفی {len(Xte_sample)} ردیف از Test (هزینه‌ی O(features²)).",
        "",
        "فاز ۴ چهار تعامل معنادار یافت و فاز ۵ هرکدام را به‌صورت **یک ستون مهندسی‌شده‌ی صریح** "
        "ساخت. پس سنجه‌ی درست اینجا اهمیت خودِ آن ستون است، نه SHAP-interaction بین دو ستون "
        "پایه — که همان‌طور که ستون‌های آخر نشان می‌دهند، مدل اصلاً روی بعضی ستون‌های پایه "
        "split نمی‌زند (اهمیت دقیقاً صفر).",
        "",
        "| ستون مهندسی‌شده | جفت پایه | ΔAIC فاز ۴ | اهمیت SHAP ستون مهندسی‌شده | اهمیت پایه‌ی اول | اهمیت پایه‌ی دوم |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in inter_df.iterrows():
        lines.append(f"| `{r['engineered_col']}` | {r['base_pair']} | {r['phase4_delta_aic']:.1f} | "
                     f"{r['shap_importance_engineered_col']:.5f} | {r['base_a_importance']:.5f} | "
                     f"{r['base_b_importance']:.5f} |")
    lines += [
        "",
        "🔎 **یافته:** `city`، `Meal` و `RestaurantType` خام اهمیت SHAP **دقیقاً صفر** دارند — "
        "مدل هرگز رویشان split نمی‌زند، چون `RestaurantName` (رتبه ۱) و ستون‌های تعاملی "
        "مهندسی‌شده همان اطلاعات را با وضوح بیشتر می‌دهند. یعنی تصمیم فاز ۵ (ساختن ستون "
        "تعاملی صریح به‌جای اتکا به کشف خودکار تعامل توسط درخت) واقعاً کار کرد. ⚠️ ولی رتبه‌بندی "
        "ΔAIC فاز ۴ با اهمیت واقعی در مدل نهایی **همخوان نیست**: `dow_x_type` بالاترین ΔAIC "
        "(۵۶.۲) را داشت ولی اهمیتش در مدل یک‌دهم `dow_x_city` (ΔAIC=۲۰.۷) است — معناداری آماری "
        "در یک مدل خطی سطح-جمعیت، سودمندی پیش‌بینی در یک مدل درختی را پیش‌بینی نمی‌کند.",
        "",
        "### قوی‌ترین تعامل‌هایی که مدل واقعاً استفاده می‌کند (بدون فرض قبلی)",
        "",
        "| جفت فیچر | میانگین |SHAP interaction| |",
        "|---|---|",
    ]
    for a, b, v in top_pairs:
        lines.append(f"| `{a}` × `{b}` | {v:.6f} |")
    lines.append("")
    lines.append("هر ۸ جفت برتر شامل `RestaurantName` یا `dow_x_city` است — یعنی ساختار تعاملی "
                 "واقعی مدل حول همان دو فیچر برتر بند ۹.۱ می‌چرخد، نه تعامل‌های جدید و ناشناخته.")

    report = "\n".join(lines)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "9.2_shap_analysis.md").write_text(report + "\n")
    mean_abs.to_csv(OUT_DIR / "9.2_shap_mean_abs.csv", header=["mean_abs_shap"])
    inter_df.to_csv(OUT_DIR / "9.2_interaction_validation.csv", index=False)
    print(report)
    print(f"\nذخیره شد در {OUT_DIR}/9.2_shap_analysis.md")


if __name__ == "__main__":
    main()
