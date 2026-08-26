"""بند ۹.۵ WBS فاز ۹ — مدل جانشین قابل‌فهم (درخت تصمیم کم‌عمق روی پیش‌بینی‌های قهرمان).

هدف بند ۹.۵ صریح است: «چند قاعده‌ی ساده که به مدیر سلف قابل توضیح است» — و WBS
می‌گوید «این احتمالاً چیزی است که واقعاً در سلف استفاده خواهد شد». پس خروجی اصلی این
ماژول **قواعد به زبان کسب‌وکار** است، نه یک درخت آکادمیک.

⚠️ **جانشین، جایگزین نیست.** درخت روی **پیش‌بینی‌های قهرمان** برازش می‌شود (نه روی
واقعیت)، پس $R^2$ آن می‌گوید «چقدر از رفتار قهرمان را بازتولید می‌کند»، نه «چقدر خوب
پیش‌بینی می‌کند». هر دو عدد گزارش می‌شوند تا این تمایز گم نشود: وفاداری به قهرمان
(fidelity) و کیفیت واقعی خودِ درخت روی داده‌ی حقیقی.

قواعد در فضای **نرخ عدم‌دریافت $\\rho$** بیان می‌شوند و بعد به زبان عملیاتی («چند درصد
کمتر از رزرو بپز») ترجمه می‌شوند، چون تصمیم واقعی سلف همان است (بند ۱۰.۱).

اجرا: ``python -m src.models.run_phase9_surrogate``
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree

from src.baselines import pinball_loss
from src.config import FIGURES_DIR, REPORTS_DIR, set_global_seed
from src.eda_lib.figio import save_fig
from src.models.axes import TUNING_TAU
from src.models.phase9_common import fit_champion

OUT_DIR = REPORTS_DIR / "phase9"
FIG_DIR = FIGURES_DIR / "phase9"
TAU = TUNING_TAU
DEPTHS = [2, 3, 4]
CHOSEN_DEPTH = 3


def _surrogate_design(X: pd.DataFrame) -> pd.DataFrame:
    """ستون‌های `category` به کد عددی؛ درخت sklearn dtype رشته‌ای نمی‌پذیرد."""
    out = X.copy()
    for c in out.columns:
        if str(out[c].dtype) == "category":
            out[c] = out[c].cat.codes
    return out


def _rules_to_business(tree: DecisionTreeRegressor, feature_names: list[str],
                       X: pd.DataFrame) -> list[dict]:
    """هر برگ را به یک قاعده‌ی عملیاتی تبدیل می‌کند (شرط‌ها + عدد پخت پیشنهادی)."""
    t = tree.tree_
    leaf_of = tree.apply(X)
    rules = []

    def walk(node: int, conditions: list[str]) -> None:
        if t.children_left[node] == t.children_right[node]:  # برگ
            rho = float(t.value[node][0][0])
            n = int((leaf_of == node).sum())
            rules.append({"conditions": list(conditions), "rho_hat": rho, "n_test_rows": n,
                          "cook_pct_of_reservation": 100.0 * (1.0 - rho)})
            return
        feat = feature_names[t.feature[node]]
        thr = t.threshold[node]
        walk(t.children_left[node], conditions + [f"`{feat}` ≤ {thr:.3f}"])
        walk(t.children_right[node], conditions + [f"`{feat}` > {thr:.3f}"])

    walk(0, [])
    return sorted(rules, key=lambda r: -r["n_test_rows"])


def main() -> None:
    set_global_seed()
    model, Xtr, Xte, train, test, cols = fit_champion(TAU)

    champion_pred_tr = model.predict(Xtr)
    champion_pred_te = model.predict(Xte)
    Xtr_s, Xte_s = _surrogate_design(Xtr), _surrogate_design(Xte)

    depth_rows = []
    trees = {}
    for depth in DEPTHS:
        surrogate = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=50, random_state=42)
        surrogate.fit(Xtr_s, champion_pred_tr)          # ⬅️ روی پیش‌بینی قهرمان، نه واقعیت
        trees[depth] = surrogate
        sur_pred_te = surrogate.predict(Xte_s)
        depth_rows.append({
            "depth": depth,
            "n_leaves": int(surrogate.get_n_leaves()),
            # وفاداری: چقدر رفتار قهرمان را بازتولید می‌کند
            "fidelity_r2_vs_champion": float(r2_score(champion_pred_te, sur_pred_te)),
            # کیفیت واقعی: pinball خودِ درخت روی داده‌ی حقیقی
            "own_pinball_on_truth": float(pinball_loss(test["rho"].to_numpy(), sur_pred_te, TAU).mean()),
        })
    depth_df = pd.DataFrame(depth_rows)

    champ_pinball = float(pinball_loss(test["rho"].to_numpy(), champion_pred_te, TAU).mean())
    b3_pinball = float(pd.read_csv(OUT_DIR.parent / "phase8" / "8.1_holdout_metrics.csv",
                                   index_col=0).loc["B3_empirical_quantile", "pinball"])

    chosen = trees[CHOSEN_DEPTH]
    rules = _rules_to_business(chosen, cols, Xte_s)

    fig, ax = plt.subplots(figsize=(20, 9))
    plot_tree(chosen, feature_names=cols, filled=True, rounded=True, fontsize=7, ax=ax,
              precision=3, impurity=False)
    ax.set_title(f"Surrogate decision tree (depth={CHOSEN_DEPTH}) fitted on champion predictions")
    save_fig(fig, "9.5_surrogate_tree.png", FIG_DIR)
    plt.close(fig)

    tree_text = export_text(chosen, feature_names=cols, decimals=3)
    (OUT_DIR / "9.5_surrogate_tree.txt").write_text(tree_text + "\n")

    chosen_row = depth_df[depth_df["depth"] == CHOSEN_DEPTH].iloc[0]

    lines = [
        "# بند ۹.۵ — مدل جانشین قابل‌فهم",
        "",
        f"> درخت تصمیم کم‌عمق روی **پیش‌بینی‌های قهرمان** (`lightgbm_quantile`, τ={TAU}) "
        "برازش شده، نه روی واقعیت. آموزش روی همان train مرز holdout بند ۸.۱، ارزیابی روی Test.",
        "",
        "## انتخاب عمق",
        "",
        "| عمق | تعداد برگ | وفاداری به قهرمان (R² روی پیش‌بینی) | pinball خودِ درخت روی واقعیت |",
        "|---|---|---|---|",
    ]
    for _, r in depth_df.iterrows():
        mark = " ⬅️ انتخاب‌شده" if int(r["depth"]) == CHOSEN_DEPTH else ""
        lines.append(f"| {int(r['depth'])}{mark} | {int(r['n_leaves'])} | "
                     f"{r['fidelity_r2_vs_champion']:.3f} | {r['own_pinball_on_truth']:.5f} |")
    lines += [
        "",
        "⚠️ **دو ستون آخر دو چیز متفاوت‌اند و نباید قاطی شوند.** وفاداری می‌گوید درخت چقدر "
        "**رفتار قهرمان** را بازتولید می‌کند؛ ستون آخر می‌گوید خودِ درخت روی **واقعیت** چقدر "
        "خوب است. برای مقایسه: قهرمان روی همین Test "
        f"pinball={champ_pinball:.5f} و مرجع B3 {b3_pinball:.5f} دارد.",
        "",
        f"عمق {CHOSEN_DEPTH} انتخاب شد: وفاداری {chosen_row['fidelity_r2_vs_champion']:.3f} با فقط "
        f"{int(chosen_row['n_leaves'])} برگ — سقف بند ۹.۵ WBS (عمق ۳-۴) و همچنان قابل توضیح روی یک کاغذ.",
        "",
        "## قواعد استخراج‌شده (به ترتیب فراوانی در Test)",
        "",
        "| # | شرط‌ها | ρ̂ | پیشنهاد پخت | ردیف Test |",
        "|---|---|---|---|---|",
    ]
    for i, r in enumerate(rules, 1):
        cond = " **و** ".join(r["conditions"]) if r["conditions"] else "(بدون شرط)"
        lines.append(f"| {i} | {cond} | {r['rho_hat']:.4f} | "
                     f"{r['cook_pct_of_reservation']:.1f}٪ رزرو | {r['n_test_rows']} |")

    biggest = rules[0]
    span_hi = max(rules, key=lambda r: r["rho_hat"])
    span_lo = min(rules, key=lambda r: r["rho_hat"])
    used_features = [cols[f] for f in sorted(set(chosen.tree_.feature[chosen.tree_.feature >= 0]))]
    lines += [
        "",
        "## پیام قابل انتقال به مدیر سلف",
        "",
        f"- پرتکرارترین حالت ({biggest['n_test_rows']} از {len(Xte)} ردیف Test): "
        f"**{biggest['cook_pct_of_reservation']:.0f}٪ رزرو بپزید** "
        f"(یعنی حدود {100 - biggest['cook_pct_of_reservation']:.0f}٪ کمتر از کل رزرو).",
        f"- محافظه‌کارانه‌ترین قاعده (کمترین ریسک کمبود): {span_lo['cook_pct_of_reservation']:.0f}٪ رزرو · "
        f"تهاجمی‌ترین قاعده (بیشترین صرفه‌جویی): {span_hi['cook_pct_of_reservation']:.0f}٪ رزرو — "
        f"کل دامنه‌ی تصمیم فقط "
        f"{abs(span_hi['cook_pct_of_reservation'] - span_lo['cook_pct_of_reservation']):.0f} "
        "واحد درصد است.",
        "",
        f"🔎 **کدام فیچرها به قاعده تبدیل شدند؟** درخت جانشین روی "
        f"{'، '.join('`' + c + '`' for c in used_features)} split زد — یعنی **نه** روی "
        "`RestaurantName`/`dow_x_city` که در بند ۹.۱/۹.۲ دو فیچر برتر قهرمان بودند. دلیلش "
        "قابل‌فهم است: آن دو فیچر دسته‌ای پرسطح‌اند (۳۰ و ۴۲ سطح) و یک درخت عمق-۳ نمی‌تواند "
        "با ۷ split آن‌ها را مفید تقسیم کند، ولی فیچرهای نرخ-تاریخی پیوسته دقیقاً همان "
        "اطلاعات سلف را به شکل فشرده‌تر حمل می‌کنند (نرخ تاریخی هر سلول، خودش هویت سلف را "
        "در خود دارد). **پیام عملیاتی:** قاعده‌ی ساده لازم نیست نام سلف را بداند — کافی است "
        "نرخ عدم‌دریافت تاریخی همان سلف/روز را بداند.",
        "",
        "⚠️ **این قواعد جایگزین مدل نیستند.** آن‌ها تقریبی از رفتار قهرمان‌اند برای توضیح‌دادن و "
        "اعتمادسازی؛ عدد عملیاتی روزانه باید از خودِ مدل بیاید، نه از این جدول. "
        f"(pinball درخت روی واقعیت = {chosen_row['own_pinball_on_truth']:.5f} در برابر "
        f"قهرمان {champ_pinball:.5f}.)",
        "",
        "نمودار درخت: `reports/figures/phase9/9.5_surrogate_tree.png` · "
        "متن کامل درخت: `reports/phase9/9.5_surrogate_tree.txt`",
    ]

    report = "\n".join(lines)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "9.5_surrogate.md").write_text(report + "\n")
    depth_df.to_csv(OUT_DIR / "9.5_depth_selection.csv", index=False)
    pd.DataFrame([{**r, "conditions": " AND ".join(r["conditions"])} for r in rules]).to_csv(
        OUT_DIR / "9.5_rules.csv", index=False)
    print(report)
    print(f"\nذخیره شد در {OUT_DIR}/9.5_surrogate.md")


if __name__ == "__main__":
    main()
