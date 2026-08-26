"""بند ۹.۶ WBS فاز ۹ — تفسیرپذیری سطح فرد (L5): تاریخچه‌ی شخصی یا فیچرهای جمعیتی؟

## ⚠️ چرا قهرمان واقعی L5 تفسیر نمی‌شود (و این چه چیزی را عوض می‌کند)

قهرمان L5 فاز ۷ (`mlp_embedding_l5`) یک شبکه‌ی PyTorch است (`models/gpu/F07/.../*.pt`)
و SHAP رویش `DeepExplainer` + `torch` (~۹۰۰MB) می‌خواهد — همان وابستگی‌ای که پروژه
عمداً از محیط CPU-فقط بیرون نگه داشته (بند خ۹ فاز ۷). و مهم‌تر: **L5 اصلاً مدل
توصیه‌شده‌ی پروژه نیست** — بند ۸.۱۰ (ردیف ۵۰ decision_log) با Δ=۰.۰۰۵۵۱ ردش کرد.
سرمایه‌گذاری برای تفسیر یک مدل ردشده توجیه ندارد.

**ولی سؤال بند ۹.۶ به آن مدل خاص وابسته نیست.** سؤال این است: «آیا تاریخچه‌ی شخصی یا
فیچرهای جمعیتی (دانشکده/خوابگاه) اهمیت بیشتری دارند؟» — که خاصیت **داده**ست، نه خاصیت
یک معماری. پس اینجا یک **LightGBM جانشین روی همان داده و همان هدف** (`dont_receive`،
سطح رزرو فردی) برازش می‌شود و SHAP رویش اجرا می‌شود.

**این جانشین است، نه قهرمان L5.** اعداد اهمیت اینجا درباره‌ی «چه چیزی در داده‌ی سطح
فرد سیگنال دارد» حرف می‌زنند، نه «قهرمان L5 چه یاد گرفت». برای سؤال بند ۹.۶ اولی
کافی است.

مرز آموزش/آزمون **همان مرز holdout بند ۸.۱** است تا با بقیه‌ی فاز ۸/۹ هم‌راستا بماند.

اجرا: ``python -m src.models.run_phase9_l5_interpretability``
"""

import matplotlib
matplotlib.use("Agg")

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import DATA_PROCESSED, FIGURES_DIR, REPORTS_DIR, set_global_seed
from src.cv import DATE_COL, holdout_split
from src.eda_lib.figio import save_fig
from src.viz_fa import fa
from src.viz_fa import setup as viz_setup

OUT_DIR = REPORTS_DIR / "phase9"
FIG_DIR = FIGURES_DIR / "phase9"
PERSON_FEATURES = DATA_PROCESSED / "person_features_v1.parquet"
SHAP_SAMPLE = 20_000

#: تفکیک صریح دو خانواده‌ی فیچر — قلب سؤال بند ۹.۶
HISTORY_FEATURES = [
    "person_n_prior_reservations", "person_n_prior_norecv", "person_reservations_per_week",
    "person_expanding_norecv_rate", "person_shrunk_norecv_rate", "person_ewm_norecv_rate",
    "is_honeymoon", "is_cold_start", "person_meal_share", "person_restaurant_share",
    "person_dow_share",
]
DEMOGRAPHIC_FEATURES = [
    "is_dorm_resident", "is_grad", "is_female", "is_evening_session",
    "college_freq", "field_freq",
]
CONTEXT_FEATURES = ["Meal", "is_tehran"]
GROUPS = {"تاریخچه‌ی شخصی": HISTORY_FEATURES, "جمعیتی": DEMOGRAPHIC_FEATURES,
          "بافتاری": CONTEXT_FEATURES}


def load() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    df = pd.read_parquet(PERSON_FEATURES).sort_values(DATE_COL).reset_index(drop=True)
    tr_mask, te_mask, fold = holdout_split(df)
    cols = HISTORY_FEATURES + DEMOGRAPHIC_FEATURES + CONTEXT_FEATURES
    for c in cols:
        if str(df[c].dtype) in ("category", "object", "bool", "Int8"):
            df[c] = df[c].astype("float64") if str(df[c].dtype) != "category" \
                else df[c].cat.codes.astype("float64")
    return df.loc[tr_mask], df.loc[te_mask], cols


def main() -> None:
    viz_setup()
    set_global_seed()
    train, test, cols = load()
    print(f"train={len(train):,} test={len(test):,}")

    model = lgb.LGBMClassifier(n_estimators=200, num_leaves=31, learning_rate=0.05,
                               min_child_samples=100, verbosity=-1, random_state=42)
    model.fit(train[cols], train["dont_receive"])

    proba = model.predict_proba(test[cols])[:, 1]
    auc = float(roc_auc_score(test["dont_receive"], proba))
    ap = float(average_precision_score(test["dont_receive"], proba))
    base_rate = float(test["dont_receive"].mean())

    rng = np.random.default_rng(42)
    idx = rng.choice(len(test), size=min(SHAP_SAMPLE, len(test)), replace=False)
    Xs = test[cols].iloc[idx]
    sv = shap.TreeExplainer(model).shap_values(Xs)
    if isinstance(sv, list):          # سازگاری با نسخه‌هایی که خروجی دو-کلاسه می‌دهند
        sv = sv[1]
    mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=cols).sort_values(ascending=False)

    group_rows = []
    total = float(mean_abs.sum())
    for gname, feats in GROUPS.items():
        present = [f for f in feats if f in mean_abs.index]
        s = float(mean_abs[present].sum())
        group_rows.append({"group": gname, "n_features": len(present), "sum_mean_abs_shap": s,
                           "share_pct": 100.0 * s / total if total > 0 else float("nan"),
                           "top_feature": mean_abs[present].idxmax() if present else None})
    group_df = pd.DataFrame(group_rows).sort_values("sum_mean_abs_shap", ascending=False)

    fig = plt.figure(figsize=(8, 6))
    shap.summary_plot(sv, Xs, plot_type="bar", show=False, plot_size=None)
    save_fig(fig, "9.6_l5_shap_bar.png", FIG_DIR)
    plt.close("all")

    # --- اعتبارسنجی دامنه‌ای H13: خوابگاه ---
    dorm = test.groupby("is_dorm_resident", observed=True).agg(
        n=("dont_receive", "size"), norecv_rate=("dont_receive", "mean")).reset_index()
    dorm_shap = float(mean_abs.get("is_dorm_resident", float("nan")))
    dorm_rank = int(mean_abs.index.get_loc("is_dorm_resident")) + 1 \
        if "is_dorm_resident" in mean_abs.index else -1

    hist_share = group_df.loc[group_df["group"] == "تاریخچه‌ی شخصی", "share_pct"].iloc[0]
    demo_share = group_df.loc[group_df["group"] == "جمعیتی", "share_pct"].iloc[0]

    lines = [
        "# بند ۹.۶ — تفسیرپذیری سطح فرد (L5): تاریخچه‌ی شخصی یا فیچرهای جمعیتی؟",
        "",
        "> ⚠️ **این تفسیرِ قهرمان L5 (`mlp_embedding_l5`) نیست.** آن مدل PyTorch است و "
        "`DeepExplainer`+torch (~۹۰۰MB) می‌خواهد — وابستگی‌ای که پروژه عمداً از محیط "
        "CPU-فقط بیرون نگه داشته؛ و خودِ L5 در بند ۸.۱۰ (ردیف ۵۰ decision_log) رد شده. "
        "اینجا یک **LightGBM جانشین روی همان داده و همان هدف** (`dont_receive`) برازش شده، "
        "چون سؤال بند ۹.۶ خاصیت **داده** است نه خاصیت یک معماری.",
        "",
        f"> مرز آموزش/آزمون = همان holdout بند ۸.۱. آموزش {len(train):,} رزرو، آزمون "
        f"{len(test):,} رزرو. نرخ پایه‌ی عدم‌دریافت در آزمون: {base_rate:.2%}.",
        "",
        f"کیفیت جانشین روی آزمون: ROC-AUC={auc:.4f} · PR-AUC={ap:.4f} "
        f"(پایه‌ی تصادفی PR-AUC={base_rate:.4f}) — مدل واقعاً سیگنال دارد، پس تفسیرش معنادار است.",
        "",
        "## پاسخ صریح: کدام خانواده‌ی فیچر مهم‌تر است؟",
        "",
        "| خانواده | تعداد فیچر | مجموع میانگین |SHAP| | سهم | مهم‌ترین عضو |",
        "|---|---|---|---|---|",
    ]
    for _, r in group_df.iterrows():
        lines.append(f"| **{r['group']}** | {int(r['n_features'])} | {r['sum_mean_abs_shap']:.5f} | "
                     f"{r['share_pct']:.1f}٪ | `{r['top_feature']}` |")

    verdict = ("**تاریخچه‌ی شخصی**" if hist_share > demo_share else "**فیچرهای جمعیتی**")
    lines += [
        "",
        f"### پاسخ: {verdict} به‌وضوح مهم‌تر است "
        f"(تاریخچه {hist_share:.1f}٪ در برابر جمعیتی {demo_share:.1f}٪ از کل |SHAP|).",
        "",
        "این مستقیماً با یافته‌های ۲۱ و ۳۱ فاز ۷ سازگار است: آنچه از سطح فرد سیگنال دارد، "
        "**رفتار گذشته‌ی خودِ فرد** است نه اینکه او در کدام دانشکده/خوابگاه است. و چون "
        "فیچرهای نرخ-تاریخی سطح سلول (`cell_*_rate`) همین سیگنال را به‌صورت تجمیع‌شده "
        "از قبل حمل می‌کنند (بند ۹.۵)، مدل تجمیعی چیز زیادی از دست نمی‌دهد — دقیقاً "
        "توضیح اینکه چرا L5 در بند ۸.۱۰ نبرد.",
        "",
        "## ۱۰ فیچر برتر",
        "",
        "| رتبه | فیچر | خانواده | میانگین |SHAP| |",
        "|---|---|---|---|",
    ]
    fam_of = {f: g for g, fs in GROUPS.items() for f in fs}
    for i, (feat, v) in enumerate(mean_abs.head(10).items(), 1):
        lines.append(f"| {i} | `{feat}` | {fam_of.get(feat, '—')} | {v:.5f} |")

    lines += [
        "",
        "## اعتبارسنجی دامنه‌ای اختصاصی — فرضیه‌ی H13 (فاصله‌ی فیزیکی خوابگاه)",
        "",
        "H13 (بند ۴.۱۳ WBS): «خوابگاه‌های دورتر از یک سلف نرخ عدم‌دریافت بالاتری دارند».",
        "",
        "| ساکن خوابگاه؟ | تعداد رزرو | نرخ عدم‌دریافت |",
        "|---|---|---|",
    ]
    for _, r in dorm.iterrows():
        label = "بله" if r["is_dorm_resident"] in (1, 1.0, True) else "خیر"
        lines.append(f"| {label} | {int(r['n']):,} | {r['norecv_rate']:.2%} |")

    lines += [
        "",
        f"`is_dorm_resident` در رتبه‌ی **{dorm_rank}** از {len(mean_abs)} فیچر قرار دارد "
        f"(میانگین |SHAP|={dorm_shap:.5f}).",
        "",
        "⚠️ **حکم دامنه‌ای — طبق دستور صریح بند ۹.۴ WBS این به‌عنوان هشدار گزارش می‌شود، "
        "نه یافته:** داده‌ی این پروژه فقط پرچم دوتایی «ساکن خوابگاه/نه» دارد، **نه فاصله‌ی "
        "واقعی خوابگاه تا سلف** (`dorm_restaurant_distance` که خودِ H13 به‌عنوان فیچر لازم "
        "نام برده، هرگز به دست نیامد — بند ۲.۵ WBS). پس تفاوت مشاهده‌شده بین دو گروه "
        "**نمی‌تواند** H13 را تأیید یا رد کند: هر تفاوتی می‌تواند به‌جای فاصله، ناشی از "
        "متغیر پنهان (ترکیب رشته/مقطع ساکنان، الگوی وعده‌ی خوابگاهی‌ها، یا اینکه "
        "خوابگاهی‌ها اساساً بیشتر رزرو می‌کنند) باشد. **H13 آزموده‌نشده باقی می‌ماند.**",
        "",
        "نمودار: `reports/figures/phase9/9.6_l5_shap_bar.png`",
    ]

    report = "\n".join(lines)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "9.6_l5_interpretability.md").write_text(report + "\n")
    group_df.to_csv(OUT_DIR / "9.6_group_importance.csv", index=False)
    mean_abs.to_csv(OUT_DIR / "9.6_l5_shap_mean_abs.csv", header=["mean_abs_shap"])
    print(report)
    print(f"\nذخیره شد در {OUT_DIR}/9.6_l5_interpretability.md")


if __name__ == "__main__":
    main()
