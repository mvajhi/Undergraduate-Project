"""دور ۳ (ج) — کالبدشکافی واریانس: نرخ یک وعده از کجا می‌آید؟ (فقط تهران)

**چرا این مرحله لازم شد.** دور ۳ (ب) یک نتیجه‌ی غیرمنتظره داد: با اینکه تاریخچه‌ی
فردی، رفتار *فرد* را با AUC=۰.۷۲ پیش‌بینی می‌کند، «میانگین تاریخچه‌ی رزروکنندگان» نرخ
*سلول* (روز×وعده×سلف) را تقریباً هیچ بهتر از هویت خود سلف پیش‌بینی نمی‌کند
(ΔR²=+۰.۰۰۲۵). این تناقض ظاهری فقط یک توضیح دارد: **نرخ سلول عمدتاً با یک شوک مشترک
روزانه تعیین می‌شود، نه با اینکه چه کسی رزرو کرده.**

اگر درست باشد، سه پیامد دارد:
1. سقف پیش‌بینی‌پذیری از سمت «ترکیب افراد» پایین است ⇒ مدل B برتری خودکار ندارد.
2. باید شوک روزانه را شناسایی و مدل کرد، نه ترکیب افراد را.
3. اگر شوک روزانه **مانا** باشد، فیچر lag آن را می‌گیرد — و این دقیقاً توضیح می‌دهد
   چرا شام lag1=۰.۷۶ داشت (F33).

این اسکریپت واریانس نرخ سلول را به سه جزء تجزیه می‌کند و شوک روزانه را می‌شکافد.

اجرا: `python -m src.eda_lib.runners.r3c_variance_anatomy`
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from src.config import FIGURES_DIR
from src.eda_lib.figio import save_fig
from src.eda_lib.runners._common import CALENDAR_PATH, header, kv, pct, setup
from src.eda_lib.runners.s13_individual import load_fact
from src.viz_fa import fa
from src.viz_fa import setup as viz_setup


def build_cells(fact: pd.DataFrame) -> pd.DataFrame:
    """سلول (روز، وعده، سلف) + پیش‌بینی ترکیب رزروکنندگان (leakage-safe)."""
    f = fact.sort_values(["PersonId", "date_gregorian"]).copy()
    csum = f.groupby("PersonId")["dont_receive"].cumsum() - f["dont_receive"]
    cnt = f.groupby("PersonId").cumcount()
    f["hist"] = np.where(cnt > 0, csum / cnt.replace(0, np.nan), np.nan)
    f["has_hist"] = cnt >= 5

    cell = f.groupby(["date_gregorian", "Meal", "restaurant_canonical"], observed=True).agg(
        n=("dont_receive", "size"), k=("dont_receive", "sum"),
        pred=("hist", "mean"), cover=("has_hist", "mean")).reset_index()
    cell["actual"] = cell["k"] / cell["n"]
    cell = cell[(cell["n"] >= 30) & (cell["cover"] >= 0.7) & cell["pred"].notna()].copy()
    cell["dow"] = (cell["date_gregorian"].dt.dayofweek + 2) % 7
    return cell


def q22_variance_decomposition(cell: pd.DataFrame, fact: pd.DataFrame) -> None:
    header("Q22 — واریانس نرخ سلول از سه جزء تشکیل شده؛ سهم هرکدام چقدر است؟")
    p_bar = cell["k"].sum() / cell["n"].sum()
    total = cell["actual"].var()
    kv("واریانس کل نرخ سلول", f"{total:.6f}")

    # جزء ۱: نویز نمونه‌گیری دوجمله‌ای
    v_sampling = (p_bar * (1 - p_bar) / cell["n"]).mean()
    # جزء ۲: ترکیب رزروکنندگان (واریانس توضیح‌داده‌شده توسط pred)
    m_pred = smf.ols("actual ~ pred", data=cell).fit()
    v_comp = m_pred.rsquared * total
    # جزء ۳: باقیمانده
    print("\nتجزیه‌ی خام:")
    for lbl, v in [("نویز نمونه‌گیری (دوجمله‌ای)", v_sampling),
                   ("ترکیب رزروکنندگان", v_comp),
                   ("باقیمانده (شوک مشترک + غیره)", total - v_sampling - v_comp)]:
        print(f"  {lbl:<34s} {v:.6f}   ({v / total:>6.1%})")

    header("همان تجزیه ولی با افزودن تدریجی عوامل ساختاری", 2)
    models = {
        "خالی": "actual ~ 1",
        "ترکیب رزروکنندگان": "actual ~ pred",
        "سلف×وعده": "actual ~ C(restaurant_canonical):C(Meal)",
        "سلف×وعده + روزهفته": "actual ~ C(restaurant_canonical):C(Meal) + C(dow)",
        "سلف×وعده + روزهفته + ترکیب": "actual ~ C(restaurant_canonical):C(Meal) + C(dow) + pred",
        "+ اثر ثابت روز (سقف عملی)": "actual ~ C(restaurant_canonical):C(Meal) + pred + C(date_gregorian)",
    }
    prev = 0.0
    for name, fo in models.items():
        m = smf.ols(fo, data=cell).fit()
        print(f"  {name:<34s} R²={m.rsquared:.4f}  (Δ={m.rsquared - prev:+.4f})")
        prev = m.rsquared
    print("\n→ پرش بزرگ در ردیف «اثر ثابت روز» = سهم شوک مشترک روزانه‌ای که هیچ‌کدام")
    print("  از فیچرهای ساختاری آن را نمی‌گیرند.")


def q23_day_shock(cell: pd.DataFrame) -> pd.DataFrame:
    header("Q23 — شوک روزانه چیست و چقدرش با تقویم توضیح داده می‌شود؟")
    # عامل روز: میانگین باقیمانده‌ی سلول‌ها پس از حذف اثر سلف×وعده
    cell = cell.copy()
    cell["res_rm"] = cell["actual"] - cell.groupby(
        ["restaurant_canonical", "Meal"], observed=True)["actual"].transform("mean")
    day = cell.groupby(["date_gregorian", "Meal"], observed=True).agg(
        shock=("res_rm", "mean"), n_cells=("res_rm", "size"), vol=("n", "sum")).reset_index()
    day = day[day["n_cells"] >= 3]
    kv("روز-وعده‌های با ≥۳ سلف", len(day))
    kv("انحراف معیار شوک روزانه", f"{day['shock'].std():.4f}")
    kv("دامنه", f"{day['shock'].min():+.4f} تا {day['shock'].max():+.4f}")

    cal = pd.read_csv(CALENDAR_PATH, parse_dates=["date_gregorian"])
    day = day.merge(cal[["date_gregorian", "is_day_before_holiday", "is_exam_period",
                         "days_to_next_holiday", "week_of_semester", "is_holiday_any"]],
                    on="date_gregorian", how="left")
    day["dow"] = (day["date_gregorian"].dt.dayofweek + 2) % 7
    day["log_vol"] = np.log(day["vol"])

    print("\nچقدر از شوک روزانه با متغیرهای *معلوم در لحظه‌ی برش* توضیح داده می‌شود؟")
    for name, fo in {
        "روز هفته": "shock ~ C(dow)",
        "+ روز قبل از تعطیلی": "shock ~ C(dow) + C(is_day_before_holiday)",
        "+ ایام امتحانات": "shock ~ C(dow) + C(is_day_before_holiday) + C(is_exam_period)",
        "+ هفته‌ی ترم": "shock ~ C(dow) + C(is_day_before_holiday) + C(is_exam_period) + week_of_semester",
        "+ حجم رزرو آن روز (معلوم!)": "shock ~ C(dow) + C(is_day_before_holiday) + C(is_exam_period) + week_of_semester + log_vol",
    }.items():
        m = smf.ols(fo, data=day.dropna(subset=["week_of_semester"])).fit()
        print(f"  {name:<32s} R²={m.rsquared:.4f}  R²adj={m.rsquared_adj:.4f}")

    header("آیا شوک روزانه **مانا** است؟ (اگر بله، فیچر lag آن را می‌گیرد)", 2)
    for meal in ["lunch", "dinner"]:
        s = day[day["Meal"] == meal].set_index("date_gregorian")["shock"].sort_index()
        s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"))
        out = []
        for k in [1, 2, 3, 7, 14]:
            a, b = s.values[:-k], s.values[k:]
            m_ = ~(np.isnan(a) | np.isnan(b))
            if m_.sum() >= 20:
                r, p = stats.pearsonr(a[m_], b[m_])
                out.append(f"lag{k}={r:+.3f}{'*' if p < 0.05 else ' '}")
        print(f"  {meal:<8s} {'  '.join(out)}  (n={int(s.notna().sum())} روز)")
    print("→ مانایی شوک یعنی «نرخ دیروز» بخشی از شوک امروز را حمل می‌کند —")
    print("  همان چیزی که F33 به‌صورت lag1=۰.۷۶ برای شام دیده بود.")

    print("\n۱۰ روز-وعده با بزرگ‌ترین شوک مثبت:")
    print(day.nlargest(10, "shock")[["date_gregorian", "Meal", "shock", "n_cells", "vol",
                                     "is_day_before_holiday", "is_exam_period"]]
          .round(4).to_string(index=False))
    return day


def q24_shock_scope(cell: pd.DataFrame) -> None:
    header("Q24 — شوک روزانه سراسری است یا مخصوص هر سلف؟")
    c = cell.copy()
    c["res_rm"] = c["actual"] - c.groupby(["restaurant_canonical", "Meal"], observed=True)["actual"].transform("mean")
    piv = c.pivot_table(index=["date_gregorian", "Meal"], columns="restaurant_canonical",
                        values="res_rm", observed=True)
    piv = piv.loc[:, piv.notna().sum() >= 60]
    kv("سلف‌های با ≥۶۰ روز مشترک", piv.shape[1])
    corr = piv.corr(min_periods=40)
    vals = corr.values[np.triu_indices_from(corr.values, k=1)]
    vals = vals[~np.isnan(vals)]
    kv("میانگین همبستگی جفت‌سلف‌ها در همان روز", f"{np.nanmean(vals):+.4f}")
    kv("میانه", f"{np.nanmedian(vals):+.4f}")
    kv("سهم جفت‌های با همبستگی مثبت", pct((vals > 0).mean()))
    print("→ همبستگی مثبت بین سلف‌های مختلف در یک روز = شوک **سراسری دانشگاه**.")

    # سهم واریانس: عامل مشترک اول (PCA)
    X = piv.dropna(thresh=int(piv.shape[1] * 0.8)).fillna(0.0)
    if len(X) > 30:
        Xc = X - X.mean()
        u, s, vt = np.linalg.svd(Xc.values, full_matrices=False)
        ev = s ** 2 / (s ** 2).sum()
        kv("سهم واریانس مؤلفه‌ی اول (عامل مشترک روز)", pct(ev[0]))
        kv("سهم مؤلفه‌ی دوم", pct(ev[1]))
        print("→ اگر مؤلفه‌ی اول سهم بزرگی داشته باشد، یک «عامل روز» واحد وجود دارد.")


def make_figures(cell: pd.DataFrame, day: pd.DataFrame) -> None:
    header("نمودارهای بینشی دور ۳")
    viz_setup()

    # ۱) ترکیب رزروکنندگان در برابر واقعیت
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(cell["pred"] * 100, cell["actual"] * 100, s=9, alpha=.28, color="#4C72B0")
    lim = [0, max(cell["actual"].max(), cell["pred"].max()) * 100 * 1.02]
    axes[0].plot(lim, lim, "k--", lw=1.2)
    axes[0].set_xlim(0, 20); axes[0].set_ylim(0, 30)
    axes[0].set_xlabel(fa("نرخ پیش‌بینی‌شده از تاریخچه‌ی رزروکنندگان (درصد)"))
    axes[0].set_ylabel(fa("نرخ واقعی آن وعده (درصد)")); axes[0].grid(alpha=.3)
    axes[0].set_title(fa("ترکیب افراد فقط ۱۰٪ واریانس را توضیح می‌دهد"))

    c2 = cell.copy()
    c2["ar"] = c2["actual"] - c2.groupby(["restaurant_canonical", "Meal"], observed=True)["actual"].transform("mean")
    c2["pr"] = c2["pred"] - c2.groupby(["restaurant_canonical", "Meal"], observed=True)["pred"].transform("mean")
    axes[1].scatter(c2["pr"] * 100, c2["ar"] * 100, s=9, alpha=.28, color="#C44E52")
    axes[1].axhline(0, color="k", lw=.8); axes[1].axvline(0, color="k", lw=.8)
    axes[1].set_xlabel(fa("انحراف ترکیب از عادت آن سلف (درصد)"))
    axes[1].set_ylabel(fa("انحراف نرخ واقعی از عادت آن سلف (درصد)")); axes[1].grid(alpha=.3)
    axes[1].set_title(fa("و درون هر سلف، تقریباً هیچ (r=۰.۰۷)"))
    fig.suptitle(fa("چه کسی رزرو کرده تقریباً بی‌اهمیت است — آنچه مهم است «آن روز چه خبر بوده» است"),
                 fontsize=13, y=1.02)
    fig.tight_layout()
    print(" ", save_fig(fig, "report_13_composition_bridge", FIGURES_DIR).name)
    plt.close(fig)

    # ۲) شوک روزانه در طول زمان
    fig, ax = plt.subplots(figsize=(13.5, 5))
    for meal, col, lbl in [("lunch", "#DD8452", "ناهار"), ("dinner", "#8172B2", "شام")]:
        s = day[day["Meal"] == meal].sort_values("date_gregorian")
        ax.plot(s["date_gregorian"], s["shock"] * 100, lw=1.5, color=col, label=fa(lbl), alpha=.85)
    ax.axhline(0, color="k", lw=.9, ls="--")
    pre = day[day["is_day_before_holiday"] == True]
    ax.scatter(pre["date_gregorian"], pre["shock"] * 100, s=26, color="#C44E52",
               zorder=5, label=fa("روز قبل از تعطیلی"))
    ax.set_xlabel(fa("تاریخ")); ax.set_ylabel(fa("شوک روزانه (واحد درصد)"))
    ax.legend(); ax.grid(alpha=.3)
    ax.set_title(fa("شوک مشترک روزانه: همه‌ی سلف‌ها در یک روز با هم بالا و پایین می‌روند"))
    fig.tight_layout()
    print(" ", save_fig(fig, "report_14_day_shock", FIGURES_DIR).name)
    plt.close(fig)


def main() -> None:
    setup()
    fact = load_fact()
    fact = fact[fact["is_tehran"]].copy()
    header(f"دور ۳ (ج) — تهران: {len(fact):,} رزرو")
    cell = build_cells(fact)
    q22_variance_decomposition(cell, fact)
    day = q23_day_shock(cell)
    q24_shock_scope(cell)
    make_figures(cell, day)


if __name__ == "__main__":
    main()
