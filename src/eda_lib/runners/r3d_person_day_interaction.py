"""دور ۳ (د) — آخرین لایه: تعامل فرد×روز، و سقف پیش‌بینی‌پذیری (فقط تهران).

دور ۳ (ج) نشان داد نرخ سلول را **شوک سراسری روز** تعیین می‌کند (اثر ثابت روز
ΔR²=+۰.۲۳۸)، نه ترکیب افراد (ΔR²=+۰.۰۰۲). این چهار سؤال باقی می‌ماند:

- Q25: آیا ترکیب افراد **جایی** مهم می‌شود؟ (سلول کوچک؟ وعده‌ی خاص؟ نوع سلف؟)
- Q26: سقف پیش‌بینی شوک روزانه با اطلاعات لحظه‌ی برش چقدر است؟ (اعتبارسنجی زمانی واقعی)
- Q27: **در سطح خودِ رزرو فردی**، سهم «فرد» در برابر «روز» چقدر است؟ این عدد مستقیماً
  می‌گوید مدل B نهایتاً چقدر می‌تواند از مدل A جلو بزند.
- Q28: **⭐ آیا شوک روز برای همه یکسان است؟** فرضیه: در روزهای بد، آدم‌های «بی‌ثبات»
  بیشتر می‌ریزند تا آدم‌های وفادار. اگر درست باشد، ترکیب افراد **غیرخطی** اثر دارد و
  دلیل اینکه میانگین ساده‌اش کار نکرد همین است.

اجرا: `python -m src.eda_lib.runners.r3d_person_day_interaction`
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from src.config import FIGURES_DIR
from src.eda_lib.figio import save_fig
from src.eda_lib.runners._common import CALENDAR_PATH, header, kv, pct, setup
from src.eda_lib.runners.r3c_variance_anatomy import build_cells
from src.eda_lib.runners.s13_individual import load_fact
from src.viz_fa import fa
from src.viz_fa import setup as viz_setup


def prep(fact: pd.DataFrame) -> pd.DataFrame:
    """تاریخچه‌ی انبساطی هر فرد (leakage-safe) + شوک روز."""
    f = fact.sort_values(["PersonId", "date_gregorian"]).copy()
    csum = f.groupby("PersonId")["dont_receive"].cumsum() - f["dont_receive"]
    cnt = f.groupby("PersonId").cumcount()
    f["hist"] = np.where(cnt >= 10, csum / cnt.replace(0, np.nan), np.nan)
    return f


def q25_where_composition_matters(cell: pd.DataFrame) -> None:
    header("Q25 — آیا ترکیب رزروکنندگان *جایی* مهم می‌شود؟")
    c = cell.copy()
    c["ar"] = c["actual"] - c.groupby(["restaurant_canonical", "Meal"], observed=True)["actual"].transform("mean")
    c["pr"] = c["pred"] - c.groupby(["restaurant_canonical", "Meal"], observed=True)["pred"].transform("mean")

    print("همبستگی درون-سلفیِ ترکیب با نرخ، به تفکیک زیرگروه:\n")
    rows = []
    c["size_bin"] = pd.qcut(c["n"], 3, labels=["سلول کوچک", "متوسط", "بزرگ"])
    for key, grouper in [("اندازه‌ی سلول", "size_bin"), ("وعده", "Meal")]:
        for lv, g in c.groupby(grouper, observed=True):
            if len(g) < 80:
                continue
            r = stats.pearsonr(g["pr"], g["ar"])
            rows.append({"بُعد": key, "گروه": lv, "n": len(g), "r": r[0], "p": r[1]})
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print("\n→ اگر همه‌جا نزدیک صفر بماند، ترکیب واقعاً اطلاعات پویا ندارد.")


def q26_predict_day_shock(cell: pd.DataFrame) -> None:
    header("Q26 — سقف پیش‌بینی شوک روزانه با اطلاعات لحظه‌ی برش (اعتبارسنجی زمانی)")
    c = cell.copy()
    c["res_rm"] = c["actual"] - c.groupby(["restaurant_canonical", "Meal"], observed=True)["actual"].transform("mean")
    day = (c.groupby(["date_gregorian", "Meal"], observed=True)
             .agg(shock=("res_rm", "mean"), vol=("n", "sum"), n_cells=("res_rm", "size"))
             .reset_index())
    day = day[day["n_cells"] >= 3].sort_values("date_gregorian")
    cal = pd.read_csv(CALENDAR_PATH, parse_dates=["date_gregorian"])
    day = day.merge(cal[["date_gregorian", "is_day_before_holiday", "is_exam_period",
                         "week_of_semester"]], on="date_gregorian", how="left")
    day["dow"] = (day["date_gregorian"].dt.dayofweek + 2) % 7
    day["log_vol"] = np.log(day["vol"])

    # lag شوک، آگاه به وعده و به قاعده‌ی برش (آخرین وعده‌ی *در دسترس*)
    day = day.sort_values(["Meal", "date_gregorian"])
    day["shock_lag1"] = day.groupby("Meal")["shock"].shift(1)
    day["shock_lag2"] = day.groupby("Meal")["shock"].shift(2)
    day = day.dropna(subset=["week_of_semester", "shock_lag1", "shock_lag2"])

    cut = day["date_gregorian"].quantile(0.75)
    tr, te = day[day["date_gregorian"] <= cut], day[day["date_gregorian"] > cut]
    kv("آموزش / آزمون (تقسیم زمانی)", f"{len(tr)} / {len(te)}")

    specs = {
        "تقویم (روزهفته+تعطیلی+امتحان)": "shock ~ C(dow) + C(is_day_before_holiday) + C(is_exam_period)",
        "+ حجم رزرو روز هدف": "shock ~ C(dow) + C(is_day_before_holiday) + C(is_exam_period) + log_vol",
        "+ شوک وعده‌ی قبلی (lag۱)": "shock ~ C(dow) + C(is_day_before_holiday) + C(is_exam_period) + log_vol + shock_lag1",
        "+ lag۲": "shock ~ C(dow) + C(is_day_before_holiday) + C(is_exam_period) + log_vol + shock_lag1 + shock_lag2",
    }
    base_mae = (te["shock"] - tr["shock"].mean()).abs().mean()
    kv("MAE خط پایه (میانگین آموزش)", f"{base_mae:.5f}")
    for name, fo in specs.items():
        m = smf.ols(fo, data=tr).fit()
        pred = m.predict(te)
        mae = (te["shock"] - pred).abs().mean()
        r2_oos = 1 - ((te["shock"] - pred) ** 2).sum() / ((te["shock"] - tr["shock"].mean()) ** 2).sum()
        print(f"  {name:<34s} MAE={mae:.5f} ({(1 - mae / base_mae):+.1%})  R²out={r2_oos:+.4f}")
    print("\n→ R²out مثبت یعنی شوک روزانه با اطلاعات مجازِ لحظه‌ی برش واقعاً قابل‌پیش‌بینی است.")


def q27_person_vs_day(f: pd.DataFrame) -> None:
    header("Q27 — در سطح خودِ رزرو: سهم «فرد» در برابر «روز» چقدر است؟")
    g = f.dropna(subset=["hist"]).copy()
    g["day_key"] = g["date_gregorian"].astype(str) + "|" + g["Meal"].astype(str)
    kv("رزروهای واجد (فرد با ≥۱۰ سابقه)", f"{len(g):,}")

    # سهم واریانس با اثر ثابت: R² هر مدل روی نتیجه‌ی باینری
    from sklearn.metrics import roc_auc_score
    y = g["dont_receive"].astype(int).values
    day_mean = g.groupby("day_key")["dont_receive"].transform("mean").values
    rest_mean = g.groupby(["restaurant_canonical", "Meal"], observed=True)["dont_receive"].transform("mean").values
    hist = g["hist"].values

    print("\nAUC تک‌متغیره روی نتیجه‌ی هر رزرو منفرد:")
    for name, x in [("تاریخچه‌ی شخص (leakage-safe)", hist),
                    ("میانگین همان روز-وعده (اوراکل)", day_mean),
                    ("میانگین همان سلف-وعده", rest_mean)]:
        print(f"  {name:<38s} AUC={roc_auc_score(y, x):.4f}")

    print("\nترکیب (رگرسیون لجستیک، اوراکلِ روز فقط برای سنجش سقف):")
    from sklearn.linear_model import LogisticRegression
    combos = {
        "فقط تاریخچه‌ی شخص": [hist],
        "فقط اثر روز (اوراکل)": [day_mean],
        "شخص + روز": [hist, day_mean],
        "شخص + روز + سلف": [hist, day_mean, rest_mean],
    }
    for name, xs in combos.items():
        X = np.column_stack(xs)
        lr = LogisticRegression(max_iter=300).fit(X, y)
        print(f"  {name:<28s} AUC={roc_auc_score(y, lr.predict_proba(X)[:, 1]):.4f}")
    print("\n→ «اثر روز» اینجا اوراکل است (از نتیجه‌ی واقعی همان روز ساخته شده) و فقط")
    print("  سقف را نشان می‌دهد. اینکه تاریخچه‌ی شخص چقدر به آن نزدیک است، می‌گوید")
    print("  مدل B چقدر از اطلاعاتِ در دسترسِ لحظه‌ی برش استخراج می‌کند.")


def q28_person_day_interaction(f: pd.DataFrame) -> pd.DataFrame:
    header("Q28 — ⭐ آیا شوک روز برای همه یکسان است، یا بی‌ثبات‌ها بیشتر می‌ریزند؟")
    g = f.dropna(subset=["hist"]).copy()
    g["day_key"] = g["date_gregorian"].astype(str) + "|" + g["Meal"].astype(str)
    day_rate = g.groupby("day_key")["dont_receive"].transform("mean")
    g["day_z"] = day_rate - day_rate.mean()
    g["h_bin"] = pd.qcut(g["hist"], 5, labels=["Q1 وفادارترین", "Q2", "Q3", "Q4", "Q5 بی‌ثبات‌ترین"],
                         duplicates="drop")

    print("نرخ عدم‌دریافت هر گروهِ تاریخچه، در روزهای عادی در برابر روزهای بد:\n")
    g["day_bin"] = pd.qcut(day_rate, [0, .5, .8, .95, 1.0],
                           labels=["روز عادی (۵۰٪ پایین)", "نسبتاً بد", "بد", "خیلی بد (۵٪ بالا)"],
                           duplicates="drop")
    t = g.pivot_table(index="h_bin", columns="day_bin", values="dont_receive",
                      aggfunc="mean", observed=True)
    print((t * 100).round(2).to_string())
    print("\nافزایش مطلق نرخ از «روز عادی» به «خیلی بد» برای هر گروه (واحد درصد):")
    delta = (t.iloc[:, -1] - t.iloc[:, 0]) * 100
    for k, v in delta.items():
        print(f"  {str(k):<18s} {v:+.2f}")
    print(f"\nنسبت افزایش بی‌ثبات‌ترین به وفادارترین: {delta.iloc[-1] / delta.iloc[0]:.2f}×")

    print("\nآزمون رسمی تعامل (رگرسیون خطی روی نتیجه‌ی باینری):")
    # `dont_receive` بولی است و patsy آن را دو-ستونی می‌کند؛ به float تبدیل می‌شود.
    g = g.assign(y=g["dont_receive"].astype(float))
    m0 = smf.ols("y ~ hist + day_z", data=g).fit()
    m1 = smf.ols("y ~ hist * day_z", data=g).fit()
    print(f"  بدون تعامل : R²={m0.rsquared:.5f}")
    print(f"  با تعامل   : R²={m1.rsquared:.5f}  ضریب تعامل={m1.params['hist:day_z']:+.4f} "
          f"(p={m1.pvalues['hist:day_z']:.3g})")
    print("→ ضریب تعامل مثبت یعنی در روزهای بد، شکاف بین وفادار و بی‌ثبات **بازتر** می‌شود")
    print("  ⇒ ترکیب افراد اثر **ضرب‌شونده** دارد، نه جمع‌شونده؛ و همین توضیح می‌دهد چرا")
    print("  میانگین ساده‌ی تاریخچه (Q18) در سطح سلول کار نکرد.")
    return g


def make_figures(g: pd.DataFrame, cell: pd.DataFrame) -> None:
    header("نمودار بینشی دور ۳ (د)")
    viz_setup()
    day_rate = g.groupby(g["date_gregorian"].astype(str) + "|" + g["Meal"].astype(str))["dont_receive"].transform("mean")
    g = g.assign(day_bin=pd.qcut(day_rate, [0, .5, .8, .95, 1.0],
                                 labels=["عادی", "نسبتاً بد", "بد", "خیلی بد"], duplicates="drop"))
    t = g.pivot_table(index="h_bin", columns="day_bin", values="dont_receive",
                      aggfunc="mean", observed=True) * 100
    fig, ax = plt.subplots(figsize=(10, 5.6))
    marks = ["o-", "s-", "^-", "d-"]
    for i, c in enumerate(t.columns):
        ax.plot(range(len(t)), t[c], marks[i % 4], lw=2.1, ms=8, label=fa(str(c)))
    ax.set_xticks(range(len(t)))
    ax.set_xticklabels([fa(str(i)) for i in t.index])
    ax.set_xlabel(fa("گروه‌بندی دانشجویان بر اساس تاریخچه‌ی شخصی"))
    ax.set_ylabel(fa("نرخ عدم‌دریافت (درصد)"))
    ax.legend(title=fa("وضعیت آن روز")); ax.grid(alpha=.3)
    ax.set_title(fa("در روزهای بد، دانشجویان بی‌ثبات چند برابر بیشتر می‌ریزند تا وفادارها"))
    fig.tight_layout()
    print(" ", save_fig(fig, "report_15_person_day_interaction", FIGURES_DIR).name)
    plt.close(fig)


def main() -> None:
    setup()
    fact = load_fact()
    fact = fact[fact["is_tehran"]].copy()
    header(f"دور ۳ (د) — تهران: {len(fact):,} رزرو")
    cell = build_cells(fact)
    f = prep(fact)
    q25_where_composition_matters(cell)
    q26_predict_day_shock(cell)
    q27_person_vs_day(f)
    g = q28_person_day_interaction(f)
    make_figures(g, cell)


if __name__ == "__main__":
    main()
