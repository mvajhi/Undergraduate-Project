"""دور ۵ / مرحله ۵ — سلف خوابگاهی در برابر دانشکده‌ای: کِی رفتارشان فرق می‌کند؟ (Q46–Q49)

**چرا این تحلیل لازم شد.** F41 (خوشه‌بندی بدون‌ناظر) خودش مرز خوابگاه/دانشکده را کشف کرد و
F12/F16 اختلاف *سطح* را ثبت کرده‌اند. ولی هیچ‌کس نپرسیده دو نوع سلف **کِی از هم جدا
می‌شوند** — یعنی کدام شرط تقویمی یکی را بالا و دیگری را پایین می‌برد. برای سیاست‌گذاری این
مهم‌تر از اختلاف سطح است: اختلاف سطح را با یک ثابت می‌شود جبران کرد، ولی اثر تقویمیِ
متفاوت یعنی **دو سیاست جدا** لازم است.

⚠️ **تله‌ی اصلی (F43 + بررسی این دور):** `RestaurantType` تقریباً با `Meal` هم‌خط است — و
هر ۴ سلف دوگانه دقیقاً ناهار=دانشکده‌ای / شام=خوابگاهی‌اند. پس هر مقایسه‌ای که وعده را
کنترل نکند، در واقع **اثر وعده** را اندازه می‌گیرد و اسمش را «اثر نوع سلف» می‌گذارد.
Q46 پیش از هر چیز همین را کمّی می‌کند.

اجرا: `python -m src.eda_lib.runners.r5d_dorm_vs_faculty`
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from src.config import FIGURES_DIR
from src.eda_lib.disruption import load_calendar
from src.eda_lib.figio import save_fig
from src.eda_lib.runners._common import (
    DOW_FA,
    PERSON_DIM_PATH,
    PERSON_FACT_PATH,
    header,
    kv,
    load_dataset,
    pct,
    setup,
)
from src.viz_fa import fa
from src.viz_fa import setup as viz_setup

FIG_DIR = FIGURES_DIR / "round5"
MIN_RES = 30
TYPE_FA = {"daneshgah": "دانشکده‌ای", "khabgah": "خوابگاهی"}


# ---------------------------------------------------------------------------
# Q46 — هم‌آمیختگی نوع سلف با وعده
# ---------------------------------------------------------------------------

def q46(cell: pd.DataFrame) -> pd.DataFrame:
    header("Q46 — چقدر «نوع سلف» و «وعده» از هم قابل‌تفکیک‌اند؟")
    ct = pd.crosstab(cell["RestaurantType"], cell["Meal"], values=cell["Res"],
                     aggfunc="sum").fillna(0)
    ct.index = [TYPE_FA.get(i, i) for i in ct.index]
    print("\n— رزرو کل بر حسب نوع × وعده —")
    print(ct.to_string(float_format=lambda v: f"{v:,.0f}"))
    print("\n— سهم سطری —")
    print(ct.div(ct.sum(axis=1), axis=0).to_string(float_format=lambda v: f"{v:.3f}"))

    chi2, p, dof, _ = stats.chi2_contingency(ct.values)
    n = ct.values.sum()
    cramer = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
    kv("کرامر V (نوع، وعده)", f"{cramer:.4f}")
    kv("چند سلف هر دو نوع را دارند", cell.groupby("RestaurantName")["RestaurantType"].nunique().gt(1).sum())

    # کدام مقایسه شناسا می‌ماند؟
    print("\n— تعداد سلولِ قابل‌استفاده برای مقایسه‌ی *درون‌وعده* —")
    tab = cell.groupby(["Meal", "RestaurantType"]).agg(
        n_cells=("rho", "size"), n_rest=("RestaurantName", "nunique"),
        Res=("Res", "sum"), rho=("rho", lambda s: np.nan)).reset_index()
    w = cell.groupby(["Meal", "RestaurantType"]).apply(
        lambda x: np.average(x["rho"], weights=x["Res"]), include_groups=False)
    tab["نرخ وزنی"] = tab.set_index(["Meal", "RestaurantType"]).index.map(w)
    tab = tab.drop(columns=["rho"])
    tab["RestaurantType"] = tab["RestaurantType"].map(TYPE_FA)
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n> فقط سطرهایی که در هر دو نوع سلولِ کافی دارند، مقایسه‌ی شناسا می‌دهند.")
    return tab


# ---------------------------------------------------------------------------
# Q47 — دو سری روزانه: چقدر هم‌حرکت، کِی واگرا
# ---------------------------------------------------------------------------

def q47(cell: pd.DataFrame, cal: pd.DataFrame) -> pd.DataFrame:
    header("Q47 — دو سری روزانه چقدر هم‌حرکت‌اند و کِی می‌پاشند؟")
    ser = (cell.groupby(["date_gregorian", "RestaurantType"])
           .apply(lambda x: pd.Series({"rho": np.average(x["rho"], weights=x["Res"]),
                                       "Res": x["Res"].sum()}), include_groups=False)
           .reset_index())
    piv = ser.pivot(index="date_gregorian", columns="RestaurantType", values="rho").dropna()
    piv.columns = [TYPE_FA[c] for c in piv.columns]
    kv("روزهای با هر دو نوع", len(piv))
    r, p = stats.pearsonr(piv["دانشکده‌ای"], piv["خوابگاهی"])
    rs, ps = stats.spearmanr(piv["دانشکده‌ای"], piv["خوابگاهی"])
    kv("همبستگی پیرسون بین دو سری", f"{r:+.4f} (p={p:.2e})")
    kv("همبستگی اسپیرمن", f"{rs:+.4f} (p={ps:.2e})")
    kv("نرخ میانگین دانشکده‌ای", f"{piv['دانشکده‌ای'].mean():.4f}")
    kv("نرخ میانگین خوابگاهی", f"{piv['خوابگاهی'].mean():.4f}")

    # واگرایی = اختلاف استانداردشده‌ی دو سری
    z = (piv - piv.mean()) / piv.std()
    piv["واگرایی"] = z["دانشکده‌ای"] - z["خوابگاهی"]
    piv = piv.reset_index().merge(cal, on="date_gregorian", how="left")
    piv["dow_name"] = piv["date_gregorian"].dt.dayofweek.map(
        lambda d: DOW_FA[(d + 2) % 7])

    cols = ["date_jalali", "dow_name", "دانشکده‌ای", "خوابگاهی", "واگرایی",
            "is_day_before_holiday", "is_day_after_holiday", "is_exam_period"]
    print("\n— ۱۰ روزی که دانشکده‌ای به‌طور نسبی بسیار بدتر بود —")
    print(piv.nlargest(10, "واگرایی")[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n— ۱۰ روزی که خوابگاهی به‌طور نسبی بسیار بدتر بود —")
    print(piv.nsmallest(10, "واگرایی")[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return piv


# ---------------------------------------------------------------------------
# Q48 — کدام ویژگی تقویمی ضریب *متفاوت* دارد؟
# ---------------------------------------------------------------------------

CAL_TERMS = [("is_day_before_holiday", "روز پیش از تعطیلی"),
             ("is_day_after_holiday", "روز پس از تعطیلی"),
             ("is_exam_period", "دوره‌ی امتحانات"),
             ("is_thu", "پنجشنبه"),
             ("is_wed", "چهارشنبه")]


def q48(cell: pd.DataFrame, cal: pd.DataFrame) -> pd.DataFrame:
    header("Q48 — کدام اهرم تقویمی برای دو نوع سلف ضریب متفاوت دارد؟")
    d = cell.merge(cal, on="date_gregorian", how="left")
    d["is_thu"] = (d["DayOfWeek"] == 5).astype(float)
    d["is_wed"] = (d["DayOfWeek"] == 4).astype(float)
    d["khabgah"] = (d["RestaurantType"] == "khabgah").astype(float)
    for c, _ in CAL_TERMS:
        d[c] = d[c].astype(float)

    # اثر ثابت سلف و وعده، هر دو — پس ضریب برهم‌کنش «تفاوت اثر تقویم بین دو نوع» است،
    # نه اختلاف سطح و نه اثر وعده (تله‌ی Q46).
    terms = " + ".join([f"{c}*khabgah" for c, _ in CAL_TERMS])
    f = f"rho ~ C(RestaurantName) + C(Meal) + {terms}"
    m = smf.wls(f, data=d, weights=d["Res"]).fit(cov_type="HC3")

    rows = []
    for c, lab in CAL_TERMS:
        main = m.params.get(c, np.nan)
        inter_key = f"{c}:khabgah"
        inter = m.params.get(inter_key, np.nan)
        rows.append({
            "اهرم": lab,
            "اثر در دانشکده‌ای": main,
            "اثر در خوابگاهی": main + inter,
            "تفاوت (برهم‌کنش)": inter,
            "p تفاوت": m.pvalues.get(inter_key, np.nan),
        })
    tbl = pd.DataFrame(rows)
    print("\n(متغیر وابسته: نرخ سلول؛ وزن=رزرو؛ اثر ثابت سلف و وعده؛ HC3)")
    print(tbl.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    kv("R² مدل", f"{m.rsquared:.4f}")
    kv("تعداد سلول", f"{int(m.nobs):,}")
    return tbl


# ---------------------------------------------------------------------------
# Q49 — واگرایی از ترکیب افراد است یا از مکان؟
# ---------------------------------------------------------------------------

def q49(cell: pd.DataFrame) -> None:
    header("Q49 — تفاوت دو نوع سلف: اثر مکان یا ترکیب افراد؟ (پل L5→L1)")
    print("آزمون درون‌فردی: کسانی که در **هر دو** نوع سلف غذا خورده‌اند، نرخشان در دو نوع")
    print("چقدر فرق می‌کند؟ اگر تفاوت بماند، اثر مکان است؛ اگر بپرد، ترکیب افراد بود.\n")

    fact = pd.read_csv(PERSON_FACT_PATH,
                       usecols=["PersonId", "date_gregorian", "Meal",
                                "restaurant_canonical", "dont_receive", "is_main_meal"])
    fact = fact[fact["is_main_meal"]]
    tmap = (cell[["RestaurantName", "Meal", "RestaurantType"]].drop_duplicates()
            .rename(columns={"RestaurantName": "restaurant_canonical"}))
    fact = fact.merge(tmap, on=["restaurant_canonical", "Meal"], how="inner")

    g = fact.groupby(["PersonId", "RestaurantType"])["dont_receive"].agg(["mean", "size"])
    g = g[g["size"] >= 10]["mean"].unstack()
    both = g.dropna()
    kv("افراد با ≥۱۰ رزرو در هر دو نوع", f"{len(both):,}")
    if len(both) < 30:
        print("  [warn] نمونه‌ی کوچک — نتیجه محتاطانه تفسیر شود")
    kv("نرخ همین افراد در سلف دانشکده‌ای", f"{both['daneshgah'].mean():.4f}")
    kv("نرخ همین افراد در سلف خوابگاهی", f"{both['khabgah'].mean():.4f}")
    diff = both["daneshgah"] - both["khabgah"]
    w = stats.wilcoxon(both["daneshgah"], both["khabgah"])
    kv("اختلاف درون‌فردی (میانگین)", f"{diff.mean():+.4f}")
    kv("اختلاف درون‌فردی (میانه)", f"{diff.median():+.4f}")
    kv("ویلکاکسون زوجی p", f"{w.pvalue:.3e}")
    kv("٪ افرادی که در دانشکده‌ای بدترند", pct(float((diff > 0).mean())))

    # مقایسه با اختلاف بین‌فردی خام (که ترکیب را کنترل نمی‌کند)
    raw = fact.groupby("RestaurantType")["dont_receive"].mean()
    kv("اختلاف خام بین دو نوع (بدون کنترل فرد)", f"{raw['daneshgah'] - raw['khabgah']:+.4f}")
    kv("سهم اختلاف خام که «اثر مکان» است",
       f"{100 * diff.mean() / (raw['daneshgah'] - raw['khabgah']):.1f}%")


def figures(piv: pd.DataFrame, q48_tbl: pd.DataFrame) -> None:
    viz_setup()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(2, 1, figsize=(13, 6.6), sharex=True)
    d = piv.sort_values("date_gregorian")
    ax[0].plot(d["date_gregorian"], d["دانشکده‌ای"], lw=1.3, label=fa("دانشکده‌ای"), color="#47c")
    ax[0].plot(d["date_gregorian"], d["خوابگاهی"], lw=1.3, label=fa("خوابگاهی"), color="#4a7")
    ax[0].set_ylabel(fa("نرخ عدم‌دریافت"))
    ax[0].legend()
    ax[0].set_title(fa("دو سری روزانه — سطح متفاوت، حرکت مشترک"))
    ax[1].fill_between(d["date_gregorian"], d["واگرایی"], 0,
                       where=d["واگرایی"] > 0, color="#47c", alpha=0.7)
    ax[1].fill_between(d["date_gregorian"], d["واگرایی"], 0,
                       where=d["واگرایی"] <= 0, color="#4a7", alpha=0.7)
    ax[1].axhline(0, c="k", lw=1)
    ax[1].set_ylabel(fa("واگرایی (استانداردشده)"))
    ax[1].set_xlabel(fa("تاریخ"))
    ax[1].set_title(fa("بالا = دانشکده‌ای نسبتاً بدتر · پایین = خوابگاهی نسبتاً بدتر"))
    fig.tight_layout()
    save_fig(fig, "R5_q47_two_series", FIG_DIR)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.4))
    x = np.arange(len(q48_tbl))
    w = 0.38
    ax.bar(x - w / 2, q48_tbl["اثر در دانشکده‌ای"], w, label=fa("دانشکده‌ای"), color="#47c")
    ax.bar(x + w / 2, q48_tbl["اثر در خوابگاهی"], w, label=fa("خوابگاهی"), color="#4a7")
    ax.axhline(0, c="k", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([fa(s) for s in q48_tbl["اهرم"]], fontsize=9)
    ax.set_ylabel(fa("اثر بر نرخ عدم‌دریافت"))
    ax.set_title(fa("اثر هر اهرم تقویمی، جدا برای دو نوع سلف"))
    ax.legend()
    for i, r in q48_tbl.reset_index().iterrows():
        if r["p تفاوت"] < 0.05:
            ax.text(i, max(r["اثر در دانشکده‌ای"], r["اثر در خوابگاهی"]) + 0.002, "*",
                    ha="center", fontsize=16)
    fig.tight_layout()
    save_fig(fig, "R5_q48_calendar_by_type", FIG_DIR)
    plt.close(fig)


def main() -> None:
    setup()
    cell = pd.read_parquet("data/interim/r5_cell_level.parquet")
    cal = load_calendar()
    q46(cell)
    piv = q47(cell, cal)
    tbl = q48(cell, cal)
    q49(cell)
    figures(piv, tbl)
    print("\n[ok] مرحله ۵ تمام شد")


if __name__ == "__main__":
    main()
