"""بستن دو ابهام باز دور ۹ — بازاجرای هم‌ارز F76 (ابهام ر) و بازآزمون F66 (ابهام س).

اجرا: python -m src.eda_lib.runners.r9k_open_items

## چرا این اجراکننده جدا از `r9b_replication` است

`r9b` هر یافته را با **ارزان‌ترین مشخصه‌ی معتبر** اجرا کرد. برای F76 آن انتخاب
(WLS روی سلول فرد×سلف) جهتِ تغییرِ برعکس داد و چون مشخصه‌اش با اصل یکی نبود،
نتیجه «بازتولیدنشده‌ی مشروط» ثبت شد (ابهام ر، ردیف ۱۱۱). اینجا **همان لجیت**
اجرا می‌شود تا مقایسه هم‌ارز باشد.

## لجیت روی ۸.۲ میلیون ردیف، بدون ۸.۲ میلیون ردیف

لجیت سطح‌رزرو با متغیرهای **تماماً رده‌ای** ریاضیاً معادل GLM دوجمله‌ای روی
سلول‌های (ترکیب رده‌ها) با (موفقیت، کل) است — همان ضرایب، همان خطای استاندارد.
پس به‌جای ۸.۲ میلیون ردیف، چند هزار سلول برازش می‌شود.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as st

from src.config import set_global_seed
from src.eda_lib import figs
from src.eda_lib.person_profile_v4 import build_person_rows_v4
from src.eda_lib.runners._common_r9 import header, md_table, setup


# ---------------------------------------------------------------------------
# ابهام (ر) — بازاجرای F76 با لجیت هم‌ارز
# ---------------------------------------------------------------------------

def f76_equivalent_logit() -> pd.DataFrame:
    """F76 با همان مشخصه‌ی اصل: لجیت سطح‌رزرو، با و بدون اثر ثابت سلف."""
    header("ابهام (ر) — بازاجرای F76 با لجیت هم‌ارز سطح رزرو", 2)
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    rows = build_person_rows_v4(columns=[
        "PersonId", "dont_receive", "is_dorm_resident", "RestaurantName",
        "Gender", "DegreeName", "EducationSession"])

    # ⚠️ چهار `.astype(str)` روی ستون رده‌ای ۸.۲ میلیونی، اوج حافظه را به
    # **۷.۹ گیگابایت** می‌برد — همان تله‌ای که `person_profile_v4` هم خورد.
    # هر محمول روی **دسته‌ها** اجرا می‌شود و `venue` کدِ رده‌ای می‌ماند.
    from src.eda_lib.person_profile_v4 import _map_categories

    venue_codes, venue_levels = pd.factorize(rows.RestaurantName)
    d = pd.DataFrame({
        "is_dorm": rows.is_dorm_resident.to_numpy().astype("int8"),
        "male": _map_categories(rows.Gender, lambda c: c.str.strip() == "مرد").astype("int8"),
        "grad": _map_categories(
            rows.DegreeName, lambda c: c.str.contains("ارشد|دكتري|دکتری|دكترا", na=False)
        ).astype("int8"),
        "night": _map_categories(
            rows.EducationSession, lambda c: c.str.contains("نوبت دوم", na=False)
        ).astype("int8"),
        "venue": venue_codes.astype("int16"),
        "y": rows.dont_receive.to_numpy().astype("int8"),
    })
    del rows

    # تجمیع به سلول: لجیت سطح‌رزرو با متغیرهای رده‌ای ⇔ GLM دوجمله‌ای روی سلول‌ها
    cells = (d.groupby(["is_dorm", "male", "grad", "night", "venue"], observed=True)
             .agg(k=("y", "sum"), n=("y", "size")).reset_index())
    print(f"۸٬۲۲۲٬۶۹۹ ردیف → **{len(cells):,} سلول** (ضرایب و خطای استاندارد یکسان)")

    out = []
    for lbl, formula in (("بدون اثر ثابت سلف", "k + I(n - k) ~ is_dorm + male + grad + night"),
                         ("با اثر ثابت سلف",
                          "k + I(n - k) ~ is_dorm + male + grad + night + C(venue)")):
        m = smf.glm(formula, data=cells, family=sm.families.Binomial()).fit()
        pseudo_r2 = 1 - m.deviance / m.null_deviance
        for var in ("is_dorm", "male", "grad", "night"):
            out.append({"مدل": lbl, "متغیر": var, "ضریب": float(m.params[var]),
                        "خطای استاندارد": float(m.bse[var]),
                        "p": float(m.pvalues[var]), "pseudo R²": pseudo_r2})
    t = pd.DataFrame(out)
    print(md_table(t.round(5)))

    piv = t.pivot(index="متغیر", columns="مدل", values="ضریب")
    piv["تغییر نسبی"] = (piv["با اثر ثابت سلف"] / piv["بدون اثر ثابت سلف"] - 1)
    print("\n▸ تغییر ضریب پس از افزودن اثر ثابت سلف:")
    print(md_table(piv.reset_index().round(4)))
    dr2 = (t[t["مدل"] == "با اثر ثابت سلف"]["pseudo R²"].iloc[0]
           - t[t["مدل"] == "بدون اثر ثابت سلف"]["pseudo R²"].iloc[0])
    print(f"\nΔpseudo R² (سلولی) = {dr2:+.5f}")
    print("⚠️ این عدد با لنگر F76 (+۰.۰۲۰۴) **قابل‌مقایسه نیست**: ضرایب و خطای")
    print("   استاندارد بین لجیت سطح‌رزرو و GLM سلولی یکسان‌اند، ولی انحراف (deviance)")
    print("   نیست — فرم سلولی تغییرات درون‌سلولی را از مخرج حذف می‌کند. فقط ستون")
    print("   **ضریب** بین دو دور قابل‌مقایسه است.")
    print("لنگر F76: is_dorm −۰.۸۰۸ → −۰.۲۰۹ (**−۷۴٪**) · مرد −۰.۰۹۷ → +۰.۰۶۳ (علامت عوض)")

    fig, ax = figs.new(w=7.6, h=4.4)
    x = np.arange(len(piv))
    ax.bar(x - 0.2, piv["بدون اثر ثابت سلف"], 0.4, color=figs.C_GREY,
           label="بدون اثر ثابت سلف")
    ax.bar(x + 0.2, piv["با اثر ثابت سلف"], 0.4, color=figs.C_MAIN,
           label="با اثر ثابت سلف")
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x, piv.index)
    ax.set_ylabel("ضریب لجیت")
    ax.set_title("بازاجرای F76 با مشخصه‌ی هم‌ارز (لجیت سطح رزرو)")
    ax.legend(fontsize=8)
    figs.save(fig, "r9k_01_f76_equivalent_logit", t,
              caption="آیا کنترل سلف اثر فرد را محو می‌کند یا تقویت؟ — با همان مشخصه‌ی اصل")
    return t


# ---------------------------------------------------------------------------
# ابهام (س) — بازآزمون F66، سه زیرفرضیه
# ---------------------------------------------------------------------------

def f66a_food_preference(min_pair: int = 6) -> dict:
    """F66-الف — آیا «ترجیح غذایی شخصی» پایدارتر از خودِ فرد است؟"""
    header("F66-الف — پایداری جفت (فرد، غذا) در برابر پایداری خودِ فرد", 2)
    rows = build_person_rows_v4(columns=["PersonId", "meal_seq", "dont_receive", "FoodName"])
    g = rows.groupby("PersonId", observed=True)["meal_seq"]
    second = (rows.meal_seq > (g.transform("min") + g.transform("max")) / 2).to_numpy()

    # پایداری خودِ فرد
    pr = (rows.assign(_h=second).groupby(["PersonId", "_h"], observed=True)
          .dont_receive.agg(["mean", "size"]).unstack("_h").dropna())
    pr.columns = ["r1", "r2", "n1", "n2"]
    pr = pr[(pr.n1 >= min_pair) & (pr.n2 >= min_pair)]
    person_stab = float(st.pearsonr(pr.r1, pr.r2).statistic)

    # پایداری جفت (فرد، غذا)
    pd_ = (rows.assign(_h=second)
           .groupby(["PersonId", "FoodName", "_h"], observed=True)
           .dont_receive.agg(["mean", "size"]).unstack("_h").dropna())
    pd_.columns = ["r1", "r2", "n1", "n2"]
    pd_ = pd_[(pd_.n1 >= min_pair) & (pd_.n2 >= min_pair)]
    pair_stab = float(st.pearsonr(pd_.r1, pd_.r2).statistic)

    print(f"آستانه: ≥{min_pair} مشاهده در **هر دو** نیمه")
    print(f"  پایداری خودِ فرد        : **{person_stab:.4f}**  (n={len(pr):,})")
    print(f"  پایداری جفت (فرد، غذا) : **{pair_stab:.4f}**  (n={len(pd_):,})")
    print(f"  لنگر F66-الف: جفت ۰.۲۳۹ < فرد ۰.۴۳۱")
    verdict = "تکرار" if pair_stab < person_stab else "بازتولیدنشده"
    print(f"  ⇒ وضعیت: **{verdict}** "
          f"({'جفت کمتر از فرد' if pair_stab < person_stab else 'جفت بیشتر از فرد'})")
    return {"person": person_stab, "pair": pair_stab, "verdict": verdict,
            "n_person": len(pr), "n_pair": len(pd_)}


def _icc_oneway(values: np.ndarray, group_codes: np.ndarray) -> float:
    """ICC یک‌طرفه از تجزیه‌ی واریانس ANOVA (همان تعریفی که F66-ب استفاده کرده).

    ⚠️ ورودی گروه **کد صحیح** است، نه رشته. نسخه‌ی نخست این تابع یک `DataFrame`
    با ستون رشته‌ای ۴ میلیونی می‌ساخت و `df.g.map(...)` می‌زد؛ همراه با
    `MultiIndex.from_arrays` در تابع فراخوان، اوج حافظه را به **۷.۹ گیگابایت**
    برد. اینجا همه‌چیز با `np.bincount` روی کدها انجام می‌شود.
    """
    y = np.asarray(values, dtype="float64")
    g = np.asarray(group_codes)
    n_i = np.bincount(g).astype("float64")
    keep = n_i > 0
    s_i = np.bincount(g, weights=y)
    k = int(keep.sum())
    if k < 2:
        return float("nan")
    m_i = np.divide(s_i, n_i, out=np.zeros_like(s_i), where=n_i > 0)
    grand = y.sum() / y.size
    ms_between = float((n_i[keep] * (m_i[keep] - grand) ** 2).sum() / (k - 1))
    ss_within = float(((y - m_i[g]) ** 2).sum())
    ms_within = ss_within / (y.size - k)
    nn = n_i[keep]
    n0 = (nn.sum() - (nn ** 2).sum() / nn.sum()) / (k - 1)
    var_b = (ms_between - ms_within) / n0
    return float(var_b / (var_b + ms_within)) if (var_b + ms_within) > 0 else float("nan")


def f66b_social_contagion() -> dict:
    """F66-ب — آیا «سرایت اجتماعی خوابگاه» پس از کنترل سلف-روز باقی می‌ماند؟"""
    header("F66-ب — ICC واحد سکونت، پیش و پس از کنترل سلف×روز", 2)
    rows = build_person_rows_v4(columns=["GroupName", "RestaurantName",
                                         "date_gregorian", "meal", "dont_receive",
                                         "is_dorm_resident"])
    d = rows[rows.is_dorm_resident]
    y = d.dont_receive.to_numpy(dtype="float64")
    # ⚠️ همه‌ی کلیدها به **کد صحیح** تبدیل می‌شوند، نه رشته و نه MultiIndex.
    # `MultiIndex.from_arrays` روی سه آرایه‌ی ۴ میلیونیِ شیء، به‌تنهایی چند
    # گیگابایت می‌گیرد (قاعده‌ی ۳ بخش حافظه‌ی AGENTS.md).
    grp, grp_levels = pd.factorize(d.GroupName)
    venue_c = pd.factorize(d.RestaurantName)[0].astype("int64")
    date_c = pd.factorize(d.date_gregorian)[0].astype("int64")
    meal_c = pd.factorize(d.meal)[0].astype("int64")
    n_meal = int(meal_c.max()) + 1
    n_date = int(date_c.max()) + 1
    cell = pd.factorize((venue_c * n_date + date_c) * n_meal + meal_c)[0]
    del rows, d, venue_c, date_c, meal_c

    icc_raw = _icc_oneway(y, grp)
    cell_mean = np.bincount(cell, weights=y) / np.bincount(cell)
    resid = y - cell_mean[cell]
    icc_ctrl = _icc_oneway(resid, grp)

    n_units = len(grp_levels)
    print(f"ساکنان خوابگاه: {y.size:,} رزرو · {n_units} واحد سکونت · "
          f"{int(cell.max()) + 1:,} سلول سلف×روز×وعده")
    print(f"  ICC خام                : **{icc_raw:+.4f}**")
    print(f"  ICC پس از کنترل سلف-روز: **{icc_ctrl:+.4f}**")
    print(f"  کاهش: {1 - icc_ctrl / icc_raw:.1%}" if icc_raw else "")
    print("  لنگر F66-ب: +۰.۰۱۹۵ → **+۰.۰۰۳۱** (کاهش ۸۴٪)")
    verdict = "تکرار" if icc_ctrl < 0.2 * icc_raw or icc_ctrl < 0.006 else "بازتولیدنشده"
    print(f"  ⇒ وضعیت: **{verdict}**")
    return {"icc_raw": icc_raw, "icc_ctrl": icc_ctrl, "n_units": n_units,
            "verdict": verdict}


def f66c_commitment() -> pd.DataFrame:
    """F66-ج — آیا ستون‌های «سیگنال تعهد» هنوز منحط‌اند؟"""
    header("F66-ج — منحط‌بودن ستون‌های سیگنال تعهد روی v4", 2)
    from src.data.query_v4 import connect

    con = connect()
    t = con.execute("""
        SELECT 'Count' AS ستون, COUNT(DISTINCT Count) AS مقدار_یکتا,
               SUM(CASE WHEN Count <> 1 THEN 1 ELSE 0 END) AS ردیف_غیرپیش‌فرض,
               COUNT(*) AS کل FROM reserve
        UNION ALL
        SELECT 'Price', COUNT(DISTINCT Price),
               SUM(CASE WHEN Price = 0 THEN 1 ELSE 0 END), COUNT(*) FROM reserve
        UNION ALL
        SELECT 'ReceptionType', COUNT(DISTINCT ReceptionType),
               SUM(CASE WHEN ReceptionType NOT IN (0,1) THEN 1 ELSE 0 END),
               COUNT(*) FROM reserve
    """).fetchdf()
    t["سهم غیرپیش‌فرض"] = t["ردیف_غیرپیش‌فرض"] / t["کل"]
    print(md_table(t.round(6)))
    print("\nلنگر F66-ج: `Count` ۱۰۰٪ برابر ۱ و `has_extras` ۱۰۰٪ True — هر دو منحط")
    print("⇒ `Count` روی v4 **منحط نیست** (۲۳٬۹۴۵ ردیف با Count>1، همان F94)")
    print("⇒ `has_extras` در استخراج v4 **اصلاً ستونی ندارد** — غیرقابل‌آزمون")
    print("⇒ اثر `Price` در بند ۹-۱۱-۴ سنجیده شد: اسپیرمن −۰.۰۰۶۹ (p=۰.۱۸)")
    con.close()
    return t


def main() -> None:
    set_global_seed()
    setup()
    header("بستن دو ابهام باز دور ۹ — (ر) و (س)")
    f76_equivalent_logit()
    a = f66a_food_preference()
    b = f66b_social_contagion()
    f66c_commitment()

    header("خلاصه‌ی حکم", 2)
    print(md_table(pd.DataFrame([
        {"زیرفرضیه": "F66-الف ترجیح غذایی شخصی",
         "عدد v2": "جفت ۰.۲۳۹ < فرد ۰.۴۳۱",
         "عدد v4": f"جفت {a['pair']:.3f} در برابر فرد {a['person']:.3f}",
         "وضعیت": a["verdict"]},
        {"زیرفرضیه": "F66-ب سرایت اجتماعی خوابگاه",
         "عدد v2": "+۰.۰۱۹۵ → +۰.۰۰۳۱",
         "عدد v4": f"{b['icc_raw']:+.4f} → {b['icc_ctrl']:+.4f}",
         "وضعیت": b["verdict"]},
        {"زیرفرضیه": "F66-ج سیگنال تعهد",
         "عدد v2": "Count منحط، has_extras منحط",
         "عدد v4": "Count منحط نیست · has_extras وجود ندارد",
         "وضعیت": "تصحیح/غیرقابل‌آزمون"},
    ])))
    header("پایان")


if __name__ == "__main__":
    main()
