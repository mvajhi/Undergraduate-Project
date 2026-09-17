"""۱۴-۴-۱ (پیگیری) و ۱۴-۴-۲ · ساختار واریانس، ناهم‌واریانسی و ICC روی ۵۹۳ روز.

اجرا: ``python -m src.eda_lib.runners.r7c_variance_structure``

⚠️ قاعده‌ی حاکم بر کل این بند (F142، بند ۱۴-۳-۲): **ICC ویژگی جفتِ (پدیده، شبکه)
است.** هر عدد ICC اینجا با شبکه‌ای که رویش محاسبه شده نقل می‌شود، وگرنه با عدد
F10 قابل‌مقایسه نیست.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import set_global_seed
from src.eda_lib.distribution_v4 import fit_families
from src.eda_lib.mapping_audit_v4 import icc_oneway
from src.eda_lib.runners._common_v4 import header, load_features, md_table, show

SIZE_BINS = [0, 5, 10, 20, 50, 100, 250, 10 ** 9]
SIZE_LABELS = ["0–5", "5–10", "10–20", "20–50", "50–100", "100–250", "250–∞"]


def _add_size_bucket(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["سطل"] = pd.cut(df.Res, bins=SIZE_BINS, labels=SIZE_LABELS, right=False)
    return df


def signal_variance(g: pd.DataFrame) -> dict[str, float]:
    """واریانس پس از کم‌کردن نویز نمونه‌گیری دوجمله‌ای.

    $\\operatorname{Var}(\\rho) = \\operatorname{Var}_{\\text{واقعی}} +
    E[\\rho(1-\\rho)/Res]$. جمله‌ی دوم **قابل‌محاسبه** است، پس جمله‌ی اول
    قابل‌استخراج. بدون این تفکیک، هر مقایسه‌ی پراکندگی بین دو مجموعه با
    دانه‌بندی متفاوت بی‌معناست.
    """
    y = g.rho.to_numpy()
    v_obs = float(np.var(y))
    v_samp = float(np.mean(y * (1 - y) / np.maximum(g.Res.to_numpy(), 1)))
    return {"Var مشاهده": v_obs, "Var نمونه‌گیری": v_samp,
            "Var سیگنال": max(v_obs - v_samp, 0.0), "سهم نویز": v_samp / v_obs if v_obs else np.nan}


def nested_icc(df: pd.DataFrame) -> pd.DataFrame:
    """ICC سه‌سطحی — روز، پردیس، و سلف **درون** پردیس.

    سلف‌درون‌پردیس جدا محاسبه می‌شود چون شهر ≡ پردیس است (F146) و ICC خام سلف
    اثر پردیس را با خودش حمل می‌کند.
    """
    y = df.rho.to_numpy()
    day = (df.date_gregorian.astype(str) + "|" + df.meal.astype(str)).to_numpy()
    rows = [
        {"سطح": "روز×وعده", **icc_oneway(y, day)},
        {"سطح": "پردیس (≡شهر)", **icc_oneway(y, df.campus.astype(str).to_numpy())},
        {"سطح": "نوع سلف", **icc_oneway(y, df.venue_type.astype(str).to_numpy())},
        {"سطح": "سلف", **icc_oneway(y, df.restaurant.astype(str).to_numpy())},
        {"سطح": "غذا", **icc_oneway(y, df.dish.astype(str).to_numpy())},
    ]
    # سلف درون پردیس: اثر پردیس را اول بردار
    y_dm = y - pd.Series(y).groupby(pd.Series(df.campus.astype(str).to_numpy())).transform("mean").to_numpy()
    rows.append({"سطح": "سلف | پس از حذف پردیس", **icc_oneway(y_dm, df.restaurant.astype(str).to_numpy())})
    out = pd.DataFrame(rows)[["سطح", "k", "icc", "var_between", "var_within"]]
    out.columns = ["سطح", "گروه", "ICC", "Var بین", "Var درون"]
    out["n مؤثر (نسبت)"] = 1.0 / (1.0 + (len(df) / out["گروه"] - 1) * out["ICC"])
    return out


def main() -> None:
    set_global_seed()
    f = load_features(columns=["date_gregorian", "meal", "restaurant", "dish", "Res", "NoRecv",
                               "academic_year", "venue_type", "campus", "city"])
    f["rho"] = f.NoRecv / f.Res
    f = _add_size_bucket(f)

    header("۱ · ⭐ ناهم‌واریانسی پس از حذف نویز نمونه‌گیری — بازآزمون F06/F08")
    rows = []
    for lab in SIZE_LABELS:
        g = f[f["سطل"] == lab]
        if len(g) < 300:
            continue
        sv = signal_variance(g)
        rows.append({"سطل Res": lab, "سلول": len(g), "μ": float(g.NoRecv.sum() / g.Res.sum()),
                     **sv, "SD سیگنال": np.sqrt(sv["Var سیگنال"])})
    het = pd.DataFrame(rows)
    show(het)
    print("\n--- Markdown ---\n" + md_table(het, "{:.5f}"))
    r = het["SD سیگنال"].iloc[0] / het["SD سیگنال"].iloc[-1]
    print(f"\n⇒ SD سیگنال از {het['SD سیگنال'].iloc[0]:.4f} به {het['SD سیگنال'].iloc[-1]:.4f} می‌رسد — "
          f"دامنه‌ی {r:.1f} برابری **پس از** حذف نویز نمونه‌گیری.")
    print("  یعنی وزن‌دهی با Res که فقط نویز دوجمله‌ای را تصحیح می‌کند، **کافی نیست**.")

    header("۲ · φ بتا-دوجمله‌ای به تفکیک اندازه — آیا یک φ سراسری معنا دارد؟")
    rows = []
    for lab in SIZE_LABELS:
        g = f[f["سطل"] == lab]
        if len(g) < 300:
            continue
        ff = fit_families(g.NoRecv.to_numpy(), g.Res.to_numpy())
        bb = ff[ff["خانواده"] == "بتا-دوجمله‌ای"].iloc[0]
        zi = ff[ff["خانواده"] == "ZI بتا-دوجمله‌ای"].iloc[0]
        rows.append({"سطل Res": lab, "n": len(g), "μ": bb["μ"], "φ": bb["φ"], "π₀": zi["π₀"],
                     "ΔAIC نفع ZI": float(ff[ff["خانواده"] == "بتا-دوجمله‌ای"]["ΔAIC"].iat[0])})
    phi = pd.DataFrame(rows)
    show(phi)
    print("\n--- Markdown ---\n" + md_table(phi, "{:.4f}"))

    header("۳ · ⭐⭐ رشد پراکندگی بین سه سال — تجزیه‌ی بین‌سلفی/درون‌سلفی")
    rows = []
    for yr, g in f.groupby("academic_year"):
        y = g.rho.to_numpy()
        v = icc_oneway(y, g.restaurant.astype(str).to_numpy())
        samp = float(np.mean(y * (1 - y) / np.maximum(g.Res.to_numpy(), 1)))
        wsig = max(v["var_within"] - samp, 0.0)
        rows.append({"سال": yr, "سلف": v["k"], "Var کل": float(np.var(y)),
                     "Var بین‌سلفی": v["var_between"], "Var درون‌سلفی سیگنال": wsig,
                     "نویز نمونه‌گیری": samp, "ICC خام": v["icc"],
                     "ICC بدون‌نویز": v["var_between"] / (v["var_between"] + wsig)})
    yr_tab = pd.DataFrame(rows)
    show(yr_tab)
    print("\n--- Markdown ---\n" + md_table(yr_tab, "{:.5f}"))
    for c in ["Var کل", "Var بین‌سلفی", "Var درون‌سلفی سیگنال", "نویز نمونه‌گیری"]:
        a, b = yr_tab[c].iat[0], yr_tab[c].iat[-1]
        print(f"  {c:24s}: {a:.5f} → {b:.5f}  ({b/a - 1:+.1%})")

    header("۴ · همان، با مجموعه‌ی سلف **ثابت** — تفکیک ورود سلف تازه از واگرایی")
    common = set.intersection(*[set(g.restaurant.unique()) for _, g in f.groupby("academic_year")])
    fc = f[f.restaurant.isin(common)]
    print(f"  {len(common)} سلف در هر سه سال حاضرند ({fc.Res.sum()/f.Res.sum():.1%} پرس)")
    rows = []
    for yr, g in fc.groupby("academic_year"):
        ff = fit_families(g.NoRecv.to_numpy(), g.Res.to_numpy())
        bb = ff[ff["خانواده"] == "بتا-دوجمله‌ای"].iloc[0]
        v = icc_oneway(g.rho.to_numpy(), g.restaurant.astype(str).to_numpy())
        rows.append({"سال": yr, "n": len(g), "μ": bb["μ"], "φ": bb["φ"], "ICC سلف": v["icc"],
                     "Var بین": v["var_between"], "Var درون": v["var_within"]})
    fixed = pd.DataFrame(rows)
    show(fixed)
    print("\n--- Markdown ---\n" + md_table(fixed, "{:.5f}"))

    header("۵ · کوانتایل در برابر میانگین — آیا پایداری F102 به نقطه‌ی تصمیم می‌رسد؟")

    def wq(v, w, q):
        i = np.argsort(v)
        v, w = np.asarray(v)[i], np.asarray(w)[i]
        return float(np.interp(q, np.cumsum(w) / np.sum(w), v))

    rows = []
    for yr, g in f.groupby("academic_year"):
        w, v = g.Res.to_numpy(), g.rho.to_numpy()
        rows.append({"سال": yr, "ρ وزنی": float(g.NoRecv.sum() / g.Res.sum()),
                     "q10": wq(v, w, .10), "q20": wq(v, w, .20), "میانه": wq(v, w, .50),
                     "q80": wq(v, w, .80), "q90": wq(v, w, .90), "q90−q10": wq(v, w, .90) - wq(v, w, .10)})
    q = pd.DataFrame(rows)
    show(q)
    print("\n--- Markdown ---\n" + md_table(q, "{:.4f}"))
    for c in ["ρ وزنی", "q20", "میانه", "q90", "q90−q10"]:
        a, b = q[c].iat[0], q[c].iat[-1]
        print(f"  {c:9s}: {a:.4f} → {b:.4f}  ({b/a - 1:+.1%})")

    header("۶ · ICC لانه‌ای روی شبکه‌ی عملیاتی L1 v4")
    ni = nested_icc(f)
    show(ni)
    print("\n--- Markdown ---\n" + md_table(ni, "{:.4f}"))
    print("\n⚠️ این اعداد روی شبکه‌ی L1 v4 (۶۳٬۹۳۱ سلول، میانه Res=۵۲) محاسبه شده‌اند.")
    print("   F10 روی شبکه‌ی v2 (۷٬۵۷۹ سلول، میانه Res=۱۵۵) بود — F142، قابل‌مقایسه نیستند.")


if __name__ == "__main__":
    main()
