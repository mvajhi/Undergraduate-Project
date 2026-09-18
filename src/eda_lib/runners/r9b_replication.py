"""بند ۹-۴ — دفتر بازآزمون سطح فرد: یافته‌هایی که دور ۸ معلق گذاشت.

اجرا: python -m src.eda_lib.runners.r9b_replication

این اجراکننده **فقط** یافته‌هایی را حساب می‌کند که جای دیگری از دور ۹ حساب
نشده‌اند. F73/F74 در ۹-۵، F75/F77/F78/F79 در ۹-۶، و F86 در ۹-۱۲ آمده‌اند و
اینجا فقط به آن‌ها ارجاع می‌شود — بازمحاسبه‌ی دوباره، دو عدد کمی متفاوت تولید
می‌کند و همان چیزی است که دفتر بازآزمون باید جلویش را بگیرد.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as st

from src.config import set_global_seed
from src.eda_lib import figs
from src.eda_lib.person_profile_v4 import build_person_rows_v4
from src.eda_lib.runners._common_r9 import (
    N_MIN_MAIN, at_threshold, header, load_profile, md_table, setup,
)


def f46_persistence(rows: pd.DataFrame) -> dict:
    """F46/F193 — پایداری رفتار فرد بین نیمه‌ی اول و دوم حضور خودش."""
    header("F46/F193 — پایداری نرخ فردی (نیمه‌ی اول در برابر نیمه‌ی دوم)", 2)
    g = rows.groupby("PersonId", observed=True)["meal_seq"]
    lo, hi = g.transform("min"), g.transform("max")
    second = rows.meal_seq > (lo + hi) / 2
    t = (rows.assign(_h=second.to_numpy())
         .groupby(["PersonId", "_h"], observed=True)["dont_receive"]
         .agg(["mean", "size"]).unstack("_h"))
    t.columns = ["rho_1", "rho_2", "n_1", "n_2"]
    t = t.dropna()
    for n_min in (20, 50):
        s = t[(t.n_1 >= n_min / 2) & (t.n_2 >= n_min / 2)]
        pe = st.pearsonr(s.rho_1, s.rho_2)
        sp = st.spearmanr(s.rho_1, s.rho_2)
        print(f"  n_min={n_min:3d} (هر نیمه ≥{n_min // 2}): n={len(s):6d} · "
              f"پیرسون={pe.statistic:.4f} · اسپیرمن={sp.statistic:.4f}")
        if n_min == 20:
            out = {"pearson": float(pe.statistic), "spearman": float(sp.statistic),
                   "n": len(s)}
    print("  لنگر: F46 (v2) پیرسون +۰.۶۳۶ · F193 (v4، دور ۷) ۰.۶۳۱")
    return out


def f64_intensity(P: pd.DataFrame) -> dict:
    """F64 — شدت استفاده قوی‌ترین همبسته‌ی سطح فرد."""
    header("F64 — همبستگی شدت مصرف با نرخ", 2)
    out = {}
    for n_min in (1, 20):
        S = at_threshold(P, n_min)
        sp = st.spearmanr(S.n_res, S.rho_eb)
        q = pd.qcut(S.n_res.rank(method="first"), 4, labels=["Q1 کم‌مصرف", "Q2", "Q3", "Q4 پرمصرف"])
        m = S.groupby(q, observed=True).rho_eb.mean()
        print(f"  n_min={n_min:2d}: اسپیرمن(تعداد رزرو، نرخ EB) = {sp.statistic:.4f} · "
              f"Q1={m.iloc[0]:.4f} Q4={m.iloc[-1]:.4f} (نسبت {m.iloc[0] / m.iloc[-1]:.2f}×)")
        if n_min == 20:
            out = {"spearman": float(sp.statistic), "ratio": float(m.iloc[0] / m.iloc[-1])}
    print("  لنگر F64 (v2): اسپیرمن −۰.۲۶۲ · چارک کم‌مصرف ۰.۱۳۴۴ در برابر پرمصرف ۰.۰۵۹۳ (۲.۳×)")
    return out


def f55_state_vs_trait(rows: pd.DataFrame, max_lag: int = 10) -> pd.DataFrame:
    """F55/F194 — همبستگی درون‌فردی با تأخیر، پس از حذف اثر ثابت فرد."""
    header("F55/F194 — همبستگی درون‌فردی lag-k (اثر ثابت فرد حذف‌شده)", 2)
    d = rows[["PersonId", "meal_seq", "dont_receive"]].sort_values(["PersonId", "meal_seq"])
    n = d.groupby("PersonId", observed=True).PersonId.transform("size")
    d = d[n >= N_MIN_MAIN]
    y = d.dont_receive.to_numpy(dtype="float64")
    pid = d.PersonId.to_numpy()
    mean = d.groupby("PersonId", observed=True).dont_receive.transform("mean").to_numpy()
    r = y - mean                                   # حذف اثر ثابت فرد
    rows_out = []
    for k in range(1, max_lag + 1):
        same = pid[k:] == pid[:-k]
        a, b = r[k:][same], r[:-k][same]
        rows_out.append({"lag": k, "n جفت": int(same.sum()),
                         "همبستگی درون‌فردی": float(np.corrcoef(a, b)[0, 1])})
    t = pd.DataFrame(rows_out)
    print(md_table(t.round(4)))
    print("  لنگر: F55 (v2) lag-1 = +۰.۰۴۹، «پس از ۳ روز صفر» · "
          "F194 (v4، دور ۷) lag-1 = +۰.۰۹۱۳، lag-3 = +۰.۰۴۰۲")
    return t


def f49_coldstart(rows: pd.DataFrame) -> dict:
    """F49 — آیا نرخ فرد کم‌سابقه از باسابقه تفکیک‌پذیر است؟"""
    header("F49 — cold-start در برابر باسابقه", 2)
    g = rows.groupby("PersonId", observed=True)
    t = pd.DataFrame({"n": g.size(), "rho": g.dont_receive.mean()})
    cold = t[t.n.between(1, 4)]
    est = t[t.n >= 20]
    mw = st.mannwhitneyu(cold.rho, est.rho)
    # اندازه‌ی اثر کلیف از آماره‌ی U (بدون ساخت ماتریس n×m)
    delta = 2 * mw.statistic / (len(cold) * len(est)) - 1
    print(f"  cold-start (۱-۴ رزرو): n={len(cold):5d} میانگین={cold.rho.mean():.4f} "
          f"میانه={cold.rho.median():.4f} انحراف={cold.rho.std():.4f}")
    print(f"  باسابقه (≥۲۰ رزرو):   n={len(est):5d} میانگین={est.rho.mean():.4f} "
          f"میانه={est.rho.median():.4f} انحراف={est.rho.std():.4f}")
    print(f"  Mann-Whitney p={mw.pvalue:.3e} · دلتای کلیف={delta:+.4f}")
    print("  لنگر F49 (v2): میانگین ۰.۲۳۷ در برابر ۰.۰۹۰، p=۰.۱۱۲، δ=+۰.۰۱۷، میانه‌ی cold صفر")
    return {"p": float(mw.pvalue), "delta": float(delta)}


def f65_honeymoon(rows: pd.DataFrame) -> pd.DataFrame:
    """F65/F203 — دانشجوی تازه‌وارد ابتدا بهتر است یا بدتر؟"""
    header("F65/F203 — باقیمانده نسبت به میانگین خودِ فرد، بر حسب رتبه‌ی رزرو", 2)
    d = rows[["PersonId", "meal_seq", "dont_receive"]].sort_values(["PersonId", "meal_seq"])
    n = d.groupby("PersonId", observed=True).PersonId.transform("size")
    d = d[n >= 150].copy()                       # برای دیدن تا رتبه‌ی ۱۵۰
    d["rank"] = d.groupby("PersonId", observed=True).cumcount() + 1
    d["resid"] = (d.dont_receive.astype(float)
                  - d.groupby("PersonId", observed=True).dont_receive.transform("mean"))
    bins = [0, 5, 10, 25, 50, 100, 150]
    d["band"] = pd.cut(d["rank"], bins=bins)
    t = d.groupby("band", observed=True).resid.agg(["mean", "size"]).reset_index()
    t["band"] = t["band"].astype(str)
    print(md_table(t.round(5)))
    print(f"  افراد واجد شرایط (≥۱۵۰ رزرو): {d.PersonId.nunique():,}")
    print("  لنگر F65 (v2): رزرو ۱-۵ = −۰.۰۱۸۰ → رزرو ۱۰۱-۱۵۰ = +۰.۰۰۹۱")

    fig, ax = figs.new(w=7.0, h=4.2)
    x = np.arange(len(t))
    ax.bar(x, t["mean"], color=[figs.C_OK if v < 0 else figs.C_ALT for v in t["mean"]])
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x, t["band"], fontsize=8)
    ax.set_xlabel("رتبه‌ی رزرو در تاریخچه‌ی فرد")
    ax.set_ylabel("باقیمانده نسبت به میانگین خودِ فرد")
    ax.set_title("دوره‌ی ماه‌عسل — بازآزمون F65")
    figs.save(fig, "r9b_01_honeymoon", t,
              caption="منفی یعنی بهتر از میانگین خودِ فرد؛ شیب صعودی یعنی بی‌قیدتر شدن")
    return t


def f62_person_day(P: pd.DataFrame) -> dict:
    """F62/F201 — آیا اثر فرد و روز ضرب‌شونده است؟"""
    header("F62/F201 — شکاف پرریسک و کم‌ریسک در روز خوب و روز بد", 2)
    S = at_threshold(P, N_MIN_MAIN).dropna(subset=["rho_on_bad_days", "rho_on_good_days"])
    tier = pd.qcut(S.rho_eb.rank(method="first"), 3,
                   labels=["کم‌ریسک", "میان‌ریسک", "پرریسک"])
    m = S.groupby(tier, observed=True)[["rho_on_good_days", "rho_on_bad_days"]].mean()
    gap_good = m.loc["پرریسک", "rho_on_good_days"] - m.loc["کم‌ریسک", "rho_on_good_days"]
    gap_bad = m.loc["پرریسک", "rho_on_bad_days"] - m.loc["کم‌ریسک", "rho_on_bad_days"]
    print(md_table(m.reset_index().round(4)))
    print(f"  شکاف در روز خوب: {gap_good:+.4f} · در روز بد: {gap_bad:+.4f} · "
          f"نسبت **{gap_bad / gap_good:.2f}×**")
    print("  لنگر: F62 (v2) نسبت ۳.۳۶× · F201 (v4، دور ۷) نسبت ۲.۳۶×")

    fig, ax = figs.new(w=7.0, h=4.2)
    x = np.arange(len(m))
    ax.bar(x - 0.2, m.rho_on_good_days, 0.4, color=figs.C_OK, label="روز خوب (دهک پایین شوک)")
    ax.bar(x + 0.2, m.rho_on_bad_days, 0.4, color=figs.C_ALT, label="روز بد (دهک بالای شوک)")
    ax.set_xticks(x, m.index)
    ax.set_ylabel("نرخ عدم‌دریافت")
    ax.set_title(f"برهم‌کنش فرد × روز — نسبت شکاف {gap_bad / gap_good:.2f}×")
    ax.legend(fontsize=8)
    figs.save(fig, "r9b_02_person_day_interaction", m.reset_index(),
              caption="اگر اثرها جمع‌شونده بودند دو شکاف برابر می‌شدند")
    return {"gap_good": float(gap_good), "gap_bad": float(gap_bad),
            "ratio": float(gap_bad / gap_good)}


def f76_dorm_venue(rows: pd.DataFrame) -> pd.DataFrame:
    """F76 — چقدر از «مزیت خوابگاهی‌بودن» مزیتِ خودِ سلف است؟"""
    header("F76 — ضریب is_dorm پیش و پس از افزودن اثر ثابت سلف", 2)
    import statsmodels.formula.api as smf

    d = rows[["PersonId", "is_dorm_resident", "RestaurantName", "dont_receive"]]
    # تجمیع در سطح (فرد، سلف) تا لجیت روی ۸ میلیون ردیف اجرا نشود
    g = (d.groupby(["PersonId", "is_dorm_resident", "RestaurantName"], observed=True)
         .dont_receive.agg(["sum", "size"]).reset_index())
    g.columns = ["PersonId", "is_dorm", "venue", "k", "n"]
    g = g[g.n >= 5]
    g["is_dorm"] = g.is_dorm.astype(int)
    g["rate"] = g.k / g.n

    m1 = smf.wls("rate ~ is_dorm", data=g, weights=g.n).fit(cov_type="HC3")
    m2 = smf.wls("rate ~ is_dorm + C(venue)", data=g, weights=g.n).fit(cov_type="HC3")
    t = pd.DataFrame([
        {"مدل": "بدون اثر ثابت سلف", "ضریب is_dorm": float(m1.params["is_dorm"]),
         "p": float(m1.pvalues["is_dorm"]), "R²": float(m1.rsquared)},
        {"مدل": "با اثر ثابت سلف", "ضریب is_dorm": float(m2.params["is_dorm"]),
         "p": float(m2.pvalues["is_dorm"]), "R²": float(m2.rsquared)},
    ])
    print(md_table(t.round(5)))
    shrink = 1 - t.loc[1, "ضریب is_dorm"] / t.loc[0, "ضریب is_dorm"]
    print(f"  کاهش ضریب پس از کنترل سلف: **{shrink:.1%}** (لنگر F76: −۷۴٪)")
    print(f"  n = {len(g):,} زوج (فرد، سلف) با ≥۵ رزرو")
    return t


def f85_demo_shock(P: pd.DataFrame) -> pd.DataFrame:
    """F85 — آیا جمعیت‌شناسی در روز بد فشرده می‌شود؟"""
    header("F85 — نسبت نرخ جمعیتی در روز خوب و روز بد", 2)
    S = at_threshold(P, N_MIN_MAIN).dropna(subset=["rho_on_bad_days", "rho_on_good_days"])
    S = S.assign(
        grad=S.demo_degree.astype(str).str.contains("ارشد|دكتري|دکتری|دكترا", na=False),
        night=S.demo_session.astype(str).str.contains("نوبت دوم", na=False),
        male=S.demo_gender.astype(str).str.strip().eq("مرد"))
    rows_out = []
    for col, lbl in (("demo_is_dorm", "خوابگاهی"), ("grad", "تحصیلات تکمیلی"),
                     ("night", "شبانه/نوبت دوم"), ("male", "مرد")):
        m = S.groupby(S[col].astype(bool), observed=True)[
            ["rho_on_good_days", "rho_on_bad_days"]].mean()
        if len(m) < 2:
            continue
        rows_out.append({
            "ویژگی": lbl,
            "نسبت نرخ — روز خوب": m.loc[True, "rho_on_good_days"] / m.loc[False, "rho_on_good_days"],
            "نسبت نرخ — روز بد": m.loc[True, "rho_on_bad_days"] / m.loc[False, "rho_on_bad_days"],
        })
    t = pd.DataFrame(rows_out)
    t["جهت"] = np.where(t["نسبت نرخ — روز بد"] < t["نسبت نرخ — روز خوب"],
                        "فشرده می‌شود (هم‌راستا با F85)", "باز می‌شود (خلاف F85)")
    print(md_table(t.round(4)))
    print("  لنگر F85 (v2): نسبت نرخ تحصیلات تکمیلی ۱.۱۷ (روز خوب) → ۰.۹۳ (روز بد)")
    return t


def main() -> None:
    set_global_seed()
    setup()
    P = load_profile()
    rows = build_person_rows_v4(columns=["PersonId", "meal_seq", "dont_receive",
                                         "is_dorm_resident", "RestaurantName"])
    header("۹-۴ دفتر بازآزمون سطح فرد")
    print(f"افراد {P.PersonId.nunique():,} · رزرو {len(rows):,}")
    f46_persistence(rows)
    f64_intensity(P)
    f55_state_vs_trait(rows)
    f49_coldstart(rows)
    f65_honeymoon(rows)
    f62_person_day(P)
    f76_dorm_venue(rows)
    f85_demo_shock(P)
    header("پایان بند ۹-۴")


if __name__ == "__main__":
    main()
