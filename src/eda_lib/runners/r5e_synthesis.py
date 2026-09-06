"""دور ۵ / مرحله ۶ — ترکیب: آیا هر سه یافته یک سازوکار واحدند؟ (Q50–Q52)

**چرا این مرحله لازم شد.** سه مرحله‌ی قبل هر کدام مستقل به یک الگوی هم‌شکل رسیدند:

- Q42: ریسک جمعیت‌شناختی (شبانه‌بودن، تحصیلات تکمیلی) تقریباً فقط **بیرون** خوابگاه وجود
  دارد؛ داخل خوابگاه صاف می‌شود و یکی از آن‌ها حتی علامت عوض می‌کند.
- Q48: تفاوت سلف دانشکده‌ای و خوابگاهی نه در تعطیلات، بلکه در **آخر هفته‌ی کاری**
  (چهارشنبه، و با اغماض پنجشنبه) ظاهر می‌شود.
- F57 (دور ۳): متغیر پنهانِ کل پدیده «غیبت از دانشگاه» است، نه تصمیم درباره‌ی غذا.

اگر این هر سه یک چیز باشند، یک پیش‌بینی **آزمون‌پذیر** دارند: شکاف جمعیت‌شناختی باید دقیقاً
در شرایطی باز شود که «حضور در دانشگاه اختیاری» است، و در شرایطی که نیست بسته بماند. Q50 و
Q51 همین را می‌آزمایند. Q52 سپس می‌پرسد این همه برای سیاست هدف‌گیری چقدر می‌ارزد.

اجرا: `python -m src.eda_lib.runners.r5e_synthesis`
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from src.config import FIGURES_DIR
from src.eda_lib.figio import save_fig
from src.eda_lib.runners._common import PERSON_FACT_PATH, header, kv, pct, setup
from src.viz_fa import fa
from src.viz_fa import setup as viz_setup

FIG_DIR = FIGURES_DIR / "round5"


def build_person_day() -> pd.DataFrame:
    """سطح رزرو (فرد × روز × وعده) + شوک روز + ویژگی جمعیتی فرد."""
    per = pd.read_parquet("data/interim/r5_person_level.parquet")
    day = pd.read_parquet("data/interim/r5_day_level.parquet")

    fact = pd.read_csv(PERSON_FACT_PATH,
                       usecols=["PersonId", "date_gregorian", "Meal",
                                "dont_receive", "is_main_meal"],
                       parse_dates=["date_gregorian"])
    fact = fact[fact["is_main_meal"]].drop(columns=["is_main_meal"])

    dcols = ["date_gregorian", "shock", "Res", "DayOfWeek",
             "is_day_before_holiday", "is_day_after_holiday", "is_exam_period"]
    fact = fact.merge(day[dcols], on="date_gregorian", how="inner")
    pcols = ["PersonId", "is_dorm_resident", "Gender", "DegreeName",
             "edu_group", "eb_rate", "n_res"]
    fact = fact.merge(per[pcols], on="PersonId", how="inner")

    fact["is_wed"] = (fact["DayOfWeek"] == 4)
    fact["night"] = (fact["edu_group"] == "شبانه/نوبت دوم")
    fact["grad"] = fact["DegreeName"].str.contains("ارشد|دکتری")
    fact["male"] = (fact["Gender"] == "مرد")
    # سطل شوک روز: سه‌گانه بر پایه‌ی صدک‌های شوک وزنی روز
    q = fact["shock"].quantile([1 / 3, 2 / 3]).to_numpy()
    fact["shock_bucket"] = pd.cut(fact["shock"], [-np.inf, q[0], q[1], np.inf],
                                  labels=["روز خوب", "روز عادی", "روز بد"])
    return fact


# ---------------------------------------------------------------------------
# Q50 — آیا شکاف جمعیت‌شناختی در «روز بد» باز می‌شود؟
# ---------------------------------------------------------------------------

def q50(f: pd.DataFrame) -> pd.DataFrame:
    header("Q50 — آیا شکاف جمعیت‌شناختی در روزهای بد باز می‌شود؟")
    print("F62 نشان داد اثر *تاریخچه‌ی فردی* با شوک روز ضرب می‌شود (۳.۳۶×). آیا")
    print("جمعیت‌شناسی هم همین رفتار را دارد، یا اثرش جمع‌شونده و ثابت است؟\n")

    rows = []
    for var, lab in [("is_dorm_resident", "ساکن خوابگاه"), ("night", "شبانه"),
                     ("grad", "تحصیلات تکمیلی"), ("male", "مرد")]:
        t = f.groupby(["shock_bucket", var], observed=True)["dont_receive"].mean().unstack()
        gap = (t[True] - t[False])
        for b in t.index:
            rows.append({"ویژگی": lab, "سطل روز": b,
                         "نرخ گروه=بله": t.loc[b, True], "نرخ گروه=خیر": t.loc[b, False],
                         "شکاف": gap[b], "نسبت": t.loc[b, True] / t.loc[b, False]})
    tbl = pd.DataFrame(rows)
    print(tbl.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n— آزمون رسمی برهم‌کنش (لجیت روی رزرو، خوشه‌بندی خطا روی روز) —")
    res = []
    for var, lab in [("is_dorm_resident", "ساکن خوابگاه"), ("night", "شبانه"),
                     ("grad", "تحصیلات تکمیلی"), ("male", "مرد")]:
        d = f[[var, "shock", "dont_receive", "date_gregorian"]].copy()
        d[var] = d[var].astype(float)
        d["dont_receive"] = d["dont_receive"].astype(float)
        m = smf.logit(f"dont_receive ~ {var} * shock", data=d).fit(
            disp=0, cov_type="cluster", cov_kwds={"groups": d["date_gregorian"]})
        res.append({"ویژگی": lab, "اثر اصلی": m.params[var], "شوک": m.params["shock"],
                    "برهم‌کنش": m.params[f"{var}:shock"],
                    "p برهم‌کنش": m.pvalues[f"{var}:shock"]})
    print(pd.DataFrame(res).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return tbl


# ---------------------------------------------------------------------------
# Q51 — آیا مزیت خوابگاه در «روزهای حضور اختیاری» بزرگ‌تر است؟
# ---------------------------------------------------------------------------

def q51(f: pd.DataFrame) -> pd.DataFrame:
    header("Q51 — مزیت خوابگاه در کدام روزها بزرگ‌ترین است؟")
    print("پیش‌بینی سازوکار «غیبت»: هرچه حضور در دانشگاه اختیاری‌تر باشد، مزیت ساکن")
    print("خوابگاه (که برای غذا نیازی به آمدن ندارد) باید بیشتر شود.\n")

    conds = [("is_wed", "چهارشنبه"), ("is_day_before_holiday", "روز پیش از تعطیلی"),
             ("is_day_after_holiday", "روز پس از تعطیلی"), ("is_exam_period", "امتحانات")]
    rows = []
    base = f.groupby("is_dorm_resident")["dont_receive"].mean()
    rows.append({"شرط": "— همه‌ی روزها —", "n رزرو": len(f),
                 "خوابگاهی": base[True], "غیرخوابگاهی": base[False],
                 "مزیت خوابگاه": base[False] - base[True]})
    for c, lab in conds:
        sub = f[f[c].astype(bool)]
        t = sub.groupby("is_dorm_resident")["dont_receive"].mean()
        rows.append({"شرط": lab, "n رزرو": len(sub),
                     "خوابگاهی": t[True], "غیرخوابگاهی": t[False],
                     "مزیت خوابگاه": t[False] - t[True]})
    tbl = pd.DataFrame(rows)
    print(tbl.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n— آزمون رسمی: برهم‌کنش خوابگاه × هر شرط (خطای خوشه‌ای روی روز) —")
    d = f.copy()
    d["dont_receive"] = d["dont_receive"].astype(float)
    d["dorm"] = d["is_dorm_resident"].astype(float)
    for c, _ in conds:
        d[c] = d[c].astype(float)
    terms = " + ".join([f"dorm*{c}" for c, _ in conds])
    m = smf.logit(f"dont_receive ~ {terms}", data=d).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": d["date_gregorian"]})
    out = []
    for c, lab in conds:
        k = f"dorm:{c}"
        out.append({"شرط": lab, "برهم‌کنش با خوابگاه": m.params[k], "p": m.pvalues[k]})
    print(pd.DataFrame(out).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return tbl


# ---------------------------------------------------------------------------
# Q52 — این همه برای سیاست هدف‌گیری چقدر می‌ارزد؟
# ---------------------------------------------------------------------------

def q52(f: pd.DataFrame) -> None:
    header("Q52 — سقف عملی سیاست هدف‌گیری: سه منبع اطلاعات در برابر هم")
    per = pd.read_parquet("data/interim/r5_person_level.parquet")
    total = per["n_norecv"].sum()
    n_top = int(round(0.10 * len(per)))

    def share(score: pd.Series, label: str) -> dict:
        idx = score.sort_values(ascending=False).index[:n_top]
        got = per.loc[idx, "n_norecv"].sum()
        return {"معیار هدف‌گیری": label, "٪ هدررفت پوشش‌داده‌شده": 100 * got / total}

    rows = []
    # سقف نظری: انتخاب کامل بر پایه‌ی خودِ هدررفت (اوراکل)
    rows.append(share(per["n_norecv"], "اوراکل (تعداد واقعی هدررفت)"))
    # تاریخچه‌ی فردی (کوچک‌سازی‌شده) — در دسترسِ لحظه‌ی برش
    rows.append(share(per["eb_rate"] * per["n_res"], "تاریخچه‌ی فردی × حجم"))
    rows.append(share(per["eb_rate"], "فقط نرخ تاریخچه‌ی فردی"))
    # فقط جمعیت‌شناسی (cold-start) — نرخ گروهیِ برازش‌شده
    d = per.copy()
    d["dorm"] = d["is_dorm_resident"].astype(float)
    d["night"] = (d["edu_group"] == "شبانه/نوبت دوم").astype(float)
    d["grad"] = d["DegreeName"].str.contains("ارشد|دکتری").astype(float)
    d["male"] = (d["Gender"] == "مرد").astype(float)
    d["teh"] = d["is_tehran"].astype(float)
    m = smf.ols("eb_rate ~ dorm + night + grad + male + teh + C(CollegeName)", data=d).fit()
    d["demo_pred"] = m.fittedvalues
    rows.append(share(d["demo_pred"], "فقط جمعیت‌شناسی (cold-start)"))
    rows.append(share(d["demo_pred"] * d["n_res"], "جمعیت‌شناسی × حجم"))
    rows.append({"معیار هدف‌گیری": "تصادفی (پایه)", "٪ هدررفت پوشش‌داده‌شده": 10.0})

    tbl = pd.DataFrame(rows).sort_values("٪ هدررفت پوشش‌داده‌شده", ascending=False)
    print("\n(هدف‌گیری بدترین ۱۰٪ افراد؛ عدد = چند درصد از کل پرس هدررفته پوشش داده می‌شود)")
    print(tbl.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    kv("R² جمعیت‌شناسی روی نرخ EB", f"{m.rsquared:.4f}")

    print("\n— برای مقایسه: سقف هدف‌گیری در بعد **روز** —")
    day = pd.read_parquet("data/interim/r5_day_level.parquet")
    tot_d = day["NoRecv"].sum()
    for k in [10, 20, 30]:
        top = day.nlargest(k, "NoRecv")["NoRecv"].sum()
        kv(f"{k} روز بدتر (از {len(day)}) پوشش می‌دهد", f"{100 * top / tot_d:.1f}%")


def figures(f: pd.DataFrame, q50_tbl: pd.DataFrame, q51_tbl: pd.DataFrame) -> None:
    viz_setup()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    order = ["روز خوب", "روز عادی", "روز بد"]
    for lab, mk in [("ساکن خوابگاه", "o"), ("شبانه", "s"),
                    ("تحصیلات تکمیلی", "^"), ("مرد", "d")]:
        t = q50_tbl[q50_tbl["ویژگی"] == lab].set_index("سطل روز").loc[order]
        axes[0].plot(range(3), t["شکاف"], marker=mk, lw=1.8, label=fa(lab))
        axes[1].plot(range(3), t["نسبت"], marker=mk, lw=1.8, label=fa(lab))
    for a, ttl, hl in [(axes[0], "شکاف مطلق نرخ", 0.0), (axes[1], "نسبت نرخ", 1.0)]:
        a.axhline(hl, c="k", lw=1, ls="--")
        a.set_xticks(range(3))
        a.set_xticklabels([fa(s) for s in order])
        a.set_title(fa(ttl))
        a.legend(fontsize=8)
    fig.suptitle(fa("آیا شکاف جمعیت‌شناختی با بدترشدن روز باز می‌شود؟"))
    fig.tight_layout()
    save_fig(fig, "R5_q50_demo_by_shock", FIG_DIR)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    t = q51_tbl
    y = np.arange(len(t))
    ax.barh(y, t["مزیت خوابگاه"], color="#4a7")
    ax.set_yticks(y)
    ax.set_yticklabels([fa(s) for s in t["شرط"]])
    ax.axvline(t.loc[0, "مزیت خوابگاه"], ls="--", c="k", lw=1)
    ax.set_xlabel(fa("مزیت ساکن خوابگاه (واحد نرخ)"))
    ax.set_title(fa("مزیت خوابگاه در کدام شرایط بزرگ‌تر است؟ (خط‌چین: میانگین کل)"))
    fig.tight_layout()
    save_fig(fig, "R5_q51_dorm_advantage", FIG_DIR)
    plt.close(fig)


def main() -> None:
    setup()
    f = build_person_day()
    kv("رزروهای وارد تحلیل", f"{len(f):,}")
    t50 = q50(f)
    t51 = q51(f)
    q52(f)
    figures(f, t50, t51)
    print("\n[ok] مرحله ۶ تمام شد")


if __name__ == "__main__":
    main()
