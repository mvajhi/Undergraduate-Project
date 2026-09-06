"""دور ۵ / مرحله ۴ — «بدترین روز» یعنی کدام روز؟ (Q43–Q45)

**چرا این تحلیل لازم شد.** بند «روزهایی که بدترین به‌نظر می‌رسند، بدترین نیستند» در فصل ۳
گزارش، بر پایه‌ی F54 نوشته شده و یک نکته‌ی *توصیفی* می‌گوید: آن روزها کم‌حجم‌اند. ولی
هیچ‌جا نگفته آن روزها **از نظر تقویمی چه‌اند** — بین دو تعطیلی؟ پیش از بلوک بلند؟ دوره‌ی
امتحانات؟ — و نگفته الگو در سلف خوابگاهی و دانشکده‌ای یکی است یا نه. برای سیاست‌گذاری،
«این روز کم‌حجم بود» بی‌فایده است؛ «این روزها روزِ پس از بلوک تعطیل‌اند» قابل‌اقدام است.

سه رتبه‌بندی رقیب از «بدترین روز» ساخته می‌شود (نرخ خام، هدررفت مطلق، شوک باقیمانده) و
سپس هر فهرست با تقویم تشریح می‌شود.

اجرا: `python -m src.eda_lib.runners.r5c_worst_days`
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

from src.config import FIGURES_DIR
from src.eda_lib.disruption import load_calendar
from src.eda_lib.figio import save_fig
from src.eda_lib.runners._common import DOW_FA, header, kv, load_dataset, pct, setup
from src.viz_fa import fa
from src.viz_fa import setup as viz_setup

FIG_DIR = FIGURES_DIR / "round5"
MIN_RES = 30      # حداقل رزرو سلول (F5.1)
TOP_N = 15        # طول هر فهرست «بدترین»

CAL_FLAGS = [
    ("is_day_before_holiday", "روز پیش از تعطیلی"),
    ("is_day_after_holiday", "روز پس از تعطیلی"),
    ("is_bridge_day", "روز پل"),
    ("is_exam_period", "دوره‌ی امتحانات"),
    ("is_final_exam_period", "امتحانات پایان‌ترم"),
    ("is_midterm_period", "میان‌ترم"),
    ("is_inter_semester_break", "بین دو نیمسال"),
    ("is_nowruz_block", "بلوک نوروز"),
    ("is_add_drop_period", "حذف و اضافه"),
]


# ---------------------------------------------------------------------------
# ساخت پایه
# ---------------------------------------------------------------------------

def build(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """سلول (روز، وعده، سلف) + جدول روز با سه معیار «بدی»."""
    cell = df.groupby(["date_gregorian", "Meal", "RestaurantName"], as_index=False).agg(
        Res=("Res", "sum"), NoRecv=("NoRecv", "sum"),
        DayOfWeek=("DayOfWeek", "first"), RestaurantType=("RestaurantType", "first"))
    cell["rho"] = cell["NoRecv"] / cell["Res"]
    cell = cell[cell["Res"] >= MIN_RES].copy()

    # شوک = باقیمانده پس از حذف اثر ثابت سلف/وعده/روزهفته (رگرسیون وزنی — F5.1)
    cell["dow"] = cell["DayOfWeek"].astype(str)
    m = smf.wls("rho ~ C(RestaurantName) + C(Meal) + C(dow)",
                data=cell, weights=cell["Res"]).fit()
    cell["shock"] = m.resid

    day = cell.groupby("date_gregorian").apply(
        lambda x: pd.Series({
            "Res": x["Res"].sum(), "NoRecv": x["NoRecv"].sum(),
            "shock": np.average(x["shock"], weights=x["Res"]),
            "n_cells": len(x), "DayOfWeek": x["DayOfWeek"].iloc[0],
        }), include_groups=False).reset_index()
    day["rho"] = day["NoRecv"] / day["Res"]

    cal = load_calendar()
    day = day.merge(cal, on="date_gregorian", how="left", validate="one_to_one")
    day["dow_name"] = day["DayOfWeek"].map(DOW_FA)
    day["vol_pctile"] = 100 * day["Res"].rank(pct=True)
    return cell, day


# ---------------------------------------------------------------------------
# Q43 — سه رتبه‌بندی
# ---------------------------------------------------------------------------

RANKS = [("rho", "الف) نرخ خام", False),
         ("NoRecv", "ب) هدررفت مطلق", False),
         ("shock", "ج) شوک باقیمانده", False)]


def q43(day: pd.DataFrame) -> pd.DataFrame:
    header("Q43 — سه رتبه‌بندی رقیب از «بدترین روز»")
    kv("روزهای سرویس", len(day))
    kv("کل هدررفت (پرس)", f"{int(day['NoRecv'].sum()):,}")

    for col, label, _ in RANKS:
        day[f"top_{col}"] = False
        idx = day.sort_values(col, ascending=False).index[:TOP_N]
        day.loc[idx, f"top_{col}"] = True

    print(f"\n— همپوشانی فهرست‌های {TOP_N}تایی —")
    for a, b in [("rho", "NoRecv"), ("rho", "shock"), ("NoRecv", "shock")]:
        n = int((day[f"top_{a}"] & day[f"top_{b}"]).sum())
        kv(f"{a} ∩ {b}", f"{n} از {TOP_N}")

    for col, label, _ in RANKS:
        print(f"\n===== {label} — {TOP_N} روز بالا =====")
        cols = ["date_jalali", "dow_name", "Res", "NoRecv", "rho", "shock",
                "vol_pctile", "holiday_name", "days_to_next_holiday",
                "days_since_last_holiday", "holiday_block_length"]
        t = day.sort_values(col, ascending=False).head(TOP_N)[cols]
        print(t.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return day


# ---------------------------------------------------------------------------
# Q44 — آناتومی تقویمی هر فهرست
# ---------------------------------------------------------------------------

def q44(day: pd.DataFrame) -> pd.DataFrame:
    header("Q44 — آناتومی تقویمی: چه چیزی هر فهرست را می‌سازد؟")
    rows = []
    base = {}
    for flag, label in CAL_FLAGS:
        base[label] = float(day[flag].mean())

    for col, label, _ in RANKS:
        sub = day[day[f"top_{col}"]]
        rec = {"فهرست": label}
        for flag, lab in CAL_FLAGS:
            rec[lab] = float(sub[flag].mean())
        rec["میانه‌ی صدک حجم"] = float(sub["vol_pctile"].median())
        rec["میانه‌ی فاصله تا تعطیلی بعد"] = float(sub["days_to_next_holiday"].median())
        rec["میانه‌ی فاصله از تعطیلی قبل"] = float(sub["days_since_last_holiday"].median())
        rec["٪ پنجشنبه"] = float((sub["dow_name"] == "پنجشنبه").mean())
        rec["٪ چهارشنبه"] = float((sub["dow_name"] == "چهارشنبه").mean())
        rows.append(rec)
    rec = {"فهرست": "— پایه (همه‌ی روزها) —"}
    rec.update(base)
    rec["میانه‌ی صدک حجم"] = float(day["vol_pctile"].median())
    rec["میانه‌ی فاصله تا تعطیلی بعد"] = float(day["days_to_next_holiday"].median())
    rec["میانه‌ی فاصله از تعطیلی قبل"] = float(day["days_since_last_holiday"].median())
    rec["٪ پنجشنبه"] = float((day["dow_name"] == "پنجشنبه").mean())
    rec["٪ چهارشنبه"] = float((day["dow_name"] == "چهارشنبه").mean())
    rows.append(rec)

    tbl = pd.DataFrame(rows).set_index("فهرست").T
    print("\n(نسبت روزهای هر فهرست که پرچم را دارند)")
    print(tbl.to_string(float_format=lambda v: f"{v:.3f}"))

    # آزمون فیشر برای پرچم‌های کلیدی، هر فهرست در برابر بقیه‌ی روزها
    print("\n— آزمون دقیق فیشر: هر پرچم × هر فهرست —")
    res = []
    for col, label, _ in RANKS:
        for flag, lab in CAL_FLAGS:
            a = int((day[f"top_{col}"] & day[flag]).sum())
            b = int((day[f"top_{col}"] & ~day[flag]).sum())
            c = int((~day[f"top_{col}"] & day[flag]).sum())
            d = int((~day[f"top_{col}"] & ~day[flag]).sum())
            if a + c == 0:
                continue
            odds, p = stats.fisher_exact([[a, b], [c, d]])
            if p < 0.20:
                res.append({"فهرست": label, "پرچم": lab, "در فهرست": f"{a}/{TOP_N}",
                            "نسبت شانس": odds, "p": p})
    print(pd.DataFrame(res).to_string(index=False, float_format=lambda v: f"{v:.4f}")
          if res else "  هیچ پرچمی p<0.20 نداد")
    return tbl


# ---------------------------------------------------------------------------
# Q45 — آیا «بدترین ظاهری» چیزی بیش از کم‌حجمی است؟
# ---------------------------------------------------------------------------

def q45(day: pd.DataFrame) -> None:
    header("Q45 — آیا فهرست ظاهری پس از کنترل حجم، امضای تقویمی دارد؟")
    d = day.copy()
    d["log_res"] = np.log(d["Res"])
    d["y"] = d["top_rho"].astype(float)

    feats = ["log_res", "is_day_before_holiday", "is_day_after_holiday",
             "is_exam_period", "is_inter_semester_break", "is_nowruz_block"]

    # ⚠️ جدایی کامل: هیچ روزِ فهرست ظاهری در دوره‌ی امتحانات نیست، پس MLE بی‌کران می‌شود
    # (ضریب −۱۸.۷ و شکست وارون‌سازی هسیان در نسخه‌ی اول). خودِ این جدایی یک یافته است،
    # نه نقص — پس هم گزارش می‌شود و هم با جریمه‌ی L2 برآورد کران‌دار گرفته می‌شود.
    print("— بررسی جدایی کامل (پیش‌نیاز تفسیر ضرایب) —")
    sep = []
    for f in feats[1:]:
        a = int((d["y"].astype(bool) & d[f].astype(bool)).sum())
        n_flag = int(d[f].astype(bool).sum())
        sep.append({"پرچم": f, "روز پرچم‌دار": n_flag, "از فهرست ظاهری": a,
                    "جدایی کامل؟": "بله" if (a == 0 and n_flag > 0) else "خیر"})
    print(pd.DataFrame(sep).to_string(index=False))

    X0 = sm.add_constant(d[["log_res"]])
    m0 = sm.Logit(d["y"], X0).fit(disp=0)
    kv("فقط حجم — pseudo-R²", f"{m0.prsquared:.4f}")
    kv("ضریب log(حجم)", f"{m0.params['log_res']:+.4f} (p={m0.pvalues['log_res']:.2e})")

    X1 = sm.add_constant(d[feats].astype(float))
    m1 = sm.Logit(d["y"], X1).fit_regularized(disp=0, alpha=1.0, L1_wt=0.0, maxiter=500)
    llf1 = sm.Logit(d["y"], X1).loglike(m1.params)
    kv("حجم + تقویم (جریمه‌ی L2) — log-lik", f"{llf1:.3f}")
    print("\n(ضرایب با جریمه‌ی L2؛ به‌دلیل جدایی، خطای معیار تفسیرپذیر نیست)")
    print(pd.DataFrame({"ضریب (L2)": m1.params}).to_string(float_format=lambda v: f"{v:.4f}"))

    # آزمون معتبر برای «تقویم فراتر از حجم»: جایگشت **همتاشده بر رتبه‌ی حجم**.
    # ⚠️ نسخه‌ی اول این آزمون وزن ۱/حجم داشت که کل جرم را روی کم‌حجم‌ترین روزها می‌برد و
    # نولی می‌ساخت که خودش از نمونه‌ی واقعی کم‌حجم‌تر بود. همتاسازی رتبه‌ای این را ندارد:
    # به‌ازای هر روزِ فهرست، یک روز با حجم مشابه (±۷ رتبه) قرعه می‌شود.
    rng = np.random.default_rng(42)
    cal_only = feats[1:]
    d = d.sort_values("Res").reset_index(drop=True)
    d["vol_rank"] = np.arange(len(d))
    obs = float(d.loc[d["y"] == 1, cal_only].sum(axis=1).mean())
    target_ranks = d.loc[d["y"] == 1, "vol_rank"].to_numpy()
    null = np.empty(5000)
    for i in range(5000):
        picks = []
        for r in target_ranks:
            lo, hi = max(0, r - 7), min(len(d) - 1, r + 7)
            cand = [c for c in range(lo, hi + 1) if c not in picks]
            picks.append(rng.choice(cand))
        null[i] = float(d.iloc[picks][cal_only].sum(axis=1).mean())
    p_perm = float((null >= obs).mean())
    kv("میانگین پرچم تقویمی در فهرست ظاهری", f"{obs:.3f}")
    kv("میانگین تحت نولِ همتاشده بر حجم", f"{null.mean():.3f}")
    kv("p جایگشتی (یک‌دامنه)", f"{p_perm:.4f}")

    print("\n— همبستگی سه معیار با حجم روز —")
    for col, label, _ in RANKS:
        r, p = stats.spearmanr(day["Res"], day[col])
        kv(f"اسپیرمن(حجم، {label})", f"{r:+.4f} (p={p:.2e})")


def figures(day: pd.DataFrame) -> None:
    viz_setup()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    d = day.sort_values("date_gregorian")

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    specs = [("rho", "نرخ عدم‌دریافت روزانه", "#c44"),
             ("NoRecv", "هدررفت مطلق (پرس)", "#47c"),
             ("shock", "شوک باقیمانده", "#4a7")]
    for ax, (col, lab, c) in zip(axes, specs):
        ax.plot(d["date_gregorian"], d[col], lw=1, color="#888")
        top = d[d[f"top_{col}"]]
        ax.scatter(top["date_gregorian"], top[col], color=c, zorder=5, s=34)
        ax.set_ylabel(fa(lab))
        ax.set_title(fa(f"{lab} — {TOP_N} روز بدتر با نقطه‌ی رنگی"))
    axes[-1].set_xlabel(fa("تاریخ"))
    fig.suptitle(fa("سه تعریف «بدترین روز» سه فهرست تقریباً مجزا می‌دهند"))
    fig.tight_layout()
    save_fig(fig, "R5_q43_three_rankings", FIG_DIR)
    plt.close(fig)

    # حجم در برابر نرخ، با رنگ‌بندی فهرست
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    ax.scatter(d["Res"], d["rho"], s=18, c="#bbb", label=fa("سایر روزها"))
    for col, lab, c in specs:
        t = d[d[f"top_{col}"]]
        ax.scatter(t["Res"], t["rho"], s=48, c=c, label=fa(f"بدتر بر اساس {lab}"))
    ax.set_xlabel(fa("حجم رزرو روزانه"))
    ax.set_ylabel(fa("نرخ عدم‌دریافت روزانه"))
    ax.set_title(fa("چرا رتبه‌بندی بر پایه‌ی نرخ، تصمیم‌گیر را گمراه می‌کند"))
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, "R5_q43_volume_vs_rate", FIG_DIR)
    plt.close(fig)


def main() -> None:
    setup()
    df = load_dataset()
    cell, day = build(df)
    day = q43(day)
    q44(day)
    q45(day)
    figures(day)
    day.to_parquet("data/interim/r5_day_level.parquet", index=False)
    cell.to_parquet("data/interim/r5_cell_level.parquet", index=False)
    print("\n[ok] مرحله ۴ تمام شد")


if __name__ == "__main__":
    main()
