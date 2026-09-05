"""دور ۴ — رفتار حول اختلال سرویس: روزهای شوک، روزهای قطع، و روزهای بازگشت.

**چرا این دور لازم شد.** آرشیو ۳۹ رویدادی (`events.csv`، ردیف ۲۰ decision_log) ستون
`observed_in_data` دارد که برای هر رویداد $\\rho$ و z آن را ثبت می‌کند، ولی این ستون یک
*توصیف* است نه یک *آزمون*: نه فاصله‌ی اطمینان دارد، نه گروه کنترل همتا، و نه چیزی درباره‌ی
روزهای **پس از** رویداد می‌گوید. سه سؤال بی‌پاسخ ماند:

1. آیا روزی در بازه‌ی داده هست که رزروِ بسته‌شده را یک قطع ناگهانی بلاتکلیف گذاشته باشد؟
2. اثر روزهای شوکِ با سرو کامل چقدر است، با فاصله‌ی اطمینان، و چند روز دوام می‌آورد؟
3. پس از یک شکاف سرویس، نرخ عدم‌دریافت در روزهای بازگشت چه می‌کند؟

سؤال ۳ مستقیماً به یک بدهی شناخته‌شده‌ی پروژه وصل است: ردیف ۳۲ و ۴۹ decision_log هر دو
پنجره‌ی بازگشتِ پس از رمضان را «رژیم غیرعادی» خواندند و بر همان اساس پروتکل اعتبارسنجی را
عوض کردند — ولی خودِ آن غیرعادی‌بودن هرگز اندازه‌گیری نشد.

اجرا: `python -m src.eda_lib.runners.r4_disruption`
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from src.config import FIGURES_DIR
from src.eda_lib.disruption import (
    add_run_position,
    day_disruption_table,
    expand_events,
    load_events,
    meal_gap_table,
    unit_service_matrix,
)
from src.eda_lib.figio import save_fig
from src.eda_lib.runners._common import DOW_FA, boot_ci, header, kv, load_dataset, pct, setup
from src.viz_fa import fa
from src.viz_fa import setup as viz_setup

MIN_RES = 30          # حداقل رزرو یک سلول تا نرخش نویز مخرج کوچک نباشد (F5.1)
FIG_DIR = FIGURES_DIR / "round4"


def wmean(x: np.ndarray, w: np.ndarray) -> float:
    return float(np.average(x, weights=w))


def wboot_ci(x, w, n_boot: int = 4000, alpha: float = 0.05, seed: int = 42):
    """فاصله‌ی اطمینان bootstrap برای **میانگین وزنی**.

    ⚠️ چرا وزنی و نه ساده: میانگین ساده‌ی شوکِ سلول‌ها، سلف کوچک را هم‌وزن سلف بزرگ
    می‌کند و دقیقاً همان مصنوعی را برمی‌گرداند که F5.1 هشدار داده بود (نرخ پرت با
    اندازه‌ی رزرو معکوس است). در نسخه‌ی اول این تحلیل، میانگین ساده اثر «روز اول
    بازگشت» را ۵ برابر بزرگ‌تر نشان می‌داد (+۰.۰۷۸ در برابر +۰.۰۱۵ وزنی).
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    stats_ = np.array([np.average(x[i], weights=w[i]) for i in idx])
    lo, hi = np.percentile(stats_, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return wmean(x, w), float(lo), float(hi)


# ---------------------------------------------------------------------------
# پایه: شوک روز×وعده
# ---------------------------------------------------------------------------

def build_cells(df: pd.DataFrame) -> pd.DataFrame:
    """سطح (روز، وعده، سلف) با نرخ و وزن رزرو."""
    cell = df.groupby(["date_gregorian", "Meal", "RestaurantName"], as_index=False).agg(
        Res=("Res", "sum"), NoRecv=("NoRecv", "sum"), DayOfWeek=("DayOfWeek", "first"))
    cell["rho"] = cell["NoRecv"] / cell["Res"]
    return cell[cell["Res"] >= MIN_RES].copy()


def build_shock(cell: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """شوک = باقیمانده‌ی نرخ پس از حذف اثر ثابت سلف، وعده و روزِ هفته.

    رگرسیون وزنی است (وزن = تعداد رزرو) تا سلف کوچک به‌اندازه‌ی سلف بزرگ در تعریف
    «روز عادی» سهم نداشته باشد — همان قاعده‌ای که F5.1 تحمیل می‌کند.
    """
    c = cell.copy()
    c["dow"] = c["DayOfWeek"].astype(str)
    model = smf.wls("rho ~ C(RestaurantName) + C(Meal) + C(dow)", data=c, weights=c["Res"]).fit()
    c["shock"] = model.resid
    meal_day = (c.groupby(["date_gregorian", "Meal"])
                .apply(lambda x: pd.Series({
                    "shock": np.average(x["shock"], weights=x["Res"]),
                    "Res": x["Res"].sum(), "n_cells": len(x)}), include_groups=False)
                .reset_index())
    return c, meal_day


# ---------------------------------------------------------------------------
# Q29 — تبارشناسی اختلال
# ---------------------------------------------------------------------------

def q29_taxonomy(day: pd.DataFrame) -> pd.DataFrame:
    header("Q29 — در ۱۴۲ روزِ سرویس، «اختلال» چند نوع است و هرکدام چند روز؟")
    tab = (day.groupby("disruption_class")
           .agg(n_days=("date_gregorian", "size"),
                n_with_event=("n_events", lambda s: int((s > 0).sum())))
           .reset_index().sort_values("n_days", ascending=False))
    tab["سهم روز"] = (tab["n_days"] / len(day)).map(pct)
    print(tab.to_string(index=False))

    hard = day[day["disruption_class"].isin(["قطع_کامل", "قطع_گسترده"])]
    non_ramadan = hard[~hard["event_ids"].fillna("").str.contains("ramadan")]
    kv("روزهای قطع کامل/گسترده", len(hard))
    kv("از این‌ها خارج از رمضان", len(non_ramadan))
    print()
    print("روزهای قطع خارج از رمضان:")
    print(non_ramadan[["date_gregorian", "date_jalali", "n_expected", "n_closed",
                       "disruption_class", "event_ids"]].to_string(index=False))
    print()
    print("> رمضان تنها منبع قطع گسترده است. تنها استثنا 2024-02-10 است که هیچ رویداد")
    print("> ثبت‌شده‌ای ندارد و در تقویم هم تعطیل نیست.")
    return tab


# ---------------------------------------------------------------------------
# Q30 — آیا قطعی، رزرو بسته‌شده را بلاتکلیف گذاشت؟
# ---------------------------------------------------------------------------

def q30_stranded_reservations(day: pd.DataFrame, fact: pd.DataFrame) -> pd.DataFrame:
    header("Q30 — آیا هیچ قطعی رزروِ از پیش بسته‌شده را بلاتکلیف گذاشت؟")
    print("رزرو ۷۲ ساعت زودتر بسته می‌شود، پس یک قطع ناگهانی باید ردی در داده‌ی سطح فرد")
    print("بگذارد: رزروی که ثبت شده ولی هیچ‌کس دریافت نکرده. این آزمونِ وجودِ سناریوی")
    print("«تعطیلی ناگهانی با رزروِ سوخته» است.")
    print()
    hard = day[day["disruption_class"].isin(["قطع_کامل", "قطع_گسترده"])]
    rows = []
    for r in hard.itertuples(index=False):
        sub = fact[fact["date_gregorian"] == r.date_gregorian]
        for meal in ["lunch", "dinner"]:
            s_ = sub[sub["Meal"] == meal]
            rows.append({"date_gregorian": r.date_gregorian, "date_jalali": r.date_jalali,
                         "class": r.disruption_class, "Meal": meal,
                         "n_reservations": len(s_),
                         "n_received": int((~s_["dont_receive"]).sum())})
    out = pd.DataFrame(rows)
    out["stranded"] = (out["n_reservations"] > 0) & (out["n_received"] == 0)
    print(out[out["n_reservations"] > 0].to_string(index=False))
    print()
    kv("جفت (روز، وعده) با رزرو ولی صفر دریافت", int(out["stranded"].sum()))
    kv("مجموع رزرو سوخته", int(out.loc[out["stranded"], "n_reservations"].sum()))
    print()
    print("> ⭐ در ۱۱ روز از ۱۲ روزِ قطع، تعداد رزرو هم صفر است — یعنی سرویس *پیش از*")
    print("> باز شدن رزرو برداشته شده بود (رمضان: ناهار از قبل حذف). این قطع نیست، برنامه است.")
    print("> تنها استثنای واقعی 1403-01-21 ناهار است: ۱۶۳ رزرو، صفر دریافت، همه با وضعیت")
    print("> «منقضی شده» و همه در یک سلف. یعنی سناریوی «رزروِ سوخته» در بازه‌ی قفل‌شده")
    print("> دقیقاً **یک‌بار** و در **یک سلف** رخ داده: ۱۶۳ رزرو از ۲٬۰۴۹٬۳۲۲ (%s)."
          % pct(163 / 2_049_322))
    print("> پیامد مدل‌سازی: با یک مشاهده، این سناریو آموختنی نیست و باید بیرون از مدل")
    print("> (قاعده‌ی عملیاتی) مدیریت شود — نه با فیچر و نه با وزن‌دهی.")
    return out


# ---------------------------------------------------------------------------
# Q31 — اثر روزهای شوکِ با سرو کامل، با فاصله‌ی اطمینان
# ---------------------------------------------------------------------------

SHOCK_EVENTS = {
    "kerman_bombing": "انفجار کرمان",
    "kerman_mourning_day": "عزای عمومی کرمان",
    "raisi_helicopter_crash": "سوگواری سقوط بالگرد",
    "iran_strike_israel": "حمله‌ی ایران به اسرائیل",
    "bus_drivers_strike": "اعتصاب اتوبوسرانی",
    "snow_episode_01": "برف ۸ بهمن",
    "snow_episode_02": "برف ۱۲–۱۳ بهمن",
    "snow_episode_04": "برف ۹ اسفند",
}


def q31_event_study(meal_day: pd.DataFrame, horizon: int = 3) -> pd.DataFrame:
    header("Q31 — روزهای شوکِ با سرو کامل: اثر روی روز رویداد و تا ۳ روز بعد")
    ev = load_events()
    ev_days = expand_events(ev)
    served = ev[ev["service_status"] == "سرو_کامل"]["event_id"].tolist()

    sd = meal_day["shock"].std()
    kv("انحراف معیار شوک روز×وعده", f"{sd:.4f} (واحد نرخ)")
    base = meal_day["shock"]
    kv("میانگین شوک همه‌ی روزها", f"{base.mean():+.4f}")
    print()

    idx = meal_day.set_index(["date_gregorian", "Meal"])["shock"]
    rows = []
    for eid, label in SHOCK_EVENTS.items():
        if eid not in served:
            continue
        dates = sorted(ev_days.loc[ev_days["event_id"] == eid, "date_gregorian"])
        for h in range(horizon + 1):
            vals = []
            for d0 in dates:
                target = d0 + pd.Timedelta(h, unit="D")
                for meal in ["lunch", "dinner"]:
                    if (target, meal) in idx.index:
                        vals.append(idx.loc[(target, meal)])
            if vals:
                rows.append({"event": label, "h": h, "n_obs": len(vals),
                             "shock_mean": float(np.mean(vals)),
                             "in_sd": float(np.mean(vals) / sd)})
    out = pd.DataFrame(rows)
    piv = out.pivot(index="event", columns="h", values="in_sd").round(2)
    piv.columns = [f"h={c}" for c in piv.columns]
    print("اثر بر حسب انحراف معیار شوک روزانه (h=0 روز رویداد):")
    print(piv.to_string())
    print()

    # آزمون تجمیعی: آیا مجموعه‌ی روزهای شوک با بقیه فرق دارد؟
    ev_dates = set(ev_days[ev_days["event_id"].isin([e for e in SHOCK_EVENTS if e in served])]["date_gregorian"])
    md = meal_day.copy()
    md["is_event"] = md["date_gregorian"].isin(ev_dates)
    a = md.loc[md["is_event"], "shock"].values
    b = md.loc[~md["is_event"], "shock"].values
    diff = a.mean() - b.mean()
    t, p = stats.ttest_ind(a, b, equal_var=False)
    pooled = np.sqrt((a.var(ddof=1) * (len(a) - 1) + b.var(ddof=1) * (len(b) - 1)) / (len(a) + len(b) - 2))
    d_cohen = diff / pooled
    rng = np.random.default_rng(42)
    boots = np.array([rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(4000)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    kv("n روزِ شوک (روز×وعده)", len(a))
    kv("اختلاف میانگین شوک", f"{diff:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    kv("اندازه اثر Cohen's d", f"{d_cohen:+.3f}")
    kv("t-test Welch p", f"{p:.3f}")
    print()
    if lo <= 0 <= hi:
        print("> ⭐ فاصله‌ی اطمینان صفر را در بر می‌گیرد: روزهای شوکِ خبری، در سطح تجمیعی")
        print("> اثر قابل‌تشخیصی روی نرخ عدم‌دریافت ندارند. برچسب «روز پرالتهاب» اطلاعاتی")
        print("> فراتر از تقویم و هویت سلف به مدل اضافه نمی‌کند.")
    return out


# ---------------------------------------------------------------------------
# Q32 — روزهای بازگشت پس از شکاف
# ---------------------------------------------------------------------------

def q32_return_curve(cell: pd.DataFrame, cell_shock: pd.DataFrame,
                     gap_min: int = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
    header(f"Q32 — پس از یک شکاف ≥{gap_min} روزه، نرخ در روزهای بازگشت چه می‌کند؟")
    gaps = add_run_position(meal_gap_table(cell), gap_min=gap_min)
    cs = cell_shock.merge(gaps[["date_gregorian", "Meal", "run_pos", "gap_len", "gap_days"]],
                          on=["date_gregorian", "Meal"], how="left")

    print("شکاف‌های شناسایی‌شده:")
    g = gaps[gaps["gap_days"] >= gap_min][["Meal", "date_gregorian", "gap_days"]]
    print(g.sort_values("date_gregorian").to_string(index=False))
    print()

    rows = []
    for pos in range(1, 6):
        sub = cs[cs["run_pos"] == pos]
        if len(sub) == 0:
            continue
        m, lo, hi = wboot_ci(sub["shock"].values, sub["Res"].values)
        rows.append({"روز بازگشت": pos, "n_cell": len(sub), "Res": int(sub["Res"].sum()),
                     "rho": sub["NoRecv"].sum() / sub["Res"].sum(),
                     "شوک": m, "lo": lo, "hi": hi})
    far = cs[(cs["run_pos"].isna()) | (cs["run_pos"] > 5)]
    m, lo, hi = wboot_ci(far["shock"].values, far["Res"].values)
    rows.append({"روز بازگشت": "پایه (>۵)", "n_cell": len(far), "Res": int(far["Res"].sum()),
                 "rho": far["NoRecv"].sum() / far["Res"].sum(),
                 "شوک": m, "lo": lo, "hi": hi})
    out = pd.DataFrame(rows)
    out[["شوک", "lo", "hi", "rho"]] = out[["شوک", "lo", "hi", "rho"]].round(4)
    print(out.to_string(index=False))
    print()

    fs = cs[cs["run_pos"].isin([1, 2, 3, 4])]
    diff = wmean(fs["shock"].values, fs["Res"].values) - wmean(far["shock"].values, far["Res"].values)
    rng = np.random.default_rng(42)
    boots = []
    for _ in range(4000):
        i = rng.integers(0, len(fs), len(fs))
        j = rng.integers(0, len(far), len(far))
        boots.append(np.average(fs["shock"].values[i], weights=fs["Res"].values[i])
                     - np.average(far["shock"].values[j], weights=far["Res"].values[j]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    kv("چهار روز اول بازگشت − پایه (وزنی)", f"{diff:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    kv("نرخ خام چهار روز اول", f"{fs['NoRecv'].sum() / fs['Res'].sum():.4f}")
    kv("نرخ خام پایه", f"{far['NoRecv'].sum() / far['Res'].sum():.4f}")
    kv("به نسبت نرخ پایه", pct(diff / (far["NoRecv"].sum() / far["Res"].sum())))
    return cs, out


# ---------------------------------------------------------------------------
# Q33 — طول شکاف: پاسخ مدرج یا آستانه‌ای؟
# ---------------------------------------------------------------------------

def q33_gap_dose_response(cs: pd.DataFrame) -> pd.DataFrame:
    header("Q33 — آیا اثر «بازگشت» با طول شکاف مدرج می‌شود؟")
    cs = cs.copy()

    def bucket(g):
        if pd.isna(g):
            return "نامعلوم"
        if g == 1:
            return "۱ روز (پیوسته)"
        if g == 2:
            return "۲ روز"
        if g == 3:
            return "۳ روز (آخر هفته)"
        if g <= 7:
            return "۴–۷ روز"
        if g <= 20:
            return "۸–۲۰ روز"
        return "بیش از ۲۰ روز"

    cs["gap_bucket"] = cs["gap_days"].map(bucket)
    order = ["۱ روز (پیوسته)", "۲ روز", "۳ روز (آخر هفته)", "۴–۷ روز", "۸–۲۰ روز", "بیش از ۲۰ روز"]
    rows = []
    for b in order:
        sub = cs[cs["gap_bucket"] == b]
        if len(sub) < 3:
            continue
        m, lo, hi = wboot_ci(sub["shock"].values, sub["Res"].values)
        rows.append({"شکاف پیش از این روز": b, "n_cell": len(sub), "Res": int(sub["Res"].sum()),
                     "rho": round(sub["NoRecv"].sum() / sub["Res"].sum(), 4),
                     "شوک": round(m, 4), "lo": round(lo, 4), "hi": round(hi, 4)})
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    print()
    sub = cs[cs["gap_days"].notna()]
    r, p = stats.spearmanr(sub["gap_days"], sub["shock"])
    kv("همبستگی اسپیرمن طول شکاف ↔ شوک", f"{r:+.3f} (p={p:.4f})")
    return out


# ---------------------------------------------------------------------------
# Q34 — سطح فرد: بازگشت را چه کسی بد می‌کند؟
# ---------------------------------------------------------------------------

def q34_who_defects_on_return(fact: pd.DataFrame, cs: pd.DataFrame) -> pd.DataFrame:
    header("Q34 — نرخ بالای روزهای بازگشت را چه کسی می‌سازد؟ (سطح فرد)")
    print("دو فرضیه‌ی رقیب — (الف) همه کمی بدتر می‌شوند، (ب) همان اقلیت بی‌ثبات F62 بدتر")
    print("می‌شود و بقیه دست‌نخورده می‌مانند. تفاوتشان برای تصمیم پخت مهم است.")
    print()
    ret_days = set(cs.loc[cs["run_pos"].isin([1, 2, 3, 4]), "date_gregorian"])

    f = fact.sort_values(["PersonId", "date_gregorian"]).copy()
    csum = f.groupby("PersonId")["dont_receive"].cumsum() - f["dont_receive"]
    cnt = f.groupby("PersonId").cumcount()
    f["hist"] = np.where(cnt >= 5, csum / cnt.replace(0, np.nan), np.nan)
    f = f[f["hist"].notna()].copy()
    f["is_return"] = f["date_gregorian"].isin(ret_days)

    f["hist_q"] = pd.qcut(f["hist"], 4, labels=["Q1 وفادارترین", "Q2", "Q3", "Q4 بی‌ثبات‌ترین"])
    tab = (f.groupby(["hist_q", "is_return"], observed=True)["dont_receive"]
           .agg(["mean", "size"]).reset_index())
    piv = tab.pivot(index="hist_q", columns="is_return", values="mean")
    npiv = tab.pivot(index="hist_q", columns="is_return", values="size")
    piv.columns = ["روز عادی", "روز بازگشت"]
    piv["اختلاف (واحد درصد)"] = (100 * (piv["روز بازگشت"] - piv["روز عادی"])).round(2)
    piv["نسبت"] = (piv["روز بازگشت"] / piv["روز عادی"]).round(2)
    piv["n روز بازگشت"] = npiv[True].values
    print(piv.round(4).to_string())
    print()

    m = smf.logit("dont_receive ~ hist + is_return + hist:is_return", data=f.assign(
        dont_receive=f["dont_receive"].astype(int), is_return=f["is_return"].astype(int))).fit(disp=0)
    kv("ضریب برهم‌کنش hist×بازگشت", f"{m.params['hist:is_return']:+.3f} (p={m.pvalues['hist:is_return']:.4f})")
    return piv


# ---------------------------------------------------------------------------
# نمودار
# ---------------------------------------------------------------------------

def make_figures(meal_day: pd.DataFrame, ret: pd.DataFrame, dose: pd.DataFrame, day: pd.DataFrame) -> None:
    viz_setup()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ⚠️ دو تله‌ی رسم که در نسخه‌ی اول این نمودار افتادیم:
    # (۱) matplotlib روزهای بی‌سرویس را با خط مستقیم وصل می‌کند و شکاف ۲۹ روزه‌ی رمضان
    #     به شکل یک روند نزولیِ جعلی درمی‌آید — با reindex روی بازه‌ی کامل و NaN شکسته می‌شود؛
    # (۲) شوک شام در ۱۴۰۲-۱۱-۲۱ برابر ۰.۹۵ است و کل محور را می‌بلعد، در حالی که آن روز
    #     فقط ۲ ردیف و ۱۳۳ رزرو دارد (F54/F5.1). محور به بازه‌ی معنادار محدود و آن نقطه
    #     برچسب‌گذاری می‌شود، نه اینکه حذف شود.
    full = pd.date_range(meal_day["date_gregorian"].min(), meal_day["date_gregorian"].max(), freq="D")
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    hard = day[day["disruption_class"].isin(["قطع_کامل", "قطع_گسترده"])]
    for ax, meal, lab in zip(axes, ["lunch", "dinner"], ["ناهار", "شام"]):
        s_ = (meal_day[meal_day["Meal"] == meal]
              .set_index("date_gregorian")["shock"].reindex(full))
        ax.axhline(0, color="#999", lw=0.8)
        for d0 in hard["date_gregorian"]:
            ax.axvspan(d0, d0 + pd.Timedelta(1, unit="D"), color="#e53e3e", alpha=0.12)
        ax.plot(s_.index, s_.values, lw=1.2, color="#2b6cb0")
        lo, hi = np.nanpercentile(s_.values, [0.5, 99]) if s_.notna().any() else (0, 1)
        pad = 0.25 * (hi - lo)
        ax.set_ylim(min(lo - pad, -0.05), hi + pad)
        out = s_[s_ > hi + pad]
        for d0, v in out.items():
            ax.annotate(fa(f"{v:.2f} ↑"), xy=(d0, hi + pad), xytext=(0, -12),
                        textcoords="offset points", ha="center", fontsize=8, color="#c53030")
        ax.set_ylabel(fa("شوک روز (واحد نرخ)"))
        ax.set_title(fa(f"سری شوک روزانه — {lab}  (نوار قرمز: روز قطع گسترده/کامل؛ خط بریده: روز بی‌سرویس)"))
    axes[1].set_xlabel(fa("تاریخ"))
    fig.tight_layout()
    save_fig(fig, "r4_1_shock_series_disruption", FIG_DIR)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    d = ret[ret["روز بازگشت"] != "پایه (>۵)"]
    x = np.arange(len(d))
    ax.errorbar(x, d["شوک"], yerr=[d["شوک"] - d["lo"], d["hi"] - d["شوک"]],
                fmt="o-", capsize=4, color="#2b6cb0")
    basev = float(ret.loc[ret["روز بازگشت"] == "پایه (>۵)", "شوک"].iloc[0])
    ax.axhline(basev, color="#e53e3e", ls="--", lw=1.2, label=fa("روزهای عادی"))
    ax.set_xticks(x)
    ax.set_xticklabels([fa(f"روز {int(v)}") for v in d["روز بازگشت"]])
    ax.set_ylabel(fa("شوک (واحد نرخ)"))
    ax.set_title(fa("نرخ عدم‌دریافت در روزهای بازگشت پس از شکاف سرویس"))
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "r4_2_return_curve", FIG_DIR)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.errorbar(np.arange(len(dose)), dose["شوک"],
                yerr=[dose["شوک"] - dose["lo"], dose["hi"] - dose["شوک"]],
                fmt="s-", capsize=4, color="#805ad5")
    ax.axhline(0, color="#999", lw=0.8)
    ax.set_xticks(np.arange(len(dose)))
    ax.set_xticklabels([fa(v) for v in dose["شکاف پیش از این روز"]], rotation=20, ha="right")
    ax.set_ylabel(fa("شوک (واحد نرخ)"))
    ax.set_title(fa("پاسخ مدرج: شوک بر حسب طول شکاف پیش از آن روز"))
    fig.tight_layout()
    save_fig(fig, "r4_3_gap_dose_response", FIG_DIR)
    plt.close(fig)
    print(f"\n[fig] سه نمودار در {FIG_DIR}")


def main() -> None:
    setup()
    df = load_dataset()
    cell = build_cells(df)
    cell_shock, meal_day = build_shock(cell)
    day = day_disruption_table()

    fact = pd.read_csv(
        "data/processed/person_reservation_fact_v3.csv",
        usecols=["date_gregorian", "Meal", "PersonId", "dont_receive", "is_main_meal"],
        parse_dates=["date_gregorian"], low_memory=False)
    fact = fact[fact["is_main_meal"]].drop(columns="is_main_meal")

    q29_taxonomy(day)
    q30_stranded_reservations(day, fact)
    q31_event_study(meal_day)
    cs, ret_tbl = q32_return_curve(cell, cell_shock)
    dose = q33_gap_dose_response(cs)
    q34_who_defects_on_return(fact, cs)
    make_figures(meal_day, ret_tbl, dose, day)


if __name__ == "__main__":
    main()
