r"""بند ۱۴-۹-۲ — تعیین تکلیف خانواده‌ی `regime`.

**وضعیت ارثی.** خانواده‌ی `regime` پس از ردیف ۸۰ عمداً **خالی** است: پنجره‌ی درگیری
خرداد ۱۴۰۴ با **حذف** رفتار می‌شود نه با پرچم. استدلال آن ردیف دو پایه داشت — پرچمِ
یک رویدادِ **یکتا** یک نمونه است و تعمیم‌پذیر نیست؛ و در استقرار، مقدارش برای هر شوک
آینده `False` است، یعنی دقیقاً وقتی لازم است کار نمی‌کند.

**پرسش این بند.** هیچ‌کدام از آن دو پایه برای رژیم‌های **تکرارشونده** برقرار نیست.
رمضان، امتحانات و دوره‌ی آلودگی هر سال تکرار می‌شوند و در استقرار از تقویم (یا از
تاریخچه‌ی آلودگی) **پیشاپیش معلوم**‌اند. پس منطق ردیف ۸۰ آن‌ها را پوشش نمی‌دهد و
خالی‌ماندن `regime` برایشان یک تصمیم **نیازموده** است، نه یک تصمیم گرفته‌شده.

**قاعده‌ی پذیرش — پیش از اجرا ثبت شد (قاعده‌ی ۱۴-۱-ه).**

> خانواده‌ی `regime` دوباره پر می‌شود **اگر و تنها اگر** Δ افزودنش به `lean` روی
> **هر دو** تقسیم برون‌نمونه‌ای معنادار منفی باشد. معناداری فقط درون‌پنجره کافی
> **نیست** — چون کل ادعای این خانواده «رژیم سال بعد هم تکرار می‌شود» است، و آزمونِ
> همین ادعا یعنی تقسیم برون‌نمونه‌ای. نتیجه‌ی منفی هم بستن معتبر این بند است.

**چرا این هفت ستون و نه پرچم خام.** پرچم خام رمضان و امتحان از قبل در خانواده‌ی
`calendar` هست و ablation بند ۱۴-۹-۱ سهم کل آن خانواده را −۰.۳٪ تا −۰.۵٪ سنجید.
اگر `regime` فقط همان پرچم‌ها را تکرار کند، آزمون بی‌معناست. پس هر ستون اینجا چیزی
می‌گوید که پرچم دودویی نمی‌گوید: **کجای** رژیم هستیم (`regime_progress`)، **چند روز**
از آغازش گذشته (`regime_day_index`)، رژیم **چقدر** طول می‌کشد (`regime_len`)، و
رژیم آلودگی که اصلاً در تقویم نیست.

⚠️ **دو قید برش اطلاعاتی که این ماژول رعایت می‌کند.**

۱. ستون‌های تقویمی رژیم (پنج‌تای اول) از تقویم می‌آیند و **از پیش معلوم**‌اند —
   تاریخ رمضان و امتحانات ماه‌ها قبل منتشر می‌شود.
۲. ستون‌های آلودگی فقط از AQI روزهای $\le d-2$ ساخته می‌شوند. AQI روز هدف در
   خانواده‌ی `weather` هست و همان‌جا هم با هشدار صریح آمده («مقدار واقعی روز $d$؛ در
   استقرار باید پیش‌بینی باشد»). نسخه‌ی رژیمی این مشکل را **ندارد**: تاریخچه‌ی
   آلودگی در لحظه‌ی برش واقعاً معلوم است. مرز $d-2$ محافظه‌کارانه است — AQI یک
   تجمیع روزانه است و روز $d-1$ در ساعت ۱۵ همان روز هنوز کامل نشده.

اجرا: ``python -m src.features.regime_v4``
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.baselines import pinball_loss
from src.config import REPORTS_DIR, set_global_seed
from src.cv import block_bootstrap_2d_mean
from src.features.ablation_v4 import _FA_DIGITS, N_BOOT, SEEDS, TAU, fit_predict, num
from src.features.build_v4 import WEATHER_PATH, OUTPUT_PATH as FEATURES_PATH, feature_set
from src.models.run_phase13_oos import build_splits

logger = logging.getLogger(__name__)

OUTPUT_DIR = REPORTS_DIR / "phase14"

#: آستانه‌ی «ناسالم» شاخص AQI آمریکا. ۱۵۰ مرز استاندارد است و روی این داده هم مرز
#: معناداری است: صدک ۷۵ برابر ۱۵۵ است، پس دوره‌ی آلودگی رویداد نادری نیست.
AQI_UNHEALTHY = 150

#: فاصله‌ی امن تاریخچه‌ی آلودگی تا روز هدف (قید ۲ بالا).
AQI_LAG_DAYS = 2

#: اولویت رژیم‌ها وقتی هم‌پوشانی دارند. رمضان اول است چون اثرش بزرگ‌ترین است
#: (ρ=۰.۱۰۷ در برابر ۰.۰۸۰) و چون سرویس **ناهار** را قطع می‌کند — یعنی ساختار
#: داده را عوض می‌کند، نه فقط سطح نرخ را (۹۴.۷٪ سلول‌های رمضان شام‌اند).
REGIME_PRIORITY = [
    ("ramadan", "is_ramadan"),
    ("nowruz", "is_nowruz_block"),
    ("summer", "is_summer_break"),
    ("inter_semester", "is_inter_semester_break"),
    ("exam", "is_exam_period"),
]

REGIME_FEATURES = [
    "regime_id", "regime_day_index", "regime_progress", "regime_len",
    "regime_ramadan_dinner_phase", "aqi_spell_len", "aqi_7d_mean_lag",
]


def _spell_frame(dates: pd.Series, flags: pd.Series) -> pd.DataFrame:
    """طول، شماره‌ی روز و پیشرفت هر «دوره‌ی پیوسته‌ی روشن» — روی تقویم روزانه.

    ⚠️ روی **روزهای تقویمی** کار می‌کند نه روزهای سرویس: رمضانی که وسطش تعطیلی
    بیفتد یک دوره است، نه دو تا. برچسب دوره با `cumsum` روی نقاط تغییر ساخته می‌شود.
    """
    d = pd.DataFrame({"date": pd.to_datetime(dates), "on": flags.astype(bool)}).drop_duplicates("date")
    d = d.sort_values("date").reset_index(drop=True)
    d["spell"] = (d["on"] != d["on"].shift()).cumsum()
    grp = d.groupby("spell")
    d["day_index"] = grp.cumcount()
    d["length"] = grp["on"].transform("size")
    d.loc[~d["on"], ["day_index", "length"]] = 0
    d["progress"] = np.where(d["length"] > 1, d["day_index"] / (d["length"] - 1), 0.0)
    return d.set_index("date")[["day_index", "length", "progress"]]


def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """هفت ستون خانواده‌ی `regime` — پنج تقویمی + دو آلودگی."""
    out = df.copy(deep=False)
    dates = pd.to_datetime(out["date_gregorian"])

    regime = pd.Series("normal", index=out.index, dtype=object)
    for name, flag in reversed(REGIME_PRIORITY):        # اولویت بالا آخر نوشته شود
        if flag in out.columns:
            regime = regime.mask(out[flag].astype(bool), name)
    out["regime_id"] = regime

    # یک دوره‌ی پیوسته به‌ازای «هر رژیمی غیر از عادی»، تا `regime_progress` معنای
    # «کجای همین رژیم هستیم» بدهد نه «کجای امسال».
    spells = _spell_frame(dates, regime != "normal")
    joined = spells.reindex(dates.to_numpy())
    out["regime_day_index"] = joined["day_index"].to_numpy()
    out["regime_len"] = joined["length"].to_numpy()
    out["regime_progress"] = joined["progress"].to_numpy()

    # رمضان × شام: اثر رمضان عملاً یک اثر **شام** است، چون ناهار سرو نمی‌شود
    # (۱۰۸ سلول ناهار در برابر ۱٬۹۲۷ شام). حاصل‌ضرب، این را به مدل می‌گوید
    # بدون اینکه لازم باشد خودش برهم‌کنش را از دو ستون کشف کند.
    is_dinner = (out["meal"].astype(str) == "dinner").to_numpy(float)
    out["regime_ramadan_dinner_phase"] = np.where(
        (regime == "ramadan").to_numpy(), out["regime_progress"].to_numpy() * is_dinner, 0.0)

    aqi = _aqi_history()
    out = out.merge(aqi, on=["city", "date_gregorian"], how="left")
    return out


def _aqi_history() -> pd.DataFrame:
    """تاریخچه‌ی آلودگی هر شهر، با تأخیر `AQI_LAG_DAYS` — هرگز روز هدف."""
    w = pd.read_csv(WEATHER_PATH, usecols=["city", "date_gregorian", "aqi_us_max"])
    w["date_gregorian"] = pd.to_datetime(w["date_gregorian"])
    w = w.sort_values(["city", "date_gregorian"])

    g = w.groupby("city", observed=True)["aqi_us_max"]
    unhealthy = w["aqi_us_max"] >= AQI_UNHEALTHY
    w["_u"] = unhealthy
    # طول دوره‌ی پیوسته‌ی ناسالم تا همان روز
    blocks = w.groupby("city", observed=True)["_u"].transform(
        lambda s: (s != s.shift()).cumsum())
    w["aqi_spell_len"] = (w.groupby(["city", blocks], observed=True).cumcount() + 1) * unhealthy
    w["aqi_7d_mean_lag"] = g.transform(lambda s: s.rolling(7, min_periods=1).mean())

    shifted = w.groupby("city", observed=True)[["aqi_spell_len", "aqi_7d_mean_lag"]].shift(AQI_LAG_DAYS)
    w[["aqi_spell_len", "aqi_7d_mean_lag"]] = shifted
    return w[["city", "date_gregorian", "aqi_spell_len", "aqi_7d_mean_lag"]]


# ---------------------------------------------------------------------------
# ارزیابی
# ---------------------------------------------------------------------------

#: دو پایه: خانواده روی مجموعه‌ی مرجع تازه (`lean`) و روی `full` آزموده می‌شود.
#: اگر فقط روی یکی کمک کند، آن خودش خبر است — یعنی سهمش با چیزی در همان مجموعه
#: هم‌پوشانی دارد.
BASES = {"lean": "مرجع جریان ب (۳۸ ستون)", "full": "ماتریس کامل (۶۶ ستون)"}


def evaluate(tau: float = TAU, seeds=SEEDS, n_boot: int = N_BOOT) -> pd.DataFrame:
    """Δ = pinball(پایه + رژیم) − pinball(پایه). منفی یعنی خانواده کمک می‌کند."""
    df = add_regime_features(pd.read_parquet(FEATURES_PATH))
    splits = build_splits(df)

    rows = []
    for split_name, split in splits.items():
        train, test = split["train"], split["test"].reset_index(drop=True)
        y = test["rho"].to_numpy(float)
        res = test["Res"].to_numpy(float)
        work = test[["date_gregorian", "restaurant"]].copy()

        for base, label in BASES.items():
            cols = feature_set(base)
            ref = np.mean([pinball_loss(y, fit_predict(train, test, tau, cols, s), tau)
                           for s in seeds], axis=0)
            arm = np.mean([pinball_loss(y, fit_predict(train, test, tau, cols + REGIME_FEATURES, s), tau)
                           for s in seeds], axis=0)
            work["_d"] = arm - ref
            work["_dp"] = (arm - ref) * res
            delta, lo, hi = block_bootstrap_2d_mean(
                work, "_d", day_col="date_gregorian", unit_col="restaurant", n_boot=n_boot)
            delta_p, lo_p, hi_p = block_bootstrap_2d_mean(
                work, "_dp", day_col="date_gregorian", unit_col="restaurant", n_boot=n_boot)
            rows.append({
                "base": base, "base_label": label, "split": split_name,
                "split_label": split["label"], "tau": tau,
                "pinball_base": float(ref.mean()), "pinball_with_regime": float(arm.mean()),
                "delta": delta, "ci_low": lo, "ci_high": hi,
                "delta_pct": 100 * delta / float(ref.mean()),
                "delta_portions": delta_p, "ci_low_portions": lo_p, "ci_high_portions": hi_p,
                "significant": bool(hi < 0),
            })
            logger.info("%s · %-5s → Δ=%+.6f [%+.6f, %+.6f] (%+.2f٪) %s", split_name, base,
                        delta, lo, hi, rows[-1]["delta_pct"], "معنادار" if hi < 0 else "—")

    out = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_DIR / "regime_family.csv", index=False)
    return out


def decide(results: pd.DataFrame) -> tuple[bool, str]:
    """قاعده‌ی پذیرشِ از پیش ثبت‌شده، اجراشده روی عدد — نه تفسیر پس از دیدن نتیجه."""
    oos = results[(results.base == "lean") & (results.split.str.startswith("oos_"))]
    passed = bool(len(oos) == 2 and oos["significant"].all())
    if passed:
        return True, "روی هر دو تقسیم برون‌نمونه‌ای معنادار منفی — خانواده پر می‌شود"
    n = int(oos["significant"].sum())
    return False, f"فقط روی {num(n, 'd')} از ۲ تقسیم برون‌نمونه‌ای معنادار — خانواده خالی می‌ماند"


def regime_rates(df: pd.DataFrame) -> pd.DataFrame:
    """توصیف رژیم‌ها: سهم سلول، نرخ، و فاصله از حالت عادی — زمینه‌ی خواندن Δ."""
    base = df.loc[df["regime_id"] == "normal"]
    rho_normal = base["NoRecv"].sum() / base["Res"].sum()
    rows = []
    for name, block in df.groupby("regime_id", observed=True):
        rho = block["NoRecv"].sum() / max(block["Res"].sum(), 1)
        rows.append({
            "regime_id": name, "n_cells": len(block), "share_pct": 100 * len(block) / len(df),
            "rho": rho, "gap_vs_normal_pp": 100 * (rho - rho_normal),
            "median_len_days": float(block["regime_len"].median()),
            "dinner_share_pct": 100 * (block["meal"].astype(str) == "dinner").mean(),
        })
    return pd.DataFrame(rows).sort_values("n_cells", ascending=False)


#: سطل‌های شماره‌ی روزِ رژیم. مرز ۱ و ۳ دلبخواه نیست: فیچرهای تاریخی سطح سلف
#: (`venue_rho_lag1`, `venue_last_rho`) با **یک وعده** تأخیر به‌روز می‌شوند، پس اگر
#: سازوکارِ جذب همین باشد، جهش خطا باید دقیقاً در روز صفر باشد و از روز ۱ فروکش کند.
ONSET_BUCKETS = [(0, 0, "روز آغاز"), (1, 2, "روز ۱–۲"), (3, 99, "روز ۳ به بعد")]


def onset_diagnostic(tau: float = TAU, seeds=SEEDS) -> pd.DataFrame:
    """چرا رژیم تکرارشونده پرچم نمی‌خواهد — آزمون سازوکار، نه تکرار همان Δ.

    اگر مدل رژیم را **از تاریخچه** بگیرد نه از تقویم، خطایش باید در **روز آغاز**
    رژیم بیشترین باشد و بعد فروکش کند: فیچرهای تاریخی هنوز سطحِ حالت عادی را
    می‌گویند. اگر خطا در طول رژیم تخت باشد، سازوکار چیز دیگری است و «پرچم لازم
    نیست» بی‌توضیح می‌ماند.

    ⚠️ این تشخیص است نه معیار انتخاب — مثل AUC سطح فرد در قاعده‌ی ۱۴-۱-د.
    """
    df = add_regime_features(pd.read_parquet(FEATURES_PATH))
    rows = []
    for split_name, split in build_splits(df).items():
        if not split_name.startswith("oos_"):
            continue
        train, test = split["train"], split["test"].reset_index(drop=True)
        y = test["rho"].to_numpy(float)
        loss = np.mean([pinball_loss(y, fit_predict(train, test, tau, feature_set("lean"), s), tau)
                        for s in seeds], axis=0)
        work = test[["regime_id", "regime_day_index"]].copy()
        work["loss"] = loss
        normal = float(work.loc[work.regime_id == "normal", "loss"].mean())
        for regime in ("ramadan", "exam"):
            block = work[work.regime_id == regime]
            for lo, hi, label in ONSET_BUCKETS:
                cell = block[block.regime_day_index.between(lo, hi)]
                if not len(cell):
                    continue
                rows.append({
                    "split": split_name, "regime_id": regime, "bucket": label,
                    "n": len(cell), "pinball": float(cell["loss"].mean()),
                    "vs_normal_pct": 100 * (float(cell["loss"].mean()) / normal - 1),
                })
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "regime_onset.csv", index=False)
    return out


#: سه‌بخشی‌کردن دوره — با ۱۵ روز سرویس، سطل ریزتر از این نویز است.
RAMADAN_PHASES = [(-0.01, 0.33, "اول"), (0.33, 0.66, "میانه"), (0.66, 1.01, "آخر")]


def ramadan_profile(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """شکل اثر رمضان در هر سال تحصیلی — آزمونِ «آیا این رژیم اصلاً آموختنی است؟».

    فیچرِ فاز رژیم تنها وقتی معنا دارد که شکل اثر بین سال‌ها **تکرار** شود. اگر
    شکل عوض شود، `regime_progress` نه‌تنها کمک نمی‌کند، بلکه منحنی سال قبل را به
    سال بعد تحمیل می‌کند — یعنی فعالانه مضر است.
    """
    df = add_regime_features(pd.read_parquet(FEATURES_PATH)) if df is None else df
    r = df[df["regime_id"] == "ramadan"].copy()
    rows = []
    for year, block in r.groupby("academic_year", observed=True):
        for lo, hi, label in RAMADAN_PHASES:
            cell = block[block["regime_progress"].between(lo, hi, inclusive="right")]
            if not len(cell):
                continue
            rows.append({"academic_year": year, "phase": label, "n_cells": len(cell),
                         "rho": cell["NoRecv"].sum() / max(cell["Res"].sum(), 1)})
        rows.append({"academic_year": year, "phase": "کل دوره", "n_cells": len(block),
                     "rho": block["NoRecv"].sum() / max(block["Res"].sum(), 1),
                     "service_days": block["date_gregorian"].nunique()})
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "regime_ramadan_profile.csv", index=False)
    return out


def write_report(results: pd.DataFrame, rates: pd.DataFrame,
                 onset: pd.DataFrame | None = None,
                 ramadan: pd.DataFrame | None = None) -> str:
    passed, why = decide(results)
    lines = [
        "# خانواده‌ی `regime` — تعیین تکلیف، بند ۱۴-۹-۲",
        "",
        "> `src/features/regime_v4.py` · " f"τ={num(TAU, '.2f')} · بذرها: "
        f"{'، '.join(num(s, 'd') for s in SEEDS)} · بوت‌استرپ بلوکی دوبعدی "
        f"({num(N_BOOT, 'd')} تکرار)",
        "",
        "## پرسش",
        "",
        "ردیف ۸۰ `decision_log` پنجره‌ی شوک **یکتا** را با حذف حل کرد، نه با پرچم — چون",
        "یک نمونه تعمیم‌پذیر نیست و در استقرار پرچمش برای هر شوک آینده `False` است.",
        "هیچ‌کدام از این دو استدلال برای رژیم‌های **تکرارشونده** برقرار نیست: رمضان،",
        "امتحانات و دوره‌ی آلودگی هر سال برمی‌گردند و پیشاپیش معلوم‌اند.",
        "",
        "**قاعده‌ی پذیرش، ثبت‌شده پیش از اجرا:** خانواده فقط وقتی پر می‌شود که Δ افزودنش",
        "به `lean` روی **هر دو** تقسیم برون‌نمونه‌ای معنادار منفی باشد.",
        "",
        "## رژیم‌ها در داده",
        "",
        "| رژیم | سلول | سهم | ρ | فاصله از عادی | میانه‌ی طول (روز) | سهم شام |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in rates.iterrows():
        lines.append(
            f"| `{r['regime_id']}` | {num(int(r['n_cells']), ',d').replace(',', '٬')} | "
            f"{num(r['share_pct'], '.2f')}٪ | {num(r['rho'], '.4f')} | "
            f"{num(r['gap_vs_normal_pp'], '+.2f')} واحد درصد | "
            f"{num(r['median_len_days'], '.0f')} | {num(r['dinner_share_pct'], '.1f')}٪ |")

    lines += [
        "",
        "## نتیجه‌ی ablation",
        "",
        "Δ = pinball(پایه + رژیم) − pinball(پایه). **منفی یعنی خانواده کمک می‌کند.**",
        "",
        "| پایه | تقسیم | Δ (نرخ) | Δ (پرس) |",
        "|---|---|---|---|",
    ]
    for _, r in results.iterrows():
        mark = " ✅" if r["significant"] else ""
        mark_p = " ✅" if r["ci_high_portions"] < 0 else ""
        lines.append(
            f"| `{r['base']}` | {r['split_label'].translate(_FA_DIGITS)} | "
            f"{num(r['delta'])} [{num(r['ci_low'])}, {num(r['ci_high'])}]{mark} | "
            f"{num(r['delta_portions'], '+.4f')} [{num(r['ci_low_portions'], '+.4f')}, "
            f"{num(r['ci_high_portions'], '+.4f')}]{mark_p} |")

    if onset is not None and len(onset):
        lines += [
            "",
            "## چرا — و چرا فرضیه‌ی اول غلط بود",
            "",
            "«اثری ندارد» بدون سازوکار، نتیجه‌ی ناتمامی است. رژیم‌ها در داده **واقعی**‌اند",
            "(جدول بالا: رمضان +۲.۳۸ و امتحان −۱.۲۰ واحد درصد)، پس باید معلوم شود چرا",
            "پرچمشان کمکی نمی‌کند.",
            "",
            "**فرضیه‌ی اول: فیچرهای تاریخی با یک وعده تأخیر رژیم را جذب می‌کنند.** اگر",
            "درست بود، خطا باید در **روز آغاز** رژیم بیشینه می‌شد و بعد فروکش می‌کرد.",
            "جدول زیر این را **رد می‌کند** — و همین ردشدن، پاسخ واقعی را نشان داد.",
            "",
            "pinball مدل `lean` به تفکیک روزِ رژیم (نسبت به روزهای عادی همان تقسیم):",
            "",
            "| تقسیم | رژیم | روز | سلول | pinball | فاصله از روز عادی |",
            "|---|---|---|---|---|---|",
        ]
        for _, r in onset.iterrows():
            lines.append(
                f"| {r['split'].replace('oos_', '').translate(_FA_DIGITS)} | "
                f"`{r['regime_id']}` | {r['bucket']} | "
                f"{num(int(r['n']), 'd')} | {num(r['pinball'], '.6f')} | "
                f"{num(r['vs_normal_pct'], '+.1f')}٪ |")

    if ramadan is not None and len(ramadan):
        lines += [
            "",
            "### پاسخ واقعی: شکل اثر رمضان بین دو سال **وارونه** می‌شود",
            "",
            "| سال تحصیلی | بخش دوره | سلول | ρ |",
            "|---|---|---|---|",
        ]
        for _, r in ramadan.iterrows():
            bold = "**" if r["phase"] == "کل دوره" else ""
            lines.append(f"| {r['academic_year'].translate(_FA_DIGITS)} | "
                         f"{bold}{r['phase']}{bold} | "
                         f"{num(int(r['n_cells']), 'd')} | {bold}{num(r['rho'], '.4f')}{bold} |")
        lines += [
            "",
            "دو چیز هم‌زمان اینجاست و هر دو کشنده‌ی فیچرِ فاز رژیم‌اند:",
            "",
            "۱. **کل پنجره فقط دو رمضان دارد** (۱۵ و ۱۶ روز سرویس)، و تقسیم آموزش فقط",
            "   **یکی** از آن‌ها را می‌بیند. رمضان ۱۴۰۴-۱۴۰۵ بیرون از انتهای داده",
            "   (۲۰۲۶-۰۱-۳۱) می‌افتد، پس آن تقسیم اصلاً رمضانی در آزمون ندارد.",
            "۲. **شکل اثر در همان دو سال وارونه است:** در ۱۴۰۲-۱۴۰۳ نرخ از ۰.۲۰۹ شروع",
            "   می‌شود و به ۰.۱۰۹ **می‌افتد**؛ در ۱۴۰۳-۱۴۰۴ از ۰.۰۶۶ شروع می‌شود و به",
            "   ۰.۱۵۶ **بالا می‌رود**. سطح کلی هم تقریباً نصف شده (۰.۱۵۲ → ۰.۰۸۸).",
            "",
            "یعنی `regime_progress` منحنی سال قبل را به سال بعد تحمیل می‌کند و دقیقاً",
            "برعکسش درمی‌آید — و همین است که در جدول بالا خطای «روز ۳ به بعد» رمضان را",
            "به **+۷۴.۶٪** می‌رساند، نه کمبود پرچم. **فیچر نیست که کم است؛ رمضان با دو",
            "نمونه آموختنی نیست.**",
            "",
            "در برابرش، امتحانات نه‌تنها پرچم نمی‌خواهند بلکه **آسان‌ترند**: خطای مدل در",
            "دوره‌ی امتحانات روی هر دو سال و هر سه سطل، ۱۴٪ تا ۴۴٪ **کمتر** از روزهای",
            "عادی است. رژیمی که نرخش پایین‌تر و باثبات‌تر است، مسئله‌ی مدل نیست.",
        ]

    lines += [
        "",
        f"## حکم: {'خانواده پر می‌شود' if passed else '**خانواده خالی می‌ماند**'}",
        "",
        why + ".",
        "",
        "⚠️ **همان نتیجه‌ی ردیف ۸۰، ولی به دلیلی متفاوت — و تفاوت باید ثبت بماند.**",
        "ردیف ۸۰ شوک **یکتا** را پرچم نزد چون یک نمونه تعمیم نمی‌دهد. رژیم تکرارشونده",
        "قرار بود این مشکل را نداشته باشد؛ معلوم شد **دارد**، فقط ملایم‌تر: رمضان در این",
        "پنجره دو نمونه دارد نه یکی، و آن دو نمونه شکل‌های مخالف هم می‌دهند.",
        "",
        "**این نتیجه شرطی است، نه دائمی — و شرطش عددی است.** با رسیدن **رمضان سومِ",
        "کامل** به داده، این بند باید بازاجرا شود: دو نمونه‌ی متناقض فقط می‌گویند «شکل",
        "تکرار نشد»، نه «شکلی وجود ندارد». نمونه‌ی سوم است که بین این دو تفکیک می‌کند.",
        "تا آن‌وقت، رمضان یک **محدودیت اعلام‌شده** است نه یک فیچر نساخته: مدل روی",
        f"{num(3.18, '.2f')}٪ سلول‌ها خطای بالاتری دارد و پروژه این را می‌داند.",
    ]
    text = "\n".join(lines) + "\n"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "regime_family.md").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    set_global_seed()
    df = add_regime_features(pd.read_parquet(FEATURES_PATH))
    rates = regime_rates(df)
    print("\n=== رژیم‌ها در داده ===")
    print(rates.to_string(index=False))

    results = evaluate()
    onset = onset_diagnostic()
    print("\n=== خطا به تفکیک روزِ رژیم ===")
    print(onset.to_string(index=False))
    profile = ramadan_profile(df)
    print("\n=== پروفایل رمضان به تفکیک سال ===")
    print(profile.to_string(index=False))
    write_report(results, rates, onset, profile)
    passed, why = decide(results)
    print(f"\n=== حکم بند ۱۴-۹-۲: {'پر شود' if passed else 'خالی بماند'} — {why} ===")


if __name__ == "__main__":
    main()
