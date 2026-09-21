r"""بند ۱۴-۹-۳ — بازطراحی فیچرهای تاریخی با ۶۰۰ روز.

خانواده‌ی `history` هجده ستون دارد و **تنها خانواده‌ای است که در هر دو فضای نرخ و
پرس سهم اثبات‌شده دارد** (F276، F278). ولی هر سه انتخاب طراحی‌اش با افق ۱۴۲ روزه
بسته شده‌اند، و هر سه با ۶۰۰ روز دوباره قابل پرسیدن‌اند:

| انتخاب | چرا آن‌موقع بدیهی بود | چرا حالا نیست |
|---|---|---|
| پنجره‌ی **انبساطی** | با ۱۴۲ روز، پنجره‌ی متحرک ۱۸۰ روزه بی‌معناست | ۶۰۰ روز یعنی «تاریخچه‌ی تازه» و «تاریخچه‌ی کامل» دو چیز متفاوتند |
| آستانه‌ی `same_meal_lag` | فیلتر فاصله، `cell_rho_lag1` را به پوشش ۱۴.۷٪ می‌رساند (F120، F220) | با داده‌ی بیشتر شاید آستانه‌ی دیگری بهتر باشد — یا اصلاً این ستون‌ها زائد باشند |
| پیشین Beta ‏(۰.۹۷۵۸, ۱۱.۱۵۰۶) | روی پنجره‌ی قفل‌شده برازش شده بود | باید روی ۶۰۰ روز بازبرازش شود و تفاوتش گزارش |

**یک چیزِ چهارم که ضمن همین بند آشکار شد.** خانواده‌ی تاریخی v4 اصلاً **کوچک‌سازی
بیزی ندارد** — نرخ‌هایش خام‌اند (`nr_sum / res_sum`) و فقط `hist_n` کنارشان است.
طراحی فاز ۵ نسخه‌ی کوچک‌شده داشت (`cell_shrunk_rate`). با ۳۶.۹٪ سلول‌های L1 که کمتر
از ۱۰ رزرو دارند (F104)، این حذف بی‌هزینه نیست و باید آزموده شود، نه فرض.

**قاعده‌ی پذیرش — پیش از اجرا (قاعده‌ی ۱۴-۱-ه).** هر تغییر طراحی فقط وقتی جایگزین
نسخه‌ی فعلی می‌شود که Δ‌اش روی **هر دو** تقسیم برون‌نمونه‌ای معنادار منفی باشد.
«بهتر شدن درون‌پنجره» کافی نیست — همان اشتباهی که F123 هزینه‌اش را نشان داد.

اجرا: ``python -m src.features.history_v4``
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.baselines import BETA_ALPHA, BETA_BETA, pinball_loss
from src.config import REPORTS_DIR, set_global_seed
from src.cv import block_bootstrap_2d_mean
from src.features.ablation_v4 import _FA_DIGITS, N_BOOT, SEEDS, TAU, fit_predict, num
from src.features.build_v4 import OUTPUT_PATH as FEATURES_PATH, feature_set
from src.features.cutoff import meal_seq, rolling_stat_at_cutoff, same_meal_lag
from src.models.run_phase13_oos import TRAIN_YEAR, build_splits

logger = logging.getLogger(__name__)

OUTPUT_DIR = REPORTS_DIR / "phase14"

#: پنجره‌های متحرک نامزد. ۳۰ = «ماه گذشته» · ۹۰ = «نیمسال جاری» · ۱۸۰ = «یک ترم کامل».
ROLL_WINDOWS = (30, 90, 180)

#: پیشین سطح **فرد** که دور ۹ روی v4 برازش کرد (F259) — برای مقایسه، نه استفاده اینجا.
PERSON_PRIOR_V4 = (1.2717, 10.1638)


# ---------------------------------------------------------------------------
# ۱) پیشین Beta
# ---------------------------------------------------------------------------

def fit_beta_binomial(k: np.ndarray, n: np.ndarray) -> tuple[float, float]:
    r"""برازش بیشینه‌درست‌نمایی Beta-Binomial روی جفت‌های (موفقیت، کل).

    ⚠️ **چرا نه میانگین‌گیری ساده‌ی نرخ‌ها.** نرخ سلولی با ۳ رزرو و سلولی با ۳۰۰ رزرو
    دو برآورد با دقت کاملاً متفاوت‌اند؛ برازش روی نرخ‌ها هر دو را هم‌وزن می‌کند و
    پیشین را **پهن‌تر از واقع** می‌سازد. درست‌نمایی Beta-Binomial خودش $n_i$ را وزن
    می‌دهد.
    """
    from scipy.optimize import minimize
    from scipy.special import betaln

    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    ok = np.isfinite(k) & np.isfinite(n) & (n > 0)
    k, n = k[ok], n[ok]

    def nll(theta: np.ndarray) -> float:
        a, b = np.exp(theta)
        return -float(np.sum(betaln(k + a, n - k + b) - betaln(a, b)))

    start = np.log([max(BETA_ALPHA, 1e-3), max(BETA_BETA, 1e-3)])
    res = minimize(nll, start, method="Nelder-Mead",
                   options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 4000})
    a, b = np.exp(res.x)
    return float(a), float(b)


def refit_prior(df: pd.DataFrame | None = None) -> dict:
    """پیشین سطح **سلول**، بازبرازش روی سال آموزش پنجره‌ی v4.

    ⚠️ فقط روی سال آموزش، نه کل پنجره. پیشین یک **پارامتر ساخت فیچر** است؛ برازشش
    روی داده‌ای که بعداً آزمون می‌شود، همان نشتی ظریفی است که ممیزی بند ۱۴-۱۲-۳
    دنبالش می‌گردد — حتی اگر فقط دو عدد باشد.
    """
    df = pd.read_parquet(FEATURES_PATH) if df is None else df
    train = df[df["academic_year"] == TRAIN_YEAR]
    a, b = fit_beta_binomial(train["NoRecv"].to_numpy(), train["Res"].to_numpy())
    a_all, b_all = fit_beta_binomial(df["NoRecv"].to_numpy(), df["Res"].to_numpy())
    return {
        "v2_alpha": BETA_ALPHA, "v2_beta": BETA_BETA,
        "v2_mean": BETA_ALPHA / (BETA_ALPHA + BETA_BETA),
        "v2_strength": BETA_ALPHA + BETA_BETA,
        "v4_train_alpha": a, "v4_train_beta": b, "v4_train_mean": a / (a + b),
        "v4_train_strength": a + b, "n_train_cells": int(len(train)),
        "v4_all_alpha": a_all, "v4_all_beta": b_all, "v4_all_mean": a_all / (a_all + b_all),
        "v4_all_strength": a_all + b_all, "n_all_cells": int(len(df)),
        "person_v4_alpha": PERSON_PRIOR_V4[0], "person_v4_beta": PERSON_PRIOR_V4[1],
        "person_v4_mean": PERSON_PRIOR_V4[0] / sum(PERSON_PRIOR_V4),
        "person_v4_strength": sum(PERSON_PRIOR_V4),
    }


# ---------------------------------------------------------------------------
# ۲) ساخت نسخه‌های جایگزین
# ---------------------------------------------------------------------------

#: هر نسخه: ستون‌هایی که به `lean` **اضافه** می‌شوند و ستون‌هایی که **برداشته** می‌شوند.
#: نسخه‌ی مرجع (`lean` دست‌نخورده) در جدول Δ=۰ است و ردیف ندارد.
VARIANTS: dict[str, dict] = {
    "roll30": {"add": ["cell_roll30_rho", "venue_roll30_rho"], "drop": [],
               "question": "تاریخچه‌ی یک‌ماهه کنار انبساطی"},
    "roll90": {"add": ["cell_roll90_rho", "venue_roll90_rho"], "drop": [],
               "question": "تاریخچه‌ی سه‌ماهه کنار انبساطی"},
    "roll180": {"add": ["cell_roll180_rho", "venue_roll180_rho"], "drop": [],
                "question": "تاریخچه‌ی شش‌ماهه کنار انبساطی"},
    "roll_all": {"add": [f"{lvl}_roll{w}_rho" for w in ROLL_WINDOWS for lvl in ("cell", "venue")],
                 "drop": [], "question": "هر سه پنجره با هم"},
    "roll90_replaces_expanding": {
        "add": ["cell_roll90_rho", "venue_roll90_rho"],
        "drop": ["cell_hist_rho", "venue_hist_rho"],
        "question": "⭐ متحرک **به‌جای** انبساطی، نه کنارش"},
    "shrunk": {"add": ["cell_shrunk_rho", "venue_shrunk_rho"], "drop": [],
               "question": "کوچک‌سازی بیزی با پیشین بازبرازش‌شده"},
    "shrunk_replaces_raw": {
        "add": ["cell_shrunk_rho", "venue_shrunk_rho"],
        "drop": ["cell_hist_rho", "venue_hist_rho"],
        "question": "کوچک‌شده **به‌جای** خام"},
    "no_lag_cols": {"add": [], "drop": ["cell_rho_lag1", "cell_rho_lag7"],
                    "question": "⭐ آیا lagهای کم‌پوشش زائدند؟"},
    "lag_buffer27": {"add": ["cell_rho_lag1_b27", "cell_rho_lag7_b27"],
                     "drop": ["cell_rho_lag1", "cell_rho_lag7"],
                     "question": "آستانه‌ی فاصله‌ی ۴ هفته به‌جای ۱ هفته"},
    # ⚠️ پس‌رویدادی (post-hoc) — **پس از** دیدن نتایج بالا ساخته شد، نه پیش از آن.
    # سه تغییری که هرکدام جداگانه روی هر سه تقسیم علامت منفی گرفتند ولی از سد قاعده‌ی
    # پذیرش رد نشدند، اینجا با هم آزموده می‌شوند. **قاعده‌ی پذیرش شاملش نمی‌شود**:
    # ترکیبی که پس از دیدن نتیجه ساخته شود، هر چقدر هم خوب، شاهدِ هم‌سطحِ آزمون
    # از‌پیش‌ثبت‌شده نیست. نتیجه‌اش فقط **نامزدِ جریان پ** است.
    "combined_posthoc": {
        "add": ["cell_roll90_rho", "venue_roll90_rho", "cell_shrunk_rho", "venue_shrunk_rho"],
        "drop": ["cell_hist_rho", "venue_hist_rho", "cell_rho_lag1", "cell_rho_lag7"],
        "question": "⚠️ پس‌رویدادی: هر سه تغییر هم‌جهت با هم"},
}


def build_variant_columns(df: pd.DataFrame, prior: dict | None = None) -> pd.DataFrame:
    """همه‌ی ستون‌های نامزد را یک‌بار می‌سازد — هر نسخه زیرمجموعه‌ای از آن‌هاست."""
    prior = prior or refit_prior(df)
    out = df.sort_values(["date_gregorian", "meal"], kind="stable").reset_index(drop=True)
    out["meal_seq"] = meal_seq(out["date_gregorian"], out["meal"])

    levels = {"cell": ["restaurant", "dish", "meal"], "venue": ["restaurant", "meal"]}
    for name, keys in levels.items():
        for w in ROLL_WINDOWS:
            nr = rolling_stat_at_cutoff(out, keys, "NoRecv", w, out_prefix="_nr")
            res = rolling_stat_at_cutoff(out, keys, "Res", w, out_prefix="_res")
            with np.errstate(invalid="ignore", divide="ignore"):
                out[f"{name}_roll{w}_rho"] = np.where(
                    res["_res_sum"] > 0, nr["_nr_sum"] / res["_res_sum"], np.nan)

        # کوچک‌سازی بیزی روی همان جمع‌های انبساطی که ماتریس از قبل دارد:
        # res_sum = hist_n × hist_res_mean  و  nr_sum = hist_rho × res_sum.
        res_sum = out[f"{name}_hist_n"] * out[f"{name}_hist_res_mean"]
        nr_sum = out[f"{name}_hist_rho"].fillna(0) * res_sum
        a, b = prior["v4_train_alpha"], prior["v4_train_beta"]
        out[f"{name}_shrunk_rho"] = (nr_sum.fillna(0) + a) / (res_sum.fillna(0) + a + b)

    wide = same_meal_lag(out, ["restaurant", "dish", "meal"], "rho", [1, 7], gap_buffer_days=27)
    out["cell_rho_lag1_b27"] = wide["rho_lag1"].to_numpy()
    out["cell_rho_lag7_b27"] = wide["rho_lag7"].to_numpy()
    return out


def coverage_table(df: pd.DataFrame) -> pd.DataFrame:
    """پوشش و همبستگی هر ستون نامزد — زمینه‌ی خواندن Δ، نه جایگزینش."""
    cols = sorted({c for v in VARIANTS.values() for c in v["add"]}
                  | {"cell_hist_rho", "venue_hist_rho", "cell_rho_lag1", "cell_rho_lag7"})
    rows = []
    for c in cols:
        if c not in df.columns:
            continue
        rows.append({"column": c, "coverage_pct": 100 * df[c].notna().mean(),
                     "corr_with_rho": float(df[c].corr(df["rho"]))})
    out = pd.DataFrame(rows).sort_values("coverage_pct")
    out.to_csv(OUTPUT_DIR / "history_coverage.csv", index=False)
    return out


# ---------------------------------------------------------------------------
# ۳) ارزیابی
# ---------------------------------------------------------------------------

def evaluate(tau: float = TAU, seeds=SEEDS, n_boot: int = N_BOOT) -> pd.DataFrame:
    """Δ هر نسخه نسبت به `lean` دست‌نخورده، روی هر سه تقسیم."""
    prior = refit_prior()
    df = build_variant_columns(pd.read_parquet(FEATURES_PATH), prior)
    splits = build_splits(df)
    base_cols = feature_set("lean")

    rows = []
    for split_name, split in splits.items():
        train, test = split["train"], split["test"].reset_index(drop=True)
        y = test["rho"].to_numpy(float)
        res = test["Res"].to_numpy(float)
        work = test[["date_gregorian", "restaurant"]].copy()
        ref = np.mean([pinball_loss(y, fit_predict(train, test, tau, base_cols, s), tau)
                       for s in seeds], axis=0)

        for name, spec in VARIANTS.items():
            cols = [c for c in base_cols if c not in spec["drop"]] + spec["add"]
            arm = np.mean([pinball_loss(y, fit_predict(train, test, tau, cols, s), tau)
                           for s in seeds], axis=0)
            work["_d"] = arm - ref
            work["_dp"] = (arm - ref) * res
            delta, lo, hi = block_bootstrap_2d_mean(
                work, "_d", day_col="date_gregorian", unit_col="restaurant", n_boot=n_boot)
            delta_p, lo_p, hi_p = block_bootstrap_2d_mean(
                work, "_dp", day_col="date_gregorian", unit_col="restaurant", n_boot=n_boot)
            rows.append({
                "variant": name, "question": spec["question"], "split": split_name,
                "split_label": split["label"], "n_cols": len(cols), "tau": tau,
                "pinball_ref": float(ref.mean()), "pinball_variant": float(arm.mean()),
                "delta": delta, "ci_low": lo, "ci_high": hi,
                "delta_pct": 100 * delta / float(ref.mean()),
                "delta_portions": delta_p, "ci_low_portions": lo_p, "ci_high_portions": hi_p,
                "better": bool(hi < 0), "worse": bool(lo > 0),
            })
            logger.info("%s · %-26s → Δ=%+.6f [%+.6f, %+.6f] (%+.2f٪) %s", split_name, name,
                        delta, lo, hi, rows[-1]["delta_pct"],
                        "بهتر" if hi < 0 else ("بدتر" if lo > 0 else "—"))

    out = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_DIR / "history_variants.csv", index=False)
    return out


def decide(results: pd.DataFrame) -> pd.DataFrame:
    """قاعده‌ی پذیرشِ از پیش ثبت‌شده: معناداری روی **هر دو** تقسیم برون‌نمونه‌ای."""
    rows = []
    for name, block in results.groupby("variant", sort=False):
        oos = block[block.split.str.startswith("oos_")]
        adopt = bool(len(oos) == 2 and oos["better"].all())
        harmful = bool(oos["worse"].any())
        rows.append({
            "variant": name, "n_better_oos": int(oos["better"].sum()),
            "n_worse_oos": int(oos["worse"].sum()),
            "verdict": "پذیرفته" if adopt else ("⚠️ مضر" if harmful else "رد"),
        })
    return pd.DataFrame(rows)


SPLIT_ORDER = ["in_window", "oos_1403-1404", "oos_1404-1405"]
SPLIT_FA = {"in_window": "درون‌پنجره", "oos_1403-1404": "۱۴۰۳-۱۴۰۴", "oos_1404-1405": "۱۴۰۴-۱۴۰۵"}


def write_report(results: pd.DataFrame, prior: dict, coverage: pd.DataFrame) -> str:
    verdicts = decide(results).set_index("variant")
    lines = [
        "# فیچرهای تاریخی با ۶۰۰ روز — بند ۱۴-۹-۳",
        "",
        f"> `src/features/history_v4.py` · τ={num(TAU, '.2f')} · بذرها: "
        f"{'، '.join(num(s, 'd') for s in SEEDS)} · بوت‌استرپ بلوکی دوبعدی "
        f"({num(N_BOOT, 'd')} تکرار) · پایه: `lean` ({num(len(feature_set('lean')), 'd')} ستون)",
        "",
        "**قاعده‌ی پذیرش، ثبت‌شده پیش از اجرا:** هر تغییر طراحی فقط وقتی جایگزین نسخه‌ی",
        "فعلی می‌شود که Δ‌اش روی **هر دو** تقسیم برون‌نمونه‌ای معنادار منفی باشد.",
        "",
        "## ۱. پیشین Beta — بازبرازش روی ۶۰۰ روز",
        "",
        "| پیشین | α | β | میانگین | قدرت (رزرو) | برازش روی |",
        "|---|---|---|---|---|---|",
        f"| v2 (در کد فعلی) | {num(prior['v2_alpha'], '.4f')} | {num(prior['v2_beta'], '.4f')} | "
        f"{num(prior['v2_mean'], '.4f')} | {num(prior['v2_strength'], '.2f')} | پنجره‌ی قفل‌شده |",
        f"| **v4، سال آموزش** | {num(prior['v4_train_alpha'], '.4f')} | "
        f"{num(prior['v4_train_beta'], '.4f')} | **{num(prior['v4_train_mean'], '.4f')}** | "
        f"{num(prior['v4_train_strength'], '.2f')} | "
        f"{num(prior['n_train_cells'], ',d').replace(',', '٬')} سلول |",
        f"| v4، کل پنجره (مرجع) | {num(prior['v4_all_alpha'], '.4f')} | "
        f"{num(prior['v4_all_beta'], '.4f')} | {num(prior['v4_all_mean'], '.4f')} | "
        f"{num(prior['v4_all_strength'], '.2f')} | "
        f"{num(prior['n_all_cells'], ',d').replace(',', '٬')} سلول |",
        f"| سطح **فرد** v4 (F259) | {num(prior['person_v4_alpha'], '.4f')} | "
        f"{num(prior['person_v4_beta'], '.4f')} | {num(prior['person_v4_mean'], '.4f')} | "
        f"{num(prior['person_v4_strength'], '.2f')} | دور ۹ |",
        "",
        "## ۲. پوشش ستون‌های نامزد",
        "",
        "| ستون | پوشش | r با هدف |",
        "|---|---|---|",
    ]
    for _, r in coverage.iterrows():
        lines.append(f"| `{r['column']}` | {num(r['coverage_pct'], '.1f')}٪ | "
                     f"{num(r['corr_with_rho'], '+.3f')} |")

    lines += [
        "",
        "## ۳. Δ هر نسخه نسبت به `lean`",
        "",
        "Δ = pinball(نسخه) − pinball(`lean`). **منفی یعنی نسخه بهتر است.**",
        "",
        "| نسخه | پرسش | " + " | ".join(SPLIT_FA[s] for s in SPLIT_ORDER) + " | حکم |",
        "|---|---|" + "---|" * (len(SPLIT_ORDER) + 1),
    ]
    for name, spec in VARIANTS.items():
        block = results[results.variant == name]
        cells = []
        for split in SPLIT_ORDER:
            row = block[block.split == split]
            if not len(row):
                cells.append("—")
                continue
            r = row.iloc[0]
            mark = " ✅" if r["better"] else (" ❌" if r["worse"] else "")
            cells.append(f"{num(r['delta'])} [{num(r['ci_low'])}, {num(r['ci_high'])}]{mark}")
        lines.append(f"| `{name}` | {spec['question']} | " + " | ".join(cells)
                     + f" | **{verdicts.loc[name, 'verdict']}** |")

    lines += ["", "✅ = معنادار بهتر · ❌ = معنادار **بدتر**.", ""]

    oos = results[results.split.str.startswith("oos_")]
    best = oos.groupby("variant")["delta_pct"].mean().idxmin()
    best_pct = oos[oos.variant == best]["delta_pct"].mean()
    comb = oos[oos.variant == "combined_posthoc"]["delta_pct"].mean()
    lag7 = coverage[coverage.column == "cell_rho_lag7"]["coverage_pct"].iloc[0]
    lines += [
        "## این جدول چه می‌گوید",
        "",
        "**۱. هر سه انتخاب طراحی، تأیید می‌شوند — نه چون بهترین‌اند، چون جایگزینی",
        "برتری‌اش اثبات نشد.** هیچ‌کدام از نُه نسخه از سد قاعده‌ی پذیرش رد نشد. این",
        "«همه‌چیز خوب است» نیست؛ یعنی با این پنجره و این معیار، تفاوت‌ها از نویز جدا",
        "نمی‌شوند.",
        "",
        f"**۲. قوی‌ترین نامزد `{best}` است** (میانگین برون‌نمونه "
        f"{num(best_pct, '+.2f')}٪): روی **هر سه** تقسیم علامت منفی دارد و روی یکی",
        "معنادار است. رد شد چون قاعده هر دو تقسیم را می‌خواست، نه یکی. ⚠️ یک قید مهم",
        "هم دارد که در جریان پ باید برداشته شود: هایپرپارامتر قهرمان روی ماتریسِ با",
        "**تاریخچه‌ی انبساطی** تنظیم شده؛ نسخه‌ای که خودِ تاریخچه را عوض می‌کند، با",
        "پیکربندی تنظیم‌شده‌ی خودش شانس منصفانه‌تری دارد — قرینه‌ی همان درس F139.",
        "",
        f"**۳. ⚠️ ترکیب پس‌رویدادیِ سه تغییر هم‌جهت، از بهترین جزئش **بدتر** شد** "
        f"({num(comb, '+.2f')}٪ در برابر {num(best_pct, '+.2f')}٪). ثبتش برای این است",
        "که وسوسه‌اش طبیعی است: سه چیز که هرکدام کمی کمک می‌کنند، لزوماً با هم بیشتر",
        "کمک نمی‌کنند — به‌ویژه وقتی هر سه از یک منبع (تاریخچه‌ی همان سلول) می‌آیند.",
        "",
        f"**۴. `cell_rho_lag7` عملاً یک ستون مرده است: پوشش {num(lag7, '.2f')}٪.**",
        "یعنی روی بیش از ۹۹٪ ردیف‌ها `NaN` است. برداشتن هر دو ستون lag روی هر سه تقسیم",
        "علامت منفی می‌دهد (یعنی بی‌ضرر یا کمی مفید). توصیه: حذفشان در بند ۱۴-۱۲-۲ بر",
        "پایه‌ی **پوشش** ثبت شود، نه بر پایه‌ی Δ — چون Δشان معنادار نیست و استناد به",
        "آن، همان بیش‌تفسیری است که این سند از آن پرهیز می‌کند. این ادامه‌ی مستقیم",
        "F220 است (`cell_rho_lag1` ‏۸۵.۳٪ تهی، و در فاز ۷ و ۹ فیچر شمرده شد).",
        "",
        "**۵. پیشین Beta بازبرازش شد و نتیجه‌اش برای اینجا کم‌اثر ولی برای بند ۱۴-۱۱",
        "تعیین‌کننده است.** قدرت پیشین سطح سلول از ۱۲.۱۳ به ۲۷.۵۸ رفت (میانگین تقریباً",
        "ثابت: ۰.۰۸۰۵ → ۰.۰۷۷۱)، یعنی با ۶۰۰ روز، نرخ‌های سلولی **کم‌پراکنده‌تر** از",
        "آنچه پنجره‌ی قفل‌شده نشان می‌داد درآمدند و کوچک‌سازی قوی‌تری می‌طلبند. ولی",
        "کوچک‌سازی به‌عنوان **فیچر** از سد قاعده رد نشد.",
        "",
        "⚠️ **و یک ناسازگاری که همین بازبرازش آشکارش کرد:** `composition_v4.py` نرخ",
        "**فرد** را با پیشین سطح **سلول** کوچک می‌کند — میانگین ۰.۰۸۰۵ در حالی که",
        "پیشین واقعیِ سطح فرد روی همین داده میانگین ۰.۱۱۱۲ دارد (F259، دور ۹). تفاوت",
        "۳۸٪ در لنگر کوچک‌سازی. این می‌تواند بخشی از دلیل ضعف خانواده‌ی `composition`",
        "باشد (F277) و **آزمونش کار بند ۱۴-۱۱ است**، نه اینجا.",
        "",
    ]
    text = "\n".join(lines) + "\n"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "history_features.md").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    set_global_seed()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    prior = refit_prior()
    pd.Series(prior).to_csv(OUTPUT_DIR / "history_prior.csv", header=False)
    print("\n=== پیشین Beta ===")
    for k, v in prior.items():
        print(f"  {k:22s} {v:,.4f}" if isinstance(v, float) else f"  {k:22s} {v}")

    df = build_variant_columns(pd.read_parquet(FEATURES_PATH), prior)
    cov = coverage_table(df)
    print("\n=== پوشش ستون‌های نامزد ===")
    print(cov.to_string(index=False))

    results = evaluate()
    write_report(results, prior, cov)
    print("\n=== حکم‌ها ===")
    print(decide(results).to_string(index=False))


if __name__ == "__main__":
    main()
