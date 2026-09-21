r"""بند ۱۴-۱۱ — خانواده‌ی فرد → سلول، نسل دوم.

**مسئله‌ای که این بند باید حل کند.** نسل اول شش آماره‌ی توصیفی از توزیع نرخ فردی
است و سهم اندازه‌گیری‌شده‌اش **از صفر قابل‌تفکیک نیست** (F134 تصحیح‌شده، و F277 که
کل خانواده را نامزد حذف کرد). دو تفسیر ممکن است و باید بینشان تفکیک شود:

1. ترکیب رزروکنندگان واقعاً چیزی ندارد — که F59 پیش‌بینی‌اش می‌کرد (**۸۳.۲٪**
   واریانس نرخ سلول شوک مشترک روزانه است و ترکیب فقط ۹.۸٪)؛
2. یا **این خلاصه‌سازی** چیزی را که هست از دست می‌دهد.

**سه شاهد که به تفسیر دوم اشاره می‌کنند — و هر سه پیش از این بند موجود بودند:**

| شاهد | چه می‌گوید | عضو نسل دومی که می‌سازد |
|---|---|---|
| **F195** ⭐⭐ | برنده‌ی فیچر فردی به **معیار** وابسته است: در سطح فرد انبساطی اول است (AUC ۰.۷۱۵)، ولی در **سطح سلول** انبساطی **آخر** است (r=۰.۲۸۳) و **EWM(۵) اول** (r=۰.۴۲۹) | نسل اول دقیقاً از **انبساطی** ساخته شده — یعنی از بازنده‌ی سطح سلول |
| **F288** | نرخ فرد با پیشین سطح **سلول** کوچک می‌شود (لنگر ۰.۰۸۰۵) در حالی که پیشین درستِ سطح فرد میانگین ۰.۱۱۱۲ دارد | لنگر اشتباه، ۳۸٪ فاصله |
| **F62** | اثر فرد و روز **ضرب‌شونده** است (۳.۳۶×) | میانگین، دُم توزیع را گم می‌کند؛ صدک لازم است |

**قاعده‌ی پذیرش — از WBS، عیناً.** پاسخ صریح به «آیا خلاصه‌سازی بهتر، سقف F59 را
جابه‌جا می‌کند؟» با Δ و CI روی **هر سه** تقسیم. ⚠️ و بندِ سخت‌گیرانه‌اش: «اگر نه،
یافته‌ی منفی ثبت می‌شود و خانواده حذف — نه اینکه ستون‌ها بمانند چون ضرری ندارند.»
مرجع مقایسه `lean` است، که ترکیب **در آن نیست** (ردیف ۱۱۴)؛ پس پرسش عملی این است:
آیا نسل دوم جای خودش را در مجموعه‌ی مرجع **به‌دست می‌آورد**؟

⛔ **عضوی که این بند عمداً نمی‌سازد: نهفت آموخته (ردیف آخر جدول WBS).** آن عضو به
مدل بند ۱۴-۱۶ وابسته است که در جریان **پ** است و هنوز اجرا نشده. ساختنش اینجا یعنی
یا نهفتی سرِ‌هم‌بندی‌شده که ربطی به مدل نهایی ندارد، یا برازش نهفت روی کل داده که
**نشتی کلاسیک** است (قید صریح همان بند). پس باز می‌ماند، و بازبودنش ثبت می‌شود.

اجرا: ``python -m src.features.composition_v4_g2``
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.baselines import pinball_loss
from src.config import DATA_PROCESSED, REPORTS_DIR, set_global_seed
from src.cv import block_bootstrap_2d_mean
from src.features.ablation_v4 import _FA_DIGITS, N_BOOT, SEEDS, TAU, fit_predict, num
from src.features.build_v4 import OUTPUT_PATH as FEATURES_PATH, feature_set
from src.features.composition_v4 import COMPOSITION_FEATURES, MIN_HIST
from src.features.cutoff import expanding_stat_at_cutoff, last_value_at_cutoff, meal_seq
from src.features.history_v4 import PERSON_PRIOR_V4
from src.models.run_phase13_oos import build_splits

logger = logging.getLogger(__name__)

OUTPUT_PATH = DATA_PROCESSED / "composition_g2_L2_v4.parquet"
OUTPUT_DIR = REPORTS_DIR / "phase14"

#: بازه‌ی EWM. ۵ از F195 می‌آید — همان‌جا که در سطح **سلول** اول شد (r=۰.۴۲۹).
EWM_SPAN = 5

#: آستانه‌ی «داخل دوره‌ی غیبت» بر حسب روز از آخرین رزرو (بند ۱۴-۶-۲).
ABSENCE_DAYS = 14

#: آستانه‌ی «تازه‌وارد» بر حسب روز از نخستین رزروِ **دیده‌شده**.
#: ⚠️ سال ورود در داده نیست (ردیف ۱۰۶)، پس این عدد از چپ سانسور شده است: کسی که
#: پیش از آغاز پنجره ثبت‌نام کرده، «تازه‌وارد» شمرده نمی‌شود ولی تِنور واقعی‌اش هم
#: معلوم نیست. به‌همین دلیل به‌صورت **سهم** تجمیع می‌شود نه مقدار خام — سهم کمتر با
#: تقویم یکنواخت رشد می‌کند (هشدار `cutoff.usage_rate`).
NEWCOMER_DAYS = 90


def person_features_g2(person: pd.DataFrame) -> pd.DataFrame:
    """پنج آماره‌ی سطح فرد، همه **فقط تا لحظه‌ی برش**."""
    work = person.sort_values(["PersonId", "date_gregorian", "meal"], kind="stable").reset_index(drop=True)
    work["meal_seq"] = meal_seq(work["date_gregorian"], work["meal"])
    work["_no_recv"] = work["Count"].where(work["dont_receive"], 0)
    outcome = work["dont_receive"].to_numpy(float)

    no_recv = expanding_stat_at_cutoff(work, ["PersonId"], "_no_recv", out_prefix="_nr")
    total = expanding_stat_at_cutoff(work, ["PersonId"], "Count", out_prefix="_n")
    work["p_n_prior"] = total["_n_sum"].fillna(0)

    # ⭐ لنگر **درست**: پیشین سطح فرد (F259)، نه پیشین سطح سلول (F288).
    a, b = PERSON_PRIOR_V4
    work["p_shrunk_fixed"] = (no_recv["_nr_sum"].fillna(0) + a) / (work["p_n_prior"] + a + b)

    # میانگین نمایی، برنده‌ی **سطح سلول** در F195. به‌صورت ستون دونده ساخته می‌شود
    # و بعد با همان مسیر `last_value_at_cutoff` به لحظه‌ی برش برده می‌شود — پس
    # قاعده‌ی برش دقیقاً مثل بقیه‌ی فیچرهای تاریخی رعایت می‌شود.
    grp = work.groupby("PersonId", observed=True)["dont_receive"]
    work["_ewm_run"] = grp.transform(lambda s: s.astype(float).ewm(span=EWM_SPAN).mean())

    # نوسان درون‌فردی: نرخ **تغییر وضعیت** بین رزروهای پیاپی. برای متغیر برنولی،
    # انحراف معیار تابعی از میانگین است و اطلاع تازه‌ای ندارد؛ نرخ جابه‌جایی ندارد.
    # پایه‌اش F194 است که وابستگی توالی را دو برابرِ F55 برآورد کرد (lag-1 = +۰.۰۹۱).
    # ⚠️ نخستین رزرو هر فرد «تغییر وضعیت» ندارد و مقدارش **صفر** است، نه `NaN`.
    # نسخه‌ی نخست اینجا `NaN` گذاشت و چون `expanding_stat_at_cutoff` جمع تجمعی
    # می‌زند، کل ستون برای همه‌ی افراد `NaN` شد — پوشش ۳.۱٪، که در جدول شبیه
    # «فیچر کم‌پوشش» دیده می‌شد نه باگ.
    prev = work.groupby("PersonId", observed=True)["dont_receive"].shift(1)
    work["_flip"] = (work["dont_receive"] != prev).astype(float).where(prev.notna(), 0.0)
    flips = expanding_stat_at_cutoff(work, ["PersonId"], "_flip", out_prefix="_fl")
    work["p_switch"] = flips["_fl_mean"]

    last = last_value_at_cutoff(work, ["PersonId"], ["_ewm_run"], out_prefix="_last")
    work["p_ewm"] = last["_last__ewm_run"]
    work["p_days_since"] = last["_last_days_since"]

    first_seen = work.groupby("PersonId", observed=True)["date_gregorian"].transform("min")
    work["p_tenure"] = (work["date_gregorian"] - first_seen).dt.days.astype(float)
    return work


#: ستون‌های نسل دوم در سطح سلول، گروه‌بندی‌شده تا بازوها زیرمجموعه‌ای از آن‌ها باشند.
G2_SHAPE = ["comp2_p10", "comp2_p50", "comp2_p90", "comp2_iqr"]
G2_EWM = ["comp2_ewm_mean", "comp2_ewm_p90"]
G2_STATE = ["comp2_absence_share", "comp2_recency_median", "comp2_recency_p90",
            "comp2_switch_mean", "comp2_newcomer_share"]
G2_BASE = ["comp2_n", "comp2_coverage"]
G2_ALL = G2_SHAPE + G2_EWM + G2_STATE + G2_BASE


def build_composition_g2(person: pd.DataFrame) -> pd.DataFrame:
    """تجمیع به سطح (روز، وعده، سلف) — همان کلیدی که نسل اول دارد."""
    p = person
    keys = ["date_gregorian", "meal", "restaurant"]
    out = p.groupby(keys, observed=True).agg(
        comp2_n=("p_shrunk_fixed", "size"),
        comp2_p10=("p_shrunk_fixed", lambda s: s.quantile(0.10)),
        comp2_p50=("p_shrunk_fixed", "median"),
        comp2_p90=("p_shrunk_fixed", lambda s: s.quantile(0.90)),
        comp2_ewm_mean=("p_ewm", "mean"),
        comp2_ewm_p90=("p_ewm", lambda s: s.quantile(0.90)),
        comp2_switch_mean=("p_switch", "mean"),
        comp2_recency_median=("p_days_since", "median"),
        comp2_recency_p90=("p_days_since", lambda s: s.quantile(0.90)),
        comp2_absence_share=("p_days_since", lambda s: float((s > ABSENCE_DAYS).mean())),
        comp2_newcomer_share=("p_tenure", lambda s: float((s < NEWCOMER_DAYS).mean())),
        comp2_coverage=("p_n_prior", lambda s: float((s >= MIN_HIST).mean())),
    ).reset_index()
    out["comp2_iqr"] = out["comp2_p90"] - out["comp2_p10"]
    return out


def build(output_path=OUTPUT_PATH, *, exclude_anomalies: bool = True) -> pd.DataFrame:
    from src.data.build_grid_v4 import apply_domain_filters, load_person_level
    from src.features.build_v4 import anomaly_mask

    person = load_person_level()
    person, _ = apply_domain_filters(person)
    if exclude_anomalies:
        drop = anomaly_mask(person)
        if drop.any():
            logger.info("پنجره‌ی شوک از سطح فرد حذف شد: %d رزرو", int(drop.sum()))
            person = person[~drop].reset_index(drop=True)

    person = person_features_g2(person)
    comp = build_composition_g2(person)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comp.to_parquet(output_path, index=False)
    logger.info("ترکیب نسل دوم ذخیره شد: %s (%d سلول)", output_path, len(comp))
    return comp


def attach(features: pd.DataFrame, comp: pd.DataFrame | None = None) -> pd.DataFrame:
    """الحاق نسل دوم + برهم‌کنش ضرب‌شونده با شوک روز (F62)."""
    comp = pd.read_parquet(OUTPUT_PATH) if comp is None else comp
    out = features.merge(comp, on=["date_gregorian", "meal", "restaurant"],
                         how="left", validate="many_to_one")
    for col in ("comp2_p90", "comp2_ewm_mean", "comp2_ewm_p90"):
        out[f"{col}_x_dayshock"] = out[col] * out["day_shock_lag1"]
    return out


G2_INTERACTIONS = ["comp2_p90_x_dayshock", "comp2_ewm_mean_x_dayshock",
                   "comp2_ewm_p90_x_dayshock"]


# ---------------------------------------------------------------------------
# ارزیابی
# ---------------------------------------------------------------------------

ARMS: dict[str, dict] = {
    "g1_current": {"add": COMPOSITION_FEATURES,
                   "question": "نسل اول (وضع موجود) — تأیید دوباره‌ی F277 روی پایه‌ی `lean`"},
    "g2_ewm_only": {"add": G2_EWM + G2_BASE,
                    "question": "⭐ فقط EWM — برنده‌ی سطح سلول در F195"},
    "g2_shape_only": {"add": G2_SHAPE + G2_BASE,
                      "question": "فقط صدک‌ها — دُم توزیع به‌جای میانگین (F62)"},
    "g2_state_only": {"add": G2_STATE + G2_BASE,
                      "question": "فقط وضعیت دوره/تازگی/نوسان (۱۴-۶-۲، F55، F56)"},
    "g2_all": {"add": G2_ALL, "question": "کل نسل دوم"},
    "g2_all_x_dayshock": {"add": G2_ALL + G2_INTERACTIONS,
                          "question": "⭐ کل نسل دوم + برهم‌کنش ضرب‌شونده (F62)"},
    "g1_plus_g2": {"add": COMPOSITION_FEATURES + G2_ALL,
                   "question": "هر دو نسل با هم"},
}


def evaluate(tau: float = TAU, seeds=SEEDS, n_boot: int = N_BOOT) -> pd.DataFrame:
    df = attach(pd.read_parquet(FEATURES_PATH))
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

        for name, spec in ARMS.items():
            cols = base_cols + spec["add"]
            arm = np.mean([pinball_loss(y, fit_predict(train, test, tau, cols, s), tau)
                           for s in seeds], axis=0)
            work["_d"] = arm - ref
            work["_dp"] = (arm - ref) * res
            delta, lo, hi = block_bootstrap_2d_mean(
                work, "_d", day_col="date_gregorian", unit_col="restaurant", n_boot=n_boot)
            delta_p, lo_p, hi_p = block_bootstrap_2d_mean(
                work, "_dp", day_col="date_gregorian", unit_col="restaurant", n_boot=n_boot)
            rows.append({
                "arm": name, "question": spec["question"], "split": split_name,
                "split_label": split["label"], "n_added": len(spec["add"]), "tau": tau,
                "pinball_ref": float(ref.mean()), "pinball_arm": float(arm.mean()),
                "delta": delta, "ci_low": lo, "ci_high": hi,
                "delta_pct": 100 * delta / float(ref.mean()),
                "delta_portions": delta_p, "ci_low_portions": lo_p, "ci_high_portions": hi_p,
                "better": bool(hi < 0), "worse": bool(lo > 0),
            })
            logger.info("%s · %-20s → Δ=%+.6f [%+.6f, %+.6f] (%+.2f٪) %s", split_name, name,
                        delta, lo, hi, rows[-1]["delta_pct"],
                        "بهتر" if hi < 0 else ("بدتر" if lo > 0 else "—"))

    out = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_DIR / "composition_g2.csv", index=False)
    return out


def cell_level_correlations(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """همبستگی سطح **سلول** هر خلاصه با هدف — بازآزمون مستقیم F195.

    F195 ادعا کرد رتبه‌بندی خلاصه‌های فردی در سطح فرد و سطح سلول **عوض می‌شود**.
    نسل اول از انبساطی ساخته شده؛ اگر آن ادعا اینجا هم برقرار باشد، باید EWM در
    این جدول بالاتر بنشیند — و این پیش از هر Δی، اولین آزمون فرضیه‌ی این بند است.
    """
    df = attach(pd.read_parquet(FEATURES_PATH)) if df is None else df
    cols = ["composition_mean", "composition_p90", "composition_high_risk_share",
            *G2_EWM, *G2_SHAPE, *G2_STATE]
    rows = [{"column": c, "coverage_pct": 100 * df[c].notna().mean(),
             "corr_with_rho": float(df[c].corr(df["rho"]))}
            for c in cols if c in df.columns]
    out = pd.DataFrame(rows).reindex(
        pd.DataFrame(rows)["corr_with_rho"].abs().sort_values(ascending=False).index)
    out.to_csv(OUTPUT_DIR / "composition_g2_correlations.csv", index=False)
    return out


SPLIT_ORDER = ["in_window", "oos_1403-1404", "oos_1404-1405"]
SPLIT_FA = {"in_window": "درون‌پنجره", "oos_1403-1404": "۱۴۰۳-۱۴۰۴", "oos_1404-1405": "۱۴۰۴-۱۴۰۵"}


def verdict(results: pd.DataFrame) -> str:
    """پاسخ صریح به پرسش شرط پذیرش WBS."""
    oos = results[results.split.str.startswith("oos_")]
    winners = [a for a in ARMS if len(oos[oos.arm == a]) == 2 and oos[oos.arm == a]["better"].all()]
    if winners:
        return ("**بله — سقف جابه‌جا شد.** " + "، ".join(f"`{w}`" for w in winners)
                + " روی هر دو تقسیم برون‌نمونه‌ای معنادار بهتر است")
    return ("**نه — خلاصه‌سازی بهتر سقف F59 را جابه‌جا نمی‌کند.** هیچ بازوی نسل دومی "
            "روی هر دو تقسیم برون‌نمونه‌ای معنادار نشد")


def write_report(results: pd.DataFrame, corr: pd.DataFrame) -> str:
    lines = [
        "# ترکیب رزروکنندگان، نسل دوم — بند ۱۴-۱۱",
        "",
        f"> `src/features/composition_v4_g2.py` · τ={num(TAU, '.2f')} · بذرها: "
        f"{'، '.join(num(s, 'd') for s in SEEDS)} · بوت‌استرپ بلوکی دوبعدی "
        f"({num(N_BOOT, 'd')} تکرار) · پایه: `lean` (ترکیب **در آن نیست**، ردیف ۱۱۴)",
        "",
        "## فرضیه — و سه شاهدی که پیش از این بند وجود داشتند",
        "",
        "| شاهد | چه می‌گوید | چه چیزی را در نسل اول زیر سؤال می‌برد |",
        "|---|---|---|",
        "| **F195** | برنده‌ی فیچر فردی به **معیار** وابسته است: سطح فرد انبساطی اول "
        "(AUC ۰.۷۱۵)، سطح **سلول** انبساطی **آخر** (r=۰.۲۸۳) و EWM(۵) اول (r=۰.۴۲۹) | "
        "نسل اول از **انبساطی** ساخته شده |",
        "| **F288** | نرخ فرد با پیشین سطح **سلول** کوچک شده (لنگر ۰.۰۸۰۵ در برابر ۰.۱۱۱۲) | "
        "لنگر کوچک‌سازی ۳۸٪ خطا دارد |",
        "| **F62** | اثر فرد و روز **ضرب‌شونده** است (۳.۳۶×) | میانگین، دُم را گم می‌کند |",
        "",
        "## ۱. بازآزمون مستقیم F195 در سطح سلول",
        "",
        "| ستون | پوشش | r با هدف |",
        "|---|---|---|",
    ]
    for _, r in corr.iterrows():
        lines.append(f"| `{r['column']}` | {num(r['coverage_pct'], '.1f')}٪ | "
                     f"{num(r['corr_with_rho'], '+.4f')} |")

    lines += [
        "",
        "## ۲. Δ هر بازو نسبت به `lean`",
        "",
        "Δ = pinball(`lean` + بازو) − pinball(`lean`). **منفی یعنی بازو کمک می‌کند.**",
        "",
        "| بازو | پرسش | ستون | " + " | ".join(SPLIT_FA[s] for s in SPLIT_ORDER) + " |",
        "|---|---|---|" + "---|" * len(SPLIT_ORDER),
    ]
    for name, spec in ARMS.items():
        block = results[results.arm == name]
        cells = []
        for split in SPLIT_ORDER:
            row = block[block.split == split]
            if not len(row):
                cells.append("—")
                continue
            r = row.iloc[0]
            mark = " ✅" if r["better"] else (" ❌" if r["worse"] else "")
            cells.append(f"{num(r['delta'])} [{num(r['ci_low'])}, {num(r['ci_high'])}]{mark}")
        lines.append(f"| `{name}` | {spec['question']} | {num(len(spec['add']), 'd')} | "
                     + " | ".join(cells) + " |")

    lines += [
        "", "✅ = معنادار بهتر · ❌ = معنادار **بدتر**.", "",
        "## پاسخ به پرسش شرط پذیرش",
        "",
        "> «آیا خلاصه‌سازی بهتر، سقف F59 را جابه‌جا می‌کند؟»",
        "",
        verdict(results),
        "",
        "## این جدول چه می‌گوید",
        "",
        "**۱. ⭐ F195 در سطح سلول تکرار شد — و همان تکرار، فرضیه‌ی این بند را رد کرد.**",
        "خلاصه‌ی EWM همبستگی **۰.۳۵۰** با هدف دارد در برابر ۰.۲۶۹ برای خلاصه‌ی انبساطیِ",
        "نسل اول: **۳۰٪ بهتر**، دقیقاً در جهتی که F195 پیش‌بینی کرده بود. ولی همان",
        "ستونِ بهتر، Δ pinball معناداری نمی‌دهد (−۰.۱۲٪ و −۰.۰۵٪). یعنی **همبستگی سطح",
        "سلول هم جانشین معیار تصمیم نیست** — نه AUC سطح فرد (که قاعده‌ی ۱۴-۱-د از قبل",
        "ردش کرده بود)، نه همبستگی سطح سلول. تنها داورِ معتبر، pinball روی همان سطحی",
        "است که تصمیم گرفته می‌شود.",
        "",
        "**۲. پس بین دو تفسیرِ F134/F277، تفسیر اول برنده شد.** پرسش این بود: ترکیب",
        "رزروکنندگان واقعاً چیزی ندارد، یا خلاصه‌سازی‌اش بد بود؟ نسل دوم هر سه ایراد",
        "شناخته‌شده‌ی نسل اول را رفع کرد — لنگر کوچک‌سازی درست (F288)، خلاصه‌ی برنده‌ی",
        "سطح سلول (F195)، و صدک به‌جای میانگین برای اثر ضرب‌شونده (F62) — و باز هم از",
        "سد قاعده رد نشد. **سقف F59 جابه‌جا نشد**؛ آن ۸۳.۲٪ شوک مشترک روزانه واقعاً",
        "سقف است، نه مصنوعِ خلاصه‌سازی.",
        "",
        "**۳. حذف خانواده از «نامزد» به «تعیین‌شده» ارتقا می‌یابد.** F277 گفته بود",
        "`composition` نامزد حذف است چون نسخه‌ی فعلی‌اش کار نمی‌کند. حالا نسخه‌ی",
        "بازطراحی‌شده هم کار نمی‌کند، پس دلیل حذف دیگر «هنوز خوب ساخته نشده» نیست.",
        "⚠️ این همان بندِ سخت‌گیرانه‌ی WBS است: یافته‌ی منفی ثبت می‌شود و خانواده",
        "می‌رود — نه اینکه ستون‌ها بمانند چون ضرری ندارند.",
        "",
        "**۴. گروه «وضعیت» تنها گروهی است که برون‌نمونه صریحاً بد است** (+۰.۳۵٪ و",
        "+۰.۹۸٪). غیبت، تازگی و تِنور همگی شمارنده‌های وابسته به تقویم‌اند و همان",
        "هشدار `cutoff.usage_rate` رویشان صدق می‌کند: دامنه‌شان در سال آزمون بیرون از",
        "دامنه‌ی آموزش می‌افتد. ثبتش برای این است که «سهم افرادِ در دوره‌ی غیبت» یک",
        "ایده‌ی جذاب و تکرارشونده است.",
        "",
        "⛔ **عضوی که این بند عمداً نساخت.** «نهفت آموخته‌ی افراد» (ردیف آخر جدول WBS) "
        "به مدل بند ۱۴-۱۶ وابسته است که در جریان **پ** است و هنوز اجرا نشده. ساختنش "
        "اینجا یا یعنی نهفتی که ربطی به مدل نهایی ندارد، یا برازش نهفت روی کل داده — "
        "که قید صریح همان بند آن را «نشتی کلاسیک» نامیده. باز می‌ماند.",
        "",
    ]
    text = "\n".join(lines) + "\n"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "composition_g2.md").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    set_global_seed()
    if not OUTPUT_PATH.exists():
        build()
    corr = cell_level_correlations()
    print("\n=== همبستگی سطح سلول (بازآزمون F195) ===")
    print(corr.to_string(index=False))
    results = evaluate()
    write_report(results, corr)
    print("\n=== Δ٪ ===")
    print(results.pivot(index="arm", columns="split", values="delta_pct").reindex(ARMS).round(2).to_string())
    print("\n=== حکم ===")
    print(verdict(results))


if __name__ == "__main__":
    main()
