"""دور ۵ / مرحله ۳ — ساختار **درون‌گروهی** هدررفت (Q40–Q42)

**چرا این تحلیل لازم شد.** F16 می‌گوید ساکنان خوابگاه *به‌طور میانگین* نرخ کمتری دارند و
Q37 همین دور نشان داد در دم توزیع هم کم‌نماینده‌اند (غنی‌شدگی ۰.۶۰). ولی هیچ‌کدام نمی‌گوید
**داخل خوابگاه** چه کسی هدر می‌دهد — و این دقیقاً سؤال سیاست‌گذار است، چون مداخله‌ی
خوابگاهی روی ساکنان خوابگاه اعمال می‌شود، نه روی مقایسه‌ی آن‌ها با غیرساکنان.

F66 «سرایت اجتماعی خوابگاهی» را رد کرد، ولی آن سؤال متفاوتی بود (آیا هم‌خوابگاهی‌ها
*هم‌زمان* بد می‌شوند؟) و چیزی درباره‌ی **تفاوت پایدار بین ساختمان‌ها** نمی‌گوید.

سه سؤال: پروفایل داخل خوابگاه (Q40)، همان در آینه‌ی غیرخوابگاهی (Q41)، و آزمون صریح
اینکه آیا سازوکار داخل و بیرون **یکی** است (Q42) — خطر پارادوکس سیمپسون که F32 در سطح
سلف نشان داد.

اجرا: `python -m src.eda_lib.runners.r5b_within_strata`
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.proportion import proportion_confint

from src.config import FIGURES_DIR
from src.eda_lib.figio import save_fig
from src.eda_lib.runners._common import header, kv, pct, setup
from src.viz_fa import fa
from src.viz_fa import setup as viz_setup

FIG_DIR = FIGURES_DIR / "round5"
PER_PATH = "data/interim/r5_person_level.parquet"
MIN_N_UNIT = 50     # حداقل نفر در یک ساختمان/واحد تا واردش کنیم
DEMO_COLS = [("Gender", "جنسیت"), ("DegreeName", "مقطع"),
             ("edu_group", "دوره"), ("CollegeName", "دانشکده")]


def load_per() -> pd.DataFrame:
    per = pd.read_parquet(PER_PATH)
    # دهک «درون‌گروهی»: بدترین ۱۰٪ *همان زیرجامعه* — تا مقایسه‌ی سازوکارها زیر سایه‌ی
    # اختلاف نرخ پایه‌ی دو زیرجامعه گم نشود (نرخ پایه خودش یافته‌ی F16/Q37 است).
    per["worst_within"] = False
    for _, idx in per.groupby("is_dorm_resident").groups.items():
        sub = per.loc[idx]
        k = int(round(0.10 * len(sub)))
        top = sub.sort_values("eb_rate", ascending=False).index[:k]
        per.loc[top, "worst_within"] = True
    return per


def enrich(sub: pd.DataFrame, flag: str, col: str, min_n: int) -> pd.DataFrame:
    base = sub[flag].mean()
    rows = []
    for cat, g in sub.groupby(col, observed=True):
        if len(g) < min_n:
            continue
        k, n = int(g[flag].sum()), len(g)
        lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
        rows.append({"دسته": str(cat), "n": n, "نرخ عضویت": k / n,
                     "غنی‌شدگی": (k / n) / base, "CI پایین": lo / base, "CI بالا": hi / base,
                     "نرخ EB میانگین": float(g["eb_rate"].mean())})
    return pd.DataFrame(rows).sort_values("غنی‌شدگی", ascending=False)


def unit_spread_test(sub: pd.DataFrame, unit_col: str, label: str) -> None:
    """آیا تفاوت بین واحدها واقعی است یا نوسان نمونه‌گیری؟

    کروسکال-والیس روی نرخ EB افراد + سهم واریانس بین‌واحدی (ANOVA یک‌طرفه‌ی η²).
    """
    g = [x["eb_rate"].to_numpy() for _, x in sub.groupby(unit_col, observed=True)
         if len(x) >= MIN_N_UNIT]
    if len(g) < 3:
        print(f"  [skip] {label}: کمتر از ۳ واحد با n≥{MIN_N_UNIT}")
        return
    H, p = stats.kruskal(*g)
    allv = np.concatenate(g)
    ss_between = sum(len(x) * (x.mean() - allv.mean()) ** 2 for x in g)
    eta2 = ss_between / ((allv - allv.mean()) ** 2).sum()
    kv(f"{label}: کروسکال-والیس H", f"{H:.1f} (p={p:.2e}، {len(g)} واحد)")
    kv(f"{label}: η² بین‌واحدی", f"{eta2:.4f}")


def q40(per: pd.DataFrame) -> dict:
    header("Q40 — داخل ساکنان خوابگاه: هدردهنده چه ویژگی‌ای دارد؟")
    dorm = per[per["is_dorm_resident"]].copy()
    kv("تعداد ساکن خوابگاه", f"{len(dorm):,}")
    kv("نرخ EB میانگین این زیرجامعه", f"{dorm['eb_rate'].mean():.4f}")

    print("\n— ساختمان خوابگاه (۱۰ بالا / ۵ پایین، تعریف: بدترین ۱۰٪ *داخل خوابگاه*) —")
    t = enrich(dorm, "worst_within", "dorm_canonical", MIN_N_UNIT)
    print(pd.concat([t.head(10), t.tail(5)]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    unit_spread_test(dorm, "dorm_canonical", "ساختمان خوابگاه")

    out = {"dorm_unit": t}
    for col, label in DEMO_COLS:
        e = enrich(dorm, "worst_within", col, 60)
        out[col] = e
        print(f"\n— {label} داخل خوابگاه —")
        print((e.head(10) if len(e) > 10 else e).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return out


def q41(per: pd.DataFrame, dorm_out: dict) -> dict:
    header("Q41 — آینه: داخل غیرساکنان (واحد دانشکده‌ای)")
    non = per[~per["is_dorm_resident"]].copy()
    kv("تعداد غیرساکن", f"{len(non):,}")
    kv("نرخ EB میانگین این زیرجامعه", f"{non['eb_rate'].mean():.4f}")

    print("\n— واحد دانشکده‌ای (۱۰ بالا / ۵ پایین) —")
    t = enrich(non, "worst_within", "dorm_canonical", MIN_N_UNIT)
    print(pd.concat([t.head(10), t.tail(5)]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    unit_spread_test(non, "dorm_canonical", "واحد دانشکده‌ای")

    out = {"unit": t}
    for col, label in DEMO_COLS:
        e = enrich(non, "worst_within", col, 60)
        out[col] = e
        print(f"\n— {label} داخل غیرخوابگاهی —")
        print((e.head(10) if len(e) > 10 else e).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # آیا رتبه‌بندی دانشکده‌ها بین دو زیرجامعه یکی است؟
    header("Q41ب — آیا رتبه‌بندی دانشکده‌ها در دو زیرجامعه یکسان است؟", level=2)
    a = dorm_out["CollegeName"].set_index("دسته")["نرخ EB میانگین"]
    b = out["CollegeName"].set_index("دسته")["نرخ EB میانگین"]
    common = a.index.intersection(b.index)
    if len(common) >= 5:
        rho, p = stats.spearmanr(a[common], b[common])
        r, pp = stats.pearsonr(a[common], b[common])
        kv("تعداد دانشکده‌ی مشترک", len(common))
        kv("همبستگی رتبه‌ای اسپیرمن", f"{rho:+.4f} (p={p:.4f})")
        kv("همبستگی پیرسون", f"{r:+.4f} (p={pp:.4f})")
        cmp = pd.DataFrame({"خوابگاهی": a[common], "غیرخوابگاهی": b[common]})
        cmp["اختلاف"] = cmp["خوابگاهی"] - cmp["غیرخوابگاهی"]
        print("\n" + cmp.sort_values("اختلاف").to_string(float_format=lambda v: f"{v:.4f}"))
        out["college_cmp"] = cmp
    return out


def q42(per: pd.DataFrame) -> pd.DataFrame:
    header("Q42 — آزمون برهم‌کنش صریح: آیا سازوکار داخل و بیرون خوابگاه یکی است؟")
    d = per[per["n_res"] >= 5].copy()
    d["is_dorm"] = d["is_dorm_resident"].astype(float)
    d["male"] = (d["Gender"] == "مرد").astype(float)
    d["night"] = (d["edu_group"] == "شبانه/نوبت دوم").astype(float)
    d["grad"] = d["DegreeName"].str.contains("ارشد|دکتری").astype(float)
    d["tehran"] = d["is_tehran"].astype(float)

    base = ["male", "night", "grad", "tehran"]
    rows = []
    # مدل خطی روی نرخ EB (پیوسته، توان بیشتر از عضویت باینری در دهک)
    for v in base:
        X = pd.DataFrame({
            "const": 1.0, "is_dorm": d["is_dorm"], v: d[v],
            f"{v}×خوابگاه": d[v] * d["is_dorm"],
        })
        m = sm.OLS(d["eb_rate"], X).fit(cov_type="HC3")
        # اثر ساده در هر زیرجامعه
        eff_non = m.params[v]
        eff_dorm = m.params[v] + m.params[f"{v}×خوابگاه"]
        rows.append({
            "متغیر": v,
            "اثر در غیرخوابگاهی": eff_non,
            "اثر در خوابگاهی": eff_dorm,
            "ضریب برهم‌کنش": m.params[f"{v}×خوابگاه"],
            "p برهم‌کنش": m.pvalues[f"{v}×خوابگاه"],
            "علامت عوض می‌شود؟": "بله" if eff_non * eff_dorm < 0 else "خیر",
        })
    tbl = pd.DataFrame(rows)
    print("\n(متغیر وابسته: نرخ EB فرد؛ خطای معیار مقاوم HC3)")
    print(tbl.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    return tbl


def figures(per: pd.DataFrame, dorm_out: dict, non_out: dict, inter: pd.DataFrame) -> None:
    viz_setup()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # شکل ۱ — پراکندگی واحدها در دو زیرجامعه
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
    for a, (t, lab) in zip(axes, [(dorm_out["dorm_unit"], "ساختمان خوابگاه"),
                                  (non_out["unit"], "واحد دانشکده‌ای")]):
        tt = t.sort_values("نرخ EB میانگین")
        y = np.arange(len(tt))
        a.barh(y, tt["نرخ EB میانگین"], color="#4a7" if "خوابگاه" in lab else "#47c")
        a.set_yticks(y)
        a.set_yticklabels([fa(s[:38]) for s in tt["دسته"]], fontsize=7)
        a.axvline(per["eb_rate"].mean(), ls="--", c="k", lw=1)
        a.set_title(fa(f"{lab} (n≥{MIN_N_UNIT})"))
        a.set_xlabel(fa("میانگین نرخ کوچک‌سازی‌شده"))
    fig.suptitle(fa("پراکندگی نرخ بین واحدهای سکونت/تحصیل — خط‌چین: میانگین کل"))
    fig.tight_layout()
    save_fig(fig, "R5_q40_units", FIG_DIR)
    plt.close(fig)

    # شکل ۲ — برهم‌کنش: اثر هر ویژگی در دو زیرجامعه
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    x = np.arange(len(inter))
    w = 0.38
    ax.bar(x - w / 2, inter["اثر در غیرخوابگاهی"], w, label=fa("غیرخوابگاهی"), color="#47c")
    ax.bar(x + w / 2, inter["اثر در خوابگاهی"], w, label=fa("خوابگاهی"), color="#4a7")
    ax.axhline(0, c="k", lw=1)
    lbl = {"male": "مرد", "night": "شبانه", "grad": "تحصیلات تکمیلی", "tehran": "تهران"}
    ax.set_xticks(x)
    ax.set_xticklabels([fa(lbl[v]) for v in inter["متغیر"]])
    ax.set_ylabel(fa("اثر بر نرخ کوچک‌سازی‌شده"))
    ax.set_title(fa("اثر هر ویژگی جمعیتی، جدا در دو زیرجامعه"))
    ax.legend()
    for i, r in inter.reset_index().iterrows():
        if r["p برهم‌کنش"] < 0.05:
            ax.text(i, max(r["اثر در غیرخوابگاهی"], r["اثر در خوابگاهی"]) + 0.001, "*",
                    ha="center", fontsize=15)
    fig.tight_layout()
    save_fig(fig, "R5_q42_interaction", FIG_DIR)
    plt.close(fig)


def main() -> None:
    setup()
    per = load_per()
    d_out = q40(per)
    n_out = q41(per, d_out)
    inter = q42(per)
    figures(per, d_out, n_out, inter)
    print("\n[ok] مرحله ۳ تمام شد")


if __name__ == "__main__":
    main()
