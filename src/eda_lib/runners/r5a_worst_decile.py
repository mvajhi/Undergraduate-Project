"""دور ۵ / مرحله ۱-۲ — «بدترین ۱۰٪» چه کسانی‌اند؟ (Q35–Q39)

**چرا این تحلیل لازم شد.** دفتر حقایق اثر *حاشیه‌ای* جمعیت‌شناسی را دارد (F16 خوابگاه،
F50 دوره، F51 دانشکده، F52 جنسیت/مقطع) — یعنی مقایسه‌ی **میانگین** گروه‌ها. ولی سیاست
هدف‌گیری به میانگین نگاه نمی‌کند؛ به **دمِ** توزیع نگاه می‌کند. اثری که در میانگین ناچیز
است می‌تواند در دم قوی باشد، یا برعکس. هیچ‌کدام تا امروز آزموده نشده بود.

پیش از هر پروفایلی، یک گام روش‌شناختی اجباری: خودِ «بدترین ۱۰٪» سه تعریف رقیب دارد و
F49 صریحاً هشدار داده که نرخ خام فرد کم‌رزرو نباید مستقیم استفاده شود. Q35 این را حل
می‌کند، Q36 می‌گوید هدف‌گیری اصلاً صرف می‌کند یا نه، و Q37–Q39 پروفایل می‌سازند.

اجرا: `python -m src.eda_lib.runners.r5a_worst_decile`
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import optimize, special
from statsmodels.stats.proportion import proportion_confint

from src.config import FIGURES_DIR
from src.eda_lib.figio import save_fig
from src.eda_lib.individual_helpers import lorenz_curve, pareto_share
from src.eda_lib.runners._common import (
    PERSON_DIM_PATH,
    PERSON_FACT_PATH,
    header,
    kv,
    pct,
    setup,
)
from src.viz_fa import fa
from src.viz_fa import setup as viz_setup

FIG_DIR = FIGURES_DIR / "round5"
DECILE = 0.10          # «بدترین ۱۰٪»
MIN_RES_PROFILE = 5    # حداقل رزرو برای ورود به تحلیل پروفایل (نه برای تعریف دهک)


# ---------------------------------------------------------------------------
# بارگذاری
# ---------------------------------------------------------------------------

def load_person_level() -> pd.DataFrame:
    """یک ردیف به‌ازای هر فرد: تعداد رزرو، تعداد عدم‌دریافت، نرخ خام، + ویژگی‌های جمعیتی."""
    fact = pd.read_csv(
        PERSON_FACT_PATH,
        usecols=["PersonId", "Meal", "restaurant_canonical", "city", "is_tehran",
                 "dont_receive", "is_main_meal"],
    )
    fact = fact[fact["is_main_meal"]]
    g = fact.groupby("PersonId")
    per = pd.DataFrame({
        "n_res": g.size(),
        "n_norecv": g["dont_receive"].sum().astype(int),
    })
    # سلف غالب فرد و شهر غالب — برای Q39 (اثر مکان در برابر اثر فرد).
    # ⚠️ mode() گروهی روی ۲۶٬۷۶۸ گروه هم کند است و هم روی گروه تمام-NaN خطا می‌دهد؛
    # مسیر value_counts→idxmax هر دو مشکل را ندارد.
    def modal(col: str) -> pd.Series:
        c = fact.groupby(["PersonId", col], observed=True).size()
        return c.reset_index(name="_n").sort_values("_n").groupby("PersonId")[col].last()

    per["main_rest"] = modal("restaurant_canonical")
    per["main_city"] = modal("city")
    per["share_dinner"] = g["Meal"].agg(lambda s: (s == "dinner").mean())
    per["is_tehran"] = (per["main_city"] == "تهران")
    per["raw_rate"] = per["n_norecv"] / per["n_res"]
    per = per.reset_index()

    dim = pd.read_csv(PERSON_DIM_PATH)
    per = per.merge(dim, on="PersonId", how="left", validate="one_to_one")
    return per


# ---------------------------------------------------------------------------
# Q35 — سه تعریف رقیب از «بدترین ۱۰٪»
# ---------------------------------------------------------------------------

def fit_beta_binomial(y: np.ndarray, n: np.ndarray) -> tuple[float, float]:
    """برآورد MLE پارامترهای پیشین بتا-دوجمله‌ای.

    خروجی `alpha+beta` تفسیر مستقیم دارد: «پیشین به‌اندازه‌ی چند رزرو شاهد می‌ارزد» —
    یعنی فردی با رزرو کمتر از این عدد، نرخش عمدتاً از جمعیت قرض گرفته می‌شود نه از خودش.
    """
    def nll(p):
        a, b = np.exp(p)
        return -np.sum(
            special.betaln(y + a, n - y + b) - special.betaln(a, b)
        )

    res = optimize.minimize(nll, x0=np.log([1.0, 10.0]), method="Nelder-Mead",
                            options={"maxiter": 4000, "xatol": 1e-8, "fatol": 1e-8})
    a, b = np.exp(res.x)
    return float(a), float(b)


def add_definitions(per: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, dict]:
    """سه تعریف «بدترین ۱۰٪» را به‌صورت ستون بولین اضافه می‌کند."""
    rng = np.random.default_rng(seed)
    n_top = int(round(DECILE * len(per)))

    # (الف) نرخ خام — با شکن قرعه‌ای، چون گره‌های نرخ ۱.۰ و ۰.۰ عظیم‌اند
    per["_tie"] = rng.random(len(per))
    order_raw = per.sort_values(["raw_rate", "_tie"], ascending=[False, False]).index
    per["worst_raw"] = False
    per.loc[order_raw[:n_top], "worst_raw"] = True

    # (ب) نرخ کوچک‌سازی‌شده‌ی امپریکال-بیز (F49)
    a, b = fit_beta_binomial(per["n_norecv"].to_numpy(float), per["n_res"].to_numpy(float))
    per["eb_rate"] = (per["n_norecv"] + a) / (per["n_res"] + a + b)
    order_eb = per.sort_values(["eb_rate", "_tie"], ascending=[False, False]).index
    per["worst_eb"] = False
    per.loc[order_eb[:n_top], "worst_eb"] = True

    # (ج) تعداد مطلق پرس هدررفته — تعریف کسب‌وکاری
    order_abs = per.sort_values(["n_norecv", "_tie"], ascending=[False, False]).index
    per["worst_abs"] = False
    per.loc[order_abs[:n_top], "worst_abs"] = True

    per.drop(columns=["_tie"], inplace=True)
    return per, {"alpha": a, "beta": b, "prior_strength": a + b, "n_top": n_top}


def jaccard(s1: pd.Series, s2: pd.Series) -> float:
    inter = int((s1 & s2).sum())
    union = int((s1 | s2).sum())
    return inter / union if union else float("nan")


def q35(per: pd.DataFrame, meta: dict) -> pd.DataFrame:
    header("Q35 — «بدترین ۱۰٪» یعنی چه؟ سه تعریف رقیب")
    kv("تعداد افراد", f"{len(per):,}")
    kv("اندازه‌ی دهک", f"{meta['n_top']:,}")
    kv("پیشین بتا-دوجمله‌ای α", f"{meta['alpha']:.4f}")
    kv("پیشین بتا-دوجمله‌ای β", f"{meta['beta']:.4f}")
    kv("قدرت پیشین (α+β) ≡ «معادل چند رزرو»", f"{meta['prior_strength']:.2f}")
    kv("میانگین پیشین α/(α+β)", f"{meta['alpha'] / meta['prior_strength']:.5f}")

    total_waste = per["n_norecv"].sum()
    rows = []
    for key, label in [("worst_raw", "الف) نرخ خام"),
                       ("worst_eb", "ب) نرخ کوچک‌سازی‌شده (EB)"),
                       ("worst_abs", "ج) تعداد مطلق هدررفت")]:
        sub = per[per[key]]
        rows.append({
            "تعریف": label,
            "میانه‌ی رزرو": float(sub["n_res"].median()),
            "میانگین رزرو": float(sub["n_res"].mean()),
            "٪ افراد با ≤۵ رزرو": 100 * float((sub["n_res"] <= 5).mean()),
            "میانگین نرخ خام": float(sub["raw_rate"].mean()),
            "کل هدررفت": int(sub["n_norecv"].sum()),
            "٪ از کل هدررفت": 100 * float(sub["n_norecv"].sum() / total_waste),
        })
    tbl = pd.DataFrame(rows)
    print("\n" + tbl.to_string(index=False))

    print("\n— همپوشانی زوجی (ژاکار) —")
    kv("خام ∩ EB", f"{jaccard(per['worst_raw'], per['worst_eb']):.4f}")
    kv("خام ∩ مطلق", f"{jaccard(per['worst_raw'], per['worst_abs']):.4f}")
    kv("EB ∩ مطلق", f"{jaccard(per['worst_eb'], per['worst_abs']):.4f}")

    kv("کل هدررفت (پرس)", f"{int(total_waste):,}")
    kv("میانه‌ی رزرو کل جمعیت", f"{per['n_res'].median():.0f}")
    return tbl


# ---------------------------------------------------------------------------
# Q36 — تمرکز هدررفت
# ---------------------------------------------------------------------------

def q36(per: pd.DataFrame) -> pd.DataFrame:
    header("Q36 — تمرکز هدررفت بین افراد (لورنتس/جینی/پارتو)")
    pop, cum, gini = lorenz_curve(per["n_norecv"])
    kv("ضریب جینی (تعداد عدم‌دریافت)", f"{gini:.4f}")
    pop_r, cum_r, gini_r = lorenz_curve(per["n_res"])
    kv("ضریب جینی (تعداد رزرو) — مبنای مقایسه", f"{gini_r:.4f}")

    par = pareto_share(per["n_norecv"], [0.01, 0.05, 0.10, 0.20, 0.50])
    print("\n" + par.to_string(index=False))

    kv("٪ افراد با صفر هدررفت", pct(float((per["n_norecv"] == 0).mean())))
    return par, (pop, cum, gini), (pop_r, cum_r, gini_r)


# ---------------------------------------------------------------------------
# Q37 — پروفایل تک‌متغیره‌ی دهک بدتر
# ---------------------------------------------------------------------------

DEMOS = [
    ("is_dorm_resident", "ساکن خوابگاه"),
    ("Gender", "جنسیت"),
    ("DegreeName", "مقطع"),
    ("edu_group", "دوره"),
    ("CollegeName", "دانشکده"),
    ("main_city", "شهر"),
]


def add_edu_group(per: pd.DataFrame) -> pd.DataFrame:
    """۱۱ مقدار `EducationSession` را به ۴ گروه معنادار جمع می‌کند (بقیه ناچیزند)."""
    def m(s: str) -> str:
        if s.startswith("روزانه - شهریه"):
            return "شهریه‌پرداز"
        if s.startswith("روزانه"):
            return "روزانه"
        if "نوبت دوم" in s or "شبانه" in s:
            return "شبانه/نوبت دوم"
        return "سایر"
    per["edu_group"] = per["EducationSession"].map(m)
    return per


def enrichment_table(per: pd.DataFrame, flag: str, col: str, min_n: int = 60) -> pd.DataFrame:
    """نسبت غنی‌شدگی هر دسته در دهک بدتر، با فاصله‌ی اطمینان ویلسون.

    غنی‌شدگی = P(عضو دهک | دسته) ÷ ۰.۱۰. مقدار ۱ یعنی دسته دقیقاً سهم منصفانه‌اش را دارد.
    """
    base = per[flag].mean()
    rows = []
    for cat, sub in per.groupby(col, observed=True):
        n = len(sub)
        if n < min_n:
            continue
        k = int(sub[flag].sum())
        p = k / n
        lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
        rows.append({
            "دسته": str(cat), "n": n, "عضو دهک": k,
            "نرخ عضویت": p, "غنی‌شدگی": p / base,
            "CI پایین": lo / base, "CI بالا": hi / base,
            "نرخ خام میانه": float(sub["raw_rate"].median()),
            "نرخ EB میانگین": float(sub["eb_rate"].mean()),
        })
    return pd.DataFrame(rows).sort_values("غنی‌شدگی", ascending=False)


def q37(per: pd.DataFrame, flag: str = "worst_eb") -> dict:
    header(f"Q37 — پروفایل تک‌متغیره‌ی دهک بدتر (تعریف: {flag})")
    out = {}
    for col, label in DEMOS:
        t = enrichment_table(per, flag, col)
        out[col] = t
        print(f"\n— {label} ({col}) —")
        show = t.head(12) if len(t) > 12 else t
        print(show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        if len(t) > 12:
            print("  … (فقط ۱۲ ردیف بالا)")
    return out


# ---------------------------------------------------------------------------
# Q38 / Q39 — چندمتغیره و اثر مکان
# ---------------------------------------------------------------------------

def design_matrix(per: pd.DataFrame, with_rest: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    d = per[per["n_res"] >= MIN_RES_PROFILE].copy()
    parts = [
        pd.get_dummies(d["Gender"], prefix="g", drop_first=True),
        pd.get_dummies(d["edu_group"], prefix="edu", drop_first=True),
        pd.get_dummies(d["DegreeName"], prefix="deg", drop_first=True),
        pd.DataFrame({"is_dorm": d["is_dorm_resident"].astype(float)}, index=d.index),
        pd.DataFrame({"is_tehran": d["is_tehran"].astype(float)}, index=d.index),
        pd.get_dummies(d["CollegeName"], prefix="col", drop_first=True),
    ]
    if with_rest:
        parts.append(pd.get_dummies(d["main_rest"], prefix="r", drop_first=True))
    X = pd.concat(parts, axis=1).astype(float)
    X = X.loc[:, X.sum() >= 30]                       # دسته‌های خیلی نادر حذف
    X = sm.add_constant(X, has_constant="add")
    return X, d


def fit_logit(X: pd.DataFrame, y: pd.Series, label: str):
    model = sm.Logit(y.astype(float), X)
    res = model.fit(disp=0, method="lbfgs", maxiter=400)
    kv(f"{label} — pseudo-R² (McFadden)", f"{res.prsquared:.4f}")
    kv(f"{label} — n", f"{int(res.nobs):,}")
    return res


def q38_q39(per: pd.DataFrame, flag: str = "worst_eb") -> dict:
    header("Q38 — مدل چندمتغیره‌ی عضویت در دهک بدتر")
    X, d = design_matrix(per, with_rest=False)
    y = d[flag]
    res = fit_logit(X, y, "بدون اثر سلف")

    coefs = pd.DataFrame({
        "ضریب": res.params, "خطای معیار": res.bse, "p": res.pvalues,
        "نسبت شانس": np.exp(res.params),
    }).drop(index="const")
    key = coefs.loc[[i for i in coefs.index
                     if i.startswith(("g_", "edu_", "deg_", "is_dorm", "is_tehran"))]]
    print("\n— ضرایب کلیدی (به‌جز ۳۴ دامی دانشکده) —")
    print(key.sort_values("ضریب", ascending=False).to_string(float_format=lambda v: f"{v:.4f}"))

    header("Q39 — همان مدل + اثر ثابت سلف (اثر مکان در برابر اثر فرد)")
    X2, d2 = design_matrix(per, with_rest=True)
    res2 = fit_logit(X2, d2[flag], "با اثر سلف")
    coefs2 = pd.DataFrame({"ضریب": res2.params, "p": res2.pvalues}).drop(index="const")
    key2 = coefs2.loc[[i for i in coefs2.index
                       if i.startswith(("g_", "edu_", "deg_", "is_dorm", "is_tehran"))]]
    cmp = key[["ضریب", "p"]].join(key2, rsuffix="_باسلف", how="outer")
    cmp["تغییر ٪"] = 100 * (cmp["ضریب_باسلف"] - cmp["ضریب"]) / cmp["ضریب"].abs()
    print("\n— مقایسه‌ی ضرایب پیش و پس از کنترل سلف —")
    print(cmp.to_string(float_format=lambda v: f"{v:.4f}"))
    kv("ΔpseudoR² از افزودن سلف", f"{res2.prsquared - res.prsquared:+.4f}")
    return {"no_rest": res, "with_rest": res2, "cmp": cmp}


# ---------------------------------------------------------------------------
# شکل‌ها
# ---------------------------------------------------------------------------

def figures(per: pd.DataFrame, lor, lor_res, enr: dict) -> None:
    viz_setup()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # شکل ۱ — سه تعریف: توزیع تعداد رزرو + سهم از هدررفت
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    bins = np.logspace(0, np.log10(per["n_res"].max()), 40)
    # رنگ‌ها عمداً با پنل راست یکی است — سه تعریف باید در هر دو پنل یک رنگ داشته باشند
    for key, lab, c in [("worst_raw", "نرخ خام", "#c44"),
                        ("worst_eb", "نرخ کوچک‌سازی‌شده", "#4a7"),
                        ("worst_abs", "تعداد مطلق", "#47c")]:
        ax[0].hist(per.loc[per[key], "n_res"], bins=bins, histtype="step", lw=2,
                   color=c, label=fa(lab))
    ax[0].set_xscale("log")
    ax[0].set_xlabel(fa("تعداد رزرو فرد (مقیاس لگاریتمی)"))
    ax[0].set_ylabel(fa("فراوانی"))
    ax[0].set_title(fa("دهک بدتر با سه تعریف: چه کسانی وارد می‌شوند؟"))
    ax[0].legend()

    shares = [100 * per.loc[per[k], "n_norecv"].sum() / per["n_norecv"].sum()
              for k in ["worst_raw", "worst_eb", "worst_abs"]]
    ax[1].bar([fa("نرخ خام"), fa("کوچک‌سازی‌شده"), fa("تعداد مطلق")], shares,
              color=["#c44", "#4a7", "#47c"])
    ax[1].axhline(10, ls="--", c="k", lw=1)
    ax[1].set_ylabel(fa("٪ از کل پرس هدررفته"))
    ax[1].set_title(fa("سهم دهک از هدررفت واقعی (خط‌چین: سهم منصفانه ۱۰٪)"))
    for i, v in enumerate(shares):
        ax[1].text(i, v + 1, f"{v:.1f}%", ha="center")
    fig.tight_layout()
    save_fig(fig, "R5_q35_definitions", FIG_DIR)
    plt.close(fig)

    # شکل ۲ — لورنتس
    pop, cum, gini = lor
    pop_r, cum_r, gini_r = lor_res
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot(pop, cum, lw=2.2, label=fa(f"عدم‌دریافت (جینی {gini:.3f})"))
    ax.plot(pop_r, cum_r, lw=1.6, ls="-.", label=fa(f"رزرو (جینی {gini_r:.3f})"))
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel(fa("کسر تجمعی افراد (از کم‌هدررفت به پرهدررفت)"))
    ax.set_ylabel(fa("کسر تجمعی مقدار"))
    ax.set_title(fa("تمرکز هدررفت بین افراد"))
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "R5_q36_lorenz", FIG_DIR)
    plt.close(fig)

    # شکل ۳ — غنی‌شدگی جمعیتی
    keys = ["is_dorm_resident", "Gender", "edu_group", "DegreeName"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    for a, k in zip(axes, keys):
        t = enr[k].sort_values("غنی‌شدگی")
        y = np.arange(len(t))
        a.errorbar(t["غنی‌شدگی"], y,
                   xerr=[t["غنی‌شدگی"] - t["CI پایین"], t["CI بالا"] - t["غنی‌شدگی"]],
                   fmt="o", capsize=3)
        a.axvline(1, ls="--", c="k", lw=1)
        a.set_yticks(y)
        a.set_yticklabels([fa(s) for s in t["دسته"]], fontsize=8)
        a.set_title(fa(dict(DEMOS)[k]))
        a.set_xlabel(fa("غنی‌شدگی"))
    fig.suptitle(fa("غنی‌شدگی دسته‌های جمعیتی در بدترین دهک (۱ = سهم منصفانه)"))
    fig.tight_layout()
    save_fig(fig, "R5_q37_enrichment", FIG_DIR)
    plt.close(fig)


def main() -> None:
    setup()
    per = load_person_level()
    per = add_edu_group(per)
    per, meta = add_definitions(per)

    q35(per, meta)
    par, lor, lor_res = q36(per)
    enr = q37(per)
    q38_q39(per)

    figures(per, lor, lor_res, enr)
    per.to_parquet("data/interim/r5_person_level.parquet", index=False)
    print("\n[ok] سطح فرد ذخیره شد: data/interim/r5_person_level.parquet")


if __name__ == "__main__":
    main()
