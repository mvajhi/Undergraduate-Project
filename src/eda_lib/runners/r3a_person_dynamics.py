"""دور ۳ (الف) — دینامیک رفتار فردی: عدم‌دریافت «صفت» است یا «حالت»؟

**چرا این سؤال مهم است.** دور ۱ نشان داد نرخ شخصی بین دو نیمه‌ی بازه همبستگی ۰.۶۴
دارد (F46) — یعنی تفاوت *پایدار* بین افراد وجود دارد. ولی این چیزی درباره‌ی ساختار
*درون* هر فرد نمی‌گوید. دو دنیای کاملاً متفاوت با همان r=۰.۶۴ سازگارند:

- **صفت (trait):** هر فرد یک نرخ ثابت دارد و رخدادها مستقل و پراکنده‌اند
  → فیچر درست: میانگین بلندمدت شخص.
- **حالت (state):** رخدادها خوشه‌ای‌اند (یک هفته‌ی بد، بعد عادی)
  → فیچر درست: میانگین وزن‌دار به تازگی، یا وضعیت رزرو قبلی.

این اسکریپت با چهار روش مستقل تفکیک می‌کند: ماتریس گذار، آزمون runs، خودهمبستگی
درون‌فردی، و مقایسه‌ی قدرت پیش‌بینی «تازگی» در برابر «بلندمدت».

**دامنه:** فقط تهران (طبق نظر ذی‌نفع، مقایسه‌ی بین‌شهری دیگر موضوع نیست؛ ضمناً حذف
شهر یک مخدوش‌کننده‌ی بزرگ را از همه‌ی تحلیل‌های زیر برمی‌دارد).

اجرا: `python -m src.eda_lib.runners.r3a_person_dynamics`
"""

import numpy as np
import pandas as pd
from scipy import stats

from src.eda_lib.runners._common import PERSON_DIM_PATH, header, kv, pct, setup
from src.eda_lib.runners.s13_individual import load_fact

MIN_RES = 20  # حداقل رزرو برای اینکه آماره‌ی درون‌فردی معنا داشته باشد


def load_tehran() -> tuple[pd.DataFrame, pd.DataFrame]:
    fact = load_fact()
    fact = fact[fact["is_tehran"]].copy()
    dim = pd.read_csv(PERSON_DIM_PATH)
    header(f"دامنه‌ی دور ۳: فقط تهران — {len(fact):,} رزرو · {fact['PersonId'].nunique():,} فرد")
    return fact, dim


def _order(fact: pd.DataFrame) -> pd.DataFrame:
    """توالی زمانی رزروهای هر فرد (ناهار قبل از شام در یک روز)."""
    f = fact.copy()
    f["meal_ord"] = (f["Meal"].astype(str) == "dinner").astype(int)
    f = f.sort_values(["PersonId", "date_gregorian", "meal_ord"], kind="stable")
    f["seq"] = f.groupby("PersonId").cumcount()
    return f


def q12_trait_or_state(fact: pd.DataFrame) -> pd.DataFrame:
    header("Q12 — عدم‌دریافت «صفت» است یا «حالت»؟")
    f = _order(fact)
    f["prev"] = f.groupby("PersonId")["dont_receive"].shift(1)
    f["prev_date"] = f.groupby("PersonId")["date_gregorian"].shift(1)
    f["gap_days"] = (f["date_gregorian"] - f["prev_date"]).dt.days
    sub = f.dropna(subset=["prev"])

    print("### ماتریس گذار (احتمال شرطی عدم‌دریافت بر اساس رزرو قبلی همان فرد)")
    p_base = f["dont_receive"].mean()
    p_after_show = sub.loc[~sub["prev"].astype(bool), "dont_receive"].mean()
    p_after_no = sub.loc[sub["prev"].astype(bool), "dont_receive"].mean()
    kv("P(عدم‌دریافت) پایه", f"{p_base:.4f}")
    kv("P(عدم‌دریافت | رزرو قبلی دریافت شد)", f"{p_after_show:.4f}")
    kv("P(عدم‌دریافت | رزرو قبلی دریافت نشد)", f"{p_after_no:.4f}")
    kv("نسبت شانس (odds ratio)", f"{(p_after_no / (1 - p_after_no)) / (p_after_show / (1 - p_after_show)):.2f}")
    kv("نسبت خطر (risk ratio)", f"{p_after_no / p_after_show:.2f}")

    print("\n### آیا این فقط بازتاب تفاوت *بین* افراد است؟")
    print("کنترل: مقایسه‌ی همان آماره پس از حذف اثر ثابت فرد (باقیمانده‌ی درون‌فردی)")
    pr = f.groupby("PersonId")["dont_receive"].transform("mean")
    f["resid"] = f["dont_receive"].astype(float) - pr
    f["prev_resid"] = f.groupby("PersonId")["resid"].shift(1)
    s2 = f.dropna(subset=["prev_resid"])
    r_within = stats.pearsonr(s2["prev_resid"], s2["resid"])
    kv("همبستگی درون‌فردی lag-1 (اثر ثابت فرد حذف‌شده)", f"{r_within[0]:+.4f} (p={r_within[1]:.3g})")
    print("→ اگر نزدیک صفر باشد: «صفت» غالب است. اگر مثبت و معنادار: «حالت»/خوشه‌ای هم هست.")

    print("\n### اثر فاصله‌ی زمانی: آیا وابستگی با فاصله محو می‌شود؟")
    sub2 = f.dropna(subset=["prev_resid", "gap_days"])
    sub2 = sub2[sub2["gap_days"] <= 30]
    b = pd.cut(sub2["gap_days"], [0, 1, 2, 3, 7, 14, 30],
               labels=["همان روز/۱", "۲", "۳", "۴-۷", "۸-۱۴", "۱۵-۳۰"], include_lowest=True)
    rows = []
    for lv, g in sub2.groupby(b, observed=True):
        if len(g) < 200:
            continue
        r = stats.pearsonr(g["prev_resid"], g["resid"])
        rows.append({"فاصله (روز)": lv, "n": len(g), "همبستگی درون‌فردی": r[0], "p": r[1]})
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    print("\n### آزمون runs (Wald-Wolfowitz) — آیا رخدادها خوشه‌ای‌اند؟")
    per = f.groupby("PersonId")["dont_receive"].agg(["size", "sum"])
    keep = per[(per["size"] >= MIN_RES) & (per["sum"] >= 3) & (per["sum"] <= per["size"] - 3)].index
    g = f[f["PersonId"].isin(keep)]
    runs = g.groupby("PersonId")["dont_receive"].apply(lambda s: int((s != s.shift()).sum()))
    n1 = g.groupby("PersonId")["dont_receive"].sum()
    n0 = g.groupby("PersonId")["dont_receive"].size() - n1
    exp_runs = 2 * n1 * n0 / (n1 + n0) + 1
    var_runs = (2 * n1 * n0 * (2 * n1 * n0 - n1 - n0)) / ((n1 + n0) ** 2 * (n1 + n0 - 1))
    z = (runs - exp_runs) / np.sqrt(var_runs)
    kv("افراد واجد شرایط", len(keep))
    kv("میانگین z آماره‌ی runs", f"{z.mean():+.4f}")
    kv("سهم افراد با z<−1.96 (خوشه‌ای)", pct((z < -1.96).mean()))
    kv("سهم افراد با z>+1.96 (متناوب)", pct((z > 1.96).mean()))
    t = stats.ttest_1samp(z.dropna(), 0)
    kv("آزمون t یک‌نمونه‌ای روی z", f"t={t.statistic:.2f} p={t.pvalue:.3g}")
    print("→ z منفی یعنی runs کمتر از انتظار ⇒ رخدادها کنار هم جمع شده‌اند (خوشه‌ای).")
    return f


def q19_learning_curve(f: pd.DataFrame) -> None:
    header("Q19 — منحنی یادگیری: آیا رفتار فرد در طول سابقه‌اش تغییر می‌کند؟")
    per = f.groupby("PersonId")["dont_receive"].size()
    keep = per[per >= 60].index
    g = f[f["PersonId"].isin(keep)].copy()
    kv("افراد با ≥۶۰ رزرو", len(keep))
    print("\nنرخ عدم‌دریافت بر حسب شماره‌ی ترتیبی رزرو (اثر ثابت فرد حذف‌شده):")
    g["resid"] = g["dont_receive"].astype(float) - g.groupby("PersonId")["dont_receive"].transform("mean")
    b = pd.cut(g["seq"], [0, 5, 10, 20, 40, 60, 100, 150, 400],
               labels=["۱-۵", "۶-۱۰", "۱۱-۲۰", "۲۱-۴۰", "۴۱-۶۰", "۶۱-۱۰۰", "۱۰۱-۱۵۰", "۱۵۰+"],
               include_lowest=True)
    t = g.groupby(b, observed=True).agg(n=("resid", "size"), نرخ_خام=("dont_receive", "mean"),
                                        باقیمانده=("resid", "mean"))
    t["se"] = g.groupby(b, observed=True)["resid"].sem()
    print(t.round(4).to_string())
    print("→ باقیمانده‌ی مثبت در ابتدا یعنی افراد در شروع بدتر از میانگین خودشان‌اند (یادگیری/عادت‌سازی).")


def q20_recency_vs_longrun(f: pd.DataFrame) -> None:
    header("Q20 — کدام فیچر تاریخی قوی‌تر است: میانگین بلندمدت یا وزن‌دار به تازگی؟")
    print("روش leakage-safe: برای هر رزرو، فقط از رزروهای *قبلی* همان فرد استفاده می‌شود.\n")
    g = f.sort_values(["PersonId", "seq"]).copy()
    y = g["dont_receive"].astype(float)

    # میانگین انبساطی (بلندمدت) تا پیش از رزرو جاری
    csum = g.groupby("PersonId")["dont_receive"].cumsum() - g["dont_receive"]
    cnt = g.groupby("PersonId").cumcount()
    g["exp_mean"] = np.where(cnt > 0, csum / cnt.replace(0, np.nan), np.nan)

    # میانگین متحرک k رزرو آخر
    for k in [1, 3, 5, 10]:
        g[f"roll_{k}"] = (g.groupby("PersonId")["dont_receive"]
                            .transform(lambda s: s.shift(1).rolling(k, min_periods=1).mean()))

    # میانگین با میرایی نمایی
    for half in [5, 20]:
        alpha = 1 - 0.5 ** (1 / half)
        g[f"ewm_{half}"] = (g.groupby("PersonId")["dont_receive"]
                              .transform(lambda s: s.shift(1).ewm(alpha=alpha, adjust=False).mean()))

    valid = g[cnt >= 10].copy()
    kv("رزروهای واجد ارزیابی (پس از ۱۰ رزرو اول هر فرد)", f"{len(valid):,}")
    print("\nقدرت پیش‌بینی هر فیچر (AUC تک‌متغیره روی همان رزروها):")
    from sklearn.metrics import roc_auc_score
    rows = []
    for c in ["exp_mean", "roll_1", "roll_3", "roll_5", "roll_10", "ewm_5", "ewm_20"]:
        m = valid[c].notna()
        auc = roc_auc_score(valid.loc[m, "dont_receive"], valid.loc[m, c])
        r = stats.spearmanr(valid.loc[m, c], valid.loc[m, "dont_receive"])[0]
        rows.append({"فیچر": c, "AUC": auc, "Spearman": r, "n": int(m.sum())})
    res = pd.DataFrame(rows).sort_values("AUC", ascending=False)
    print(res.round(4).to_string(index=False))

    print("\nترکیب بلندمدت + تازگی (رگرسیون لجستیک دو-فیچره):")
    from sklearn.linear_model import LogisticRegression
    m = valid[["exp_mean", "roll_3"]].notna().all(axis=1)
    X = valid.loc[m, ["exp_mean", "roll_3"]].values
    yy = valid.loc[m, "dont_receive"].values
    lr = LogisticRegression(max_iter=400).fit(X, yy)
    auc_both = roc_auc_score(yy, lr.predict_proba(X)[:, 1])
    auc_long = roc_auc_score(yy, valid.loc[m, "exp_mean"])
    kv("AUC فقط بلندمدت", f"{auc_long:.4f}")
    kv("AUC بلندمدت + تازگی", f"{auc_both:.4f}")
    kv("ΔAUC", f"{auc_both - auc_long:+.4f}")
    kv("ضرایب", f"exp_mean={lr.coef_[0][0]:+.3f} · roll_3={lr.coef_[0][1]:+.3f}")
    print("→ اگر ضریب roll_3 معنادار و ΔAUC مثبت باشد، «حالت» اطلاعات مستقل از «صفت» دارد.")


def q13_person_vs_demographics(f: pd.DataFrame, dim: pd.DataFrame) -> None:
    header("Q13 — تاریخچه‌ی شخصی در برابر فیچرهای جمعیتی: کدام بیشتر توضیح می‌دهد؟")
    per = f.groupby("PersonId").agg(n=("dont_receive", "size"), rate=("dont_receive", "mean"))
    per = per[per["n"] >= MIN_RES].join(dim.set_index("PersonId"))
    per = per.dropna(subset=["CollegeName", "is_dorm_resident"])
    kv("افراد در تحلیل", len(per))

    import statsmodels.formula.api as smf
    per = per.copy()
    per["is_dorm"] = per["is_dorm_resident"].astype(int)
    per["is_female"] = (per["Gender"].astype(str).str.strip() == "زن").astype(int)
    per["is_grad"] = per["DegreeName"].astype(str).str.contains("ارشد|دکتری|PhD", na=False).astype(int)

    models = {
        "خوابگاه": "rate ~ is_dorm",
        "+ جنسیت + مقطع": "rate ~ is_dorm + is_female + is_grad",
        "+ نوع دوره": "rate ~ is_dorm + is_female + is_grad + C(EducationSession)",
        "+ دانشکده (۲۶ سطح)": "rate ~ is_dorm + is_female + is_grad + C(EducationSession) + C(CollegeName)",
        "+ رشته (کاردینالیتی بالا)": "rate ~ is_dorm + is_female + is_grad + C(EducationSession) + C(CollegeName) + C(FieldName)",
    }
    for name, fo in models.items():
        try:
            m = smf.ols(fo, data=per).fit()
            print(f"  {name:<28s} R²={m.rsquared:.4f}  R²adj={m.rsquared_adj:.4f}  (k={int(m.df_model)})")
        except Exception as e:
            print(f"  {name:<28s} خطا: {e}")

    print("\n**مقایسه‌ی حیاتی:** کل توان توضیحی جمعیت‌شناسی در برابر پایداری خودِ فرد")
    print("(همبستگی نیمه‌اول/نیمه‌دوم یعنی R² قابل‌دستیابی از تاریخچه‌ی شخصی)")
    fs = f.sort_values("date_gregorian")
    mid = fs["date_gregorian"].quantile(0.5)
    h1 = fs[fs.date_gregorian < mid].groupby("PersonId")["dont_receive"].agg(["mean", "size"])
    h2 = fs[fs.date_gregorian >= mid].groupby("PersonId")["dont_receive"].agg(["mean", "size"])
    b = h1.join(h2, lsuffix="_1", rsuffix="_2", how="inner")
    b = b[(b["size_1"] >= MIN_RES) & (b["size_2"] >= MIN_RES)]
    r = stats.pearsonr(b["mean_1"], b["mean_2"])[0]
    kv("  همبستگی تاریخچه‌ی شخصی (نیمه۱ → نیمه۲)", f"{r:.4f}")
    kv("  R² معادل تاریخچه‌ی شخصی", f"{r ** 2:.4f}")
    print(f"  → تاریخچه‌ی شخصی حدود {r ** 2 / max(1e-9, 0.0580):.0f} برابر بیشتر از کل جمعیت‌شناسی توضیح می‌دهد.")


def main() -> None:
    setup()
    fact, dim = load_tehran()
    f = q12_trait_or_state(fact)
    q19_learning_curve(f)
    q20_recency_vs_longrun(f)
    q13_person_vs_demographics(fact, dim)


if __name__ == "__main__":
    main()
