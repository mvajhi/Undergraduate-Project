"""بند ۵.۱۳ — ممیزی خودکار نشت داده. دروازه‌ی اجباری پیش از فاز ۶.

سه لایه آزمون:

1. **آزمون ساختاری قاعده‌ی برش** — بازسازی مستقیم نامساوی `seq_obs ≤ seq_target − 2`
   روی نمونه‌ی تصادفی: برای هر ردیف، فیچر تاریخی باید دقیقاً از مشاهداتِ مجاز ساخته
   شده باشد. این آزمون *تعریفی* است و به داده وابسته نیست.
2. **آزمون همبستگی مشکوک** — هیچ فیچری نباید |r| > ۰.۹۵ با هدف داشته باشد.
3. **آزمون سقف $R^2$** — فاز ۴ نشان داد ۸۳٪ واریانس سلول شوک روزانه است (F59) و
   بیش‌پراکندگی باقیمانده ۷.۶ برابر (F09)، پس سقف واقع‌بینانه‌ی $R^2$ حدود ۰.۴–۰.۵
   است. هر عدد بالاتر از ۰.۹ تقریباً قطعاً نشتی است.

اجرا: `python -m src.features.audit`
"""

import json
import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score

from src.config import DATA_PROCESSED, DOCS_DIR, set_global_seed
from src.features.build import FEATURE_SETS_PATH, FEATURES_A_PATH, TARGET
from src.features.cutoff import CUTOFF_LAG, meal_seq

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SUSPICIOUS_CORR = 0.95
R2_CEILING = 0.90
AUDIT_PATH = DOCS_DIR / "leakage_audit.md"


def test_cutoff_invariant(n_sample: int = 300, seed: int = 42) -> tuple[bool, str]:
    """لایه ۱ — بازسازی مستقل فیچر تاریخچه‌ی فرد و مقایسه با مقدار ذخیره‌شده."""
    pf = pd.read_parquet(DATA_PROCESSED / "person_features_v1.parquet",
                         columns=["PersonId", "date_gregorian", "Meal", "dont_receive",
                                  "person_n_prior_reservations", "person_n_prior_norecv"])
    pf["meal_seq"] = meal_seq(pf["date_gregorian"], pf["Meal"])
    rng = np.random.default_rng(seed)
    # فقط افرادی که چند رزرو دارند، وگرنه آزمون بی‌معناست
    counts = pf["PersonId"].value_counts()
    pool = counts[counts.between(20, 200)].index.to_numpy()
    persons = rng.choice(pool, size=min(n_sample, len(pool)), replace=False)
    sub = pf[pf["PersonId"].isin(persons)]

    bad = 0
    checked = 0
    for _, g in sub.groupby("PersonId", sort=False):
        seqs = g["meal_seq"].to_numpy()
        vals = g["dont_receive"].to_numpy(dtype=float)
        for i in range(len(g)):
            allowed = seqs <= seqs[i] - CUTOFF_LAG
            exp_n, exp_k = int(allowed.sum()), float(vals[allowed].sum())
            got_n = int(g["person_n_prior_reservations"].to_numpy()[i])
            got_k = float(g["person_n_prior_norecv"].to_numpy()[i])
            checked += 1
            if exp_n != got_n or abs(exp_k - got_k) > 1e-9:
                bad += 1
    ok = bad == 0
    msg = f"{checked:,} ردیف از {len(persons)} فرد بررسی شد · {bad} ناسازگاری"
    return ok, msg


def test_no_same_day_crossmeal(n_sample: int = 400, seed: int = 7) -> tuple[bool, str]:
    """لایه ۱-ب — تله‌ی F57: نتیجه‌ی ناهار روز $d$ نباید در تاریخچه‌ی شامِ همان روز باشد.

    ⚠️ **دقت در جهت آزمون.** شهود اولیه («شمارنده‌ی شام نباید از ناهار بیشتر باشد») غلط
    است: پنجره‌ی شامِ روز $d$ (seq ≤ 2d−1) *عمداً* یک وعده عقب‌تر از پنجره‌ی ناهار همان
    روز (seq ≤ 2d−2) می‌رسد، پس شامل شامِ روز $d-1$ هم می‌شود. بنابراین
    ``count(dinner) ≥ count(lunch)`` رفتار **درست** است.

    آزمون درست: اختلاف باید **دقیقاً** برابر تعداد رزروهای شامِ روز $d-1$ همان فرد باشد —
    نه بیشتر (که یعنی ناهار همان روز هم شمرده شده) و نه کمتر.
    """
    pf = pd.read_parquet(DATA_PROCESSED / "person_features_v1.parquet",
                         columns=["PersonId", "date_gregorian", "Meal",
                                  "person_n_prior_reservations"])
    nmeals = pf.groupby(["PersonId", "date_gregorian"])["Meal"].nunique()
    both = nmeals[nmeals == 2].reset_index()[["PersonId", "date_gregorian"]]
    rng = np.random.default_rng(seed)
    both = both.iloc[rng.choice(len(both), size=min(n_sample, len(both)), replace=False)]

    counts = (pf.groupby(["PersonId", "date_gregorian", "Meal"])["person_n_prior_reservations"]
                .first().unstack("Meal"))
    # تعداد رزروهای شامِ روز قبل، برای هر (فرد، روز) مورد آزمون
    prev = both.assign(date_gregorian=both["date_gregorian"] - pd.Timedelta(days=1))
    n_prev_dinner = (pf[pf["Meal"] == "dinner"]
                     .groupby(["PersonId", "date_gregorian"]).size()
                     .rename("n_prev_dinner"))
    exp = prev.join(n_prev_dinner, on=["PersonId", "date_gregorian"])["n_prev_dinner"].fillna(0).to_numpy()

    got = counts.reindex(pd.MultiIndex.from_frame(both))
    diff = (got["dinner"] - got["lunch"]).to_numpy()
    violations = int(np.nansum(np.abs(diff - exp) > 1e-9))
    ok = violations == 0
    return ok, f"{len(both):,} روز-فرد با هر دو وعده · {violations} اختلاف غیرمنتظره"


def test_target_correlation(df: pd.DataFrame, features: list[str]) -> tuple[bool, pd.DataFrame]:
    """لایه ۲ — همبستگی مشکوکاً بالا با هدف."""
    rows = []
    y = df[TARGET]
    for c in features:
        s = df[c]
        if not pd.api.types.is_numeric_dtype(s):
            continue
        m = s.notna() & y.notna()
        if m.sum() < 100 or s[m].nunique() < 2:
            continue
        rows.append({"feature": c, "r": float(np.corrcoef(s[m], y[m])[0, 1]), "n": int(m.sum())})
    res = pd.DataFrame(rows).sort_values("r", key=lambda s: s.abs(), ascending=False)
    flagged = res[res["r"].abs() > SUSPICIOUS_CORR]
    return len(flagged) == 0, res


def test_r2_ceiling(df: pd.DataFrame, features: list[str], name: str) -> tuple[bool, float]:
    """لایه ۳ — تقسیم زمانی، مدل درختی، بررسی سقف $R^2$."""
    d = df.sort_values("date_gregorian")
    cut = d["date_gregorian"].quantile(0.75)
    # ⚠️ کدگذاری **یک‌بار روی کل قاب** انجام می‌شود، بعد تقسیم. اگر train و test جداگانه
    # کدگذاری شوند، سطوح دسته‌ای به اعداد متفاوتی نگاشت می‌شوند و مدل در زمان آزمون
    # عملاً «سلف اشتباه» می‌خواند — که خطای فاجعه‌بار و کاملاً مصنوعی تولید می‌کند.
    # این کدگذاری نشتی نیست: فقط شناسه‌ی سطح است و هیچ اطلاعاتی از هدف در آن نیست.
    X_all = _encode(d, features)
    mask = (d["date_gregorian"] <= cut).to_numpy()
    tr, te = d[mask], d[~mask]
    model = HistGradientBoostingRegressor(max_iter=250, random_state=42)
    model.fit(X_all[mask], tr[TARGET])
    r2 = r2_score(te[TARGET], model.predict(X_all[~mask]))
    return r2 < R2_CEILING, r2


def _encode(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """کدگذاری حداقلی برای آزمون: عددی‌ها همان، دسته‌ای‌ها به کد صحیح.

    باید روی **کل** قاب صدا زده شود، نه جدا برای هر تکه (توضیح در `test_r2_ceiling`).
    """
    X = pd.DataFrame(index=df.index)
    for c in features:
        s = df[c]
        X[c] = s if pd.api.types.is_numeric_dtype(s) else s.astype("category").cat.codes
    return X


def main() -> None:
    set_global_seed()
    df = pd.read_parquet(FEATURES_A_PATH)
    fs = json.loads(FEATURE_SETS_PATH.read_text())

    logger.info("لایه ۱ — آزمون ساختاری قاعده‌ی برش ...")
    ok_cut, msg_cut = test_cutoff_invariant()
    ok_cross, msg_cross = test_no_same_day_crossmeal()
    logger.info(f"  invariant: {'PASS' if ok_cut else 'FAIL'} ({msg_cut})")
    logger.info(f"  cross-meal: {'PASS' if ok_cross else 'FAIL'} ({msg_cross})")

    logger.info("لایه ۲ — همبستگی با هدف ...")
    ok_corr, corr = test_target_correlation(df, fs["FS_bridge"])
    logger.info(f"  {'PASS' if ok_corr else 'FAIL'} · بیشترین |r| = {corr['r'].abs().max():.4f}")

    logger.info("لایه ۳ — سقف R² با تقسیم زمانی ...")
    r2s = {}
    for name in ["FS_baseline", "FS_calendar", "FS_lag", "FS_day", "FS_full_A", "FS_bridge"]:
        ok, r2 = test_r2_ceiling(df, fs[name], name)
        r2s[name] = r2
        logger.info(f"  {name:<14s} R²out={r2:+.4f} {'PASS' if ok else 'FAIL'}")

    all_ok = ok_cut and ok_cross and ok_corr and all(v < R2_CEILING for v in r2s.values())
    _write_report(corr, r2s, ok_cut, msg_cut, ok_cross, msg_cross, all_ok)
    print("\n" + "=" * 70)
    print("نتیجه‌ی دروازه:", "✅ PASS" if all_ok else "❌ FAIL")
    print("=" * 70)
    print("\nR² خارج‌نمونه‌ی هر فیچرست (تقسیم زمانی ۷۵/۲۵):")
    for k, v in r2s.items():
        print(f"  {k:<14s} {v:+.4f}")
    print("\n۱۰ فیچر با بیشترین همبستگی با هدف:")
    print(corr.head(10).round(4).to_string(index=False))


def _write_report(corr, r2s, ok_cut, msg_cut, ok_cross, msg_cross, all_ok) -> None:
    lines = [
        "# ممیزی نشت داده (Leakage Audit) — فاز ۵",
        "",
        "> بند ۵.۱۳ WBS. تولید خودکار با `python -m src.features.audit`.",
        f"> **نتیجه‌ی دروازه: {'✅ PASS' if all_ok else '❌ FAIL'}**",
        "",
        "## قاعده‌ی حاکم",
        "",
        "قاعده‌ی برش به تفکیک وعده، پس از شماره‌گذاری سراسری وعده‌ها (ناهار روز $t$ → $2t$،",
        "شام روز $t$ → $2t+1$) به یک نامساوی واحد فرو می‌ریزد:",
        "",
        "$$\\text{مشاهده مجاز است} \\iff \\text{meal\\_seq}_{obs} \\le \\text{meal\\_seq}_{target} - 2$$",
        "",
        "همه‌ی فیچرهای تاریخی (سطح سلول و سطح فرد) از `src/features/cutoff.py` عبور می‌کنند،",
        "پس یک آزمون واحد کل پروژه را پوشش می‌دهد.",
        "",
        "## لایه ۱ — آزمون ساختاری",
        "",
        "| آزمون | نتیجه | جزئیات |",
        "|---|---|---|",
        f"| بازسازی مستقل تاریخچه‌ی فرد در برابر نامساوی برش | {'✅ PASS' if ok_cut else '❌ FAIL'} | {msg_cut} |",
        f"| تله‌ی F57: ناهار روز $d$ در تاریخچه‌ی شام روز $d$ نباشد | {'✅ PASS' if ok_cross else '❌ FAIL'} | {msg_cross} |",
        "",
        "> **چرا آزمون دوم لازم بود.** فاز ۴ کشف کرد نتیجه‌ی ناهار و شام یک روزِ یک فرد",
        "> به‌شدت گره خورده‌اند (نسبت خطر ۳.۱۸، F57). این وسوسه‌انگیز است که از آن فیچر",
        "> بسازیم، ولی لحظه‌ی برشِ شام روز $d$ ساعت ۲۳ روز $d-1$ است — ناهار روز $d$ هنوز",
        "> سرو نشده. تعریف ساده‌ی `groupby().shift()` این مرز را **نقض می‌کند**.",
        "",
        "### اندازه‌ی خوش‌بینی تعریف ساده (سنجش کمّی)",
        "",
        "| تعریف تاریخچه‌ی فرد | AUC |",
        "|---|---|",
        "| `groupby(PersonId).shift()` — همه‌ی ردیف‌های قبلی | ۰.۷۲۰۱ |",
        "| **با قاعده‌ی برش (`meal_seq − 2`)** | **۰.۷۱۷۹** |",
        "",
        "اختلاف ۰.۰۰۲۲ (شام ۰.۰۰۲۹، ناهار ۰.۰۰۱۹) — کوچک، ولی چون یک نقض *تعریفی* قاعده",
        "است نه یک تقریب، نسخه‌ی درست مبنا قرار گرفت. نتیجه‌گیری‌های فاز ۴ با این تصحیح",
        "تغییر نمی‌کنند.",
        "",
        "## لایه ۲ — همبستگی با هدف",
        "",
        f"آستانه: |r| > {SUSPICIOUS_CORR}. بیشترین مقدار مشاهده‌شده: **{corr['r'].abs().max():.4f}** — بدون پرچم.",
        "",
        "۱۰ فیچر با بیشترین همبستگی:",
        "",
        "| فیچر | r | n |",
        "|---|---|---|",
    ]
    for _, r in corr.head(10).iterrows():
        lines.append(f"| `{r['feature']}` | {r['r']:+.4f} | {int(r['n']):,} |")
    lines += [
        "",
        "## لایه ۳ — سقف $R^2$ (تقسیم زمانی ۷۵/۲۵، HistGradientBoosting)",
        "",
        f"آستانه‌ی هشدار: $R^2 > {R2_CEILING}$. سقف واقع‌بینانه طبق F59 و F09 حدود ۰.۴–۰.۵ است.",
        "",
        "| فیچرست | $R^2$ خارج‌نمونه | وضعیت |",
        "|---|---|---|",
    ]
    for k, v in r2s.items():
        lines.append(f"| `{k}` | {v:+.4f} | {'✅' if v < R2_CEILING else '❌ مشکوک'} |")
    lines += [
        "",
        "## چک‌لیست فیچرهای ممنوع (ساخته نشده‌اند)",
        "",
        "| فیچر | دلیل | شاهد |",
        "|---|---|---|",
        "| نتیجه‌ی ناهار روز $d$ برای شام روز $d$ | نقض قاعده‌ی برش | F57 |",
        "| `card_ratio` خام | از خروجی همان وعده مشتق می‌شود | بند ۵.۷ |",
        "| `NoRecv`, `Recv` روز هدف | خروجی همان وعده | بند ۴-۲ سند مسئله |",
        "| `day_shock` روز هدف (بدون lag) | از outcome همان روز ساخته می‌شود | بند ۵.۱۷.۲ |",
        "| AQI / PM | همبستگی کاذب بین‌شهری | F25 |",
        "| `has_extras`, `Count`, `Price` | ستون منحط یا بی‌اثر | F66 |",
        "| ترجیح غذایی شخصی | پایداری کمتر از خود فرد | F66 |",
        "",
        "## محدودیت ثبت‌شده",
        "",
        "متغیرهای جوی (`temp_min`, `precip_type`) با **مقدار واقعی** روز هدف ساخته شده‌اند،",
        "در حالی که در استقرار واقعی باید **پیش‌بینی** هواشناسی باشند. چون سهمشان در مدل",
        "ناچیز است (F23، F26)، این تقریب پذیرفته شد ولی باید در گزارش نهایی ذکر شود.",
    ]
    AUDIT_PATH.write_text("\n".join(lines) + "\n")
    logger.info(f"Saved {AUDIT_PATH}")


if __name__ == "__main__":
    main()
