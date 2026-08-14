"""بند ۵.۱۶ — فیچرهای سطح فرد (مدل B).

اولویت‌ها **تجربی** تعیین شده‌اند، نه شهودی. فاز ۴ (F55, F56) قدرت هر کاندید را
تک‌متغیره سنجید:

| فیچر | AUC | تصمیم |
|---|---|---|
| میانگین انبساطی | ۰.۷۲۰۰ | ⭐ اصلی |
| EWM (نیمه‌عمر ۵) | ۰.۷۱۲۸ | مکمل |
| rolling(۱۰) | ۰.۶۸۴۵ | اختیاری |
| rolling(۳) | ۰.۶۲۶۷ | ❌ ساخته نمی‌شود |
| آخرین نتیجه | ۰.۵۷۱۸ | ❌ ساخته نمی‌شود |

همه‌ی فیچرها از `src.features.cutoff` عبور می‌کنند، یعنی مرزشان `meal_seq − 2` است نه
«ردیف قبلی». این تفاوت برای شام روزهایی که فرد ناهار هم داشته اهمیت دارد (F57).

اجرا: `python -m src.features.person_features` (تولید و ذخیره‌ی جدول فیچر فرد)
"""

import logging

import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED
from src.features.cutoff import CUTOFF_LAG, expanding_stat_at_cutoff, meal_seq, shrunk_rate, usage_rate

logger = logging.getLogger(__name__)

#: پارامترهای پیشین Beta از برازش با تصحیح بیش‌پراکندگی (بند ۴.۸، یافته‌ی F8.3)
BETA_ALPHA, BETA_BETA = 0.9758, 11.1506

#: آستانه‌ی cold-start — بند ۵.۱۶.۴ (بالا برده شد از ۵ به ۱۰: فیچر انبساطی زیر ۱۰ رزرو ناپایدار است)
COLD_START_K = 10

OUTPUT_PATH = DATA_PROCESSED / "person_features_v1.parquet"


def _ewm_at_cutoff(df: pd.DataFrame, half_life: float) -> pd.Series:
    """EWM انبساطی با مرز برش (نه صرفاً shift(1)).

    برای هر فرد، مقادیر بر اساس `meal_seq` مرتب می‌شوند و EWM روی مقادیر *مجاز* محاسبه
    می‌شود: ابتدا EWM معمولی ساخته می‌شود، سپس با `searchsorted` مقدارِ متناظر با
    `seq − CUTOFF_LAG` برداشته می‌شود.
    """
    alpha = 1 - 0.5 ** (1 / half_life)
    out = np.full(len(df), np.nan)
    work = df[["PersonId", "meal_seq", "dont_receive"]].copy()
    work["_row"] = np.arange(len(work))
    for _, g in work.groupby("PersonId", observed=True, sort=False):
        g = g.sort_values("meal_seq", kind="stable")
        seqs = g["meal_seq"].to_numpy()
        vals = g["dont_receive"].to_numpy(dtype=float)
        ew = pd.Series(vals).ewm(alpha=alpha, adjust=False).mean().to_numpy()
        ew_shift = np.concatenate([[np.nan], ew])  # ew_shift[k] = EWM پس از k مشاهده
        k = np.searchsorted(seqs, seqs - CUTOFF_LAG, side="right")
        out[g["_row"].to_numpy()] = ew_shift[k]
    return pd.Series(out, index=df.index)


def build_person_features(fact: pd.DataFrame) -> pd.DataFrame:
    """جدول فیچر سطح فرد، هم‌اندیس با `fact`."""
    f = fact.copy()
    f["meal_seq"] = meal_seq(f["date_gregorian"], f["Meal"])
    f = f.sort_values(["PersonId", "meal_seq"], kind="stable")

    out = pd.DataFrame(index=f.index)

    # --- ۵.۱۶.۱ تاریخچه‌ی رفتاری ---
    hist = expanding_stat_at_cutoff(f, ["PersonId"], "dont_receive", out_prefix="p_hist")
    # شمارنده‌ی خام نگه داشته می‌شود (برای آستانه‌ی cold-start و گزارش)، ولی فیچرِ
    # ورودی مدل نسخه‌ی **اشباع‌شونده** است — دلیل کامل در `cutoff.saturate`.
    out["person_n_prior_reservations"] = hist["p_hist_count"]
    out["person_n_prior_norecv"] = hist["p_hist_sum"]
    # نرخ، نه شمارنده: شمارنده یک شاخص زمان است (Spearman ۰.۷۶ با تقویم) و در تقسیم
    # زمانی برون‌یابی‌ناپذیر می‌شود. «رزرو در هفته» بین افرادِ هم‌زمان تفاوت دارد ولی
    # با گذشت زمان سیستماتیک رشد نمی‌کند. جزئیات در `cutoff.usage_rate`.
    day_num = (pd.to_datetime(f["date_gregorian"]) - pd.to_datetime(f["date_gregorian"]).min()).dt.days
    first_seen = day_num.groupby(f["PersonId"]).transform("min")
    out["person_reservations_per_week"] = usage_rate(hist["p_hist_count"], first_seen, day_num)
    out["person_expanding_norecv_rate"] = hist["p_hist_mean"]
    out["person_shrunk_norecv_rate"] = shrunk_rate(hist["p_hist_sum"], hist["p_hist_count"],
                                                   BETA_ALPHA, BETA_BETA)
    out["person_ewm_norecv_rate"] = _ewm_at_cutoff(f, half_life=5)

    # اثر ماه‌عسل (F65): شماره‌ی ترتیبی رزرو جاری
    # اثر ماه‌عسل (F65) کوتاه و **کران‌دار** است (~۲۰ رزرو اول)، پس فلگ باینری هم
    # سیگنال را می‌گیرد و هم شاخص زمان نمی‌سازد.
    out["is_honeymoon"] = (hist["p_hist_count"] < 20).astype(int)

    # --- ۵.۱۶.۴ cold-start ---
    out["is_cold_start"] = (hist["p_hist_count"] < COLD_START_K).astype(int)

    # --- ۵.۱۶.۵ زمینه‌ی فرد در این وعده (سهم تاریخی، leakage-safe) ---
    for col, name in [("Meal", "meal"), ("restaurant_canonical", "restaurant")]:
        f["_is"] = 1.0
        num = expanding_stat_at_cutoff(f, ["PersonId", col], "_is", out_prefix="n")["n_count"]
        den = hist["p_hist_count"]
        out[f"person_{name}_share"] = np.where(den > 0, num / den.replace(0, np.nan), np.nan)
    f = f.drop(columns="_is")
    dow = ((pd.to_datetime(f["date_gregorian"]).dt.dayofweek + 2) % 7).rename("dow")
    f2 = f.assign(dow=dow, _is=1.0)
    num_dow = expanding_stat_at_cutoff(f2, ["PersonId", "dow"], "_is", out_prefix="n")["n_count"]
    out["person_dow_share"] = np.where(hist["p_hist_count"] > 0,
                                       num_dow / hist["p_hist_count"].replace(0, np.nan), np.nan)

    return out.reindex(fact.index)


def attach_demographics(person_feats: pd.DataFrame, fact: pd.DataFrame,
                        dim: pd.DataFrame) -> pd.DataFrame:
    """۵.۱۶.۲ — فیچرهای جمعیتی ثابت (اولویت پایین‌تر، عمدتاً برای cold-start)."""
    d = dim.set_index("PersonId")
    cols = ["is_dorm_resident", "EducationSession", "CollegeName", "FieldName",
            "DegreeName", "Gender"]
    dm = fact[["PersonId"]].join(d[cols], on="PersonId")
    out = person_feats.copy()
    out["is_dorm_resident"] = dm["is_dorm_resident"].astype("boolean").astype("Int8")
    out["is_grad"] = dm["DegreeName"].astype(str).str.contains("ارشد|دکتری|PhD", na=False).astype(int)
    out["is_female"] = (dm["Gender"].astype(str).str.strip() == "زن").astype(int)
    out["is_evening_session"] = dm["EducationSession"].astype(str).str.contains("شبانه|نوبت دوم",
                                                                               na=False).astype(int)
    # کدگذاری فراوانی برای کاردینالیتی بالا (بند ۵.۱۰) — امن و بدون نشت هدف
    for col, name in [("CollegeName", "college"), ("FieldName", "field")]:
        freq = dm[col].value_counts(normalize=True)
        out[f"{name}_freq"] = dm[col].map(freq).astype(float)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    from src.config import set_global_seed
    from src.eda_lib.runners.s13_individual import load_fact

    set_global_seed()
    logger.info("Loading individual fact table ...")
    fact = load_fact()
    dim = pd.read_csv(DATA_PROCESSED / "person_dim_v3.csv")
    logger.info(f"{len(fact):,} reservations")

    logger.info("Building person features (cut-off aware) ...")
    feats = build_person_features(fact)
    feats = attach_demographics(feats, fact, dim)

    keys = fact[["PersonId", "date_gregorian", "Meal", "restaurant_canonical",
                 "city", "is_tehran", "dont_receive"]].reset_index(drop=True)
    out = pd.concat([keys, feats.reset_index(drop=True)], axis=1)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUTPUT_PATH, index=False)
    logger.info(f"Saved {OUTPUT_PATH} ({len(out):,} rows, {out.shape[1]} cols)")

    print("\nپوشش فیچرها:")
    for c in ["person_expanding_norecv_rate", "person_ewm_norecv_rate",
              "person_shrunk_norecv_rate", "person_meal_share", "person_dow_share"]:
        print(f"  {c:<34s} غیرتهی={out[c].notna().mean():.1%}  میانگین={out[c].mean():.4f}")
    print(f"  is_cold_start                      سهم={out['is_cold_start'].mean():.1%}")


if __name__ == "__main__":
    main()
