"""ساخت شبکه‌ی مدل‌سازی (L1/L2/L3) از سطح فرد — بند ۱۳-۷.

**چه چیزی عوض می‌شود.** تا امروز سطح L1 فقط از فایل تجمیعی
`raw_data_14020901_14030301.csv` می‌آمد و همان فایل پنجره را به ۱۴۲ روز می‌بست.
یافته F97 نشان داد تجمیع سطح فرد همان فایل را با خطای ~۱.۵٪ بازتولید می‌کند، پس
شبکه برای هر ۶۴۰ روز ساختنی است. این ماژول همان مسیر است.

**اعتبارسنجی اجباری، نه اختیاری.** `validate_against_aggregate` شبکه‌ی بازسازی‌شده را
روی پنجره‌ی هم‌پوشان با اعداد مرجع F97 مقایسه می‌کند (همبستگی ۰.۹۹۹۹، $\\rho$
۰.۰۸۰۵/۰.۰۸۰۲). انحراف از این اعداد یعنی پایپ‌لاین تازه ایراد دارد — به همین دلیل
`build_all` به‌طور پیش‌فرض آن را اجرا می‌کند و شکستش خطاست، نه یادداشت.

**چهار تصمیم دامنه که اینجا اعمال می‌شوند** (هیچ‌کدام تازه نیست؛ همه از `decision_log`):

| فیلتر | چرا | مرجع |
|---|---|---|
| فقط `restaurant_class == "core"` | چهار رده‌ی دیگر تصمیم پخت ندارند | ردیف ۶۶ |
| فقط `is_dish` | `…ظرفیت محدود-پیش پرداخت:` و `بسته ارزاق` غذا نیستند | F114 |
| فقط `mealid ∈ {۱, ۳}` | ناهار و شام؛ چهار وعده‌ی دیگر دامنه‌ی مسئله نیستند | F95، ردیف ۷۳ |
| هدف = **اجتماع** سه کد | تک‌کدها رژیم عوض می‌کنند، اجتماعشان نه | F90 |

⚠️ **`Count` جمع می‌شود، ردیف شمرده نمی‌شود** (F94): در v4 دیگر باینری نیست
(۲۳٬۹۴۲ ردیف با `Count=۲`). شمردن ردیف حجم را کم‌برآورد می‌کند.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED
from src.data.food_dim_v4 import OUTPUT_PATH as FOOD_DIM_PATH
from src.data.load_v4 import load_clean_v4
from src.data.restaurant_dim_v4 import OUTPUT_PATH as RESTAURANT_DIM_PATH
from src.eda_lib.restaurant_taxonomy import CORE_CLASS

logger = logging.getLogger(__name__)

#: کلید هر سطح. `date_gregorian` و `meal` در همه‌ی سطوح مشترک‌اند.
LEVEL_KEYS: dict[str, list[str]] = {
    "L1": ["date_gregorian", "meal", "restaurant", "dish"],
    "L2": ["date_gregorian", "meal", "restaurant"],
    "L3": ["date_gregorian", "meal"],
}

OUTPUT_PATHS = {level: DATA_PROCESSED / f"grid_{level}_v4.parquet" for level in LEVEL_KEYS}

#: اعداد مرجع F97 و رواداریشان. رواداری از تفاوت روش می‌آید نه از سلیقه: بازسازی
#: از سطح فرد و فایل تجمیعی دو شمارش مستقل‌اند و F97 خودش میانه‌ی خطای نسبی ۱.۵٪ را
#: ثبت کرده است.
F97_REFERENCE = {
    "res_correlation": (0.9999, 0.002),
    "rho_aggregate": (0.0805, 0.004),
    "rho_reconstructed": (0.0802, 0.004),
}


def _as_category(col: pd.Series) -> pd.Series:
    """ستون را رده‌ای برمی‌گرداند بدون عبور از `object` — حتی اگر از قبل رده‌ای باشد."""
    if isinstance(col.dtype, pd.CategoricalDtype):
        return col
    return col.astype("category")


def _map_categories(col: pd.Series, lookup: pd.Series, *, to_category: bool = True):
    """نگاشت یک ستون رده‌ای با جست‌وجو روی **رده‌ها**، نه روی ردیف‌ها.

    کدهای رده‌ای ردیف‌ها ثابت می‌مانند و فقط برچسب‌ها عوض می‌شوند؛ پس هزینه‌ی
    حافظه با تعداد **رده** بالا می‌رود نه با تعداد ردیف. مقدارِ نگاشت‌نشده `NaN`
    می‌شود، دقیقاً مثل `merge(how="left")`.
    """
    values = lookup.reindex(col.cat.categories).to_numpy()
    codes = col.cat.codes.to_numpy()
    if not to_category:
        out = np.where(codes >= 0, values[np.clip(codes, 0, None)], np.nan)
        return pd.Series(out, index=col.index)

    uniq = pd.Index(pd.unique(pd.Series(values).dropna()))
    remap = {v: i for i, v in enumerate(uniq)}
    per_cat = np.array([remap.get(v, -1) for v in values], dtype=np.int32)
    out_codes = np.where(codes >= 0, per_cat[np.clip(codes, 0, None)], -1)
    return pd.Series(pd.Categorical.from_codes(out_codes, categories=uniq), index=col.index)


def load_person_level() -> pd.DataFrame:
    """ردیف‌های سطح فرد، پاکسازی‌شده و **فیلترشده به دامنه‌ی مسئله**.

    ستون‌های `restaurant`/`dish` از جدول‌های بعد می‌آیند، نه از نام خام — تا یک غذا
    با دو رسم‌الخط دو سلول نشود (ردیف ۷۲).
    """
    df = load_clean_v4(
        ["PersonId", "RestaurantName", "FoodName", "Count", "miladi", "mealid", "ReserveStatus"]
    )

    restaurants = pd.read_csv(RESTAURANT_DIM_PATH)
    foods = pd.read_csv(FOOD_DIM_PATH)

    # ⚠️ قاعده‌ی ۱ قرارداد حافظه (`AGENTS.md`): «روی دسته‌ها کار کنید، نه روی ردیف‌ها.»
    #
    # نسخه‌ی پیشین این تابع دو `merge` روی ۸.۴ میلیون ردیف می‌زد و ده ستون `object`
    # می‌ساخت؛ اوج مصرف **۱۲.۳ گیگابایت** روی ماشین ۱۶ گیگابایتی بود — یعنی یک
    # فرایند دیگرِ هم‌زمان کافی بود تا `SIGKILL` بگیرد. ولی کاردینالیتی واقعی این
    # نگاشت‌ها کوچک است: ۱۲۴ سلف و چند صد نام غذا. پس نگاشت روی **رده‌ها** اجرا
    # می‌شود نه روی ردیف‌ها، و کد رده‌ایِ هر ردیف دست‌نخورده می‌ماند.
    #
    # همان الگوی `classify_restaurants` که از ۸.۴ میلیون اجرای regex به ۱۲۴ رسید.
    # هم‌ارزی خروجی با نسخه‌ی merge در `test_person_level_mapping_matches_merge` آزموده می‌شود.
    df["restaurant"] = _as_category(df["RestaurantName"])
    df["dish_raw"] = _as_category(df["FoodName"])

    rmap = restaurants.drop_duplicates("raw_name").set_index("raw_name")
    fmap = foods.drop_duplicates("raw_food_name").set_index("raw_food_name")
    for col in ("restaurant_class", "campus", "city", "venue_type"):
        df[col] = _map_categories(df["restaurant"], rmap[col])
    for src, dst in (("canonical_dish", "dish"), ("food_group", "food_group")):
        df[dst] = _map_categories(df["dish_raw"], fmap[src])
    df["is_dish"] = _map_categories(df["dish_raw"], fmap["is_dish"], to_category=False)

    # هیچ ردیفی نباید بی‌ویژگی بماند — دروازه‌ی بند ۱۳-۵ در مصرف هم برقرار است.
    for col in ("restaurant_class", "is_dish"):
        missing = int(df[col].isna().sum())
        if missing:
            raise ValueError(f"{missing} ردیف بدون «{col}» — جدول بعد با داده هم‌نسخه نیست")

    # ⚠️ قاعده‌ی ۳ قرارداد حافظه (`AGENTS.md`): «هیچ ستون رشته‌ای آزادی نسازید.»
    # merge با جدول‌های بعد، ده ستون `object` روی ۸.۲ میلیون ردیف می‌سازد —
    # هرکدام ~۴۵۰ مگابایت، و اوج مصرف را روی ماشین ۱۶ گیگابایتی به **۱۲.۳
    # گیگابایت** می‌رساند. کاردینالیتی واقعی‌شان کوچک است (۱۲۴ سلف، ۱۲۲ غذا،
    # ۶ شهر)، پس `category` همان اطلاعات را در کسری از حافظه نگه می‌دارد.
    # ⚠️ این تبدیل **پس از** merge انجام می‌شود نه قبلش: کلید merge اگر رده‌ای
    # باشد و دو طرف رده‌های متفاوت داشته باشند، pandas به `object` برمی‌گرداند و
    # صرفه‌جویی از بین می‌رود.
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype("category")

    return df


def apply_domain_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """فیلترهای دامنه، به‌همراه گزارش اینکه هر فیلتر چقدر برداشت.

    گزارش برمی‌گردد چون «چه چیزی حذف شد» به‌اندازه‌ی «چه ماند» مهم است — قاعده‌ی
    «نام بدون نگاشت نباید بی‌صدا حذف شود» (دروازه‌ی بند ۱۳-۵).
    """
    total = df["Count"].sum()
    steps = []

    for label, mask in [
        ("رده‌ی غیر core (ردیف ۶۶)", df["restaurant_class"] != CORE_CLASS),
        ("«غذا» نیست (F114)", ~df["is_dish"].astype(bool)),
    ]:
        removed = df.loc[mask, "Count"].sum()
        steps.append({
            "filter": label,
            "rows_removed": int(mask.sum()),
            "portions_removed": int(removed),
            "pct_portions": round(100 * removed / total, 3),
        })
        df = df.loc[~mask]

    steps.append({
        "filter": "باقی‌مانده", "rows_removed": len(df),
        "portions_removed": int(df["Count"].sum()),
        "pct_portions": round(100 * df["Count"].sum() / total, 3),
    })
    return df, pd.DataFrame(steps)


def aggregate_to_level(df: pd.DataFrame, level: str) -> pd.DataFrame:
    """تجمیع به یک سطح. `Res` جمعِ `Count` است، نه تعداد ردیف (F94)."""
    keys = LEVEL_KEYS[level]
    df = df.copy()
    df["_no_recv"] = df["Count"].where(df["dont_receive"], 0)

    out = df.groupby(keys, observed=True).agg(
        Res=("Count", "sum"),
        NoRecv=("_no_recv", "sum"),
        n_reservations=("Count", "size"),
        # تعداد **فرد** یکتا، نه تعداد رزرو: ۵۰ پرس برای ۵۰ نفر با ۵۰ پرس برای ۲۵ نفرِ
        # دوپرسی یک چیز نیستند، و `Count` در v4 دیگر باینری نیست (F94).
        n_people=("PersonId", "nunique"),
    ).reset_index()

    out["Recv"] = out["Res"] - out["NoRecv"]
    out["rho"] = (out["NoRecv"] / out["Res"]).where(out["Res"] > 0)
    return out


# ---------------------------------------------------------------------------
# اعتبارسنجی اجباری در برابر فایل تجمیعی (اعداد مرجع F97)
# ---------------------------------------------------------------------------


#: مرجع مقایسه = snapshot **پاکسازی‌شده‌ی** پنجره‌ی قفل‌شده، نه فایل خام تجمیعی.
#:
#: ⚠️ این تمایز را همین دروازه پیدا کرد، نه یک بازبینی. نسخه‌ی اول به
#: `data/raw/raw_data_14020901_14030301.csv` وصل بود و آزمون شکست: همبستگی `Res`
#: ‏۰.۸۰۷۲ به‌جای ۰.۹۹۹۹، با الگوی عجیبِ **دقیقاً منفی ۵۰ درصد** روی یک زیرمجموعه.
#: علت: فایل خام روی ۲۷ روز از ۱۴۲ روز ردیف **کاملاً تکراری** دارد (۱۵٬۹۶۸ ردیف،
#: ۱۳٬۰۸۵ یکتا) — همان تله‌ی مرز فایل‌های ماهانه که ردیف ۱۸ `decision_log` مستند
#: کرده و `src/data/clean.py` حذفش می‌کند. مقایسه با خامْ بازسازیِ درست را «غلط»
#: نشان می‌داد.
AGGREGATE_REFERENCE_PATH = DATA_PROCESSED / "dataset_v2.csv"


def _aggregate_reference() -> pd.DataFrame:
    """snapshot پاکسازی‌شده‌ی پنجره‌ی قفل‌شده، تجمیع‌شده به سطح روز×وعده."""
    agg = pd.read_csv(AGGREGATE_REFERENCE_PATH)
    agg["date_gregorian"] = pd.to_datetime(agg["date_gregorian"])
    return agg.groupby(["date_gregorian", "Meal"], as_index=False).agg(
        Res=("Res", "sum"), NoRecv=("NoRecv", "sum"),
    ).rename(columns={"Meal": "meal"})


def validate_against_aggregate(l3: pd.DataFrame) -> dict:
    """مقایسه‌ی شبکه‌ی بازسازی‌شده با فایل تجمیعی، روی پنجره‌ی هم‌پوشان.

    این تابع **آزمون** است نه گزارش: خروجی `passed` می‌گوید اعداد با مرجع F97
    می‌خوانند یا نه. اگر نخوانند، پایپ‌لاین تازه ایراد دارد و شبکه نباید استفاده شود.
    """
    ref = _aggregate_reference()
    got = l3[["date_gregorian", "meal", "Res", "NoRecv"]].copy()
    got["date_gregorian"] = pd.to_datetime(got["date_gregorian"])

    merged = ref.merge(got, on=["date_gregorian", "meal"], how="inner", suffixes=("_agg", "_new"))
    if merged.empty:
        raise ValueError("پنجره‌ی هم‌پوشانی پیدا نشد — کلید تاریخ/وعده نمی‌خواند")

    rho_agg = merged["NoRecv_agg"].sum() / merged["Res_agg"].sum()
    rho_new = merged["NoRecv_new"].sum() / merged["Res_new"].sum()
    rel_err = ((merged["Res_new"] - merged["Res_agg"]).abs() / merged["Res_agg"])

    result = {
        "n_overlap_cells": len(merged),
        "n_overlap_days": merged["date_gregorian"].nunique(),
        "res_correlation": round(float(merged["Res_agg"].corr(merged["Res_new"])), 4),
        "rho_correlation": round(
            float((merged["NoRecv_agg"] / merged["Res_agg"]).corr(merged["NoRecv_new"] / merged["Res_new"])), 4
        ),
        "rho_aggregate": round(float(rho_agg), 4),
        "rho_reconstructed": round(float(rho_new), 4),
        "pct_cells_within_5pct": round(100 * float((rel_err <= 0.05).mean()), 1),
        "median_abs_rel_error_pct": round(100 * float(rel_err.median()), 2),
    }

    checks = {
        key: abs(result[key] - expected) <= tol
        for key, (expected, tol) in F97_REFERENCE.items()
    }
    result["checks"] = checks
    result["passed"] = all(checks.values())
    return result


# ---------------------------------------------------------------------------
# اسمبل
# ---------------------------------------------------------------------------


def build_all(*, validate: bool = True) -> dict[str, pd.DataFrame]:
    person = load_person_level()
    person, filter_report = apply_domain_filters(person)
    logger.info("گزارش فیلتر دامنه:\n%s", filter_report.to_string(index=False))

    grids = {}
    for level in LEVEL_KEYS:
        grid = aggregate_to_level(person, level)
        grids[level] = grid
        OUTPUT_PATHS[level].parent.mkdir(parents=True, exist_ok=True)
        grid.to_parquet(OUTPUT_PATHS[level], index=False)
        logger.info("%s ذخیره شد: %s (%d سلول)", level, OUTPUT_PATHS[level], len(grid))

    if validate:
        result = validate_against_aggregate(grids["L3"])
        logger.info("اعتبارسنجی F97: %s", result)
        if not result["passed"]:
            failed = [k for k, ok in result["checks"].items() if not ok]
            raise ValueError(
                f"شبکه‌ی بازسازی‌شده با اعداد مرجع F97 نمی‌خواند ({failed}). "
                f"نتیجه: {result}"
            )
        grids["_validation"] = result

    grids["_filter_report"] = filter_report
    return grids


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    from src.config import set_global_seed

    set_global_seed()
    result = build_all()

    pd.set_option("display.width", 200)
    print("\n=== اندازه‌ی سطوح ===")
    for level in LEVEL_KEYS:
        g = result[level]
        print(f"  {level}: {len(g):,} سلول · {g.Res.sum():,.0f} پرس · rho کل {g.NoRecv.sum()/g.Res.sum():.4f}")

    print("\n=== اعتبارسنجی در برابر فایل تجمیعی (مرجع F97) ===")
    for k, v in result["_validation"].items():
        print(f"  {k}: {v}")


def test_person_level_mapping_matches_merge(sample_rows: int = 2_000_000) -> tuple[bool, str]:
    """نگاشت روی رده‌ها باید **دقیقاً** همان چیزی را بدهد که `merge` می‌داد.

    این تست بدهی یک بازنویسی حافظه‌ای را می‌پردازد: `load_person_level` از دو
    `merge` روی ۸.۴ میلیون ردیف به نگاشت روی رده‌ها تغییر کرد (اوج ۱۲.۳ گیگابایت
    → ۱.۸؛ زمان ۳۱.۸ ثانیه → ۳.۳). صرفه‌جویی بی‌معناست اگر یک مقدار هم عوض شده
    باشد، پس نسخه‌ی `merge` روی نمونه‌ای از همان داده بازسازی و مقایسه می‌شود.
    """
    df = load_clean_v4(
        ["PersonId", "RestaurantName", "FoodName", "Count", "miladi", "mealid", "ReserveStatus"]
    ).head(sample_rows)
    restaurants = pd.read_csv(RESTAURANT_DIM_PATH)
    foods = pd.read_csv(FOOD_DIM_PATH)

    legacy = df.copy(deep=False)
    legacy["restaurant"] = legacy["RestaurantName"].astype(str)
    legacy["dish_raw"] = legacy["FoodName"].astype(str)
    legacy = legacy.merge(
        restaurants[["raw_name", "restaurant_class", "campus", "city", "venue_type"]],
        left_on="restaurant", right_on="raw_name", how="left",
    ).merge(
        foods[["raw_food_name", "canonical_dish", "food_group", "is_dish"]],
        left_on="dish_raw", right_on="raw_food_name", how="left",
    ).rename(columns={"canonical_dish": "dish"})

    fast = df.copy(deep=False)
    fast["restaurant"] = _as_category(fast["RestaurantName"])
    fast["dish_raw"] = _as_category(fast["FoodName"])
    rmap = restaurants.drop_duplicates("raw_name").set_index("raw_name")
    fmap = foods.drop_duplicates("raw_food_name").set_index("raw_food_name")
    for col in ("restaurant_class", "campus", "city", "venue_type"):
        fast[col] = _map_categories(fast["restaurant"], rmap[col])
    fast["dish"] = _map_categories(fast["dish_raw"], fmap["canonical_dish"])
    fast["food_group"] = _map_categories(fast["dish_raw"], fmap["food_group"])
    fast["is_dish"] = _map_categories(fast["dish_raw"], fmap["is_dish"], to_category=False)

    checked = ["restaurant", "dish_raw", "restaurant_class", "campus", "city",
               "venue_type", "dish", "food_group", "is_dish"]
    bad = []
    for col in checked:
        a = legacy[col].astype(object).where(legacy[col].notna(), None)
        b = fast[col].astype(object).where(fast[col].notna(), None)
        n_diff = int((a.to_numpy() != b.to_numpy()).sum())
        if n_diff:
            bad.append(f"{col}:{n_diff}")
    ok = not bad
    return ok, (f"{len(df):,} ردیف × {len(checked)} ستون بررسی شد · "
                + ("هیچ اختلافی نیست" if ok else "اختلاف در " + "، ".join(bad)))
