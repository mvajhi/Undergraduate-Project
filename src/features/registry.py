"""بند ۵.۱۵ — تولید خودکار `doc/feature_registry.md`.

رجیستری از خودِ ماتریس فیچر ساخته می‌شود، نه دستی — تا هرگز از کد عقب نیفتد. برای هر
فیچر: خانواده، ردیف پشتیبان دفتر حقایق، وابستگی زمانی، پوشش، و فیچرست‌های شامل آن.

اجرا: `python -m src.features.registry`
"""

import json
import logging

import pandas as pd

from src.config import DOCS_DIR, set_global_seed
from src.features.build import FEATURE_SETS_PATH, FEATURES_A_PATH, TARGET

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REGISTRY_PATH = DOCS_DIR / "feature_registry.md"

#: (خانواده، بند WBS، وابستگی زمانی، ردیف پشتیبان دفتر حقایق)
META: dict[str, tuple[str, str, str, str]] = {
    "cell_dow_shrunk_rate": ("خط پایه", "5.4", "انبساطی تا لحظه‌ی برش", "F8.3، بند ۲.۳"),
    "cell_expanding_rate": ("خط پایه", "5.4", "انبساطی تا لحظه‌ی برش", "F15"),
    "cell_shrunk_rate": ("خط پایه", "5.4", "انبساطی تا لحظه‌ی برش", "F8.3"),
    "cell_dow_expanding_rate": ("خط پایه", "5.4", "انبساطی تا لحظه‌ی برش", "F18"),
    "RestaurantName": ("هویت", "5.10", "ثابت", "F15 (η²=۰.۳۲۳)"),
    "Meal": ("هویت", "5.10", "ثابت", "F17"),
    "FoodType": ("هویت", "5.10", "ثابت", "F67"),
    "RestaurantType": ("هویت", "5.10", "ثابت", "F16"),
    "city": ("هویت", "5.10", "ثابت", "F12، F13"),
    "is_khabgah": ("هویت", "5.18", "ثابت", "F16"),
    "is_lunch": ("هویت", "5.18", "ثابت", "F17"),
    "log_res": ("مقیاس رزرو", "5.5", "معلوم در لحظه‌ی برش (رزرو ۷۲ ساعت زودتر بسته می‌شود)", "F54"),
    "dow": ("تقویمی", "5.1", "از پیش معلوم", "F18"),
    "jmonth": ("تقویمی", "5.1", "از پیش معلوم", "F21"),
    "day_of_month": ("تقویمی", "5.1", "از پیش معلوم", "—"),
    "week_of_semester": ("تقویمی", "5.1", "از پیش معلوم", "F61"),
    "is_holiday_any": ("تقویمی", "5.1", "از پیش معلوم", "F19"),
    "is_day_before_holiday": ("تقویمی ⭐", "5.1", "از پیش معلوم", "**F19** (+۳.۱ واحد درصد)"),
    "is_day_after_holiday": ("تقویمی", "5.1", "از پیش معلوم", "F19"),
    "is_bridge_day": ("تقویمی", "5.1", "از پیش معلوم", "F19"),
    "days_to_next_holiday": ("تقویمی", "5.1", "از پیش معلوم", "F19"),
    "days_since_last_holiday": ("تقویمی", "5.1", "از پیش معلوم", "F19"),
    "holiday_block_length": ("تقویمی", "5.1", "از پیش معلوم", "F19"),
    "pre_holiday_x_block_len": ("تقویمی", "5.1", "از پیش معلوم", "F19 (برهم‌کنش)"),
    "is_exam_period": ("تقویمی", "5.1", "از پیش معلوم", "F20"),
    "is_final_exam_period": ("تقویمی", "5.1", "از پیش معلوم", "F20"),
    "days_to_exam_start": ("تقویمی", "5.1", "از پیش معلوم", "F20"),
    "is_ramadan": ("تقویمی", "5.1", "از پیش معلوم", "F22 (فلگ گزارشی)"),
    "res_vs_history": ("مقیاس رزرو", "5.5", "معلوم در لحظه‌ی برش", "F54"),
    "res_vs_dow_history": ("مقیاس رزرو", "5.5", "معلوم در لحظه‌ی برش", "F54"),
    "log_daily_total_res": ("مقیاس رزرو ⭐", "5.5", "معلوم در لحظه‌ی برش", "**F61** (R² شوک ۰.۳۹۶→۰.۴۲۳)"),
    "food_expanding_rate": ("غذا", "5.6", "انبساطی تا لحظه‌ی برش", "F67"),
    "food_shrunk_rate": ("غذا", "5.6", "انبساطی تا لحظه‌ی برش", "F67 + F8.3"),
    "is_new_food": ("غذا", "5.6", "معلوم در لحظه‌ی برش", "F67"),
    "competitor_food_rate": ("غذا", "5.6", "انبساطی تا لحظه‌ی برش", "F67"),
    "food_rate_minus_competitor": ("غذا", "5.6", "انبساطی تا لحظه‌ی برش", "F67"),
    "temp_min": ("خارجی ⚠️", "5.8", "**مقدار واقعی روز d** — در استقرار باید پیش‌بینی باشد", "F26 (اثر ناچیز)"),
    "precip_type": ("خارجی ⚠️", "5.8", "**مقدار واقعی روز d** — در استقرار باید پیش‌بینی باشد", "F23، F24"),
    "is_snow_day": ("خارجی ⚠️", "5.8", "**مقدار واقعی روز d** — در استقرار باید پیش‌بینی باشد", "F23"),
    "day_shock_lag1": ("عامل روز ⭐", "5.17", "آخرین وعده‌ی هم‌نوعِ در دسترس", "**F59، F60، F61**"),
    "day_shock_lag2": ("عامل روز", "5.17", "دو وعده‌ی هم‌نوع قبل", "F61"),
    "day_shock_lag7": ("عامل روز", "5.17", "هفت وعده‌ی هم‌نوع قبل", "F61 (شام lag۷=+۰.۵۰۵)"),
    "day_shock_roll_mean_7": ("عامل روز", "5.17", "میانگین متحرک با shift(1)", "F60"),
    "dow_x_type": ("برهم‌کنش", "5.18", "ثابت/تقویمی", "F42 (ΔAIC=۵۶.۲)"),
    "meal_x_type": ("برهم‌کنش", "5.18", "ثابت", "F42 (ΔAIC=۵۲.۷)"),
    "dow_x_city": ("برهم‌کنش", "5.18", "ثابت/تقویمی", "F42 (ΔAIC=۲۰.۷)"),
    "city_x_meal": ("برهم‌کنش", "5.18", "ثابت", "F42 (ΔAIC=۱۹.۴)"),
    "composition_mean": ("پل A↔B", "5.19", "هویت رزروکنندگان معلوم + تاریخچه تا برش", "F63 (شام r=+۰.۵۲)"),
    "composition_std": ("پل A↔B", "5.19", "همان", "F62 (ساختار ضرب‌شونده)"),
    "composition_p90": ("پل A↔B", "5.19", "همان", "F62"),
    "composition_high_risk_share": ("پل A↔B", "5.19", "همان", "F62"),
    "composition_coverage": ("پل A↔B", "5.19", "همان", "کیفیت فیچر"),
    "composition_n": ("پل A↔B", "5.19", "همان", "—"),
    "composition_mean_dinner_only": ("پل A↔B", "5.19", "همان", "**F63**"),
    "composition_mean_x_dayshock": ("برهم‌کنش ضربی ⭐", "5.18", "همان", "**F62** (نسبت ۳.۳۶×)"),
    "composition_p90_x_dayshock": ("برهم‌کنش ضربی", "5.18", "همان", "F62"),
    "composition_high_risk_share_x_dayshock": ("برهم‌کنش ضربی", "5.18", "همان", "F62"),
}

REJECTED = [
    ("`days_since_same_food_served`", "5.6", "مصنوع تقویمی؛ پس از کنترل تقویم p=۰.۳۲", "F27"),
    ("`food_popularity_score` شخصی", "5.6", "پایداری جفت (فرد،غذا) ۰.۲۳۹ < پایداری خود فرد ۰.۴۳۱", "F66"),
    ("`has_extras`, `n_extras`", "5.6", "ستون منحط — ۱۰۰٪ رکوردها True", "F66"),
    ("`Count`", "5.16", "ستون منحط — ۱۰۰٪ برابر ۱", "F66"),
    ("`Price`", "5.16", "اثر باقیمانده ~۰.۰۰۱، Spearman +۰.۰۱۴", "F66"),
    ("`aqi`, `aqi_above_150`, `pm2_5`, `pm10`", "5.8", "همبستگی کاذب بین‌شهری؛ داخل تهران p=۰.۶۵", "**F25**"),
    ("مقدار پیوسته‌ی بارش", "5.8", "«نوع» معنادار (p=۲.۶e−۶) ولی «مقدار» نه (p=۰.۷۰)", "F24"),
    ("`card_ratio` خام", "5.7", "از خروجی همان وعده مشتق می‌شود — نشت مستقیم", "بند ۴-۲"),
    ("`person_rolling_norecv_rate_3`", "5.16", "AUC=۰.۶۲۷ در برابر ۰.۷۲۰ برای انبساطی", "F55، F56"),
    ("`person_last_outcome`", "5.16", "AUC=۰.۵۷۲ — ضعیف‌ترین کاندید", "F56"),
    ("ترجیح غذایی شخصی `person × food`", "5.16.6", "پایداری کمتر از خود فرد", "F66"),
    ("فیچر سرایت هم‌خوابگاهی", "5.16.6", "ICC پس از کنترل سلف-روز ۰.۰۰۳", "F66"),
    ("نتیجه‌ی ناهار روز d برای شام روز d", "5.13", "**نقض قاعده‌ی برش** — تله‌ی اصلی", "**F57**"),
    ("`cell_n_prior_days`, `food_n_prior_servings`", "5.4/5.6",
     "شاخص زمان (Spearman ۰.۹۵ و ۰.۷۲ با تقویم)؛ به‌تنهایی R²out را از +۰.۰۸ به −۱.۰۹ می‌برد", "ممیزی فاز ۵"),
    ("`person_n_prior_reservations` خام", "5.16.1",
     "همان مشکل (Spearman ۰.۷۶ با تقویم) — با نرخ `person_reservations_per_week` جایگزین شد", "ممیزی فاز ۵"),
]


def main() -> None:
    set_global_seed()
    df = pd.read_parquet(FEATURES_A_PATH)
    fs = json.loads(FEATURE_SETS_PATH.read_text())

    rows = []
    for name, (fam, wbs, timing, fact) in META.items():
        if name not in df.columns:
            continue
        s = df[name]
        cov = f"{s.notna().mean():.1%}"
        if pd.api.types.is_numeric_dtype(s):
            corr = s.corr(df[TARGET])
            corr_s = f"{corr:+.3f}" if pd.notna(corr) else "—"
        else:
            corr_s = "(دسته‌ای)"
        sets = [k.replace("FS_", "") for k, v in fs.items() if name in v]
        rows.append((name, fam, wbs, timing, fact, cov, corr_s, "، ".join(sets)))

    lines = [
        "# رجیستری ویژگی (Feature Registry) — فاز ۵",
        "",
        "> بند ۵.۱۵ WBS. **تولید خودکار** با `python -m src.features.registry` — دستی ویرایش نشود.",
        f"> ماتریس: `data/processed/features_A_v1.parquet` ({len(df):,} ردیف × {df.shape[1]} ستون)",
        "> ممیزی نشت: `doc/leakage_audit.md` · قواعد الزام‌آور: `doc/data_facts_register.md`",
        "",
        "**قاعده‌ی حاکم فاز ۵:** هیچ فیچری ساخته نشده که به یک ردیف دفتر حقایق یا دانش دامنه‌ای",
        "صریح متصل نباشد. ستون «شاهد» همان اتصال است.",
        "",
        "## فیچرهای ساخته‌شده",
        "",
        "| فیچر | خانواده | بند | وابستگی زمانی | شاهد | پوشش | r با هدف | فیچرست‌ها |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda x: (x[1], x[0])):
        lines.append("| `{}` | {} | {} | {} | {} | {} | {} | {} |".format(*r))

    lines += [
        "",
        "## فیچرهای **رد‌شده** — و دلیلشان",
        "",
        "> این جدول به‌اندازه‌ی جدول بالا مهم است: هر ردیف یک فیچری است که نسخه‌ی ۲.۰ WBS",
        "> پیشنهاد کرده بود یا شهود معمول می‌ساخت، ولی شواهد فاز ۴/۵ ردش کرد.",
        "",
        "| فیچر | بند | دلیل رد | شاهد |",
        "|---|---|---|---|",
    ]
    for name, wbs, why, fact in REJECTED:
        lines.append(f"| {name} | {wbs} | {why} | {fact} |")

    lines += [
        "",
        "## فیچرست‌ها (محور آزمایش فاز ۷)",
        "",
        "| فیچرست | تعداد فیچر | $R^2$ خارج‌نمونه* |",
        "|---|---|---|",
        "| `FS_baseline` | {} | +۰.۰۵۶ |".format(len(fs["FS_baseline"])),
        "| `FS_calendar` | {} | +۰.۱۳۶ |".format(len(fs["FS_calendar"])),
        "| `FS_lag` | {} | +۰.۰۸۲ |".format(len(fs["FS_lag"])),
        "| **`FS_day`** | {} | **+۰.۱۳۹** |".format(len(fs["FS_day"])),
        "| `FS_full_A` | {} | +۰.۱۱۹ |".format(len(fs["FS_full_A"])),
        "| `FS_bridge` | {} | +۰.۱۱۵ |".format(len(fs["FS_bridge"])),
        "",
        "\\* تقسیم زمانی منفرد ۷۵/۲۵ با HistGradientBoosting — فقط برای **ممیزی**، نه انتخاب مدل.",
        "بازه‌ی آزمون این تقسیم (۱۴۰۳-۰۱-۲۸ تا ۱۴۰۳-۰۳-۰۱) غیرعادی‌ترین بخش داده است",
        "(پس از رمضان + سوگواری ملی)، پس این اعداد **کران پایین** محافظه‌کارانه‌اند.",
        "انتخاب واقعی مدل با پروتکل walk-forward فاز ۶ انجام می‌شود.",
        "",
        "**مشاهده‌ی قابل‌توجه:** `FS_day` (با فیچرهای عامل روز) بهترین است و `FS_lag`",
        "به‌تنهایی از `FS_calendar` بدتر — سازگار با F59: آنچه اهمیت دارد شوک مشترک روز",
        "است، نه تاریخچه‌ی خودِ سلف. افزودن فیچرهای بیشتر (`FS_full_A`, `FS_bridge`) کمی",
        "بدتر می‌کند که با ۵٬۶۸۴ ردیف آموزش و ۶۰+ فیچر، نشانه‌ی بیش‌برازش است — بند ۵.۱۲",
        "(انتخاب ویژگی داخل fold) در فاز ۷ باید جدی گرفته شود.",
    ]
    REGISTRY_PATH.write_text("\n".join(lines) + "\n")
    logger.info(f"Saved {REGISTRY_PATH} ({len(rows)} features documented)")
    print(f"فیچرهای مستندشده: {len(rows)} · رد‌شده: {len(REJECTED)}")


if __name__ == "__main__":
    main()
