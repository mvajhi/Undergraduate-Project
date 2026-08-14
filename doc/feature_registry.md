# رجیستری ویژگی (Feature Registry) — فاز ۵

> بند ۵.۱۵ WBS. **تولید خودکار** با `python -m src.features.registry` — دستی ویرایش نشود.
> ماتریس: `data/processed/features_A_v1.parquet` (7,579 ردیف × 100 ستون)
> ممیزی نشت: `doc/leakage_audit.md` · قواعد الزام‌آور: `doc/data_facts_register.md`

**قاعده‌ی حاکم فاز ۵:** هیچ فیچری ساخته نشده که به یک ردیف دفتر حقایق یا دانش دامنه‌ای
صریح متصل نباشد. ستون «شاهد» همان اتصال است.

## فیچرهای ساخته‌شده

| فیچر | خانواده | بند | وابستگی زمانی | شاهد | پوشش | r با هدف | فیچرست‌ها |
|---|---|---|---|---|---|---|---|
| `city_x_meal` | برهم‌کنش | 5.18 | ثابت | F42 (ΔAIC=۱۹.۴) | 100.0% | (دسته‌ای) | full_A، bridge |
| `dow_x_city` | برهم‌کنش | 5.18 | ثابت/تقویمی | F42 (ΔAIC=۲۰.۷) | 100.0% | (دسته‌ای) | full_A، bridge |
| `dow_x_type` | برهم‌کنش | 5.18 | ثابت/تقویمی | F42 (ΔAIC=۵۶.۲) | 100.0% | +0.140 | full_A، bridge |
| `meal_x_type` | برهم‌کنش | 5.18 | ثابت | F42 (ΔAIC=۵۲.۷) | 100.0% | +0.093 | full_A، bridge |
| `composition_high_risk_share_x_dayshock` | برهم‌کنش ضربی | 5.18 | همان | F62 | 92.6% | +0.232 | bridge |
| `composition_p90_x_dayshock` | برهم‌کنش ضربی | 5.18 | همان | F62 | 92.6% | +0.247 | bridge |
| `composition_mean_x_dayshock` | برهم‌کنش ضربی ⭐ | 5.18 | همان | **F62** (نسبت ۳.۳۶×) | 92.6% | +0.247 | bridge |
| `day_of_month` | تقویمی | 5.1 | از پیش معلوم | — | 100.0% | +0.088 | calendar، lag، day، full_A، bridge |
| `days_since_last_holiday` | تقویمی | 5.1 | از پیش معلوم | F19 | 83.0% | -0.014 | calendar، lag، day، full_A، bridge |
| `days_to_exam_start` | تقویمی | 5.1 | از پیش معلوم | F20 | 100.0% | -0.079 | calendar، lag، day، full_A، bridge |
| `days_to_next_holiday` | تقویمی | 5.1 | از پیش معلوم | F19 | 100.0% | -0.082 | calendar، lag، day، full_A، bridge |
| `dow` | تقویمی | 5.1 | از پیش معلوم | F18 | 100.0% | +0.144 | calendar، lag، day، full_A، bridge |
| `holiday_block_length` | تقویمی | 5.1 | از پیش معلوم | F19 | 2.5% | +0.473 | calendar، lag، day، full_A، bridge |
| `is_bridge_day` | تقویمی | 5.1 | از پیش معلوم | F19 | 100.0% | +0.061 | calendar، lag، day، full_A، bridge |
| `is_day_after_holiday` | تقویمی | 5.1 | از پیش معلوم | F19 | 100.0% | -0.059 | calendar، lag، day، full_A، bridge |
| `is_exam_period` | تقویمی | 5.1 | از پیش معلوم | F20 | 100.0% | -0.095 | calendar، lag، day، full_A، bridge |
| `is_final_exam_period` | تقویمی | 5.1 | از پیش معلوم | F20 | 100.0% | -0.060 | calendar، lag، day، full_A، bridge |
| `is_holiday_any` | تقویمی | 5.1 | از پیش معلوم | F19 | 100.0% | +0.012 | calendar، lag، day، full_A، bridge |
| `is_ramadan` | تقویمی | 5.1 | از پیش معلوم | F22 (فلگ گزارشی) | 100.0% | +0.158 | calendar، lag، day، full_A، bridge |
| `jmonth` | تقویمی | 5.1 | از پیش معلوم | F21 | 100.0% | +0.099 | calendar، lag، day، full_A، bridge |
| `pre_holiday_x_block_len` | تقویمی | 5.1 | از پیش معلوم | F19 (برهم‌کنش) | 100.0% | — | calendar، lag، day، full_A، bridge |
| `week_of_semester` | تقویمی | 5.1 | از پیش معلوم | F61 | 83.4% | -0.127 | calendar، lag، day، full_A، bridge |
| `is_day_before_holiday` | تقویمی ⭐ | 5.1 | از پیش معلوم | **F19** (+۳.۱ واحد درصد) | 100.0% | +0.015 | calendar، lag، day، full_A، bridge |
| `is_snow_day` | خارجی ⚠️ | 5.8 | **مقدار واقعی روز d** — در استقرار باید پیش‌بینی باشد | F23 | 100.0% | +0.042 | full_A، bridge |
| `precip_type` | خارجی ⚠️ | 5.8 | **مقدار واقعی روز d** — در استقرار باید پیش‌بینی باشد | F23، F24 | 100.0% | (دسته‌ای) | full_A، bridge |
| `temp_min` | خارجی ⚠️ | 5.8 | **مقدار واقعی روز d** — در استقرار باید پیش‌بینی باشد | F26 (اثر ناچیز) | 100.0% | -0.083 | full_A، bridge |
| `cell_dow_expanding_rate` | خط پایه | 5.4 | انبساطی تا لحظه‌ی برش | F18 | 94.1% | +0.413 | lag، day، full_A، bridge |
| `cell_dow_shrunk_rate` | خط پایه | 5.4 | انبساطی تا لحظه‌ی برش | F8.3، بند ۲.۳ | 100.0% | +0.399 | baseline، calendar، lag، day، full_A، bridge |
| `cell_expanding_rate` | خط پایه | 5.4 | انبساطی تا لحظه‌ی برش | F15 | 98.9% | +0.387 | lag، day، full_A، bridge |
| `cell_shrunk_rate` | خط پایه | 5.4 | انبساطی تا لحظه‌ی برش | F8.3 | 100.0% | +0.388 | lag، day، full_A، bridge |
| `day_shock_lag2` | عامل روز | 5.17 | دو وعده‌ی هم‌نوع قبل | F61 | 90.7% | +0.129 | day، full_A، bridge |
| `day_shock_lag7` | عامل روز | 5.17 | هفت وعده‌ی هم‌نوع قبل | F61 (شام lag۷=+۰.۵۰۵) | 80.1% | +0.012 | day، full_A، bridge |
| `day_shock_roll_mean_7` | عامل روز | 5.17 | میانگین متحرک با shift(1) | F60 | 98.5% | +0.176 | day، full_A، bridge |
| `day_shock_lag1` | عامل روز ⭐ | 5.17 | آخرین وعده‌ی هم‌نوعِ در دسترس | **F59، F60، F61** | 92.6% | +0.249 | day، full_A، bridge |
| `competitor_food_rate` | غذا | 5.6 | انبساطی تا لحظه‌ی برش | F67 | 85.6% | +0.030 | full_A، bridge |
| `food_expanding_rate` | غذا | 5.6 | انبساطی تا لحظه‌ی برش | F67 | 93.7% | +0.101 | full_A، bridge |
| `food_rate_minus_competitor` | غذا | 5.6 | انبساطی تا لحظه‌ی برش | F67 | 83.0% | +0.060 | full_A، bridge |
| `food_shrunk_rate` | غذا | 5.6 | انبساطی تا لحظه‌ی برش | F67 + F8.3 | 100.0% | +0.098 | full_A، bridge |
| `is_new_food` | غذا | 5.6 | معلوم در لحظه‌ی برش | F67 | 100.0% | -0.036 | full_A، bridge |
| `log_res` | مقیاس رزرو | 5.5 | معلوم در لحظه‌ی برش (رزرو ۷۲ ساعت زودتر بسته می‌شود) | F54 | 100.0% | -0.216 | calendar، lag، day، full_A، bridge |
| `res_vs_dow_history` | مقیاس رزرو | 5.5 | معلوم در لحظه‌ی برش | F54 | 94.1% | -0.173 | full_A، bridge |
| `res_vs_history` | مقیاس رزرو | 5.5 | معلوم در لحظه‌ی برش | F54 | 98.9% | -0.204 | full_A، bridge |
| `log_daily_total_res` | مقیاس رزرو ⭐ | 5.5 | معلوم در لحظه‌ی برش | **F61** (R² شوک ۰.۳۹۶→۰.۴۲۳) | 100.0% | -0.098 | full_A، bridge |
| `FoodType` | هویت | 5.10 | ثابت | F67 | 100.0% | (دسته‌ای) | calendar، lag، day، full_A، bridge |
| `Meal` | هویت | 5.10 | ثابت | F17 | 100.0% | (دسته‌ای) | calendar، lag، day، full_A، bridge |
| `RestaurantName` | هویت | 5.10 | ثابت | F15 (η²=۰.۳۲۳) | 100.0% | (دسته‌ای) | calendar، lag، day، full_A، bridge |
| `RestaurantType` | هویت | 5.10 | ثابت | F16 | 100.0% | (دسته‌ای) | calendar، lag، day، full_A، bridge |
| `city` | هویت | 5.10 | ثابت | F12، F13 | 100.0% | (دسته‌ای) | calendar، lag، day، full_A، bridge |
| `is_khabgah` | هویت | 5.18 | ثابت | F16 | 100.0% | -0.106 | calendar، lag، day، full_A، bridge |
| `is_lunch` | هویت | 5.18 | ثابت | F17 | 100.0% | +0.123 | calendar، lag، day، full_A، bridge |
| `composition_coverage` | پل A↔B | 5.19 | همان | کیفیت فیچر | 100.0% | +0.027 | bridge |
| `composition_high_risk_share` | پل A↔B | 5.19 | همان | F62 | 100.0% | +0.360 | bridge |
| `composition_mean` | پل A↔B | 5.19 | هویت رزروکنندگان معلوم + تاریخچه تا برش | F63 (شام r=+۰.۵۲) | 100.0% | +0.378 | bridge |
| `composition_mean_dinner_only` | پل A↔B | 5.19 | همان | **F63** | 100.0% | -0.062 | bridge |
| `composition_n` | پل A↔B | 5.19 | همان | — | 100.0% | -0.160 | bridge |
| `composition_p90` | پل A↔B | 5.19 | همان | F62 | 100.0% | +0.374 | bridge |
| `composition_std` | پل A↔B | 5.19 | همان | F62 (ساختار ضرب‌شونده) | 100.0% | +0.342 | bridge |

## فیچرهای **رد‌شده** — و دلیلشان

> این جدول به‌اندازه‌ی جدول بالا مهم است: هر ردیف یک فیچری است که نسخه‌ی ۲.۰ WBS
> پیشنهاد کرده بود یا شهود معمول می‌ساخت، ولی شواهد فاز ۴/۵ ردش کرد.

| فیچر | بند | دلیل رد | شاهد |
|---|---|---|---|
| `days_since_same_food_served` | 5.6 | مصنوع تقویمی؛ پس از کنترل تقویم p=۰.۳۲ | F27 |
| `food_popularity_score` شخصی | 5.6 | پایداری جفت (فرد،غذا) ۰.۲۳۹ < پایداری خود فرد ۰.۴۳۱ | F66 |
| `has_extras`, `n_extras` | 5.6 | ستون منحط — ۱۰۰٪ رکوردها True | F66 |
| `Count` | 5.16 | ستون منحط — ۱۰۰٪ برابر ۱ | F66 |
| `Price` | 5.16 | اثر باقیمانده ~۰.۰۰۱، Spearman +۰.۰۱۴ | F66 |
| `aqi`, `aqi_above_150`, `pm2_5`, `pm10` | 5.8 | همبستگی کاذب بین‌شهری؛ داخل تهران p=۰.۶۵ | **F25** |
| مقدار پیوسته‌ی بارش | 5.8 | «نوع» معنادار (p=۲.۶e−۶) ولی «مقدار» نه (p=۰.۷۰) | F24 |
| `card_ratio` خام | 5.7 | از خروجی همان وعده مشتق می‌شود — نشت مستقیم | بند ۴-۲ |
| `person_rolling_norecv_rate_3` | 5.16 | AUC=۰.۶۲۷ در برابر ۰.۷۲۰ برای انبساطی | F55، F56 |
| `person_last_outcome` | 5.16 | AUC=۰.۵۷۲ — ضعیف‌ترین کاندید | F56 |
| ترجیح غذایی شخصی `person × food` | 5.16.6 | پایداری کمتر از خود فرد | F66 |
| فیچر سرایت هم‌خوابگاهی | 5.16.6 | ICC پس از کنترل سلف-روز ۰.۰۰۳ | F66 |
| نتیجه‌ی ناهار روز d برای شام روز d | 5.13 | **نقض قاعده‌ی برش** — تله‌ی اصلی | **F57** |
| `cell_n_prior_days`, `food_n_prior_servings` | 5.4/5.6 | شاخص زمان (Spearman ۰.۹۵ و ۰.۷۲ با تقویم)؛ به‌تنهایی R²out را از +۰.۰۸ به −۱.۰۹ می‌برد | ممیزی فاز ۵ |
| `person_n_prior_reservations` خام | 5.16.1 | همان مشکل (Spearman ۰.۷۶ با تقویم) — با نرخ `person_reservations_per_week` جایگزین شد | ممیزی فاز ۵ |

## فیچرست‌ها (محور آزمایش فاز ۷)

| فیچرست | تعداد فیچر | $R^2$ خارج‌نمونه* |
|---|---|---|
| `FS_baseline` | 1 | +۰.۰۵۶ |
| `FS_calendar` | 31 | +۰.۱۳۶ |
| `FS_lag` | 43 | +۰.۰۸۲ |
| **`FS_day`** | 47 | **+۰.۱۳۹** |
| `FS_full_A` | 62 | +۰.۱۱۹ |
| `FS_bridge` | 72 | +۰.۱۱۵ |

\* تقسیم زمانی منفرد ۷۵/۲۵ با HistGradientBoosting — فقط برای **ممیزی**، نه انتخاب مدل.
بازه‌ی آزمون این تقسیم (۱۴۰۳-۰۱-۲۸ تا ۱۴۰۳-۰۳-۰۱) غیرعادی‌ترین بخش داده است
(پس از رمضان + سوگواری ملی)، پس این اعداد **کران پایین** محافظه‌کارانه‌اند.
انتخاب واقعی مدل با پروتکل walk-forward فاز ۶ انجام می‌شود.

**مشاهده‌ی قابل‌توجه:** `FS_day` (با فیچرهای عامل روز) بهترین است و `FS_lag`
به‌تنهایی از `FS_calendar` بدتر — سازگار با F59: آنچه اهمیت دارد شوک مشترک روز
است، نه تاریخچه‌ی خودِ سلف. افزودن فیچرهای بیشتر (`FS_full_A`, `FS_bridge`) کمی
بدتر می‌کند که با ۵٬۶۸۴ ردیف آموزش و ۶۰+ فیچر، نشانه‌ی بیش‌برازش است — بند ۵.۱۲
(انتخاب ویژگی داخل fold) در فاز ۷ باید جدی گرفته شود.
