# دیکشنری/شناسنامه‌ی داده (Data Manifest)

هر فایل خام که وارد `data/raw/` می‌شود، پیش از استفاده در هر نوت‌بوک یا اسکریپت باید در این جدول ثبت شود: هش SHA-256 آن (برای اثبات یکسان‌بودن snapshot، مستقل از DVC) و pointer داخل DVC (برای بازیابی). طبق بند 0.3 WBS: **هر snapshot داده یک شناسه دارد که در گزارش قابل ذکر است.**

| فایل | تاریخ افزوده‌شدن | SHA-256 | DVC pointer | توضیح |
|---|---|---|---|---|
| `data/raw/raw_data_14020901_14030301.csv` | 2026-08-11 | `c4b8b3b027a944ebaf8f76c5afa5a2d45e3eec82aceb3de44303c1ad6ea9efc4` | `data/raw/raw_data_14020901_14030301.csv.dvc` (md5: `4cce1bff71271d9a03a3bd59aa2cf2b2`) | داده‌ی خام اصلی از سامانه‌ی تغذیه‌ی دانشگاه تهران، آذر ۱۴۰۲ تا خرداد ۱۴۰۳ (بند ۳-۱ سند مسئله)، ۱۵٬۹۶۸ رکورد |
| `data/raw/UT_calender_1402_1043.pdf` | 2026-08-11 | `e650cd7eb2f984e6aae42b9133e087668abf3c6a11b14ee0b9cf28a2df136102` | `data/raw/UT_calender_1402_1043.pdf.dvc` (md5: `db57e315e29a1fc22b1752cece728a68`) | تقویم آموزشی نیمسال اول ۱۴۰۲-۱۴۰۳ دانشگاه تهران (کلیه مقاطع)، مصوب معاونت آموزشی — منبع فیچرهای تقویم آموزشی بند ۲-۲-۴ WBS (شروع/پایان ترم، میان‌ترم، حذف‌واضافه، امتحانات) |
| `data/raw/per_person_raw_data/ReserveAzar-1402.xlsx` | 2026-08-12 | `707e24c9779ffdfb993ffffab5b1e686e7cc1b69db2c1db35663ed6fec28a4a9` | `data/raw/per_person_raw_data.dvc` (dir md5: `daf2f6fd583b26398340081ee6b99795.dir`) | داده‌ی خام سطح رزرو فردی (`PersonId`)، آذر ۱۴۰۲ — منبع مدل B (بند ۳-۱-ب سند مسئله) |
| `data/raw/per_person_raw_data/ReserveBaham1402.xlsx` | 2026-08-12 | `be9e466720b7de7061999553f4b2851efe4189c76fc8d627f448be980f280e99` | `data/raw/per_person_raw_data.dvc` (dir md5: `daf2f6fd583b26398340081ee6b99795.dir`) | داده‌ی خام سطح رزرو فردی، بهمن ۱۴۰۲ |
| `data/raw/per_person_raw_data/ReserveDay1402-part1.xlsx` | 2026-08-12 | `13429ff6a2d892f57c436050e7a6d2b484fba7ed5ebff19d1940f76273eb86a9` | `data/raw/per_person_raw_data.dvc` (dir md5: `daf2f6fd583b26398340081ee6b99795.dir`) | داده‌ی خام سطح رزرو فردی، دی ۱۴۰۲. **⚠️ اصلاحیه‌ی فاز ۳ (decision_log #۱۸):** برخلاف نامش «بخش ۱»، این فایل و `part2` دو نیمه نیستند — `DataFrame.equals()` نشان داد کاملاً یکسانند (۴۳۷٬۴۷۰ ردیف). فقط همین فایل (`part1`) در پایپ‌لاین استفاده می‌شود |
| `data/raw/per_person_raw_data/ReserveDay1402-part2.xlsx` | 2026-08-12 | `6e70aeb32e29c12450b5e0e6814f736655656b3ebf86c34983e2e94abf4783c3` | `data/raw/per_person_raw_data.dvc` (dir md5: `daf2f6fd583b26398340081ee6b99795.dir`) | **⚠️ کنار گذاشته‌شده از پایپ‌لاین فاز ۳ (decision_log #۱۸):** بایت‌به‌بایت یکسان با `ReserveDay1402-part1.xlsx` (تأیید‌شده با `DataFrame.equals()`)، نه نیمه‌ی دوم دی‌ماه؛ بارگذاری هر دو، دی‌ماه را دوبار می‌شمارد. فایل طبق قاعده‌ی فقط‌خواندنی‌بودن `data/raw/` حذف فیزیکی نشده، فقط در `src/data/clean_individual.py::EXCLUDED_INDIVIDUAL_FILES` نادیده گرفته می‌شود |
| `data/raw/per_person_raw_data/ReserveEsfand1402.xlsx` | 2026-08-12 | `4ae8be2001b833d5b5a3c08be5be42e9304baa7922713898d405377f6be61ef8` | `data/raw/per_person_raw_data.dvc` (dir md5: `daf2f6fd583b26398340081ee6b99795.dir`) | داده‌ی خام سطح رزرو فردی، اسفند ۱۴۰۲ |
| `data/raw/per_person_raw_data/ReserveFarvardin-1403.xlsx` | 2026-08-12 | `71c002559ff5a69696c417a8d26ec9cce75e7d5345fdfbabebbe342a515303e6` | `data/raw/per_person_raw_data.dvc` (dir md5: `daf2f6fd583b26398340081ee6b99795.dir`) | داده‌ی خام سطح رزرو فردی، فروردین ۱۴۰۳ |
| `data/raw/per_person_raw_data/ReserveOrdibehesht-1403.xlsx` | 2026-08-12 | `846eb4e5afe96e38722e8ac652319ca4668ba02a5f7d27be73d6f9b44df1a487` | `data/raw/per_person_raw_data.dvc` (dir md5: `daf2f6fd583b26398340081ee6b99795.dir`) | داده‌ی خام سطح رزرو فردی، اردیبهشت ۱۴۰۳ |
| `data/raw/per_person_raw_data/dailysell/DailySellAzar-1402.xlsx` | 2026-08-12 | `8c1fa49320e7e9af3e8ac9e05eded3a45ff6fba440413b1b4ce2f746b1cd87a8` | `data/raw/per_person_raw_data.dvc` (dir md5: `daf2f6fd583b26398340081ee6b99795.dir`) | فروش روزانه (دیده‌بان مکمل، غیر رزرو)، آذر ۱۴۰۲ |
| `data/raw/per_person_raw_data/dailysell/DailySellBahman1402.xlsx` | 2026-08-12 | `19b832e9947df5d2d32b551fbe9355471b84b44f47410b7419213f6765557206` | `data/raw/per_person_raw_data.dvc` (dir md5: `daf2f6fd583b26398340081ee6b99795.dir`) | فروش روزانه، بهمن ۱۴۰۲ |
| `data/raw/per_person_raw_data/dailysell/DailySellBahman1402(1).xlsx` | 2026-08-12 | `35f685fe2579a1173e4cfceb20c164a019f5ea4888d83dab72bb0e39bb447c2e` | `data/raw/per_person_raw_data.dvc` (dir md5: `daf2f6fd583b26398340081ee6b99795.dir`) | فروش روزانه، بهمن ۱۴۰۲ — **⚠️ حذف‌شده از هر snapshot پردازش‌شده (decision_log #۱۸):** نسخه‌ی تکراری تأییدشده‌ی `DailySellBahman1402.xlsx` (کلید/مجموع `Count` صد‌درصد یکسان). ضمناً کل زیرپوشه‌ی `dailysell/` در فاز ۳ از هر دو پایپ‌لاین کنار گذاشته شد چون فاقد `PersonId`/`Reserveid` است |
| `data/raw/per_person_raw_data/dailysell/DailySellDay1402.xlsx` | 2026-08-12 | `77b30daf695612d36164b5ebede0c577ee5b89a7cb90c4a7a1af77218afffe32` | `data/raw/per_person_raw_data.dvc` (dir md5: `daf2f6fd583b26398340081ee6b99795.dir`) | فروش روزانه، دی ۱۴۰۲ |
| `data/raw/per_person_raw_data/dailysell/esfand1402_rozfrosh.xlsx` | 2026-08-12 | `e7254646e6a782a2c67c51d761dbeb813c1530f9446162f052e34d8e6129e3ad` | `data/raw/per_person_raw_data.dvc` (dir md5: `daf2f6fd583b26398340081ee6b99795.dir`) | فروش روزانه، اسفند ۱۴۰۲ |
| `data/raw/per_person_raw_data/dailysell/DailySellFarvardin-1403.xlsx` | 2026-08-12 | `ff210165e1d929807ce6c1e0b8ea2a50c5e49b4dd137755fc33ca775ea18f797` | `data/raw/per_person_raw_data.dvc` (dir md5: `daf2f6fd583b26398340081ee6b99795.dir`) | فروش روزانه، فروردین ۱۴۰۳ |
| `data/raw/per_person_raw_data/dailysell/DailySellOrdibehesht-1403.xlsx` | 2026-08-12 | `6a1058e9f3e880a59a6944c63b393867f082a858d70c1c0a698dc41712c3fb74` | `data/raw/per_person_raw_data.dvc` (dir md5: `daf2f6fd583b26398340081ee6b99795.dir`) | فروش روزانه، اردیبهشت ۱۴۰۳ |
| `data/external/weather_aqi_tehran.csv` | 2026-08-12 | `9183fb947744f088a515b6b73b5ab535b090a187ffe89f8ec09f009b82e6afe4` | `src/data/weather.py` (استخراج خودکار از APIهای Open-Meteo) | داده‌های روزانه دما، بارش، برف، باد، PM2.5, PM10 و AQI تهران (پردیس مرکزی دانشگاه تهران $35.705^\circ N, 51.396^\circ E$)، بازه ۲۰۲۳-۱۱-۰۱ تا ۲۰۲۴-۰۶-۳۰ (پوشش آذر ۱۴۰۲ تا خرداد ۱۴۰۳) |
| `data/external/events.csv` | 2026-08-13 | `de6feab694d29aa0ff75440db3bb7faa1dbd9a6df5cc4feaa55dacfb0dc9951b` | `src/data/events.py` (اسکریپت بازتولیدپذیر؛ ردیف‌های خبری دستی + ردیف‌های مشتق از `weather_aqi_tehran.csv` + برچسب‌گذاری خودکار اثر از فایل تجمیعی) | آرشیو رویدادهای بافتاری بازه‌ی قفل‌شده‌ی داده (۲۰۲۳-۱۱-۲۲ تا ۲۰۲۴-۰۵-۲۱): **۳۹ رویداد** با پوشش هر ۶ ماه شمسی — ماه رمضان ۱۴۴۵ و عید فطر (توقف ساختاری ۲۹ روزه‌ی سرو ناهار و ۲ روزه‌ی سرو شام)، بلوک نوروز، انفجار کرمان و روز عزای عمومی، سقوط بالگرد رئیس‌جمهور و ۵ روز سوگواری ملی، هر دو مرحله‌ی انتخابات مجلس دوازدهم، ۴ رویداد امنیتی ایران–اسرائیل، ۱۰ بازی تیم ملی (جام ملت‌های آسیا، مقدماتی جام جهانی، دوستانه) و ۲ دربی تهران، یک اعتصاب اتوبوسرانی، و ۱۴ دوره‌ی آلودگی هوا/بارش برف شدید مشتق از داده‌ی Open-Meteo پروژه (آستانه‌ها: AQI حداکثر ≥۱۵۵ برای ≥۲ روز پیوسته؛ برف ≥۱ سانتی‌متر). ستون‌ها: `event_id, date_start, date_end, date_jalali_start, date_jalali_end, event_type, scope, affected_meal, service_status, confidence, description, expected_effect, observed_in_data, source_url`. `affected_meal` = فرضیه‌ی دستی؛ `service_status` (سرو_کامل/سرو_ناقص/بدون_سرو) و `observed_in_data` ($\rho$ واقعی و z آن نسبت به میانه‌ی همان روزِ هفته) هر دو از داده مشتق می‌شوند و دستی نوشته نمی‌شوند. **⚠️ طبق بند ۲-۴ WBS این فایل هرگز نباید فیچر یادگیری مدل شود** (هر رویداد یکتا فقط ۱ مشاهده در ۱۴۲ روز داده دارد ⇒ overfit قطعی)؛ نقش آن فقط توضیح داده‌های پرت در فاز ۴ است. تنها استثنا `ماه_رمضان` است که یک رژیم ساختاری است نه رویداد نقطه‌ای و فیچر `is_ramadan` آن به جدول تقویم تعلق دارد (بند ۵-۱ WBS). جزئیات روش و دلیل کنارگذاشتن آرشیو خبری برای تعطیلی‌های آلودگی در `doc/decision_log.md` ردیف ۲۰ |
| `data/external/calendar_tehran.csv` | 2026-08-13 | `334593211f48945a457a9505c87de6993b90461547f9b7a3a7f250f2990337ff` | `src/data/calendar.py` (استخراج خودکار: پکیج `holidays` + تقویم آموزشی از PDF) | تقویم روزانه‌ی ۲۰۲۳-۱۱-۰۱ تا ۲۰۲۴-۰۶-۳۰ (۲۴۳ ردیف) — تبدیل شمسی/میلادی، تعطیلات ملی/مذهبی (پکیج `holidays`، ایران)، بلوک نوروز و تعطیلات بین‌ترمی (استخراج از `data/raw/UT_calender_1402_1043.pdf`)، فیچرهای مشتق (`days_to_next_holiday`, `is_bridge_day`, `holiday_block_length`, ...) و فیچرهای تقویم آموزشی (`semester`, `week_of_semester`, `is_exam_period`, ...) طبق بند ۲-۲ WBS. **اعتبارسنجی:** نرخ تطابق `day_of_week` مستقل با ستون `DayOfWeek` فایل تجمیعی = ۱٫۰۰؛ هر ۴۰ روز غایب فایل تجمیعی توسط `is_holiday_any` (۳۹ روز) یا `is_bridge_day` (۱ روز) توضیح داده می‌شود — صفر شکاف توضیح‌نیافته. جزئیات در `doc/decision_log.md` ردیف ۱۵ |
| `data/external/restaurant_mapping.csv` | 2026-08-13 | `0f70a48330034978044045a77c43e0fb1d97d2bf700202c4d96a7c243a5289cf` | `src/data/mapping.py::build_restaurant_mapping` (نرمال‌سازی متن + تطبیق فازی `rapidfuzz` + بازبینی دستی کامل) | نگاشت ۷۳ نام خام `RestaurantName` فایل فردی → ۳۱ نام کانونیک فایل تجمیعی (بند ۳-۳ WBS)؛ ستون‌های `raw_name, canonical_name, match_score, method`. جزئیات روش/دو خطای بازبینی دستی در `doc/decision_log.md` ردیف ۱۶ |
| `data/external/dorm_mapping.csv` | 2026-08-13 | `94f8be1e54888475de53d01a99d761715cacaffb5b3ba5f44e1505d377114a84` | `src/data/mapping.py::build_dorm_mapping` (نرمال‌سازی متن + بررسی خودهمبستگی فازی) | نگاشت ۷۱ نام خام `GroupName` (خوابگاه/گروه فردی) — پس از بررسی داده‌محور، نگاشت نهایی همانی (identity) است، جزئیات در `doc/decision_log.md` ردیف ۱۶ |
| `data/external/food_mapping.csv` | 2026-08-13 | `8b9ec247b6eec505ae69d942d6da78ba4126d7b38a483b166c47844b40f6758a` | `src/data/mapping.py::build_food_mapping` (تجزیه‌ی `+`/`،`/پرانتز + تطبیق فازی) | نگاشت ۶۸۲ نام خام `FoodName` فایل فردی → غذای اصلی/کانونیک + `has_extras`/`extras_list` (بند ۳-۴ WBS)؛ ۶۵۲ مورد (۹۵٫۶٪) با امتیاز≥۹۰ به یکی از ۷۸ غذای کانونیک فایل تجمیعی نگاشته شد، ۳۰ مورد بدون نگاشت اجباری ماندند (صبحانه/نسخه‌ی سویا/پیتزا و... که واقعاً معادلی ندارند). جزئیات در `doc/decision_log.md` ردیف ۱۶ |
| `data/processed/dataset_v1.csv` | 2026-08-13 | `f6b939a813bfc6ad0df1239f519fe78c20cf2e3b303642ded3715e40e4b9e793` | `data/processed/dataset_v1.csv.dvc` (md5: `c376e7871e195232b7251cf35cb7310c`) — بازتولید با `src/data/make_dataset.py` | **snapshot قفل‌شده‌ی مدل A (بند ۳-۱۱ WBS، معیار پایان M1).** ۷٬۵۹۱ ردیف در سطح $(d,m,r,f)$، بازه‌ی ۱۴۰۲-۰۹-۰۱ تا ۱۴۰۳-۰۳-۰۱؛ از فایل تجمیعی خام با نرمال‌سازی متن، حذف ۲٬۸۸۳ ردیف کاملاً تکراری، تست‌های T1-T5 (صفر نقض)، تجمیع روی جنسیت و ستون `is_served`. جزئیات در `doc/decision_log.md` ردیف ۱۷ |
| `data/interim/coverage_grid_v1.csv` | 2026-08-13 | `cd0c1d1943e24db36a2a3e584bbdfcf57920608ed79c7627ae5dda7942d498ce` | `src/data/clean.py::build_full_grid` (فراخوانی‌شده در `make_dataset.py`) | حاصل‌ضرب دکارتی کامل (تاریخ×سلف×وعده×غذا) — ۹۰۸٬۵۴۴ ردیف — با ستون `absence_reason` ∈ {served, holiday, restaurant_meal_not_offered, restaurant_closed_that_day, food_not_on_menu_that_day} (بند ۳-۱۰ WBS، گزارش پوشش). داده‌ی میانی/مشتق حجیم (۹۱ مگابایت)، نه بخشی از snapshot قفل‌شده‌ی نهایی — **نه در git نه در DVC** (در `.gitignore`، `data/interim/*`)، قابل‌بازتولید کامل و ارزان با `python -m src.data.make_dataset` |
| `data/processed/person_dim_v1.csv` | 2026-08-13 | `50c53a2886b999307ae4c0d4a7a019990f653eaaa9a53c2fcda177974cbe9929` | `data/processed/person_dim_v1.csv.dvc` (md5: `f703c4f2e6b68a039143f54615b1b3a4`) — بازتولید با `src/data/make_dataset_individual.py` | **snapshot قفل‌شده‌ی بُعد فرد — مدل B (بند ۳-۱۲ WBS).** ۲۶٬۷۷۲ ردیف، یک به ازای هر `PersonId`، آخرین مقدار شناخته‌شده‌ی ویژگی‌های جمعیتی (دانشکده/رشته/مقطع/خوابگاه/جنسیت). جزئیات در `doc/decision_log.md` ردیف ۱۹ |
| `data/processed/person_reservation_fact_v1.csv` | 2026-08-13 | `ee8c1789380cae9735605a2ba0613f93c9e587a665e9c843b0b2f2b2581b24ad` | `data/processed/person_reservation_fact_v1.csv.dvc` (md5: `60251a84bc2e4e77f2069c0df3a074e5`) — بازتولید با `src/data/make_dataset_individual.py` | **⚠️ ۶۸۴ مگابایت — به همین دلیل DVC (نه git مستقیم) برای این فایل الزامی بود.** snapshot قفل‌شده‌ی رویداد رزرو — مدل B (بند ۳-۱۲ WBS، معیار پایان M1).** ۲٬۱۲۷٬۴۴۹ ردیف (۶ فایل خام، `ReserveDay1402-part2.xlsx` کنار گذاشته‌شده)، کلید یکتا (`Reserveid`,`source_file`,`FoodName`)؛ شامل `dont_receive` (نگاشت آماری `ReserveStatus`، نرخ کلی ۸٫۲٪) و `restaurant_canonical`/`food_canonical`/`has_extras`. جزئیات در `doc/decision_log.md` ردیف ۱۹ |

> **یادداشت snapshot سطح فردی:** ۱۴ فایل بالا با یک `dvc add data/raw/per_person_raw_data` به‌عنوان یک snapshot واحد (پوشه) اضافه شده‌اند؛ به همین دلیل همه یک DVC pointer مشترک (سطح دایرکتوری) دارند اما هش SHA-256 هرکدام مستقل ثبت شده تا صحت تک‌تک فایل‌ها قابل اثبات باشد. زیرپوشه‌ی `dailysell/` گزارش فروش روزانه است (نه رزرو)، از منبع/گرانولاریت متفاوت با فایل‌های `Reserve*` — پیش از استفاده باید مشخص شود این دو منبع مکمل‌اند یا هم‌پوشان (بند ۲-۱-۴ WBS). **به‌روزرسانی فاز ۳ (decision_log #۱۸):** `DailySellBahman1402(1).xlsx` رسماً تکراری تشخیص داده شد و کنار گذاشته شد؛ کل `dailysell/` (فاقد `PersonId`/`Reserveid`) از پایپ‌لاین‌های `make_dataset.py`/`make_dataset_individual.py` خارج ماند — هیچ‌کدام حذف فیزیکی نشدند (طبق قاعده‌ی فقط‌خواندنی‌بودن `data/raw/`)، فقط نادیده گرفته می‌شوند.

## نحوه‌ی افزودن یک فایل خام جدید
```bash
dvc add data/raw/<file>
sha256sum data/raw/<file>
git add data/raw/<file>.dvc data/raw/.gitignore
```
سپس یک ردیف جدید به جدول بالا اضافه کنید و در `doc/decision_log.md` اگر منبع/بازه‌ی زمانی جدید است ثبت کنید.

## بازتولید یک snapshot
```bash
dvc pull   # با استفاده از remote تنظیم‌شده (اگر بعداً اضافه شود)
sha256sum data/raw/<file>   # باید با مقدار ثبت‌شده در جدول بالا یکی باشد
```
> **وضعیت فعلی:** هنوز remote برای DVC تنظیم نشده؛ cache فقط محلی است (`.dvc/cache`). این تصمیم در `doc/decision_log.md` ثبت شده و باید پیش از هر جابه‌جایی/اشتراک‌گذاری پروژه به ماشین دیگر بازنگری شود.

## بازرسی یکپارچگی (بند ۲.۱.۱ WBS)

نتایج قابل‌بازتولید با `python -m src.data.inspect_raw` (`src/data/inspect_raw.py`).

### `raw_data_14020901_14030301.csv` (فایل تجمیعی)
- **تعداد ردیف:** ۱۵٬۹۶۸ | **تعداد ستون:** ۲۰
- **بازه‌ی تاریخ:** ۱۴۰۲-۰۹-۰۱ تا ۱۴۰۳-۰۳-۰۱ (شمسی) / ۲۰۲۳-۱۱-۲۲ تا ۲۰۲۴-۰۵-۲۱ (میلادی) — منطبق با نام فایل
- **کدگذاری:** UTF-8 | **جداکننده:** کاما (`,`)
- **مقدار گمشده:** صفر در تمام ۲۰ ستون
- **تست‌های صحت (بند ۳-۶ WBS):** T1 (Reservation=Card+Code+DontReceive) ✅ صفر نقض | T3 (DontReceive≤Reservation) ✅ صفر نقض | T4 (Reservation>0) ✅ صفر نقض
- **⚠️ یافته‌ی جدید — T5 (کلید تکراری (d,m,r,f,g)):** ۲٬۸۸۳ نقض، که همگی دقیقاً معادل ۲٬۸۸۳ ردیف **کاملاً تکراری** (تمام ستون‌ها یکسان) هستند. تصمیم حذف/نگه‌داشت این ردیف‌ها **در همین بند گرفته نمی‌شود** — طبق بند ۳-۸ WBS («تکرار کامل → حذف») به فاز ۳ موکول می‌شود؛ در اینجا فقط وجود و اندازه‌ی آن ثبت می‌شود تا در قفل نسخه‌ی فاز ۳ (بند ۳-۱۱) بی‌صدا از دست نرود.
- **DayOfWeek:** تأیید شد قرارداد شنبه=۰ … جمعه=۶ است (نرخ تطابق با روز هفته‌ی گرگوری واقعی = ۱.۰۰۰).

### فایل‌های سطح فردی (`per_person_raw_data/`)
- کدگذاری/ساختار: `.xlsx` (openpyxl)، بدون جداکننده (باینری اکسل)؛ ستون‌های متنی چند مقدار حاوی نویز `_x000D_\n` (Windows carriage-return باقی‌مانده از خروجی اکسل) هستند — طبق بند ۳-۱۲ WBS پاکسازی آن به فاز ۳ موکول است، اینجا فقط وجودش تأیید می‌شود.
- تعداد ردیف و بازه‌ی تاریخ هر فایل: در `doc/data_dictionary_individual.md` (بند ۲.۱.۴) ثبت شده — حجم کل ~۲.۵۶ میلیون ردیف در ۷ فایل ماهانه.

---

## snapshot نسخه‌ی ۲ (اصلاحیه‌ی شروع فاز ۴) — ۲۰۲۶-۰۸-۱۳

نسخه‌ی ۱ (قفل‌شده در M1) با دو اصلاح دامنه‌ای باطل شد؛ نسخه‌ی ۲ جایگزین آن در تمام تحلیل‌های فاز ۴ به بعد است. نسخه‌ی ۱ حذف نمی‌شود (ردیابی‌پذیری)، ولی **نباید مبنای هیچ تحلیل جدیدی باشد**. جزئیات چرایی: ردیف‌های ۲۱، ۲۲ و ۲۳ `doc/decision_log.md`.

| فایل | ردیف | تغییر نسبت به v1 |
|---|---|---|
| `data/processed/dataset_v2.csv` | ۷٬۵۷۹ | −۱۲ ردیف (حذف «دوره حوزه علوم انسانی»)؛ +۴ ستون `city`, `province`, `is_tehran`, `distance_km_to_tehran_campus` |
| `data/processed/person_dim_v2.csv` | ۲۶٬۷۷۲ | +۱ ستون `is_dorm_resident` |
| `data/processed/person_reservation_fact_v2.csv` | ۲٬۱۲۷٬۲۴۰ | −۲۰۹ ردیف (همان حذف)؛ +۲ ستون `city`, `is_tehran` |
| `data/interim/coverage_grid_v2.csv` | — | بازتولید متناظر |
| `data/external/campus_geo.csv` | ۳۰ | **جدید** — نگاشت سلف → شهر/استان/مختصات |
| `data/external/excluded_restaurants.csv` | ۲ | **جدید** — فهرست صریح سلف‌های کنارگذاشته‌شده |
| `data/external/weather_aqi_by_city.csv` | ۱٬۴۵۸ | **جدید** — هواشناسی/AQI روزانه‌ی ۶ شهر (جایگزین `weather_aqi_tehran.csv` برای الحاق) |

**بازتولید:**
```
python -m src.data.campus_geo
python -m src.data.weather --by-city
python -m src.data.make_dataset
python -m src.data.make_dataset_individual
```

**⚠️ قاعده‌ی الحاق هواشناسی:** از این پس کلید الحاق `(city, date_gregorian)` است، نه `date_gregorian` تنها.

### اصلاحیه‌ی نسخه‌ی ۳ فایل‌های سطح فردی — ۲۰۲۶-۰۸-۱۳ (دور ۲ فاز ۴)

دو نقص داده‌ای که در دور ۲ فاز ۴ کشف شد (ردیف ۲۷ `doc/decision_log.md`) رفع شد. **`dataset_v2.csv` (تجمیعی) بدون تغییر می‌ماند** — منبعش فایل خام دیگری است و هیچ‌کدام از دو نقص را ندارد.

| فایل | ردیف | تغییر نسبت به v2 |
|---|---|---|
| `data/processed/person_reservation_fact_v3.csv` | ۲٬۱۱۰٬۳۹۷ | −۱۶٬۸۴۳ ردیف (تکرار بین‌فایلی روز ۱۴۰۳-۰۲-۰۱)؛ +۱ ستون `is_main_meal` |
| `data/processed/person_dim_v3.csv` | ۲۶٬۷۷۲ | بدون تغییر ساختاری |

**دامنه‌ی تحلیل مدل B:** فقط ردیف‌های `is_main_meal=True` (ناهار/شام) = ۲٬۰۴۹٬۳۲۲ ردیف. صبحانه و سحری (۶۱٬۰۷۳ ردیف، ۲.۹٪) خارج از واحد تحلیل پروژه‌اند.

**⚠️ نسخه‌های v1 و v2 فایل‌های فردی نباید مبنای هیچ تحلیل جدیدی باشند.**

---

## قفل فاز ۷ — `cv_folds.json` (دروازه‌ی انصاف A1، بند 7.7.3 `doc/WBS-phase7-modeling.md`)

| فایل | تاریخ افزوده‌شدن | SHA-256 | ردیابی | توضیح |
|---|---|---|---|---|
| `data/processed/cv_folds.json` | 2026-08-14 | `bd08d6f7c801ee0611121e404774251de07a480ac1589b12eb7c64f8044b78d4` | git مستقیم (فایل کوچک، مثل `feature_sets_v1.json`) — بازتولید با `python -m src.cv` | ۵ fold walk-forward گسترشی، از ۱۴۲ تاریخ یکتای `features_A_v1.parquet` (`n_folds=5, min_train_days=60`). **منجمد است** — `src/cv.py::main()` اگر فایل موجود باشد بازنویسی نمی‌کند. |

### هش snapshot داده‌ی هر سطح — `data_snapshot_hash` (بند 7.7.2)

این دو هش، مقدار param اجباری `data_snapshot_hash` هر MLflow run فاز ۷ هستند و در سلول ۳ هر نوت‌بوک GPU (بند 7.8.2) به‌صورت `assert` بررسی می‌شوند — چون نوت‌بوک کولب داده را از Drive می‌خواند، نه از DVC، و بدون این assert هیچ اثباتی وجود ندارد که همان فایل مخزن آپلود شده است.

| فایل | سطح داده | SHA-256 | مصرف‌کننده |
|---|---|---|---|
| `data/processed/features_A_v1.parquet` | L1 (سلول $(d,m,r,f)$، ۷٬۵۷۹ ردیف) | `68b4cb8517d292599b2f161f779758b9f3254d60302849f39d81650d0bd9fba0` | خ۱، خ۲، خ۶، خ۷-الف، خ۸، خ۹، خ۱۰، خ۱۱ |
| `data/processed/person_features_v1.parquet` | L5 (رزرو فردی، ۲٬۰۴۹٬۳۲۲ ردیف) | `90097e5f3b7d4572ee94c9a1a09ae0d0ec65115c413445a4751e20c48ac35de0` | خ۷-ب (سطح فرد)، فیچرهای کوهورت اسپرینت B |
| `data/processed/feature_sets_v1.json` | — (تعریف فیچرست‌ها) | `0c7d6a59a8e596681266d9a165ad2413cd09aed43fddfc4bc3ad9bf0a550cc3e` | همه |

⚠️ **هر run آموزش مدل فاز ۷ (هر خانواده، هر سطح داده) باید foldهایش را از همین فایل بخواند** (`src.cv.load_cv_folds()`) و `cv_folds_hash` بازگشتی را به‌عنوان param در MLflow ثبت کند — نه اینکه `WalkForwardSplitter` را مستقیم و دوباره صدا بزند. اگر داده یا پروتکل CV تغییر کرد و این فایل عمداً بازتولید شد، هش جدید باید هم اینجا و هم در تمام ارجاعات (`AGENTS.md`) به‌روزرسانی شود، و **تمام runهای قبلی با هش قدیمی از جدول مقایسه‌ی فاز ۷ حذف می‌شوند** (بند 7.7.3).

---

## snapshot نسخه‌ی ۴ (استخراج تازه‌ی سطح فرد) — ۲۰۲۶-۰۹-۰۵

استخراج تازه‌ی سامانه‌ی تغذیه، دریافت‌شده در ۲۰۲۶-۰۹-۱۷. **هنوز به هیچ پایپ‌لاینی وصل
نیست و هیچ snapshot پردازش‌شده‌ای از آن ساخته نشده** — ثبت اینجا طبق قاعده‌ی «هر فایل خام
پیش از استفاده باید ثبت شود» انجام شده، نه به‌معنای ورود آن به مطالعه. برنامه‌ی کار:
`doc/WBS-phase13-data-expansion.md`.

| فایل | تاریخ افزوده‌شدن | SHA-256 | DVC pointer | توضیح |
|---|---|---|---|---|
| `data/raw/StudentReserve.rar` | 2026-09-17 | `aa31001417b4c8de4bdc3f43d8f4f080a588eb24dd9028e04fc022edefc8fb70` | `data/raw/StudentReserve.rar.dvc` (md5: `3a339b36f719af40c8e7cf715efe20da`) | آرشیو حاوی `StudentReserve.parquet` (۳۲۳٬۱۷۴٬۵۸۶ بایت، ۸٬۵۶۷٬۲۶۵ ردیف، ۲۹ ستون، ۸۶ row group). **نسخه‌ی کانونیک پیشنهادی.** اسکیما دقیقاً همان فایل‌های `Reserve*.xlsx` سطح فرد. بازه: ۲۰۲۳-۰۹-۲۵ تا ۲۰۲۶-۰۲-۲۰ (۶۴۰ روز فعال، سال‌های شمسی ۱۴۰۲ تا ۱۴۰۴)، ۴۶٬۶۵۹ فرد، ۱۲۷ سلف، ۱٬۷۶۲ نام غذا |
| `data/raw/StudentReserve.parquet` (استخراج‌شده، **ورودی پایپ‌لاین**) | 2026-09-17 | `91508f9e3d787b26be9f6737d24c97259c217b874a89bfdfba7cda3408a2ee4a` | `data/raw/StudentReserve.parquet.dvc` (md5: `9d2aace7d2462056b4bc7dff435bfc1d`) | هش فایل استخراج‌شده، برای `assert` سلول ۳ نوت‌بوک‌های GPU (بند ۷-۸-۲) |
| `data/raw/parquet.rar` | 2026-09-17 | `f8ce8288c7bd94406fa1f98ce64e9f5362df0c436bcd02497635522d4f98e6d9` | `data/raw/parquet.rar.dvc` (md5: `d5a03dc1a65d039d5f1db5dde4804061`) | ۸۶ قطعه `parquet_output/part_0001..0086.parquet` (مجموع ۳۲۴٬۰۹۵٬۰۱۲ بایت). **⚠️ داده‌ی تکراری — همان ۸٬۵۶۷٬۲۶۵ ردیف `StudentReserve.parquet` با ترتیب ردیف متفاوت** (یافته F87: چندمجموعه‌ی `Reserveid`/`PersonId`/`mealid` در دو آرشیو یکسان است). بارگذاری هم‌زمان هر دو کل داده را دو بار می‌شمارد — همان تله‌ی `ReserveDay1402-part2.xlsx` در ردیف ۱۸ `decision_log`. طبق قاعده‌ی فقط‌خواندنی‌بودن `data/raw/` حذف فیزیکی نمی‌شود؛ قطعه‌ها برای اجرای حافظه‌محدود (کولب/کگل) مفیدند |

### تقویم آموزشی — سه PDF تازه (۲۰۲۶-۰۹-۱۷)

تنها قلم بیرونی‌ای که پروژه نمی‌توانست خودش تولید کند (بند ۱۳-۶، «مسدود»). با رسیدن این
سه فایل، `semester`، `week_of_semester` و `is_exam_period` برای هر سه سال تحصیلی ساخته
می‌شوند. استخراج **مختصات‌محور و خودارزیاب** است، نه رونویسی دستی:
`src/data/academic_calendar_pdf.py`.

| فایل | تاریخ افزوده‌شدن | SHA-256 | پوشش |
|---|---|---|---|
| `data/raw/تقویم آموزشی-1403-1404.pdf` | 2026-09-17 | `945c7981198cc03e211865b37e23fc3615b5049a420808461bdc140bede868f8` | هر دو نیمسال ۱۴۰۳-۱۴۰۴ (چیدمان جدولی، رویداد داخل سلول) |
| `data/raw/تقویم_آموزشی.pdf` | 2026-09-17 | `b4c805e2663691bdfa0032e4991b3b46ade47002d64e1c7d794f73973edffc0f` | هر دو نیمسال ۱۴۰۴-۱۴۰۵ (چیدمان ستون رویداد). ⚠️ صفحه‌ی ۲ **جایگزین شده** — سطر زیر |
| `data/raw/اصلاحیه_سوم__تقویم_دانشگاهی_نیمسال_2-1404 (1).pdf` | 2026-09-17 | `0f4acbd6c718efadbc5675412fffd68114b72ebcf743ef752448e6af0479e4b2` | نیمسال دوم ۱۴۰۴-۱۴۰۵، **اصلاحیه‌ی سوم** — مرجع معتبر این نیمسال |

> **⚠️ دو نسخه از یک نیمسال.** صفحه‌ی ۲ `تقویم_آموزشی.pdf` و فایل اصلاحیه هر دو نیمسال
> دوم ۱۴۰۴-۱۴۰۵‌اند و **یکی نیستند**: اصلاحیه کل نیمسال را یک هفته جلو می‌برد (شروع
> کلاس‌ها ۲۵ بهمن → ۲ اسفند). اصلاحیه متأخرتر است (ساخت ۲۰۲۶-۰۶-۲۲ در برابر ۲۰۲۵-۰۵-۰۳)
> و در `CALENDAR_SOURCES` مرجع است؛ نسخه‌ی اول با پرچم `is_superseded` در جدول رویدادها
> نگه داشته می‌شود. این نیمسال **بیرون از پنجره‌ی داده** است (داده تا ۲۰۲۶-۰۲-۲۰).

> **ثبت در git، نه DVC.** برخلاف بقیه‌ی `data/raw/`، این سه فایل مستقیماً در git‌اند
> (روی‌هم ۱٫۴ مگابایت). برای فایل مرجعِ کوچکی که هرگز تغییر نمی‌کند این ساده‌تر است و
> `git clone` تنهایی کافی است — `dvc pull` لازم نیست. `UT_calender_1402_1043.pdf` به
> دلایل تاریخی در DVC مانده.

**خروجی‌های مشتق‌شده:** `data/external/academic_calendar_events_v4.csv` (۱۵۵ رویداد،
۷ صفحه‌ی نیمسال) · `data/external/calendar_tehran_v4.csv` (۹۴۳ روز، ۲۰۲۳-۰۹-۰۱ تا
۲۰۲۶-۰۳-۳۱ — هم‌مرز `weather_aqi_by_city_v4.csv`).

> **⚠️ سه نکته پیش از استفاده:** (۱) کدهای `ReserveStatus` در طول این بازه دو بار معنایشان
> جابه‌جا می‌شود — فقط **اجتماع** سه کد عدم‌دریافت پایدار است (بند «قاعده‌ی پایداری هدف»
> `AGENTS.md`، یافته F90). (۲) نگاشت‌های `restaurant_mapping.csv` و `food_mapping.csv`
> فقط ۴۷ از ۱۲۷ سلف و ۴۹۶ از ۱٬۷۶۲ غذا را پوشش می‌دهند (F91). (۳) `calendar_tehran.csv`
> و `weather_aqi_tehran.csv` هرکدام فقط ۲۴۳ روز از ~۸۷۹ روز این بازه را دارند (F92).

## foldهای پنجره‌ی v4 — `data/processed/cv_folds_v4.json`

| کلید | مقدار |
|---|---|
| `cv_folds_hash` | **`941c700896055e72`** |
| محور تقسیم | سال تحصیلی (F103) |
| ردیف کل | ۶۳٬۹۳۱ (سلول L1، **بدون** پنجره‌ی درگیری خرداد ۱۴۰۴) |
| تقسیم‌ها | `in_window` ۱۹٬۸۹۲/۴٬۹۷۴ · `oos_1403-1404` ۲۴٬۸۶۶/۲۳٬۸۶۸ · `oos_1404-1405` ۲۴٬۸۶۶/۱۳٬۱۴۳ |

⚠️ این هش **عمداً** با `cv_folds_hash` فاز ۷ نمی‌خواند — پنجره‌ی داده عوض شده، پس
هیچ‌کدام از ~۱۶۹ پیکربندی فاز ۷ با runهای `phase13_oos` در یک جدول قابل‌مقایسه
نیستند. ناسازگاری واقعی است و ثبتش تنها راهی است که بی‌صدا نماند (بند ۱۳-۱۲ پ).

⚠️ هش نسبت به نخستین اجرای بند ۱۳-۹ **عوض شده**، چون پنجره‌ی درگیری خرداد ۱۴۰۴ از
شبکه حذف شد (ردیف ۸۰، F135). runهای پیش از آن تاریخ با هش قدیمی ثبت شده‌اند و با
جدول فعلی قابل‌مقایسه نیستند.
