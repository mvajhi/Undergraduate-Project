# دیکشنری/شناسنامه‌ی داده (Data Manifest)

هر فایل خام که وارد `data/raw/` می‌شود، پیش از استفاده در هر نوت‌بوک یا اسکریپت باید در این جدول ثبت شود: هش SHA-256 آن (برای اثبات یکسان‌بودن snapshot، مستقل از DVC) و pointer داخل DVC (برای بازیابی). طبق بند 0.3 WBS: **هر snapshot داده یک شناسه دارد که در گزارش قابل ذکر است.**

| فایل | تاریخ افزوده‌شدن | SHA-256 | DVC pointer | توضیح |
|---|---|---|---|---|
| `data/raw/raw_data_14020901_14030301.csv` | 2026-08-11 | `c4b8b3b027a944ebaf8f76c5afa5a2d45e3eec82aceb3de44303c1ad6ea9efc4` | `data/raw/raw_data_14020901_14030301.csv.dvc` (md5: `4cce1bff71271d9a03a3bd59aa2cf2b2`) | داده‌ی خام اصلی از سامانه‌ی تغذیه‌ی دانشگاه تهران، آذر ۱۴۰۲ تا خرداد ۱۴۰۳ (بند ۳-۱ سند مسئله)، ۱۵٬۹۶۸ رکورد |
| `data/raw/UT_calender_1402_1043.pdf` | 2026-08-11 | `e650cd7eb2f984e6aae42b9133e087668abf3c6a11b14ee0b9cf28a2df136102` | `data/raw/UT_calender_1402_1043.pdf.dvc` (md5: `db57e315e29a1fc22b1752cece728a68`) | تقویم آموزشی نیمسال اول ۱۴۰۲-۱۴۰۳ دانشگاه تهران (کلیه مقاطع)، مصوب معاونت آموزشی — منبع فیچرهای تقویم آموزشی بند ۲-۲-۴ WBS (شروع/پایان ترم، میان‌ترم، حذف‌واضافه، امتحانات) |
| `data/raw/per_person_raw_data/ReserveAzar-1402.xlsx` | 2026-08-12 | `707e24c9779ffdfb993ffffab5b1e686e7cc1b69db2c1db35663ed6fec28a4a9` | `data/raw/per_person_raw_data.dvc` (dir md5: `daf2f6fd583b26398340081ee6b99795.dir`) | داده‌ی خام سطح رزرو فردی (`PersonId`)، آذر ۱۴۰۲ — منبع مدل B (بند ۳-۱-ب سند مسئله) |
| `data/raw/per_person_raw_data/ReserveBaham1402.xlsx` | 2026-08-12 | `be9e466720b7de7061999553f4b2851efe4189c76fc8d627f448be980f280e99` | `data/raw/per_person_raw_data.dvc` (dir md5: `daf2f6fd583b26398340081ee6b99795.dir`) | داده‌ی خام سطح رزرو فردی، بهمن ۱۴۰۲ |
| `data/raw/per_person_raw_data/ReserveDay1402-part1.xlsx` | 2026-08-12 | `13429ff6a2d892f57c436050e7a6d2b484fba7ed5ebff19d1940f76273eb86a9` | `data/raw/per_person_raw_data.dvc` (dir md5: `daf2f6fd583b26398340081ee6b99795.dir`) | داده‌ی خام سطح رزرو فردی، دی ۱۴۰۲ (بخش ۱) |
| `data/raw/per_person_raw_data/ReserveDay1402-part2.xlsx` | 2026-08-12 | `6e70aeb32e29c12450b5e0e6814f736655656b3ebf86c34983e2e94abf4783c3` | `data/raw/per_person_raw_data.dvc` (dir md5: `daf2f6fd583b26398340081ee6b99795.dir`) | داده‌ی خام سطح رزرو فردی، دی ۱۴۰۲ (بخش ۲) |
| `data/raw/per_person_raw_data/ReserveEsfand1402.xlsx` | 2026-08-12 | `4ae8be2001b833d5b5a3c08be5be42e9304baa7922713898d405377f6be61ef8` | `data/raw/per_person_raw_data.dvc` (dir md5: `daf2f6fd583b26398340081ee6b99795.dir`) | داده‌ی خام سطح رزرو فردی، اسفند ۱۴۰۲ |
| `data/raw/per_person_raw_data/ReserveFarvardin-1403.xlsx` | 2026-08-12 | `71c002559ff5a69696c417a8d26ec9cce75e7d5345fdfbabebbe342a515303e6` | `data/raw/per_person_raw_data.dvc` (dir md5: `daf2f6fd583b26398340081ee6b99795.dir`) | داده‌ی خام سطح رزرو فردی، فروردین ۱۴۰۳ |
| `data/raw/per_person_raw_data/ReserveOrdibehesht-1403.xlsx` | 2026-08-12 | `846eb4e5afe96e38722e8ac652319ca4668ba02a5f7d27be73d6f9b44df1a487` | `data/raw/per_person_raw_data.dvc` (dir md5: `daf2f6fd583b26398340081ee6b99795.dir`) | داده‌ی خام سطح رزرو فردی، اردیبهشت ۱۴۰۳ |
| `data/raw/per_person_raw_data/dailysell/DailySellAzar-1402.xlsx` | 2026-08-12 | `8c1fa49320e7e9af3e8ac9e05eded3a45ff6fba440413b1b4ce2f746b1cd87a8` | `data/raw/per_person_raw_data.dvc` (dir md5: `daf2f6fd583b26398340081ee6b99795.dir`) | فروش روزانه (دیده‌بان مکمل، غیر رزرو)، آذر ۱۴۰۲ |
| `data/raw/per_person_raw_data/dailysell/DailySellBahman1402.xlsx` | 2026-08-12 | `19b832e9947df5d2d32b551fbe9355471b84b44f47410b7419213f6765557206` | `data/raw/per_person_raw_data.dvc` (dir md5: `daf2f6fd583b26398340081ee6b99795.dir`) | فروش روزانه، بهمن ۱۴۰۲ |
| `data/raw/per_person_raw_data/dailysell/DailySellBahman1402(1).xlsx` | 2026-08-12 | `35f685fe2579a1173e4cfceb20c164a019f5ea4888d83dab72bb0e39bb447c2e` | `data/raw/per_person_raw_data.dvc` (dir md5: `daf2f6fd583b26398340081ee6b99795.dir`) | فروش روزانه، بهمن ۱۴۰۲ — فایل تکمیلی/بازنویسی‌شده؛ نیازمند بررسی هم‌پوشانی با `DailySellBahman1402.xlsx` پیش از استفاده (بند ۲-۱-۴) |
| `data/raw/per_person_raw_data/dailysell/DailySellDay1402.xlsx` | 2026-08-12 | `77b30daf695612d36164b5ebede0c577ee5b89a7cb90c4a7a1af77218afffe32` | `data/raw/per_person_raw_data.dvc` (dir md5: `daf2f6fd583b26398340081ee6b99795.dir`) | فروش روزانه، دی ۱۴۰۲ |
| `data/raw/per_person_raw_data/dailysell/esfand1402_rozfrosh.xlsx` | 2026-08-12 | `e7254646e6a782a2c67c51d761dbeb813c1530f9446162f052e34d8e6129e3ad` | `data/raw/per_person_raw_data.dvc` (dir md5: `daf2f6fd583b26398340081ee6b99795.dir`) | فروش روزانه، اسفند ۱۴۰۲ |
| `data/raw/per_person_raw_data/dailysell/DailySellFarvardin-1403.xlsx` | 2026-08-12 | `ff210165e1d929807ce6c1e0b8ea2a50c5e49b4dd137755fc33ca775ea18f797` | `data/raw/per_person_raw_data.dvc` (dir md5: `daf2f6fd583b26398340081ee6b99795.dir`) | فروش روزانه، فروردین ۱۴۰۳ |
| `data/raw/per_person_raw_data/dailysell/DailySellOrdibehesht-1403.xlsx` | 2026-08-12 | `6a1058e9f3e880a59a6944c63b393867f082a858d70c1c0a698dc41712c3fb74` | `data/raw/per_person_raw_data.dvc` (dir md5: `daf2f6fd583b26398340081ee6b99795.dir`) | فروش روزانه، اردیبهشت ۱۴۰۳ |
| `data/external/weather_aqi_tehran.csv` | 2026-08-12 | `9183fb947744f088a515b6b73b5ab535b090a187ffe89f8ec09f009b82e6afe4` | `src/data/weather.py` (استخراج خودکار از APIهای Open-Meteo) | داده‌های روزانه دما، بارش، برف، باد، PM2.5, PM10 و AQI تهران (پردیس مرکزی دانشگاه تهران $35.705^\circ N, 51.396^\circ E$)، بازه ۲۰۲۳-۱۱-۰۱ تا ۲۰۲۴-۰۶-۳۰ (پوشش آذر ۱۴۰۲ تا خرداد ۱۴۰۳) |

> **یادداشت snapshot سطح فردی:** ۱۴ فایل بالا با یک `dvc add data/raw/per_person_raw_data` به‌عنوان یک snapshot واحد (پوشه) اضافه شده‌اند؛ به همین دلیل همه یک DVC pointer مشترک (سطح دایرکتوری) دارند اما هش SHA-256 هرکدام مستقل ثبت شده تا صحت تک‌تک فایل‌ها قابل اثبات باشد. زیرپوشه‌ی `dailysell/` گزارش فروش روزانه است (نه رزرو)، از منبع/گرانولاریت متفاوت با فایل‌های `Reserve*` — پیش از استفاده باید مشخص شود این دو منبع مکمل‌اند یا هم‌پوشان (بند ۲-۱-۴ WBS). فایل `DailySellBahman1402(1).xlsx` به‌نظر نسخه‌ی دوم/تکراری `DailySellBahman1402.xlsx` است و رفع ابهامش پیش از فاز ۳ لازم است.

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
