"""Build the event/context archive for the study window (WBS بند ۲.۴).

خروجی: `data/external/events.csv` — یک ردیف به‌ازای هر رویداد (تک‌روزه یا چندروزه) که ممکن است
رفتار دریافت غذا را در بازه‌ی ۱۴۰۲-۰۹-۰۱ تا ۱۴۰۳-۰۳-۰۱ (۲۰۲۳-۱۱-۲۲ تا ۲۰۲۴-۰۵-۲۱) توضیح دهد.

⚠️ **این جدول فیچر یادگیری نیست.** با ۱۴۲ روز داده هر رویداد یکتا فقط ۱ مشاهده دارد و ورود آن
به‌عنوان فیچر Overfit قطعی است (هشدار روش‌شناختی بند ۲.۴ WBS). نقش این جدول **توضیح داده‌های پرت
در فاز ۴** است. تنها استثنا `ماه_رمضان` است که یک رژیم ساختاری ۲۹روزه است نه یک رویداد نقطه‌ای —
پیشنهاد فیچرسازی آن (`is_ramadan`) در بند ۵.۱ WBS آمده و جای آن جدول تقویم است، نه این فایل.

منابع ردیف‌ها سه دسته‌اند و ستون `source_url` آن را مشخص می‌کند:
1. **آرشیو خبری** — لینک عمومی قابل‌بازبینی (ویکی‌پدیا، خبرگزاری‌ها).
2. **مشتق از داده‌ی خودمان** (`derived:weather_aqi_tehran.csv`) — دوره‌های آلودگی/برف شدید که
   مستقیماً از `data/external/weather_aqi_tehran.csv` با آستانه‌های صریح زیر استخراج می‌شوند.
   این دسته عمداً از خبر استخراج نشد: جست‌وجوی خبری برای تعطیلی‌های آلودگی ۱۴۰۲ نتایج آلوده به
   سال‌های ۱۴۰۳/۱۴۰۴ برمی‌گرداند (دیدن decision_log ردیف ۱۶) و ثبت تاریخ اشتباه بدتر از نبودِ ردیف است.
3. **تقویم قمری/رسمی** — رمضان و عید فطر، که با شکاف واقعی سرو در داده تأیید شده‌اند.

ستون `observed_in_data` توسط `annotate_observed_effect()` به‌صورت خودکار پر می‌شود و مقدار واقعی
$\rho$ و انحراف آن از میانه‌ی همان روزِ هفته را نشان می‌دهد — یعنی هر ادعای «اثر مورد انتظار»
بلافاصله در کنار شاهد تجربی‌اش قرار می‌گیرد.
"""

import logging

import jdatetime
import pandas as pd

from src.config import DATA_EXTERNAL
from src.data.inspect_raw import load_aggregate

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_PATH = DATA_EXTERNAL / "events.csv"
WEATHER_PATH = DATA_EXTERNAL / "weather_aqi_tehran.csv"

# بازه‌ی داده‌ی سامانه‌ی تغذیه (بند ۲.۱) — رویدادهای خارج از این بازه ثبت نمی‌شوند.
WINDOW_START = "2023-11-22"
WINDOW_END = "2024-05-21"

# آستانه‌های دسته‌ی «مشتق از داده‌ی خودمان»
AQI_UNHEALTHY = 155  # AQI آمریکا: ۱۵۱+ = «ناسالم برای همه»؛ ۱۵۵ حاشیه‌ی اطمینان می‌دهد
AQI_MIN_RUN_DAYS = 2  # فقط دوره‌های پیوسته‌ی ۲ روز به بالا (یک روز تکی نویز است)
HEAVY_SNOW_CM = 1.0  # بارش برف روزانه بر حسب سانتی‌متر

COLUMNS = [
    "event_id",
    "date_start",
    "date_end",
    "date_jalali_start",
    "date_jalali_end",
    "event_type",
    "scope",
    "affected_meal",
    "service_status",
    "confidence",
    "description",
    "expected_effect",
    "observed_in_data",
    "source_url",
]

# `affected_meal` = *فرضیه‌ی* ما درباره‌ی وعده‌ی متأثر (ناهار | شام | هردو) — یک ادعای دستی.
# `service_status` = *واقعیت* استخراج‌شده از داده (سرو_کامل | سرو_ناقص | بدون_سرو) — هرگز دستی
# نوشته نمی‌شود. جداکردن این دو عمدی است: نسخه‌ی قبلی این فایل چند رویداد را دستی «بدون سرویس»
# برچسب زده بود که غلط بود (مثلاً جمعه ۲۹ دی که به‌دلیل ایام امتحانات سرو داشت).


def _jalali(date: str) -> str:
    ts = pd.Timestamp(date)
    return jdatetime.date.fromgregorian(date=ts.date()).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# ۲.۴.۱ رویدادهای دارای منبع خبری/تقویمی
# ---------------------------------------------------------------------------
# مرتب بر اساس تاریخ. `affected_meal` مقادیر مجاز: ناهار | شام | هردو.
# وضعیت واقعی سرو دستی نوشته نمی‌شود — ستون مشتق `service_status` آن را از داده می‌سازد.

MANUAL_EVENTS: list[dict] = [
    {
        "event_id": "derby_102",
        "date_start": "2023-12-14",
        "event_type": "رویداد_ورزشی_باشگاهی",
        "scope": "استان_تهران",
        "affected_meal": "شام",
        "confidence": "متوسط",
        "description": "دربی ۱۰۲ تهران، استقلال - پرسپولیس، ورزشگاه آزادی، پنجشنبه ۲۳ آذر ۱۴۰۲؛ نتیجه: تساوی ۱-۱",
        "expected_effect": "اثر انتظاری ضعیف: پنجشنبه است و حجم رزرو شام پنجشنبه‌ها ذاتاً بسیار کم است (۲۵۶ رزرو)، پس حتی اثر واقعی هم در نویز گم می‌شود",
        "source_url": "https://fa.wikipedia.org/wiki/%D9%81%D9%87%D8%B1%D8%B3%D8%AA_%D8%A8%D8%A7%D8%B2%DB%8C%E2%80%8C%D9%87%D8%A7%DB%8C_%D8%B4%D9%87%D8%B1%D8%A2%D9%88%D8%B1%D8%AF_%D8%AA%D9%87%D8%B1%D8%A7%D9%86",
    },
    {
        "event_id": "kerman_bombing",
        "date_start": "2024-01-03",
        "event_type": "رویداد_امنیتی",
        "scope": "ملی",
        "affected_meal": "هردو",
        "confidence": "بالا",
        "description": "انفجار دوگانه‌ی تروریستی در مراسم سالگرد قاسم سلیمانی، گلزار شهدای کرمان، چهارشنبه ۱۳ دی ۱۴۰۲ ساعت ۱۵:۵۰–۱۶:۰۰؛ ۸۹ کشته و ۲۸۴ زخمی؛ داعش خراسان مسئولیت را پذیرفت",
        "expected_effect": "زمان انفجار (~۱۶:۰۰) پس از پایان بازه‌ی ناهار و پیش از شام است؛ احتمال کاهش حضور در شام همان روز و روز بعد؛ ⚠️ این تاریخ با ردیف `pollution_closure_13dey` هم‌پوشان است و اثر دو رویداد قابل تفکیک نیست",
        "source_url": "https://en.wikipedia.org/wiki/2024_Kerman_bombings",
    },
    {
        "event_id": "pollution_closure_13dey",
        "date_start": "2024-01-03",
        "event_type": "تعطیلی_اضطراری_آلودگی",
        "scope": "استان_تهران",
        "affected_meal": "هردو",
        "confidence": "متوسط",
        "description": "تعطیلی حضوری مدارس تهران (چهارشنبه ۱۳ دی ۱۴۰۲) به‌دلیل آلودگی هوا؛ مشخص نیست کلاس‌های دانشگاهی و ادارات هم مشمول تعطیلی بودند یا فقط مدارس K-12 استان تهران",
        "expected_effect": "احتمال کاهش حضور دانشجویان و افزایش نرخ عدم‌دریافت وعده‌های همان روز؛ در صورت تعطیلی حضوری کلاس‌های دانشگاه اثر به‌مراتب قوی‌تر خواهد بود؛ ⚠️ هم‌پوشان با `kerman_bombing`",
        "source_url": "https://www.delgarm.com/%D9%88%D8%B6%D8%B9%DB%8C%D8%AA-%D8%AA%D8%B9%D8%B7%DB%8C%D9%84%DB%8C-%D9%85%D8%AF%D8%A7%D8%B1%D8%B3-%D8%AA%D9%87%D8%B1%D8%A7%D9%86-%DA%86%D9%87%D8%A7%D8%B1%D8%B4%D9%86%D8%A8%D9%87-13-%D8%AF%DB%8C-1402.a306460",
    },
    {
        "event_id": "kerman_mourning_day",
        "date_start": "2024-01-04",
        "event_type": "سوگواری_ملی",
        "scope": "ملی",
        "affected_meal": "هردو",
        "confidence": "بالا",
        "description": "روز عزای عمومی سراسری اعلام‌شده توسط رئیس‌جمهور برای قربانیان انفجار کرمان؛ پنجشنبه ۱۴ دی ۱۴۰۲",
        "expected_effect": "عزای عمومی لزوماً تعطیلی رسمی نیست و سلف‌ها سرو داشتند؛ احتمال کاهش خفیف حضور و لغو مراسم/کلاس‌های فوق‌برنامه",
        "source_url": "https://www.aljazeera.com/news/2024/1/4/confusion-speculation-in-iran-after-twin-blasts-kill-more-than-80-people",
    },
    {
        "event_id": "friendly_burkina_faso",
        "date_start": "2024-01-05",
        "event_type": "رویداد_ورزشی_ملی",
        "scope": "ملی",
        "affected_meal": "شام",
        "confidence": "بالا",
        "description": "بازی دوستانه‌ی تدارکاتی ایران - بورکینافاسو، جزیره‌ی کیش، جمعه ۱۵ دی ۱۴۰۲؛ نتیجه: برد ۲-۱ ایران",
        "expected_effect": "بدون اثر قابل‌سنجش — جمعه‌ی بدون سرو (خارج از ایام امتحانات)؛ فقط برای کامل‌بودن آرشیو بازی‌های تیم ملی ثبت شده",
        "source_url": "https://en.wikipedia.org/wiki/Iran_national_football_team_results_(2020%E2%80%93present)",
    },
    {
        "event_id": "friendly_indonesia",
        "date_start": "2024-01-09",
        "event_type": "رویداد_ورزشی_ملی",
        "scope": "ملی",
        "affected_meal": "شام",
        "confidence": "متوسط",
        "description": "بازی دوستانه‌ی تدارکاتی ایران - اندونزی، الریان قطر، سه‌شنبه ۱۹ دی ۱۴۰۲؛ نتیجه: برد ۵-۰ ایران",
        "expected_effect": "بازی دوستانه (نه رقابتی) با جذابیت کم؛ اثر انتظاری بسیار ضعیف‌تر از بازی‌های جام ملت‌ها",
        "source_url": "https://en.wikipedia.org/wiki/Iran_national_football_team_results_(2020%E2%80%93present)",
    },
    {
        "event_id": "asian_cup_palestine",
        "date_start": "2024-01-14",
        "event_type": "رویداد_ورزشی_ملی",
        "scope": "ملی",
        "affected_meal": "شام",
        "confidence": "بالا",
        "description": "بازی گروهی ایران - فلسطین؛ جام ملت‌های آسیا ۲۰۲۳ قطر؛ یکشنبه، ساعت ۲۱:۰۰ به‌وقت تهران (۲۰:۳۰ به‌وقت قطر)؛ نتیجه: برد ۴-۱ ایران",
        "expected_effect": "احتمال کاهش نرخ دریافت شام یا جابه‌جایی زمان دریافت به‌خاطر هم‌زمانی با بازی",
        "source_url": "https://en.wikipedia.org/wiki/2023_AFC_Asian_Cup_Group_C",
    },
    {
        "event_id": "asian_cup_hong_kong",
        "date_start": "2024-01-19",
        "event_type": "رویداد_ورزشی_ملی",
        "scope": "ملی",
        "affected_meal": "شام",
        "confidence": "بالا",
        "description": "بازی گروهی هنگ‌کنگ - ایران؛ جام ملت‌های آسیا؛ جمعه، ساعت ۲۱:۰۰ به‌وقت تهران (۲۰:۳۰ قطر)؛ نتیجه: برد ۱-۰ ایران",
        "expected_effect": "جمعه است ولی چون داخل ایام امتحانات پایان‌ترم قرار دارد استثنائاً سرو داشته (یکی از تنها ۵ جمعه‌ی سرودار کل بازه)؛ حجم رزرو جمعه‌ها پایین است و اثر باید با احتیاط تفسیر شود",
        "source_url": "https://en.wikipedia.org/wiki/2023_AFC_Asian_Cup_Group_C",
    },
    {
        "event_id": "asian_cup_uae",
        "date_start": "2024-01-23",
        "event_type": "رویداد_ورزشی_ملی",
        "scope": "ملی",
        "affected_meal": "شام",
        "confidence": "بالا",
        "description": "بازی گروهی ایران - امارات؛ جام ملت‌های آسیا؛ سه‌شنبه، ساعت ۱۸:۳۰ به‌وقت تهران (۱۸:۰۰ قطر)؛ نتیجه: برد ۲-۱ ایران و صعود به‌عنوان صدرنشین گروه C",
        "expected_effect": "نزدیک به آغاز بازه‌ی معمول سرو شام؛ احتمال کاهش جزئی نرخ دریافت",
        "source_url": "https://en.wikipedia.org/wiki/2023_AFC_Asian_Cup_Group_C",
    },
    {
        "event_id": "asian_cup_syria_r16",
        "date_start": "2024-01-31",
        "event_type": "رویداد_ورزشی_ملی",
        "scope": "ملی",
        "affected_meal": "هردو",
        "confidence": "بالا",
        "description": "یک‌هشتم‌نهایی ایران - سوریه؛ جام ملت‌های آسیا؛ چهارشنبه، ساعت ۱۶:۳۰ به‌وقت تهران (۱۶:۰۰ قطر)؛ نتیجه: مساوی ۱-۱ در وقت قانونی و برد ۵-۳ ایران در ضربات پنالتی",
        "expected_effect": "بین بازه‌ی معمول ناهار و شام؛ ممکن است رفتار دریافت وعده‌ی عصر/شام روز را تحت تأثیر قرار دهد",
        "source_url": "https://www.espn.com/soccer/match/_/gameId/698049/syria-iran",
    },
    {
        "event_id": "asian_cup_japan_qf",
        "date_start": "2024-02-03",
        "event_type": "رویداد_ورزشی_ملی",
        "scope": "ملی",
        "affected_meal": "ناهار",
        "confidence": "بالا",
        "description": "یک‌چهارم‌نهایی ایران - ژاپن؛ جام ملت‌های آسیا؛ شنبه، ساعت ۱۵:۰۰ به‌وقت تهران (۱۱:۳۰ UTC)؛ نتیجه: برد دراماتیک ۲-۱ ایران با پنالتی دقیقه ۹۶",
        "expected_effect": "بلافاصله پس از پایان بازه‌ی معمول سرو ناهار؛ احتمال تجمع دانشجویان برای تماشای بازی هم‌زمان با ناهار و کاهش نرخ دریافت؛ این روز (۱۴ بهمن) آخرین روز امتحانات پایان‌ترم است و سرو داشت — درست پیش از شروع تعطیلات بین‌ترم",
        "source_url": "https://www.aljazeera.com/sports/2024/2/3/iran-beat-japan-2-1-for-a-place-in-afc-asian-cup-2023-semifinal",
    },
    {
        "event_id": "asian_cup_qatar_sf",
        "date_start": "2024-02-07",
        "event_type": "رویداد_ورزشی_ملی",
        "scope": "ملی",
        "affected_meal": "شام",
        "confidence": "بالا",
        "description": "نیمه‌نهایی ایران - قطر (میزبان)؛ جام ملت‌های آسیا؛ چهارشنبه، ساعت ۱۸:۳۰ به‌وقت تهران (۱۸:۰۰ قطر)؛ نتیجه: باخت ۳-۲ ایران و حذف از جام",
        "expected_effect": "داخل بازه‌ی تعطیلات بین‌ترم (۱۵ تا ۲۰ بهمن) و بدون سرو؛ باخت پرحاشیه (اعتراض به داوری) ممکن است اثر روانی روزهای پس از بازگشایی ترم را داشته باشد",
        "source_url": "https://www.aljazeera.com/sports/2024/2/7/qatar-edge-iran-3-2-in-dramatic-asian-cup-2023-semifinal",
    },
    {
        "event_id": "majlis_election_round1",
        "date_start": "2024-03-01",
        "event_type": "انتخابات",
        "scope": "ملی",
        "affected_meal": "هردو",
        "confidence": "بالا",
        "description": "مرحله‌ی اول انتخابات دوازدهمین دوره‌ی مجلس شورای اسلامی و ششمین دوره‌ی مجلس خبرگان رهبری؛ جمعه ۱۱ اسفند ۱۴۰۲؛ مشارکت ۴۰٫۶٪ (کمترین در تاریخ انتخابات مجلس)",
        "expected_effect": "احتمال سفر برخی دانشجویان به شهر محل رأی‌گیری خود و کاهش رزرو/حضور در روزهای مجاور؛ چون جمعه است و سرو ندارد، اثر فقط به‌صورت سرریز روی چهارشنبه/پنجشنبه قبل و شنبه بعد قابل جست‌وجو است",
        "source_url": "https://en.wikipedia.org/wiki/2024_Iranian_legislative_election",
    },
    {
        "event_id": "bus_drivers_strike",
        "date_start": "2024-03-06",
        "event_type": "اعتصاب",
        "scope": "استان_تهران",
        "affected_meal": "هردو",
        "confidence": "متوسط",
        "description": "اعتصاب رانندگان اتوبوسرانی تهران (خط ۷ سامانه BRT ۴) در اعتراض به عدم پرداخت عیدی پایان سال از سوی شهرداری؛ چهارشنبه ۱۶ اسفند ۱۴۰۲",
        "expected_effect": "احتمال کاهش جزئی حضور دانشجویانی که با اتوبوس شهری تردد می‌کنند؛ اثر محتمل محدود چون بخش زیادی از دانشجویان خوابگاهی نزدیک پردیس هستند",
        "source_url": "https://ir.voanews.com/a/dozens-of-tehran-bus-drivers-strike-on-wednesday-over-unpaid-benefits/7517281.html",
    },
    {
        "event_id": "ramadan_1445",
        "date_start": "2024-03-12",
        "date_end": "2024-04-09",
        "event_type": "ماه_رمضان",
        "scope": "ملی",
        "affected_meal": "هردو",
        "confidence": "بالا",
        "description": "ماه رمضان ۱۴۴۵ در ایران: ۱ رمضان = سه‌شنبه ۲۲ اسفند ۱۴۰۲ (۱۲ مارس ۲۰۲۴) تا ۳۰ رمضان = ۲۱ فروردین ۱۴۰۳ (۹ آوریل ۲۰۲۴). **مهم‌ترین رویداد ساختاری کل بازه‌ی داده**: سرو ناهار در تمام این ۲۹ روز کاملاً متوقف شده و فقط شام (افطار) سرو شده است — در داده‌ی تجمیعی هیچ ردیف ناهاری بین ۲۰۲۴-۰۳-۱۲ و ۲۰۲۴-۰۴-۰۹ وجود ندارد",
        "expected_effect": "قطع کامل سری زمانی ناهار به‌مدت ۲۹ روز (شکاف ساختاری، نه داده‌ی گمشده) و تغییر رژیم سری شام؛ نرخ عدم‌دریافت شام در روزهای نخست رمضان به‌شدت بالا می‌رود (سازگاری رفتاری با افطار). ⚠️ هر فیچر lag/rolling که از این مرز عبور کند بی‌معنا است؛ بند ۵.۲ و ۵.۳ باید این شکاف را صریحاً مدیریت کند. توصیه: افزودن `is_ramadan` به `src/data/calendar.py` طبق بند ۵.۱ WBS",
        "source_url": "https://www.moroccoworldnews.com/2024/03/22232/ramadan-2024-official-start-date-in-iran-will-be-march-12/",
    },
    {
        "event_id": "derby_103",
        "date_start": "2024-03-13",
        "event_type": "رویداد_ورزشی_باشگاهی",
        "scope": "استان_تهران",
        "affected_meal": "شام",
        "confidence": "متوسط",
        "description": "دربی ۱۰۳ تهران، پرسپولیس - استقلال، ورزشگاه آزادی، شامگاه چهارشنبه ۲۳ اسفند ۱۴۰۲ ساعت ۲۰:۰۰",
        "expected_effect": "هم‌زمان با بازه‌ی سرو افطار در دومین روز رمضان؛ ⚠️ اثر آن از اثر شروع رمضان (`ramadan_1445`) قابل تفکیک نیست و نباید جداگانه تفسیر شود",
        "source_url": "https://www.mehrnews.com/news/6050100/",
    },
    {
        "event_id": "nowruz_1403",
        "date_start": "2024-03-18",
        "date_end": "2024-03-29",
        "event_type": "تعطیلات_نوروز",
        "scope": "ملی",
        "affected_meal": "هردو",
        "confidence": "بالا",
        "description": "بلوک تعطیلات نوروز ۱۴۰۳ در تقویم آموزشی دانشگاه تهران؛ در داده‌ی سامانه هیچ سروی بین ۲۰۲۴-۰۳-۱۸ و ۲۰۲۴-۰۳-۲۹ ثبت نشده است",
        "expected_effect": "شکاف ساختاری ۱۲ روزه در هر دو سری ناهار و شام؛ در `calendar_tehran.csv` با `is_nowruz_block` پوشش داده شده — این ردیف فقط برای انسجام آرشیو رویدادی است",
        "source_url": "derived:calendar_tehran.csv",
    },
    {
        "event_id": "wcq_turkmenistan_home",
        "date_start": "2024-03-21",
        "event_type": "رویداد_ورزشی_ملی",
        "scope": "ملی",
        "affected_meal": "شام",
        "confidence": "بالا",
        "description": "مقدماتی جام جهانی ۲۰۲۶، ایران - ترکمنستان، تهران، ۲ فروردین ۱۴۰۳؛ نتیجه: برد ۵-۰ ایران",
        "expected_effect": "بدون اثر قابل‌سنجش — داخل بلوک تعطیلات نوروز و بدون سرو",
        "source_url": "https://en.wikipedia.org/wiki/Iran_national_football_team_results_(2020%E2%80%93present)",
    },
    {
        "event_id": "wcq_turkmenistan_away",
        "date_start": "2024-03-26",
        "event_type": "رویداد_ورزشی_ملی",
        "scope": "ملی",
        "affected_meal": "شام",
        "confidence": "بالا",
        "description": "مقدماتی جام جهانی ۲۰۲۶، ترکمنستان - ایران، عشق‌آباد، ۷ فروردین ۱۴۰۳؛ نتیجه: برد ۱-۰ ایران",
        "expected_effect": "بدون اثر قابل‌سنجش — داخل بلوک تعطیلات نوروز و بدون سرو",
        "source_url": "https://en.wikipedia.org/wiki/Iran_national_football_team_results_(2020%E2%80%93present)",
    },
    {
        "event_id": "damascus_consulate_strike",
        "date_start": "2024-04-01",
        "event_type": "رویداد_امنیتی",
        "scope": "بین‌المللی",
        "affected_meal": "هردو",
        "confidence": "بالا",
        "description": "حمله‌ی هوایی اسرائیل به بخش کنسولی سفارت ایران در دمشق، ۱۳ فروردین ۱۴۰۳؛ ۱۶ کشته شامل سردار محمدرضا زاهدی؛ نقطه‌ی شروع رویارویی مستقیم ایران و اسرائیل",
        "expected_effect": "روز تعطیل رسمی (شهادت امام علی + روز طبیعت) و بدون سرو؛ اثر آن از طریق فضای امنیتی روزهای بعد و رویداد `iran_strike_israel` دنبال می‌شود",
        "source_url": "https://en.wikipedia.org/wiki/Israeli_airstrike_on_the_Iranian_consulate_in_Damascus",
    },
    {
        "event_id": "eid_fitr_1445",
        "date_start": "2024-04-10",
        "date_end": "2024-04-11",
        "event_type": "عید_مذهبی",
        "scope": "ملی",
        "affected_meal": "شام",
        "confidence": "بالا",
        "description": "عید فطر ۱۴۴۵؛ چهارشنبه ۲۲ فروردین ۱۴۰۳ (۱۰ آوریل ۲۰۲۴) و روز تعطیل پس از آن. در داده: سرو ناهار دقیقاً از همین روز از سر گرفته شده و در عوض سرو شام در ۲۲ و ۲۳ فروردین کاملاً متوقف بوده",
        "expected_effect": "نقطه‌ی بازگشت رژیم ناهار و شکاف دوروزه در سری شام؛ حجم رزرو ناهار در روزهای نخست پس از عید هنوز به سطح عادی نرسیده و به‌تدریج بازمی‌گردد",
        "source_url": "https://www.zakat.org/when-is-eid-al-fitr-2024",
    },
    {
        "event_id": "iran_strike_israel",
        "date_start": "2024-04-13",
        "date_end": "2024-04-14",
        "event_type": "رویداد_امنیتی",
        "scope": "بین‌المللی",
        "affected_meal": "شام",
        "confidence": "بالا",
        "description": "عملیات «وعده‌ی صادق»: شلیک بیش از ۳۰۰ پهپاد و موشک ایران به اسرائیل؛ آغاز پرتاب‌ها شامگاه شنبه ۲۵ فروردین ۱۴۰۳ و اصابت حوالی ۲:۰۰ بامداد یکشنبه ۲۶ فروردین به‌وقت ایران؛ نخستین حمله‌ی مستقیم نظامی ایران به اسرائیل",
        "expected_effect": "هم‌زمانی پرتاب‌ها با بازه‌ی سرو شام شنبه؛ انتظار افزایش نرخ عدم‌دریافت شام و احتمال ماندن دانشجویان پای اخبار",
        "source_url": "https://simple.wikipedia.org/wiki/April_2024_Iranian_strikes_against_Israel",
    },
    {
        "event_id": "isfahan_strike",
        "date_start": "2024-04-19",
        "event_type": "رویداد_امنیتی",
        "scope": "بین‌المللی",
        "affected_meal": "هردو",
        "confidence": "بالا",
        "description": "حمله‌ی هوایی اسرائیل به یک سایت پدافند هوایی نزدیک اصفهان، ساعت ۵:۲۳ بامداد جمعه ۳۱ فروردین ۱۴۰۳؛ ایران خسارت را تکذیب کرد",
        "expected_effect": "جمعه است و سرو ندارد؛ اثر احتمالی فقط از طریق فضای امنیتی و سرریز به شنبه ۱ اردیبهشت",
        "source_url": "https://en.wikipedia.org/wiki/April_2024_Israeli_strikes_on_Iran",
    },
    {
        "event_id": "majlis_election_round2",
        "date_start": "2024-05-10",
        "event_type": "انتخابات",
        "scope": "ملی",
        "affected_meal": "هردو",
        "confidence": "بالا",
        "description": "مرحله‌ی دوم انتخابات دوازدهمین دوره‌ی مجلس شورای اسلامی در ۲۲ حوزه‌ی انتخابیه از جمله تهران؛ جمعه ۲۱ اردیبهشت ۱۴۰۳؛ رأی‌گیری الکترونیکی در تهران",
        "expected_effect": "مشابه مرحله‌ی اول: احتمال سفر دانشجویان به شهر محل رأی‌گیری؛ جمعه و بدون سرو، پس فقط اثر سرریز روی روزهای مجاور قابل بررسی است",
        "source_url": "https://www.irna.ir/news/85421995/",
    },
    {
        "event_id": "raisi_helicopter_crash",
        "date_start": "2024-05-19",
        "date_end": "2024-05-21",
        "event_type": "سوگواری_ملی",
        "scope": "ملی",
        "affected_meal": "هردو",
        "confidence": "بالا",
        "description": "سقوط بالگرد حامل ابراهیم رئیسی رئیس‌جمهور و حسین امیرعبداللهیان وزیر امور خارجه در ورزقان آذربایجان شرقی، یکشنبه ۳۰ اردیبهشت ۱۴۰۳ حوالی ساعت ۱۳:۳۰؛ تأیید مرگ در ۳۱ اردیبهشت و اعلام ۵ روز عزای عمومی توسط رهبری. تعطیلی سراسری ادارات برای تشییع در ۲ خرداد اعلام شد که **خارج از بازه‌ی داده‌ی ماست** (داده در ۱ خرداد ۱۴۰۳ تمام می‌شود)",
        "expected_effect": "قوی‌ترین رویداد انتهای بازه: انتظار افزایش پلکانی نرخ عدم‌دریافت در سه روز پایانی داده به‌دلیل عزای عمومی، لغو کلاس‌ها و مراسم تشییع؛ ⚠️ چون این سه روز آخرین روزهای سری زمانی‌اند، در تقسیم زمانی فاز ۶ کاملاً داخل مجموعه‌ی آزمون می‌افتند و می‌توانند ارزیابی را سوگیر کنند — این نکته باید در طراحی backtest لحاظ شود",
        "source_url": "https://en.wikipedia.org/wiki/2024_Varzaqan_helicopter_crash",
    },
]


# ---------------------------------------------------------------------------
# ۲.۴.۲ رویدادهای مشتق از داده‌ی آب‌وهوا/کیفیت هوای خودمان
# ---------------------------------------------------------------------------


def _load_weather() -> pd.DataFrame:
    w = pd.read_csv(WEATHER_PATH)
    return w[(w.date_gregorian >= WINDOW_START) & (w.date_gregorian <= WINDOW_END)].reset_index(drop=True)


def _consecutive_runs(dates: list[str]) -> list[tuple[str, str]]:
    """رشته‌های تاریخ مرتب را به بازه‌های پیوسته (شروع، پایان) تبدیل می‌کند."""
    if not dates:
        return []
    ts = sorted(pd.Timestamp(d) for d in dates)
    runs, start, prev = [], ts[0], ts[0]
    for d in ts[1:]:
        if (d - prev).days == 1:
            prev = d
            continue
        runs.append((start, prev))
        start = prev = d
    runs.append((start, prev))
    return [(s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d")) for s, e in runs]


def derive_pollution_episodes(weather: pd.DataFrame) -> list[dict]:
    """دوره‌های پیوسته‌ی آلودگی «ناسالم برای همه» را از داده‌ی AQI خودمان استخراج می‌کند."""
    hits = weather.loc[weather.aqi_us_max >= AQI_UNHEALTHY, "date_gregorian"].tolist()
    runs = [
        (s, e) for s, e in _consecutive_runs(hits) if (pd.Timestamp(e) - pd.Timestamp(s)).days + 1 >= AQI_MIN_RUN_DAYS
    ]
    events = []
    for i, (start, end) in enumerate(runs, start=1):
        n_days = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
        block = weather[(weather.date_gregorian >= start) & (weather.date_gregorian <= end)]
        peak = int(block.aqi_us_max.max())
        events.append(
            {
                "event_id": f"aqi_episode_{i:02d}",
                "date_start": start,
                "date_end": end,
                "event_type": "آلودگی_هوای_شدید",
                "scope": "استان_تهران",
                "affected_meal": "هردو",
                "confidence": "بالا",
                "description": (
                    f"دوره‌ی پیوسته‌ی {n_days} روزه با AQI حداکثر روزانه ≥ {AQI_UNHEALTHY} "
                    f"(«ناسالم برای همه‌ی گروه‌ها»)؛ اوج دوره: AQI={peak}. "
                    "استخراج‌شده از داده‌ی Open-Meteo پروژه، نه از خبر — پس تاریخ‌ها قطعی و بازتولیدپذیرند"
                ),
                "expected_effect": (
                    "کاهش تردد اختیاری و احتمال افزایش نرخ عدم‌دریافت؛ ⚠️ این ردیف جایگزین «تعطیلی رسمی» "
                    "نیست — آلودگی شدید همیشه به تعطیلی منجر نشده است"
                ),
                "source_url": "derived:weather_aqi_tehran.csv",
            }
        )
    return events


def derive_snow_events(weather: pd.DataFrame) -> list[dict]:
    """روزهای بارش برف قابل‌توجه را از داده‌ی هواشناسی خودمان استخراج می‌کند."""
    hits = weather.loc[weather.snowfall_sum >= HEAVY_SNOW_CM, "date_gregorian"].tolist()
    events = []
    for i, (start, end) in enumerate(_consecutive_runs(hits), start=1):
        block = weather[(weather.date_gregorian >= start) & (weather.date_gregorian <= end)]
        total = round(float(block.snowfall_sum.sum()), 2)
        tmin = float(block.temp_min.min())
        events.append(
            {
                "event_id": f"snow_episode_{i:02d}",
                "date_start": start,
                "date_end": end,
                "event_type": "بارش_برف_سنگین",
                "scope": "استان_تهران",
                "affected_meal": "هردو",
                "confidence": "بالا",
                "description": (
                    f"بارش برف مجموعاً {total} سانتی‌متر با کمینه‌ی دمای {tmin}°C؛ "
                    "استخراج‌شده از داده‌ی Open-Meteo پروژه"
                ),
                "expected_effect": "اختلال تردد و احتمال افزایش نرخ عدم‌دریافت، به‌ویژه برای دانشجویان غیرخوابگاهی",
                "source_url": "derived:weather_aqi_tehran.csv",
            }
        )
    return events


# ---------------------------------------------------------------------------
# ۲.۴.۳ برچسب‌گذاری اثر مشاهده‌شده در داده
# ---------------------------------------------------------------------------


def _daily_rho(df_aggregate: pd.DataFrame) -> pd.DataFrame:
    """نرخ عدم‌دریافت روزانه به تفکیک وعده، به‌همراه انحراف از میانه‌ی همان روزِ هفته."""
    g = (
        df_aggregate.groupby(["DateReserveGregorian", "Meal"])
        .agg(res=("Reservation", "sum"), dr=("DontReceive", "sum"))
        .reset_index()
    )
    g["rho"] = g.dr / g.res
    # روزهایی با حجم رزرو ناچیز (مثلاً اولین روز ترم) نویز شدید تولید می‌کنند و از مبنای
    # میانه‌ی روزِ هفته کنار گذاشته می‌شوند.
    base = g[g.res > 500]
    dow = pd.to_datetime(g.DateReserveGregorian).dt.weekday
    g["dow"] = (dow + 2) % 7
    base = base.assign(dow=(pd.to_datetime(base.DateReserveGregorian).dt.weekday + 2) % 7)
    med = base.groupby(["dow", "Meal"])["rho"].median().rename("rho_dow_median")
    sd = base.groupby("Meal")["rho"].transform(lambda s: s - s.median()).groupby(base.Meal).std()
    g = g.merge(med, on=["dow", "Meal"], how="left")
    g["z"] = (g.rho - g.rho_dow_median) / g.Meal.map(sd)
    return g.set_index(["DateReserveGregorian", "Meal"])


def _service_status(rho: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> str:
    """وضعیت واقعی سرو در بازه‌ی رویداد، مستقیماً از داده‌ی تجمیعی."""
    n_slots = n_served = 0
    for d in pd.date_range(start, end, freq="D"):
        key = d.strftime("%Y-%m-%d")
        for meal in ("lunch", "dinner"):
            n_slots += 1
            n_served += (key, meal) in rho.index
    if n_served == 0:
        return "بدون_سرو"
    return "سرو_کامل" if n_served == n_slots else "سرو_ناقص"


def annotate_observed_effect(events: list[dict], df_aggregate: pd.DataFrame) -> list[dict]:
    """برای هر رویداد، $\\rho$ واقعی و انحراف استانداردشده‌اش از میانه‌ی همان روزِ هفته را می‌نویسد.

    این ستون ادعای `expected_effect` را در برابر شاهد تجربی قرار می‌دهد — بدون آن، آرشیو رویدادی
    فقط فهرستی از حدس‌هاست.
    """
    rho = _daily_rho(df_aggregate)
    out = []
    for ev in events:
        ev = dict(ev)
        start = pd.Timestamp(ev["date_start"])
        end = pd.Timestamp(ev.get("date_end") or ev["date_start"])
        ev["service_status"] = _service_status(rho, start, end)
        parts = []
        for d in pd.date_range(start, end, freq="D"):
            key = d.strftime("%Y-%m-%d")
            served = [m for m in ("lunch", "dinner") if (key, m) in rho.index]
            if not served:
                parts.append(f"{key}: بدون سرو")
                continue
            for m in served:
                r = rho.loc[(key, m)]
                parts.append(f"{key} {m}: ρ={r.rho:.3f} (z={r.z:+.1f})")
        # برای رویدادهای طولانی فقط سر و ته بازه ثبت می‌شود تا سلول از کنترل خارج نشود.
        if len(parts) > 6:
            parts = parts[:3] + [f"… ({len(parts) - 6} سطر میانی حذف شد)"] + parts[-3:]
        ev["observed_in_data"] = " | ".join(parts)
        out.append(ev)
    return out


# ---------------------------------------------------------------------------
# اسمبل نهایی
# ---------------------------------------------------------------------------


def build_events(output_path=OUTPUT_PATH) -> pd.DataFrame:
    weather = _load_weather()
    events = MANUAL_EVENTS + derive_pollution_episodes(weather) + derive_snow_events(weather)

    for ev in events:
        ev.setdefault("date_end", ev["date_start"])
        ev["date_jalali_start"] = _jalali(ev["date_start"])
        ev["date_jalali_end"] = _jalali(ev["date_end"])

    out_of_window = [e["event_id"] for e in events if not (WINDOW_START <= e["date_start"] <= WINDOW_END)]
    if out_of_window:
        raise ValueError(f"رویداد خارج از بازه‌ی داده: {out_of_window}")

    events = annotate_observed_effect(events, load_aggregate())

    df = pd.DataFrame(events)[COLUMNS].sort_values(["date_start", "event_id"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} events to {output_path}")
    return df


if __name__ == "__main__":
    df_events = build_events()
    print(f"\n{len(df_events)} رویداد ثبت شد.\n")
    print("توزیع نوع رویداد:")
    print(df_events.event_type.value_counts().to_string())
    print("\nتوزیع وعده‌ی متأثر:")
    print(df_events.affected_meal.value_counts().to_string())
    print("\nپوشش ماهانه (بر اساس ماه شمسی شروع):")
    print(df_events.date_jalali_start.str[:7].value_counts().sort_index().to_string())
