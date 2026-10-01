# راهبرد انتشار: حوزه‌ها، موضوع مقاله‌ها و مجله‌های پیشنهادی

> **تاریخ:** ۱۴۰۵-۰۷-۰۹ (2026-10-01) · **وضعیت:** پیشنهاد، هنوز تصمیمی ثبت نشده
> **پرسش:** این پروژه در چه حوزه‌هایی قابل انتشار است، چند مقاله از آن درمی‌آید، و هر
> مقاله به کدام مجله برود؟
>
> ⚠️ **محدودیت روش جست‌وجو.** دسترسی مستقیم به پایگاه‌های علمی (Crossref، OpenAlex،
> Semantic Scholar، arXiv، ScienceDirect، MDPI، PubMed) در محیط اجرای این تحلیل توسط
> سیاست شبکه مسدود بود. جست‌وجو با موتور جست‌وجوی وب انجام شد. شاخص‌های مجله (IF،
> چارک، APC) از منابع ثانویه آمده‌اند و گاهی با هم اختلاف دارند؛ **پیش از ارسال روی
> JCR/Scimago و صفحه‌ی خود مجله تأیید شوند.**

---

## خلاصه در یک صفحه

پروژه **یک مقاله‌ی اصلی قوی** و **دو مقاله‌ی جانبی مستقل** دارد. بقیه‌ی یافته‌ها یا
بخشی از همین سه مقاله‌اند یا ارزش مقاله‌ی جدا ندارند.

| # | مقاله | پرسش پژوهشی | حوزه | مجله‌ی هدف اول | جایگزین |
|---|---|---|---|---|---|
| **P1** | پیش‌بینی کوانتایلی نرخ عدم‌دریافت برای تصمیم پخت | آیا مدل یادگیری ماشین نسبت به قاعده‌ی ساده ارزش دارد، و آیا در سال‌های بعد هم منتقل می‌شود؟ | پیش‌بینی + پژوهش عملیات | **International Journal of Forecasting** | Journal of Cleaner Production ← Journal of Forecasting / Waste Management & Research |
| **P2** | چه کسی غذای رزروشده را جا می‌گذارد؟ | تمرکز هدررفت در سطح فرد چقدر است و هدف‌گیری آینده‌نگر چقدر از آن را می‌گیرد؟ | رفتار مصرف‌کننده + پایداری | **Resources, Conservation & Recycling** | Sustainable Production and Consumption ← Socio-Economic Planning Sciences |
| **P3** | چهار خطای بی‌صدا در ارزیابی پیش‌بینی | کدام انتخاب‌های ارزیابی، بدون هیچ خطایی، نتیجه را جابه‌جا می‌کنند؟ | روش‌شناسی ارزیابی | **Foresight** (IIF) | یادداشت پژوهشی در IJF |
| P4 | مقاله‌ی داده (مشروط) | — | داده‌ی باز | Data in Brief | Scientific Data |

**ترتیب پیشنهادی:** ارائه‌ی P1 در **ISF 2027** (پافوس قبرس، ۲۰ تا ۲۳ ژوئن ۲۰۲۷) و
ارسال هم‌زمانش به IJF؛ P2 موازی با آن (سطح داده و پرسش جداست)؛ P3 پس از پذیرش P1.

**سه دلیل اصلی این انتخاب:**

۱. **قوی‌ترین دارایی پروژه «اعتبار بیرونی» است، نه مدل.** قهرمان یک LightGBM کوانتایلی
   است، یعنی روش تازه‌ای در کار نیست. چیزی که در ادبیات کم است: ارزیابی برون‌نمونه‌ای
   روی **دو سال تحصیلی بعدی**، ۲۵ مدل در جدول نهایی، و بوت‌استرپ بلوکی دوبعدی. این
   دقیقاً همان چیزی است که IJF صریحاً می‌خواهد («empirical studies, evaluation
   activities, implementation research»).
۲. **نزدیک‌ترین کار پیشین ایرانی است و بیرون از IJF چاپ شده.** Faezirad و همکاران
   (۲۰۲۱) همین مسئله را با شبکه‌ی عصبی روی داده‌ی یک دانشگاه ایرانی در
   *Waste Management & Research* منتشر کرده‌اند. این هم شاهد تناسب موضوع با آن
   مجله است و هم یعنی داوران آن را می‌شناسند؛ مقاله باید صریحاً خودش را از آن
   متمایز کند (بخش ۴).
۳. **پیام «سود اصلی از قاعده‌ی تصمیم است نه از مدل» یک نقطه‌ی قوت است، نه ضعف.**
   در برابر وضع موجود (پخت = رزرو) دورریز حدود ۶۰٪ کم می‌شود، ولی برتری قهرمان بر
   خط پایه‌ی ساده‌ی کوانتایل تجربی فقط ۶.۷ تا ۸.۱٪ pinball است. ادبیات پیش‌بینی این
   نوع صداقت را می‌پسندد؛ ادبیات پایداری عدد ۶۰٪ را. هر مقاله یکی از این دو را
   سرتیتر می‌کند.

---

## ۱. روش کار

تحلیل در چهار گام انجام شد تا قابل بازبینی باشد:

| گام | کار | خروجی |
|---|---|---|
| ۱. فهرست دارایی‌ها | خواندن هدفمند اسناد مرجع مخزن (`doc/project-definition.md`، `doc/findings_register.md`، `reports/phase7/model_comparison.md`، `reports/phase8/8.10_L1L2_vs_L5.md`، `reports/phase10/10.4_annual_savings.md`، `reports/phase13/oos_validation.md`، `reports/phase14/*insight_report_fa.md`، `reports/phase14/stream_b_synthesis.md`، `doc/Literature_Review.md`) | جدول بخش ۲ |
| ۲. نگاشت به حوزه | برای هر دارایی، رشته‌هایی که آن را «سهم تازه» حساب می‌کنند | جدول بخش ۳ |
| ۳. جست‌وجوی پیشینه‌ی نزدیک | ۳۴ پرس‌وجو در شش محور (فهرست کامل در پیوست الف) | جدول بخش ۴ |
| ۴. غربال مجله | چهار معیار: **(الف)** شاهد تناسب، یعنی مجله کار مشابهی چاپ کرده باشد؛ **(ب)** تناسب سطح مجله با قوت سهم؛ **(پ)** هزینه و امکان پرداخت از ایران؛ **(ت)** الزامات خاص (داده، کد، اجرای واقعی) | بخش ۵ |

معیار (الف) مهم‌ترین است: مجله‌ای که در پنج سال گذشته مقاله‌ای نزدیک به این موضوع
چاپ کرده، احتمال رد اولیه به دلیل «خارج از حوزه» را کم می‌کند.

---

## ۲. دارایی‌های قابل‌انتشار پروژه

هر ردیف یک «ادعای قابل‌دفاع» است با شاهدش در مخزن. ستون آخر می‌گوید کدام مقاله آن را
حمل می‌کند.

| # | ادعا | عدد کلیدی | شاهد | قوت | مقاله |
|---|---|---|---|---|---|
| ۱ | داده‌ی کم‌نظیر: رزرو پیش‌پرداخت با outcome فردی، سه سال تحصیلی | حدود ۸٫۲ میلیون رزرو · حدود ۴۶٫۵ هزار فرد · ۷۱ سلف · ۵۹۳ روز سرویس (مهر ۱۴۰۲ تا اسفند ۱۴۰۴) | `reports/phase14/insight_report_fa.md`، `reports/phase14/round9_insight_report_fa.md` | بالا | همه |
| ۲ | صورت‌بندی: رزرو «کوواریت آینده‌ی معلوم» است؛ کوانتایل در فضای نرخ برابر $\tau_\rho = C_o/(C_u+C_o)$ است نه فرمول کلاسیک | — | `doc/project-definition.md` بند ۲-۲، ردیف ۳۶ | متوسط (نتیجه‌ی ریاضی ساده، ولی خطای رایج) | P1 |
| ۳ | قاعده‌ی برش اطلاعاتی **به تفکیک وعده** (شام دیروز برای ناهار امروز نشت است) | — | `AGENTS.md`، ردیف ۹ | متوسط | P1، P3 |
| ۴ | محک ۱۱ خانواده‌ی مدل؛ در جدول نهایی ۲۵ مدل از ۸ خانواده، و فقط ۵ مدل خط پایه‌ی کوانتایل تجربی را معنادار بردند | قهرمان: `lightgbm_quantile` | `reports/phase7/model_comparison.md` | بالا | P1 |
| ۵ | **انتقال به سال‌های بعد:** برتری بر خط پایه در هر دو سال بعد معنادار می‌ماند، ولی با فاصله‌ی زمانی کم می‌شود | ۹٫۳٪ درون‌پنجره ← ۶٫۷٪ و ۸٫۱٪؛ افت قهرمان +۱٫۰٪ و +۶٫۲٪ | `reports/phase13/oos_validation.md` | **بسیار بالا** | P1 |
| ۶ | ارزش تصمیم: در برابر «پخت = رزرو» | کاهش دورریز ۶۰ تا ۶۲٪ با نرخ کمبود سلولی ۶ تا ۸٪ (سه تقسیم) | همان | بالا | P1 |
| ۷ | **بیشتر ارزش از قاعده‌ی تصمیم است، نه از مدل:** سود خالص قهرمان نسبت به خط پایه روی پنجره‌ی Test عملاً صفر | −۰٫۰۰۳ میلیارد تومان | یافته‌ی ۴۸، `reports/phase10/10.2_newsvendor.md` | بالا (و جذاب برای IJF) | P1 |
| ۸ | $\tau = 0.20$ روی سه تقسیم بهینه می‌ماند؛ پوشش بدکالیبره است ولی تصمیم خراب نمی‌شود (سقف‌گیری پرس) | پوشش ۰٫۳۳ تا ۰٫۳۹ در برابر نرخ کمبود ۰٫۰۶ تا ۰٫۰۸ | `reports/phase13/oos_validation.md` بند ۳ و ۴ | بالا | P1 |
| ۹ | ACI شکاف پوشش را بست، CQR ساده بدترش کرد | +۱۳٫۶٪ ← +۰٫۱٪ | یافته‌های ۲۰، ۲۲، ۴۱ | متوسط | P1 |
| ۱۰ | **مدل سطح فرد برای تصمیم تجمیعی بازنده است** (برخلاف ادعای رایج در ادبیات هواپیمایی و هتل) | L5 حدود ۴۶٪ pinball بدتر | `reports/phase8/8.10_L1L2_vs_L5.md`، یافته‌های ۲۱ و ۳۱ | بالا | P1، P2 |
| ۱۱ | سقف پیش‌بینی‌پذیری: ۸۳٪ واریانس یک شوک روزانه‌ی مشترک است | $R^2$ اطلاعات ایستا ۰٫۱۹ در برابر وضعیت روز×سلف ۰٫۷۶ | F59، `reports/phase14/insight_report_fa.md` | بالا | P1 |
| ۱۲ | رانش صعودی هدف: ۹۴٪ آن تغییر رفتار درون سلف است، نه جابه‌جایی جمعیت | ۷٫۷۱٪ ← ۸٫۲۵٪ ← ۸٫۶۵٪ | همان | متوسط | P1 یا P2 |
| ۱۳ | تمرکز هدررفت روی موقعیت‌ها | جینی ۰٫۷۱۶؛ ۱۰٪ موقعیت‌ها = ۵۳٫۴٪ هدررفت | همان | متوسط | P1 |
| ۱۴ | تمرکز در سطح فرد **به آستانه وابسته است** | سهم ۱۰٪ بدترین افراد بین ۱۱٫۸٪ و ۴۱٫۸٪ | F261، `reports/phase14/round9_insight_report_fa.md` | بالا | P2 |
| ۱۵ | عدد قابل‌تعهد هدف‌گیری آینده‌نگر ۳۰٪ است نه ۴۲٪ | ۲۹٫۵ تا ۳۱٫۲٪ | F249، ردیف ۱۰۴ | **بسیار بالا** | P2 |
| ۱۶ | معیار هدف‌گیری «نرخ × تعداد» است؛ گروه «کم‌مصرف × پرنرخ» بدتر از انتخاب تصادفی است | بازده ۰٫۶۲× | همان | بالا | P2 |
| ۱۷ | غیبت بلوکی، نه بی‌قیدی روزانه؛ فضای رفتاری پیوسته است، نه خوشه‌ای | ژاکارد ۰٫۹۹ در کنار آزمون پوچی که همان خوشه‌ها را نویز می‌خواند | F252، F254 | بالا | P2 |
| ۱۸ | اعتبار سازه‌ی هدف: کد تک‌وضعیتی تا ۱۰ برابر نوسان می‌کند، اجتماع کدها پایدار است | ۸٫۳۱٪ پایدار در ۲۸ ماه | F90 | متوسط | P1 (بخش داده) |
| ۱۹ | CI بوت‌استرپ سطری ۲٫۳ برابر باریک‌تر از واقعیت است | ICC روز ۰٫۲۲۵، سلف ۰٫۲۰۵؛ n مؤثر ۳۳۲ از ۳٬۸۸۰ | F138، `reports/phase13/oos_validation.md` | بالا | P3 |
| ۲۰ | ablation با مدل تنظیم‌نشده ارزش فیچر را بیش‌برآورد می‌کند | −۱٫۶٪ ← −۰٫۲٪ | F139 | بالا | P3 |
| ۲۱ | نشت **تجمیعی** از سه لایه‌ی ممیزی رایج رد می‌شود | همبستگی ۰٫۱۴۹ ولی اثر ۲٫۴ برابر | F292، F299 | بالا | P3 |
| ۲۲ | باگ‌های بی‌صدا: علامت در LP کوانتایل، سیم‌کشی فیچرست، پیش‌فرض کتابخانه به‌جای هایپرپارامتر قهرمان | — | یافته‌های ۴ و ۳۴، F136 | متوسط | P3 |
| ۲۳ | شوک خرداد ۱۴۰۴: قفل ۷۲ ساعته‌ی رزرو زمان ترک شهر را آشکار می‌کند | ρ ‏۰٫۴۷ تا ۰٫۶۷؛ حذف ۰٫۵۲٪ حجم هزینه‌ی انتقال را از +۱۱٫۶٪ به +۱٫۰٪ می‌برد | F135، F137 | پایین برای مقاله‌ی جدا | P1 (بخش استواری) |
| ۲۴ | رمضان مسئله‌ی ظرفیت است نه نرخ | افت حجم ۶ برابری در برابر نرخ ۱٫۵۸× | `reports/phase14/insight_report_fa.md` | پایین (۳۳ روز داده) | — |

**ردیف‌های ۲۳ و ۲۴ مقاله‌ی جدا نمی‌شوند.** شوک خرداد ۱۳ روز است و از نظر سیاسی
حساس؛ جایش یک کادر استواری در P1 است. رمضان با دو رمضانِ ناهم‌پوشان قابل اتکا نیست.

---

## ۳. نقشه‌ی حوزه‌ها: این پروژه کجا «سهم تازه» حساب می‌شود؟

| حوزه | چرا مناسب است | شاهد تناسب (کار مشابه چاپ‌شده) | تناسب |
|---|---|---|---|
| **پیش‌بینی (Forecasting)** | کوانتایل با کوواریت آینده‌ی معلوم، ارزیابی چندساله، مقایسه با خط پایه‌ی ساده، کالیبراسیون | IJF: Weatherford & Kimes (2003) پیش‌بینی هتل با رزرو پیشین؛ Qu, Timmermann & Zhu (2024) مقایسه‌ی پیش‌بینی پانلی. J. Forecasting: Kim (2026) پسماند غذای سلف نظامی | **قوی** |
| **پژوهش عملیات / نیوزوندور داده‌محور** | تصمیم پخت = نیوزوندور؛ کوانتایل در فضای متمم؛ سنجش ارزش مدل در برابر قاعده | EJOR: Huber et al. (2019)؛ Operations Research: Ban & Rudin (2019)؛ C&IE: overbooking هتل بر پایه‌ی احتمال no-show (2023) | **قوی**، ولی سهم نظری کم است |
| **پسماند غذا و تولید پاک** | ۶۶۲ هزار پرس دورریز، ۶۰٪ کاهش، برآورد مالی | JCP: Rodrigues et al. (2024)؛ WM&R: Faezirad et al. (2021)؛ ESPR: مرور Rodrigues & Miguéis (2025)؛ Sustainability: Turker (2025) | **قوی** |
| **رفتار مصرف‌کننده و سیاست‌گذاری** | تمرکز، پایداری رفتار، حدود هدف‌گیری | RCR: آزمون مداخله در کترینگ مدرسه (2021)؛ SEPS: Malefors et al. (2022)؛ JoES: مرور no-show از Amberger & Schreyer (2024) | **قوی** برای P2 |
| **مدیریت مهمان‌نوازی و خدمات غذایی** | سلف به‌عنوان عملیات خدماتی | IJHM: پیش‌بینی لغو رزرو هتل؛ J. Foodservice Business Research | متوسط: مخاطب مدیریتی است و عمق آماری را کمتر ارج می‌نهد |
| **پایداری در آموزش عالی** | عملیات دانشگاه | IJSHE؛ Leal Filho et al. (2023) در EDS | متوسط: سطح مجله پایین‌تر از قوت داده |
| **روش‌شناسی یادگیری ماشین / تکرارپذیری** | نشت، بوت‌استرپ، ablation | Patterns: Kapoor & Narayanan (2023) | ضعیف برای مقاله‌ی جدا، مگر با تعمیم به داده‌های دیگر |
| **داده‌ی باز** | داده‌ی تراکنش سطح فرد در محیط دانشگاهی کمیاب است | داده‌ی ۸ ساله‌ی خرید غذای EPFL (۳۸ میلیون تراکنش) | مشروط به مجوز انتشار داده |
| **مجلات داخلی** | مخاطب مستقیم: مدیران تغذیه‌ی دانشگاه‌ها | مدیریت صنعتی دانشگاه تهران (IMJ): هم Faezirad et al. را چاپ کرده | جایگزین، نه هدف اول |

---

## ۴. پیشینه‌ی نزدیک و شکافی که پروژه پر می‌کند

### ۴-۱. نزدیک‌ترین کارها

| کار | داده | روش | نتیجه | تفاوت با این پروژه |
|---|---|---|---|---|
| **Faezirad, Pooya, Naji-Azimi & Amir Haeri (2021)**، *Waste Management & Research* 39(8):1027–1038 | سامانه‌ی رزرو یک دانشگاه بزرگ ایرانی با یارانه | ANN دومرحله‌ای، تحلیل هزینه‌ی دورریز و کمبود | کاهش دورریز تا ۷۹٪ | سطح روزانه/تجمیعی؛ بدون ارزیابی برون‌نمونه‌ای چندساله؛ بدون مقایسه با خط پایه‌ی کوانتایل ساده؛ بدون سطح فرد |
| **Faezirad et al.**، Industrial Management Journal (دانشگاه تهران) | ۷ سال داده‌ی همان دانشگاه، دورریز ~۱۰٪ | ANN با تابع خطای وزن‌دار (GPS) | کاهش دورریز روزانه بیش از ۸۰٪ | پیش‌بینی نقطه‌ای با وزن نامتقارن؛ سطح سلف×غذا ندارد |
| **Rodrigues, Miguéis, Freitas & Machado (2024)**، *J. Cleaner Production* 435:140265 | ۳ سلف | RF، LSTM و دو خط پایه | کاهش تقاضای برآورده‌نشده ۳ تا ۱۶٪ | بدون رزرو پیشین؛ پیش‌بینی تقاضا نه نرخ عدم‌حضور |
| **Turker (2025)**، *Sustainability* 17(2):379 | سلف پردیس | GBR، XGB، Lasso | $R^2$ ‏۰٫۹۹۹ (لاسو) | معیار متقارن؛ $R^2$ نزدیک ۱ پرسش نشت برمی‌انگیزد |
| **Malefors et al. (2022)**، *Socio-Economic Planning Sciences* 82 | کترینگ عمومی سوئد، دوره‌ی کووید | پیش‌بینی حضور مهمان | خطای ۲ تا ۳٪ | بدون رزرو؛ مدرسه |
| **Kim (2026)**، *Journal of Forecasting* 45(4) | سلف‌های نظامی آمریکا | RF و یادگیری عمیق | پیش‌بینی پسماند بشقاب | هدف پسماند بشقاب است نه عدم‌حضور |
| **Weatherford & Kimes (2003)**، *IJF* 19(3):401–415 | هتل | مدل‌های pickup | pickup و رگرسیون بهترین | قیاس ساختاری مستقیم: رزرو معلوم، حضور نامعلوم |
| Hotel overbooking based on no-show probability forecasts (2023)، *C&IE* | هتل | احتمال no-show فردی ← تصمیم تجمیعی | — | این پروژه نشان می‌دهد مسیر فردی برای تصمیم تجمیعی **بازنده** است (ردیف ۱۰) |
| **Rodrigues & Miguéis (2025)**، *ESPR* 32:21301–21337 | مرور PRISMA، ۶۵ مطالعه | — | پیش‌بینی رایج‌ترین رویکرد است | مرجع برای قرار دادن P1 در ادبیات |
| **Qu, Timmermann & Zhu (2024)**، *IJF* 40(3):918–941 | پیش‌بینی‌های IMF | آزمون برابری دقت پانلی | — | پشتوانه‌ی نظری بوت‌استرپ بلوکی دوبعدی P1 و P3 |
| **Kapoor & Narayanan (2023)**، *Patterns* | ۲۹۴ مقاله در ۱۷ رشته | رده‌بندی نشت | نشت فراگیر است | P3 لایه‌ی چهارمی می‌افزاید که در رده‌بندی آن‌ها نیست |

### ۴-۲. شکاف

جست‌وجو هیچ مطالعه‌ای پیدا نکرد که هم‌زمان این پنج ویژگی را داشته باشد:

۱. پیش‌بینی **کوانتایلی** نرخ عدم‌حضور در محیط **رزرو پیش‌پرداخت** غذا، در سطح
   (روز، وعده، سلف، غذا).
۲. ارزیابی **برون‌نمونه‌ای روی سال‌های تحصیلی بعدی**، نه فقط تقسیم زمانی درون یک
   پنجره.
۳. تفکیک **ارزش قاعده‌ی تصمیم** از **ارزش مدل** (پخت = رزرو ← کوانتایل تجربی ← ML).
۴. مقایسه‌ی مستقیم **مدل فردی و مدل تجمیعی** برای یک تصمیم واحد.
۵. استنباط آماری که وابستگی درون‌روز و درون‌سلف را لحاظ کند.

⚠️ **مهم برای داوری:** کار Faezirad و همکاران از همین کشور، با همین نوع سامانه و همین
مسئله است. مقاله‌ی P1 باید در مقدمه صریحاً بگوید چه چیزی را **فراتر** از آن می‌برد
(ویژگی‌های ۱ تا ۵ بالا). داوری که آن کار را می‌شناسد و نبیند به آن ارجاع شده، احتمالاً
اصالت را زیر سؤال می‌برد. همین منطق برای بخش ۲-۳ `doc/Literature_Review.md` هم صادق
است: آن فصل این دو مقاله را ندارد.

---

## ۵. مقاله‌ها: موضوع، ادعا و مجله

### P1: مقاله‌ی اصلی

**عنوان کاری:**
*From reservations to cooking quantities: quantile forecasting of no-show rates in
pre-booked university dining, with a two-year out-of-sample evaluation*

**پرسش‌ها:**
۱. در برابر وضع موجود (پخت = رزرو) و خط پایه‌ی کوانتایل تجربی گروهی، ML چقدر ارزش
   اضافه می‌کند؟
۲. آیا این ارزش به سال‌های تحصیلی بعد منتقل می‌شود و با فاصله‌ی زمانی چه می‌کند؟
۳. آیا داده‌ی سطح فرد برای این تصمیم تجمیعی کمکی می‌کند؟

**ساختار پیشنهادی:** مسئله و صورت‌بندی (ردیف‌های ۲ و ۳) ← داده و اعتبار سازه‌ی هدف
(۱ و ۱۸) ← سقف پیش‌بینی‌پذیری (۱۱) ← محک مدل‌ها (۴ و ۱۰) ← انتقال (۵) ← تصمیم و
کالیبراسیون (۶، ۷، ۸، ۹) ← استواری: شوک خرداد ۱۴۰۴ (۲۳) ← پیامد مدیریتی.

**نردبان مجله:**

| ترتیب | مجله | IF تقریبی | چرا | چه چیزی را باید پررنگ کرد |
|---|---|---|---|---|
| ۱ | **International Journal of Forecasting** (Elsevier/IIF) | ~۷٫۱ | حوزه صریحاً «empirical studies, evaluation activities, implementation research» را می‌خواهد؛ سابقه‌ی پیش‌بینی بر پایه‌ی رزرو (Weatherford & Kimes) و مقایسه‌ی پانلی (Qu et al.) | انتقال چندساله، تفکیک ارزش قاعده از مدل، بازنده‌بودن مدل فردی، استنباط دوبعدی. کد و داده‌ی تکرار الزامی است |
| ۲ | **Journal of Cleaner Production** | ~۹٫۸ تا ۱۰٫۷ (منابع ناهمسان) | Rodrigues et al. (2024) دقیقاً همین نوع کار را آنجا چاپ کرده | عدد دورریز، صرفه‌جویی پرس، برآورد سالانه، تمرکز هدررفت |
| ۳ | **Journal of Forecasting** (Wiley) یا **Waste Management & Research** (SAGE) | ~۲٫۶ / ~۵٫۷ | J. Forecasting سلف نظامی (Kim 2026) را چاپ کرده؛ WM&R کار Faezirad را | گزینه‌ی امن‌تر با نرخ پذیرش بالاتر |

**چرا IJF اول و نه JCP؟** سهم واقعی پروژه **روش ارزیابی و صداقت در گزارش ارزش** است، نه
عدد دورریز. در JCP همان عدد ۶۰٪ را مقاله‌های دیگر با روش ضعیف‌تر هم ادعا کرده‌اند
(Faezirad ‏۷۹٪، Turker با $R^2$ ‏۰٫۹۹۹)، پس رقابت روی عدد است و پروژه با عدد
محافظه‌کارانه‌اش ضرر می‌کند. در IJF رقابت روی دقت ارزیابی است و پروژه آنجا قوی است.
اگر IJF در مرحله‌ی دفتری رد کرد، همان داده با قاب پایداری به JCP می‌رود.

**دو در دیگر، فقط پس از پایلوت.** EJOR بخش «Innovative Applications of O.R.» و مجله‌ی
INFORMS Journal on Applied Analytics هر دو به **اجرای واقعی** ارزش زیادی می‌دهند. پروتکل
پایلوت (`reports/phase10/10.5_pilot_protocol.md`) طراحی شده ولی اجرا نشده. اگر معاونت
دانشجویی آن را اجرا کند، این دو مجله واقع‌بینانه می‌شوند.

### P2: رفتار فردی و حدود هدف‌گیری

**عنوان کاری:**
*Who leaves the food they booked? Concentration, persistence and the prospective
limits of targeting no-shows among 46,000 university students*

**ادعاهای اصلی:** ردیف‌های ۱۲، ۱۴، ۱۵، ۱۶ و ۱۷، به‌علاوه‌ی نتیجه‌ی ردیف ۱۰ به‌عنوان
پیوند با P1 (هدف‌گیری فردی مکمل است؛ اهرم اصلی تصمیم پخت است: مداخله‌ای که نرخ
گروه هدف را ۲۰٪ کم کند، کل هدررفت را حدود ۳٫۸٪ کم می‌کند، در برابر ۴۷٫۵٪ قاعده‌ی
پخت، که خودش مرز بالاست).

**چرا مقاله‌ی جداست و تکه‌تکه‌سازی (salami slicing) نیست:** واحد تحلیل (فرد در برابر
سلول)، پرسش (چه کسی، نه چقدر)، مخاطب (سیاست‌گذار مداخله در برابر برنامه‌ریز تولید) و
روش (بخش‌بندی، پایداری بین‌سالی، آزمون پوچی خوشه) همه متفاوت‌اند.

| ترتیب | مجله | IF تقریبی | چرا |
|---|---|---|---|
| ۱ | **Resources, Conservation & Recycling** | ~۱۰٫۹ | مداخله‌ی رفتاری در پسماند غذای کترینگ را چاپ می‌کند؛ یافته‌ی «۳۰٪ نه ۴۲٪» و «نرخ × تعداد» پیام سیاستی روشن دارند |
| ۲ | **Sustainable Production and Consumption** | ~۹٫۶ تا ۱۲٫۱ (منابع ناهمسان) | مصرف پایدار در سطح فرد |
| ۳ | **Socio-Economic Planning Sciences** | ~۵٫۴ تا ۶٫۲ | Malefors et al. (2022) را چاپ کرده؛ قاب برنامه‌ریزی عمومی |
| جایگزین | International Journal of Sustainability in Higher Education | ~۳٫۴ | اگر قاب «عملیات دانشگاه» انتخاب شود |

⚠️ **پیش‌نیاز P2:** این مقاله با داده‌ی سطح فرد کار می‌کند. بیشتر این مجلات بیانیه‌ی
اخلاق (تأیید کمیته‌ی اخلاق یا معافیت آن) و توضیح ناشناس‌سازی می‌خواهند. در مخزن هیچ
سندی درباره‌ی تأیید اخلاقی پیدا نشد (جست‌وجوی «اخلاق/ethic/IRB» در `doc/`).

### P3: یادداشت روش‌شناختی

**عنوان کاری:**
*Four silent failures in forecast evaluation: evidence from a 25-model benchmark*

**ادعاها:** ردیف‌های ۱۹ تا ۲۲. ویژگی مشترکشان این است که **عدد تولید می‌کنند و خطا
نمی‌دهند**؛ همین است که آن‌ها را برای متخصصان پیش‌بینی خواندنی می‌کند.

| ترتیب | مجله | چرا |
|---|---|---|
| ۱ | **Foresight: The International Journal of Applied Forecasting** (IIF) | مجله‌ی کاربردی IIF، مقاله‌های «درس‌آموخته» و مطالعه‌ی موردی را داوری و منتشر می‌کند؛ آستانه‌ی ورود پایین‌تر و خوانندگان مستقیم |
| ۲ | یادداشت پژوهشی (research note) در IJF | فقط اگر لایه‌ی چهارم ممیزی نشت با شبیه‌سازی روی داده‌ی دیگری هم تعمیم داده شود |

**چرا Patterns نه؟** کار Kapoor & Narayanan در آن مجله بر ۲۹۴ مقاله در ۱۷ رشته تکیه
داشت. یافته‌های این پروژه از **یک** داده‌اند؛ بدون تعمیم، داور آن مجله «مطالعه‌ی موردی»
می‌بیند.

### P4: مقاله‌ی داده (مشروط)

اگر معاونت دانشجویی اجازه‌ی انتشار داده‌ی **تجمیعی سطح سلول** را بدهد (نه سطح فرد)،
یک مقاله‌ی داده هم ارجاع‌پذیری P1 را بالا می‌برد و هم الزام تکرارپذیری IJF را برآورده
می‌کند.

| مجله | نکته |
|---|---|
| **Data in Brief** (Elsevier) | دسترسی آزاد اجباری، APC حدود ۱٬۶۰۰ دلار؛ می‌تواند همراه P1 منتشر شود |
| Scientific Data (Nature) | آستانه‌ی بالاتر؛ فقط اگر داده‌ی چندساله و مستندسازی کامل منتشر شود |

بدون مجوز، این مسیر بسته است و P1 باید داده را «در دسترس با درخواست» اعلام کند.

### مجلات داخلی

**Industrial Management Journal (مدیریت صنعتی، دانشگاه تهران):** از ۲۰۲۵ انگلیسی‌زبان
است، داوری دوسوکور، حدود ۱۲ هفته تا انتشار، نرخ پذیرش حدود ۲۰٪، و کار Faezirad و
همکاران را چاپ کرده. **گزینه‌ی خوبی است برای وقتی که P1 در مسیر بین‌المللی گیر کند**،
نه برای انتشار موازی. انتشار هم‌زمان یک یافته در مجله‌ی داخلی و خارجی انتشار تکراری
است.

Interdisciplinary Journal of Management Studies (نام قبلی: Iranian Journal of
Management Studies)، دانشگاه تهران: وضعیت نمایه‌اش در Scopus در منابع ناهمسان است؛
پیش از انتخاب تأیید شود.

### همایش

**International Symposium on Forecasting (ISF 2027)**، پافوس قبرس، ۲۰ تا ۲۳ ژوئن ۲۰۲۷.
ارائه‌ی چکیده مانع انتشار در مجله نیست و فرصت گرفتن بازخورد از داوران بالقوه‌ی IJF است.
مهلت ارسال چکیده‌ی ۲۰۲۷ در نتایج جست‌وجو نبود؛ صفحه‌ی رویدادهای IIF را دنبال کنید.

---

## ۶. ریسک‌ها و پیش‌نیازها

| # | ریسک | چرا مهم است | اقدام |
|---|---|---|---|
| ۱ | **مجوز داده** | داده با مکاتبه‌ی رسمی از معاونت دانشجویی آمده (`doc/project-definition.md` بند ۳-۱) | مجوز کتبی برای انتشار نتایج، و جداگانه برای انتشار داده‌ی تجمیعی |
| ۲ | **اخلاق پژوهش** | P2 تحلیل رفتار ۴۶ هزار فرد است | تأیید یا معافیت کمیته‌ی اخلاق پیش از ارسال |
| ۳ | **تکه‌تکه‌سازی** | راهنمای IJF افشای مقاله‌های هم‌داده را الزامی می‌داند | هر مقاله پرسش جدا، ارجاع متقابل، بدون شکل تکراری؛ در نامه‌ی پوششی ذکر شود |
| ۴ | **ردیف ۸۲** | نتایج فاز ۱۳ وارد گزارش کارشناسی نشدند | آن تصمیم فقط درباره‌ی گزارش نهایی است؛ مقاله **باید** شامل فاز ۱۳ باشد، چون قوی‌ترین شاهد است، و باید بگوید اعداد درون‌پنجره خوش‌بینانه‌اند (F123) |
| ۵ | **کار Faezirad** | هم‌کشور، هم‌مسئله | ارجاع و تمایز صریح (بخش ۴-۲) |
| ۶ | **عدد ریالی** | ۴۴ تا ۸۱ میلیارد تومان با قیمت ۱۴۰۵ و برای خواننده‌ی خارجی بی‌معناست | گزارش به پرس و نسبت هزینه؛ عدد ریالی فقط در پیوست با فرض صریح |
| ۷ | **هزینه‌ی انتشار** | پرداخت ارزی از ایران دشوار است | مجلات هیبرید (IJF، JCP، RCR، WM&R، J. Forecasting) مسیر اشتراکی بدون هزینه دارند؛ Data in Brief و MDPI ندارند |
| ۸ | **بسته‌ی تکرار** | IJF کد و داده‌ی تکرار می‌خواهد | کد `src/` آماده است؛ نسخه‌ی تمیز با داده‌ی تجمیعی یا مصنوعی |
| ۹ | **انتقال عدد** | هر عدد باید پنجره‌اش معلوم باشد (`AGENTS.md`) | جدول «کدام عدد از کدام پنجره» در مقاله |

---

## ۷. گام‌های بعدی پیشنهادی

۱. **با استاد راهنما:** انتخاب نردبان مجله‌ی P1 و تعیین نویسندگان.
۲. **با معاونت دانشجویی:** مجوز انتشار نتایج و، در صورت امکان، داده‌ی تجمیعی.
۳. **کمیته‌ی اخلاق:** برای P2.
۴. **تکمیل پیشینه:** افزودن Faezirad et al. (2021، و نسخه‌ی IMJ)، Turker (2025)،
   Kim (2026)، Weatherford & Kimes (2003)، Qu et al. (2024) و Rodrigues & Miguéis (2025)
   به `doc/Literature_Review.md`، و خواندن متن کامل آن‌ها (این تحلیل فقط چکیده‌ها را
   دیده است).
۵. **پیش‌نویس P1** به انگلیسی، با جدول «کدام عدد از کدام پنجره».
۶. **چکیده‌ی ISF 2027** به محض اعلام مهلت.

اگر یکی از این مسیرها انتخاب شد، تصمیم در `doc/decision_log.md` ثبت شود.

---

## پیوست الف: فهرست پرس‌وجوها

| محور | پرس‌وجو | یافته‌ی کلیدی |
|---|---|---|
| حوزه‌ی مستقیم | university canteen food waste forecasting machine learning pre-booked meals reservation no-show | Faezirad et al. 2021؛ Turker 2025؛ پروژه‌ی Roskilde/CheffrBook |
| حوزه‌ی مستقیم | machine learning demand forecasting university dining hall food waste reduction journal 2025 | Turker 2025 (Sustainability) |
| حوزه‌ی مستقیم | "Preventing food waste in subsidy-based university dining systems" | WM&R 39(8):1027–1038 |
| حوزه‌ی مستقیم | Faezirad Pooya university dining reservation show no-show | نسخه‌ی IMJ دانشگاه تهران |
| حوزه‌ی مستقیم | "Demand Prediction in University Booking Systems to Reduce Food Waste…" | ANN با خطای وزن‌دار، ۷ سال داده |
| حوزه‌ی مستقیم | "Preventing food waste: Forecasting the number of meals for canteen kitchens" | Roskilde؛ کاهش ۲۵٪ و ۹٪ |
| حوزه‌ی مستقیم | "Reducing Food Waste in Campus Dining…" authors | Turker، Sustainability 17(2):379 |
| حوزه‌ی مستقیم | "Leveraging Machine Learning to Predict Food Waste Quantity" military | Kim 2026، J. Forecasting 45(4) |
| حوزه‌ی مستقیم (فارسی) | پیش‌بینی تقاضای غذا سلف دانشگاه رزرو عدم مراجعه دانشجویان مقاله | فقط آیین‌نامه‌های رزرو؛ مقاله‌ی فارسی تازه‌ای پیدا نشد |
| نیوزوندور | data-driven newsvendor quantile regression food service catering 2024 2025 | نیوزوندور سانسورشده، رویکرد کانفورمال، برنامه‌ریزی منوی استوار |
| نیوزوندور | SARIMA quantile regression food production hospital catering | SARIMA-QR (IJPE) |
| کترینگ | Malefors guest attendance forecasting public catering | SEPS 2022؛ ۲۰٪ مهمانان = ۶۰٪ پسماند بشقاب |
| مرور | literature review quantitative approaches food waste … 2025 | Rodrigues & Miguéis، ESPR 2025 |
| مرور | "Toward food waste reduction at universities" | Leal Filho et al.، EDS 2024 |
| no-show | no-show prediction meal reservation students cafeteria individual | مطالعه‌ی مستقیمی یافت نشد |
| no-show | hotel booking cancellation no-show prediction overbooking | C&IE 2023؛ IJHM |
| no-show | hotel revenue management pickup model advance booking | Weatherford & Kimes، IJF 2003 |
| no-show | heterogeneity individual no-show concentration targeting | Amberger & Schreyer 2024؛ مداخله‌های سلامت |
| رفتار | university cafeteria food waste behavior intervention RCR/Appetite | مداخله‌های پیش‌سفارش و آموزشی |
| روش | Kapoor Narayanan leakage reproducibility Patterns | ۲۹۴ مقاله، ۸ نوع نشت |
| روش | comparing forecast accuracy panel cross-sectional dependence block bootstrap | Qu, Timmermann & Zhu، IJF 2024 |
| روش | Qu Timmermann "Comparing forecasting performance with panel data" | IJF 40(3):918–941 |
| روش | adaptive conformal inference quantile forecast calibration demand | ACI در سری زمانی؛ کاربرد عملیاتی کم |
| داده | Scientific Data descriptor food service reservation dataset | داده‌ی EPFL (۳۸ میلیون تراکنش) |
| مجله | Waste Management & Research impact factor 2024 | ~۵٫۷، Q1/Q2 |
| مجله | Journal of Cleaner Production / RCR impact factor 2024 | ~۹٫۸ تا ۱۰٫۷ / ~۱۰٫۹ |
| مجله | IJF / EJOR impact factor 2024 | ~۷٫۱ / ~۶٫۰ |
| مجله | Sustainable Production and Consumption / SEPS impact factor | ~۹٫۶ تا ۱۲٫۱ / ~۵٫۴ تا ۶٫۲ |
| مجله | IJHM / J. Foodservice Business Research impact factor | ~۸٫۳ / ~۳٫۵ |
| مجله | IJSHE impact factor | ~۳٫۴ |
| مجله | IJPE / C&IE impact factor | ~۸٫۸ / ~۶٫۵ |
| مجله | J. Forecasting / Waste Management impact factor | ~۲٫۶ / ~۷٫۴ |
| مجله | Foresight IIF submission | مقاله‌ی کاربردی و مطالعه‌ی موردی |
| مجله | IJF guide for authors / forecasting practice | الزام افشای مقاله‌های هم‌داده؛ الزام کد و داده |
| مجله | Data in Brief scope APC | IF ~۱٫۹، APC ~۱٬۶۰۰ دلار |
| مجله | Industrial Management Journal UT scope | انگلیسی از ۲۰۲۵، ~۱۲ هفته، پذیرش ~۲۰٪ |
| مجله | Iranian Journal of Management Studies Scopus | وضعیت Scopus ناهمسان |
| همایش | International Symposium on Forecasting 2027 | پافوس، ۲۰ تا ۲۳ ژوئن ۲۰۲۷ |
| فراخوان | special issue call for papers 2026 food waste forecasting | ویژه‌نامه‌ی Annals of OR بسته شده |

## پیوست ب: منابع

- Faezirad, M., Pooya, A., Naji-Azimi, Z., & Amir Haeri, M. (2021). Preventing food waste in subsidy-based university dining systems: An artificial neural network-aided model under uncertainty. *Waste Management & Research*, 39(8), 1027–1038. https://journals.sagepub.com/doi/10.1177/0734242X211017974
- Faezirad, M., et al. Demand prediction in university booking systems to reduce food waste using neural networks including weighted error function. *Industrial Management Journal*. https://imj.ut.ac.ir/article_85301.html?lang=en
- Rodrigues, M., Miguéis, V., Freitas, S., & Machado, T. (2024). Machine learning models for short-term demand forecasting in food catering services. *Journal of Cleaner Production*, 435, 140265. https://www.researchgate.net/publication/377153878
- Rodrigues, M., & Miguéis, V. (2025). A literature review on the quantitative approaches to food waste. *Environmental Science and Pollution Research*, 32, 21301–21337. https://link.springer.com/article/10.1007/s11356-025-36937-9
- Turker, G. F. (2025). Reducing food waste in campus dining: A data-driven approach to demand prediction and sustainability. *Sustainability*, 17(2), 379. https://doi.org/10.3390/su17020379
- Kim (2026). Leveraging machine learning to predict food waste quantity: Military dining facilities. *Journal of Forecasting*, 45(4). https://onlinelibrary.wiley.com/doi/10.1002/for.70128
- Malefors, C., et al. (2022). Food waste reduction and economic savings in times of crisis. *Socio-Economic Planning Sciences*, 82. https://ideas.repec.org/a/eee/soceps/v82y2022ipas0038012121000331.html
- Leal Filho, W., et al. (2024). Toward food waste reduction at universities. *Environment, Development and Sustainability*, 26, 16585–16606. https://link.springer.com/article/10.1007/s10668-023-03300-2
- Weatherford, L. R., & Kimes, S. E. (2003). A comparison of forecasting methods for hotel revenue management. *International Journal of Forecasting*, 19(3), 401–415. https://ecommons.cornell.edu/bitstream/handle/1813/72130/Kimes30_Comparison_of_Forecasting.pdf
- Hotel overbooking based on no-show probability forecasts (2023). *Computers & Industrial Engineering*. https://www.sciencedirect.com/science/article/abs/pii/S0360835223002504
- Qu, R., Timmermann, A., & Zhu, Y. (2024). Comparing forecasting performance with panel data. *International Journal of Forecasting*, 40(3), 918–941. https://ideas.repec.org/a/eee/intfor/v40y2024i3p918-941.html
- Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. *Patterns*. https://www.cell.com/patterns/fulltext/S2666-3899(23)00159-9
- Amberger, C., & Schreyer, D. (2024). What do we know about no-show behavior? *Journal of Economic Surveys*. https://onlinelibrary.wiley.com/doi/10.1111/joes.12534
- Testing interventions to reduce food waste in school catering (2021). *Resources, Conservation and Recycling*. https://www.sciencedirect.com/science/article/pii/S0921344921006066
- EPFL campus food purchase dataset (arXiv). https://arxiv.org/pdf/2102.08755
- IJF Guide for Authors. https://www.sciencedirect.com/journal/international-journal-of-forecasting/publish/guide-for-authors
- Foresight article submission (IIF). https://forecasters.org/foresight/submit-article/
- International Symposium on Forecasting. https://forecasters.org/events/symposium-on-forecasting/
- Industrial Management Journal, About. https://imj.ut.ac.ir/journal/about
- Interdisciplinary Journal of Management Studies, Indexing. https://ijms.ut.ac.ir/journal/indexing
- Data in Brief. https://www.sciencedirect.com/journal/data-in-brief
- شاخص‌های مجله (ثانویه): https://www.scimagojr.com/journalsearch.php?q=26492&tip=sid · https://exaly.com/journal/16916/international-journal-of-forecasting/impact-factor · https://www.bioxbio.com/journal/RESOUR-CONSERV-RECY · https://exaly.com/journal/29254/sustainable-production-and-consumption/impact-factor · https://www.journalmetrics.org/journal/international-journal-of-hospitality-management · https://exaly.com/journal/23965/international-journal-of-sustainability-in-highe/impact-factor · https://exaly.com/journal/13787/computers-and-industrial-engineering/impact-factor · https://exaly.com/journal/13835/waste-management/impact-factor · https://en.wikipedia.org/wiki/Journal_of_Forecasting
