# شکل‌های گزارش نهایی

**منبع حقیقت هر شکل چاپ‌شده‌ی فصل ۳ و ۴، `reports/figures/` است.** هر فایل داخل
`final_report/img/` که مقابلش یک شکل تولیدی (نه لوگو/besm) باشد، **symlink** به
یک فایل در `reports/figures/` است، نه کپی. با بازتولید یک شکل (دستور زیر) نسخه‌ی
داخل گزارش هم خودکار به‌روز می‌شود — نیازی به کپی دستی نیست.

```bash
PYTHONPATH=. .venv/bin/python tools/report_figures/make_figs.py            # perm_importance.png (+ 8.8_reliability_fa.png، دیگر در گزارش نمی‌آید)
PYTHONPATH=. .venv/bin/python tools/report_figures/make_surrogate_tree.py  # surrogate_tree.png (شکل ۴-۳)
PYTHONPATH=. .venv/bin/python tools/report_figures/make_residual_figs.py   # residual_hist.png, resid_acf.png (پانل‌های آ و ب شکل ۴-۱)
PYTHONPATH=. .venv/bin/python -m src.models.run_phase10_tau_curve          # tau_tradeoff.png (بند ۱۰.۷)
PYTHONPATH=. .venv/bin/python tools/report_figures/make_featureset_fig.py   # feature_sets.png (شکل ۳-۱)
PYTHONPATH=. .venv/bin/python tools/report_figures/make_eda_figs.py        # نُه شکل بند ۳-۳-۳ (شکل‌های ۳-۲ تا ۳-۱۰)
```

`resid_vs_res.png` (پانل ج شکل ۴-۱) اسکریپت جدا ندارد — کپی/تبدیلی لازم نداشت، پس مستقیم
symlink به خروجی `src/models/run_phase8_residuals.py` است.

## نگاشت symlink → مبدأ

| `final_report/img/` | `reports/figures/` |
|---|---|
| `perm_importance.png` | `phase9/9.1_perm_importance_fa.png` |
| `surrogate_tree.png` | `phase9/9.5_surrogate_tree_fa.png` |
| `residual_hist.png` | `phase8/8.2_residual_hist_fa.png` |
| `resid_acf.png` | `phase8/8.2_resid_acf_fa.png` |
| `tau_tradeoff.png` | `phase10/10.7_tau_tradeoff_fa.png` |
| `resid_vs_res.png` | `phase8/8.2_resid_vs_res.png` |
| `feature_sets.png` | `5.12_feature_sets_fa.png` |
| `target_distribution.png` | `report_01_target_distribution_fa.png` |
| `city_effect.png` | `report_02_city_effect_fa.png` |
| `aqi_spurious.png` | `report_03_aqi_spurious_fa.png` |
| `pre_holiday.png` | `report_07_pre_holiday_fa.png` |
| `acf_by_meal.png` | `report_08_acf_by_meal_fa.png` |
| `daily_series.png` | `report_09_daily_series_volume_fa.png` |
| `dorm_faculty.png` | `report_10_dorm_faculty_fa.png` (پنل راست) |
| `lorenz.png` | `report_12_lorenz_fa.png` |
| `day_shock.png` | `report_14_day_shock_fa.png` |

پسوند `_fa` یعنی «نسخه‌ی گزارش‌آماده»: بدون عنوان تکراری با کپشن، با برچسب فارسی/mathtext
سالم — نسخه‌ی خام هم‌بند (مثلاً `phase9/9.1_feature_importance.csv` یا
`phase8/8.2_residual_hist.png` بدون `_fa`) برای EDA/نوت‌بوک دست‌نخورده می‌ماند.

## چرا این شکل‌ها نسخه‌ی جدا از EDA خام دارند

۱. **`τ` و `Δ` با فونت Vazirmatn مربع خالی چاپ می‌شوند.** `$\tau$` (mathtext) هم
   جواب نمی‌دهد: patch بیدی `src/viz_fa` واژه‌ی فارسیِ کنارِ بخش ریاضی را وارونه
   می‌کند («اسمی» ← «یمسا»). راه‌حل، نوشتن معادل فارسی است («کوانتایل اسمی»).
۲. **عنوان داخل شکل با کپشن لاتک تکراری است** و اغلب واژه‌ی «قهرمان» دارد که از
   بدنه‌ی گزارش حذف شده.
۳. گاهی شکل چندپنلی است و پنلی را نشان می‌دهد که متن گزارش صریحاً ردش می‌کند
   (`9.1_feature_importance.png`: پنل چپ همان اهمیت درختی سوگیردار است).

### نمودارهای کاوش داده (بند ۳-۳-۳)

نُه شکل این بند سه جور ساخته می‌شوند و همه‌شان از `make_eda_figs.py` بیرون می‌آیند:

۱. **بازتولید از داده** — `report_01` (توزیع هدف) و `report_14` (شوک روزانه). شوک دقیقاً
   طبق رابطه‌ی (۱) گزارش محاسبه می‌شود؛ محور تاریخِ شکل شمسی است و شکاف روزهای بدون سرو
   با خط وصل نمی‌شود (برخلاف نسخه‌ی خام EDA).
۲. **بازسازی** — `report_12` (منحنی لورنتس). نسخه‌ی خام در برچسب‌هایش نویسه‌ی «⇒» دارد که
   در Vazirmatn مربع خالی چاپ می‌شود؛ نسخه‌ی گزارش با متن فارسی بازنویسی شده. ضریب جینی
   خروجی (۰.۵۶۲) با `data_facts_register` (F47) تطبیق داده شده است.
۳. **برش نوار عنوان** — شش شکل دیگر؛ از یکی‌شان (`report_10`) فقط پنل راست با
   `crop_right_panel()` جدا شده تا در شکل سه‌پنلی بند ۳-۳-۳ بنشیند. `strip_title()` نوار عنوان بالای PNG را می‌برد و
   عنوان پنل‌ها را نگه می‌دارد؛ هیچ عددی بازمحاسبه نمی‌شود. مرز برش از پروفایل سطرهای
   دارای جوهر پیدا می‌شود، پس اگر نسخه‌ی خام بازتولید شود همچنان کار می‌کند.

اعداد این اسکریپت‌ها ساخته نمی‌شوند، فقط از `reports/` خوانده می‌شوند؛ ساختار درخت
جانشین از `reports/phase9/9.5_surrogate_tree.txt` دستی منتقل شده چون مدلش ذخیره نشده.
