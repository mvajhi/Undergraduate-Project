# شکل‌های گزارش نهایی

**منبع حقیقت هر شکل چاپ‌شده‌ی فصل ۴، `reports/figures/` است.** هر فایل داخل
`final_report/img/` که مقابلش یک شکل تولیدی (نه لوگو/besm) باشد، **symlink** به
یک فایل در `reports/figures/` است، نه کپی. با بازتولید یک شکل (دستور زیر) نسخه‌ی
داخل گزارش هم خودکار به‌روز می‌شود — نیازی به کپی دستی نیست.

```bash
PYTHONPATH=. .venv/bin/python tools/report_figures/make_figs.py            # reliability.png, perm_importance.png
PYTHONPATH=. .venv/bin/python tools/report_figures/make_surrogate_tree.py  # surrogate_tree.png (شکل ۴-۴)
PYTHONPATH=. .venv/bin/python tools/report_figures/make_residual_figs.py   # residual_hist.png, resid_acf.png (شکل ۴-۳)
PYTHONPATH=. .venv/bin/python -m src.models.run_phase10_tau_curve          # tau_tradeoff.png (بند ۱۰.۷)
```

`resid_vs_res.png` (شکل ۴-۲) اسکریپت جدا ندارد — کپی/تبدیلی لازم نداشت، پس مستقیم
symlink به خروجی `src/models/run_phase8_residuals.py` است.

## نگاشت symlink → مبدأ

| `final_report/img/` | `reports/figures/` |
|---|---|
| `reliability.png` | `phase8/8.8_reliability_fa.png` |
| `perm_importance.png` | `phase9/9.1_perm_importance_fa.png` |
| `surrogate_tree.png` | `phase9/9.5_surrogate_tree_fa.png` |
| `residual_hist.png` | `phase8/8.2_residual_hist_fa.png` |
| `resid_acf.png` | `phase8/8.2_resid_acf_fa.png` |
| `tau_tradeoff.png` | `phase10/10.7_tau_tradeoff_fa.png` |
| `resid_vs_res.png` | `phase8/8.2_resid_vs_res.png` |

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

اعداد این اسکریپت‌ها ساخته نمی‌شوند، فقط از `reports/` خوانده می‌شوند؛ ساختار درخت
جانشین از `reports/phase9/9.5_surrogate_tree.txt` دستی منتقل شده چون مدلش ذخیره نشده.
