# خ۶ — فرایند گاوسی روی L1 (GPyTorch، اجرای GPU)

> اجرای GPU، بند 7.8 `doc/WBS-phase7-modeling.md`. خانواده F06. سخت‌افزار: Tesla T4 · torch=2.10.0+cu128 · CUDA=12.8

## R0 — آزمایش دود (اجراپذیری + سیم‌چین نشتی)

| مدل | pinball | B3 | پوشش | R² | زمان |
|---|---|---|---|---|---|
| `gp_quantile` | 0.01641 | 0.01375 | 0.186 | -0.304 | 21.0s |
| `gp_heteroscedastic` | 0.01644 | 0.01375 | 0.244 | -0.173 | 29.6s |

## R2 — تنظیم با بودجه‌ی زمانی

| مدل | بهترین pinball | B3 | trial | همگرا (A6) | پایداری (۷.۶.۳) | شکست | ساعت-هسته |
|---|---|---|---|---|---|---|---|
| `gp_quantile` | **0.01794** | 0.01590 | 5 | ✅ | 3/5 | 0 | 0.57h |
| `gp_heteroscedastic` | **0.02161** | 0.01590 | 1 | ✅ | 5/5 | 0 | 0.36h |

## S3 — قهرمان‌ها: سه seed، کالیبراسیون ACI، آزمون Diebold-Mariano

| مدل | pinball(ردیفی) | B3(ردیفی) | Δ | DM p | معنادار؟ | پوشش | پوشش پس از ACI |
|---|---|---|---|---|---|---|---|
| `gp_quantile` | 0.01578 | 0.01335 | +0.00244 | 0.0000 | ❌ خیر | 0.1399 (-0.0601) | 0.2000 (+0.0000) |

### پراکندگی بین seedها (قاعده‌ی A7)

| مدل | seed 42 | seed 1234 | دامنه |
|---|---|---|---|
| `gp_quantile` | 0.01580 | 0.01578 | 0.00002 |

### فایل‌های مدل ذخیره‌شده

- `gp_quantile`: 30 فایل زیر `models/gpu/F06/gp_quantile/`

## یادداشت‌ها و تفسیر

- جدول مقایسه‌ی ۷ ترکیب کرنل (بند 7.15.3/7.15.6): `reports/gpu/F06_kernel_comparison.csv`.
- طول‌مقیاس ARD هر فیچر: `reports/gpu/F06_ard_lengthscales.csv` — خروجی تفسیری خانواده.
- GP روی L1 اجرا شد نه L3 — روی GPU محدودیت مقیاس‌پذیری بند 7.15.4 عملاً برطرف است.
- کوانتایل مستقیم از پسین گاوسی گرفته شد (مسیر Q4)، بدون آفست تجربی باقیمانده.
- K7 واریانس نویز را تابع log Res کرد — پاسخ مستقیم به F06؛ نتیجه‌اش در جدول کرنل.

## مقایسه‌ی ترکیب‌های کرنل (بند 7.15.3)

| ترکیب کرنل          |   pinball |     B3 |   پوشش |   LML (میانگین fold) |   ثانیه |
|:--------------------|----------:|-------:|-------:|---------------------:|--------:|
| K4_matern           |   0.02053 | 0.0159 | 0.13   |                -1.01 |   347.1 |
| K1_rbf              |   0.02062 | 0.0159 | 0.1023 |                -1.06 |   359.7 |
| K2_periodic         |   0.02099 | 0.0159 | 0.0968 |                -1.36 |   306.3 |
| K5_full             |   0.02103 | 0.0159 | 0.0891 |                -1.03 |   484.7 |
| K3_rbf_x_periodic   |   0.02107 | 0.0159 | 0.108  |                -1.06 |   452.2 |
| K7_heteroscedastic  |   0.02113 | 0.0159 | 0.1072 |                -1.03 |   970.5 |
| K6_full_plus_linear |   0.02114 | 0.0159 | 0.1826 |                -0.83 |   629.8 |

## ده فیچر با کوچک‌ترین طول‌مقیاس ARD

| kernel_param                                                 | feature                                |   lengthscale |
|:-------------------------------------------------------------|:---------------------------------------|--------------:|
| covar_module.kernels.1.base_kernel.raw_lengthscale           | RestaurantName=ژئوفیزیک                |      0.268668 |
| covar_module.kernels.1.base_kernel.raw_lengthscale           | RestaurantName=علوم اجتماعی            |      0.3      |
| covar_module.kernels.0.kernels.0.base_kernel.raw_lengthscale | RestaurantName=بیوشیمی و بیوفیزیک      |      0.345897 |
| covar_module.kernels.0.kernels.0.base_kernel.raw_lengthscale | RestaurantName=علوم اجتماعی            |      0.381926 |
| covar_module.kernels.0.kernels.0.base_kernel.raw_lengthscale | RestaurantName=مطالعات جهان            |      0.585211 |
| covar_module.kernels.1.base_kernel.raw_lengthscale           | RestaurantName=بیوشیمی و بیوفیزیک      |      0.623998 |
| covar_module.kernels.1.base_kernel.raw_lengthscale           | is_ramadan                             |      0.693147 |
| covar_module.kernels.0.kernels.0.base_kernel.raw_lengthscale | is_ramadan                             |      0.693147 |
| covar_module.kernels.0.kernels.0.base_kernel.raw_lengthscale | is_day_before_holiday                  |      0.759269 |
| covar_module.kernels.1.base_kernel.raw_lengthscale           | RestaurantName=روانشناسی و علوم تربیتی |      0.798892 |
