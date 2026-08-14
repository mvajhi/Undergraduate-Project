"""بند 7.9.1 سند فاز ۷ — محورهای آزمایش، به‌صورت **قابل‌اجرا** نه فقط جدول در سند.

## چرا این ماژول وجود دارد

سند فاز ۷ نُه محور آزمایش تعریف کرده (سطح داده، دامنه‌ی مدل، فیچرست، معماری، …). ولی
سندِ تنها هیچ چیزی را اجبار نمی‌کند: یک اجرا می‌تواند بی‌خبر فقط یک نقطه از این فضا را
بزند و در گزارش چنان دیده شود که انگار «مدل خطی آزموده شد». این ماژول آن جدول را به یک
**قرارداد** تبدیل می‌کند:

1. هر اجرا باید یک ``RunConfig`` بسازد که **صریحاً** موضعش روی هر نُه محور را اعلام کند —
   هیچ محوری «پیش‌فرضِ نانوشته» نمی‌ماند.
2. مقادیر نامعتبر همان‌جا رد می‌شوند، نه اینکه بی‌صدا در MLflow ثبت شوند.
3. هر محور به‌عنوان یک param در MLflow می‌رود، پس ``src/models/coverage.py`` می‌تواند
   بپرسد «کدام نقاط این فضا اصلاً زده نشده‌اند؟» و جواب **قابل‌گزارش** بدهد.

قاعده‌ی حاکم (بند 7.2، منشور انصاف): نزدن یک نقطه اشکالی ندارد؛ **نگفتنِ اینکه نزده‌ای**
اشکال دارد. گزارش پوشش، تفاوت این دو است.
"""

from dataclasses import asdict, dataclass

from src.models.registry import FAMILIES, LEVELS, STAGES

# ---------------------------------------------------------------------------
# مقادیر مجاز هر محور — بند 7.9.1
# ---------------------------------------------------------------------------

#: معماری (بند ۵.۱۷.۳): «تخت» = مدل مستقیم روی سلول؛ «دومرحله‌ای» = عامل روز ← انحراف سلف
ARCHITECTURES = ("flat", "two_stage")

#: نگاشت هدف. ``binary`` فقط در L5 معنا دارد (بند 7.1.1)
TARGETS = ("rho", "logit_rho", "arcsine_sqrt_rho", "count", "beta_binomial", "binary")

#: دامنه‌ی مدل: یک مدل سراسری، یا مدل جدا به‌ازای هر خوشه/شهر/سلف
#: ⚠️ F12: ``city`` و ``restaurant_id`` نباید هم‌زمان به‌عنوان فیچر وارد شوند
SCOPES = ("global", "per_cluster", "per_city", "per_restaurant")

#: وزن‌دهی نمونه — F06 ناهم‌واریانسی را تأیید کرده
WEIGHTINGS = ("none", "res", "sqrt_res")

#: سطح تجمیع خروجی (بند 7.24)
OUTPUT_AGGREGATIONS = ("per_food", "aggregated", "reconciled")

#: L1-L4 همگی از دیتاست **تجمیعی** مشتق می‌شوند (`dataset_v2.csv`)؛ L5 از دیتاست
#: **فردی** (`person_reservation_fact_v3.csv`) — همان مرز «مدل A / مدل B» تاریخی
#: (بند ۱.۶ WBS). برای نام‌گذاری خوانای MLflow Dataset استفاده می‌شود، تا از تب
#: Datasets هر run بی‌درنگ معلوم شود روی کدام دیتاست و با کدام فیچرست train شده.
LEVEL_DATASET_VARIANT: dict[str, str] = {
    "L1": "تجمیعی", "L2": "تجمیعی", "L3": "تجمیعی", "L4": "تجمیعی", "L5": "فردی",
}

#: شبکه‌ی τ بند 7.9.1 — کل شبکه فقط در S3 پیموده می‌شود
TAU_GRID = (0.02, 0.05, 0.10, 0.15, 0.20)

#: ⭐ **نقطه‌ی عملیاتی تثبیت‌شده‌ی پروژه** — S0/S1/S2 همگی روی همین τ اجرا می‌شوند (بند 7.3.1).
#:
#: مبنا: ردیف ۳۴ `doc/decision_log.md` — ذی‌نفع نسبت هزینه‌ی کمبود به مازاد را ~۴ برابر
#: برآورد کرد که طبق رابطه‌ی نیوزوندور ($\tau^*_\rho = C_o/(C_u+C_o)$) به ۰.۲۰ می‌افتد؛
#: تحلیل حاشیه‌ای مستقل در `reports/tau_sensitivity.md` همان نقطه را تأیید کرد.
#:
#: ⚠️ این عدد را اینجا **و** در بند 7.3.1 سند فاز ۷ **و** با یک ردیف جدید در decision_log
#: باهم عوض کنید — نه یکی‌یکی. تنظیم در یک τ و استقرار در τ دیگر یعنی قهرمان با معیار
#: اشتباه انتخاب می‌شود (رگرسیون کوانتایل به‌ازای هر τ یک مدل متفاوت است).
TUNING_TAU = 0.20

#: نام هر محور → مقادیر مجازش. کلیدها **دقیقاً** نام فیلدهای ``RunConfig``اند و همان نام
#: در MLflow هم param می‌شود، تا گزارش پوشش بتواند بدون نگاشت دستی کار کند.
AXES: dict[str, tuple] = {
    "level": LEVELS,
    "architecture": ARCHITECTURES,
    "target": TARGETS,
    "scope": SCOPES,
    "weighting": WEIGHTINGS,
    "output_aggregation": OUTPUT_AGGREGATIONS,
    "tau": TAU_GRID,
}

#: محور «فیچرست» عمداً در ``AXES`` مقدار محدود ندارد: بند 7.5 می‌گوید هر مدل حق دارد
#: فیچرست اختصاصی خودش (``FS_{model_id}``) را بسازد، پس مجموعه‌اش باز است. ولی همچنان
#: اعلام‌کردنش اجباری است و در گزارش پوشش شمرده می‌شود.
OPEN_AXES = ("feature_set",)


@dataclass(frozen=True)
class RunConfig:
    """موضع یک اجرا روی هر نُه محور آزمایش + هویت آن.

    هیچ فیلدی مقدار پیش‌فرض ندارد جز آن‌هایی که سند صریحاً «پیش‌فرض» خوانده — تا اعلام‌نکردن
    یک محور یک انتخاب آگاهانه باشد، نه فراموشی.
    """

    # هویت
    family: str
    model_id: str
    stage: str
    seed: int

    # محورهای آزمایش (بند 7.9.1)
    level: str
    feature_set: str
    tau: float
    architecture: str = "flat"
    target: str = "rho"
    scope: str = "global"
    weighting: str = "none"
    output_aggregation: str = "per_food"

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"خانواده‌ی نامعتبر: {self.family!r}")
        if self.stage not in STAGES:
            raise ValueError(f"مرحله‌ی نامعتبر: {self.stage!r}")
        for axis, allowed in AXES.items():
            value = getattr(self, axis)
            if value not in allowed:
                raise ValueError(f"مقدار نامعتبر برای محور {axis!r}: {value!r} "
                                 f"(مجاز: {allowed})")
        if self.target == "binary" and self.level != "L5":
            raise ValueError("هدف باینری فقط در سطح L5 (رزرو فردی) معنا دارد — بند 7.1.1")
        if not self.feature_set:
            raise ValueError("feature_set باید صریحاً اعلام شود (بند 7.5)")

    def to_mlflow_params(self) -> dict:
        """هر محور یک param جدا — تا ``coverage.py`` بتواند مستقیم روی MLflow پرس‌وجو کند."""
        return asdict(self)

    def axis_point(self) -> tuple:
        """موضع این اجرا در فضای محورها، بدون هویت — کلید گزارش پوشش."""
        return tuple(getattr(self, a) for a in AXES) + tuple(getattr(self, a) for a in OPEN_AXES)
