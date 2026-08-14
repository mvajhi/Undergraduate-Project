"""بند 7.0.2/7.10–7.22/7.26 سند فاز ۷ — رجیستری فراداده‌ی خانواده‌ها و مدل‌های فاز ۷.

``FAMILIES`` نقشه‌ی سیزده خانواده‌ی نسخه‌ی ۳.۰ (`doc/WBS-phase7-modeling.md`) را به‌صورت
برنامه‌ریزی‌شده نگه می‌دارد — منبع واحدی که ``src/models/naming.py`` کد خانواده را در
برابرش اعتبارسنجی می‌کند، به‌جای اینکه هر اسکریپت رشته‌ی خام "F02" را بی‌بررسی قبول کند.

``MODELS`` رجیستری تک‌تک مدل‌هاست. این فایل جای WBS را نمی‌گیرد و پیشاپیش هر ۱۶۹ مدل را
حدس نمی‌زند — هر خانواده هنگام پیاده‌سازیِ واقعی‌اش (اسپرینت S2 مربوطه)، اعضایش را با
``register()`` اضافه می‌کند. اینجا فقط شکل و قرارداد ثبت مشخص می‌شود.
"""

from dataclasses import dataclass, field

#: سطوح داده‌ی بند 7.1.1 — "*" یعنی مدل مستقل از سطح است (مثل خانواده‌ی ترکیب/کالیبراسیون)
LEVELS = ("L1", "L2", "L3", "L4", "L5")

#: مراحل قیف اجرا — بند 7.3.1
STAGES = ("S0", "S1", "S2", "S3")

#: مسیرهای استخراج کوانتایل — بند 7.23
QUANTILE_ROUTES = ("Q1", "Q2", "Q3", "Q4", "Q5")


@dataclass(frozen=True)
class Family:
    code: str                  # "F01".."F13" — پیشوند run_name (بند 7.7.1)
    wbs_section: str            # بند سند تفصیلی، مثل "7.11"
    name_fa: str
    n_models: int               # تعداد اعضا طبق جدول «نقشه‌ی سیزده خانواده» (جمع = ۱۶۹)
    levels: tuple[str, ...]     # سطوح اصلی قابل‌اعمال
    compute: str                # "CPU" یا "GPU" — بند 7.28.1


#: بند «نقشه‌ی سیزده خانواده» doc/WBS-phase7-modeling.md — مطابقت با ماتریس 7.26.
FAMILIES: dict[str, Family] = {
    "F01": Family("F01", "7.10", "خطی و تعمیم‌یافته", 17, ("L1", "L2", "L3", "L5"), "CPU"),
    "F02": Family("F02", "7.11", "درختی و بوستینگ", 16, ("L1", "L2", "L3", "L4", "L5"), "CPU"),
    "F03": Family("F03", "7.12", "سری‌زمانی کلاسیک تک‌متغیره", 14, ("L3", "L4"), "CPU"),
    "F04": Family("F04", "7.13", "سری‌زمانی چندمتغیره و پنلی", 10, ("L3", "L4"), "CPU"),
    "F05": Family("F05", "7.14", "واریانس شرطی (ARCH/GARCH)", 12, ("L3", "L4"), "CPU"),
    "F06": Family("F06", "7.15", "کرنل و بردار پشتیبان", 10, ("L1", "L2", "L3"), "CPU"),
    "F07": Family("F07", "7.16", "شبکه‌ی عصبی", 27, ("L1", "L4", "L5"), "GPU"),
    "F08": Family("F08", "7.17", "سلسله‌مراتبی و بیزی", 13, ("L1", "L2", "L5"), "CPU"),
    "F09": Family("F09", "7.18", "رگرسیون توزیعی و شمارشی", 13, ("L1", "L2"), "CPU"),
    "F10": Family("F10", "7.19", "نمونه‌محور و ناپارامتری", 9, ("L1", "L2", "L3"), "CPU"),
    "F11": Family("F11", "7.20", "تصمیم-محور (Newsvendor یادگیرنده)", 9, ("L1", "L2"), "CPU"),
    "F12": Family("F12", "7.21", "ترکیب، متامدل و آشتی سلسله‌مراتبی", 10, ("*",), "CPU"),
    "F13": Family("F13", "7.22", "کالیبراسیون و پوشش", 9, ("*",), "CPU"),
}

assert sum(f.n_models for f in FAMILIES.values()) == 169, (
    "جمع اعضای FAMILIES باید ۱۶۹ باشد (تصحیح‌شده‌ی جدول «نقشه‌ی سیزده خانواده») — "
    "اگر عمداً تغییر کرد، doc/WBS-phase7-modeling.md و AGENTS.md را هم به‌روزرسانی کنید."
)


@dataclass(frozen=True)
class ModelSpec:
    """توصیف‌گر یک مدل واحد. هر خانواده هنگام پیاده‌سازی، اعضایش را با ``register()`` ثبت می‌کند."""

    model_id: str                          # اسلاگ کوتاه snake_case، مثل "lightgbm_quantile"
    family: str                            # کد خانواده، باید در FAMILIES باشد
    levels: tuple[str, ...]                # سطوح داده‌ای که این مدل واقعاً روی آن اجرا می‌شود
    quantile_route: str                    # یکی از QUANTILE_ROUTES (بند 7.23)
    incompatible_levels: tuple[str, ...] = field(default_factory=tuple)  # با دلیل در بند 7.27

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"خانواده‌ی نامعتبر: {self.family!r} (باید یکی از {sorted(FAMILIES)} باشد)")
        bad_levels = set(self.levels) - set(LEVELS)
        if bad_levels:
            raise ValueError(f"سطح نامعتبر: {bad_levels} (باید زیرمجموعه‌ی {LEVELS} باشد)")
        if self.quantile_route not in QUANTILE_ROUTES:
            raise ValueError(f"مسیر کوانتایل نامعتبر: {self.quantile_route!r}")


#: رجیستری تک‌تک مدل‌ها — خالی شروع می‌شود، هر خانواده هنگام پیاده‌سازی پر می‌کند.
MODELS: dict[str, ModelSpec] = {}


def register(spec: ModelSpec) -> ModelSpec:
    """یک ``ModelSpec`` را در ``MODELS`` ثبت می‌کند. فراخوانی دوباره با همان ``model_id`` خطا می‌دهد
    تا تصادفی دو مدل هم‌نام (مثلاً در دو خانواده) جایگزین هم نشوند."""
    if spec.model_id in MODELS:
        raise ValueError(f"model_id تکراری: {spec.model_id!r} (قبلاً در خانواده‌ی "
                         f"{MODELS[spec.model_id].family} ثبت شده)")
    MODELS[spec.model_id] = spec
    return spec


def models_of_family(family: str) -> list[ModelSpec]:
    return [m for m in MODELS.values() if m.family == family]
