"""بند 7.6 سند فاز ۷ — رجیستری فضای هایپرپارامتر هر مدل (برای Optuna) + بودجه‌ی trial.

هر مدل، هنگام پیاده‌سازی در اسپرینت S2 خانواده‌ی خودش، فضای جستجویش را با
``register_space`` اینجا ثبت می‌کند: تابعی که یک ``optuna.Trial`` می‌گیرد و دیکشنری
هایپرپارامتر برمی‌گرداند. این فایل خودش هیچ فضایی از پیش تعریف نمی‌کند — رجیستری فقط
شکل ثبت را مشخص می‌کند؛ محتوا به‌مرور که هر خانواده پیاده‌سازی می‌شود اضافه می‌شود.

**نسخه‌بندی (بند 7.6.3):** تغییر فضای یک مدل یعنی افزایش ``version``، نه ویرایش بی‌صدا —
مقایسه‌ی نتایج S2 قدیم و جدید فقط وقتی معنا دارد که بدانیم زیر کدام فضا اجرا شده‌اند.
"""

from dataclasses import dataclass
from typing import Callable

import optuna

SpaceFn = Callable[[optuna.Trial], dict]


@dataclass(frozen=True)
class RegisteredSpace:
    model_id: str
    version: int
    n_hyperparams: int    # برای بودجه‌ی trial (بند 7.6.2) — نه لزوماً len(space)، چون
                           # بعضی پارامترها شرطی‌اند (مثلاً degree فقط وقتی kernel=poly)
    fn: SpaceFn


#: رجیستری فضاها — خالی شروع می‌شود، هر خانواده هنگام پیاده‌سازی پر می‌کند.
SPACES: dict[str, RegisteredSpace] = {}


def register_space(model_id: str, version: int, n_hyperparams: int) -> Callable[[SpaceFn], SpaceFn]:
    """دکوراتور ثبت. مثال::

        @register_space("lightgbm_quantile", version=1, n_hyperparams=8)
        def _space(trial: optuna.Trial) -> dict:
            return {
                "num_leaves": trial.suggest_int("num_leaves", 4, 256, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 200, log=True),
                ...
            }
    """
    def deco(fn: SpaceFn) -> SpaceFn:
        prev = SPACES.get(model_id)
        if prev is not None and version <= prev.version:
            raise ValueError(
                f"{model_id}: نسخه‌ی {version} باید بزرگ‌تر از نسخه‌ی ثبت‌شده‌ی قبلی "
                f"({prev.version}) باشد — بند 7.6.3 تغییر فضا را نسخه‌بندی می‌خواهد"
            )
        SPACES[model_id] = RegisteredSpace(model_id, version, n_hyperparams, fn)
        return fn
    return deco


def sample(model_id: str, trial: optuna.Trial) -> dict:
    """نمونه‌گیری یک پیکربندی از فضای ثبت‌شده‌ی ``model_id``."""
    space = SPACES.get(model_id)
    if space is None:
        raise KeyError(f"فضای هایپرپارامتر مدل {model_id!r} هنوز ثبت نشده")
    return space.fn(trial)


# ---------------------------------------------------------------------------
# بند 7.6.2 — بودجه‌ی trial بر اساس ابعاد فضا (قاعده‌ی انصاف A5)
# ---------------------------------------------------------------------------

#: (سقف تعداد هایپرپارامتر مؤثر, بودجه‌ی {S2, S3}) — به ترتیب صعودی سقف؛ آخرین ردیف
#: (۱۰+ پارامتر) طبق جدول بند 7.6.2 علاوه‌بر عدد trial به Hyperband/ASHA هم نیاز دارد —
#: آن انتخاب روش را ``trial_budget`` تعیین نمی‌کند، فقط شمارش trial را.
_BUDGET_TABLE: tuple[tuple[float, dict[str, int]], ...] = (
    (2, {"S2": 25, "S3": 50}),
    (5, {"S2": 60, "S3": 120}),
    (9, {"S2": 120, "S3": 250}),
    (float("inf"), {"S2": 150, "S3": 300}),
)


def trial_budget(n_hyperparams: int, stage: str) -> int:
    """حداقل بودجه‌ی trial **تضمین‌شده** برای مدلی با این تعداد هایپرپارامتر مؤثر (قاعده‌ی A5).

    ``stage`` فقط ``"S2"`` یا ``"S3"`` می‌پذیرد — S0/S1 بودجه‌ی ثابت خودشان را دارند
    (بند 7.3.1: S0=۱ برازش پیش‌فرض، S1=۲۰ trial تصادفی) و از این جدول نمی‌آیند.
    """
    if stage not in ("S2", "S3"):
        raise ValueError(f"trial_budget فقط برای S2/S3 معنا دارد، نه {stage!r}")
    for cap, budgets in _BUDGET_TABLE:
        if n_hyperparams <= cap:
            return budgets[stage]
    raise AssertionError("unreachable — آخرین سقف جدول بی‌نهایت است")
