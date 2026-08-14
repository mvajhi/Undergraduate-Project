"""بند 7.7.1 سند فاز ۷ — نام‌گذاری run و بازتولیدپذیری آن.

قالب اجباری::

    {family}_{model}_{level}_{target}_{featureset}_{tau}_{stage}_{seed}_{timestamp}
    مثال: F02_lightgbm_L1_rho_FSlgbm_t010_S2_s42_20260815T1130

هر اسکریپت/نوت‌بوک آموزش مدل باید نام runاش را با ``run_name()`` بسازد، نه دستی — تا قالب
هرگز به‌اشتباه ننوشته شود و بشود بعداً با ``parse_run_name()`` از روی نام run در MLflow
پرس‌وجو کرد، بدون تکیه‌ی صرف بر paramهای جداگانه.
"""

import re
from datetime import datetime

from src.models.registry import FAMILIES, LEVELS, STAGES

_MODEL_ID_RE = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")

_RUN_NAME_RE = re.compile(
    r"^(?P<family>F\d{2})_(?P<model>[a-z0-9]+(?:_[a-z0-9]+)*)_(?P<level>L\d)_"
    r"(?P<target>[a-z0-9]+)_(?P<feature_set>[A-Za-z0-9]+)_(?P<tau>t\d{3})_"
    r"(?P<stage>S\d)_s(?P<seed>\d+)_(?P<timestamp>\d{8}T\d{4})$"
)


def tau_code(tau: float) -> str:
    """۰.۱۰ → ``'t010'``، ۰.۰۲ → ``'t002'``."""
    if not 0 < tau < 1:
        raise ValueError(f"tau باید در بازه‌ی باز (0,1) باشد: {tau}")
    return f"t{round(tau * 100):03d}"


def tau_from_code(code: str) -> float:
    m = re.fullmatch(r"t(\d{3})", code)
    if not m:
        raise ValueError(f"قالب کد τ نامعتبر: {code!r} (باید مثل 't010' باشد)")
    return int(m.group(1)) / 100


def run_name(*, family: str, model: str, level: str, target: str, feature_set: str,
            tau: float, stage: str, seed: int, timestamp: datetime | None = None) -> str:
    """نام run طبق قالب بند 7.7.1. تمام اجزا اعتبارسنجی می‌شوند تا نام نامعتبر اصلاً ساخته نشود."""
    if family not in FAMILIES:
        raise ValueError(f"خانواده‌ی نامعتبر: {family!r} (باید یکی از {sorted(FAMILIES)} باشد)")
    if level not in LEVELS:
        raise ValueError(f"سطح نامعتبر: {level!r} (باید یکی از {LEVELS} باشد)")
    if stage not in STAGES:
        raise ValueError(f"مرحله‌ی نامعتبر: {stage!r} (باید یکی از {STAGES} باشد)")
    if not _MODEL_ID_RE.match(model):
        raise ValueError(f"شناسه‌ی مدل باید snake_case حروف کوچک/رقم باشد: {model!r}")
    if seed < 0:
        raise ValueError(f"seed نمی‌تواند منفی باشد: {seed}")

    ts = (timestamp or datetime.now()).strftime("%Y%m%dT%H%M")
    return "_".join([family, model, level, target, feature_set, tau_code(tau), stage, f"s{seed}", ts])


def parse_run_name(name: str) -> dict:
    """معکوس ``run_name`` — بازگرداندن اجزا از روی رشته‌ی نام run."""
    m = _RUN_NAME_RE.fullmatch(name)
    if not m:
        raise ValueError(f"run_name با قالب بند 7.7.1 مطابق نیست: {name!r}")
    d = m.groupdict()
    d["tau"] = tau_from_code(d["tau"])
    d["seed"] = int(d["seed"])
    return d
