"""Persian text normalization (WBS بند ۳.۲) — پیش‌نیاز هر نگاشت/تطبیق در فاز ۳.

بدون وابستگی به `hazm`/`parsivar` (گزینه‌ی سوم WBS): این تبدیل‌ها صرفاً رشته‌ای‌اند
(یکسان‌سازی کاراکتر، حذف کاراکتر نامرئی، جمع‌کردن فاصله) و نیازی به کتابخانه‌ی
پردازش زبان طبیعی سنگین ندارند.
"""

import re
import unicodedata

# عربی → فارسی: ي→ی, ك→ک, ى (الف مقصوره)→ی
_AR_TO_FA_LETTERS = str.maketrans({"ي": "ی", "ك": "ک", "ى": "ی"})

# ارقام عربی/فارسی → لاتین
_DIGIT_MAP = str.maketrans(
    {
        **{chr(0x0660 + i): str(i) for i in range(10)},  # ٠-٩
        **{chr(0x06F0 + i): str(i) for i in range(10)},  # ۰-۹
    }
)

# اعراب عربی (فتحه/ضمه/کسره/سکون/تشدید/تنوین...) + کشیده (تطویل)
_DIACRITICS_RE = re.compile(r"[ً-ْـ]")

# کاراکترهای نامرئی/جهت‌ده که نباید در متن قابل‌تطبیق باقی بمانند: RLM, LRM, ZWJ,
# BOM، و باقیمانده‌ی متن آلوده‌ی خروجی اکسل ویندوز (`_x000D_`, `\r`).
_INVISIBLE_RE = re.compile(r"[‎‏‍﻿]")
_WINDOWS_ARTIFACT_RE = re.compile(r"_x000D_|\r")

# نیم‌فاصله (ZWNJ, U+200C): برای هدف تطبیق/نگاشت به فاصله‌ی معمولی تبدیل می‌شود
# (نه حذف کامل) تا اختلاف نیم‌فاصله/فاصله/بدون‌فاصله بین دو منبع مانع تطبیق نشود.
_ZWNJ_RE = re.compile(r"‌")

_MULTI_SPACE_RE = re.compile(r"\s+")


def normalize_persian_text(s: str | float | None) -> str | float | None:
    """یک رشته را برای تطبیق/نگاشت قابل‌اعتماد نرمال می‌کند.

    `NaN`/`None` بدون تغییر برمی‌گردد (پانداس ستون‌های گمشده را همین‌طور پاس می‌دهد).
    """
    if not isinstance(s, str):
        return s

    s = unicodedata.normalize("NFKC", s)
    s = s.translate(_AR_TO_FA_LETTERS)
    s = s.translate(_DIGIT_MAP)
    s = _WINDOWS_ARTIFACT_RE.sub("", s)
    s = _INVISIBLE_RE.sub("", s)
    s = _ZWNJ_RE.sub(" ", s)
    s = _DIACRITICS_RE.sub("", s)
    s = _MULTI_SPACE_RE.sub(" ", s).strip()
    return s


def normalize_columns(df, columns: list[str]):
    """`normalize_persian_text` را روی چند ستون یک DataFrame اعمال می‌کند (in-place-safe: کپی برمی‌گرداند)."""
    df = df.copy()
    for col in columns:
        df[col] = df[col].map(normalize_persian_text)
    return df


if __name__ == "__main__":
    samples = [
        "سطح شهر-پروفسورحسابی",
        "خوراک سالاد الویه+دوغ",
        "مرد_x000D_\n",
        "روزانه_x000D_\n",
        "چلو خورش  قیمه   سیب زمینی",
        "١٤٠٢/٩/١",
        "كباب كوبيده",
    ]
    for s in samples:
        print(repr(s), "->", repr(normalize_persian_text(s)))
