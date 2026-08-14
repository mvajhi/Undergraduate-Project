"""بند 7.4 سند فاز ۷ — کارت مدل آزمایشی: ساخت/خواندن/سنجش کامل‌بودن.

هر مدلی که وارد S2 می‌شود باید یک کارت در ``reports/models/{model_id}.md`` داشته باشد که
دقیقاً ۱۴ بخش قالب (``reports/models/_TEMPLATE.md``) را پر کند. این ماژول فقط اسکلت و
سنجه‌ی کامل‌بودن را می‌سازد — محتوای هر بخش را خودِ کد آموزش/تحلیل هر مدل می‌نویسد.

⚠️ **چهار گام غیرقابل‌حذف** (بند 7.4): ۴ (آزمون‌های پیش‌پرواز)، ۹ (شاهد همگرایی A6)،
۱۰ (تحلیل حساسیت هایپرپارامتر)، ۱۳ (کالیبراسیون و پوشش). کارت بدون این‌ها ناقص است و
مدل وارد جدول مقایسه‌ی نهایی (بند 7.25) نمی‌شود.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

from src.config import REPORTS_DIR

CARDS_DIR = REPORTS_DIR / "models"
TEMPLATE_PATH = CARDS_DIR / "_TEMPLATE.md"

PLACEHOLDER = "_ثبت نشده._"

#: عنوان دقیق هر یک از ۱۴ گام، به همان ترتیب و شماره‌ی بند 7.4
STEPS: tuple[str, ...] = (
    "جایگاه نظری",
    "سطوح داده‌ی قابل‌اعمال",
    "نگاشت هدف",
    "آزمون‌های پیش‌پرواز",
    "فیچرست اختصاصی",
    "پیش‌پردازش",
    "فضای هایپرپارامتر",
    "راهبرد و بودجه‌ی جستجو",
    "شاهد همگرایی (A6)",
    "تحلیل حساسیت هایپرپارامتر",
    "تشخیص برازش",
    "استخراج کوانتایل",
    "کالیبراسیون و پوشش",
    "جمع‌بندی و حکم",
)

#: این گام‌ها (شماره‌ی ۱-پایه) طبق بند 7.4 «قابل‌حذف نیستند حتی برای مدل‌های ساده»
MANDATORY_STEPS: tuple[int, ...] = (4, 9, 10, 13)

_HEADER_RE = re.compile(r"^##\s*(\d+)\.")


@dataclass
class ModelCard:
    model_id: str
    #: شماره‌ی گام (۱..۱۴) → متن Markdown آن بخش
    sections: dict[int, str] = field(default_factory=dict)

    def set_section(self, step: int, content: str) -> None:
        if not 1 <= step <= len(STEPS):
            raise ValueError(f"شماره‌ی گام باید ۱ تا {len(STEPS)} باشد: {step}")
        content = content.strip()
        if content:
            self.sections[step] = content
        else:
            self.sections.pop(step, None)

    def missing_mandatory(self) -> list[int]:
        return [s for s in MANDATORY_STEPS if not self.sections.get(s)]

    def is_complete(self) -> bool:
        """کارت کامل یعنی هر ۱۴ گام نوشته شده — نه فقط چهار گام اجباری."""
        return len(self.sections) == len(STEPS) and not self.missing_mandatory()

    def require_complete(self) -> None:
        """در نقطه‌ی «نهایی‌سازی برای جدول مقایسه» صدا زده شود، نه هنگام ذخیره‌ی پیش‌نویس."""
        missing_all = [i for i in range(1, len(STEPS) + 1) if not self.sections.get(i)]
        if missing_all:
            raise ValueError(
                f"کارت {self.model_id!r} ناقص است — گام‌های غایب: {missing_all} "
                f"(اجباری‌ها در این فهرست: {[m for m in missing_all if m in MANDATORY_STEPS]})"
            )

    def to_markdown(self) -> str:
        lines = [f"# کارت مدل — {self.model_id}", ""]
        for i, title in enumerate(STEPS, start=1):
            star = " ⚠️" if i in MANDATORY_STEPS else ""
            lines += [f"## {i}. {title}{star}", "", self.sections.get(i, PLACEHOLDER), ""]
        return "\n".join(lines)

    def save(self, path: Path | None = None) -> Path:
        """همیشه می‌نویسد، حتی اگر ناقص باشد — کارت‌ها تدریجی تکمیل می‌شوند. برای اطمینان از
        کامل‌بودن پیش از ورود به جدول مقایسه از ``require_complete()`` استفاده کنید."""
        out = path or (CARDS_DIR / f"{self.model_id}.md")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.to_markdown())
        return out


def load(model_id: str, path: Path | None = None) -> ModelCard:
    """بازخوانی یک کارت ذخیره‌شده — برای پرسش برنامه‌ای «کدام مدل‌ها کارت کامل دارند؟»
    (دروازه‌ی M4، بند 7.29.2)."""
    p = path or (CARDS_DIR / f"{model_id}.md")
    card = ModelCard(model_id=model_id)
    current: int | None = None
    buf: list[str] = []

    def flush() -> None:
        if current is not None:
            card.set_section(current, "\n".join(buf))

    for line in p.read_text().splitlines():
        m = _HEADER_RE.match(line)
        if m:
            flush()
            current = int(m.group(1))
            buf = []
        elif current is not None:
            buf.append(line)
    flush()

    for step, text in list(card.sections.items()):
        if text == PLACEHOLDER:
            del card.sections[step]
    return card
