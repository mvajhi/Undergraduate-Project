"""استخراج تقویم آموزشی دانشگاه تهران از PDF — مختصات‌محور، نه رونویسی دستی.

**چرا این ماژول وجود دارد.** تقویم آموزشی ۱۴۰۲-۱۴۰۳ در `src/data/calendar.py` دستی
رونویسی شده بود و دو بار (هر دو در ۲۰۲۶-۰۸-۱۳) اصلاح شد، چون خواندن یک جدول راست‌به‌چپ
از خروجی `pdftotext -layout` با چشم خطاپذیر است: ستون‌ها در متن چپ‌به‌راست چاپ می‌شوند
ولی محتوا راست‌به‌چپ است. با رسیدن سه PDF تازه (سه نیمسال دیگر) رونویسی دستی یعنی سه
برابر شدن همان ریسک. اینجا به‌جای چشم، از مختصات `x`/`y` هر واژه استفاده می‌شود.

**خودارزیابی.** لنگر تاریخ حدس زده نمی‌شود: همه‌ی شنبه‌های سال تحصیلی امتحان می‌شوند و
تنها شنبه‌ای پذیرفته می‌شود که **هر عدد روزِ هر سلول جدول** با حساب تقویمی بخواند
(`_solve_anchor`). اگر جدول بد خوانده شده باشد هیچ لنگری صددرصد جواب نمی‌دهد و تابع
خطا می‌دهد، نه یک تقویم اشتباهِ بی‌صدا.

دو چیدمان در PDFهای موجود هست و هر کدام یک تابع دارد:

| چیدمان | نمونه | رویداد کجاست | تابع |
|---|---|---|---|
| جدولی | `UT_calender_1402_1043.pdf`, `تقویم آموزشی-1403-1404.pdf` | داخل سلول روز | `parse_grid_calendar` |
| ستون رویداد | `تقویم_آموزشی.pdf`, `اصلاحیه_سوم…pdf` | جمله‌ی «۱۲ الی ۱۴ مهر: حذف و اضافه» در ستون کنار جدول | `parse_event_column_calendar` |

هر دو یک ساختار خروجی می‌دهند (`CalendarEvent`) تا مصرف‌کننده تفاوتشان را نبیند.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import jdatetime

logger = logging.getLogger(__name__)

#: شنبه=۰ … جمعه=۶ — همان قرارداد `src/data/calendar.py` (نرخ تطابق ۱.۰۰ با ستون
#: `DayOfWeek` فایل تجمیعی، بند ۲.۱ WBS).
WEEKDAY_HEADERS = ["شنبه", "یکشنبه", "دوشنبه", "سه شنبه", "چهارشنبه", "پنجشنبه", "جمعه"]

JALALI_MONTHS = {
    "فروردین": 1, "اردیبهشت": 2, "خرداد": 3, "تیر": 4, "مرداد": 5, "شهریور": 6,
    "مهر": 7, "آبان": 8, "آذر": 9, "دی": 10, "بهمن": 11, "اسفند": 12,
}


def norm_fa(text: str) -> str:
    """نرمال‌سازی متن استخراج‌شده از PDF فارسی.

    سه کار می‌کند: (۱) اشکال نمایشی عربی (`ﺷﻨﺒﻪ`) را به حروف پایه برمی‌گرداند،
    (۲) `ي`/`ك` عربی را به `ی`/`ک` فارسی می‌برد، (۳) نیم‌فاصله و نشانگرهای جهت را
    حذف می‌کند.

    ⚠️ لیگاتور `لا` در این PDFها (تولید Microsoft Word) وارونه استخراج می‌شود —
    `کلاس` به‌صورت `کالس` درمی‌آید. اینجا **تصحیح نمی‌شود**؛ به‌جایش الگوهای
    `EVENT_PATTERNS` هر دو رسم‌الخط را می‌پذیرند، چون وارونگی همیشگی نیست و تصحیح
    کورکورانه می‌تواند واژه‌های سالم را خراب کند.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("ي", "ی").replace("ك", "ک").replace("ۀ", "ه").replace("ة", "ه")
    for ch in "‌‎‏‪‫‬؜":
        text = text.replace(ch, "")
    return text.strip()


def squash(text: str) -> str:
    """حذف هر فاصله و نویسه‌ی غیرحرفی — برای تطبیق کلیدواژه روی متنی که واژه‌هایش
    در PDF تکه‌تکه شده‌اند (`میلا`+`د`+`ر`+`سو`+`ل` → `میلادرسول`)."""
    return re.sub(r"[^\w]", "", norm_fa(text))


# ---------------------------------------------------------------------------
# کلیدواژه‌ی رویدادها
# ---------------------------------------------------------------------------

#: هر رویداد: کد → فهرست الگوهای بی‌فاصله. ترتیب مهم است — اولین تطابق برنده است،
#: پس الگوهای خاص‌تر (`پایانامتحاناتمیانترم`) باید پیش از عام‌تر (`پایانامتحانات`) بیایند.
EVENT_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("midterm_start", ("شروعامتحاناتمیانترم", "شروعامتحانمیانترم")),
    ("midterm_end", ("پایانامتحاناتمیانترم", "پایانامتحانمیانترم")),
    ("midterm_mark", ("امتحاناتمیانترم", "امتحانمیانترم")),
    ("final_start", ("شروعامتحانات", "شروعامتحان")),
    ("final_end", ("پایانامتحانات",)),
    ("general_exam", ("امتحاندروسعمومی", "امتحاناتامتحاندروسعمومی")),
    # `کالس` = رسم‌الخط وارونه‌ی `کلاس` در خروجی PDF (توضیح در norm_fa)
    ("class_start", ("شروعکلاس", "شروعکالس")),
    ("class_end", ("پایانکلاس", "پایانکالس")),
    ("makeup_classes", ("کلاسهایجبرانی", "کالسهایجبرانی", "کلاسهاییجبرانی")),
    ("add_drop", ("حذفواضافه",)),
    ("emergency_drop", ("حذفاضطراری",)),
    ("final_control", ("کنترلنهایی",)),
    ("evaluation", ("ارزشیابی",)),
    ("registration", ("شروعثبتنام", "ثبتنامواانتخابواحد")),
]


def _transpositions(word: str) -> list[str]:
    """خود واژه به‌علاوه‌ی همه‌ی حالت‌هایی که دقیقاً دو نویسه‌ی مجاورش جابه‌جا شده‌اند."""
    return [word] + [word[:i] + word[i + 1] + word[i] + word[i + 2:] for i in range(len(word) - 1)]


def _fuzzy_contains(haystack: str, needle: str) -> bool:
    """تطبیق با تحمل **یک** جابه‌جایی دو نویسه‌ی مجاور.

    ⚠️ چرا لازم است: بعضی PDFها پسوند واژه را وارونه ذخیره می‌کنند. نمونه‌ی واقعی —
    `حذف اضطراری` در صفحه‌ی ۱ تقویم ۱۴۰۳-۱۴۰۴ به‌صورت دو span ذخیره شده:
    `«حذف اضطرا»` و `«ير»`، یعنی `ری` وارونه. تطبیق دقیق این سلول را **بی‌صدا** رد
    می‌کرد و سه روز حذف اضطراری از تقویم می‌افتاد. تحمل به یک جابه‌جایی محدود است
    تا تطبیق‌های تصادفی راه پیدا نکنند.
    """
    return any(v in haystack for v in _transpositions(needle))


def classify_event(text: str) -> str | None:
    """کد رویداد را از متن سلول/جمله درمی‌آورد؛ `None` یعنی متن رویداد تقویمی نیست
    (مثلاً نام یک تعطیلی مذهبی، که از پکیج `holidays` می‌آید نه از اینجا)."""
    squashed = squash(text)
    for code, patterns in EVENT_PATTERNS:
        if any(_fuzzy_contains(squashed, p) for p in patterns):
            return code
    return None


#: رویدادهایی که هر نیمسال **باید** داشته باشد. نبودشان یعنی استخراج یک سلول را از
#: دست داده — همان حالتی که تا پیش از `_fuzzy_contains` بی‌صدا رخ می‌داد.
REQUIRED_EVENT_CODES = frozenset({"class_start", "class_end", "add_drop"})


# ---------------------------------------------------------------------------
# ساختار خروجی
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalendarEvent:
    academic_year: str          # "1403-1404"
    semester: int               # ۱ یا ۲
    event_code: str             # کلید `EVENT_PATTERNS`
    date_jalali: str            # "1403-07-01"
    date_gregorian: str         # "2024-09-22"
    source_pdf: str
    source_page: int            # ۱-پایه
    raw_text: str


def _to_gregorian(y: int, m: int, d: int) -> str:
    return jdatetime.date(y, m, d).togregorian().strftime("%Y-%m-%d")


def _jalali_str(y: int, m: int, d: int) -> str:
    return f"{y:04d}-{m:02d}-{d:02d}"


# ---------------------------------------------------------------------------
# خواندن واژه‌ها از PDF
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Word:
    x0: float  # لبه‌ی چپ
    x1: float  # لبه‌ی راست
    y: float   # بالای واژه
    text: str

    @property
    def x(self) -> float:
        """مرکز افقی — مبنای تخصیص ستون."""
        return (self.x0 + self.x1) / 2


def read_words(pdf_path: Path | str, page: int) -> list[Word]:
    """واژه‌های یک صفحه با مختصاتشان. `page` صفر-پایه است."""
    import pymupdf

    with pymupdf.open(str(pdf_path)) as doc:
        raw = doc[page].get_text("words")
    words = []
    for x0, y0, x1, y1, text, *_ in raw:
        text = norm_fa(text)
        if text:
            words.append(Word(x0=x0, x1=x1, y=y0, text=text))
    return words


def _find_weekday_columns(words: list[Word]) -> list[float]:
    """مرکز افقی هفت ستون روز هفته، از سرستون جدول. خروجی به ترتیب شنبه…جمعه.

    سرستون‌ها گاهی تکه‌تکه‌اند (`سه`+`شنبه`) و گاهی در چند سطر (`شنبه` و `یکشنبه` با
    `y` متفاوت). پس به‌جای تکیه بر یک سطر، همه‌ی واژه‌های `*شنبه`/`جمعه` در نیمه‌ی
    بالایی صفحه جمع می‌شوند و هفت خوشه‌ی افقیِ راست‌ترین انتخاب می‌شوند.
    """
    y_limit = max(w.y for w in words) * 0.35
    cands = [w for w in words if w.y < y_limit and w.text in {"شنبه", "جمعه", "یکشنبه", "دوشنبه", "پنجشنبه", "چهارشنبه"}]
    if len(cands) < 6:
        raise ValueError(f"سرستون روزهای هفته پیدا نشد (فقط {len(cands)} نامزد)")

    # `جمعه` چپ‌ترین ستون است و `شنبه` راست‌ترین؛ فاصله‌ی ستون‌ها از همین دو درمی‌آید.
    x_fri = min(w.x for w in cands if w.text == "جمعه")
    x_sat = max(w.x for w in cands)
    step = (x_sat - x_fri) / 6
    return [x_sat - i * step for i in range(7)]


def _column_of(x: float, centers: list[float]) -> int | None:
    """نزدیک‌ترین ستون روز؛ `None` اگر واژه بیرون از جدول باشد (ستون «ماه»/«هفته»
    یا ستون رویداد)."""
    step = abs(centers[0] - centers[-1]) / 6
    best = min(range(7), key=lambda i: abs(x - centers[i]))
    return best if abs(x - centers[best]) <= step * 0.6 else None


_MONTH_IN_TOKEN = re.compile("|".join(JALALI_MONTHS))


def _day_number(text: str) -> int | None:
    """عدد روز از یک واژه.

    PDF گاهی عدد و نام ماه را به یک واژه می‌چسباند، و **ترتیبشان ثابت نیست** —
    `«۱مهر»` در یک صفحه و `«تیر۲»` در صفحه‌ی بعد (بازمانده‌ی راست‌به‌چپ‌بودن متن).
    پس نام ماه حذف می‌شود و بقیه‌ی واژه اگر یک عدد ۱ تا ۳۱ بود، همان عدد روز است.
    اعداد سه‌رقمی به بالا (سال تحصیلی `۱۴۰۳`) رد می‌شوند.
    """
    if re.fullmatch(r"\d{3,}", text):
        return None
    stripped = _MONTH_IN_TOKEN.sub("", text).strip()
    if not re.fullmatch(r"\d{1,2}", stripped):
        return None
    value = int(stripped)
    return value if 1 <= value <= 31 else None


# ---------------------------------------------------------------------------
# چیدمان جدولی — رویداد داخل سلول
# ---------------------------------------------------------------------------


def _build_day_grid(words: list[Word], centers: list[float]) -> list[dict]:
    """سطرهای جدول: هر سطر `{"y": float, "days": {col: day_number}}`.

    سطرها از روی `y` عددهای روز خوشه‌بندی می‌شوند (نه از روی برچسب «هفته اول/دوم»،
    که در بعضی PDFها تکراری است — مثلاً «هفته هفتم» دوبار).
    """
    numbers = []
    for w in words:
        col = _column_of(w.x, centers)
        day = _day_number(w.text)
        if col is not None and day is not None:
            numbers.append((w.y, col, day))
    numbers.sort()

    rows: list[dict] = []
    for y, col, day in numbers:
        if rows and abs(y - rows[-1]["y"]) <= 8 and col not in rows[-1]["days"]:
            rows[-1]["days"][col] = day
        else:
            rows.append({"y": y, "days": {col: day}})

    # سطرهایی با کمتر از سه روز معمولاً سرریز متن‌اند نه سطر تقویم
    return [r for r in rows if len(r["days"]) >= 3]


#: کمینه‌ی نسبت سلول‌هایی که باید با لنگر بخوانند. ۱۰۰٪ نیست چون خودِ PDFهای دانشگاه
#: غلط چاپی دارند (نمونه‌ی مستند: هفته‌ی پانزدهم خرداد ۱۴۰۴ به‌جای ۱۰…۱۶ عددهای
#: ۱۰،۱۲،۱۳،۱۳،۱۴،۱۵،۱۶ را نشان می‌دهد). آستانه آن‌قدر بالاست که یک لنگرِ غلط — که
#: تقریباً همه‌ی سلول‌ها را خراب می‌کند — هرگز از آن رد نشود.
ANCHOR_MIN_MATCH = 0.95


def _solve_anchor(rows: list[dict], year_start: int) -> tuple[jdatetime.date, list[str]]:
    """تاریخ شنبه‌ی سلول بالا-راست جدول را پیدا می‌کند — با آزمون، نه با حدس.

    همه‌ی شنبه‌های سال تحصیلی نامزدند و برای هرکدام شمرده می‌شود چند سلول با حساب
    `لنگر + ۷×سطر + ستون` می‌خوانند. بهترین نامزد باید هم از `ANCHOR_MIN_MATCH` رد
    شود و هم **یکتا** باشد؛ وگرنه خطا داده می‌شود — یعنی جدول بد استخراج شده و
    نتیجه‌ی غلط بی‌صدا تولید نمی‌شود.

    خروجی دوم فهرست سلول‌های ناخوان است. خالی‌نبودنش یعنی PDF مبدأ غلط چاپی دارد؛
    مصرف‌کننده باید گزارشش کند، نه نادیده بگیرد.
    """
    total = sum(len(row["days"]) for row in rows)
    if total == 0:
        raise ValueError("جدول هیچ عدد روزی ندارد")

    def mismatches_for(anchor: jdatetime.date) -> list[str]:
        bad = []
        for r, row in enumerate(rows):
            for c, expected in sorted(row["days"].items()):
                actual = anchor + jdatetime.timedelta(days=7 * r + c)
                if actual.day != expected:
                    bad.append(f"سطر {r} ستون {c}: PDF «{expected}» ولی حساب «{actual.day}»")
        return bad

    scored = []
    day = jdatetime.date(year_start, 5, 1)
    end = jdatetime.date(year_start + 1, 7, 1)
    while day <= end:
        if day.weekday() == 0:  # jdatetime: شنبه = ۰
            bad = mismatches_for(day)
            scored.append((len(bad), day, bad))
        day += jdatetime.timedelta(days=1)

    scored.sort(key=lambda t: t[0])
    n_bad, best, bad = scored[0]
    if 1 - n_bad / total < ANCHOR_MIN_MATCH:
        raise ValueError(
            f"هیچ لنگری با جدول نخواند (سال {year_start}، بهترین نامزد {best} با "
            f"{n_bad} از {total} سلول ناخوان) — جدول احتمالاً بد استخراج شده است"
        )
    if len(scored) > 1 and scored[1][0] == n_bad:
        raise ValueError(f"لنگر یکتا نیست: {best} و {scored[1][1]} هر دو {n_bad} خطا دارند")
    return best, bad


def parse_grid_calendar(
    pdf_path: Path | str,
    page: int,
    academic_year: str,
    semester: int,
) -> list[CalendarEvent]:
    """چیدمان جدولی: متن رویداد زیر عدد روز، در همان سلول."""
    pdf_path = Path(pdf_path)
    words = read_words(pdf_path, page)
    centers = _find_weekday_columns(words)
    rows = _build_day_grid(words, centers)
    anchor, source_typos = _solve_anchor(rows, int(academic_year.split("-")[0]))
    if source_typos:
        logger.warning(
            "غلط چاپی در %s صفحه‌ی %d (لنگر %s): %s",
            pdf_path.name, page + 1, anchor, " · ".join(source_typos),
        )

    row_tops = [r["y"] for r in rows]

    def row_of(y: float) -> int | None:
        """سطری که یک واژه‌ی رویداد به آن تعلق دارد: آخرین سطری که عددش بالای واژه است."""
        prior = [i for i, top in enumerate(row_tops) if top - 2 <= y]
        return prior[-1] if prior else None

    # متن رویداد در PDF تکه‌تکه است؛ تکه‌های هم‌سلول (همان سطر، همان ستون، `y` نزدیک)
    # به‌ترتیب راست‌به‌چپ به‌هم چسبانده می‌شوند.
    cells: dict[tuple[int, int], list[tuple[float, float, str]]] = {}
    for w in words:
        col = _column_of(w.x, centers)
        if col is None or _day_number(w.text) is not None:
            continue
        row = row_of(w.y)
        if row is None:
            continue
        cells.setdefault((row, col), []).append((w.y, -w.x, w.text))

    events: list[CalendarEvent] = []
    for (row, col), frags in sorted(cells.items()):
        frags.sort()
        text = "".join(t for _, _, t in frags)
        code = classify_event(text)
        if code is None:
            continue
        date = anchor + jdatetime.timedelta(days=7 * row + col)
        events.append(
            CalendarEvent(
                academic_year=academic_year,
                semester=semester,
                event_code=code,
                date_jalali=_jalali_str(date.year, date.month, date.day),
                date_gregorian=date.togregorian().strftime("%Y-%m-%d"),
                source_pdf=pdf_path.name,
                source_page=page + 1,
                raw_text=text,
            )
        )
    return events


# ---------------------------------------------------------------------------
# چیدمان ستون رویداد — جمله‌ی «۱۲ الی ۱۴ مهر: حذف و اضافه»
# ---------------------------------------------------------------------------

_MONTH_RE = "|".join(JALALI_MONTHS)

#: بیشینه‌ی فاصله‌ی افقی (point) بین دو تکه‌ی رقمی که یک عدد واحدند. در PDFهای مبدأ
#: این تکه‌ها دقیقاً به‌هم چسبیده‌اند (`۱`@[۱۶۷٫۰۰, ۱۷۰٫۱۰] و `۸`@[۱۷۰٫۱۲, ۱۷۵٫۲۳])
#: در حالی که دو عدد جدا دست‌کم ۱۰ point فاصله دارند.
_DIGIT_JOIN_GAP = 1.5

_DIGITS_ONLY = re.compile(r"^\d+$")


def _merge_digit_runs(words: list[Word]) -> list[str]:
    """واژه‌های یک خط (مرتب‌شده راست‌به‌چپ) را به رشته‌های خوانا تبدیل می‌کند.

    ⚠️ چرا لازم است: متن فارسی راست‌به‌چپ است ولی **عددها داخل همان متن چپ‌به‌راست**
    خوانده می‌شوند. استخراج‌کننده گاهی یک عدد دورقمی را دو واژه می‌کند؛ بازچینی
    راست‌به‌چپِ ساده آن را وارونه می‌کند و «۱۸ اسفند» به «۸ ۱ اسفند» تبدیل می‌شود —
    که بی‌صدا به «۸ اسفند» تفسیر می‌شد. اینجا تکه‌های رقمیِ **چسبیده** دوباره
    چپ‌به‌راست به‌هم وصل می‌شوند.
    """
    out: list[str] = []
    i = 0
    while i < len(words):
        if _DIGITS_ONLY.match(words[i].text):
            run = [words[i]]
            # واژه‌ها راست‌به‌چپ مرتب‌اند، پس واژه‌ی بعدی چپِ فعلی است.
            while (
                i + 1 < len(words)
                and _DIGITS_ONLY.match(words[i + 1].text)
                and abs(run[-1].x0 - words[i + 1].x1) <= _DIGIT_JOIN_GAP
            ):
                i += 1
                run.append(words[i])
            out.append("".join(w.text for w in sorted(run, key=lambda w: w.x0)))
        else:
            out.append(words[i].text)
        i += 1
    return out


def _event_column_lines(words: list[Word], centers: list[float]) -> list[str]:
    """جمله‌های ستون رویداد، هر خط یک رشته (راست‌به‌چپ بازچینی‌شده)."""
    step = abs(centers[0] - centers[-1]) / 6
    x_limit = min(centers) - step * 0.6  # هرچه چپ‌تر از ستون جمعه
    outside = [w for w in words if w.x < x_limit]

    lines: dict[int, list[Word]] = {}
    for w in outside:
        lines.setdefault(round(w.y / 4), []).append(w)

    out = []
    for key in sorted(lines):
        ws = sorted(lines[key], key=lambda w: -w.x0)
        out.append(" ".join(_merge_digit_runs(ws)))
    return out


def _parse_date_expression(text: str, year_start: int) -> list[tuple[int, int, int]]:
    """بخشِ پیش از «:» را به فهرست `(سال, ماه, روز)` تبدیل می‌کند.

    سه شکلی که در این PDFها دیده می‌شود، همگی پشتیبانی می‌شوند:

    | نمونه | تفسیر |
    |---|---|
    | `۲ اسفند` | یک روز |
    | `۱۶ الی ۱۸ اسفند` · `۲۵ و ۲۶ اسفند` | چند روز، یک ماه |
    | `۲۳ خرداد الی ۳ تیر` | بازه‌ی **بین دو ماه** |

    قاعده‌ی تخصیص ماه: هر عدد به اولین نام ماهی که **پس از** آن آمده می‌چسبد؛ اگر
    ماهی پس از آن نبود، به آخرین ماه دیده‌شده. همین قاعده حالت سوم را درست می‌کند،
    که نسخه‌ی رگکس-تک‌ماهه‌ی قبلی بی‌صدا فقط انتهای بازه را می‌گرفت.
    """
    tokens = re.findall(rf"\d{{1,2}}(?!\d)|{_MONTH_RE}", text)
    pending: list[int] = []
    resolved: list[tuple[int, int]] = []  # (ماه, روز)
    for token in tokens:
        if token in JALALI_MONTHS:
            month = JALALI_MONTHS[token]
            resolved.extend((month, d) for d in pending)
            pending = []
        else:
            pending.append(int(token))
    if pending and resolved:  # عددهای بی‌ماه در انتها → آخرین ماه
        resolved.extend((resolved[-1][0], d) for d in pending)

    out = []
    for month, day in resolved:
        if not 1 <= day <= 31:
            continue
        # سال تحصیلی از شهریور شروع می‌شود: ماه‌های ۶ به بعد سال اول، ماه‌های ۱ تا ۵
        # سال دوم (نیمسال دوم از بهمن ۱۴۰۴ تا مرداد ۱۴۰۵ می‌رود).
        out.append((year_start if month >= 6 else year_start + 1, month, day))
    return out


def parse_event_column_calendar(
    pdf_path: Path | str,
    page: int,
    academic_year: str,
    semester: int,
) -> list[CalendarEvent]:
    """چیدمان ستون رویداد: تاریخ صریح در خودِ جمله است، پس جدول فقط برای یافتن مرز
    ستون رویداد لازم است و لنگر تاریخ حل نمی‌شود."""
    pdf_path = Path(pdf_path)
    words = read_words(pdf_path, page)
    centers = _find_weekday_columns(words)
    year_start = int(academic_year.split("-")[0])

    events: list[CalendarEvent] = []
    for line in _event_column_lines(words, centers):
        head, sep, label = line.partition(":")
        if not sep:
            continue
        code = classify_event(label)
        if code is None:
            continue

        dates = _parse_date_expression(head, year_start)
        if not dates:
            continue

        # جمله می‌تواند بازه («۱۲ الی ۱۴ مهر») یا فهرست («۲۱ و ۲۲ مهر») باشد. هر دو
        # به‌صورت «اولین و آخرین» ثبت می‌شوند؛ پرکردن میانه کار مصرف‌کننده است و
        # `raw_text` همیشه جمله‌ی کامل را نگه می‌دارد.
        for y, m_, d in sorted({min(dates), max(dates)}):
            events.append(
                CalendarEvent(
                    academic_year=academic_year,
                    semester=semester,
                    event_code=code,
                    date_jalali=_jalali_str(y, m_, d),
                    date_gregorian=_to_gregorian(y, m_, d),
                    source_pdf=pdf_path.name,
                    source_page=page + 1,
                    raw_text=line.strip(),
                )
            )
    return events


# ---------------------------------------------------------------------------
# بلوک نوروز — از برچسب ستون «هفته»، نه از عدد ثابت در کد
# ---------------------------------------------------------------------------


def _week_label_rows(words: list[Word], centers: list[float]) -> list[tuple[float, str]]:
    """برچسب ستون «هفته» هر سطر: `(y, «هفتهاول»)`. برچسب‌ها در PDF تکه‌تکه‌اند، پس
    تکه‌های هم‌ارتفاع به‌هم چسبانده و فاصله‌هایشان حذف می‌شود."""
    step = abs(centers[0] - centers[-1]) / 6
    right_of_saturday = [w for w in words if w.x > centers[0] + step * 0.6]

    lines: dict[int, list[Word]] = {}
    for w in right_of_saturday:
        lines.setdefault(round(w.y / 4), []).append(w)

    out = []
    for key in sorted(lines):
        ws = sorted(lines[key], key=lambda w: -w.x0)
        out.append((key * 4.0, squash("".join(w.text for w in ws))))
    return out


def _nowruz_dates_from_grid(
    words: list[Word],
    centers: list[float],
    rows: list[dict],
    anchor: jdatetime.date,
) -> list[str]:
    """تاریخ‌های هفته‌های برچسب‌خورده‌ی «نوروز» — چیدمان جدولی، با حساب لنگر."""
    labels = _week_label_rows(words, centers)
    dates: list[str] = []
    for y, label in labels:
        if "نوروز" not in label:
            continue
        row = min(range(len(rows)), key=lambda i: abs(rows[i]["y"] - y))
        if abs(rows[row]["y"] - y) > 14:
            continue
        for col in sorted(rows[row]["days"]):
            d = anchor + jdatetime.timedelta(days=7 * row + col)
            dates.append(_jalali_str(d.year, d.month, d.day))
    return sorted(set(dates))


def _nowruz_dates_from_day_numbers(
    words: list[Word], centers: list[float], year_start: int
) -> list[str]:
    """تاریخ‌های هفته‌های «نوروز» — چیدمان ستون رویداد، بدون لنگر.

    در این چیدمان هفته‌ها سرِ مرز ماه به دو سطر فیزیکی می‌شکنند، پس حساب
    `لنگر + ۷×سطر + ستون` کار نمی‌کند. ولی لازم هم نیست: نوروز همیشه در فروردین است،
    پس خودِ عددهای روزِ همان سطر کافی‌اند.
    """
    labels = _week_label_rows(words, centers)
    nowruz_y = [y for y, label in labels if "نوروز" in label]
    if not nowruz_y:
        return []

    dates = []
    for w in words:
        day = _day_number(w.text)
        if day is None or _column_of(w.x, centers) is None:
            continue
        if any(abs(w.y - y) <= 8 for y in nowruz_y):
            dates.append(_jalali_str(year_start + 1, 1, day))
    return sorted(set(dates))


@dataclass(frozen=True)
class PageParse:
    """نتیجه‌ی کامل خواندن یک صفحه‌ی تقویم."""

    events: list[CalendarEvent]
    nowruz_dates: list[str]      # تاریخ جلالی، از برچسب «نوروز» ستون هفته
    source_typos: list[str]      # سلول‌های ناخوانِ خودِ PDF (خالی = بی‌عیب)


def parse_calendar_page(
    pdf_path: Path | str,
    page: int,
    academic_year: str,
    semester: int,
    layout: str,
) -> PageParse:
    """نقطه‌ی ورود واحد — `layout` یکی از `"grid"` یا `"event_column"`."""
    pdf_path = Path(pdf_path)
    year_start = int(academic_year.split("-")[0])
    words = read_words(pdf_path, page)
    centers = _find_weekday_columns(words)

    if layout == "grid":
        rows = _build_day_grid(words, centers)
        anchor, typos = _solve_anchor(rows, year_start)
        return PageParse(
            events=parse_grid_calendar(pdf_path, page, academic_year, semester),
            nowruz_dates=_nowruz_dates_from_grid(words, centers, rows, anchor),
            source_typos=typos,
        )
    if layout == "event_column":
        return PageParse(
            events=parse_event_column_calendar(pdf_path, page, academic_year, semester),
            nowruz_dates=_nowruz_dates_from_day_numbers(words, centers, year_start),
            source_typos=[],
        )
    raise ValueError(f"چیدمان ناشناخته: {layout!r}")
