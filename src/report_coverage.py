"""ضمانت مکانیکی نقشه‌ی منبع — بند «چرا این سند وجود دارد» در `doc/report-source-map.md`.

بدون این اسکریپت، `doc/report-source-map.md` و `doc/findings_register.md` فقط یک امیدند:
ممکن است روز نوشتنشان کامل باشند ولی فردا که گزارش جدیدی در `reports/` اضافه شد، بی‌صدا
قدیمی شوند. این اسکریپت با **شکست سخت** (exit code≠۰) تضمین می‌کند:

1. هر فایل `.md` زیر `reports/` (به‌جز `_TEMPLATE.md`) — با نام یا با ریشه‌ی نامش (بدون
   پسوند، چون جدول‌ها گاه از نماد براکت `{a,b,c}.md` استفاده می‌کنند) — در
   `doc/report-source-map.md` ظاهر شده باشد.
2. هر ردیف شماره‌دار `doc/decision_log.md` حداقل یک‌بار در `doc/report-source-map.md` یا
   `doc/findings_register.md` ارجاع شده باشد — چه به‌صورت نثر «ردیف N»، چه به‌صورت عدد خام
   داخل ستون «تصمیم‌ها (ردیف)» یک جدول (شامل فهرست‌های کاما-جدا و بازه‌های خط‌فاصله‌ای).
3. هر حقیقت `F##` در `doc/data_facts_register.md` حداقل یک‌بار در `doc/report-source-map.md`
   ارجاع شده باشد.
4. هر یافته‌ای که در `doc/decision_log.md` به‌صورت «یافته‌ی N» ارجاع شده، در
   `doc/findings_register.md` شناسه‌ی متناظر داشته باشد.

اجرا: ``python -m src.report_coverage`` — خروجی کامل در `reports/phase11/coverage_gaps.md`.
"""

import re
import sys
from pathlib import Path

from src.config import REPORTS_DIR, ROOT_DIR

DOC_DIR = ROOT_DIR / "doc"
SOURCE_MAP = DOC_DIR / "report-source-map.md"
FINDINGS_REGISTER = DOC_DIR / "findings_register.md"
DECISION_LOG = DOC_DIR / "decision_log.md"
FACTS_REGISTER = DOC_DIR / "data_facts_register.md"
OUT_PATH = REPORTS_DIR / "phase11" / "coverage_gaps.md"

_PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
_ASCII_DIGITS = "0123456789"
_DIGIT_MAP = str.maketrans(_PERSIAN_DIGITS, _ASCII_DIGITS)

#: دنباله‌ای از عدد/جداکننده (کاما، ویرگول فارسی، «و»، فاصله، خط‌فاصله‌ی بازه) پس از «ردیف»
_ROW_LIST_RE = re.compile(
    r"ردیف(?:‌های)?\s*((?:[0-9]+\s*(?:[-–−]\s*[0-9]+)?\s*(?:[،,]|\s+و\s+)?\s*)+)")
_RANGE_RE = re.compile(r"^([0-9]+)\s*[-–−]\s*([0-9]+)$")
_NUMBER_TOKEN_RE = re.compile(r"[0-9]+(?:\s*[-–−]\s*[0-9]+)?")


def to_ascii_digits(text: str) -> str:
    return text.translate(_DIGIT_MAP)


def _expand_number_tokens(chunk: str) -> set[int]:
    out = set()
    for tok in _NUMBER_TOKEN_RE.findall(chunk):
        m = _RANGE_RE.match(tok.strip())
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            out.update(range(lo, hi + 1))
        else:
            out.add(int(tok.strip()))
    return out


def find_report_files() -> list[Path]:
    return sorted(p for p in REPORTS_DIR.rglob("*.md")
                 if p.name != "_TEMPLATE.md" and p != OUT_PATH)


def find_decision_log_rows(text: str) -> set[int]:
    return {int(m) for m in re.findall(r"^\|\s*(\d+)\s*\|", text, flags=re.MULTILINE)}


def find_fact_ids(text: str) -> set[str]:
    return set(re.findall(r"\bF(\d{2})\b", text))


def find_referenced_decision_rows(text: str) -> set[int]:
    ascii_text = to_ascii_digits(text)
    rows: set[int] = set()

    # مسیر ۱: نثر «ردیف N[، M، ...]»
    for m in _ROW_LIST_RE.finditer(ascii_text):
        rows |= _expand_number_tokens(m.group(1))

    # مسیر ۲: عدد خام داخل ستون «تصمیم‌ها (ردیف)» یک جدول — ستون سوم از ۷ (0-indexed=2)
    for line in ascii_text.splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 3:
            continue
        col = cells[2]
        if col in ("—", "-", "") or "تصمیم" in col:
            continue
        if re.fullmatch(r"[0-9،,\s\-–−وارجاعبند\.decision_logردیفی۰-۹A-Za-z()]*", col) and re.search(r"[0-9]", col):
            rows |= _expand_number_tokens(col)

    return rows


def find_referenced_fact_ids(text: str) -> set[str]:
    return set(re.findall(r"\bF(\d{2})\b", text))


def find_finding_refs_in_decision_log(text: str) -> set[str]:
    ascii_text = to_ascii_digits(text)
    return set(re.findall(r"یافته‌?\s*ی?\s*(\d+)(?:-(?:الف|ب))?", ascii_text))


def find_finding_ids_in_register(text: str) -> set[str]:
    ids = set()
    for m in re.finditer(r"^\|\s*(\d+(?:-(?:الف|ب))?)\s*\|", text, flags=re.MULTILINE):
        ids.add(m.group(1))
    return ids


def main() -> int:
    source_map_text = SOURCE_MAP.read_text() if SOURCE_MAP.exists() else ""
    findings_text = FINDINGS_REGISTER.read_text() if FINDINGS_REGISTER.exists() else ""
    decision_log_text = DECISION_LOG.read_text() if DECISION_LOG.exists() else ""
    facts_text = FACTS_REGISTER.read_text() if FACTS_REGISTER.exists() else ""

    combined_map_and_findings = source_map_text + "\n" + findings_text

    # --- بررسی ۱: هر گزارش reports/*.md ارجاع شده (نام کامل یا ریشه‌ی بدون پسوند) ---
    all_reports = find_report_files()
    unreferenced_reports = []
    for p in all_reports:
        rel = p.relative_to(ROOT_DIR).as_posix()
        name = p.name
        stem = p.stem
        if rel in source_map_text or name in source_map_text or stem in source_map_text:
            continue
        unreferenced_reports.append(rel)

    # --- بررسی ۲: هر ردیف decision_log ارجاع شده ---
    all_rows = find_decision_log_rows(decision_log_text)
    referenced_rows = find_referenced_decision_rows(combined_map_and_findings)
    unreferenced_rows = sorted(all_rows - referenced_rows)

    # --- بررسی ۳: هر حقیقت F## ارجاع شده ---
    all_facts = find_fact_ids(facts_text)
    referenced_facts = find_referenced_fact_ids(source_map_text)
    unreferenced_facts = sorted(all_facts - referenced_facts)

    # --- بررسی ۴: هر «یافته‌ی N» در decision_log شناسه‌ی متناظر در findings_register دارد ---
    finding_refs = find_finding_refs_in_decision_log(decision_log_text)
    finding_ids = find_finding_ids_in_register(findings_text)
    finding_ids_bare = {fid.split("-")[0] for fid in finding_ids}
    missing_findings = sorted(finding_refs - finding_ids_bare, key=int)

    n_gaps = (len(unreferenced_reports) + len(unreferenced_rows)
             + len(unreferenced_facts) + len(missing_findings))

    lines = [
        "# گزارش پوشش نقشه‌ی منبع",
        "",
        "> تولید خودکار با `python -m src.report_coverage`.",
        "",
        f"## خلاصه: {n_gaps} شکاف",
        "",
        f"- گزارش‌های بی‌ارجاع: {len(unreferenced_reports)} از {len(all_reports)}",
        f"- ردیف‌های decision_log بی‌ارجاع: {len(unreferenced_rows)} از {len(all_rows)}",
        f"- حقایق F## بی‌ارجاع: {len(unreferenced_facts)} از {len(all_facts)}",
        f"- یافته‌های ارجاع‌شده در decision_log بی‌شناسه در findings_register: {len(missing_findings)}",
        "",
    ]
    if unreferenced_reports:
        lines += ["## گزارش‌های بی‌ارجاع", ""] + [f"- `{r}`" for r in unreferenced_reports] + [""]
    if unreferenced_rows:
        lines += ["## ردیف‌های decision_log بی‌ارجاع", ""] + [f"- ردیف {r}" for r in unreferenced_rows] + [""]
    if unreferenced_facts:
        lines += ["## حقایق F## بی‌ارجاع", ""] + [f"- F{r}" for r in unreferenced_facts] + [""]
    if missing_findings:
        lines += ["## یافته‌های بی‌شناسه", ""] + [f"- یافته‌ی {r} (در decision_log ارجاع شده، در findings_register نیست)" for r in missing_findings] + [""]
    if n_gaps == 0:
        lines += ["✅ همه‌ی بررسی‌ها پاس شدند — هیچ شکافی پیدا نشد."]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    return 1 if n_gaps > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
