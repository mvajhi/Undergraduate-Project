"""ساخت داده‌ی داشبورد دانش پروژه.

این اسکریپت مخزن را می‌خواند و همه‌ی دانش پروژه را در یک فایل
`dashboard/data/dashboard-data.js` تجمیع می‌کند. داشبورد هیچ منبع دیگری ندارد —
هر عددی که نشان می‌دهد از همین‌جا می‌آید و مسیر فایل شاهدش هم همراهش ذخیره می‌شود.

اجرا:  python dashboard/build.py
"""

from __future__ import annotations

import csv
import io
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "doc"
REPORTS = ROOT / "reports"
OUT = Path(__file__).resolve().parent / "data" / "dashboard-data.js"


# ---------------------------------------------------------------- ابزار عمومی

def read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


FA_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٪", "0123456789%")


def to_en_digits(s: str) -> str:
    return s.translate(FA_DIGITS)


# اسناد مخزن ریاضی درون‌خطی لاتک دارند. داشبورد لاتک رندر نمی‌کند، پس نمادهای
# پرتکرار به یونیکد تبدیل می‌شوند و بقیه‌ی `$...$` فقط از دلار خالی می‌شود.
TEX_SYMBOLS = {
    r"\tau": "τ", r"\rho": "ρ", r"\alpha": "α", r"\beta": "β", r"\gamma": "γ",
    r"\delta": "δ", r"\Delta": "Δ", r"\epsilon": "ε", r"\eta": "η", r"\theta": "θ",
    r"\lambda": "λ", r"\mu": "μ", r"\sigma": "σ", r"\Sigma": "Σ", r"\phi": "φ",
    r"\chi": "χ", r"\pi": "π", r"\hat": "", r"\bar": "", r"\text": "",
    r"\times": "×", r"\cdot": "·", r"\le": "≤", r"\ge": "≥", r"\neq": "≠",
    r"\approx": "≈", r"\sim": "~", r"\in": "∈", r"\pm": "±", r"\to": "→",
    r"\ldots": "…", r"\dots": "…", r"\%": "٪", r"\,": " ", r"\;": " ", r"\ ": " ",
}
SUP = {"*": "*", "2": "²", "3": "³", "+": "⁺", "-": "⁻"}
SUB = {"o": "₀", "u": "ᵤ", "0": "₀", "1": "₁", "2": "₂", "i": "ᵢ", "t": "ₜ"}


def delatex(s: str) -> str:
    def one(m):
        body = m.group(1)
        for k, v in TEX_SYMBOLS.items():
            body = body.replace(k, v)
        body = re.sub(r"\^\{?([*23+-])\}?", lambda x: SUP.get(x.group(1), "^" + x.group(1)), body)
        body = re.sub(r"_\{?([a-z0-9])\}?", lambda x: SUB.get(x.group(1), "_" + x.group(1)), body)
        return body.replace("{", "").replace("}", "").strip()

    return re.sub(r"\$([^$]*)\$", one, s)


def strip_md(s: str) -> str:
    """حذف نشانه‌های markdown و لاتک از یک سلول جدول، برای نمایش و جست‌وجو."""
    s = re.sub(r"`([^`]*)`", r"\1", s)
    s = re.sub(r"\*\*([^*]*)\*\*", r"\1", s)
    s = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\1", s)
    s = delatex(s)
    return s.strip()


def split_row(line: str) -> list[str]:
    """یک ردیف جدول markdown را به سلول‌ها بشکن (با احترام به `|` داخل کد و لاتک)."""
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    cells, buf, in_code, in_math = [], [], False, False
    i = 0
    while i < len(line):
        ch = line[i]
        if ch == "`":
            in_code = not in_code
        elif ch == "$":
            in_math = not in_math
        elif ch == "\\" and i + 1 < len(line):
            buf.append(ch)
            i += 1
            buf.append(line[i])
            i += 1
            continue
        if ch == "|" and not in_code and not in_math:
            cells.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
        i += 1
    cells.append("".join(buf).strip())
    return cells


SEP_RE = re.compile(r"^\s*\|?[\s:|-]+\|[\s:|-]*$")


def iter_tables(text: str):
    """هر جدول markdown را با آخرین عنوان بالای سرش برگردان."""
    section = ""
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("#"):
            section = strip_md(line.lstrip("#").strip())
            i += 1
            continue
        if line.lstrip().startswith("|") and i + 1 < len(lines) and SEP_RE.match(lines[i + 1]):
            headers = [strip_md(c) for c in split_row(line)]
            rows, j = [], i + 2
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                cells = split_row(lines[j])
                if not SEP_RE.match(lines[j]):
                    rows.append(cells)
                j += 1
            yield section, headers, rows
            i = j
            continue
        i += 1


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        out = []
        for row in reader:
            clean = {}
            for k, v in row.items():
                key = (k or "name").strip() or "name"
                clean[key] = v
            out.append(clean)
        return out


def num(value, default=None):
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return default


# ------------------------------------------------------------ دفتر حقایق F##

FACT_RE = re.compile(r"^\*{0,2}`?(F\d+)`?\*{0,2}")


def build_facts() -> list[dict]:
    text = read(DOC / "data_facts_register.md")
    facts = []
    for section, headers, rows in iter_tables(text):
        for cells in rows:
            if not cells:
                continue
            m = FACT_RE.match(cells[0].strip())
            if not m:
                continue
            get = lambda i: strip_md(cells[i]) if i < len(cells) else ""
            facts.append(
                {
                    "id": m.group(1),
                    "section": section,
                    "fact": get(1),
                    "evidence": get(2),
                    "effect": get(3),
                    "impact": get(4),
                    "starred": "⭐" in cells[1] if len(cells) > 1 else False,
                    "source": "doc/data_facts_register.md",
                }
            )
    # چند حقیقت در منبع به‌جای ردیف جدول، به‌صورت بلوک نقل‌قول نوشته شده‌اند
    # (در حال حاضر فقط F54). بدون این پاس، آن‌ها از داشبورد غایب می‌مانند.
    seen = {f["id"] for f in facts}
    section = ""
    for line in text.splitlines():
        if line.startswith("#"):
            section = strip_md(line.lstrip("#").strip())
            continue
        m = re.match(r"^\s*>\s*\*\*(F\d+)\s*[—–-]\s*([^*]*)\*\*:?\s*(.*)$", line)
        if not m or m.group(1) in seen:
            continue
        seen.add(m.group(1))
        facts.append(
            {
                "id": m.group(1),
                "section": section,
                "fact": strip_md(m.group(2).strip().rstrip(":")),
                "evidence": strip_md(m.group(3)),
                "effect": "",
                "impact": "",
                "starred": False,
                "calloutOnly": True,
                "source": "doc/data_facts_register.md",
            }
        )

    facts.sort(key=lambda f: int(f["id"][1:]))
    return facts


# -------------------------------------------------------------- ثبت یافته‌ها

FINDING_RE = re.compile(r"^(\d+(?:-(?:الف|ب))?)$")


def build_findings() -> list[dict]:
    text = read(DOC / "findings_register.md")
    findings = []
    for section, headers, rows in iter_tables(text):
        if not headers or "یافته" not in " ".join(headers):
            continue
        for cells in rows:
            if not cells:
                continue
            fid = strip_md(cells[0])
            if not FINDING_RE.match(fid):
                continue
            get = lambda i: strip_md(cells[i]) if i < len(cells) else ""
            status = get(4)
            findings.append(
                {
                    "id": fid,
                    "sort": int(to_en_digits(fid).split("-")[0]),
                    "group": section,
                    "finding": get(1),
                    "phase": get(2),
                    "evidence": get(3),
                    "status": status,
                    "statusKind": classify_status(status),
                    "crossref": get(5),
                    "source": "doc/findings_register.md",
                }
            )
    findings.sort(key=lambda f: (f["sort"], f["id"]))
    return findings


def classify_status(status: str) -> str:
    s = status.replace("‌", "")
    if s.startswith("ردشده") or "ردشده" in s:
        return "rejected"
    if s.startswith("رفعشده") or "رفعشده" in s:
        return "resolved"
    if s.startswith("باز") or s.startswith("باز "):
        return "open"
    if "تأییدشده" in s or "تاییدشده" in s:
        return "confirmed"
    return "other"


# ------------------------------------------------------------ دفترچه تصمیمات

def build_decisions() -> list[dict]:
    text = read(DOC / "decision_log.md")
    decisions = []
    for section, headers, rows in iter_tables(text):
        if not headers or headers[0].strip() != "#":
            continue
        for cells in rows:
            if not cells or not re.fullmatch(r"\d+", strip_md(cells[0])):
                continue
            get = lambda i: strip_md(cells[i]) if i < len(cells) else ""
            decisions.append(
                {
                    "id": int(strip_md(cells[0])),
                    "title": get(1),
                    "date": get(2),
                    "options": get(3),
                    "reason": get(4),
                    "effect": get(5),
                    "source": "doc/decision_log.md",
                }
            )
    decisions.sort(key=lambda d: d["id"])

    # ⚠️ منبع یک برخورد شناسه‌ی واقعی دارد: شماره‌ی ۶۲ دو ردیف کاملاً متفاوت است.
    # شماره عوض نمی‌شود (ردیف‌های decision_log از جاهای دیگر ارجاع داده شده‌اند)؛
    # طبق همان قاعده‌ای که findings_register برای ۲۸/۲۹ به کار برد، پسوند الف/ب
    # می‌گیرند و برخورد به‌عنوان هشدار کیفیت داده به داشبورد گزارش می‌شود.
    counts = {}
    for d in decisions:
        counts[d["id"]] = counts.get(d["id"], 0) + 1
    suffixes = ["الف", "ب", "پ", "ت"]
    used = {}
    for d in decisions:
        d["label"] = str(d["id"])
        d["collision"] = counts[d["id"]] > 1
        if d["collision"]:
            k = used.get(d["id"], 0)
            used[d["id"]] = k + 1
            d["label"] = f"{d['id']}-{suffixes[k]}" if k < len(suffixes) else f"{d['id']}-{k}"

    # پیوست: سندهای تحلیل عمیق `doc/decisions/NN-*.md`
    deep = {}
    for path in sorted((DOC / "decisions").glob("*.md")):
        m = re.match(r"(\d+)-", path.name)
        if m:
            deep[int(m.group(1))] = rel(path)
    for d in decisions:
        if d["id"] in deep:
            d["deepDive"] = deep[d["id"]]
    return decisions


# ------------------------------------------------------------- رجیستری فیچر

def build_features() -> list[dict]:
    text = read(DOC / "feature_registry.md")
    features = []
    for section, headers, rows in iter_tables(text):
        if not headers or headers[0].strip() != "فیچر":
            continue
        for cells in rows:
            name = strip_md(cells[0]) if cells else ""
            if not name or name == "فیچر":
                continue
            get = lambda i: strip_md(cells[i]) if i < len(cells) else ""
            features.append(
                {
                    "name": name,
                    "family": get(1),
                    "clause": get(2),
                    "timing": get(3),
                    "evidence": get(4),
                    "coverage": get(5),
                    "corr": get(6),
                    "sets": get(7),
                    "starred": "⭐" in cells[1] if len(cells) > 1 else False,
                    "risky": "⚠️" in " ".join(cells),
                    "source": "doc/feature_registry.md",
                }
            )
    return features


# ------------------------------------------------------- پیشرفت فازها (TODO)

PHASE_TITLES = {
    "00": "زیرساخت و حاکمیت پروژه",
    "01": "تثبیت و تکمیل تعریف مسئله",
    "02": "جمع‌آوری داده‌های بیرونی و مکمل",
    "03": "پاکسازی و یکپارچه‌سازی",
    "04": "کاوش اکتشافی داده",
    "05": "مهندسی ویژگی",
    "06": "طراحی پروتکل اعتبارسنجی",
    "07": "مدل‌سازی و تنظیم",
    "08": "ارزیابی و تحلیل خطا",
    "09": "تفسیرپذیری",
    "10": "لایه‌ی تصمیم و تحلیل کسب‌وکاری",
    "11": "مستندسازی و ارائه",
}

CHECK_RE = re.compile(r"^(\s*)[-*]\s*\[([ xX])\]\s*(.+)$")


def build_phases() -> list[dict]:
    phases = []
    for path in sorted((DOC / "progress").glob("*.md")):
        code = path.name[:2]
        text = read(path)
        items, group = [], ""
        for line in text.splitlines():
            if line.startswith("#"):
                group = strip_md(line.lstrip("#").strip())
                continue
            m = CHECK_RE.match(line)
            if not m:
                continue
            items.append(
                {
                    "done": m.group(2).lower() == "x",
                    "text": strip_md(m.group(3)),
                    "group": group,
                }
            )
        done = sum(1 for i in items if i["done"])
        phases.append(
            {
                "code": code,
                "title": PHASE_TITLES.get(code, path.stem),
                "file": rel(path),
                "total": len(items),
                "done": done,
                "pct": round(100 * done / len(items)) if items else 0,
                "items": items,
            }
        )
    return phases


# ----------------------------------------------------------- مقایسه‌ی مدل‌ها

def build_models() -> dict:
    rows = read_csv(REPORTS / "phase7" / "model_comparison.csv")
    models = []
    for r in rows:
        models.append(
            {
                "model": r.get("model_id", ""),
                "family": r.get("family", ""),
                "level": r.get("level", ""),
                "target": r.get("target", ""),
                "pinball": num(r.get("pinball_rate")),
                "pinballPortions": num(r.get("pinball_portions")),
                "shortage": num(r.get("shortage_rate")),
                "wasteReduction": num(r.get("waste_reduction_pct")),
                "coverage": num(r.get("coverage")),
                "coverageGap": num(r.get("coverage_gap")),
                "rmse": num(r.get("RMSE_rho")),
                "mae": num(r.get("MAE_rho")),
                "r2": num(r.get("R2_rho")),
                "delta": num(r.get("delta_vs_B3")),
                "ciLo": num(r.get("delta_vs_B3_ci_lo")),
                "ciHi": num(r.get("delta_vs_B3_ci_hi")),
                "beatsB3": str(r.get("beats_B3_significant", "")).strip().lower() == "true",
                "fitSeconds": num(r.get("fit_seconds")),
                "nTrials": num(r.get("n_trials")),
                "compute": r.get("compute", ""),
                "metricBasis": r.get("metric_basis", ""),
                "sigMethod": r.get("significance_method", ""),
                "dmP": num(r.get("dm_p_value")),
            }
        )
    models.sort(key=lambda m: (m["pinball"] is None, m["pinball"]))

    cards = []
    for path in sorted((REPORTS / "models").glob("*.md")):
        if path.stem.startswith("_"):
            continue
        first = next((l for l in read(path).splitlines() if l.startswith("#")), "")
        cards.append(
            {
                "name": path.stem,
                "title": strip_md(first.lstrip("#").strip()),
                "file": rel(path),
            }
        )
    return {"comparison": models, "cards": cards}


# ------------------------------------------------------- فاز ۸ — ارزیابی نهایی

def build_evaluation() -> dict:
    holdout = []
    for r in read_csv(REPORTS / "phase8" / "8.1_holdout_metrics.csv"):
        holdout.append(
            {
                "model": r.get("name", ""),
                "pinball": num(r.get("pinball")),
                "pinballPortions": num(r.get("pinball_portions")),
                "mae": num(r.get("MAE_rho")),
                "rmse": num(r.get("RMSE_rho")),
                "r2": num(r.get("R2_rho")),
                "coverage": num(r.get("coverage")),
                "coverageGap": num(r.get("coverage_gap")),
                "shortage": num(r.get("shortage_rate")),
                "wasteReduction": num(r.get("waste_reduction_pct")),
                "shortagePortions": num(r.get("shortage_portions")),
                "surplusPortions": num(r.get("surplus_portions")),
                "fitSeconds": num(r.get("fit_seconds")),
            }
        )
    holdout.sort(key=lambda m: (m["pinball"] is None, m["pinball"]))

    significance = []
    for r in read_csv(REPORTS / "phase8" / "8.1_holdout_significance.csv"):
        significance.append({k: (num(v) if num(v) is not None else v) for k, v in r.items()})

    breakdown = {}
    labels = {
        "dow_name": "روز هفته",
        "Meal": "وعده",
        "FoodType": "نوع غذا",
        "res_quartile": "چارک حجم رزرو",
        "RestaurantName": "سلف",
        "ym": "ماه",
        "day_type": "نوع روز",
    }
    for path in sorted((REPORTS / "phase8").glob("8.3_by_*.csv")):
        rows = read_csv(path)
        if not rows:
            continue
        key = list(rows[0].keys())[0]
        breakdown[key] = {
            "label": labels.get(key, key),
            "file": rel(path),
            "rows": [
                {
                    "name": r.get(key, ""),
                    "n": num(r.get("n")),
                    "pinball": num(r.get("pinball")),
                    "shortage": num(r.get("shortage_rate")),
                    "coverage": num(r.get("coverage")),
                    "coverageGap": num(r.get("coverage_gap")),
                }
                for r in rows
            ],
        }

    reliability = [
        {"tau": num(r.get("tau")), "coverage": num(r.get("coverage")), "gap": num(r.get("gap"))}
        for r in read_csv(REPORTS / "phase8" / "8.8_reliability_table.csv")
    ]
    learning = [
        {k: num(v) for k, v in r.items()}
        for r in read_csv(REPORTS / "phase8" / "8.9_learning_curve.csv")
    ]
    seeds = [
        {k: (num(v) if num(v) is not None else v) for k, v in r.items()}
        for r in read_csv(REPORTS / "phase8" / "8.6_seed_sensitivity.csv")
    ]
    return {
        "holdout": holdout,
        "significance": significance,
        "breakdown": breakdown,
        "reliability": reliability,
        "learning": learning,
        "seeds": seeds,
    }


# -------------------------------------------------------- فاز ۹ — تفسیرپذیری

def build_interpretability() -> dict:
    importance = []
    for r in read_csv(REPORTS / "phase9" / "9.1_feature_importance.csv"):
        importance.append(
            {
                "feature": r.get("name", ""),
                "gain": num(r.get("gain")),
                "split": num(r.get("split")),
                "gainRank": num(r.get("gain_rank")),
                "permMean": num(r.get("perm_importance_mean")),
                "permRank": num(r.get("perm_rank")),
            }
        )
    shap = [
        {"feature": r.get("name", ""), "shap": num(r.get("mean_abs_shap"))}
        for r in read_csv(REPORTS / "phase9" / "9.2_shap_mean_abs.csv")
    ]
    shapL5 = [
        {"feature": r.get("name", ""), "shap": num(r.get("mean_abs_shap"))}
        for r in read_csv(REPORTS / "phase9" / "9.6_l5_shap_mean_abs.csv")
    ]
    groups = [
        {
            "group": r.get("group", ""),
            "nFeatures": num(r.get("n_features")),
            "sumShap": num(r.get("sum_mean_abs_shap")),
            "share": num(r.get("share_pct")),
            "topFeature": r.get("top_feature", ""),
        }
        for r in read_csv(REPORTS / "phase9" / "9.6_group_importance.csv")
    ]
    return {"importance": importance, "shap": shap, "shapL5": shapL5, "groups": groups}


# --------------------------------------------------- فاز ۱۰ — لایه‌ی تصمیم

def build_decision_layer() -> dict:
    curve = [
        {
            "tau": num(r.get("tau")),
            "shortage": num(r.get("shortage_rate")),
            "wasteReduction": num(r.get("waste_reduction_pct")),
            "portionsSaved": num(r.get("portions_saved")),
        }
        for r in read_csv(REPORTS / "phase10" / "10.7_tau_curve.csv")
    ]
    unmet = [
        {k: num(v) for k, v in r.items()}
        for r in read_csv(REPORTS / "phase10" / "10.7_tau_curve_unmet.csv")
    ]
    net = []
    for r in read_csv(REPORTS / "phase10" / "10.2_net_value_by_tau.csv"):
        net.append(
            {
                "model": r.get("model", ""),
                "tau": num(r.get("tau")),
                "cuOverCo": num(r.get("Cu_over_Co")),
                "wasteSaved": num(r.get("waste_saved_toman")),
                "shortageCost": num(r.get("shortage_cost_toman")),
                "netValue": num(r.get("net_value_toman")),
                "shortagePortions": num(r.get("shortage_portions")),
                "portionsSaved": num(r.get("portions_saved_vs_B0")),
            }
        )
    scenarios = []
    for r in read_csv(REPORTS / "phase10" / "10.3_policy_scenarios.csv"):
        scenarios.append(
            {
                "scenario": r.get("scenario", ""),
                "tau": num(r.get("tau")),
                "shortage": num(r.get("shortage_rate")),
                "shortagePerDay": num(r.get("shortage_portions_per_day")),
                "shortagePerMeal": num(r.get("shortage_portions_per_meal")),
                "medianPerMeal": num(r.get("median_shortage_portions_per_meal")),
                "worstMeal": num(r.get("worst_meal_shortage_portions")),
                "meanDepth": num(r.get("mean_shortage_depth")),
                "unmetRate": num(r.get("unmet_demand_rate")),
                "fillRate": num(r.get("fill_rate")),
                "annualPortions": num(r.get("annual_portions_saved")),
                "annualPortionsLo": num(r.get("annual_portions_saved_lo")),
                "annualPortionsHi": num(r.get("annual_portions_saved_hi")),
                "annualToman": num(r.get("annual_toman")),
                "annualTomanLo": num(r.get("annual_toman_lo")),
                "annualTomanHi": num(r.get("annual_toman_hi")),
            }
        )
    annual = [
        {k: (num(v) if num(v) is not None else v) for k, v in r.items()}
        for r in read_csv(REPORTS / "phase10" / "10.4_annual_estimate.csv")
    ]
    tauStar = [
        {k: (num(v) if num(v) is not None else v) for k, v in r.items()}
        for r in read_csv(REPORTS / "phase10" / "10.2_tau_star_table.csv")
    ]
    return {
        "tauCurve": curve,
        "tauUnmet": unmet,
        "netValue": net,
        "scenarios": scenarios,
        "annual": annual,
        "tauStar": tauStar,
    }


# ------------------------------------------------------------- گالری شکل‌ها

FIG_GROUPS = {
    "": "شکل‌های گزارش و EDA (فاز ۴)",
    "phase8": "فاز ۸ — ارزیابی و تحلیل خطا",
    "phase9": "فاز ۹ — تفسیرپذیری",
    "phase10": "فاز ۱۰ — لایه‌ی تصمیم",
    "round4": "EDA دور ۴ — اختلال سرویس",
    "round5": "EDA دور ۵ — آناتومی هدررفت",
    "slides": "شکل‌های ارائه",
}


def build_figures() -> list[dict]:
    base = REPORTS / "figures"
    figures = []
    if not base.exists():
        return figures
    for path in sorted(base.rglob("*.png")):
        group = str(path.parent.relative_to(base))
        group = "" if group == "." else group
        name = path.stem
        figures.append(
            {
                "name": name,
                "path": "figures/" + str(path.relative_to(base)),
                "repoPath": rel(path),
                "group": FIG_GROUPS.get(group, group or "سایر"),
                "persian": name.endswith("_fa"),
                "sizeKB": round(path.stat().st_size / 1024),
            }
        )
    return figures


# ------------------------------------------------------------- نقشه‌ی اسناد

DOC_NOTES = {
    "doc/project-definition.md": "تعریف رسمی مسئله، قاعده‌ی برش اطلاعاتی، متغیر هدف",
    "doc/WBS-food-demand-forecasting.md": "نقشه‌ی کامل فازها و بسته‌های کاری",
    "doc/WBS-phase7-modeling.md": "پروتکل تفصیلی فاز ۷ (مدل‌سازی)",
    "doc/phase7-execution-standard.md": "رویه‌ی الزام‌آور اجرای فاز ۷",
    "doc/decision_log.md": "هر تصمیم روش‌شناختی پروژه (منبع فصل روش‌شناسی)",
    "doc/data_facts_register.md": "دفتر حقایق EDA — خروجی فاز ۴ و دروازه‌ی M2",
    "doc/findings_register.md": "فضای شناسه‌ی یافته‌های فازهای ۰، ۷، ۸، ۹، ۱۰",
    "doc/data_manifest.md": "شناسنامه‌ی داده، هش snapshot، قفل foldها",
    "doc/data_dictionary.md": "دیکشنری داده‌ی تجمیعی",
    "doc/data_dictionary_individual.md": "دیکشنری داده‌ی سطح فردی (L5)",
    "doc/feature_registry.md": "رجیستری خودکار فیچرهای فاز ۵",
    "doc/leakage_audit.md": "ممیزی نشت اطلاعاتی هر فیچر",
    "doc/report-source-map.md": "نقشه‌ی ادعا ← شاهد، بر اساس ۱۱ فصل گزارش",
    "doc/report-outline.md": "ساختار فصل‌های گزارش نهایی",
    "doc/model-evaluation-metrics.md": "مرجع معیارهای ارزیابی",
    "doc/Literature_Review.md": "مرور ادبیات",
    "doc/gpu-run-playbook.md": "رویه‌ی اجرای کولب/کگل",
    "doc/site.md": "طرح سایت تعاملی روز پروژه",
    "doc/poster-a1-plan.md": "طرح پوستر A1",
    "doc/peresentation.md": "طرح ارائه",
    "doc/data_requests.md": "اقلام داده‌ی درخواستی از ذی‌نفع",
}


def build_docs() -> list[dict]:
    docs = []
    for path in sorted(DOC.rglob("*.md")):
        text = read(path)
        first = next((l for l in text.splitlines() if l.startswith("#")), "")
        docs.append(
            {
                "path": rel(path),
                "title": strip_md(first.lstrip("#").strip()) or path.stem,
                "note": DOC_NOTES.get(rel(path), ""),
                "sizeKB": round(path.stat().st_size / 1024),
                "lines": text.count("\n") + 1,
                "group": str(path.parent.relative_to(DOC)) or ".",
            }
        )
    for path in sorted(REPORTS.rglob("*.md")):
        text = read(path)
        first = next((l for l in text.splitlines() if l.startswith("#")), "")
        docs.append(
            {
                "path": rel(path),
                "title": strip_md(first.lstrip("#").strip()) or path.stem,
                "note": "",
                "sizeKB": round(path.stat().st_size / 1024),
                "lines": text.count("\n") + 1,
                "group": "reports/" + (str(path.parent.relative_to(REPORTS)) if path.parent != REPORTS else "."),
            }
        )
    return docs


# --------------------------------------------------------------- ثابت‌ها/متا

LEVELS = [
    {"id": "L1", "unit": "(d,m,r,f) — روز × وعده × سلف × نوع غذا", "size": "۷٬۵۷۹", "note": "سطح خروجی عملیاتی و سطح قهرمان پروژه"},
    {"id": "L2", "unit": "(d,m,r) — روز × وعده × سلف", "size": "~۴٬۱۰۰", "note": "تجمیع روی نوع غذا"},
    {"id": "L3", "unit": "(d,m) — عامل مشترک روز", "size": "۲۵۶", "note": "سطح سری‌زمانی خانواده‌ی ۳"},
    {"id": "L4", "unit": "سری (m,r)", "size": "۴۱ سری · میانه ۱۹۷ نقطه", "note": "سطح DFM خانواده‌ی ۴"},
    {"id": "L5", "unit": "(p,d,m,r,f) — رزرو فردی", "size": "۲٬۰۴۹٬۳۲۲ رزرو · ۲۶٬۷۶۸ فرد", "note": "سطح رفتار فردی (مدل B سابق)"},
]

CUTOFF = [
    {"target": "ناهار روز d", "cut": "۱۵:۰۰ روز d−۱", "last": "ناهار روز d−۱", "forbidden": "شام روز d−۱"},
    {"target": "شام روز d", "cut": "۲۳:۰۰ روز d−۱", "last": "شام روز d−۱", "forbidden": "—"},
]

MILESTONES = [
    {"id": "M1", "title": "پایان فاز ۳ — داده قفل‌شده", "phase": "03"},
    {"id": "M2", "title": "پایان فاز ۴ — دفتر حقایق", "phase": "04"},
    {"id": "M3", "title": "پایان فاز ۶ — خط پایه + پروتکل", "phase": "06"},
    {"id": "M4", "title": "پایان فاز ۷ — قهرمان مدل‌سازی", "phase": "07"},
    {"id": "M5", "title": "گزارش نهایی", "phase": "11"},
]

CHAPTERS = [
    "مقدمه و بیان مسئله", "مرور ادبیات", "داده و روش جمع‌آوری", "EDA و یافته‌ها",
    "مهندسی ویژگی", "روش‌شناسی و اعتبارسنجی", "نتایج", "تفسیر (فاز ۹)",
    "تحلیل کسب‌وکاری (فاز ۱۰)", "محدودیت‌ها", "کار آینده",
]


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
    except Exception:
        return "—"


def main() -> None:
    facts = build_facts()
    findings = build_findings()
    decisions = build_decisions()
    features = build_features()
    phases = build_phases()
    models = build_models()
    evaluation = build_evaluation()
    interp = build_interpretability()
    decision_layer = build_decision_layer()
    figures = build_figures()
    docs = build_docs()

    collisions = sorted({d["id"] for d in decisions if d.get("collision")})
    missing_facts = sorted(
        set(range(1, max(int(f["id"][1:]) for f in facts) + 1))
        - {int(f["id"][1:]) for f in facts}
    ) if facts else []

    champion = models["comparison"][0] if models["comparison"] else {}
    aggressive = next((s for s in decision_layer["scenarios"] if s["tau"] == 0.2), {})

    data = {
        "meta": {
            "builtAt": datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M"),
            "gitCommit": git_commit(),
            "title": "داشبورد دانش پروژه",
            "subtitle": "پیش‌بینی نرخ عدم‌دریافت غذا در سلف‌های دانشگاه تهران",
            "counts": {
                "facts": len(facts),
                "findings": len(findings),
                "decisions": len(decisions),
                "features": len(features),
                "models": len(models["comparison"]),
                "modelCards": len(models["cards"]),
                "figures": len(figures),
                "docs": len(docs),
                "phases": len(phases),
                "phaseItems": sum(p["total"] for p in phases),
                "phaseItemsDone": sum(p["done"] for p in phases),
            },
            "dataQuality": {
                "decisionIdCollisions": collisions,
                "missingFactIds": ["F%02d" % n for n in missing_facts],
            },
        },
        "project": {
            "levels": LEVELS,
            "cutoff": CUTOFF,
            "milestones": MILESTONES,
            "chapters": CHAPTERS,
            "champion": champion,
            "aggressive": aggressive,
        },
        "facts": facts,
        "findings": findings,
        "decisions": decisions,
        "features": features,
        "phases": phases,
        "models": models,
        "evaluation": evaluation,
        "interpretability": interp,
        "decisionLayer": decision_layer,
        "figures": figures,
        "docs": docs,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    OUT.write_text(
        "// تولید خودکار با `python dashboard/build.py` — دستی ویرایش نشود.\n"
        "window.__DASH__ = " + payload + ";\n",
        encoding="utf-8",
    )
    size = OUT.stat().st_size / 1024
    print(f"نوشته شد: {rel(OUT)}  ({size:.0f} کیلوبایت)")
    for key, value in data["meta"]["counts"].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
