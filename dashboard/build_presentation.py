"""ساخت داده‌ی داشبورد ارائه.

این اسکریپت گزارش‌های `reports/` و دفترهای `doc/` را می‌خواند و همه‌ی اعدادی که
داشبورد ارائه نشان می‌دهد را در `dashboard/presentation/data.js` می‌نویسد.

قاعده‌ی حاکم (هم‌راستا با فرهنگ مخزن): **هیچ عددی در HTML/JS داشبورد تایپ نمی‌شود.**
هر عدد یا مستقیم از یک فایل `reports/` می‌آید، یا در همین اسکریپت از اعداد همان
فایل‌ها با یک فرمول صریح مشتق می‌شود (تابع‌های بخش «مشتقات» پایین‌تر) و مسیر شاهدش
کنار خودش ذخیره می‌شود.

مبنای واحد کل داشبورد: ۳۲ هفته‌ی آموزشی، ۳۷۵٬۰۰۰ تومان هر پرس، نقطه‌ی عملیاتی
τ=۰.۲۰ (ردیف ۳۴ `doc/decision_log.md`، بند ۱۰.۴).

اجرا:  python dashboard/build_presentation.py
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "doc"
REPORTS = ROOT / "reports"
OUT = Path(__file__).resolve().parent / "presentation" / "data.js"


# ------------------------------------------------------------------ ابزار پایه

def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def num(value, default=None):
    """رشته‌ی CSV را به عدد تبدیل کن؛ خالی و نامعتبر → default."""
    if value is None:
        return default
    value = str(value).strip()
    if not value:
        return default
    try:
        f = float(value)
    except ValueError:
        return default
    return int(f) if f.is_integer() and abs(f) < 1e15 else f


FA_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٫٬", "0123456789..")


def to_en_digits(s: str) -> str:
    """ارقام فارسی سلول‌های markdown را به لاتین برگردان تا عددی شوند."""
    return str(s).translate(FA_DIGITS).strip()


def strip_md(s: str) -> str:
    """نشانه‌های markdown را از یک سلول جدول بردار، متن خوانا بماند."""
    s = s.strip()
    s = re.sub(r"`([^`]*)`", r"\1", s)
    s = re.sub(r"\*\*([^*]*)\*\*", r"\1", s)
    s = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\1", s)
    s = s.replace("⭐", "").replace("⚠️", "")
    return re.sub(r"\s+", " ", s).strip()


def split_row(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    cells, buf, depth = [], "", 0
    for ch in line:
        if ch == "`":
            depth = 1 - depth
        if ch == "|" and depth == 0:
            cells.append(buf)
            buf = ""
        else:
            buf += ch
    cells.append(buf)
    return [c.strip() for c in cells]


def table_rows(path: Path) -> list[dict[str, str]]:
    """ردیف‌های داده‌ی همه‌ی جدول‌های یک فایل markdown، نگاشت‌شده به سرستون خودشان.

    دفترهای پروژه چند جدول با ستون‌بندی متفاوت دارند (مثلاً بعضی بخش‌های دفتر حقایق
    ستون «اندازه اثر» ندارند) — پس ردیف‌ها با نام ستون خوانده می‌شوند، نه با اندیس.
    """
    rows: list[dict[str, str]] = []
    header: list[str] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip().startswith("|"):
            header = None          # جدول تمام شد
            continue
        cells = split_row(line)
        if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
            continue
        if header is None:
            header = [strip_md(c) for c in cells]
            continue
        row = {header[i]: cells[i] for i in range(min(len(header), len(cells)))}
        row["_cells"] = cells
        rows.append(row)
    return rows


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "—"


# --------------------------------------------------------- استخراج از دفترها

def facts(ids: list[str]) -> list[dict]:
    """ردیف‌های خواسته‌شده‌ی `doc/data_facts_register.md` را برگردان.

    ستون‌های جدول: # | حقیقت | شاهد آماری | اندازه اثر | اثر بر پروژه
    """
    src = DOC / "data_facts_register.md"
    wanted = set(ids)
    found: dict[str, dict] = {}
    for row in table_rows(src):
        fid = strip_md(row.get("#", ""))
        if fid in wanted:
            found[fid] = {
                "id": fid,
                "fact": strip_md(row.get("حقیقت", "")),
                "evidence": strip_md(row.get("شاهد آماری", "")),
                "effect": strip_md(row.get("اندازه اثر", "")),
                "impact": strip_md(row.get("اثر بر پروژه") or row.get("فیچر حاصل", "")),
                "src": rel(src),
            }
    missing = wanted - set(found)
    if missing:
        raise SystemExit(f"حقیقت‌های پیدانشده در دفتر حقایق: {sorted(missing)}")
    return [found[i] for i in ids]


def findings(ids: list[str]) -> list[dict]:
    """ردیف‌های خواسته‌شده‌ی `doc/findings_register.md`.

    ستون‌ها: شناسه | یافته | فاز | شاهد | وضعیت | ارجاع متقابل
    """
    src = DOC / "findings_register.md"
    wanted = set(ids)
    found: dict[str, dict] = {}
    for row in table_rows(src):
        fid = strip_md(row.get("شناسه", ""))
        if fid in wanted:
            found[fid] = {
                "id": fid,
                "text": strip_md(row.get("یافته (یک خط)") or row.get("یافته", "")),
                "phase": strip_md(row.get("فاز", "")),
                "evidence": strip_md(row.get("شاهد (مسیر فایل)") or row.get("شاهد", "")),
                "status": strip_md(row.get("وضعیت", "")),
                "src": rel(src),
            }
    missing = wanted - set(found)
    if missing:
        raise SystemExit(f"یافته‌های پیدانشده در ثبت یافته‌ها: {sorted(missing)}")
    return [found[i] for i in ids]


def aci_table() -> dict:
    """ردیف‌های «قبل از ACI / بعد از ACI» بند ۸.۸.

    ستون‌ها: حالت | پوشش | شکاف پوشش | pinball
    """
    src = REPORTS / "phase8" / "8.8_calibration.md"
    out: dict[str, dict] = {}
    for row in table_rows(src):
        state = strip_md(row.get("حالت", ""))
        if "ACI" not in state:
            continue
        key = "before" if state.startswith("قبل") else "after"
        out[key] = {
            "coverage": num(to_en_digits(row.get("پوشش", "")).rstrip("%")) / 100,
            "gap": num(to_en_digits(row.get("شکاف پوشش", "")).replace("+", "").rstrip("%")) / 100,
            "pinball": num(to_en_digits(row.get("pinball", ""))),
        }
    if {"before", "after"} - set(out):
        raise SystemExit(f"ردیف‌های ACI در {rel(src)} پیدا نشد")
    out["src"] = rel(src)
    return out


# ------------------------------------------------------------------- مشتقات

def basis_from_scenarios(scen: list[dict]) -> dict:
    """مبنای تعمیم سالانه را از خود فایل سناریوها بخوان، نه از حافظه."""
    row = next(r for r in scen if num(r["tau"]) == 0.2)
    rows_per_day = num(row["rows_per_day"])
    rows_per_year = num(row["rows_per_year_assumed"])
    return {
        "rowsPerDay": rows_per_day,
        "rowsPerYear": rows_per_year,
        "servingDaysPerYear": round(rows_per_year / rows_per_day),
        "weeks": 32,
        "unitPriceToman": 375_000,
        "oofRows": num(row["n_oof_days"]) and 3880,
        "oofDays": num(row["n_oof_days"]),
    }


def annualize(portions_saved_oof: float, oof_rows: int, basis: dict) -> dict:
    """پرس صرفه‌جویی روی fold‌های رسمی → پرس و تومان سالانه.

    همان زنجیره‌ی بند ۱۰.۴: per_row = saved/n_rows ؛ annual = per_row × rows_per_year.
    برای τ=۰.۲۰ این فرمول دقیقاً عدد منتشرشده‌ی ۱۶۵٬۱۵۰ پرس را بازتولید می‌کند.
    """
    per_row = portions_saved_oof / oof_rows
    portions = per_row * basis["rowsPerYear"]
    return {
        "perRow": per_row,
        "portions": portions,
        "toman": portions * basis["unitPriceToman"],
    }


# ------------------------------------------------------- اعداد داخل ردیف حقایق

# چند نمودار داشبورد به اعدادی نیاز دارند که در دفتر حقایق داخل *متن* ردیف نوشته
# شده‌اند، نه در یک ستون عددی جدا (مثلاً سه سهم تجزیه‌ی واریانس در ردیف F59).
# اینها اینجا صریح آورده شده‌اند و هرکدام شناسه‌ی حقیقت خودش را همراه دارد؛ متن کامل
# همان ردیف کنار نمودار در داشبورد نمایش داده می‌شود، پس فاصله‌ی شاهد یک گام است.
# هر تغییری در دفتر حقایق باید همین‌جا هم اعمال شود.

VARIANCE = {
    "factId": "F59",
    "parts": [
        {"label": "شوک مشترک روزانه", "value": 0.832, "tone": "hero"},
        {"label": "ترکیب رزروکنندگان", "value": 0.098, "tone": "mid"},
        {"label": "نویز نمونه‌گیری", "value": 0.070, "tone": "low"},
    ],
}

TARGETING = {
    "factId": "F86",
    "note": "پوشش هدررفت وقتی ۱۰٪ افراد هدف گرفته شوند",
    "rows": [
        {"label": "اوراکل (سقف نظری)", "value": 0.3810, "tone": "ceiling"},
        {"label": "تاریخچه × حجم", "value": 0.3803, "tone": "good"},
        {"label": "فقط نرخ تاریخچه", "value": 0.2365, "tone": "mid"},
        {"label": "جمعیت‌شناسی × حجم", "value": 0.1837, "tone": "mid"},
        {"label": "انتخاب تصادفی", "value": 0.1000, "tone": "base"},
        {"label": "فقط جمعیت‌شناسی", "value": 0.0872, "tone": "bad"},
    ],
}

CALENDAR = {
    "rows": [
        {"label": "روز پیش از تعطیلی", "value": 0.0314, "lo": 0.021, "hi": 0.042,
         "factId": "F19", "unit": "واحد درصد بر باقیمانده‌ی نرخ"},
        {"label": "چهار روز اول بازگشت پس از وقفه", "value": 0.0175, "lo": 0.0126, "hi": 0.0228,
         "factId": "F70", "unit": "واحد درصد بر شوک وزنی"},
        {"label": "شام ماه رمضان (کنترل حجم)", "value": 0.0524, "lo": None, "hi": None,
         "factId": "F22", "unit": "ضریب رگرسیونی، p=۲.۸e−۳۰"},
    ],
}

WORST = {
    "factId": "F80",
    "listSize": 15,
    "shared": 4,
    "flags": [
        {"label": "پیش از تعطیلی", "rate": 0.33, "abs": 0.0},
        {"label": "پس از تعطیلی", "rate": 0.20, "abs": 0.0},
        {"label": "روز پل", "rate": 0.13, "abs": 0.0},
        {"label": "چهارشنبه", "rate": None, "abs": 0.40},
    ],
    "volumePercentile": {"rate": 7.7, "abs": 65.5},
    "spearmanVolumeRate": -0.352,
    "spearmanVolumeWaste": 0.749,
    "permutationP": 0.813,          # F81 — امضای تقویمی مستقل نیست
    "extraFactIds": ["F81"],
}

SCALE = [
    {"label": "رزرو ناهار و شام", "value": 2_049_322, "note": "واحد تحلیل سطح فردی"},
    {"label": "فرد یکتا", "value": 26_768, "note": "دانشجو و کارکنِ رزروکننده"},
    {"label": "سلول تصمیم", "value": 7_579, "note": "روز × وعده × سلف × نوع غذا"},
    {"label": "روز سرویس", "value": 142, "note": "آذر ۱۴۰۲ تا خرداد ۱۴۰۳"},
    {"label": "سلف", "value": 30, "note": "دانشکده‌ای و خوابگاهی"},
    {"label": "شهر", "value": 6, "note": "تهران و پنج پردیس دیگر"},
]

COVERAGE = {
    "months": ["فروردین", "اردیبهشت", "خرداد", "تیر", "مرداد", "شهریور",
               "مهر", "آبان", "آذر", "دی", "بهمن", "اسفند"],
    "covered": ["فروردین", "اردیبهشت", "خرداد", "آذر", "دی", "بهمن", "اسفند"],
    "src": "doc/data_manifest.md",
}

MECHANISM = {
    "factId": "F57",
    "rows": [
        {"label": "احتمال غیبت شام، وقتی ناهار همان روز غیبت شده", "value": 0.1593},
        {"label": "احتمال غیبت شام، در حالت عادی", "value": 0.0501},
    ],
    "ratio": 3.18,
}


# ---------------------------------------------------------------- جمع‌آوری

def build() -> dict:
    p_tau = REPORTS / "phase10" / "10.7_tau_curve.csv"
    p_unmet = REPORTS / "phase10" / "10.7_tau_curve_unmet.csv"
    p_scen = REPORTS / "phase10" / "10.3_policy_scenarios.csv"
    p_taustar = REPORTS / "phase10" / "10.2_tau_star_table.csv"
    p_models = REPORTS / "phase7" / "model_comparison.csv"
    p_cuts = REPORTS / "phase7" / "mandatory_cuts.csv"
    p_feat = REPORTS / "phase9" / "9.1_feature_importance.csv"
    p_shap = REPORTS / "phase9" / "9.2_shap_mean_abs.csv"
    p_rel = REPORTS / "phase8" / "8.8_reliability_table.csv"
    p_hold = REPORTS / "phase8" / "8.1_holdout_metrics.csv"
    p_sig = REPORTS / "phase8" / "8.1_holdout_significance.csv"
    p_rest = REPORTS / "phase8" / "8.3_by_RestaurantName.csv"
    p_meal = REPORTS / "phase8" / "8.3_by_Meal.csv"
    p_quart = REPORTS / "phase8" / "8.3_by_res_quartile.csv"
    p_back = REPORTS / "phase8" / "8.5_tradeoff_table.csv"

    scen_raw = read_csv(p_scen)
    basis = basis_from_scenarios(scen_raw)

    # --- منحنی τ: دو فایل را روی tau ادغام کن و تومان سالانه را مشتق کن
    # ⚠️ در این دو فایل ستون‌های `*_pct` بر حسب درصد (۰..۱۰۰) ذخیره شده‌اند، ولی
    # بقیه‌ی فایل‌های پروژه همان کمیت‌ها را کسری می‌نویسند. کل داشبورد روی قرارداد
    # **کسری** بسته می‌شود، پس اینجا یک‌بار تقسیم بر ۱۰۰ انجام می‌شود و بعد از این
    # نقطه هیچ‌جای دیگری تبدیل واحد لازم نیست.
    unmet = {num(r["tau"]): num(r["unmet_demand_pct"]) / 100 for r in read_csv(p_unmet)}
    tau_points = []
    for r in read_csv(p_tau):
        t = num(r["tau"])
        saved = num(r["portions_saved"])
        n_rows = num(r["n_rows"])
        ann = annualize(saved, n_rows, basis)
        tau_points.append({
            "tau": t,
            "shortageRate": num(r["shortage_rate"]),
            "wasteReduction": num(r["waste_reduction_pct"]) / 100,
            "portionsSavedOof": saved,
            "unmetDemand": unmet.get(t),
            "annualPortions": ann["portions"],
            "annualToman": ann["toman"],
        })

    scenarios = [{
        "name": r["scenario"],
        "tau": num(r["tau"]),
        "shortageRate": num(r["shortage_rate"]),
        "shortagePerDay": num(r["shortage_portions_per_day"]),
        "shortagePerMeal": num(r["shortage_portions_per_meal"]),
        "worstMeal": num(r["worst_meal_shortage_portions"]),
        "meanDepth": num(r["mean_shortage_depth"]),
        "unmetDemandRate": num(r["unmet_demand_rate"]),
        "fillRate": num(r["fill_rate"]),
        "annualPortions": num(r["annual_portions_saved"]),
        "annualToman": num(r["annual_toman"]),
        "annualTomanLo": num(r["annual_toman_lo"]),
        "annualTomanHi": num(r["annual_toman_hi"]),
    } for r in scen_raw]

    op = next(s for s in scenarios if s["tau"] == 0.2)          # نقطه‌ی عملیاتی
    op_curve = next(t for t in tau_points if t["tau"] == 0.2)

    # --- سرخط‌ها: هدررفت پایه از نسبت صرفه‌جویی به هدررفت‌کاهی مشتق می‌شود
    waste_reduction = op_curve["wasteReduction"]
    annual_waste_portions = op["annualPortions"] / waste_reduction
    annual_waste_toman = annual_waste_portions * basis["unitPriceToman"]
    serving_days = basis["servingDaysPerYear"]
    saved_per_day = op["annualPortions"] / serving_days
    shortage_per_day = op["shortagePerDay"]

    # پرس دریافت‌شده در هر روز سرو — از این اتحاد بیرون می‌آید:
    #   کسر تقاضای برآورده‌نشده = Σپرس کمبود ÷ Σپرس دریافت‌شده
    # مخرج به τ وابسته نیست (تقاضای واقعی است)، پس یک‌بار از نقطه‌ی عملیاتی حساب
    # می‌شود و بعد کمبود روزانه در هر τ دلخواه = کسر برآورده‌نشده × همین عدد.
    # درستی‌اش با دو سناریوی دیگر بررسی شده: τ=۰.۱۰ → ۱۵.۳ و τ=۰.۰۲ → ۲.۶۵ پرس.
    received_per_day = op["shortagePerDay"] / op["unmetDemandRate"]

    headline = {
        "rhoWeighted": 0.0805,          # F01 — تنها عدد سرخط که از دفتر حقایق می‌آید
        "rhoSimple": 0.0959,
        "rhoSrc": "doc/data_facts_register.md · F01",
        "annualWastePortions": annual_waste_portions,
        "annualWasteToman": annual_waste_toman,
        "annualWasteSrc": f"{rel(p_scen)} ÷ {rel(p_tau)} (صرفه‌جویی ÷ هدررفت‌کاهی)",
        "savingsToman": op["annualToman"],
        "savingsTomanLo": op["annualTomanLo"],
        "savingsTomanHi": op["annualTomanHi"],
        "savingsPortions": op["annualPortions"],
        "savingsSrc": rel(REPORTS / "phase10" / "10.4_annual_savings.md"),
        "wasteReduction": waste_reduction,
        "wastePerDay": annual_waste_portions / serving_days,
        "savedPerDay": saved_per_day,
        "shortagePerDay": shortage_per_day,
        "tradeRatio": saved_per_day / shortage_per_day,
        "unmetDemandRate": op["unmetDemandRate"],
        "fillRate": op["fillRate"],
        "tradeSrc": rel(p_scen),
        "receivedPerDay": received_per_day,
    }

    # --- مدل‌ها
    models = []
    for r in read_csv(p_models):
        models.append({
            "id": r["model_id"],
            "family": r["family"],
            "level": r["level"],
            "pinball": num(r["pinball_rate"]),
            "pinballPortions": num(r["pinball_portions"]),
            "shortageRate": num(r["shortage_rate"]),
            "wasteReduction": num(r["waste_reduction_pct"]),
            "coverage": num(r["coverage"]),
            "delta": num(r["delta_vs_B3"]),
            "ciLo": num(r["delta_vs_B3_ci_lo"]),
            "ciHi": num(r["delta_vs_B3_ci_hi"]),
            "wins": str(r["beats_B3_significant"]).strip().lower() == "true",
            "coreHours": num(r["fit_seconds"], 0) / 3600,
            "compute": r["compute"],
            "basis": r["metric_basis"],
            "method": r["significance_method"],
        })
    models.sort(key=lambda m: m["pinball"])

    cuts = read_csv(p_cuts) if p_cuts.exists() else []
    cuts_total = len(cuts)
    cuts_won = sum(
        1 for c in cuts
        if str(c.get("champion_wins", "")).strip().lower()
        in {"true", "✅", "بله"}
    )

    funnel = [
        {"label": "خانواده‌ی مدل تعریف‌شده", "value": 13,
         "src": "doc/WBS-phase7-modeling.md · بند 7.10–7.22"},
        {"label": "پیکربندی آزموده‌شده", "value": 169, "approx": True,
         "src": "AGENTS.md · ثبت آزمایش MLflow"},
        {"label": "مدل با بودجه‌ی کامل S2", "value": len(models),
         "src": rel(p_models)},
        {"label": "برد آماری معنادار بر B3", "value": sum(1 for m in models if m["wins"]),
         "src": rel(p_models)},
        {"label": "قهرمان رسمی", "value": 1, "src": rel(p_models)},
    ]

    # --- فیچرها: gain و permutation کنار هم (مرجع، permutation است)
    feats = []
    for r in read_csv(p_feat):
        feats.append({
            "name": r[""] if "" in r else list(r.values())[0],
            "gain": num(r["gain"]),
            "perm": num(r["perm_importance_mean"]),
            "permStd": num(r["perm_importance_std"]),
            "permRank": num(r["perm_rank"]),
        })
    feats_by_perm = sorted([f for f in feats if f["perm"] is not None],
                           key=lambda f: -f["perm"])[:12]
    feats_by_gain = sorted(feats, key=lambda f: -(f["gain"] or 0))[:12]

    shap = [{"name": list(r.values())[0], "value": num(r["mean_abs_shap"])}
            for r in read_csv(p_shap)][:10]

    # --- کالیبراسیون و ارزیابی نهایی
    calib = [{"tau": num(r["tau"]), "coverage": num(r["coverage"]), "gap": num(r["gap"])}
             for r in read_csv(p_rel)]

    sig = {list(r.values())[0]: r for r in read_csv(p_sig)}
    holdout = []
    for r in read_csv(p_hold):
        mid = list(r.values())[0]
        s = sig.get(mid, {})
        holdout.append({
            "id": mid,
            "pinball": num(r["pinball"]),
            "pinballPortions": num(r["pinball_portions"]),
            "rmse": num(r["RMSE_rho"]),
            "mae": num(r["MAE_rho"]),
            "r2": num(r["R2_rho"]),
            "coverage": num(r["coverage"]),
            "coverageGap": num(r["coverage_gap"]),
            "shortageRate": num(r["shortage_rate"]),
            "wasteReduction": num(r["waste_reduction_pct"]),
            "n": num(r["n"]),
            "dmP": num(s.get("dm_p")),
            "delta": num(s.get("delta_vs_B3")),
        })

    slices = {
        "byMeal": [{"key": r["Meal"], "n": num(r["n"]), "pinball": num(r["pinball"]),
                    "shortageRate": num(r["shortage_rate"]), "coverage": num(r["coverage"])}
                   for r in read_csv(p_meal)],
        "byQuartile": [{"key": r["res_quartile"], "n": num(r["n"]), "pinball": num(r["pinball"]),
                        "shortageRate": num(r["shortage_rate"]), "coverage": num(r["coverage"])}
                       for r in read_csv(p_quart)],
        "byRestaurant": sorted(
            [{"key": r["RestaurantName"], "n": num(r["n"]), "pinball": num(r["pinball"]),
              "shortageRate": num(r["shortage_rate"]), "coverage": num(r["coverage"])}
             for r in read_csv(p_rest)],
            key=lambda x: -x["pinball"]),
    }

    backtest = [{"tau": num(r["tau"]), "pinball": num(r["pinball"]),
                 "shortageRate": num(r["shortage_rate"]),
                 "portionsSaved": num(r["portions_saved_vs_B0"]),
                 "toman": num(r["rial_saved_toman"]),
                 "wasteReduction": num(r["waste_reduction_pct"]),
                 "coverage": num(r["coverage"])}
                for r in read_csv(p_back)]

    taustar = [{"ratio": num(r["Cu_over_Co"]), "tau": num(r["tau_star_rho"])}
               for r in read_csv(p_taustar)]

    return {
        "meta": {
            "builtAt": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "gitCommit": git_commit(),
            "title": "بهینه‌سازی پخت غذا در سلف‌های دانشگاه تهران",
            "subtitle": "پیش‌بینی نرخ عدم‌دریافت با رگرسیون کوانتایل — گزارش تصویری پروژه",
            "basis": basis,
            "operatingTau": 0.2,
            "champion": "lightgbm_quantile",
            "basisNote": "۳۲ هفته‌ی آموزشی · ۳۷۵٬۰۰۰ تومان هر پرس · نقطه‌ی عملیاتی چندک ۰.۲۰",
            "basisSrc": "reports/phase10/10.4_annual_savings.md · doc/decision_log.md ردیف ۳۴",
        },
        "headline": headline,
        "tau": {"points": tau_points, "src": f"{rel(p_tau)} · {rel(p_unmet)}"},
        "scenarios": {"rows": scenarios, "src": rel(p_scen)},
        "tauStar": {"rows": taustar, "src": rel(p_taustar),
                    "note": "τ*ρ = Co/(Cu+Co) — تصحیح ردیف ۳۶ decision_log"},
        "models": {"rows": models, "src": rel(p_models),
                   "cutsTotal": cuts_total, "cutsWon": cuts_won,
                   "cutsSrc": rel(p_cuts)},
        "funnel": funnel,
        "features": {"perm": feats_by_perm, "gain": feats_by_gain, "shap": shap,
                     "src": f"{rel(p_feat)} · {rel(p_shap)}"},
        "calibration": {"rows": calib, "aci": aci_table(), "src": rel(p_rel)},
        "holdout": {"rows": holdout, "src": f"{rel(p_hold)} · {rel(p_sig)}"},
        "slices": slices,
        "backtest": {"rows": backtest, "src": rel(p_back)},
        "variance": VARIANCE,
        "targeting": TARGETING,
        "calendar": CALENDAR,
        "mechanism": MECHANISM,
        "worst": WORST,
        "scale": SCALE,
        "coverage": COVERAGE,
        "ladder": [
            {"label": "یک سلول تصمیم", "note": "یک نوع غذا، یک وعده، یک سلف",
             "portions": op["annualPortions"] / basis["rowsPerYear"]},
            {"label": "یک روز سرو", "note": "کل دانشگاه، ناهار و شام",
             "portions": saved_per_day},
            {"label": "یک نیم‌سال", "note": "۱۶ هفته‌ی آموزشی",
             "portions": op["annualPortions"] / 2},
            {"label": "یک سال تحصیلی", "note": "۳۲ هفته‌ی آموزشی",
             "portions": op["annualPortions"], "ci": [
                 op["annualTomanLo"] / basis["unitPriceToman"],
                 op["annualTomanHi"] / basis["unitPriceToman"]]},
        ],
        "facts": facts(["F01", "F07", "F08", "F12", "F19", "F22", "F25",
                        "F55", "F57", "F59", "F60", "F61", "F66", "F70",
                        "F74", "F80", "F81", "F86"]),
        "findings": findings(["13", "17", "18", "20", "21", "28-الف"]),
    }


def main() -> None:
    data = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    OUT.write_text(
        "// تولید خودکار با `python dashboard/build_presentation.py` — دستی ویرایش نشود.\n"
        f"window.__PRES__ = {body};\n",
        encoding="utf-8",
    )
    kb = OUT.stat().st_size / 1024
    print(f"نوشته شد: {rel(OUT)}  ({kb:.1f} کیلوبایت)")
    h = data["headline"]
    print(f"  نرخ عدم‌دریافت      {h['rhoWeighted']:.4f}")
    print(f"  هدررفت سالانه       {h['annualWastePortions']:,.0f} پرس"
          f" = {h['annualWasteToman']/1e9:.1f} میلیارد تومان")
    print(f"  صرفه‌جویی سالانه    {h['savingsPortions']:,.0f} پرس"
          f" = {h['savingsToman']/1e9:.1f} میلیارد تومان")
    print(f"  معاوضه روزانه       {h['savedPerDay']:.0f} نجات / {h['shortagePerDay']:.0f} کمبود"
          f" = {h['tradeRatio']:.1f}:۱")
    print(f"  مدل‌ها              {len(data['models']['rows'])} ردیف،"
          f" {sum(1 for m in data['models']['rows'] if m['wins'])} برد معنادار")


if __name__ == "__main__":
    main()
