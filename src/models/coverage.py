"""گزارش پوشش محورهای آزمایش — «کدام نقاط فضای بند 7.9.1 هنوز زده نشده‌اند؟»

## چرا لازم است

منشور انصاف (بند 7.2) می‌گوید هیچ مدلی نباید پیش از رسیدن به سقف خودش بازنده اعلام شود.
ولی «سقف خودش» فقط هایپرپارامتر نیست — سطح داده، دامنه‌ی مدل (سراسری/خوشه/سلف)، فیچرست،
معماری و وزن‌دهی هم هست. بدون این گزارش، یک اجرا که فقط «L1 سراسری با FS_day» را زده در
جدول نهایی چنان دیده می‌شود که انگار کل خانواده آزموده شده.

این ماژول مستقیماً از MLflow می‌خواند (تنها منبع حقیقتِ اجراها، بند 7.7) و
``reports/phase7/axis_coverage.md`` را می‌سازد: به‌ازای هر محور، کدام مقادیر زده شده و
کدام **نشده**. اجرا: ``python -m src.models.coverage``.
"""

import mlflow
import pandas as pd

from src.config import MLFLOW_TRACKING_URI, REPORTS_DIR
from src.models.axes import AXES, OPEN_AXES
from src.models.registry import FAMILIES
from src.models.tracking import DEFAULT_EXPERIMENT

OUTPUT_PATH = REPORTS_DIR / "phase7" / "axis_coverage.md"


def load_runs(experiment: str = DEFAULT_EXPERIMENT) -> pd.DataFrame:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    exp = mlflow.get_experiment_by_name(experiment)
    if exp is None:
        return pd.DataFrame()
    return mlflow.search_runs(experiment_ids=[exp.experiment_id])


def _values_seen(runs: pd.DataFrame, axis: str) -> set:
    col = f"params.{axis}"
    if runs.empty or col not in runs.columns:
        return set()
    return {v for v in runs[col].dropna().unique()}


def build_report(runs: pd.DataFrame) -> str:
    lines = [
        "# پوشش محورهای آزمایش — فاز ۷",
        "",
        "> بند 7.9.1 سند `doc/WBS-phase7-modeling.md`. تولید خودکار با `python -m src.models.coverage`",
        "> از روی runهای ثبت‌شده در MLflow (experiment `phase7`).",
        "",
        "**این گزارش نمی‌گوید چه چیزی باید زده شود — می‌گوید چه چیزی زده *نشده*.** نزدن یک",
        "نقطه اشکالی ندارد؛ نگفتنِ اینکه نزده‌ای اشکال دارد (بند 7.2). هر نقطه‌ی نزده یا باید",
        "زده شود یا دلیل کنارگذاشتنش در `reports/phase7/incompatible_models.md` و کارت مدل بیاید.",
        "",
        f"**تعداد کل runهای ثبت‌شده: {len(runs)}**",
        "",
        "## پوشش هر محور",
        "",
        "| محور | مقادیر زده‌شده | مقادیر **زده‌نشده** | پوشش |",
        "|---|---|---|---|",
    ]

    for axis, allowed in AXES.items():
        seen_raw = _values_seen(runs, axis)
        # مقادیر MLflow رشته‌اند؛ برای مقایسه با شبکه‌ی عددی (مثل tau) هر دو را رشته می‌کنیم
        seen = {v for v in allowed if str(v) in seen_raw}
        missing = [v for v in allowed if v not in seen]
        pct = len(seen) / len(allowed) if allowed else 0.0
        icon = "✅" if not missing else ("⚠️" if seen else "❌")
        lines.append(
            f"| `{axis}` | {'، '.join(str(v) for v in sorted(seen, key=str)) or '—'} | "
            f"**{'، '.join(str(v) for v in missing) or '—'}** | {icon} {pct:.0%} |"
        )

    for axis in OPEN_AXES:
        seen = _values_seen(runs, axis)
        lines.append(f"| `{axis}` (مجموعه‌ی باز) | {'، '.join(sorted(seen)) or '—'} | — | — |")

    lines += [
        "",
        "## پوشش خانواده‌ها",
        "",
        "| خانواده | بند | مدل‌های سند | مدل‌های دارای run | مراحل زده‌شده |",
        "|---|---|---|---|---|",
    ]
    for code, fam in FAMILIES.items():
        # نام ستون‌ها همان نام فیلدهای RunConfig است (asdict) — نه نام‌های جدول 7.7.2
        if runs.empty or "params.family" not in runs.columns:
            sub = runs.iloc[0:0]
        else:
            sub = runs[runs["params.family"] == code]
        n_models = sub["params.model_id"].nunique() if "params.model_id" in sub.columns else 0
        stages = sorted(sub["params.stage"].dropna().unique()) if "params.stage" in sub.columns else []
        icon = "✅" if n_models >= fam.n_models else ("⚠️" if n_models else "❌")
        lines.append(f"| {icon} {code} — {fam.name_fa} | {fam.wbs_section} | {fam.n_models} | "
                     f"{n_models} | {'، '.join(stages) or '—'} |")

    lines += [
        "",
        "> ستون «مدل‌های سند» تعداد اعضای جدول «نقشه‌ی سیزده خانواده» است. اختلافش با ستون",
        "> بعدی یعنی چند مدل هنوز حتی یک‌بار هم اجرا نشده‌اند.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    runs = load_runs()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(build_report(runs))
    print(f"✅ {OUTPUT_PATH}  ({len(runs)} run)")


if __name__ == "__main__":
    main()
