"""بند 7.3.1 سند فاز ۷ — هارنس امکان‌سنجی S0: یک برازش با پارامتر پیش‌فرض روی نخستین fold،
τ=۰.۱۰، به‌ازای هر مدلِ یک خانواده. عمومی است — هر ماژول خانواده (``src/models/families/f0X_*.py``)
داده‌ی سطح خودش را آماده می‌کند و این هارنس را صدا می‌زند.

⚠️ **این مرحله رتبه‌بندی نیست.** S0 فقط به سه سؤال جواب می‌دهد: آیا مدل بدون خطا برازش
می‌شود؟ چقدر طول می‌کشد؟ خروجی‌اش شکل/بازه‌ی معقولی دارد؟ رتبه‌بندی کار S1 و انتخاب کار
S2/S3 است (بند 7.3.2: «S1 مدلی را حذف نمی‌کند؛ فقط ترتیب می‌دهد»).

با این حال معیارهای عملیاتی (pinball، نرخ کمبود، ٪کاهش هدررفت) **همین‌جا هم** محاسبه
می‌شوند، چون پیش‌بینی و مقدار واقعی هر دو در دست‌اند و هزینه‌شان صفر است. این اعداد
**نشانه‌ی اولیه‌اند، نه نتیجه** — با پارامتر پیش‌فرض و روی یک fold به‌دست آمده‌اند.
برای زمینه، همان معیارها برای خط پایه‌ی برنده‌ی فاز ۶ (B3) روی همان fold هم گزارش می‌شود.

نتیجه در ``reports/phase7/S0_feasibility.json`` انباشته می‌شود (کلید = family/level/model_id)،
تا اجرای S0 یک خانواده نتایج خانواده‌های دیگر را پاک نکند؛ ``S0_feasibility.md`` از روی آن
بازتولید می‌شود.
"""

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

import mlflow
import numpy as np
import pandas as pd

from src.baselines import b3_empirical_quantile, operational_metrics
from src.config import REPORTS_DIR
from src.models.axes import TUNING_TAU, RunConfig
from src.models.tracking import start_model_run

PHASE7_DIR = REPORTS_DIR / "phase7"
RESULTS_JSON = PHASE7_DIR / "S0_feasibility.json"
RESULTS_MD = PHASE7_DIR / "S0_feasibility.md"

#: بند 7.3.1 — τ ثابت برای S0/S1/S2 (نقطه‌ی عملیاتی پروژه)؛ حساسیت τ فقط در S3.
#: مقدارش از `axes.TUNING_TAU` می‌آید تا هرگز از سند و از بقیه‌ی مراحل جدا نیفتد.
S0_TAU = TUNING_TAU
#: بند 7.3.3 — دروازه‌ی S0→S1: زمان یک برازش باید کمتر از این باشد (یا مسیر GPU لازم است)
S0_TIME_LIMIT_SECONDS = 30 * 60

#: زیرمجموعه‌ی معیارهای `operational_metrics` که در جدول S0 نمایش داده می‌شوند
_REPORTED_METRICS = ("pinball", "shortage_rate", "waste_reduction_pct", "RMSE_rho", "MAE_portions")


@dataclass
class S0Result:
    family: str
    level: str
    model_id: str
    status: str  # "pass" | "slow" | "fail"
    seconds: float
    error: str | None = None
    output_mean: float | None = None
    output_min: float | None = None
    output_max: float | None = None
    mlflow_run_id: str | None = None
    #: خروجی `src.baselines.operational_metrics` — ⚠️ نشانه‌ی اولیه، نه رتبه‌بندی
    metrics: dict = field(default_factory=dict)


def run_family_s0(family: str, level: str, models: dict[str, Callable], train: pd.DataFrame,
                  test: pd.DataFrame, feature_set: str, data_snapshot_hash: str, cv_folds_hash: str,
                  dataset_source: str, tau: float = S0_TAU, target: str = "rho", seed: int = 42,
                  **axis_overrides) -> list[S0Result]:
    """هر مدل را دقیقاً یک‌بار، با پارامتر پیش‌فرض، برازش و زمان‌سنجی می‌کند — و همزمان یک
    MLflow run با tag/param اجباری بند 7.7.2 باز می‌کند (``stage="S0"``)، تا حتی نتیجه‌ی
    امکان‌سنجی هم در MLflow قابل‌پرس‌وجو باشد، نه فقط در `reports/phase7/S0_feasibility.md`.

    ``axis_overrides`` بقیه‌ی محورهای بند 7.9.1 را می‌گیرد (``scope``، ``weighting``،
    ``architecture``، ``output_aggregation``). پیش‌فرضشان همان پیش‌فرض ``RunConfig`` است —
    ولی چون هر کدام به‌عنوان param در MLflow ثبت می‌شود، `src/models/coverage.py` می‌تواند
    نشان دهد کدام نقاط این محورها هنوز زده نشده‌اند.

    خطا اجرای مدل بعدی را متوقف نمی‌کند — به‌عنوان ``status="fail"`` هم در نتیجه و هم در
    MLflow (tag ``outcome=failed``) ثبت می‌شود تا بقیه‌ی مدل‌ها هم آزموده شوند.
    """
    results = []
    for model_id, fn in models.items():
        t0 = time.time()
        cfg = RunConfig(family=family, model_id=model_id, stage="S0", seed=seed,
                        level=level, feature_set=feature_set, tau=tau, target=target,
                        **axis_overrides)
        with start_model_run(cfg, data_snapshot_hash=data_snapshot_hash, cv_folds_hash=cv_folds_hash,
                             train=train, test=test, dataset_source=dataset_source,
                             source_fn=fn) as run:
            run_id = run.info.run_id
            try:
                out = np.asarray(fn(train, test, tau), dtype=float)
                dt = time.time() - t0
                mlflow.log_metric("fit_seconds", dt)

                if out.shape != (len(test),) or not np.all(np.isfinite(out)):
                    err = f"شکل/مقدار نامعتبر: shape={out.shape}"
                    mlflow.set_tag("outcome", "failed")
                    mlflow.set_tag("error", err)
                    results.append(S0Result(family, level, model_id, "fail", dt, error=err, mlflow_run_id=run_id))
                    continue

                metrics = operational_metrics(test, out, tau)
                status = "pass" if dt <= S0_TIME_LIMIT_SECONDS else "slow"
                mlflow.set_tag("outcome", status)
                mlflow.log_metrics({
                    "output_mean": float(out.mean()),
                    "output_min": float(out.min()),
                    "output_max": float(out.max()),
                    **{k: v for k, v in metrics.items() if np.isfinite(v)},
                })
                results.append(S0Result(family, level, model_id, status, dt,
                                        output_mean=float(out.mean()), output_min=float(out.min()),
                                        output_max=float(out.max()), mlflow_run_id=run_id,
                                        metrics=metrics))
            except Exception as e:
                dt = time.time() - t0
                err = f"{type(e).__name__}: {e}"
                mlflow.log_metric("fit_seconds", dt)
                mlflow.set_tag("outcome", "failed")
                mlflow.set_tag("error", err[:250])
                results.append(S0Result(family, level, model_id, "fail", dt, error=err, mlflow_run_id=run_id))
    return results


def baseline_reference(train: pd.DataFrame, test: pd.DataFrame, tau: float = S0_TAU) -> dict:
    """معیارهای خط پایه‌ی برنده‌ی فاز ۶ (B3، کوانتایل تجربی گروهی) روی همان fold — سقفی که
    بند 7.25.4 سؤال ۱ می‌پرسد آیا هیچ مدلی آن را معناداری می‌برد یا نه."""
    return operational_metrics(test, b3_empirical_quantile(train, test, tau), tau)


def _load_store() -> dict:
    return json.loads(RESULTS_JSON.read_text()) if RESULTS_JSON.exists() else {}


def save_results(results: list[S0Result], baseline: dict | None = None) -> None:
    store = _load_store()
    for r in results:
        store[f"{r.family}/{r.level}/{r.model_id}"] = asdict(r)
    if baseline is not None:
        store["_baseline_B3"] = baseline
    PHASE7_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(json.dumps(store, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    _render_markdown(store)


def _fmt(value: float | None, spec: str) -> str:
    return format(value, spec) if value is not None and np.isfinite(value) else "—"


def _render_markdown(store: dict) -> None:
    baseline = store.get("_baseline_B3")
    models = {k: v for k, v in store.items() if not k.startswith("_")}

    lines = [
        "# امکان‌سنجی S0 — فاز ۷",
        "",
        "> بند 7.3.1 سند `doc/WBS-phase7-modeling.md`. یک برازش با پارامتر **پیش‌فرض**، روی "
        f"**نخستین fold**، τ={S0_TAU}. تولید خودکار با `python -m src.models.families.f0X_*`.",
        "",
        "## ⚠️ این جدول رتبه‌بندی نیست",
        "",
        "S0 فقط به سه سؤال جواب می‌دهد: آیا مدل بدون خطا برازش می‌شود؟ چقدر طول می‌کشد؟ خروجی‌اش",
        "شکل و بازه‌ی معقولی دارد؟ ستون‌های معیار **نشانه‌ی اولیه‌اند، نه نتیجه** — با پارامتر",
        "پیش‌فرض و روی یک fold به‌دست آمده‌اند. رتبه‌بندی کار S1 و انتخاب کار S2/S3 است، و طبق",
        "قاعده‌ی 7.3.2 هیچ مدلی پیش از گرفتن بودجه‌ی کامل S2 بازنده اعلام نمی‌شود.",
        "",
        "## وضعیت اجرا و معیارهای اولیه",
        "",
        "| خانواده | سطح | مدل | وضعیت | زمان (ث) | **pinball@۰.۱۰** | نرخ کمبود | ٪کاهش هدررفت | RMSE | MLflow | یادداشت |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    if baseline:
        lines.append(
            f"| — | — | **B3 (خط پایه‌ی فاز ۶)** | 🎯 مرجع | — | **{_fmt(baseline.get('pinball'), '.5f')}** | "
            f"{_fmt(baseline.get('shortage_rate'), '.1%')} | {_fmt(baseline.get('waste_reduction_pct'), '.1%')} | "
            f"{_fmt(baseline.get('RMSE_rho'), '.4f')} | — | سقفی که فاز ۷ باید بشکند |"
        )

    ranked = sorted(models.items(),
                    key=lambda kv: kv[1].get("metrics", {}).get("pinball") or float("inf"))
    for _key, r in ranked:
        m = r.get("metrics") or {}
        icon = {"pass": "✅", "slow": "⚠️ کند", "fail": "❌"}[r["status"]]
        run_s = f"`{r['mlflow_run_id'][:8]}`" if r.get("mlflow_run_id") else "—"
        note = (r.get("error") or "").replace("|", "\\|")
        lines.append(
            f"| {r['family']} | {r['level']} | `{r['model_id']}` | {icon} | {r['seconds']:.2f} | "
            f"{_fmt(m.get('pinball'), '.5f')} | {_fmt(m.get('shortage_rate'), '.1%')} | "
            f"{_fmt(m.get('waste_reduction_pct'), '.1%')} | {_fmt(m.get('RMSE_rho'), '.4f')} | "
            f"{run_s} | {note} |"
        )

    n_pass = sum(1 for r in models.values() if r["status"] == "pass")
    n_slow = sum(1 for r in models.values() if r["status"] == "slow")
    n_fail = sum(1 for r in models.values() if r["status"] == "fail")
    lines += [
        "",
        f"**{n_pass}/{len(models)} پاس · {n_slow} کند (>۳۰ دقیقه) · {n_fail} شکست.**",
        "",
        "ردیف‌ها بر اساس pinball مرتب شده‌اند تا خوانده‌شدنشان راحت باشد — این ترتیب **هیچ تصمیمی**",
        "را توجیه نمی‌کند (بند 7.3.2).",
        "",
        "## مسیر بعدی هر مدل",
        "",
        "| مرحله | چه چیزی اضافه می‌کند | کجا دیده می‌شود |",
        "|---|---|---|",
        "| S1 | ۲۰ trial تصادفی × ۳ fold ⇒ رتبه‌بندی مقدماتی | همین‌جا + MLflow (`stage=S1`) |",
        "| S2 | بودجه‌ی کامل Optuna + فیچرست اختصاصی + ۵ fold | `reports/models/{model_id}.md` (کارت ۱۴ گامی) |",
        "| S3 | همه‌ی τها + کالیبراسیون + SHAP | `reports/phase7/model_comparison.md` (بند 7.25) |",
    ]
    RESULTS_MD.write_text("\n".join(lines) + "\n")


def print_summary(results: list[S0Result], baseline: dict | None = None) -> bool:
    """چاپ خط‌به‌خط + برگرداندن اینکه آیا همه‌چیز pass/slow بود (نه fail)."""
    if baseline:
        print(f"  🎯 {'B3 (خط پایه‌ی فاز ۶)':<32s} {'':>7s}  pinball={baseline['pinball']:.5f}  "
              f"کمبود={baseline['shortage_rate']:.1%}")
    all_ok = True
    for r in sorted(results, key=lambda x: (x.metrics or {}).get("pinball") or float("inf")):
        icon = {"pass": "✅", "slow": "⚠️", "fail": "❌"}[r.status]
        if r.metrics:
            extra = (f"pinball={r.metrics['pinball']:.5f}  کمبود={r.metrics['shortage_rate']:.1%}  "
                     f"هدررفت−={r.metrics['waste_reduction_pct']:.1%}")
        else:
            extra = r.error or ""
        print(f"  {icon} {r.model_id:<32s} {r.seconds:6.2f}s  {extra}")
        all_ok &= r.status != "fail"
    return all_ok
