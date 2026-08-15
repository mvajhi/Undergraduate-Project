"""بند 7.3.1 سند فاز ۷ — هارنس غربالگری S1: ۲۰ trial تصادفی × ۳ fold، بدون حذف هیچ مدلی.

⚠️ **S1 مدلی را حذف نمی‌کند؛ فقط ترتیب می‌دهد** (بند 7.3.2) — رتبه‌بندی این مرحله برای
کالیبره‌کردن بودجه‌ی S2 استفاده می‌شود، نه برای بازنده اعلام‌کردن.

اجرا با استخر پردازش موازی (`ProcessPoolExecutor`): این خانواده به‌تنهایی ۱۵ مدل × ۲۰
trial × ۳ fold = ۹۰۰ برازش دارد و بیشترشان زیر یک ثانیه‌اند، ولی بعضی (مثل رگرسیون
کوانتایل ترکیبی که یک LP بزرگ حل می‌کند) چند دقیقه طول می‌کشند — بدون موازی‌سازی، این
عدم‌تعادل کل اجرا را کند می‌کند. داده‌ی هر ۳ fold فقط **یک‌بار به‌ازای هر worker** (نه
هر job) با ``initializer`` منتقل می‌شود تا هزینه‌ی pickle تکرار نشود.
"""

import os

# ⚠️ باید پیش از import شدنِ numpy/scipy (اینجا یا در هر worker fresh با spawn) تنظیم شود:
# بدون این، هر یک از N فرآیند موازی خودش می‌تواند M ریسه‌ی BLAS داخلی باز کند و روی
# ماشین ۱۲ هسته‌ای N×M >> 12 بیش‌اشتراکی ایجاد کند — دقیقاً برعکس هدف موازی‌سازی.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import importlib  # noqa: E402
import multiprocessing  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from dataclasses import asdict, dataclass, field  # noqa: E402
from typing import Callable  # noqa: E402

import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import optuna  # noqa: E402
import pandas as pd  # noqa: E402

from src.baselines import b3_empirical_quantile, operational_metrics
from src.config import REPORTS_DIR
from src.models.axes import TUNING_TAU, RunConfig
from src.models.spaces import SPACES, sample
from src.models.tracking import start_model_run

optuna.logging.set_verbosity(optuna.logging.WARNING)

PHASE7_DIR = REPORTS_DIR / "phase7"

#: بند 7.3.1
N_TRIALS = 20
N_SCREENING_FOLDS = 3
S1_TAU = TUNING_TAU
#: ۱۲ هسته‌ی سیستم — چند تا برای OS/بقیه‌ی کارها آزاد می‌ماند
DEFAULT_N_JOBS = max(1, min(10, (os.cpu_count() or 4) - 2))


@dataclass(frozen=True)
class TrialJob:
    family: str
    level: str
    model_id: str
    trial_idx: int
    hyperparams: dict
    feature_set: str
    data_snapshot_hash: str
    cv_folds_hash: str
    dataset_source: str
    target: str
    seed: int


@dataclass
class TrialResult:
    family: str
    level: str
    model_id: str
    trial_idx: int
    hyperparams: dict
    status: str  # "pass" | "fail"
    seconds: float
    fold_pinballs: list = field(default_factory=list)
    mean_pinball: float | None = None
    error: str | None = None
    mlflow_run_id: str | None = None


def build_jobs(family: str, level: str, model_ids: list[str], feature_set: str,
              data_snapshot_hash: str, cv_folds_hash: str, dataset_source: str,
              target: str = "rho", seed: int = 42, n_trials: int = N_TRIALS) -> list[TrialJob]:
    """نمونه‌گیری قطعی ۲۰ trial به‌ازای هر مدل (پیش از ارسال به استخر موازی) — یک
    ``optuna.Study`` تصادفی مستقل با seed مشتق‌شده از model_id، تا نتیجه بازتولیدپذیر
    باشد ولی مدل‌ها همان یک جریان تصادفی را کورکورانه به اشتراک نگذارند.
    """
    jobs = []
    for model_id in model_ids:
        if model_id not in SPACES:
            raise KeyError(f"فضای هایپرپارامتر {model_id!r} ثبت نشده — src/models/spaces.py را ببینید")
        model_seed = seed + (hash(model_id) % 10_000)
        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=model_seed))
        for trial_idx in range(n_trials):
            trial = study.ask()
            hp = sample(model_id, trial)
            jobs.append(TrialJob(family, level, model_id, trial_idx, hp, feature_set,
                                 data_snapshot_hash, cv_folds_hash, dataset_source, target, seed))
    return jobs


# ---------------------------------------------------------------------------
# اجرای هر job — سطح ماژول برای pickle-پذیری در ProcessPoolExecutor
# ---------------------------------------------------------------------------

_worker_state: dict = {}


def _init_worker(family_module_path: str, folds: list) -> None:
    """یک‌بار به‌ازای هر worker process صدا زده می‌شود — داده و توابع مدل را در حافظه‌ی
    همان worker نگه می‌دارد تا هر job مجبور نباشد دوباره DataFrame را pickle/منتقل کند."""
    mod = importlib.import_module(family_module_path)
    _worker_state["MODELS"] = mod.MODELS
    _worker_state["folds"] = folds


def _run_one_trial(job: TrialJob, fn: Callable, folds: list) -> TrialResult:
    """یک trial را روی هر ۳ fold غربالگری می‌کند و در **یک** MLflow run ثبت می‌شود
    (نه سه run جدا) — pinball هر fold با ``step=fold_idx``، میانگین به‌عنوان معیار اصلی."""
    t0 = time.time()
    cfg = RunConfig(family=job.family, model_id=job.model_id, stage="S1", seed=job.seed,
                    level=job.level, feature_set=job.feature_set, tau=S1_TAU, target=job.target)
    with start_model_run(cfg, data_snapshot_hash=job.data_snapshot_hash, cv_folds_hash=job.cv_folds_hash,
                         train=folds[0][0], dataset_source=job.dataset_source, source_fn=fn,
                         n_trials=N_TRIALS, sampler="RandomSampler") as run:
        run_id = run.info.run_id
        mlflow.log_params({f"hp_{k}": v for k, v in job.hyperparams.items()})
        mlflow.log_param("trial_idx", job.trial_idx)
        pinballs: list[float] = []
        try:
            for fold_idx, (tr, te) in enumerate(folds):
                out = np.asarray(fn(tr, te, S1_TAU, **job.hyperparams), dtype=float)
                if out.shape != (len(te),) or not np.all(np.isfinite(out)):
                    raise ValueError(f"شکل/مقدار نامعتبر در fold{fold_idx}: shape={out.shape}")
                m = operational_metrics(te, out, S1_TAU)
                pinballs.append(m["pinball"])
                mlflow.log_metric("pinball", m["pinball"], step=fold_idx)
                mlflow.log_metric("shortage_rate", m["shortage_rate"], step=fold_idx)
            mean_pb = float(np.mean(pinballs))
            dt = time.time() - t0
            mlflow.log_metrics({"pinball_mean": mean_pb, "fit_seconds": dt})
            mlflow.set_tag("outcome", "pass")
            return TrialResult(job.family, job.level, job.model_id, job.trial_idx, job.hyperparams,
                               "pass", dt, pinballs, mean_pb, mlflow_run_id=run_id)
        except Exception as e:
            dt = time.time() - t0
            err = f"{type(e).__name__}: {e}"
            mlflow.log_metric("fit_seconds", dt)
            mlflow.set_tag("outcome", "failed")
            mlflow.set_tag("error", err[:250])
            return TrialResult(job.family, job.level, job.model_id, job.trial_idx, job.hyperparams,
                               "fail", dt, pinballs, error=err, mlflow_run_id=run_id)


def _worker(job: TrialJob) -> TrialResult:
    fn = _worker_state["MODELS"][job.model_id]
    folds = _worker_state["folds"]
    try:
        return _run_one_trial(job, fn, folds)
    except Exception as e:  # خطای بیرون از بدنه‌ی run (مثلاً اتصال MLflow) نباید کل استخر را متوقف کند
        return TrialResult(job.family, job.level, job.model_id, job.trial_idx, job.hyperparams,
                           "fail", 0.0, error=f"خطای worker: {type(e).__name__}: {e}")


def run_family_s1(family: str, level: str, family_module_path: str, model_ids: list[str],
                  folds: list, feature_set: str, data_snapshot_hash: str, cv_folds_hash: str,
                  dataset_source: str, seed: int = 42, n_trials: int = N_TRIALS,
                  n_jobs: int = DEFAULT_N_JOBS, progress_every: int = 20) -> list[TrialResult]:
    """۲۰ trial تصادفی × ۳ fold برای هر مدل در ``model_ids``، موازی روی ``n_jobs`` پردازش.

    ``folds`` باید دقیقاً ``N_SCREENING_FOLDS`` عضو ``(train, test)`` باشد (فراخوان انتخاب
    می‌کند کدام fold‌ها — معمولاً سه‌تای اول، چون کوچک‌ترند و غربالگری را ارزان نگه می‌دارند).
    """
    if len(folds) != N_SCREENING_FOLDS:
        raise ValueError(f"S1 دقیقاً {N_SCREENING_FOLDS} fold می‌خواهد، نه {len(folds)}")

    jobs = build_jobs(family, level, model_ids, feature_set, data_snapshot_hash,
                      cv_folds_hash, dataset_source, seed=seed, n_trials=n_trials)
    print(f"S1 — {family} ({level}): {len(model_ids)} مدل × {n_trials} trial × "
         f"{N_SCREENING_FOLDS} fold = {len(jobs)} job روی {n_jobs} worker")

    results: list[TrialResult] = []
    t_start = time.time()
    # spawn (نه fork پیش‌فرض لینوکس): هر worker یک مفسر کاملاً تازه است، پس هم env varهای
    # محدودکننده‌ی ریسه‌ی بالا را از صفر می‌خواند، هم قفل‌های داخلی mlflow/threading را از
    # فرآیند والد به ارث نمی‌برد (منبع دِدلاک‌های نامشخص در فورک بعد از ایمپورت‌های سنگین).
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx, initializer=_init_worker,
                             initargs=(family_module_path, folds)) as ex:
        futures = {ex.submit(_worker, job): job for job in jobs}
        for i, fut in enumerate(as_completed(futures), start=1):
            results.append(fut.result())
            if i % progress_every == 0 or i == len(jobs):
                dt = time.time() - t_start
                n_fail = sum(1 for r in results if r.status == "fail")
                print(f"  {i}/{len(jobs)} انجام شد ({dt:.0f}s) · {n_fail} شکست تاکنون")
    return results


# ---------------------------------------------------------------------------
# مرجع خط پایه — گزارش S1 نباید بدون سقفِ مقایسه خوانده شود
# ---------------------------------------------------------------------------

def baseline_reference_multi_fold(folds: list[tuple[pd.DataFrame, pd.DataFrame]],
                                  tau: float = S1_TAU) -> float:
    """میانگین pinball خط پایه‌ی برنده‌ی فاز ۶ (B3) روی همان چند fold — سقفی که بند
    7.25.4 سؤال ۱ می‌پرسد آیا هیچ مدلی معناداری آن را می‌برد یا نه."""
    pbs = [operational_metrics(te, b3_empirical_quantile(tr, te, tau), tau)["pinball"] for tr, te in folds]
    return float(np.mean(pbs))


# ---------------------------------------------------------------------------
# گزارش
# ---------------------------------------------------------------------------

def _load_store(path) -> dict:
    import json
    return json.loads(path.read_text()) if path.exists() else {}


def save_results(results: list[TrialResult], family: str, level: str,
                 baseline_pinball: float | None = None) -> None:
    """یک فایل JSON/MD به‌ازای هر خانواده — اجرای دوباره‌ی یک خانواده نتایج بقیه را پاک نمی‌کند.

    ``baseline_pinball`` میانگین pinball خط پایه‌ی B3 (بند ۶.۵) روی همان ۳ fold است — تا
    گزارش S1، مثل S0، مرجع مقایسه داشته باشد و رتبه‌بندی «در خلأ» نباشد.
    """
    import json

    json_path = PHASE7_DIR / f"S1_screening_{family}.json"
    md_path = PHASE7_DIR / f"S1_screening_{family}.md"
    store = _load_store(json_path)
    store.setdefault(f"{family}/{level}", []).extend([_serialize(r) for r in results])
    if baseline_pinball is not None:
        store[f"_baseline_B3/{family}/{level}"] = baseline_pinball
    PHASE7_DIR.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(store, ensure_ascii=False, indent=2) + "\n")
    _render_markdown(store, md_path, family, level)


def _serialize(r: TrialResult) -> dict:
    d = asdict(r)
    d["hyperparams"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in d["hyperparams"].items()}
    return d


def _render_markdown(store: dict, path, family: str, level: str) -> None:
    rows = store.get(f"{family}/{level}", [])
    baseline = store.get(f"_baseline_B3/{family}/{level}")
    by_model: dict[str, list[dict]] = {}
    for r in rows:
        by_model.setdefault(r["model_id"], []).append(r)

    lines = [
        f"# غربالگری S1 — {family} ({level})",
        "",
        "> بند 7.3.1 سند `doc/WBS-phase7-modeling.md`. ۲۰ trial تصادفی × ۳ fold، τ="
        f"{S1_TAU}. **⚠️ هیچ مدلی حذف نشده** (بند 7.3.2) — این فقط رتبه‌بندی مقدماتی برای "
        "کالیبره‌کردن بودجه‌ی S2 است.",
        "",
    ]
    if baseline is not None:
        lines += [f"🎯 **مرجع — خط پایه‌ی فاز ۶ (B3) روی همین ۳ fold: pinball_mean = {baseline:.5f}**", ""]
    lines += [
        "## بهترین trial هر مدل (بر اساس میانگین pinball سه fold)",
        "",
        "| مدل | بهترین pinball_mean | بدترین | میانه | ٪شکست | بهترین هایپرپارامتر |",
        "|---|---|---|---|---|---|",
    ]
    summary = []
    for model_id, trials in by_model.items():
        passed = [t for t in trials if t["status"] == "pass" and t["mean_pinball"] is not None]
        n_fail = len(trials) - len(passed)
        if not passed:
            lines.append(f"| `{model_id}` | — | — | — | {n_fail/len(trials):.0%} | همه شکست خوردند |")
            continue
        vals = sorted(t["mean_pinball"] for t in passed)
        best = min(passed, key=lambda t: t["mean_pinball"])
        summary.append((model_id, best["mean_pinball"]))
        lines.append(
            f"| `{model_id}` | **{vals[0]:.5f}** | {vals[-1]:.5f} | {vals[len(vals)//2]:.5f} | "
            f"{n_fail/len(trials):.0%} | `{best['hyperparams']}` |"
        )

    summary.sort(key=lambda x: x[1])
    lines += [
        "",
        "## رتبه‌بندی مقدماتی (⚠️ فقط برای کالیبره‌کردن بودجه‌ی S2 — بند 7.3.2)",
        "",
    ]
    for i, (mid, pb) in enumerate(summary, start=1):
        mark = " 🎯 بهتر از B3" if baseline is not None and pb < baseline else ""
        lines.append(f"{i}. `{mid}` — {pb:.5f}{mark}")
    path.write_text("\n".join(lines) + "\n")
