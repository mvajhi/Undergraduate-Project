"""بند 7.3.1/7.6 سند فاز ۷ — هارنس S2: بهینه‌سازی کامل با Optuna TPE، ۵ fold،
بودجه‌ی جدول 7.6.2، شاهد همگرایی A6.

⚠️ **تفاوت ساختاری با S1.** در S1 هر ۲۰ trial با ``RandomSampler`` از پیش و مستقل
نمونه‌گیری می‌شدند — پس (مدل×trial) قابل تخت‌کردن و توزیع آزاد بین workerها بود. TPE
اما **تطبیقی و ترتیبی** است: هر trial بر مبنای نتیجه‌ی trialهای قبلیِ **همان مدل**
پیشنهاد می‌شود. پس اینجا موازی‌سازی **بین مدل‌ها**ست، نه بین trialها: هر worker یک مدل
را کامل (همه‌ی trialهایش، پشت‌سرهم) اجرا می‌کند؛ استخر به‌طور طبیعی مدل‌های سریع را زودتر
تمام و worker آزادشده را به مدل بعدی می‌دهد — مدل کند (رگرسیون کوانتایل ترکیبی) یک
worker را برای مدت طولانی اشغال می‌کند ولی بقیه را متوقف نمی‌کند.

قاعده‌ی A6 (شاهد همگرایی): منحنی best-so-far باید در ۲۵٪ پایانی trialها بهبود <۱٪
نشان دهد. اینجا محاسبه و همراه نتیجه ذخیره می‌شود — کارت مدل (بند 7.4 گام ۹) از
همین داده تغذیه می‌شود.
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import importlib  # noqa: E402
import json  # noqa: E402
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
from src.config import REPORTS_DIR, ROOT_DIR
from src.models.axes import TUNING_TAU, RunConfig
from src.models.spaces import SPACES, sample, trial_budget
from src.models.tracking import start_model_run

optuna.logging.set_verbosity(optuna.logging.WARNING)

PHASE7_DIR = REPORTS_DIR / "phase7"
#: بند 7.6.3 — «study هر مدل در optuna_studies/{model_id}.db (SQLite) ذخیره می‌شود تا
#: قابل بازتولید و ادامه‌پذیر باشد». علاوه‌براین، این تنها راهی است که بعداً (کارت مدل
#: گام ۱۰) بشود اهمیت هایپرپارامتر را با fANOVA از روی *همه‌ی* trialها محاسبه کرد —
#: نه فقط بهترین یکی، که در ModelS2Result نگه داشته می‌شود.
OPTUNA_STUDIES_DIR = ROOT_DIR / "optuna_studies"
S2_TAU = TUNING_TAU
#: بند 7.3.1 — S2 روی هر ۵ fold اجرا می‌شود (S0=۱، S1=۳)
N_S2_FOLDS = 5
#: بند 7.6.3: «سقف بالا هم خطر است» + آزمون پایداری — بهترین پیکربندی باید در ≥۳ از ۵
#: fold هم در ۱۰٪ برتر باشد؛ اینجا فقط پرچم‌گذاری می‌شود، انتخاب نهایی به کارت مدل می‌ماند
_STABILITY_TOP_FRACTION = 0.10
_STABILITY_MIN_FOLDS = 3
#: قاعده‌ی A6 — درصد پایانی trialها که باید بهبود ناچیز نشان دهند
_CONVERGENCE_TAIL_FRACTION = 0.25
_CONVERGENCE_IMPROVEMENT_THRESHOLD = 0.01


@dataclass
class ModelS2Result:
    model_id: str
    n_hyperparams: int
    n_trials: int
    best_pinball: float
    best_hyperparams: dict
    fold_pinballs_at_best: list
    trial_history: list  # [(trial_idx, mean_pinball), ...] به ترتیب اجرا — برای A6/فANOVA
    converged: bool
    seconds: float
    n_fail: int
    stable_top10pct_folds: int  # در چند از ۵ fold، بهترین پیکربندی در ۱۰٪ برتر همان fold هم بود


def _feature_set_label(quantreg: bool) -> str:
    return "FS_F01_quantreg_v1" if quantreg else "FS_F01_linear_v1"


def study_storage_url(model_id: str) -> str:
    OPTUNA_STUDIES_DIR.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{OPTUNA_STUDIES_DIR / f'{model_id}.db'}"


def _run_model_s2(model_id: str, fn: Callable, design_fn: Callable, folds: list,
                  family: str, level: str, data_snapshot_hash: str, cv_folds_hash: str,
                  dataset_source: str, seed: int = 42, quantreg: bool = False) -> ModelS2Result:
    """یک مدل را کامل با TPE تنظیم می‌کند — همه‌ی trialهایش پشت‌سرهم، هرکدام روی هر ۵ fold.

    ``study`` روی SQLite پایدار می‌شود (بند 7.6.3) — اگر این تابع قطع و دوباره اجرا شود
    (مثلاً برای رگرسیون کوانتایل ترکیبی که ساعت‌ها طول می‌کشد)، trialهای قبلی از دست
    نمی‌روند و فقط باقیِ بودجه اجرا می‌شود.
    """
    n_hp = SPACES[model_id].n_hyperparams
    n_trials = trial_budget(n_hp, "S2")
    feature_set = _feature_set_label(quantreg)

    study = optuna.create_study(
        study_name=f"{family}_{model_id}_S2", storage=study_storage_url(model_id),
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed), load_if_exists=True,
    )
    already_done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    n_remaining = max(0, n_trials - len(already_done))
    history: list[tuple[int, float]] = [(t.number, t.value if t.value is not None else float("inf"))
                                        for t in already_done]
    per_fold_pinballs: dict[int, list[float]] = {}  # trial_idx -> [pinball هر fold] — فقط trialهای این اجرا
    n_fail = 0
    t0 = time.time()

    designed_folds = [design_fn(tr, te) for tr, te in folds]  # یک‌بار برای کل study

    for _ in range(n_remaining):
        trial = study.ask()
        trial_idx = trial.number
        hp = sample(model_id, trial)
        cfg = RunConfig(family=family, model_id=model_id, stage="S2", seed=seed,
                        level=level, feature_set=feature_set, tau=S2_TAU)
        pinballs = []
        try:
            with start_model_run(cfg, data_snapshot_hash=data_snapshot_hash, cv_folds_hash=cv_folds_hash,
                                 train=folds[0][0], dataset_source=dataset_source, source_fn=fn,
                                 n_trials=n_trials, sampler="TPE") as run:
                mlflow.log_params({f"hp_{k}": v for k, v in hp.items()})
                mlflow.log_param("trial_idx", trial_idx)
                for fold_idx, (tr, te) in enumerate(folds):
                    Xtr, Xte = designed_folds[fold_idx]
                    out = np.asarray(fn(tr, te, S2_TAU, **hp), dtype=float)
                    if out.shape != (len(te),) or not np.all(np.isfinite(out)):
                        raise ValueError(f"شکل/مقدار نامعتبر در fold{fold_idx}")
                    m = operational_metrics(te, out, S2_TAU)
                    pinballs.append(m["pinball"])
                    mlflow.log_metric("pinball", m["pinball"], step=fold_idx)
                mean_pb = float(np.mean(pinballs))
                mlflow.log_metrics({"pinball_mean": mean_pb, "fit_seconds": time.time() - t0})
                mlflow.set_tag("outcome", "pass")
            study.tell(trial, mean_pb)
            history.append((trial_idx, mean_pb))
            per_fold_pinballs[trial_idx] = pinballs
        except Exception as e:
            n_fail += 1
            study.tell(trial, float("inf"))
            history.append((trial_idx, float("inf")))
            try:
                mlflow.set_tag("outcome", "failed")
                mlflow.set_tag("error", str(e)[:250])
            except Exception:
                pass

    ok_trials = [h for h in history if np.isfinite(h[1])]
    if not ok_trials:
        return ModelS2Result(model_id, n_hp, n_trials, float("nan"), {}, [], history, False,
                             time.time() - t0, n_fail, 0)

    best_idx, best_pb = min(ok_trials, key=lambda h: h[1])
    best_trial = next(t for t in study.trials if t.number == best_idx)

    # قاعده‌ی A6: بهبود بهترین-تاکنون در ۲۵٪ پایانی trialها باید ناچیز باشد
    best_so_far = []
    running_best = float("inf")
    for _, pb in history:
        running_best = min(running_best, pb)
        best_so_far.append(running_best)
    tail_n = max(1, int(len(best_so_far) * _CONVERGENCE_TAIL_FRACTION))
    tail = best_so_far[-tail_n:]
    tail_improvement = (tail[0] - tail[-1]) / tail[0] if tail[0] > 0 else 0.0
    converged = tail_improvement < _CONVERGENCE_IMPROVEMENT_THRESHOLD

    # آزمون پایداری بند 7.6.3: بهترین پیکربندی در چند fold هم در ۱۰٪ برتر همان fold بود؟
    # ⚠️ محدودیت شناخته‌شده: اگر این تابع از یک اجرای قطع‌شده ادامه یابد (resume) و
    # بهترین trial متعلق به نشست قبلی باشد، per_fold_pinballs آن را ندارد (فقط داخل
    # همین اجرا پر می‌شود) — pinball هر fold آن گم می‌شود، نه اینکه ناپایدار باشد. برای
    # F01 اهمیتی ندارد چون این اجرا از صفر شروع شده؛ برای خانواده‌های بعدی که ممکن است
    # قطع/ادامه پیدا کنند، رفعش نیازمند بازخوانی metric هر fold از MLflow است.
    stable_count = 0
    fold_pinballs_at_best = per_fold_pinballs.get(best_idx, [])
    for fold_idx in range(len(folds)):
        fold_vals = sorted(pf[fold_idx] for pf in per_fold_pinballs.values() if fold_idx < len(pf))
        if not fold_vals:
            continue
        cutoff = fold_vals[max(0, int(len(fold_vals) * _STABILITY_TOP_FRACTION) - 1)]
        if fold_idx < len(fold_pinballs_at_best) and fold_pinballs_at_best[fold_idx] <= cutoff:
            stable_count += 1

    return ModelS2Result(model_id, n_hp, n_trials, best_pb, best_trial.params,
                         fold_pinballs_at_best, history, converged, time.time() - t0, n_fail, stable_count)


# ---------------------------------------------------------------------------
# اجرای موازی بین مدل‌ها
# ---------------------------------------------------------------------------

_worker_state: dict = {}


def _init_worker(family_module_path: str, folds: list) -> None:
    mod = importlib.import_module(family_module_path)
    _worker_state["MODELS"] = mod.MODELS
    _worker_state["QUANTREG_MODELS"] = getattr(mod, "QUANTREG_MODEL_IDS", frozenset())
    _worker_state["design_linear"] = mod._design_s2
    _worker_state["folds"] = folds


def _worker(args) -> ModelS2Result:
    (model_id, family, level, data_snapshot_hash, cv_folds_hash, dataset_source, seed) = args
    fn = _worker_state["MODELS"][model_id]
    quantreg = model_id in _worker_state["QUANTREG_MODELS"]
    design_fn = lambda tr, te: _worker_state["design_linear"](tr, te, quantreg=quantreg)  # noqa: E731
    folds = _worker_state["folds"]
    try:
        return _run_model_s2(model_id, fn, design_fn, folds, family, level,
                             data_snapshot_hash, cv_folds_hash, dataset_source, seed, quantreg)
    except Exception as e:
        return ModelS2Result(model_id, 0, 0, float("nan"), {}, [], [], False, 0.0, 1, 0)  # noqa


def run_family_s2(family: str, level: str, family_module_path: str, model_ids: list[str],
                  folds: list, data_snapshot_hash: str, cv_folds_hash: str, dataset_source: str,
                  seed: int = 42, n_jobs: int = 6,
                  on_result: Callable[[str, "ModelS2Result"], None] | None = None
                  ) -> dict[str, ModelS2Result]:
    """``on_result(model_id, result)`` بعد از اتمام **هر** مدل صدا زده می‌شود (نه فقط در
    پایان کل خانواده) — چون رگرسیون کوانتایل ترکیبی می‌تواند ساعت‌ها طول بکشد و بقیه‌ی
    مدل‌های سریع نباید منتظرش بمانند تا نتیجه‌شان ذخیره/تحلیل شود."""
    if len(folds) != N_S2_FOLDS:
        raise ValueError(f"S2 دقیقاً {N_S2_FOLDS} fold می‌خواهد، نه {len(folds)}")

    jobs = [(mid, family, level, data_snapshot_hash, cv_folds_hash, dataset_source, seed)
           for mid in model_ids]
    print(f"S2 — {family} ({level}): {len(jobs)} مدل، هرکدام با بودجه‌ی خودش (جدول 7.6.2)، "
         f"روی {n_jobs} worker موازی (موازی‌سازی بین مدل‌ها، نه بین trial — TPE ترتیبی است)")

    results: dict[str, ModelS2Result] = {}
    ctx = multiprocessing.get_context("spawn")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx, initializer=_init_worker,
                             initargs=(family_module_path, folds)) as ex:
        futures = {ex.submit(_worker, job): job[0] for job in jobs}
        for fut in as_completed(futures):
            mid = futures[fut]
            r = fut.result()
            results[mid] = r
            dt = time.time() - t0
            print(f"  [{dt:6.0f}s] {mid:<32s} best={r.best_pinball:.5f}  n_trials={r.n_trials}  "
                 f"همگرا={r.converged}  پایداری={r.stable_top10pct_folds}/5  شکست={r.n_fail}")
            if on_result is not None:
                on_result(mid, r)
    return results


# ---------------------------------------------------------------------------
# گزارش
# ---------------------------------------------------------------------------

def _paths(family: str) -> tuple:
    return PHASE7_DIR / f"S2_tuning_{family}.json", PHASE7_DIR / f"S2_tuning_{family}.md"


def save_one_result(model_id: str, result: ModelS2Result, family: str, level: str) -> None:
    """ذخیره‌ی افزایشی — بعد از اتمام هر مدل صدا زده می‌شود، نه فقط در پایان کل خانواده."""
    json_path, md_path = _paths(family)
    payload = json.loads(json_path.read_text()) if json_path.exists() else {}
    payload[model_id] = asdict(result)
    PHASE7_DIR.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    _render_markdown(payload, md_path, family, level)


def save_results(results: dict[str, ModelS2Result], family: str, level: str,
                 baseline_pinball: float | None = None) -> None:
    json_path, md_path = _paths(family)
    payload = json.loads(json_path.read_text()) if json_path.exists() else {}
    payload.update({mid: asdict(r) for mid, r in results.items()})
    if baseline_pinball is not None:
        payload["_baseline_B3"] = baseline_pinball
    PHASE7_DIR.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    _render_markdown(payload, md_path, family, level)


def _render_markdown(payload: dict, path, family: str, level: str) -> None:
    baseline = payload.get("_baseline_B3")
    rows = {k: v for k, v in payload.items() if not k.startswith("_")}

    lines = [
        f"# بهینه‌سازی S2 — {family} ({level})",
        "",
        f"> بند 7.3.1/7.6 سند `doc/WBS-phase7-modeling.md`. Optuna TPE، ۵ fold، τ={S2_TAU}، "
        "بودجه‌ی trial طبق جدول 7.6.2. کارت مدل ۱۴ گامی برای هر مدل هنوز جداگانه نوشته می‌شود.",
        "",
    ]
    if baseline is not None:
        lines += [f"🎯 **مرجع — خط پایه‌ی فاز ۶ (B3) روی هر ۵ fold: pinball_mean = {baseline:.5f}**", ""]

    lines += [
        "| مدل | بهترین pinball | trialها | همگرا (A6) | پایداری (۷.۶.۳) | شکست | زمان |",
        "|---|---|---|---|---|---|---|",
    ]
    ranked = sorted(rows.items(), key=lambda kv: kv[1]["best_pinball"] if np.isfinite(kv[1]["best_pinball"]) else 1e9)
    for mid, r in ranked:
        mark = " 🎯" if baseline is not None and np.isfinite(r["best_pinball"]) and r["best_pinball"] < baseline else ""
        conv = "✅" if r["converged"] else "⚠️"
        lines.append(
            f"| `{mid}`{mark} | **{r['best_pinball']:.5f}** | {r['n_trials']} | {conv} | "
            f"{r['stable_top10pct_folds']}/5 | {r['n_fail']} | {r['seconds']:.0f}s |"
        )
    path.write_text("\n".join(lines) + "\n")


def baseline_reference_5fold(folds: list, tau: float = S2_TAU) -> float:
    pbs = [operational_metrics(te, b3_empirical_quantile(tr, te, tau), tau)["pinball"] for tr, te in folds]
    return float(np.mean(pbs))
