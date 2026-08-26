"""بند 7.8 سند فاز ۷ — هارنس مشترک اجرای GPU (کولب/کگل).

## چرا این ماژول وجود دارد و چرا `s2_runner` را دوباره استفاده نمی‌کنیم

`s2_runner` برای CPU محلی نوشته شده و موازی‌سازی‌اش **بین مدل‌ها** با
``ProcessPoolExecutor(spawn)`` است. روی کولب این الگو سه مشکل دارد:

1. یک GPU بین چند process قابل تقسیم امن نیست — دو worker هم‌زمان حافظه‌ی کارت را
   پر می‌کنند و اجرا با OOM می‌افتد. روی GPU اجرا باید **ترتیبی** باشد.
2. بودجه‌ی `s2_runner` بر حسب **تعداد trial** است، ولی محدودیت واقعی کولب
   **زمان** است (قطع session). این‌جا بودجه بر حسب دقیقه است و حلقه هر بار پیش از
   شروع trial بعدی زمان باقی‌مانده را می‌سنجد — پس نوت‌بوک همیشه به سلول بسته‌بندی
   می‌رسد، حتی اگر تنظیم ناتمام بماند (نتیجه‌ی ناقص ولی ذخیره‌شده > نتیجه‌ی گم‌شده).
3. مدل‌های این‌جا **سنگین‌اند و باید ذخیره شوند** (بند 7.29.1) — `s2_runner` فقط
   عدد pinball را نگه می‌دارد و خودِ مدل را دور می‌ریزد.

## قرارداد خانواده‌های GPU — دو مرحله‌ای، نه یک‌مرحله‌ای

خانواده‌های CPU فقط ``fit_predict(train, test, tau, **hp) -> np.ndarray`` دارند.
خانواده‌های GPU علاوه‌بر آن باید ``FITTERS`` را هم اعلام کنند::

    FITTERS: dict[str, FamilyFitter]  # هر مدل: fit / predict / save / load

دلیل: با یک تابع یک‌مرحله‌ای نمی‌شود مدل fitشده را ذخیره کرد (بیرون نمی‌آید) و
نمی‌شود یک برازش را برای چند پیش‌بینی بازاستفاده کرد — که برای ACI (بند 7.22.1
عضو ۴) دو برابر شدن هزینه‌ی برازش را می‌ساخت. ``fit_predict`` هر مدل خودکار از
همین دو تا ساخته می‌شود (``FamilyFitter.as_fit_predict``)، پس سازگاری با
`s0_runner`/`conformal`/`calibration` حفظ می‌ماند.
"""

import hashlib
import json
import platform
import time
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from src.baselines import b3_empirical_quantile, operational_metrics, pinball_loss
from src.config import MODELS_DIR, REPORTS_DIR, ROOT_DIR
from src.cv import DATE_COL, diebold_mariano, load_cv_folds, sha256_file
from src.models.axes import TUNING_TAU, RunConfig
from src.models.tracking import aggregate_fold_metrics, log_metrics_dict, start_model_run

#: خروجی‌های اجرای GPU — همه زیر یک ریشه، تا سلول بسته‌بندی (بند 7.8.2 سلول ۸) بداند چه چیزی را جمع کند
GPU_REPORTS_DIR = REPORTS_DIR / "gpu"
GPU_MODELS_DIR = MODELS_DIR / "gpu"
#: بند 7.8.2 سلول ۶ — ردیابی جدا از `mlruns/` محلی، تا ادغام دستی (بند 7.8.3 گام ۵) امن باشد
DEFAULT_TRACKING_DIR = "mlruns_gpu"
#: بند 7.6.3 — study روی SQLite تا اجرا پس از قطع session ادامه‌پذیر باشد
OPTUNA_STUDIES_DIR = ROOT_DIR / "optuna_studies"

#: بند 7.9.2 «سیم‌چین نشتی» — $R^2$ خارج‌نمونه‌ی بالاتر از این یعنی توقف و ممیزی نشت،
#: نه جشن گرفتن (سقف واقع‌بینانه‌ی پروژه ۰.۴–۰.۵، بند ۵.۱۳).
LEAK_R2_CEILING = 0.90


# ---------------------------------------------------------------------------
# قرارداد خانواده‌ی GPU
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FamilyFitter:
    """یک مدل GPU: برازش، پیش‌بینی، ذخیره، بارگذاری — هر چهار صریح.

    ``save``/``load`` اختیاری نیستند: خواسته‌ی صریح اجرای GPU این است که مدل سنگین
    پس از پایان session قابل استفاده بماند (وگرنه ساعت‌ها محاسبه با بسته‌شدن تب
    از بین می‌رود).
    """

    model_id: str
    fit: Callable          # (train_df, tau, **hp) -> model
    predict: Callable      # (model, test_df, tau) -> np.ndarray[len(test)]
    save: Callable         # (model, stem: Path) -> list[Path]
    load: Callable         # (stem: Path) -> model
    defaults: dict = field(default_factory=dict)

    def as_fit_predict(self) -> Callable:
        """امضای یک‌مرحله‌ای بند ۱.۱ استاندارد اجرا — تا `conformal`/`calibration`/
        `s0_runner` بدون هیچ تغییری این خانواده را هم بپذیرند."""
        def fit_predict(train, test, tau, **hp):
            model = self.fit(train, tau, **{**self.defaults, **hp})
            return np.clip(np.asarray(self.predict(model, test, tau), dtype=float), 0.0, 1.0)
        fit_predict.__name__ = f"fit_predict_{self.model_id}"
        fit_predict.__doc__ = f"برازش+پیش‌بینی {self.model_id} (ساخته‌شده از FamilyFitter)."
        return fit_predict


def models_from_fitters(fitters: dict[str, FamilyFitter]) -> dict[str, Callable]:
    return {mid: f.as_fit_predict() for mid, f in fitters.items()}


# ---------------------------------------------------------------------------
# محیط اجرا
# ---------------------------------------------------------------------------

def use_gpu_tracking(dirname: str = DEFAULT_TRACKING_DIR) -> str:
    """MLflow را به پوشه‌ی جدا (`mlruns_gpu/`) می‌برد — بند 7.8.2 سلول ۶.

    ⚠️ عمداً `src.models.tracking.MLFLOW_TRACKING_URI` را مستقیم عوض می‌کند (نه از
    راه env var): `.mcp.json` خودش ``MLFLOW_TRACKING_URI=mlruns`` را در محیط ست
    می‌کند و env var را سایه می‌اندازد. همین الگو در `src/models/tests.py` هم برای
    جداکردن mlruns آزمایشی استفاده شده.
    """
    import mlflow

    from src.models import coverage, tracking

    uri = str(Path(dirname).resolve())
    tracking.MLFLOW_TRACKING_URI = uri
    coverage.MLFLOW_TRACKING_URI = uri
    mlflow.set_tracking_uri(uri)
    return uri


def device_report() -> dict:
    """خلاصه‌ی سخت‌افزار — بند 7.8.2 سلول ۵: «زمان‌ها فقط با دانستن سخت‌افزار قابل تفسیرند»."""
    info = {"platform": platform.platform(), "python": platform.python_version(),
            "torch": None, "cuda": None, "device": "cpu", "gpu_name": None, "gpu_memory_gb": None}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            info["device"] = "cuda"
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    except ImportError:
        pass
    try:
        import jax

        info["jax"] = jax.__version__
        info["jax_devices"] = [str(d) for d in jax.devices()]
    except ImportError:
        pass
    return info


def torch_device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_torch_determinism(strict: bool = False) -> None:
    """بند 7.8.2 سلول ۴ + بند 7.7.4.

    ``strict=True`` (``use_deterministic_algorithms``) روی بعضی op های cuDNN
    استثنا می‌اندازد و اجرا را می‌شکند؛ چون قاعده‌ی «سه seed» (A7، بند 7.16.3)
    از قبل جبران‌کننده‌ی قطعی‌نبودن GPU است، پیش‌فرض ``False`` است و نبود قطعیت
    کامل با اجرای چندseed **گزارش** می‌شود، نه پنهان.
    """
    import torch

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if strict:
        torch.use_deterministic_algorithms(True)


# ---------------------------------------------------------------------------
# بارگذاری داده + دروازه‌ی انصاف A1 (بند 7.7.3)
# ---------------------------------------------------------------------------

@dataclass
class LevelData:
    """داده‌ی یک سطح + foldهای رسمی + هر دو هش دروازه‌ی انصاف."""

    level: str
    df: pd.DataFrame
    folds: list                       # [(train_df, test_df), ...]
    cv_folds_hash: str
    data_snapshot_hash: str
    source: str

    def summary(self) -> str:
        sizes = " · ".join(f"fold{i}: {len(tr):,}→{len(te):,}" for i, (tr, te) in enumerate(self.folds))
        return f"{self.level} — {len(self.df):,} ردیف، {len(self.folds)} fold ({sizes})"

    def first_folds(self, n: int) -> "LevelData":
        """زیرمجموعه‌ی foldهای نخست — برای غربالگری ارزان (منطق R1، بند 7.3.2: ۳ fold
        نخست) وقتی هر برازش گران است. تنظیم روی ۳ fold، تأیید نهایی روی هر ۵."""
        from dataclasses import replace

        return replace(self, folds=self.folds[:n])


def _folds_from(df: pd.DataFrame) -> tuple[list, str]:
    fold_meta, cv_hash = load_cv_folds()
    folds = []
    for f in fold_meta:
        tr_mask, te_mask = f.masks(df[DATE_COL])
        folds.append((df.loc[tr_mask], df.loc[te_mask]))
    return folds, cv_hash


def load_l1(path: Path | str | None = None) -> LevelData:
    """سطح L1 (سلول $(d,m,r,f)$، ۷٬۵۷۹ ردیف) — همان فایلی که همه‌ی خانواده‌های CPU خوانده‌اند."""
    from src.features.build import FEATURES_A_PATH

    p = Path(path) if path is not None else FEATURES_A_PATH
    df = pd.read_parquet(p).sort_values(DATE_COL).reset_index(drop=True)
    folds, cv_hash = _folds_from(df)
    return LevelData("L1", df, folds, cv_hash, sha256_file(p), str(p))


def load_l5(path: Path | str = "data/processed/person_features_v1.parquet") -> LevelData:
    """سطح L5 (رزرو فردی، ۲٬۰۴۹٬۳۲۲ ردیف).

    ⚠️ foldها **دقیقاً همان مرزهای تاریخی L1**‌اند (بند 7.16.3: «در L5 تقسیم بر
    اساس تاریخ نه فرد») — یعنی یک فرد می‌تواند هم در train و هم در test باشد، که
    اشکالی ندارد و حتی مطلوب است: در استقرار واقعی هم همان افراد برمی‌گردند.
    """
    p = Path(path)
    df = pd.read_parquet(p).sort_values(DATE_COL).reset_index(drop=True)
    folds, cv_hash = _folds_from(df)
    return LevelData("L5", df, folds, cv_hash, sha256_file(p), str(p))


def assert_fairness_gate(data: LevelData, expected_cv_folds_hash: str,
                         expected_data_snapshot_hash: str | None = None) -> None:
    """دروازه‌ی انصاف A1 (بند 7.7.3) — نامنطبق بودن هش یعنی run از جدول مقایسه حذف می‌شود،
    پس نوت‌بوک باید همین‌جا بایستد، نه اینکه ساعت‌ها محاسبه‌ی غیرقابل‌استفاده تولید کند."""
    if data.cv_folds_hash != expected_cv_folds_hash:
        raise AssertionError(
            f"cv_folds_hash نامنطبق: {data.cv_folds_hash} != {expected_cv_folds_hash} — "
            "ادامه ندهید؛ این run از جدول مقایسه‌ی فاز ۷ حذف خواهد شد (بند 7.7.3)."
        )
    if expected_data_snapshot_hash and data.data_snapshot_hash != expected_data_snapshot_hash:
        raise AssertionError(
            f"data_snapshot_hash نامنطبق: {data.data_snapshot_hash} != {expected_data_snapshot_hash}"
        )
    print(f"✅ دروازه‌ی انصاف A1 پاس شد · cv_folds_hash={data.cv_folds_hash[:12]}… · "
          f"data_snapshot_hash={data.data_snapshot_hash[:12]}…")


def baseline_b3_per_fold(folds: list, tau: float = TUNING_TAU) -> dict:
    """مرجع B3 روی همان foldها — هم میانگین fold-محور (قابل‌قیاس با `S2_tuning_*.md`)
    و هم بردار زیان تک‌ردیفی (لازم برای DM، بند ۶.۶)."""
    per_fold, losses = [], []
    for tr, te in folds:
        pred = b3_empirical_quantile(tr, te, tau)
        per_fold.append(operational_metrics(te, pred, tau)["pinball"])
        losses.append(pinball_loss(te["rho"].to_numpy(), pred, tau))
    return {"fold_pinballs": per_fold, "mean_pinball": float(np.mean(per_fold)),
            "row_losses": np.concatenate(losses)}


# ---------------------------------------------------------------------------
# R0 — آزمایش دود (بند 7.3.2: «سقف ۶۰ ثانیه، فقط اثبات اجراپذیری»)
# ---------------------------------------------------------------------------

def smoke_test(fitter: FamilyFitter, data: LevelData, tau: float = TUNING_TAU,
               hyperparams: dict | None = None, fold_idx: int = 0) -> dict:
    """یک برازش با هایپرپارامتر پیش‌فرض روی یک fold + هر ۱۳ معیار عملیاتی + سیم‌چین نشتی.

    ⚠️ **چرا معیار کامل و نه فقط زمان اجرا.** نسخه‌ی اول هارنس S0 خ۱ فقط زمان را ثبت
    می‌کرد و دو باگ واقعی (علامت LP رگرسیون کوانتایل ترکیبی، واگرایی GLM) را پنهان
    کرد — یافته‌های ۰ و ۴ `doc/progress/07-*.md`. R0 دقیقاً برای پیدا کردن همین‌هاست.
    """
    train, test = data.folds[fold_idx]
    hp = {**fitter.defaults, **(hyperparams or {})}
    t0 = time.time()
    model = fitter.fit(train, tau, **hp)
    pred = np.asarray(fitter.predict(model, test, tau), dtype=float)
    seconds = time.time() - t0

    if pred.shape != (len(test),):
        raise ValueError(f"شکل خروجی {fitter.model_id}: {pred.shape} ≠ {(len(test),)}")
    if not np.all(np.isfinite(pred)):
        raise ValueError(f"{fitter.model_id}: خروجی شامل NaN/inf است")

    pred = np.clip(pred, 0.0, 1.0)
    m = operational_metrics(test, pred, tau)
    b3 = operational_metrics(test, b3_empirical_quantile(train, test, tau), tau)["pinball"]
    leak = bool(np.isfinite(m["R2_rho"]) and m["R2_rho"] > LEAK_R2_CEILING)

    print(f"R0 {fitter.model_id:<28s} pinball={m['pinball']:.5f} (B3={b3:.5f}) "
          f"پوشش={m['coverage']:.3f} R²={m['R2_rho']:+.3f} {seconds:.1f}s"
          + ("  🔴 R²>۰.۹ — توقف و ممیزی نشت (بند 7.9.2)" if leak else ""))
    if leak:
        raise AssertionError(f"{fitter.model_id}: R²={m['R2_rho']:.3f} > {LEAK_R2_CEILING} — "
                             "سیم‌چین نشتی بند 7.9.2 فعال شد؛ پیش از ادامه فیچرها را ممیزی کنید")
    return {"model_id": fitter.model_id, "seconds": seconds, "baseline_b3": b3, **m}


# ---------------------------------------------------------------------------
# R2 — تنظیم با بودجه‌ی **زمانی** (نه فقط تعداد trial)
# ---------------------------------------------------------------------------

@dataclass
class StudyResult:
    model_id: str
    family: str
    level: str
    tau: float
    n_trials_done: int
    n_fail: int
    best_pinball: float
    best_hyperparams: dict
    best_fold_pinballs: list
    history: list                # [(trial_idx, mean_pinball, seconds), ...]
    seconds: float
    budget_minutes: float
    budget_exhausted: bool
    converged: bool              # قاعده‌ی A6
    stable_top10pct_folds: int   # آزمون پایداری بند 7.6.3
    baseline_b3: float
    device: str


def _convergence_and_stability(history: list, per_fold: dict, n_folds: int) -> tuple[bool, int, int, float]:
    ok = [(i, v) for i, v, _ in history if np.isfinite(v)]
    if not ok:
        return False, 0, -1, float("nan")
    best_so_far, running = [], float("inf")
    for _, v, _ in history:
        running = min(running, v)
        best_so_far.append(running)
    tail_n = max(1, int(len(best_so_far) * 0.25))
    tail = best_so_far[-tail_n:]
    converged = ((tail[0] - tail[-1]) / tail[0] if tail[0] > 0 else 0.0) < 0.01

    best_idx, best_val = min(ok, key=lambda kv: kv[1])
    stable = 0
    for f in range(n_folds):
        vals = sorted(pf[f] for pf in per_fold.values() if f < len(pf))
        if not vals:
            continue
        cutoff = vals[max(0, int(len(vals) * 0.10) - 1)]
        if f < len(per_fold.get(best_idx, [])) and per_fold[best_idx][f] <= cutoff:
            stable += 1
    return converged, stable, best_idx, best_val


def run_gpu_study(fitter: FamilyFitter, space_fn: Callable, data: LevelData, *,
                  family: str, feature_set: str, budget_minutes: float,
                  max_trials: int = 200, tau: float = TUNING_TAU, seed: int = 42,
                  compute: str = "colab", target: str = "rho", scope: str = "global",
                  weighting: str = "none", architecture: str = "flat",
                  output_aggregation: str = "per_food",
                  log_every_trial_to_mlflow: bool = True) -> StudyResult:
    """Optuna TPE با بودجه‌ی **زمانی**؛ هر trial روی هر ۵ fold رسمی ارزیابی می‌شود.

    حلقه پیش از شروع هر trial زمان باقی‌مانده را می‌سنجد و اگر از بودجه گذشته باشد
    با ``budget_exhausted=True`` برمی‌گردد — همان قاعده‌ای که برای CPU در
    ``s2_runner.MODEL_TIME_CAP_SECONDS`` گذاشته شد (یافته‌ی ۱۲: یک مدل ۷.۹ ساعت
    گرفت چون هیچ سقف زمانی نداشت)، این‌جا با بودجه‌ی صریح کاربر.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_STUDIES_DIR.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{OPTUNA_STUDIES_DIR / f'{fitter.model_id}.db'}"
    study = optuna.create_study(study_name=f"{family}_{fitter.model_id}_S2", storage=storage,
                                direction="minimize", load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=seed))
    b3 = baseline_b3_per_fold(data.folds, tau)
    dev = device_report()["device"]
    history: list[tuple[int, float, float]] = [
        (t.number, t.value if t.value is not None else float("inf"), 0.0)
        for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]
    per_fold: dict[int, list] = {}
    n_fail = 0
    t_start = time.time()
    budget_s = budget_minutes * 60

    print(f"\n{'='*78}\nR2 — {family}/{fitter.model_id} ({data.level}) · τ={tau} · "
          f"بودجه={budget_minutes:g} دقیقه · دستگاه={dev} · مرجع B3={b3['mean_pinball']:.5f}"
          + (f" · {len(history)} trial از اجرای قبلی بازیابی شد" if history else "")
          + f"\n{'='*78}")

    while len(history) < max_trials:
        elapsed = time.time() - t_start
        if elapsed > budget_s:
            print(f"⏱️  بودجه‌ی زمانی ({budget_minutes:g} دقیقه) تمام شد پس از "
                  f"{len(history)} trial — با بهترین نتیجه‌ی تا این لحظه ادامه می‌دهیم.")
            break

        trial = study.ask()
        # ``seed`` صریح به fit پاس داده می‌شود، نه فقط از راه بذر سراسری: هر خانواده‌ی
        # این‌جا داخل خودش ``torch.manual_seed``/``PRNGKey`` صدا می‌زند و اگر مقدارش
        # ثابت بماند، بذر سراسری بی‌اثر می‌شود (و قاعده‌ی سه seed به یک نمایش تبدیل).
        hp = {**fitter.defaults, "seed": seed, **space_fn(trial)}
        t_trial = time.time()
        cfg = RunConfig(family=family, model_id=fitter.model_id, stage="S2", seed=seed,
                        level=data.level, feature_set=feature_set, tau=tau, target=target,
                        scope=scope, weighting=weighting, architecture=architecture,
                        output_aggregation=output_aggregation)
        try:
            ctx = start_model_run(cfg, data_snapshot_hash=data.data_snapshot_hash,
                                  cv_folds_hash=data.cv_folds_hash, compute=compute,
                                  train=data.folds[0][0] if log_every_trial_to_mlflow else None,
                                  dataset_source=data.source, source_fn=fitter.fit,
                                  n_trials=max_trials, sampler="TPE")
            with ctx:
                import mlflow

                mlflow.log_params({f"hp_{k}": v for k, v in hp.items()})
                mlflow.log_param("trial_idx", trial.number)
                mlflow.set_tag("device", dev)
                pinballs, fold_metrics = [], []
                for fi, (tr, te) in enumerate(data.folds):
                    model = fitter.fit(tr, tau, **hp)
                    pred = np.clip(np.asarray(fitter.predict(model, te, tau), dtype=float), 0.0, 1.0)
                    if pred.shape != (len(te),) or not np.all(np.isfinite(pred)):
                        raise ValueError(f"خروجی نامعتبر در fold{fi}")
                    m = operational_metrics(te, pred, tau)
                    pinballs.append(m["pinball"])
                    fold_metrics.append(m)
                    log_metrics_dict(m, step=fi)
                mean_pb = float(np.mean(pinballs))
                log_metrics_dict(aggregate_fold_metrics(fold_metrics), prefix="mean_")
                mlflow.log_metrics({"pinball_mean": mean_pb,
                                    "fit_seconds": time.time() - t_trial,
                                    "baseline_b3": b3["mean_pinball"]})
                mlflow.set_tag("outcome", "pass")
            study.tell(trial, mean_pb)
            history.append((trial.number, mean_pb, time.time() - t_trial))
            per_fold[trial.number] = pinballs
            best = min(v for _, v, _ in history if np.isfinite(v))
            mark = " 🎯" if mean_pb < b3["mean_pinball"] else ""
            print(f"  trial {trial.number:>3d} | pinball={mean_pb:.5f}{mark} | "
                  f"بهترین={best:.5f} | {time.time()-t_trial:5.1f}s | "
                  f"گذشته={(time.time()-t_start)/60:5.1f}/{budget_minutes:g} دقیقه")
        except Exception as e:  # noqa: BLE001 — شکست یک trial نباید کل مطالعه را بکشد
            n_fail += 1
            study.tell(trial, float("inf"))
            history.append((trial.number, float("inf"), time.time() - t_trial))
            print(f"  trial {trial.number:>3d} | ❌ شکست: {type(e).__name__}: {str(e)[:120]}")

    converged, stable, best_idx, best_val = _convergence_and_stability(history, per_fold, len(data.folds))
    best_params = {}
    if best_idx >= 0:
        best_params = next((t.params for t in study.trials if t.number == best_idx), {})

    result = StudyResult(
        model_id=fitter.model_id, family=family, level=data.level, tau=tau,
        n_trials_done=len(history), n_fail=n_fail, best_pinball=best_val,
        best_hyperparams=best_params, best_fold_pinballs=per_fold.get(best_idx, []),
        history=[(i, v, s) for i, v, s in history], seconds=time.time() - t_start,
        budget_minutes=budget_minutes, budget_exhausted=len(history) < max_trials,
        converged=converged, stable_top10pct_folds=stable, baseline_b3=b3["mean_pinball"],
        device=dev,
    )
    print(f"\n✅ {fitter.model_id}: بهترین pinball={best_val:.5f} در برابر B3={b3['mean_pinball']:.5f} "
          f"({'برد' if best_val < b3['mean_pinball'] else 'باخت'}) · {len(history)} trial · "
          f"همگرا(A6)={'✅' if converged else '⚠️'} · پایداری={stable}/{len(data.folds)}")
    save_study_result(result)
    return result


# ---------------------------------------------------------------------------
# قهرمان‌سازی: بازبرازش با بهترین هایپرپارامتر + ذخیره‌ی مدل + کالیبراسیون + DM
# ---------------------------------------------------------------------------

@dataclass
class ChampionResult:
    model_id: str
    family: str
    level: str
    tau: float
    hyperparams: dict
    seeds: list
    per_seed_pinball: dict
    pinball_row_mean: float          # میانگین زیان هر ردیف (قابل‌قیاس با dm_test_*.md)
    pinball_fold_mean: float         # میانگین fold-محور (قابل‌قیاس با S2_tuning_*.md)
    metrics: dict                    # میانگین ۱۳ معیار عملیاتی روی foldها
    coverage: float
    coverage_gap: float
    aci_coverage: float
    aci_coverage_gap: float
    aci_pinball: float
    dm_vs_b3: dict
    baseline_b3_row: float
    baseline_b3_fold: float
    artifacts: list
    fit_seconds: float


def _oof_two_stage(fitter: FamilyFitter, folds: list, tau: float, hp: dict,
                   artifact_dir: Path | None, seed_tag: str, seed: int) -> tuple[pd.DataFrame, list]:
    """یک برازش به‌ازای هر fold؛ همان مدل هم برای پیش‌بینی test و هم برای ذخیره استفاده می‌شود."""
    parts, artifacts = [], []
    hp = {**hp, "seed": seed}
    for fi, (tr, te) in enumerate(folds):
        model = fitter.fit(tr, tau, **hp)
        pred = np.clip(np.asarray(fitter.predict(model, te, tau), dtype=float), 0.0, 1.0)
        if artifact_dir is not None:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            stem = artifact_dir / f"{fitter.model_id}__{seed_tag}__fold{fi}"
            artifacts += [str(Path(p).relative_to(ROOT_DIR)) if Path(p).is_absolute() and
                          str(p).startswith(str(ROOT_DIR)) else str(p)
                          for p in fitter.save(model, stem)]
        parts.append(pd.DataFrame({
            "fold": fi, "actual": te["rho"].to_numpy(), "pred_q": pred,
            "RestaurantName": te["RestaurantName"].to_numpy() if "RestaurantName" in te else "-",
            "Meal": te["Meal"].to_numpy(), "Res": te["Res"].to_numpy(),
            "is_tehran": te["is_tehran"].to_numpy() if "is_tehran" in te else False,
            DATE_COL: te[DATE_COL].to_numpy(),
        }))
    return pd.concat(parts, ignore_index=True), artifacts


def finalize_champion(fitter: FamilyFitter, data: LevelData, study: StudyResult, *,
                      feature_set: str, seeds: Iterable[int] = (42, 1234, 2026),
                      compute: str = "colab", run_aci: bool = True,
                      save_artifacts: bool = True, target: str = "rho",
                      output_aggregation: str = "per_food") -> ChampionResult:
    """بازبرازش بهترین پیکربندی روی هر ۵ fold و هر seed + ذخیره‌ی مدل + ACI + DM.

    **سه seed اجباری است، نه تزئینی** (بند 7.16.3 قاعده‌ی A7): قطعیت روی GPU
    تضمین‌شدنی نیست، پس یک عدد pinball از یک seed «نتیجه» نیست. اختلاف بین seedها
    خودش یکی از خروجی‌های گزارش است.

    ACI (بند 7.22.1 عضو ۴) این‌جا با **یک** برازش به‌ازای هر fold انجام می‌شود
    (`conformal.aci_from_predictions`)، نه دو تا — چون در سطح L5 هر برازش می‌تواند
    دقایق طول بکشد و نسخه‌ی refit-محور بودجه را دو برابر می‌کرد.
    """
    from src.models import conformal

    tau, hp = study.tau, study.best_hyperparams
    art_dir = GPU_MODELS_DIR / study.family / fitter.model_id if save_artifacts else None
    b3 = baseline_b3_per_fold(data.folds, tau)

    t0 = time.time()
    per_seed, oof_by_seed, artifacts = {}, {}, []
    for s in seeds:
        from src.config import set_global_seed

        set_global_seed(s)
        oof, arts = _oof_two_stage(fitter, data.folds, tau, hp, art_dir, f"s{s}", seed=s)
        oof_by_seed[s] = oof
        artifacts += arts
        per_seed[s] = float(pinball_loss(oof["actual"].to_numpy(), oof["pred_q"].to_numpy(), tau).mean())
        print(f"  seed {s}: pinball(ردیفی)={per_seed[s]:.5f}")

    best_seed = min(per_seed, key=per_seed.get)
    oof = oof_by_seed[best_seed]
    row_losses = pinball_loss(oof["actual"].to_numpy(), oof["pred_q"].to_numpy(), tau)
    fold_metrics = [operational_metrics(
        data.folds[fi][1], oof.loc[oof["fold"] == fi, "pred_q"].to_numpy(), tau)
        for fi in range(len(data.folds))]
    metrics = aggregate_fold_metrics(fold_metrics)
    coverage = float((oof["actual"] <= oof["pred_q"]).mean())

    aci_cov, aci_gap, aci_pb = float("nan"), float("nan"), float("nan")
    if run_aci:
        aci_parts = []
        for fi, (tr, te) in enumerate(data.folds):
            proper, calib = conformal._time_split(tr, DATE_COL)
            if len(calib) < 20 or len(proper) < 20:
                continue
            model = fitter.fit(proper, tau, **{**hp, "seed": best_seed})
            pred_calib = np.clip(np.asarray(fitter.predict(model, calib, tau), dtype=float), 0, 1)
            pred_test = np.clip(np.asarray(fitter.predict(model, te, tau), dtype=float), 0, 1)
            adj, _ = conformal.aci_from_predictions(
                pred_calib, calib["rho"].to_numpy(), pred_test, te["rho"].to_numpy(),
                te[DATE_COL].to_numpy(), tau)
            aci_parts.append(pd.DataFrame({"actual": te["rho"].to_numpy(), "pred_q": adj}))
        if aci_parts:
            aci = pd.concat(aci_parts, ignore_index=True)
            aci_cov = float((aci["actual"] <= aci["pred_q"]).mean())
            aci_gap = aci_cov - tau
            aci_pb = float(pinball_loss(aci["actual"].to_numpy(), aci["pred_q"].to_numpy(), tau).mean())
            print(f"  ACI: پوشش={aci_cov:.4f} (شکاف {aci_gap:+.4f}) · pinball={aci_pb:.5f}")

    dm, p = diebold_mariano(row_losses, b3["row_losses"])
    dm_result = {"dm_stat": dm, "p_value": p,
                 "delta_pinball": float(row_losses.mean() - b3["row_losses"].mean()),
                 "significant_at_0.05": bool(np.isfinite(p) and p < 0.05 and
                                             row_losses.mean() < b3["row_losses"].mean())}

    result = ChampionResult(
        model_id=fitter.model_id, family=study.family, level=data.level, tau=tau,
        hyperparams=hp, seeds=list(per_seed), per_seed_pinball=per_seed,
        pinball_row_mean=float(row_losses.mean()),
        pinball_fold_mean=float(np.mean([m["pinball"] for m in fold_metrics])),
        metrics=metrics, coverage=coverage, coverage_gap=coverage - tau,
        aci_coverage=aci_cov, aci_coverage_gap=aci_gap, aci_pinball=aci_pb,
        dm_vs_b3=dm_result, baseline_b3_row=float(b3["row_losses"].mean()),
        baseline_b3_fold=b3["mean_pinball"], artifacts=sorted(set(artifacts)),
        fit_seconds=time.time() - t0,
    )
    print(f"\n🏁 {fitter.model_id}: pinball(ردیفی)={result.pinball_row_mean:.5f} در برابر "
          f"B3={result.baseline_b3_row:.5f} · DM p={p:.4f} "
          f"{'✅ معنادار' if dm_result['significant_at_0.05'] else '❌ غیرمعنادار'} · "
          f"{len(result.artifacts)} فایل مدل ذخیره شد")
    save_champion_result(result)
    _log_champion_to_mlflow(result, data, feature_set, compute, target, output_aggregation)
    return result


def _log_champion_to_mlflow(result: ChampionResult, data: LevelData, feature_set: str,
                            compute: str, target: str, output_aggregation: str = "per_food") -> None:
    """یک run جداگانه با ``stage=S3`` برای خودِ قهرمان — با artifact مدل ضمیمه (بند 7.29.1)."""
    import mlflow

    cfg = RunConfig(family=result.family, model_id=result.model_id, stage="S3", seed=42,
                    level=result.level, feature_set=feature_set, tau=result.tau, target=target,
                    output_aggregation=output_aggregation)
    with start_model_run(cfg, data_snapshot_hash=data.data_snapshot_hash,
                         cv_folds_hash=data.cv_folds_hash, compute=compute,
                         train=data.folds[0][0], dataset_source=data.source):
        mlflow.log_params({f"hp_{k}": v for k, v in result.hyperparams.items()})
        log_metrics_dict(result.metrics, prefix="mean_")
        mlflow.log_metrics({
            "pinball_row_mean": result.pinball_row_mean,
            "pinball_fold_mean": result.pinball_fold_mean,
            "baseline_b3_row": result.baseline_b3_row,
            "coverage": result.coverage, "coverage_gap": result.coverage_gap,
            "dm_p_value": result.dm_vs_b3["p_value"] if np.isfinite(result.dm_vs_b3["p_value"]) else -1.0,
            **({"aci_coverage": result.aci_coverage, "aci_pinball": result.aci_pinball}
               if np.isfinite(result.aci_coverage) else {}),
        })
        mlflow.set_tag("champion", "true")
        art_dir = GPU_MODELS_DIR / result.family / result.model_id
        if art_dir.exists():
            mlflow.log_artifacts(str(art_dir), artifact_path="model_files")


# ---------------------------------------------------------------------------
# گزارش — همه‌چیز روی دیسک، چون session کولب از بین می‌رود
# ---------------------------------------------------------------------------

def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n")


def save_study_result(result: StudyResult) -> Path:
    p = GPU_REPORTS_DIR / f"S2_tuning_{result.family}_{result.model_id}.json"
    _write_json(p, asdict(result))
    return p


def save_champion_result(result: ChampionResult) -> Path:
    p = GPU_REPORTS_DIR / f"champion_{result.family}_{result.model_id}.json"
    _write_json(p, asdict(result))
    return p


def render_family_report(family: str, title: str, studies: list[StudyResult],
                         champions: list[ChampionResult], smoke: list[dict],
                         device: dict, notes: Iterable[str] = ()) -> str:
    """گزارش کامل فارسی همان‌جا داخل نوت‌بوک — بند 7.8.4: نتیجه باید بدون دسترسی به
    session کولب هم قابل خواندن باشد."""
    lines = [
        f"# {title}",
        "",
        f"> اجرای GPU، بند 7.8 `doc/WBS-phase7-modeling.md`. خانواده {family}. "
        f"سخت‌افزار: {device.get('gpu_name') or device.get('device')} · "
        f"torch={device.get('torch')} · CUDA={device.get('cuda')}",
        "",
        "## R0 — آزمایش دود (اجراپذیری + سیم‌چین نشتی)",
        "",
        "| مدل | pinball | B3 | پوشش | R² | زمان |",
        "|---|---|---|---|---|---|",
    ]
    for s in smoke:
        lines.append(f"| `{s['model_id']}` | {s['pinball']:.5f} | {s['baseline_b3']:.5f} | "
                     f"{s['coverage']:.3f} | {s['R2_rho']:+.3f} | {s['seconds']:.1f}s |")

    lines += ["", "## R2 — تنظیم با بودجه‌ی زمانی", "",
              "| مدل | بهترین pinball | B3 | trial | همگرا (A6) | پایداری (۷.۶.۳) | شکست | ساعت-هسته |",
              "|---|---|---|---|---|---|---|---|"]
    for st in sorted(studies, key=lambda s: s.best_pinball):
        mark = " 🎯" if st.best_pinball < st.baseline_b3 else ""
        lines.append(f"| `{st.model_id}`{mark} | **{st.best_pinball:.5f}** | {st.baseline_b3:.5f} | "
                     f"{st.n_trials_done} | {'✅' if st.converged else '⚠️'} | "
                     f"{st.stable_top10pct_folds}/5 | {st.n_fail} | {st.seconds/3600:.2f}h |")

    if champions:
        lines += ["", "## S3 — قهرمان‌ها: سه seed، کالیبراسیون ACI، آزمون Diebold-Mariano", "",
                  "| مدل | pinball(ردیفی) | B3(ردیفی) | Δ | DM p | معنادار؟ | پوشش | پوشش پس از ACI |",
                  "|---|---|---|---|---|---|---|---|"]
        for c in champions:
            sig = "✅ بله" if c.dm_vs_b3["significant_at_0.05"] else "❌ خیر"
            aci = f"{c.aci_coverage:.4f} ({c.aci_coverage_gap:+.4f})" if np.isfinite(c.aci_coverage) else "—"
            lines.append(f"| `{c.model_id}` | {c.pinball_row_mean:.5f} | {c.baseline_b3_row:.5f} | "
                         f"{c.dm_vs_b3['delta_pinball']:+.5f} | {c.dm_vs_b3['p_value']:.4f} | {sig} | "
                         f"{c.coverage:.4f} ({c.coverage_gap:+.4f}) | {aci} |")
        lines += ["", "### پراکندگی بین seedها (قاعده‌ی A7)", "",
                  "| مدل | " + " | ".join(f"seed {s}" for s in champions[0].seeds) + " | دامنه |",
                  "|---|" + "---|" * (len(champions[0].seeds) + 1)]
        for c in champions:
            vals = [c.per_seed_pinball[s] for s in c.seeds]
            lines.append(f"| `{c.model_id}` | " + " | ".join(f"{v:.5f}" for v in vals) +
                         f" | {max(vals)-min(vals):.5f} |")
        lines += ["", "### فایل‌های مدل ذخیره‌شده", ""]
        for c in champions:
            lines.append(f"- `{c.model_id}`: {len(c.artifacts)} فایل زیر "
                         f"`models/gpu/{c.family}/{c.model_id}/`")

    if notes:
        lines += ["", "## یادداشت‌ها و تفسیر", ""] + [f"- {n}" for n in notes]
    return "\n".join(lines) + "\n"


def save_family_report(family: str, text: str, slug: str) -> Path:
    p = GPU_REPORTS_DIR / f"{slug}.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    print(f"گزارش نوشته شد: {p}")
    return p


# ---------------------------------------------------------------------------
# بسته‌بندی خروجی — بند 7.8.2 سلول ۸، با تکه‌های ۱۰۰ مگابایتی
# ---------------------------------------------------------------------------

DEFAULT_PACKAGE_INCLUDE = ("mlruns_gpu", "models/gpu", "reports/gpu", "optuna_studies")


def _sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def package_outputs(tag: str, include: Iterable[str] = DEFAULT_PACKAGE_INCLUDE,
                    part_mb: int = 100, out_dir: Path | str = "gpu_outputs",
                    root: Path | str = ".") -> dict:
    """همه‌ی خروجی‌های اجرا را در یک zip می‌گذارد و به تکه‌های ``part_mb`` مگابایتی می‌شکند.

    چرا تکه‌تکه: مرورگر روی دانلود تک‌فایل بزرگ از کولب ناپایدار است و اگر وسط کار
    قطع شود، کل فایل از دست می‌رود. تکه‌ی ۱۰۰ مگابایتی قابل‌بازیابی است — هر تکه
    هش SHA-256 خودش را دارد و ``MANIFEST.json`` می‌گوید کدام تکه کجاست.

    بازیابی (محلی)::

        cat gpu_outputs_<tag>.zip.part* > gpu_outputs_<tag>.zip
        unzip gpu_outputs_<tag>.zip
    """
    root = Path(root).resolve()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    zip_path = out / f"gpu_outputs_{tag}.zip"

    n_files, total_bytes = 0, 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for rel in include:
            base = root / rel
            if not base.exists():
                print(f"  … {rel} وجود ندارد، رد شد")
                continue
            for p in sorted(base.rglob("*")):
                if p.is_file():
                    zf.write(p, arcname=str(p.relative_to(root)))
                    n_files += 1
                    total_bytes += p.stat().st_size
                elif p.is_dir() and not any(c.is_file() for c in p.rglob("*")):
                    # ⚠️ پوشه‌ی خالی (مثلاً artifacts/ یک trial بدون هیچ artifact
                    # ثبت‌شده — اکثر runهای R2) بدون entry صریح در zip گم می‌شود.
                    # کشف‌شده هنگام ادغام GPU02/03: بدون این پوشه، MLflow FileStore
                    # (``_is_valid_run_directory``: نیازمند params/metrics/artifacts
                    # هر سه) آن run را «نامعتبر» می‌بیند و بی‌صدا از get_run/search_runs
                    # حذف می‌کند — نه خطا، فقط غیبت از نتیجه؛ صدها run این‌طور از دست
                    # رفته بودند تا کشف شد.
                    zf.writestr(str(p.relative_to(root)) + "/", "")
        zf.writestr("RESTORE.md", _restore_instructions(tag))

    part_bytes = part_mb * 1024 * 1024
    parts = []
    with zip_path.open("rb") as src:
        idx = 1
        while True:
            chunk = src.read(part_bytes)
            if not chunk:
                break
            part = out / f"{zip_path.name}.part{idx:03d}"
            part.write_bytes(chunk)
            parts.append(part)
            idx += 1
    if len(parts) > 1:
        zip_path.unlink()  # فقط تکه‌ها می‌مانند — دانلود دوباره‌ی کل فایل بی‌فایده است
    else:
        parts = [zip_path]

    manifest = {
        "tag": tag, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_files": n_files, "raw_bytes": total_bytes,
        "parts": [{"name": p.name, "bytes": p.stat().st_size, "sha256": _sha256_of(p)} for p in parts],
        "restore": (f"cat {zip_path.name}.part* > {zip_path.name} && unzip {zip_path.name}"
                    if len(parts) > 1 else f"unzip {zip_path.name}"),
        "included": list(include),
    }
    _write_json(out / f"MANIFEST_{tag}.json", manifest)
    (out / "RESTORE.md").write_text(_restore_instructions(tag))

    print(f"\n📦 بسته‌بندی شد: {n_files:,} فایل ({total_bytes/1e6:.1f} MB خام) → "
          f"{len(parts)} تکه در {out}/")
    for p in parts:
        print(f"   {p.name}  {p.stat().st_size/1e6:.1f} MB")
    print(f"\nبازیابی محلی:\n   {manifest['restore']}")
    return manifest


def _restore_instructions(tag: str) -> str:
    return f"""# بازیابی خروجی اجرای GPU — {tag}

بند 7.8.3 سند فاز ۷ (چرخه‌ی رفت‌وبرگشت).

```bash
# ۱) تکه‌ها را کنار هم بگذارید و باز کنید (در ریشه‌ی مخزن)
cat gpu_outputs_{tag}.zip.part* > gpu_outputs_{tag}.zip
sha256sum -c <(python - <<'EOF'
import json
m = json.load(open("MANIFEST_{tag}.json"))
for p in m["parts"]:
    print(f"{{p['sha256']}}  {{p['name']}}")
EOF
)
unzip -o gpu_outputs_{tag}.zip

# ۲) ادغام MLflow در mlruns محلی (بند 7.8.3 گام ۵)
rsync -a mlruns_gpu/ mlruns/

# ۳) تأیید
make mlflow-ui     # runهای جدید با tag compute=colab باید دیده شوند
```

محتوای بسته:

| مسیر | چیست |
|---|---|
| `mlruns_gpu/` | همه‌ی MLflow runها (هر trial + قهرمان‌ها با artifact مدل) |
| `models/gpu/<family>/<model>/` | وزن مدل + پیش‌پردازش هر fold/seed — با `load()` همان خانواده قابل بازاستفاده |
| `reports/gpu/*.json` `*.md` | نتیجه‌ی R0/R2/قهرمان + گزارش فارسی کامل |
| `optuna_studies/*.db` | مطالعه‌ی Optuna — اجرای بعدی از همین‌جا ادامه می‌دهد (resume) |
"""


def download_parts(out_dir: Path | str = "gpu_outputs") -> None:
    """دانلود خودکار تکه‌ها روی کولب (روی محیط غیرکولب بی‌صدا رد می‌شود)."""
    try:
        from google.colab import files  # type: ignore
    except ImportError:
        print("خارج از کولب — دانلود خودکار رد شد؛ فایل‌ها روی دیسک هستند.")
        return
    for p in sorted(Path(out_dir).iterdir()):
        if p.is_file():
            files.download(str(p))
