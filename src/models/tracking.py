"""بند 7.7.1/7.7.2 سند فاز ۷ — باز کردن هر run با نام و فیلدهای اجباری، بدون فراموشی.

هر اسکریپت آموزش مدل فاز ۷ باید runاش را با ``start_model_run`` باز کند، نه
``mlflow.start_run`` خام. ورودی‌اش یک ``RunConfig`` است (نه چند آرگومان جدا) — یعنی
اعلام‌کردن موضع اجرا روی **هر نُه محور آزمایش** بند 7.9.1 از نظر فنی اجباری است، نه
یک توصیه‌ی نانوشته که فراموش شود.

چهار چیز همیشه در **جای اختصاصی خودشان** در MLflow ثبت می‌شوند، نه فقط به‌عنوان بخشی از
نام run:

1. **دیتاست** — با ``mlflow.log_input`` در تب Dataset هر run (نه فقط هش در param).
   نام دیتاست خودش می‌گوید «تجمیعی یا فردی، با کدام فیچرست» تا از تب Datasets هر run
   بدون باز کردن param جداگانه معلوم باشد.
2. **نوع مدل** — tag اختصاصی ``model_type`` (کلاس/کتابخانه‌ی واقعی، از
   ``registry.MODELS[model_id].algorithm``)، جدا از ``model_id`` که فقط اسلاگ داخلی است.
3. **مرجع کد** — tag اختصاصی ``code_ref`` (``مسیر/فایل.py:خط#نام_تابع``)، خودکار از
   ``inspect`` روی خودِ تابع fit_predict استخراج می‌شود — نه نوشته‌شده‌ی دستی که ممکن است
   با جابه‌جایی کد قدیمی بماند. کنار ``git_commit`` (که از قبل tag می‌شود)، یعنی هر run
   دقیقاً می‌گوید «کدام کامیت، کدام فایل، کدام خط» — بازتولیدش فقط یک ``git checkout`` است.
4. **خودِ مدل fitشده** (اختیاری، برای مدل‌های سنگین) — با ``log_model_fn`` که یک تابع
   ``(run) -> None`` است و مدل را با فلیور مناسب (``mlflow.sklearn``/``mlflow.pytorch``/…)
   در Model Registry ثبت می‌کند. برای مدل‌های سبک/برازش سریع (خ۱ خطی) لازم نیست — طبق
   بند 7.29.1 artifact کامل مدل برای قهرمانان S3 است، نه هر trial سبک S0/S1.

مثال استفاده::

    from src.cv import load_cv_folds
    from src.models.axes import RunConfig
    from src.models.tracking import start_model_run

    _, cv_hash = load_cv_folds()
    cfg = RunConfig(family="F02", model_id="lightgbm", stage="S2", seed=42,
                    level="L1", feature_set="FS_lgbm", tau=0.10,
                    scope="per_cluster", weighting="res")   # ← محورها صریح‌اند
    with start_model_run(cfg, data_snapshot_hash=SNAPSHOT_HASH, cv_folds_hash=cv_hash,
                         train=train_df, test=test_df, dataset_source=str(FEATURES_A_PATH),
                         source_fn=fit_predict_lightgbm,     # ← code_ref خودکار از این می‌آید
                         n_trials=120, sampler="TPE"):
        mlflow.log_params(hyperparams)
        mlflow.log_metric("pinball", value)
"""

import inspect
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator

import mlflow
import mlflow.data
import pandas as pd

from src.config import MLFLOW_TRACKING_URI, ROOT_DIR
from src.models.axes import LEVEL_DATASET_VARIANT, RunConfig
from src.models.naming import run_name
from src.models.registry import MODELS as MODEL_REGISTRY

DEFAULT_EXPERIMENT = "phase7"


def git_commit_short() -> str:
    """کوتاه‌ترین SHA کامیت فعلی — برای tag ``git_commit`` بند 7.7.2. اگر خارج از یک
    مخزن گیت اجرا شد (مثلاً روی کولب پیش از ``git clone``)، ``'unknown'`` برمی‌گرداند."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def code_reference(fn: Callable | None) -> str:
    """``مسیر/نسبی/فایل.py:خط#نام_تابع`` برای تابع fit_predict — بازتولید یعنی باز کردن
    دقیقاً همین آدرس روی همان ``git_commit``. با ``inspect`` مشتق می‌شود تا هرگز دستی
    نوشته و فراموش نشود (بند 7.7.2، `doc/phase7-execution-standard.md` بند ۱.۴)."""
    if fn is None:
        return "unknown"
    try:
        path = Path(inspect.getsourcefile(fn) or inspect.getfile(fn)).resolve()
        rel = path.relative_to(ROOT_DIR)
        _, lineno = inspect.getsourcelines(fn)
        return f"{rel}:{lineno}#{fn.__name__}"
    except (TypeError, OSError, ValueError):
        return "unknown"


def _log_dataset_input(df: pd.DataFrame, cfg: RunConfig, context: str,
                       source: str | None, snapshot_hash: str) -> None:
    """ثبت یک زیرمجموعه (train/test) در تب Dataset اختصاصی MLflow — بند 7.7.2.

    نام دیتاست خوانا و خودگویاست: ``{تجمیعی|فردی}-{feature_set}-{context}`` — تا از تب
    Datasets هر run بی‌درنگ معلوم شود روی کدام دیتاست و با کدام فیچرست train شده، بدون
    نیاز به باز کردن paramها.
    """
    variant = LEVEL_DATASET_VARIANT.get(cfg.level, cfg.level)
    name = f"{variant}-{cfg.feature_set}-{context}"
    ds = mlflow.data.from_pandas(df, source=source or "unknown", name=name, digest=snapshot_hash[:8])
    mlflow.log_input(ds, context=context, tags={"level": cfg.level, "feature_set": cfg.feature_set})


@contextmanager
def start_model_run(cfg: RunConfig, *, data_snapshot_hash: str, cv_folds_hash: str,
                    compute: str = "local", n_trials: int | None = None,
                    sampler: str | None = None,
                    train: pd.DataFrame | None = None, test: pd.DataFrame | None = None,
                    dataset_source: str | None = None, source_fn: Callable | None = None,
                    log_model_fn: Callable[[mlflow.ActiveRun], None] | None = None,
                    experiment: str = DEFAULT_EXPERIMENT) -> Iterator[mlflow.ActiveRun]:
    """باز کردن یک MLflow run با نام (بند 7.7.1) و tag/paramهای اجباری (بند 7.7.2).

    فراخوان مسئول ثبت ``hyperparams`` کامل و metricهای مدل خودش است (چون این‌ها
    مدل‌به‌مدل فرق می‌کنند)؛ این تابع فقط اسکلت مشترکی را که هرگز نباید فراموش شود تضمین
    می‌کند: نام درست، هر نُه محور آزمایش، ``cv_folds_hash``، ``data_snapshot_hash``، تب
    Dataset (اگر ``train``/``test`` داده شود)، و tagهای اختصاصی ``model_type``/``code_ref``.

    ``source_fn`` تابع fit_predict همان مدل است — اگر داده شود، ``code_ref`` خودکار از
    روی آن ساخته می‌شود (بند ۱.۴ استاندارد اجرا). ندادنش خطا نیست ولی یعنی run
    بازتولیدپذیریِ «کدام فایل، کدام خط» را ندارد — فقط برای کدهای آزمایشیِ خارج از
    ``src/models/families/`` قابل‌قبول است.

    ``log_model_fn`` قلاب اختیاری برای مدل‌های سنگین است — بعد از پایان بدنه‌ی ``with``
    صدا زده می‌شود تا فراخوان مدل fitشده را با فلیور مناسب ثبت کند؛ برای خانواده‌های سبک
    (خ۱ خطی) لازم نیست.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment)
    # بند 7.7.1 نمونه‌هایش کد فشرده‌ی بدون آندرلاین می‌خواهد ("FSlgbm")، ولی رجیستری فیچرست
    # واقعی پروژه (feature_sets_v1.json) از "FS_day" استفاده می‌کند — این‌جا فقط برای نام run
    # فشرده می‌شود؛ مقدار کامل بدون تغییر در param `feature_set` ثبت می‌ماند.
    name = run_name(family=cfg.family, model=cfg.model_id, level=cfg.level, target=cfg.target,
                    feature_set=cfg.feature_set.replace("_", ""), tau=cfg.tau,
                    stage=cfg.stage, seed=cfg.seed)

    spec = MODEL_REGISTRY.get(cfg.model_id)
    model_type = spec.algorithm if spec is not None else "unregistered"

    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tags({
            "git_commit": git_commit_short(),
            "stage": cfg.stage,
            "compute": compute,
            "family": cfg.family,
            "model_type": model_type,  # ⭐ جای اختصاصی نوع الگوریتم — جدا از model_id/نام run
            "code_ref": code_reference(source_fn),  # ⭐ فایل:خط#تابع — بازتولید = git checkout + باز کردن همین آدرس
        })
        params = {
            **cfg.to_mlflow_params(),          # هر نُه محور، هرکدام یک param جدا
            "data_snapshot_hash": data_snapshot_hash,
            "cv_folds_hash": cv_folds_hash,
        }
        if n_trials is not None:
            params["n_trials"] = n_trials
        if sampler is not None:
            params["sampler"] = sampler
        mlflow.log_params(params)

        if train is not None:
            _log_dataset_input(train, cfg, "train", dataset_source, data_snapshot_hash)
        if test is not None:
            _log_dataset_input(test, cfg, "test", dataset_source, data_snapshot_hash)

        yield run

        if log_model_fn is not None:
            log_model_fn(run)
