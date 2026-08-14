"""بند 7.7.1/7.7.2 سند فاز ۷ — باز کردن هر run با نام و فیلدهای اجباری، بدون فراموشی.

هر اسکریپت آموزش مدل فاز ۷ باید runاش را با ``start_model_run`` باز کند، نه
``mlflow.start_run`` خام. ورودی‌اش یک ``RunConfig`` است (نه چند آرگومان جدا) — یعنی
اعلام‌کردن موضع اجرا روی **هر نُه محور آزمایش** بند 7.9.1 از نظر فنی اجباری است، نه
یک توصیه‌ی نانوشته که فراموش شود.

مثال استفاده::

    from src.cv import load_cv_folds
    from src.models.axes import RunConfig
    from src.models.tracking import start_model_run

    _, cv_hash = load_cv_folds()
    cfg = RunConfig(family="F02", model_id="lightgbm", stage="S2", seed=42,
                    level="L1", feature_set="FS_lgbm", tau=0.10,
                    scope="per_cluster", weighting="res")   # ← محورها صریح‌اند
    with start_model_run(cfg, data_snapshot_hash=SNAPSHOT_HASH, cv_folds_hash=cv_hash,
                         n_trials=120, sampler="TPE"):
        mlflow.log_params(hyperparams)
        mlflow.log_metric("pinball", value)
"""

import subprocess
from contextlib import contextmanager
from typing import Iterator

import mlflow

from src.config import MLFLOW_TRACKING_URI
from src.models.axes import RunConfig
from src.models.naming import run_name

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


@contextmanager
def start_model_run(cfg: RunConfig, *, data_snapshot_hash: str, cv_folds_hash: str,
                    compute: str = "local", n_trials: int | None = None,
                    sampler: str | None = None,
                    experiment: str = DEFAULT_EXPERIMENT) -> Iterator[mlflow.ActiveRun]:
    """باز کردن یک MLflow run با نام (بند 7.7.1) و tag/paramهای اجباری (بند 7.7.2).

    فراخوان مسئول ثبت ``hyperparams`` کامل و metricهای مدل خودش است (چون این‌ها
    مدل‌به‌مدل فرق می‌کنند)؛ این تابع فقط اسکلت مشترکی را که هرگز نباید فراموش شود تضمین
    می‌کند: نام درست، هر نُه محور آزمایش، ``cv_folds_hash``، و ``data_snapshot_hash``.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment)
    # بند 7.7.1 نمونه‌هایش کد فشرده‌ی بدون آندرلاین می‌خواهد ("FSlgbm")، ولی رجیستری فیچرست
    # واقعی پروژه (feature_sets_v1.json) از "FS_day" استفاده می‌کند — این‌جا فقط برای نام run
    # فشرده می‌شود؛ مقدار کامل بدون تغییر در param `feature_set` ثبت می‌ماند.
    name = run_name(family=cfg.family, model=cfg.model_id, level=cfg.level, target=cfg.target,
                    feature_set=cfg.feature_set.replace("_", ""), tau=cfg.tau,
                    stage=cfg.stage, seed=cfg.seed)

    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tags({
            "git_commit": git_commit_short(),
            "stage": cfg.stage,
            "compute": compute,
            "family": cfg.family,
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
        yield run
