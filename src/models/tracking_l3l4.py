"""ثبت MLflow برای خ۳/خ۴ (سطح L3/L4/آشتی‌شده‌ی L1) — بند 7.7 WBS.

⚠️ **چرا این ماژول جدا از `s0_runner.py` و بقیه است.** آن‌ها فرض می‌کنند ``test`` ستون‌های
Res/Recv/rho ردیفی دارد و از ``src.baselines.operational_metrics`` عبور می‌کنند. خ۳
(هدف ``day_shock``، نه نرخ) و خ۴ (پنل، نه ردیف) این قرارداد را ندارند — پس متریک‌ها
دستی (نه از ``operational_metrics``) به ``start_model_run`` مشترک پاس داده می‌شوند.

هر (مدل × وعده) یا (مدل) **یک** MLflow run است با متریک‌های تجمیع‌شده روی هر ۵ fold
رسمی — نه یک run به‌ازای هر fold (هم‌تراز با سطح تفصیل S0/S1 در بقیه‌ی پروژه).
"""

import mlflow

from src.cv import load_cv_folds
from src.models.axes import RunConfig
from src.models.tracking import start_model_run

#: بند 7.7.2 — هش snapshot سطح L1 (`features_A_v1.parquet`) که خ۳/خ۴ از آن مشتق شده‌اند
#: (خودشان DVC-tracked جدا نیستند؛ اشتقاق کد قطعی است، نه یک snapshot تازه)
DATA_SNAPSHOT_HASH_L1 = "68b4cb8517d292599b2f161f779758b9f3254d60302849f39d81650d0bd9fba0"


def log_l3_l4_run(*, family: str, model_id: str, level: str, feature_set: str, tau: float,
                  metrics: dict[str, float], seconds: float,
                  extra_tags: dict[str, str] | None = None,
                  extra_params: dict[str, object] | None = None,
                  stage: str = "S0", seed: int = 42) -> str:
    """یک MLflow run برای مدل‌های سطح L3/L4/آشتی‌شده باز و بسته می‌کند. ``metrics``
    مستقیماً (بدون فیلتر ``operational_metrics``) ثبت می‌شود — فراخوان مسئول معنادار
    بودن کلیدهاست."""
    _, cv_folds_hash = load_cv_folds()
    cfg = RunConfig(family=family, model_id=model_id, stage=stage, seed=seed,
                    level=level, feature_set=feature_set, tau=tau)
    with start_model_run(cfg, data_snapshot_hash=DATA_SNAPSHOT_HASH_L1,
                         cv_folds_hash=cv_folds_hash) as run:
        if extra_tags:
            mlflow.set_tags(extra_tags)
        if extra_params:
            mlflow.log_params(extra_params)
        clean_metrics = {k: float(v) for k, v in metrics.items()
                         if v is not None and isinstance(v, (int, float)) and v == v}  # v==v ⇒ نه NaN
        mlflow.log_metrics(clean_metrics)
        mlflow.log_metric("fit_seconds", float(seconds))
        return run.info.run_id
