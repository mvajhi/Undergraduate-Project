"""بند 7.19 سند فاز ۷ — خانواده‌ی ۱۰: نمونه‌محور و ناپارامتری (F10).

⚠️ فقط یک نماینده طبق فهرست کوتاه اسپرینت C (`doc/decisions/37-phase7-rescope.md`
بند ۵): **kNN-Quantile** — تعمیم طبیعی خط پایه‌ی B3 (کوانتایل تجربی گروهی)؛ کوانتایل
تجربی $\\tau$ در میان $k$ همسایه‌ی نزدیک (فاصله‌ی اقلیدسی روی فیچرهای مقیاس‌شده،
بند 7.19.2 مقیاس‌بندی اجباری) به‌جای گروه‌بندی گسسته‌ی سلف×وعده×dow که B3 استفاده
می‌کند.
"""

import numpy as np
import optuna
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.models.families import common
from src.models.families.f01_linear import _design_s2
from src.models.registry import ModelSpec, register
from src.models.spaces import register_space

FAMILY = "F10"
LEVEL = "L1"


def fit_predict_knn_quantile(train: pd.DataFrame, test: pd.DataFrame, tau: float,
                             k: int = 30, **hp) -> np.ndarray:
    """کوانتایل تجربی $\\tau$ میان $k$ نزدیک‌ترین همسایه — بند 7.19.2: مقیاس‌بندی
    اجباری، وگرنه فاصله بدون معنا می‌شود."""
    Xtr, Xte = _design_s2(train, test, quantreg=False)
    scaler = StandardScaler().fit(Xtr)
    Ztr, Zte = scaler.transform(Xtr), scaler.transform(Xte)
    y = train["rho"].to_numpy()

    k_eff = min(k, len(Ztr))
    nbrs = NearestNeighbors(n_neighbors=k_eff).fit(Ztr)
    _, idx = nbrs.kneighbors(Zte)
    neighbor_rho = y[idx]  # (n_test, k)
    return np.clip(np.quantile(neighbor_rho, tau, axis=1), 0.0, 1.0)


MODELS = {"knn_quantile": fit_predict_knn_quantile}

register(ModelSpec(model_id="knn_quantile", family=FAMILY, levels=(LEVEL,), quantile_route="Q1",
                   algorithm="sklearn.NearestNeighbors + کوانتایل تجربی همسایگی"))


@register_space("knn_quantile", version=1, n_hyperparams=1)
def _space_knn_quantile(trial: optuna.Trial) -> dict:
    return {"k": trial.suggest_int("k", 5, 200, log=True)}


def main() -> None:
    from src.config import set_global_seed
    from src.cv import load_cv_folds, sha256_file
    from src.features.build import FEATURES_A_PATH
    from src.models.s0_runner import RESULTS_MD, baseline_reference, print_summary, run_family_s0, save_results

    set_global_seed()
    df = pd.read_parquet(FEATURES_A_PATH).sort_values("date_gregorian").reset_index(drop=True)
    folds, cv_folds_hash = load_cv_folds()
    f = folds[0]
    tr_mask, te_mask = f.masks(df["date_gregorian"])
    train, test = df.loc[tr_mask], df.loc[te_mask]
    data_snapshot_hash = sha256_file(FEATURES_A_PATH)

    print(f"R0 — {FAMILY} ({LEVEL}) — {f} (train={len(train):,}, test={len(test):,})")
    results = run_family_s0(FAMILY, LEVEL, MODELS, train, test, feature_set="FS_day",
                            data_snapshot_hash=data_snapshot_hash, cv_folds_hash=cv_folds_hash,
                            dataset_source=str(FEATURES_A_PATH))
    baseline = baseline_reference(train, test)
    ok = print_summary(results, baseline)
    save_results(results, baseline)
    print(f"\nذخیره شد در {RESULTS_MD}")
    if not ok:
        raise AssertionError("knn_quantile در R0 شکست خورد")


def run_s1(n_jobs: int | None = None) -> None:
    from src.config import set_global_seed
    from src.cv import load_cv_folds, sha256_file
    from src.features.build import FEATURES_A_PATH
    from src.models.s1_runner import (
        DEFAULT_N_JOBS,
        N_SCREENING_FOLDS,
        baseline_reference_multi_fold,
        run_family_s1,
        save_results,
    )

    set_global_seed()
    df = pd.read_parquet(FEATURES_A_PATH).sort_values("date_gregorian").reset_index(drop=True)
    folds, cv_folds_hash = load_cv_folds()
    data_snapshot_hash = sha256_file(FEATURES_A_PATH)

    screening_folds = []
    for f in folds[:N_SCREENING_FOLDS]:
        tr_mask, te_mask = f.masks(df["date_gregorian"])
        screening_folds.append((df.loc[tr_mask], df.loc[te_mask]))

    jobs = n_jobs or DEFAULT_N_JOBS
    results = run_family_s1(FAMILY, LEVEL, "src.models.families.f10_instance", list(MODELS),
                            screening_folds, feature_set="FS_day", data_snapshot_hash=data_snapshot_hash,
                            cv_folds_hash=cv_folds_hash, dataset_source=str(FEATURES_A_PATH), n_jobs=jobs)
    baseline = baseline_reference_multi_fold(screening_folds)
    save_results(results, FAMILY, LEVEL, baseline_pinball=baseline)
    n_fail = sum(1 for r in results if r.status == "fail")
    print(f"{len(results)} trial تمام شد · {n_fail} شکست")


def run_s2(n_jobs: int | None = None) -> None:
    from src.config import set_global_seed
    from src.cv import load_cv_folds, sha256_file
    from src.features.build import FEATURES_A_PATH
    from src.models.s2_runner import (
        N_S2_FOLDS,
        baseline_reference_5fold,
        run_family_s2,
        save_one_result,
        save_results,
    )

    set_global_seed()
    df = pd.read_parquet(FEATURES_A_PATH).sort_values("date_gregorian").reset_index(drop=True)
    folds, cv_folds_hash = load_cv_folds()
    data_snapshot_hash = sha256_file(FEATURES_A_PATH)
    if len(folds) != N_S2_FOLDS:
        raise AssertionError(f"cv_folds.json باید {N_S2_FOLDS} fold داشته باشد")

    tuning_folds = []
    for f in folds:
        tr_mask, te_mask = f.masks(df["date_gregorian"])
        tuning_folds.append((df.loc[tr_mask], df.loc[te_mask]))

    def _on_result(model_id: str, result) -> None:
        save_one_result(model_id, result, FAMILY, LEVEL)

    results = run_family_s2(FAMILY, LEVEL, "src.models.families.f10_instance", list(MODELS), tuning_folds,
                            data_snapshot_hash=data_snapshot_hash, cv_folds_hash=cv_folds_hash,
                            dataset_source=str(FEATURES_A_PATH), n_jobs=n_jobs or 1, on_result=_on_result)
    baseline = baseline_reference_5fold(tuning_folds)
    save_results(results, FAMILY, LEVEL, baseline_pinball=baseline)
    n_fail = sum(r.n_fail for r in results.values())
    print(f"{len(results)} مدل تنظیم شد · {n_fail} trial شکست‌خورده")


# ⚠️ عمداً بدون بلوک ``if __name__ == "__main__":`` — بند ۱.۵ استاندارد اجرا.
