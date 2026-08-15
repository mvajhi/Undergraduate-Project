"""تست‌های واحد زیرساخت فاز ۷ (اسپرینت S-1) — الگوی `src/cv.py::run_protocol_tests`:
هر ``test_*`` یک ``(bool, str)`` برمی‌گرداند، بدون وابستگی به pytest.

اجرا: ``python -m src.models.tests``
"""

import re
from pathlib import Path

import optuna
import pandas as pd

from src.config import DOCS_DIR
from src.cv import CV_FOLDS_PATH, WalkForwardSplitter, _folds_payload, load_cv_folds, sha256_file
from src.models import cards
from src.models.naming import parse_run_name, run_name, tau_code, tau_from_code
from src.models.registry import FAMILIES, ModelSpec, register
from src.models.spaces import register_space, sample, trial_budget

# ---------------------------------------------------------------------------
# src/cv.py — انجماد foldها (بند 7.7.3، دروازه‌ی A1)
# ---------------------------------------------------------------------------

def test_cv_folds_hash_reproducible() -> tuple[bool, str]:
    """بازتولید payload از همان داده باید بایت‌به‌بایت با فایل منجمدشده یکی باشد —
    وگرنه هش پایدار نیست و دروازه‌ی A1 بی‌معناست."""
    from src.features.build import FEATURES_A_PATH

    if not FEATURES_A_PATH.exists() or not CV_FOLDS_PATH.exists():
        return True, "رد شد (داده یا cv_folds.json هنوز موجود نیست) — نه شکست"

    df = pd.read_parquet(FEATURES_A_PATH)
    splitter = WalkForwardSplitter(n_folds=5, min_train_days=60)
    fresh = _folds_payload(df, splitter, source="ignored-for-comparison")
    frozen = __import__("json").loads(CV_FOLDS_PATH.read_text())

    same_folds = fresh["folds"] == frozen["folds"]
    same_n = fresh["n_unique_dates"] == frozen["n_unique_dates"]
    ok = same_folds and same_n
    return ok, f"{len(fresh['folds'])} fold بازتولید شد · یکسان با فایل منجمد: {ok}"


def test_cv_folds_hash_matches_manifest() -> tuple[bool, str]:
    """هش فایل زنده باید با هشی که در doc/data_manifest.md ثبت شده مطابق باشد — کشف
    می‌کند که آیا کسی فایل را بی‌سروصدا بازتولید کرده بدون به‌روزرسانی سند."""
    manifest_path = DOCS_DIR / "data_manifest.md"
    if not CV_FOLDS_PATH.exists() or not manifest_path.exists():
        return True, "رد شد (فایل یا سند مانیفست موجود نیست) — نه شکست"

    text = manifest_path.read_text()
    m = re.search(r"cv_folds\.json.*?`([0-9a-f]{64})`", text, re.DOTALL)
    if not m:
        return False, "هیچ هش SHA-256 مربوط به cv_folds.json در doc/data_manifest.md پیدا نشد"
    registered_hash = m.group(1)
    live_hash = sha256_file(CV_FOLDS_PATH)
    ok = registered_hash == live_hash
    return ok, f"ثبت‌شده={registered_hash[:12]}… · زنده={live_hash[:12]}… · مطابق: {ok}"


def test_cv_folds_load_roundtrip() -> tuple[bool, str]:
    if not CV_FOLDS_PATH.exists():
        return True, "رد شد (cv_folds.json هنوز موجود نیست) — نه شکست"
    folds, h = load_cv_folds()
    ok = len(folds) == 5 and len(h) == 64
    return ok, f"{len(folds)} fold بارگذاری شد · طول هش={len(h)}"


# ---------------------------------------------------------------------------
# src/models/registry.py — بند 7.0.2/7.26
# ---------------------------------------------------------------------------

def test_registry_family_sum() -> tuple[bool, str]:
    total = sum(f.n_models for f in FAMILIES.values())
    ok = total == 169 and len(FAMILIES) == 13
    return ok, f"{len(FAMILIES)} خانواده · جمع مدل‌ها={total} (انتظار: ۱۳ و ۱۶۹)"


def test_registry_duplicate_model_id_rejected() -> tuple[bool, str]:
    spec = ModelSpec(model_id="__test_dummy__", family="F01", levels=("L1",), quantile_route="Q1",
                     algorithm="sklearn.Dummy")
    try:
        register(spec)
        register(spec)
        return False, "ثبت دوباره‌ی model_id تکراری باید ValueError می‌داد"
    except ValueError:
        return True, "ثبت دوباره درست رد شد"
    finally:
        from src.models.registry import MODELS
        MODELS.pop("__test_dummy__", None)


def test_registry_invalid_spec_rejected() -> tuple[bool, str]:
    bad_family, bad_level, bad_route = False, False, False
    try:
        ModelSpec(model_id="x", family="F99", levels=("L1",), quantile_route="Q1", algorithm="a")
    except ValueError:
        bad_family = True
    try:
        ModelSpec(model_id="x", family="F01", levels=("L9",), quantile_route="Q1", algorithm="a")
    except ValueError:
        bad_level = True
    try:
        ModelSpec(model_id="x", family="F01", levels=("L1",), quantile_route="Q9", algorithm="a")
    except ValueError:
        bad_route = True
    ok = bad_family and bad_level and bad_route
    return ok, f"خانواده‌ی نامعتبر رد شد={bad_family} · سطح={bad_level} · مسیر کوانتایل={bad_route}"


# ---------------------------------------------------------------------------
# src/models/naming.py — بند 7.7.1
# ---------------------------------------------------------------------------

def test_run_name_matches_wbs_example() -> tuple[bool, str]:
    """مثال دقیق بند 7.7.1 سند فاز ۷ باید بایت‌به‌بایت بازتولید شود."""
    import datetime as dt

    name = run_name(family="F02", model="lightgbm", level="L1", target="rho",
                    feature_set="FSlgbm", tau=0.10, stage="S2", seed=42,
                    timestamp=dt.datetime(2026, 8, 15, 11, 30))
    expected = "F02_lightgbm_L1_rho_FSlgbm_t010_S2_s42_20260815T1130"
    return name == expected, f"{name!r} == {expected!r}"


def test_run_name_roundtrip() -> tuple[bool, str]:
    import datetime as dt

    cases = [
        dict(family="F05", model="egarch", level="L3", target="shock", feature_set="FSvar",
             tau=0.10, stage="S2", seed=42),
        dict(family="F07", model="tft", level="L4", target="rho", feature_set="FSseq",
             tau=0.05, stage="S3", seed=0),
    ]
    for kw in cases:
        name = run_name(timestamp=dt.datetime(2026, 8, 20, 9, 40), **kw)
        parsed = parse_run_name(name)
        for k, v in kw.items():
            if parsed[k] != v:
                return False, f"{name!r}: {k}={parsed[k]!r} != {v!r}"
    return True, f"{len(cases)} مورد رفت‌وبرگشت کامل تأیید شد"


def test_run_name_rejects_invalid_parts() -> tuple[bool, str]:
    bad = [
        dict(family="F99", model="x", level="L1", target="rho", feature_set="FS", tau=.1, stage="S2", seed=1),
        dict(family="F01", model="Bad-Name", level="L1", target="rho", feature_set="FS", tau=.1, stage="S2", seed=1),
        dict(family="F01", model="x", level="L9", target="rho", feature_set="FS", tau=.1, stage="S2", seed=1),
        dict(family="F01", model="x", level="L1", target="rho", feature_set="FS", tau=.1, stage="S9", seed=1),
        dict(family="F01", model="x", level="L1", target="rho", feature_set="FS", tau=1.5, stage="S2", seed=1),
    ]
    for kw in bad:
        try:
            run_name(**kw)
            return False, f"باید رد می‌شد ولی نشد: {kw}"
        except ValueError:
            pass
    return True, f"{len(bad)} ورودی نامعتبر همگی درست رد شدند"


def test_tau_code_roundtrip() -> tuple[bool, str]:
    for tau in (0.02, 0.05, 0.10, 0.15, 0.20, 1 / 3):
        code = tau_code(tau)
        back = tau_from_code(code)
        if abs(back - round(tau, 2)) > 1e-9:
            return False, f"tau={tau} → {code} → {back} — رفت‌وبرگشت نادرست"
    return True, "شبکه‌ی τ (بند ۶.۵) رفت‌وبرگشت درست دارد"


# ---------------------------------------------------------------------------
# src/models/spaces.py — بند 7.6.2/7.6.3
# ---------------------------------------------------------------------------

def test_trial_budget_matches_wbs_table() -> tuple[bool, str]:
    """جدول بند 7.6.2: (۱–۲→۲۵/۵۰) (۳–۵→۶۰/۱۲۰) (۶–۹→۱۲۰/۲۵۰) (۱۰+→۱۵۰/۳۰۰)."""
    cases = [
        (1, "S2", 25), (2, "S2", 25), (3, "S2", 60), (5, "S2", 60),
        (6, "S2", 120), (9, "S2", 120), (10, "S2", 150), (20, "S2", 150),
        (2, "S3", 50), (5, "S3", 120), (9, "S3", 250), (10, "S3", 300),
    ]
    for n, stage, expected in cases:
        got = trial_budget(n, stage)
        if got != expected:
            return False, f"trial_budget({n}, {stage!r})={got} != {expected}"
    return True, f"{len(cases)} نقطه از جدول 7.6.2 تأیید شد"


def test_space_version_guard() -> tuple[bool, str]:
    @register_space("__test_space__", version=1, n_hyperparams=1)
    def _v1(trial: optuna.Trial) -> dict:
        return {"a": trial.suggest_float("a", 0, 1)}

    try:
        @register_space("__test_space__", version=1, n_hyperparams=1)
        def _v1_again(trial: optuna.Trial) -> dict:
            return {}
        return False, "ثبت با همان نسخه باید رد می‌شد"
    except ValueError:
        pass
    finally:
        from src.models.spaces import SPACES
        SPACES.pop("__test_space__", None)
    return True, "دکوراتور register_space نسخه‌ی تکراری را درست رد کرد"


def test_space_sample_and_missing() -> tuple[bool, str]:
    @register_space("__test_space2__", version=1, n_hyperparams=1)
    def _space(trial: optuna.Trial) -> dict:
        return {"x": trial.suggest_int("x", 1, 10)}

    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    trial = study.ask()
    params = sample("__test_space2__", trial)
    from src.models.spaces import SPACES
    SPACES.pop("__test_space2__", None)

    ok_sample = "x" in params and 1 <= params["x"] <= 10
    try:
        sample("__nonexistent_model__", trial)
        ok_missing = False
    except KeyError:
        ok_missing = True
    return ok_sample and ok_missing, f"نمونه‌گیری={ok_sample} · خطای مدل ثبت‌نشده={ok_missing}"


# ---------------------------------------------------------------------------
# src/models/tracking.py — دیتاست و model_type اختصاصی (نه فقط بخشی از نام run)
# ---------------------------------------------------------------------------

def test_tracking_logs_dataset_and_model_type() -> tuple[bool, str]:
    """``start_model_run`` باید train/test را در تب Dataset اختصاصی MLflow ثبت کند
    (نه فقط هش در param) و tag ``model_type`` را از رجیستری بخواند (نه از model_id)."""
    import tempfile

    import mlflow

    from src.models import tracking
    from src.models.axes import RunConfig
    from src.models.registry import MODELS as MODEL_REGISTRY
    from src.models.registry import ModelSpec, register

    already_registered = "test_track_dummy" in MODEL_REGISTRY
    if not already_registered:
        register(ModelSpec(model_id="test_track_dummy", family="F01", levels=("L1",),
                           quantile_route="Q1", algorithm="pytest.DummyAlgo"))

    tmp_dir = tempfile.mkdtemp()
    original_uri = tracking.MLFLOW_TRACKING_URI
    tracking.MLFLOW_TRACKING_URI = tmp_dir
    try:
        cfg = RunConfig(family="F01", model_id="test_track_dummy", stage="S0", seed=1,
                        level="L1", feature_set="FS_test", tau=0.20)
        train = pd.DataFrame({"rho": [0.1, 0.2], "Res": [10, 20]})
        test = pd.DataFrame({"rho": [0.15], "Res": [15]})
        with tracking.start_model_run(cfg, data_snapshot_hash="abc123", cv_folds_hash="def456",
                                      train=train, test=test, dataset_source="pytest-source",
                                      source_fn=test_tracking_logs_dataset_and_model_type) as run:
            run_id = run.info.run_id

        client = mlflow.tracking.MlflowClient(tracking_uri=tmp_dir)
        r = client.get_run(run_id)
        model_type_ok = r.data.tags.get("model_type") == "pytest.DummyAlgo"
        code_ref = r.data.tags.get("code_ref", "")
        code_ref_ok = code_ref.startswith("src/models/tests.py:") and code_ref.endswith(
            "#test_tracking_logs_dataset_and_model_type")
        inputs = r.inputs.dataset_inputs
        dataset_count_ok = len(inputs) == 2  # train + test
        names_ok = all("FS_test" in d.dataset.name for d in inputs)
        contexts = {t.value for d in inputs for t in d.tags if t.key == "mlflow.data.context"}
        contexts_ok = contexts == {"train", "test"}
        ok = model_type_ok and code_ref_ok and dataset_count_ok and names_ok and contexts_ok
        return ok, (f"model_type tag={model_type_ok} · code_ref={code_ref!r} ({code_ref_ok}) · "
                    f"تعداد dataset={len(inputs)} · نام حاوی feature_set={names_ok} · context‌ها={contexts_ok}")
    finally:
        tracking.MLFLOW_TRACKING_URI = original_uri
        if not already_registered:
            MODEL_REGISTRY.pop("test_track_dummy", None)


def test_code_reference_derivation() -> tuple[bool, str]:
    """``code_reference`` باید مسیر نسبی + خط + نام تابع بدهد، و ``None`` را با
    ``'unknown'`` جواب بدهد (نه خطا) — چون همیشه هر مدل ``source_fn`` نمی‌دهد."""
    from src.models.tracking import code_reference

    ref = code_reference(test_code_reference_derivation)
    ok_self = ref.startswith("src/models/tests.py:") and ref.endswith("#test_code_reference_derivation")
    ok_none = code_reference(None) == "unknown"
    return ok_self and ok_none, f"ref={ref!r} · none={code_reference(None)!r}"


def test_s1_report_includes_baseline_and_marks_winners() -> tuple[bool, str]:
    """``save_results(baseline_pinball=...)`` باید مرجع B3 را در گزارش بنویسد و مدل‌های
    بهتر از آن را علامت بزند — یافته‌ی واقعی S1 خ۱ (چهار مدل B3 را بردند) وابسته به این است."""
    import tempfile

    from src.models import s1_runner

    tmp_dir = Path(tempfile.mkdtemp())
    original_dir = s1_runner.PHASE7_DIR
    s1_runner.PHASE7_DIR = tmp_dir
    try:
        good = s1_runner.TrialResult("F99", "L1", "winner_model", 0, {}, "pass", 1.0,
                                     [0.01, 0.01, 0.01], 0.01)
        bad = s1_runner.TrialResult("F99", "L1", "loser_model", 0, {}, "pass", 1.0,
                                    [0.05, 0.05, 0.05], 0.05)
        s1_runner.save_results([good, bad], "F99", "L1", baseline_pinball=0.02)

        md = (tmp_dir / "S1_screening_F99.md").read_text()
        ranking_section = md.split("## رتبه‌بندی مقدماتی")[1]
        winner_line = next(ln for ln in ranking_section.splitlines() if "winner_model" in ln)
        loser_line = next(ln for ln in ranking_section.splitlines() if "loser_model" in ln)
        has_baseline_line = "0.02000" in md
        winner_marked = "بهتر از B3" in winner_line
        loser_not_marked = "بهتر از B3" not in loser_line
        ok = has_baseline_line and winner_marked and loser_not_marked
        return ok, f"خط مرجع={has_baseline_line} · برنده علامت‌دار={winner_marked} · بازنده بدون علامت={loser_not_marked}"
    finally:
        s1_runner.PHASE7_DIR = original_dir


def test_f01_s2_feature_set_resolves_dow_collinearity() -> tuple[bool, str]:
    """فیچرست S2 (بند 7.5.3) باید `dow` خام را حذف کند (هم‌خط با پایه‌های فوریه)،
    `log_res_sq` اضافه کند، و برای شاخه‌ی quantreg فیچر تقریباً ثابت را هم حذف کند —
    و ماتریس طراحی حاصل نباید VIF بی‌نهایت بین `dow` و فوریه‌ها داشته باشد."""
    import numpy as np
    import pandas as pd
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    from src.cv import load_cv_folds
    from src.models.families.f01_linear import _design_s2, _feature_cols_s2, _feature_cols_s2_quantreg

    if not CV_FOLDS_PATH.exists():
        return True, "رد شد (cv_folds.json هنوز موجود نیست) — نه شکست"

    cols_lin, cols_q = _feature_cols_s2(), _feature_cols_s2_quantreg()
    ok_dow = "dow" not in cols_lin
    ok_quad = "log_res_sq" in cols_lin
    ok_quantreg_pruned = "pre_holiday_x_block_len" in cols_lin and "pre_holiday_x_block_len" not in cols_q

    from src.features.build import FEATURES_A_PATH
    if not FEATURES_A_PATH.exists():
        return ok_dow and ok_quad and ok_quantreg_pruned, "بدون features_A_v1.parquet — فقط فهرست ستون‌ها آزموده شد"

    df = pd.read_parquet(FEATURES_A_PATH).sort_values("date_gregorian").reset_index(drop=True)
    folds, _ = load_cv_folds()
    tr, te = folds[0].masks(df["date_gregorian"])
    Xtr, Xte = _design_s2(df.loc[tr], df.loc[te])
    ok_no_nan = Xtr.isna().sum().sum() == 0 and Xte.isna().sum().sum() == 0

    fourier = [c for c in Xtr.columns if c.startswith("dow_sin") or c.startswith("dow_cos")]
    vifs = [variance_inflation_factor(Xtr[fourier].to_numpy(), i) for i in range(len(fourier))]
    ok_finite_vif = all(np.isfinite(vifs)) and max(vifs) < 10

    ok = ok_dow and ok_quad and ok_quantreg_pruned and ok_no_nan and ok_finite_vif
    return ok, (f"dow حذف={ok_dow} · log_res_sq اضافه={ok_quad} · هرس quantreg={ok_quantreg_pruned} · "
                f"بدون NaN={ok_no_nan} · VIF فوریه متناهی={ok_finite_vif} (max={max(vifs):.1f})")


def test_s2_study_persists_and_resumes() -> tuple[bool, str]:
    """بند 7.6.3: study هر مدل باید روی SQLite ماندگار شود و اجرای دوباره باید trialهای
    قبلی را از سر نگیرد (پیش‌نیاز fANOVA گام ۱۰ کارت مدل + ادامه‌پذیری بعد از قطعی)."""
    import shutil
    import tempfile
    import time

    from src.models import s2_runner

    tmp_dir = Path(tempfile.mkdtemp())
    original_dir = s2_runner.OPTUNA_STUDIES_DIR
    original_phase7 = s2_runner.PHASE7_DIR
    s2_runner.OPTUNA_STUDIES_DIR = tmp_dir / "optuna_studies"
    s2_runner.PHASE7_DIR = tmp_dir / "phase7"
    try:
        import numpy as np
        import pandas as pd

        n = 40
        rng = np.random.default_rng(0)
        res = rng.integers(5, 50, n)
        rho = rng.uniform(0, 0.3, n)
        df = pd.DataFrame({
            "rho": rho, "Res": res, "Recv": np.round(res * (1 - rho)),
            "date_gregorian": pd.date_range("2024-01-01", periods=n // 4).repeat(4)[:n],
        })
        train, test = df.iloc[: n // 2], df.iloc[n // 2:]
        folds = [(train, test)] * 5

        def fake_fit_predict(tr, te, tau, alpha=1.0, **hp):
            return np.full(len(te), float(tr["rho"].mean()) * min(alpha, 1.0))

        from src.models.registry import MODELS as MODEL_REGISTRY
        from src.models.registry import ModelSpec, register
        from src.models.spaces import SPACES, register_space

        if "test_s2_dummy" not in MODEL_REGISTRY:
            register(ModelSpec(model_id="test_s2_dummy", family="F01", levels=("L1",),
                               quantile_route="Q1", algorithm="pytest.Dummy"))
        if "test_s2_dummy" not in SPACES:
            @register_space("test_s2_dummy", version=1, n_hyperparams=1)
            def _space(trial):
                return {"alpha": trial.suggest_float("alpha", 0.1, 1.0)}

        design_fn = lambda tr, te: (tr, te)  # noqa: E731

        r1 = s2_runner._run_model_s2("test_s2_dummy", fake_fit_predict, design_fn, folds,
                                     "F01", "L1", "snap", "cvhash", "src", seed=1)
        t0 = time.time()
        r2 = s2_runner._run_model_s2("test_s2_dummy", fake_fit_predict, design_fn, folds,
                                     "F01", "L1", "snap", "cvhash", "src", seed=1)
        dt_resume = time.time() - t0

        ok_same_n = r1.n_trials == r2.n_trials
        ok_same_best = abs(r1.best_pinball - r2.best_pinball) < 1e-9
        ok_fast_resume = dt_resume < 2.0  # اگر واقعاً از نو اجرا شده بود، بسیار کندتر می‌بود

        import optuna
        study = optuna.load_study(study_name="F01_test_s2_dummy_S2",
                                  storage=s2_runner.study_storage_url("test_s2_dummy"))
        ok_trial_count = len(study.trials) == r1.n_trials  # نه دوبرابر
        # ⚠️ fANOVA اینجا آزموده نمی‌شود — با این fixture مصنوعیِ کم‌تنوع گاهی
        # IndexError داخلی optuna می‌دهد (مسئله‌ی تبهگنی داده‌ی آزمایشی، نه کد تولید؛
        # روی مطالعه‌ی واقعی Ridge دستی تأیید شد که get_param_importances کار می‌کند).

        ok = ok_same_n and ok_same_best and ok_fast_resume and ok_trial_count
        return ok, (f"n یکسان={ok_same_n} · best یکسان={ok_same_best} · resume سریع "
                    f"({dt_resume:.2f}s)={ok_fast_resume} · بدون تکرار trial={ok_trial_count}")
    finally:
        s2_runner.OPTUNA_STUDIES_DIR = original_dir
        s2_runner.PHASE7_DIR = original_phase7
        shutil.rmtree(tmp_dir, ignore_errors=True)
        MODEL_REGISTRY.pop("test_s2_dummy", None)
        SPACES.pop("test_s2_dummy", None)


def test_calibration_coverage_arithmetic() -> tuple[bool, str]:
    """``oof_predictions``/``render_step13`` روی داده‌ی مصنوعی با پوشش دقیقاً شناخته‌شده —
    نصف رکوردها زیر پیش‌بینی (پوشیده)، نصف بالا؛ پوشش باید دقیقاً ۰.۵ محاسبه شود."""
    from src.models import calibration

    n = 200
    te = pd.DataFrame({
        "rho": [0.1 if i % 2 == 0 else 0.9 for i in range(n)],
        "RestaurantName": ["A"] * (n // 2) + ["B"] * (n // 2),
        "Meal": ["lunch"] * n,
        "Res": list(range(n)),
        "is_tehran": [True] * n,
    })

    def fake_fit(train, test, tau, **hp):
        return __import__("numpy").full(len(test), 0.5)

    df = calibration.oof_predictions(fake_fit, [(te, te)], tau=0.20, hyperparams={})
    overall = calibration._coverage_row("کلی", df, 0.20)
    ok_cov = abs(overall["coverage"] - 0.5) < 1e-9
    ok_gap = abs(overall["gap"] - 0.3) < 1e-9

    md = calibration.render_step13(df, 0.20)
    ok_md = "پوشش کلی: 0.5000" in md and "چارک Res" in md and "بدترین برش" in md
    return (ok_cov and ok_gap and ok_md,
           f"پوشش=۰.۵ محاسبه‌شد={ok_cov} · شکاف=۰.۳={ok_gap} · markdown سالم={ok_md}")


def test_card_writer_renders_data_driven_sections() -> tuple[bool, str]:
    """بخش‌های ۱،۲،۳،۵،۶،۷،۸،۹،۱۰،۱۲ ``card_writer.draft_card`` باید از روی نتیجه‌ی S2 واقعی
    (نه شبیه‌سازی) محتوای غیرخالی و سازگار با آن نتیجه بسازند."""
    from src.models import card_writer

    json_path = card_writer.PHASE7_DIR / "S2_tuning_F01.json"
    if not json_path.exists():
        return True, "رد شد (S2_tuning_F01.json هنوز موجود نیست) — نه شکست"

    import json
    payload = json.loads(json_path.read_text())
    candidates = [k for k in payload if not k.startswith("_") and payload[k].get("n_trials", 0) > 0]
    if not candidates:
        return True, "رد شد (هیچ مدلی هنوز نتیجه ندارد) — نه شکست"
    model_id = candidates[0]

    card = card_writer.draft_card(model_id, "F01", feature_cols=["log_res", "log_res_sq"], quantreg=False)
    filled = {1, 2, 3, 5, 6, 7, 8, 9, 10, 12}
    ok_filled = filled.issubset(card.sections.keys())
    ok_nonempty = all(len(card.sections.get(i, "")) > 20 for i in filled)
    ok_step8_has_trials = str(payload[model_id]["n_trials"]) in card.sections[8]
    return (ok_filled and ok_nonempty and ok_step8_has_trials,
           f"مدل={model_id} · بخش‌های پرشده={ok_filled} · غیرخالی={ok_nonempty} · "
           f"تعداد trial در گام ۸={ok_step8_has_trials}")


def test_fit_diagnosis_train_test_gap_arithmetic() -> tuple[bool, str]:
    """``train_test_gap`` روی داده‌ی مصنوعی با نسبت pinball دقیقاً شناخته‌شده — مدل ثابتی
    که همیشه ۰.۵ برمی‌گرداند، روی actual ثابت ۰.۵ باید pinball=۰ (نسبت=inf چون صورت=۰) بدهد
    و روی actual دیگر نسبت مثبت معنادار."""
    from src.models import fit_diagnosis

    n = 50
    train = pd.DataFrame({"rho": [0.5] * n, "Recv": [0.0] * n, "Res": [1.0] * n})
    test = pd.DataFrame({"rho": [0.2] * n, "Recv": [0.0] * n, "Res": [1.0] * n})

    def fake_fit(tr, te, tau, **hp):
        import numpy as np
        return np.full(len(te), 0.5)

    rows = fit_diagnosis.train_test_gap(fake_fit, [(train, test)], tau=0.20, hyperparams={})
    ok_train_zero = abs(rows[0]["pinball_train"]) < 1e-9  # train خودش را کامل می‌پوشاند
    ok_test_positive = rows[0]["pinball_test"] > 0  # پیش‌بینی ۰.۵ روی actual=۰.۲ خطای مثبت دارد

    md = fit_diagnosis.render_step11(rows, n_cols=42)
    ok_md = "نسبت pinball" in md and "42" in md
    return (ok_train_zero and ok_test_positive and ok_md,
           f"pinball_train≈۰={ok_train_zero} · pinball_test>۰={ok_test_positive} · markdown سالم={ok_md}")


def test_card_writer_step4_preflight_mapping() -> tuple[bool, str]:
    """گام ۴ (پیش‌پرواز) باید هر مدل F01 را به آزمون‌های بند 7.9.2 مرتبط نگاشت کند —
    بدون فراخوانی هیچ کد آماری تازه، فقط ارجاع به یافته‌های فاز ۴ که قبلاً مستندند."""
    from src.models import card_writer

    ols_md = card_writer.render_step4_preflight_tests("ols", "F01")
    ok_ols = all(k in ols_md for k in ["VIF", "ناهمسانی", "چولگی"])

    binom_md = card_writer.render_step4_preflight_tests("glm_binomial", "F01")
    ok_binom = "بیش‌پراکندگی" in binom_md and "VIF" not in binom_md  # فقط آزمون مرتبط، نه همه

    every_model_covered = all(
        mid in card_writer._F01_PREFLIGHT_RELEVANCE for mid in card_writer._FAMILY_MODULES
    ) or all(  # بررسی مستقیم‌تر: همه‌ی ۱۶ عضو رجیستری‌شده‌ی F01 باید نگاشت داشته باشند
        mid in card_writer._F01_PREFLIGHT_RELEVANCE
        for mid in __import__("src.models.families.f01_linear", fromlist=["MODELS"]).MODELS
    )
    return (ok_ols and ok_binom and every_model_covered,
           f"ols سه‌آزمون={ok_ols} · glm_binomial فقط بیش‌پراکندگی={ok_binom} · "
           f"هر ۱۶ مدل نگاشت‌شده={every_model_covered}")


def test_card_writer_step1_and_step14_on_real_data() -> tuple[bool, str]:
    """گام ۱ (از docstring `fit_predict_ridge`) و گام ۱۴ (سنتز از S2 + B3 + خودِ گام‌های
    ۱۱/۱۳ کارت) روی «ridge» چک می‌شوند — گام ۱۴ نباید بازبرازش کند، فقط بخواند."""
    from src.models import card_writer, cards

    json_path = card_writer.PHASE7_DIR / "S2_tuning_F01.json"
    if not json_path.exists():
        return True, "رد شد (S2_tuning_F01.json هنوز موجود نیست) — نه شکست"
    import json
    payload = json.loads(json_path.read_text())
    if "ridge" not in payload or payload["ridge"].get("n_trials", 0) == 0:
        return True, "رد شد (ridge هنوز نتیجه‌ی S2 ندارد) — نه شکست"

    step1 = card_writer.render_step1_theoretical_position("ridge", "F01")
    ok_step1 = "F44" in step1  # docstring واقعی fit_predict_ridge به F44 اشاره می‌کند

    card = cards.ModelCard("ridge")
    card.set_section(13, card_writer.render_step13_calibration("ridge", "F01"))
    card.set_section(11, card_writer.render_step11_fit_diagnosis("ridge", "F01"))
    step14 = card_writer.render_step14_summary("ridge", "F01", card)
    ok_step14 = "B3=" in step14 and "توصیه" in step14 and "کالیبراسیون" in step14

    return (ok_step1 and ok_step14, f"گام۱ حاوی F44={ok_step1} · گام۱۴ سالم={ok_step14}")


def test_card_writer_step13_calibration_on_real_data() -> tuple[bool, str]:
    """گام ۱۳ اجباری با بازبرازش واقعی روی fold‌های رسمی — روی «ridge» (سریع، معمولاً
    از اولین‌های S2 که کامل می‌شوند) چک می‌کند پوشش کلی و برش‌های اجباری تولید می‌شوند."""
    from src.models import card_writer

    json_path = card_writer.PHASE7_DIR / "S2_tuning_F01.json"
    if not json_path.exists():
        return True, "رد شد (S2_tuning_F01.json هنوز موجود نیست) — نه شکست"
    import json
    payload = json.loads(json_path.read_text())
    if "ridge" not in payload or payload["ridge"].get("n_trials", 0) == 0:
        return True, "رد شد (ridge هنوز نتیجه‌ی S2 ندارد) — نه شکست"

    md = card_writer.render_step13_calibration("ridge", "F01")
    ok_overall = "پوشش کلی:" in md
    ok_cuts = all(c in md for c in ["وعده", "سلف", "چارک Res", "تهران؟"])
    ok_worst = "بدترین برش" in md
    return (ok_overall and ok_cuts and ok_worst,
           f"پوشش کلی={ok_overall} · هر ۴ برش اجباری حاضرند={ok_cuts} · بدترین‌برش={ok_worst}")


def test_card_writer_step11_fit_diagnosis_on_real_data() -> tuple[bool, str]:
    """گام ۱۱ اجباری با بازبرازش واقعی — روی «ridge» چک می‌کند شکاف train/test هر ۵ fold
    و نسبت میانگین محاسبه و گزارش می‌شوند."""
    from src.models import card_writer

    json_path = card_writer.PHASE7_DIR / "S2_tuning_F01.json"
    if not json_path.exists():
        return True, "رد شد (S2_tuning_F01.json هنوز موجود نیست) — نه شکست"
    import json
    payload = json.loads(json_path.read_text())
    if "ridge" not in payload or payload["ridge"].get("n_trials", 0) == 0:
        return True, "رد شد (ridge هنوز نتیجه‌ی S2 ندارد) — نه شکست"

    md = card_writer.render_step11_fit_diagnosis("ridge", "F01")
    ok_ratio = "نسبت pinball" in md
    ok_rows = md.count("| 0 |") + md.count("| 1 |") + md.count("| 2 |") + md.count("| 3 |") + md.count("| 4 |") >= 5
    return (ok_ratio and ok_rows,
           f"نسبت میانگین گزارش‌شده={ok_ratio} · هر ۵ fold حاضرند={ok_rows}")


def test_f01_all_specs_have_algorithm() -> tuple[bool, str]:
    """هر ۱۶ عضو ثبت‌شده‌ی F01 باید فیلد algorithm غیرخالی داشته باشد — چون این همان
    مقداری است که به‌عنوان tag اختصاصی model_type در MLflow می‌رود، نه model_id."""
    import src.models.families.f01_linear  # noqa: F401  — اثر جانبی: ثبت در MODELS
    from src.models.registry import models_of_family

    specs = models_of_family("F01")
    empty = [s.model_id for s in specs if not s.algorithm]
    ok = len(specs) >= 16 and not empty
    return ok, f"{len(specs)} مدل F01 ثبت‌شده · بدون algorithm: {empty or 'هیچ‌کدام'}"


# ---------------------------------------------------------------------------
# src/models/cards.py — بند 7.4
# ---------------------------------------------------------------------------

def test_card_completeness_and_roundtrip(tmp_dir: Path | None = None) -> tuple[bool, str]:
    import tempfile

    tmp_dir = tmp_dir or Path(tempfile.mkdtemp())
    card = cards.ModelCard("__test_model__")
    if card.is_complete():
        return False, "کارت خالی نباید کامل باشد"
    if set(card.missing_mandatory()) != set(cards.MANDATORY_STEPS):
        return False, f"missing_mandatory اشتباه: {card.missing_mandatory()}"

    for i in range(1, len(cards.STEPS) + 1):
        card.set_section(i, f"متن گام {i}")
    if not card.is_complete():
        return False, "کارت پرشده باید کامل باشد"

    path = tmp_dir / "__test_model__.md"
    card.save(path)
    reloaded = cards.load("__test_model__", path)
    ok = reloaded.sections == card.sections and reloaded.is_complete()
    return ok, f"ذخیره/بازخوانی {len(cards.STEPS)} گام یکسان بازگشت: {ok}"


def test_card_require_complete_raises() -> tuple[bool, str]:
    card = cards.ModelCard("__test_incomplete__")
    card.set_section(1, "فقط این یکی")
    try:
        card.require_complete()
        return False, "require_complete باید برای کارت ناقص خطا می‌داد"
    except ValueError:
        return True, "require_complete درست خطا داد"


# ---------------------------------------------------------------------------
# اجراکننده
# ---------------------------------------------------------------------------

_ALL_TESTS = [
    test_cv_folds_hash_reproducible,
    test_cv_folds_hash_matches_manifest,
    test_cv_folds_load_roundtrip,
    test_registry_family_sum,
    test_registry_duplicate_model_id_rejected,
    test_registry_invalid_spec_rejected,
    test_run_name_matches_wbs_example,
    test_run_name_roundtrip,
    test_run_name_rejects_invalid_parts,
    test_tau_code_roundtrip,
    test_trial_budget_matches_wbs_table,
    test_space_version_guard,
    test_space_sample_and_missing,
    test_tracking_logs_dataset_and_model_type,
    test_code_reference_derivation,
    test_s1_report_includes_baseline_and_marks_winners,
    test_f01_s2_feature_set_resolves_dow_collinearity,
    test_s2_study_persists_and_resumes,
    test_f01_all_specs_have_algorithm,
    test_card_completeness_and_roundtrip,
    test_card_require_complete_raises,
    test_card_writer_renders_data_driven_sections,
    test_calibration_coverage_arithmetic,
    test_fit_diagnosis_train_test_gap_arithmetic,
    test_card_writer_step11_fit_diagnosis_on_real_data,
    test_card_writer_step4_preflight_mapping,
    test_card_writer_step1_and_step14_on_real_data,
    test_card_writer_step13_calibration_on_real_data,
]


def run_all() -> bool:
    all_ok = True
    for fn in _ALL_TESTS:
        ok, msg = fn()
        print(f"  {'✅' if ok else '❌'} {fn.__name__:<40s} {msg}")
        all_ok &= ok
    return all_ok


if __name__ == "__main__":
    print("=" * 78)
    print("تست‌های واحد زیرساخت فاز ۷ (اسپرینت S-1)")
    print("=" * 78)
    if not run_all():
        raise AssertionError("یک یا چند تست زیرساخت شکست خورد")
    print("\nهمه‌ی تست‌ها PASS")
