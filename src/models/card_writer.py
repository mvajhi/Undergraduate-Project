"""بند 7.4 — نویسنده‌ی نیمه‌خودکار کارت مدل.

بخش‌های **داده‌محور** (۵، ۷، ۸، ۹، ۱۰، ۱۲) مستقیماً از نتیجه‌ی S2 (`s2_runner`) و مطالعه‌ی
Optuna پایدارشده (`optuna_studies/{model_id}.db`) ساخته می‌شوند — دستی نوشته نمی‌شوند تا با
تغییر کد از هم جدا نیفتند. بخش‌های ۱، ۲، ۳، ۴، ۶ هم به‌اندازه‌ی کافی مکانیکی‌اند (۱: docstring
خودِ `fit_predict_*`؛ ۲: سطح L1 ثابت؛ ۳: نگاشت هدف از `quantile_route`؛ ۴: نگاشت مدل→آزمون
پیش‌پرواز *قبلاً*-اجراشده‌ی بند 7.9.2؛ ۶: پیش‌پردازش از `common.design_matrix`) و کامل پر
می‌شوند. بخش **روایی باقی‌مانده** (۱۴) که به قضاوت دامنه‌ای نیاز دارد و به گام‌های ۱۱/۱۳
همین کارت وابسته است، جدا سنتز می‌شود (پایین‌تر).

گام‌های ۱۱ (تشخیص برازش) و ۱۳ (کالیبراسیون) با ``render_step11_fit_diagnosis``/
``render_step13_calibration`` جدا محاسبه می‌شوند — چون برخلاف بخش‌های بالا نیازمند
**بازبرازش واقعی** مدل روی هر ۵ fold با بهترین هایپرپارامتر S2 هستند
(``src/models/fit_diagnosis.py``, ``src/models/calibration.py``)، نه فقط خواندن نتیجه‌ی
ذخیره‌شده؛ برای مدل‌های کند (مثل رگرسیون کوانتایل ترکیبی) این هزینه‌ی قابل‌توجهی دارد،
پس در ``draft_card`` با پرچم ``include_refit_diagnostics=False`` پیش‌فرض خاموش است.
"""

import functools
import importlib
import json
import re

import numpy as np
import optuna
import pandas as pd

from src.models import cards
from src.models.registry import MODELS as MODEL_REGISTRY
from src.models.s2_runner import PHASE7_DIR, study_storage_url

optuna.logging.set_verbosity(optuna.logging.WARNING)

#: خانواده → مسیر دات‌دار ماژول — برای بارگذاری ``MODELS``/``QUANTREG_MODEL_IDS`` در
#: ``render_step13_calibration`` بدون وابستگی مستقیم به یک خانواده‌ی خاص.
_FAMILY_MODULES = {"F01": "src.models.families.f01_linear", "F02": "src.models.families.f02_tree"}

#: بند 7.9.2 WBS — آزمون‌های پیش‌پرواز خ۱ که در فاز ۴/۶ **قبلاً** اجرا و در دفتر حقایق
#: ثبت شده‌اند (نه اینجا دوباره محاسبه می‌شوند — فقط ارجاع/نگاشط به مدل مربوطه‌اند).
_PREFLIGHT_TESTS = {
    "vif": ("VIF رگرسورها", "✅ انجام‌شده در فاز ۴", "F44: `temp_mean`=۱۸۲.۹ ⇒ نیازمند مهار هم‌خطی/منظم‌سازی"),
    "heteroscedasticity": ("ناهمسانی واریانس (Breusch-Pagan/White)", "✅ انجام‌شده در فاز ۴",
                          "F06: BP p=۶.۴e−۴۰، White p=۱.۴e−۵۳ ⇒ فرض همسانی واریانس OLS رد می‌شود"),
    "skew_kurtosis": ("چولگی/کشیدگی هدف", "✅ انجام‌شده در فاز ۴",
                     "F02: چولگی ۴.۰۶، کشیدگی ۳۱.۹ ⇒ OLS نامناسب، فقط مرجع مطلق"),
    "zero_inflation": ("تورم صفر و مُدبودن", "✅ انجام‌شده در فاز ۴",
                      "F03: ۴.۹۰٪ رکورد دقیقاً صفر ولی تک‌مُدی (Sarle=۰.۴۶۱) ⇒ دوبخشی کافی است، ZI کامل لازم نیست"),
    "distributional_fit": ("برازش توزیعی (Gamma در برابر Beta)", "✅ انجام‌شده در فاز ۴",
                          "F04: Gamma (KS=۰.۰۴۲۹) بهتر از Beta (۰.۰۶۱۵) برازش می‌دهد"),
    "overdispersion": ("بیش‌پراکندگی ($\\chi^2/df$)", "✅ انجام‌شده در فاز ۴",
                      "F07: نسبت بیش‌پراکندگی صعودی ۳.۷×→۱۵.۶× ⇒ GLM دوجمله‌ای رد می‌شود (بند 7.10.1 عضو ۱۴)"),
}

#: model_id → کدام آزمون‌های بالا برایش مرتبط‌اند (بند 7.9.2، ردیف «خ۱»)
_F01_PREFLIGHT_RELEVANCE = {
    "ols": ("vif", "heteroscedasticity", "skew_kurtosis"),
    "ridge": ("vif", "heteroscedasticity"),
    "lasso": ("vif", "heteroscedasticity"),
    "elasticnet": ("vif", "heteroscedasticity"),
    "adaptive_lasso": ("vif", "heteroscedasticity"),
    "group_lasso": ("vif", "heteroscedasticity"),
    "quantile_regression": ("vif", "heteroscedasticity"),
    "l1_quantile_regression": ("vif", "heteroscedasticity"),
    "composite_quantile_regression": ("vif", "heteroscedasticity"),
    "expectile_regression": ("vif", "heteroscedasticity"),
    "glm_gamma": ("vif", "distributional_fit"),
    "glm_tweedie": ("vif", "zero_inflation"),
    "beta_regression": ("vif", "distributional_fit"),
    "glm_binomial": ("overdispersion",),
    "hurdle": ("vif", "zero_inflation"),
    "gam": ("vif",),
}


def render_step4_preflight_tests(model_id: str, family: str) -> str:
    """آزمون‌های پیش‌پرواز مرتبط با این مدل — طبق بند 7.9.2، برای خ۱ همه در فاز ۴/۶
    قبلاً اجرا شده‌اند (اینجا دوباره محاسبه نمی‌شوند، فقط با کد یافته ارجاع داده می‌شوند)."""
    if family == "F02":
        return (
            "بند 7.9.2 هیچ آزمون پیش‌پروازی برای خانواده‌ی درختی/بوستینگ اجباری نکرده "
            "(جدول آن بند فقط ADF/KPSS/Ljung-Box برای سری‌زمانی، ARCH-LM برای GARCH، و "
            "VIF/Breusch-Pagan برای خطی/توزیعی را می‌خواهد). درخت/بوستینگ نه فرض خطی‌بودن "
            "دارد و نه فرض همسانی واریانس — **نبود آزمون اینجا خودش یک یافته‌ی مثبت است**، "
            "نه غفلت."
        )
    if family != "F01":
        return "_نگاشت آزمون پیش‌پرواز برای این خانواده هنوز تعریف نشده._"
    keys = _F01_PREFLIGHT_RELEVANCE.get(model_id)
    if not keys:
        return "_مدل در نگاشت آزمون‌های پیش‌پرواز خ۱ ثبت نشده._"
    lines = ["| آزمون | وضعیت | شاهد |", "|---|---|---|"]
    for k in keys:
        name, status, evidence = _PREFLIGHT_TESTS[k]
        lines.append(f"| {name} | {status} | {evidence} |")
    lines.append("\nمنبع: بند 7.9.2 `doc/WBS-phase7-modeling.md` (ردیف «خ۱»)، کدهای یافته از دفتر حقایق فاز ۴.")
    return "\n".join(lines)


_QUANTILE_ROUTE_DESC = {
    "Q1": "بومی — مدل مستقیماً Pinball@τ را کمینه می‌کند (رگرسیون کوانتایل).",
    "Q2": "از توزیع پارامتری برازش‌شده — کوانتایل با معکوس CDF توزیع فرضی محاسبه می‌شود.",
    "Q3": "از توزیع تجربی باقیمانده — پیش‌بینی میانگین + آفست کوانتایل باقیمانده به تفکیک چارک Res.",
}


def render_step1_theoretical_position(model_id: str, family: str) -> str:
    """docstring خودِ ``fit_predict_*`` معمولاً همان توجیه «چرا این مدل» بند 7.10.1 است
    (نوشته‌شده هنگام پیاده‌سازی)؛ برای مدل‌های بدون docstring (مثل ols بی‌نظم‌سازها)،
    به توضیح عمومی از رجیستری برمی‌گردد."""
    import inspect

    spec = MODEL_REGISTRY.get(model_id)
    mod = importlib.import_module(_FAMILY_MODULES[family])
    fn = mod.MODELS.get(model_id)
    doc = inspect.getdoc(fn) if fn else None
    if doc:
        return doc
    if spec is None:
        return "_مدل در رجیستری ثبت نشده._"
    return f"عضو خانواده‌ی {family} (بند 7.10.1) — پیاده‌سازی: `{spec.algorithm}`."


def load_s2_result(model_id: str, family: str) -> dict:
    path = PHASE7_DIR / f"S2_tuning_{family}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} هنوز وجود ندارد — S2 این خانواده اجرا نشده")
    payload = json.loads(path.read_text())
    if model_id not in payload:
        raise KeyError(f"{model_id!r} هنوز نتیجه‌ی S2 ندارد (احتمالاً هنوز در حال اجراست)")
    return payload[model_id]


def _load_study(model_id: str, family: str) -> optuna.Study:
    return optuna.load_study(study_name=f"{family}_{model_id}_S2", storage=study_storage_url(model_id))


def render_step5_feature_set(model_id: str, feature_cols: list[str], quantreg: bool,
                             family: str = "F01") -> str:
    if family == "F02":
        return (
            "بند 7.5.3، ردیف «درختی/بوستینگ». پایه: `FS_full_A` کامل، **بدون حذف زودهنگام** "
            "— درخت برهم‌کنش و غیرخطی را خودش کشف می‌کند؛ هرس فقط با SHAP-RFE گام ۳ که در "
            "فهرست کوتاه اسپرینت C نبود.\n\n"
            f"**{len(feature_cols)} ستون نهایی**: `{', '.join(feature_cols)}`.\n\n"
            "دسته‌ای‌ها (`RestaurantName` ۳۰ سطح، `FoodType`, `Meal`, `RestaurantType`, "
            "`city`, `precip_type`) **خام** به LightGBM/CatBoost داده می‌شوند (نه یک‌هات) — "
            "رمزگذاری بومی این کتابخانه‌ها معمولاً بهتر از یک‌هات عمل می‌کند."
        )
    bucket = "رگرسیون کوانتایل خطی" if quantreg else "خطی/منظم‌شده"
    extra = "؛ علاوه‌بر آن `pre_holiday_x_block_len` (فیچر تقریباً ثابت، <۱٪ تغییرپذیری) حذف شده" if quantreg else ""
    return (
        f"بند 7.5.3، ردیف «{bucket}». پایه: `FS_full_A` (فاز ۵؛ خودش پایه‌های فوریه + "
        f"برهم‌کنش‌های صریح + هرس VIF دما را دارد). تغییرات: حذف `dow` خام (VIF=∞ با "
        f"پایه‌های فوریه‌اش) + افزودن `log_res_sq`{extra}.\n\n"
        f"**{len(feature_cols)} ستون نهایی** (پیش از یک‌هات‌کردن دسته‌ای‌ها): "
        f"`{', '.join(feature_cols)}`.\n\n"
        "⚠️ خوشه‌ی هم‌خط دیگر (`*_expanding_rate`/`*_shrunk_rate`، VIF ۲۰۰۰-۱۰۰۰۰) عمداً "
        "حذف نشد — طبق بند 7.5.4، انتخاب فیچر این خانواده «مسیر L1» است: خودِ منظم‌سازی "
        "مدل باید این افزونگی را حل کند، نه هرس دستی پیشینی (که در آزمایش، فیچر قوی "
        "`log_res` را هم حذف می‌کرد)."
    )


def render_step2_data_levels(model_id: str) -> str:
    spec = MODEL_REGISTRY.get(model_id)
    if spec is None:
        return "_مدل در رجیستری ثبت نشده._"
    lines = [f"سطح اجراشده: **{', '.join(spec.levels)}** (بند 7.1.1)."]
    if spec.incompatible_levels:
        lines.append(f"ناسازگار با: `{', '.join(spec.incompatible_levels)}` (دلیل در بند 7.27).")
    return " ".join(lines)


def render_step3_target_mapping(model_id: str, tau: float | None = None) -> str:
    from src.models.axes import TUNING_TAU

    spec = MODEL_REGISTRY.get(model_id)
    if spec is None:
        return "_مدل در رجیستری ثبت نشده._"
    t = tau if tau is not None else TUNING_TAU
    return (
        f"هدف خام: `rho` (نرخ عدم‌دریافت، بند ۴ سند تعریف مسئله). سطح کوانتایل: "
        f"**τ={t}** (بند 7.3، ثابت پروژه). مسیر برآورد کوانتایل: **{spec.quantile_route}** "
        f"— {_QUANTILE_ROUTE_DESC.get(spec.quantile_route, '')}"
    )


def render_step6_preprocessing() -> str:
    return (
        "``src/models/families/common.py::design_matrix`` — میانه‌گذاری عددی (از train) + "
        "یک‌هات دسته‌ای (`OneHotEncoder(handle_unknown='ignore')`, fit روی train، دسته‌ی "
        "دیده‌نشده در test → همه صفر). مقیاس‌بندی (`StandardScaler`) فقط برای مدل‌های "
        "منظم‌شده/فاصله‌محور اعمال می‌شود (Ridge/Lasso/EN/QuantReg/…)، نه GLMها که به مقیاس "
        "حساس نیستند. بدون شکاف رمضان در این سطح (L1 سلولی، نه سری زمانی)."
    )


def render_step7_hyperparam_space(model_id: str, family: str) -> str:
    """جدول از روی مقادیر واقعاً کاوش‌شده در مطالعه — نه فقط تعریف نظری فضا."""
    study = _load_study(model_id, family)
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not trials:
        return "_هنوز trial کاملی موجود نیست._"
    params = sorted({k for t in trials for k in t.params})
    lines = ["| پارامتر | دامنه‌ی کاوش‌شده | تعداد مقدار یکتا | نوع |", "|---|---|---|---|"]
    for p in params:
        vals = [t.params[p] for t in trials if p in t.params]
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
            lines.append(f"| `{p}` | {min(vals):.4g} … {max(vals):.4g} | {len(set(vals))} | پیوسته |")
        else:
            uniq = sorted(set(map(str, vals)))
            lines.append(f"| `{p}` | {', '.join(uniq)} | {len(uniq)} | دسته‌ای |")
    lines.append("")
    lines.append(f"منبع فضای رسمی و بازه‌ی نظری: `src/models/families/f01_linear.py::_space_{model_id}` "
                 "(دکوراتور `register_space`، `src/models/spaces.py`).")
    return "\n".join(lines)


def render_step8_search_strategy(result: dict) -> str:
    return (
        f"روش: **Optuna TPE** (بند 7.6.1، پیش‌فرض S2) · تعداد trial: **{result['n_trials']}** "
        f"(بودجه‌ی جدول 7.6.2 برای {result['n_hyperparams']} هایپرپارامتر مؤثر) · seed: 42 · "
        f"بدون pruner (فضای کوچک؛ هر trial خودش یا سریع یا ذاتاً کند است، هرس‌کردن زودهنگام "
        f"معنا ندارد) · fold: هر ۵ (بند 7.3.1) · زمان کل: **{result['seconds']:.0f} ثانیه** "
        f"({result['n_fail']} trial شکست‌خورده از {result['n_trials']})."
    )


def render_step9_convergence(result: dict) -> str:
    history = result["trial_history"]
    finite = [v for _, v in history if np.isfinite(v)]
    if not finite:
        return "_هیچ trial موفقی برای رسم منحنی وجود ندارد — همگرایی قابل‌سنجش نیست._"
    best_so_far = list(np.minimum.accumulate(finite))
    tail_n = max(1, int(len(best_so_far) * 0.25))
    tail = best_so_far[-tail_n:]
    improvement = (tail[0] - tail[-1]) / tail[0] if tail[0] > 0 else 0.0
    verdict = "بله ✅" if result["converged"] else "نه ⚠️"
    note = ("کمتر از آستانه‌ی ۱٪ — مدل به سقف خودش رسید (قاعده‌ی A6)."
           if result["converged"] else
           "بالای آستانه‌ی ۱٪ — با بودجه‌ی بیشتر احتمالاً بهتر می‌شد؛ در بودجه‌ی فعلی سند متوقف شد.")
    return (
        f"بهبود best-so-far در ۲۵٪ پایانی trialها: **{improvement:.2%}**. {note}\n\n"
        f"**همگرایی A6: {verdict}**\n\n"
        f"بهترین-تاکنون: شروع=`{best_so_far[0]:.5f}` → پایان=`{best_so_far[-1]:.5f}` "
        f"(روی {len(finite)} trial موفق از {result['n_trials']}).\n\n"
        "| trial | pinball_mean بهترین‌تاکنون |\n|---|---|\n" +
        "\n".join(f"| {i} | {v:.5f} |" for i, v in enumerate(best_so_far)
                 if i % max(1, len(best_so_far) // 10) == 0 or i == len(best_so_far) - 1)
    )


def render_step10_importance(model_id: str, family: str) -> str:
    study = _load_study(model_id, family)
    n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    if n_complete < 3:
        return "_کمتر از ۳ trial کامل — fANOVA معنادار نیست._"
    try:
        imp = optuna.importance.get_param_importances(study)
    except Exception as e:
        return f"_fANOVA قابل‌محاسبه نبود ({type(e).__name__}: {e}) — با دست بررسی شود._"
    if not imp:
        return "_مدل هیچ هایپرپارامتر آزادی ندارد (فضای خالی) — تحلیل حساسیت بی‌معناست._"
    lines = ["| پارامتر | اهمیت (fANOVA) |", "|---|---|"]
    for k, v in imp.items():
        lines.append(f"| `{k}` | {v:.3f} |")
    top = next(iter(imp))
    lines.append(f"\n⭐ **`{top}`** مهم‌ترین پارامتر است (اهمیت={imp[top]:.3f}).")
    return "\n".join(lines)


def render_step12_quantile_extraction(model_id: str) -> str:
    spec = MODEL_REGISTRY.get(model_id)
    if spec is None:
        return "_مدل در رجیستری ثبت نشده._"
    route = spec.quantile_route
    return f"مسیر **{route}** (بند 7.23) — {_QUANTILE_ROUTE_DESC.get(route, '')}"


def _load_refit_context(model_id: str, family: str):
    """بارگذاری مشترک برای گام‌های ۱۱/۱۳ که نیازمند بازبرازش واقعی‌اند — یک‌بار داده و
    fold و بهترین هایپرپارامتر را می‌خواند، دو بار کد تکرار نمی‌شود.

    برمی‌گرداند: ``(fit_fn, folds, best_hyperparams)`` یا ``None`` اگر هنوز trial
    موفقی موجود نباشد.
    """
    from src.config import set_global_seed
    from src.cv import DATE_COL, load_cv_folds
    from src.features.build import FEATURES_A_PATH

    result = load_s2_result(model_id, family)
    if not result.get("best_hyperparams") and result.get("n_hyperparams", 0) > 0:
        return None

    mod = importlib.import_module(_FAMILY_MODULES[family])
    fit_fn = mod.MODELS[model_id]

    set_global_seed()
    df = pd.read_parquet(FEATURES_A_PATH).sort_values(DATE_COL).reset_index(drop=True)
    fold_meta, _ = load_cv_folds()
    folds = []
    for f in fold_meta:
        tr_mask, te_mask = f.masks(df[DATE_COL])
        folds.append((df.loc[tr_mask], df.loc[te_mask]))
    return fit_fn, folds, result["best_hyperparams"]


def render_step11_fit_diagnosis(model_id: str, family: str, tau: float | None = None) -> str:
    """بازبرازش واقعی + شکاف pinball train/test روی هر ۵ fold رسمی
    (``src/models/fit_diagnosis.py``) — گام ۱۱ اجباری کارت مدل، بند 7.4."""
    from src.models import fit_diagnosis
    from src.models.axes import TUNING_TAU

    ctx = _load_refit_context(model_id, family)
    if ctx is None:
        return "_هنوز trial موفقی برای بهترین هایپرپارامتر موجود نیست._"
    fit_fn, folds, hp = ctx
    t = tau if tau is not None else TUNING_TAU
    rows = fit_diagnosis.train_test_gap(fit_fn, folds, t, hp)
    return fit_diagnosis.render_step11(rows)


def render_step13_calibration(model_id: str, family: str, tau: float | None = None) -> str:
    """بازبرازش واقعی مدل با بهترین هایپرپارامتر S2 روی هر ۵ fold رسمی + سنجش پوشش
    تجربی (``src/models/calibration.py``) — گام ۱۳ اجباری کارت مدل، بند 7.4."""
    from src.models import calibration
    from src.models.axes import TUNING_TAU

    ctx = _load_refit_context(model_id, family)
    if ctx is None:
        return "_هنوز trial موفقی برای بهترین هایپرپارامتر موجود نیست._"
    fit_fn, folds, hp = ctx
    t = tau if tau is not None else TUNING_TAU
    oof = calibration.oof_predictions(fit_fn, folds, t, hp)
    return calibration.render_step13(oof, t)


@functools.lru_cache(maxsize=4)
def _b3_baseline_5fold(family: str, level: str) -> float:
    """مرجع B3 روی هر ۵ fold رسمی — مستقل از اتمام کل S2 محاسبه می‌شود (بند 7.6.3
    نتیجه‌ی نهایی همان مقدار را در ``_baseline_B3`` می‌نویسد، ولی آن فقط بعد از اتمام
    **کل** خانواده نوشته می‌شود؛ گام ۱۴ نباید منتظرش بماند — این تابع سبک است چون فقط
    کوانتایل تجربی خط پایه را می‌سنجد، نه هیچ برازش مدلی)."""
    from src.cv import DATE_COL, load_cv_folds
    from src.features.build import FEATURES_A_PATH
    from src.models.s2_runner import baseline_reference_5fold

    df = pd.read_parquet(FEATURES_A_PATH).sort_values(DATE_COL).reset_index(drop=True)
    fold_meta, _ = load_cv_folds()
    folds = []
    for f in fold_meta:
        tr_mask, te_mask = f.masks(df[DATE_COL])
        folds.append((df.loc[tr_mask], df.loc[te_mask]))
    return baseline_reference_5fold(folds)


_COVERAGE_RE = re.compile(r"پوشش کلی: ([\d.]+)\*\* \(اسمی τ=[\d.]+, شکاف=([+-][\d.]+)")
_RATIO_RE = re.compile(r"نسبت pinball\(test\)/pinball\(train\) روی ۵ fold: ([\d.]+)\*\*")


def render_step14_summary(model_id: str, family: str, card: cards.ModelCard) -> str:
    """سنتز نهایی — از نتیجه‌ی S2 (بند ۸/۹) + مرجع B3 + آنچه گام‌های ۱۱/۱۳ **همین کارت**
    (اگر پیش‌تر پر شده باشند) قبلاً محاسبه کرده‌اند می‌سازد؛ دوباره بازبرازش نمی‌کند."""
    result = load_s2_result(model_id, family)
    if not np.isfinite(result.get("best_pinball", float("nan"))):
        return "_هنوز نتیجه‌ی S2 معتبری موجود نیست._"

    b3 = _b3_baseline_5fold(family, "L1")
    beat_b3 = result["best_pinball"] < b3
    lines = [
        f"**pinball نهایی (بهترین trial، ۵ fold): {result['best_pinball']:.5f}** در برابر "
        f"مرجع B3={b3:.5f} — {'🎯 بهتر از B3' if beat_b3 else '❌ هنوز B3 را نبرده'}.",
        f"همگرایی (A6): {'✅' if result['converged'] else '⚠️'} · پایداری (بند 7.6.3): "
        f"{result['stable_top10pct_folds']}/5 fold · {result['n_trials']} trial، "
        f"{result['n_fail']} شکست.",
    ]

    cov_m = _COVERAGE_RE.search(card.sections.get(13, ""))
    if cov_m:
        cov, gap = float(cov_m.group(1)), float(cov_m.group(2))
        cal_verdict = "قابل‌قبول" if abs(gap) < 0.05 else "نیازمند کالیبراسیون متعامد (خ۱۳) پیش از استفاده‌ی عملیاتی"
        lines.append(f"کالیبراسیون (گام ۱۳): پوشش کلی={cov:.4f} (شکاف={gap:+.4f}) — {cal_verdict}.")

    ratio_m = _RATIO_RE.search(card.sections.get(11, ""))
    if ratio_m:
        ratio = float(ratio_m.group(1))
        fit_verdict = "بدون نشانه‌ی بیش‌برازش قابل‌توجه" if ratio <= 1.20 else "نشانه‌ی بیش‌برازش"
        lines.append(f"تشخیص برازش (گام ۱۱): نسبت pinball test/train={ratio:.3f} — {fit_verdict}.")

    stable = result["stable_top10pct_folds"] >= 3  # آستانه‌ی بند 7.6.3
    cal_ok = not cov_m or abs(float(cov_m.group(2))) < 0.10
    if beat_b3 and result["converged"] and cal_ok and stable:
        verdict, why = "✅ کاندید ورود به S3 (حساسیت τ + ترکیب)", "برنده‌ی B3 با همگرایی، پایداری و کالیبراسیون قابل‌قبول"
    elif beat_b3 and result["converged"] and cal_ok and not stable:
        verdict, why = ("⚠️ برد حاشیه‌ای — نیازمند تأیید آماری پیش از S3",
                        f"بهتر از B3 است ولی پایداری پایین (بند 7.6.3: فقط {result['stable_top10pct_folds']}/5 fold "
                        "در ۱۰٪ برتر) — احتمال شانسی‌بودن برد را نمی‌شود رد کرد؛ آزمون Diebold-Mariano لازم است "
                        "(یافته‌ی ۷ doc/progress: S1 هم مشابه همین را برای adaptive_lasso نشان داد که در S2 تکرار نشد)")
    else:
        verdict, why = "⚠️ نیازمند بررسی بیشتر پیش از S3", "حداقل یکی از معیارهای برد/همگرایی/کالیبراسیون هنوز برآورده نشده"
    lines.append(f"\n**توصیه:** {verdict} — {why}.")
    return "\n".join(lines)


def draft_card(model_id: str, family: str = "F01", feature_cols: list[str] | None = None,
               quantreg: bool = False, include_refit_diagnostics: bool = False) -> cards.ModelCard:
    """کارت را با بخش‌های داده‌محور/مکانیکی (۱،۲،۳،۴،۵،۶،۷،۸،۹،۱۰،۱۲) پر می‌کند؛ بقیه راهنما می‌مانند.

    ``feature_cols`` باید از ``f01_linear._feature_cols_s2()``/``_feature_cols_s2_quantreg()``
    بیاید — اینجا وابستگی مستقیم به آن ماژول ندارد تا برای خانواده‌های دیگر هم قابل‌استفاده بماند.

    ``include_refit_diagnostics=True`` گام‌های ۱۱ و ۱۳ (هر دو اجباری) را هم با یک
    بازبرازش واقعی مشترک (``_load_refit_context``) پر می‌کند — پیش‌فرض خاموش چون هزینه‌اش
    برای مدل‌های کند (رگرسیون کوانتایل ترکیبی) قابل‌توجه است؛ برای آن‌ها
    ``render_step11_fit_diagnosis``/``render_step13_calibration`` جداگانه و آگاهانه صدا زده شود.
    """
    result = load_s2_result(model_id, family)
    card = cards.ModelCard(model_id)

    card.set_section(1, render_step1_theoretical_position(model_id, family))
    card.set_section(2, render_step2_data_levels(model_id))
    card.set_section(3, render_step3_target_mapping(model_id))
    card.set_section(4, render_step4_preflight_tests(model_id, family))
    if feature_cols is not None:
        card.set_section(5, render_step5_feature_set(model_id, feature_cols, quantreg, family))
    card.set_section(6, render_step6_preprocessing())
    card.set_section(7, render_step7_hyperparam_space(model_id, family))
    card.set_section(8, render_step8_search_strategy(result))
    card.set_section(9, render_step9_convergence(result))
    card.set_section(10, render_step10_importance(model_id, family))
    card.set_section(12, render_step12_quantile_extraction(model_id))
    if include_refit_diagnostics:
        card.set_section(11, render_step11_fit_diagnosis(model_id, family))
        card.set_section(13, render_step13_calibration(model_id, family))

    # بقیه (۱۴ — و ۱۱/۱۳ اگر include_refit_diagnostics=False) عمداً دست‌نخورده می‌مانند —
    # ModelCard.to_markdown() خودش جای‌گزین «_ثبت نشده._» می‌گذارد، پس نیازی به یادداشت TODO
    # دستی اینجا نیست؛ card.missing_mandatory() برای پرس‌وجوی برنامه‌ای «چه چیزی هنوز مانده» کافی است.
    return card
