"""بند 7.4 — نویسنده‌ی نیمه‌خودکار کارت مدل.

بخش‌های **داده‌محور** (۵، ۷، ۸، ۹، ۱۰، ۱۲) مستقیماً از نتیجه‌ی S2 (`s2_runner`) و مطالعه‌ی
Optuna پایدارشده (`optuna_studies/{model_id}.db`) ساخته می‌شوند — دستی نوشته نمی‌شوند تا با
تغییر کد از هم جدا نیفتند. بخش‌های **روایی** (۱، ۲، ۳، ۴، ۶، ۱۱، ۱۳، ۱۴) که به قضاوت
دامنه‌ای نیاز دارند، با ``draft_card`` فقط یک چارچوب/راهنما می‌گیرند و باید تکمیل شوند —
جز ۲، ۳، ۶ که برای خ۱ به‌اندازه‌ی کافی مکانیکی‌اند (سطح L1 ثابت، نگاشت هدف از
``quantile_route``، پیش‌پردازش از ``common.design_matrix``) و کامل پر می‌شوند.

گام ۱۳ (کالیبراسیون) با ``render_step13_calibration`` جدا محاسبه می‌شود — چون برخلاف
بخش‌های بالا نیازمند **بازبرازش واقعی** مدل روی هر ۵ fold با بهترین هایپرپارامتر S2 است
(``src/models/calibration.py``)، نه فقط خواندن نتیجه‌ی ذخیره‌شده؛ برای مدل‌های کند
(مثل رگرسیون کوانتایل ترکیبی) این هزینه‌ی قابل‌توجهی دارد، پس در ``draft_card`` با
پرچم ``include_calibration=False`` پیش‌فرض خاموش است.
"""

import importlib
import json

import numpy as np
import optuna
import pandas as pd

from src.models import cards
from src.models.registry import MODELS as MODEL_REGISTRY
from src.models.s2_runner import PHASE7_DIR, study_storage_url

optuna.logging.set_verbosity(optuna.logging.WARNING)

#: خانواده → مسیر دات‌دار ماژول — برای بارگذاری ``MODELS``/``QUANTREG_MODEL_IDS`` در
#: ``render_step13_calibration`` بدون وابستگی مستقیم به یک خانواده‌ی خاص.
_FAMILY_MODULES = {"F01": "src.models.families.f01_linear"}

_QUANTILE_ROUTE_DESC = {
    "Q1": "بومی — مدل مستقیماً Pinball@τ را کمینه می‌کند (رگرسیون کوانتایل).",
    "Q2": "از توزیع پارامتری برازش‌شده — کوانتایل با معکوس CDF توزیع فرضی محاسبه می‌شود.",
    "Q3": "از توزیع تجربی باقیمانده — پیش‌بینی میانگین + آفست کوانتایل باقیمانده به تفکیک چارک Res.",
}


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


def render_step5_feature_set(model_id: str, feature_cols: list[str], quantreg: bool) -> str:
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


def render_step13_calibration(model_id: str, family: str, tau: float | None = None) -> str:
    """بازبرازش واقعی مدل با بهترین هایپرپارامتر S2 روی هر ۵ fold رسمی + سنجش پوشش
    تجربی (``src/models/calibration.py``) — گام ۱۳ اجباری کارت مدل، بند 7.4."""
    from src.config import set_global_seed
    from src.cv import DATE_COL, load_cv_folds
    from src.features.build import FEATURES_A_PATH
    from src.models import calibration
    from src.models.axes import TUNING_TAU

    result = load_s2_result(model_id, family)
    if not result.get("best_hyperparams") and result.get("n_hyperparams", 0) > 0:
        return "_هنوز trial موفقی برای بهترین هایپرپارامتر موجود نیست._"

    mod = importlib.import_module(_FAMILY_MODULES[family])
    fit_fn = mod.MODELS[model_id]

    set_global_seed()
    df = pd.read_parquet(FEATURES_A_PATH).sort_values(DATE_COL).reset_index(drop=True)
    fold_meta, _ = load_cv_folds()
    folds = []
    for f in fold_meta:
        tr_mask, te_mask = f.masks(df[DATE_COL])
        folds.append((df.loc[tr_mask], df.loc[te_mask]))

    t = tau if tau is not None else TUNING_TAU
    oof = calibration.oof_predictions(fit_fn, folds, t, result["best_hyperparams"])
    return calibration.render_step13(oof, t)


def draft_card(model_id: str, family: str = "F01", feature_cols: list[str] | None = None,
               quantreg: bool = False, include_calibration: bool = False) -> cards.ModelCard:
    """کارت را با بخش‌های داده‌محور/مکانیکی (۲،۳،۵،۶،۷،۸،۹،۱۰،۱۲) پر می‌کند؛ بقیه راهنما می‌مانند.

    ``feature_cols`` باید از ``f01_linear._feature_cols_s2()``/``_feature_cols_s2_quantreg()``
    بیاید — اینجا وابستگی مستقیم به آن ماژول ندارد تا برای خانواده‌های دیگر هم قابل‌استفاده بماند.

    ``include_calibration=True`` گام ۱۳ (اجباری) را هم با بازبرازش واقعی پر می‌کند —
    پیش‌فرض خاموش چون هزینه‌اش برای مدل‌های کند (رگرسیون کوانتایل ترکیبی) قابل‌توجه است؛
    برای آن‌ها ``render_step13_calibration`` جداگانه و آگاهانه صدا زده شود.
    """
    result = load_s2_result(model_id, family)
    card = cards.ModelCard(model_id)

    card.set_section(2, render_step2_data_levels(model_id))
    card.set_section(3, render_step3_target_mapping(model_id))
    if feature_cols is not None:
        card.set_section(5, render_step5_feature_set(model_id, feature_cols, quantreg))
    card.set_section(6, render_step6_preprocessing())
    card.set_section(7, render_step7_hyperparam_space(model_id, family))
    card.set_section(8, render_step8_search_strategy(result))
    card.set_section(9, render_step9_convergence(result))
    card.set_section(10, render_step10_importance(model_id, family))
    card.set_section(12, render_step12_quantile_extraction(model_id))
    if include_calibration:
        card.set_section(13, render_step13_calibration(model_id, family))

    # بقیه (۱،۴،۱۱،۱۴ — و ۱۳ اگر include_calibration=False) عمداً دست‌نخورده می‌مانند — ModelCard.to_markdown() خودش
    # جای‌گزین «_ثبت نشده._» می‌گذارد، پس نیازی به یادداشت TODO دستی اینجا نیست؛
    # card.missing_mandatory() برای پرس‌وجوی برنامه‌ای «چه چیزی هنوز مانده» کافی است.
    return card
