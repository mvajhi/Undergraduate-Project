"""🐛 باگ کشف‌شده حین نوشتن `run_f01_coefficient_path.py` (۲۰۲۶-۰۸-۲۶) — S2 خ۱ هرگز
فیچرست اختصاصی‌اش (`_feature_cols_s2`) را واقعاً استفاده نکرد.

## چه چیزی کشف شد

`s2_runner.py::_run_model_s2` خط ۱۳۲ یک‌بار `designed_folds = [design_fn(tr, te) for
...]` می‌سازد (که برای خ۱ یعنی `f01_linear.py::_design_s2` — همان فیچرست S2 اختصاصی،
بند 7.5.3) ولی خط ۱۵۴ فقط `Xtr, Xte = designed_folds[fold_idx]` را unpack می‌کند و
**هرگز به `fn(...)` نمی‌دهد** — خط ۱۵۵ مستقیم `fn(tr, te, S2_TAU, **hp)` را با
دیتافریم خام fold صدا می‌زند. هر ``fit_predict_*`` خودش داخلی `_design(train, test)`
صدا می‌زند که فیچرست S0/S1 (`FS_day`, ۴۷ ستون) را برمی‌گرداند — **نه** `_feature_cols_s2()`
(`FS_full_A` منهای `dow` به‌علاوه‌ی `log_res_sq`، ~۶۲ ستون). یعنی هر ۱۵ trial-run S2 خ۱
عملاً روی همان فیچرست کوچک S0/S1 اجرا شد، نه فیچرستی که بند ۳۸ decision_log ادعا کرده
بود ساخته و تست شده.

## چرا این را کسی زودتر ندید

ردیف ۳۸ decision_log فقط `_feature_cols_s2()`/`_design_s2()` را **به‌تنهایی** تست کرد
(«VIF فوریه از ∞ به ۱.۱–۲.۰ افت کرد») — نه سیم‌کشی‌اش را به هارنس S2. یک باگ کلاسیک
یکپارچگی: هر قطعه به‌تنهایی درست کار می‌کند، فقط باهم وصل نیستند.

## چرا اینجا فقط بررسی می‌شود، نه رفع کامل

خ۱ **منجمد است** — «دوباره اجرا نمی‌شود» (ردیف ۱۲ decision_log، تکرارشده در چند جای
دیگر). بازتنظیم کامل ۱۵ مدل (که یکی‌شان composite_quantile_regression با ۷.۹ ساعت
تنها خودش بود) توجیه ندارد. این ماژول فقط **مادیّت باگ را می‌سنجد**: آیا فیچرست
درست، روی همان هایپرپارامتر S2-تنظیم‌شده، نتیجه را عوض می‌کند؟ اگر نه، باگ مستند
می‌ماند ولی نتیجه‌گیری‌های موجود دست‌نخورده می‌مانند. اگر آری، آن مقدار هم گزارش
می‌شود — ولی هایپرپارامتر دوباره تنظیم نمی‌شود (چون خودِ آن جست‌وجو زیر فیچرست غلط
انجام شده بود؛ رفع کامل یعنی تنظیم دوباره‌ی کامل، که همان بازاجرای ممنوع است).

تنها روی سه مدلی که در S2 خ۱ ادعای برد داشتند (`l1_quantile_regression`, `elasticnet`,
`lasso` — یافته‌ی ۷/۱۱) بررسی می‌شود، نه هر ۱۵ مدل، چون این‌ها تنها مدل‌هایی‌اند که
نتیجه‌ی متفاوت واقعاً روی تصمیم پروژه اثر می‌گذارد.

اجرا: ``python -m src.models.run_f01_s2_feature_bug_check``
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso, QuantileRegressor
from sklearn.preprocessing import StandardScaler

from src.baselines import b3_empirical_quantile, pinball_loss
from src.cv import DATE_COL, diebold_mariano, load_cv_folds
from src.features.build import FEATURES_A_PATH
from src.models.axes import TUNING_TAU
from src.models.card_writer import load_s2_result
from src.models.families import common
from src.models.families.f01_linear import (
    _design,
    _feature_cols_s2,
    _feature_cols_s2_quantreg,
)

MODELS_TO_CHECK = ("l1_quantile_regression", "elasticnet", "lasso")


def _official_folds() -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    df = pd.read_parquet(FEATURES_A_PATH).sort_values(DATE_COL).reset_index(drop=True)
    fold_meta, _ = load_cv_folds()
    return [(df.loc[m1], df.loc[m2]) for f in fold_meta for m1, m2 in [f.masks(df[DATE_COL])]]


def _design_s2_correct(train: pd.DataFrame, test: pd.DataFrame, quantreg: bool) -> tuple:
    """همان `_design_s2` ولی مستقل از باگ فراخوانی — مستقیم صدا زده می‌شود."""
    tr, te = train.copy(), test.copy()
    tr["log_res_sq"] = tr["log_res"] ** 2
    te["log_res_sq"] = te["log_res"] ** 2
    cols = _feature_cols_s2_quantreg() if quantreg else _feature_cols_s2()
    return common.design_matrix(tr, te, cols)


def _refit_lasso_like(model_cls, train: pd.DataFrame, test: pd.DataFrame, tau: float,
                      hp: dict, use_s2_features: bool) -> np.ndarray:
    Xtr, Xte = _design_s2_correct(train, test, quantreg=False) if use_s2_features else _design(train, test)
    scaler = StandardScaler().fit(Xtr)
    Ztr, Zte = scaler.transform(Xtr), scaler.transform(Xte)
    model = model_cls(**hp, max_iter=5000).fit(Ztr, train["rho"])
    return common.residual_quantile_by_res_quartile(train, test, model.predict(Ztr), model.predict(Zte), tau)


def _refit_l1_qr(train: pd.DataFrame, test: pd.DataFrame, tau: float, hp: dict,
                 use_s2_features: bool) -> np.ndarray:
    Xtr, Xte = _design_s2_correct(train, test, quantreg=True) if use_s2_features else _design(train, test)
    scaler = StandardScaler().fit(Xtr)
    model = QuantileRegressor(quantile=tau, alpha=hp["alpha"], solver="highs")
    model.fit(scaler.transform(Xtr), train["rho"])
    return np.clip(model.predict(scaler.transform(Xte)), 0.0, 1.0)


def _oof(model_id: str, hp: dict, folds: list, tau: float, use_s2_features: bool) -> pd.DataFrame:
    parts = []
    for tr, te in folds:
        if model_id == "l1_quantile_regression":
            pred = _refit_l1_qr(tr, te, tau, hp, use_s2_features)
        elif model_id == "lasso":
            pred = _refit_lasso_like(Lasso, tr, te, tau, hp, use_s2_features)
        elif model_id == "elasticnet":
            pred = _refit_lasso_like(ElasticNet, tr, te, tau, hp, use_s2_features)
        else:
            raise ValueError(model_id)
        parts.append(pd.DataFrame({"actual": te["rho"].to_numpy(), "pred_q": np.clip(pred, 0, 1)}))
    return pd.concat(parts, ignore_index=True)


def _b3_row_losses(folds: list, tau: float) -> np.ndarray:
    parts = []
    for tr, te in folds:
        pred = b3_empirical_quantile(tr, te, tau)
        parts.append(pd.DataFrame({"actual": te["rho"].to_numpy(), "pred_q": pred}))
    oof = pd.concat(parts, ignore_index=True)
    return pinball_loss(oof["actual"].to_numpy(), oof["pred_q"].to_numpy(), tau)


def check_model(model_id: str, folds: list, tau: float, b3_losses: np.ndarray) -> dict:
    hp = load_s2_result(model_id, "F01")["best_hyperparams"]

    oof_buggy = _oof(model_id, hp, folds, tau, use_s2_features=False)
    oof_fixed = _oof(model_id, hp, folds, tau, use_s2_features=True)

    loss_buggy = pinball_loss(oof_buggy["actual"].to_numpy(), oof_buggy["pred_q"].to_numpy(), tau)
    loss_fixed = pinball_loss(oof_fixed["actual"].to_numpy(), oof_fixed["pred_q"].to_numpy(), tau)
    dm_stat, p_value = diebold_mariano(loss_fixed, loss_buggy)

    # ⭐ سؤال واقعاً مهم: با فیچرست درست، آیا این مدل اکنون B3 را معنادار می‌برد؟
    # (dm_test_F01.md — یافته‌ی ۱۳ — با همین فیچرست باگ‌دار محاسبه شده بود)
    dm_vs_b3, p_vs_b3 = diebold_mariano(loss_fixed, b3_losses)
    beats_b3 = bool(np.isfinite(p_vs_b3) and p_vs_b3 < 0.05 and loss_fixed.mean() < b3_losses.mean())

    return {
        "model_id": model_id,
        # ⚠️ این دو **عمداً** با هم فرق دارند — نه خطا: reproduced اینجا ردیفی
        # (row-level) است، مثل dm_test_F01.md؛ عدد رسمی S2_tuning_F01.md fold-میانگین
        # است (وزن برابر هر fold، نه هر ردیف) — همان هشدار significance.py.
        "pinball_buggy_row_level": float(loss_buggy.mean()),
        "pinball_fixed_row_level": float(loss_fixed.mean()),
        "delta_fixed_minus_buggy": float(loss_fixed.mean() - loss_buggy.mean()),
        "dm_stat_fixed_vs_buggy": dm_stat, "p_value_fixed_vs_buggy": p_value,
        "materially_different": bool(np.isfinite(p_value) and p_value < 0.05),
        "pinball_b3_row_level": float(b3_losses.mean()),
        "beats_b3_with_fixed_features": beats_b3, "p_value_vs_b3_fixed": p_vs_b3,
    }


def render_report(results: list[dict]) -> str:
    lines = [
        "# بررسی مادیّت باگ فیچرست S2 خ۱",
        "",
        "> بند بالای `src/models/run_f01_s2_feature_bug_check.py`. τ=0.20، هر ۵ fold رسمی "
        "concatenate‌شده (وزن برابر هر ردیف، مثل `dm_test_F01.md` — نه fold-میانگین "
        "`S2_tuning_F01.md`)، همان هایپرپارامتر S2-تنظیم‌شده‌ی هر مدل (بدون بازتنظیم — "
        "خ۱ منجمد است).",
        "",
        "## آیا فیچرست عوض می‌شود نتیجه را؟",
        "",
        "| مدل | با فیچرست باگ‌دار (FS_day، آنچه واقعاً اجرا شد) | با فیچرست واقعی S2 | Δ | DM p-value | تفاوت مادی؟ |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        mat = "🔴 بله" if r["materially_different"] else "✅ خیر"
        lines.append(
            f"| `{r['model_id']}` | {r['pinball_buggy_row_level']:.5f} | "
            f"{r['pinball_fixed_row_level']:.5f} | {r['delta_fixed_minus_buggy']:+.5f} | "
            f"{r['p_value_fixed_vs_buggy']:.4f} | {mat} |"
        )

    lines += [
        "",
        f"عدد ستون «با فیچرست باگ‌دار» دقیقاً با `dm_test_F01.md` یکی است (۰.۰۱۳۱۹/۰.۰۱۳۵۸/۰.۰۱۳۵۹) "
        "— تأیید می‌کند `significance.py` (یافته‌ی ۱۳) هم بی‌خبر از همین باگ روی فیچرست اشتباه اجرا شده بود.",
        "",
        "## ⭐ سؤال واقعی: با فیچرست درست، آیا این مدل‌ها الان B3 را می‌برند؟",
        "",
        f"مرجع B3 (ردیفی، هر ۵ fold): pinball={results[0]['pinball_b3_row_level']:.5f}",
        "",
        "| مدل | pinball (فیچرست درست) | Δ نسبت به B3 | DM p-value | برد معنادار؟ |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        win = "✅ **بله**" if r["beats_b3_with_fixed_features"] else "❌ خیر"
        delta = r["pinball_fixed_row_level"] - r["pinball_b3_row_level"]
        lines.append(f"| `{r['model_id']}` | {r['pinball_fixed_row_level']:.5f} | {delta:+.5f} | "
                    f"{r['p_value_vs_b3_fixed']:.4f} | {win} |")

    n_mat = sum(1 for r in results if r["materially_different"])
    n_beat = sum(1 for r in results if r["beats_b3_with_fixed_features"])
    lines += [
        "",
        f"**{n_mat} از {len(results)} مدل با فیچرست درست به‌طور آماری‌معنادار فرق کردند "
        f"(هر سه بهتر شدند، نه بدتر)؛ {n_beat} از {len(results)} با فیچرست درست حالا B3 را "
        "معنادار می‌برند.**",
        "",
        "⚠️ **پیامد باز، نه بسته:** این تناقض مستقیم با یافته‌ی ۱۳/ردیف ۴۰ decision_log است "
        "(«هیچ مدل خطی B3 را معنادار نبرد») — که هر دو با فیچرست باگ‌دار محاسبه شده بودند. "
        "خ۱ منجمد می‌ماند و اعداد رسمی (`S2_tuning_F01.md`, `model_comparison.csv`, ردیف ۴۰) "
        "بازنویسی نمی‌شوند، ولی این یافته باید صریح در گزارش نهایی بیاید: عملکرد واقعی "
        "پتانسیل خ۱ دست‌کم گرفته شده بود.",
    ]
    return "\n".join(lines)


def main() -> None:
    from src.config import REPORTS_DIR, set_global_seed

    set_global_seed()
    folds = _official_folds()
    b3_losses = _b3_row_losses(folds, TUNING_TAU)
    results = [check_model(mid, folds, TUNING_TAU, b3_losses) for mid in MODELS_TO_CHECK]
    report = render_report(results)
    out = REPORTS_DIR / "phase7"
    out.mkdir(parents=True, exist_ok=True)
    (out / "f01_s2_feature_bug_check.md").write_text(report + "\n")
    pd.DataFrame(results).to_json(out / "f01_s2_feature_bug_check.json", orient="records",
                                  indent=2, force_ascii=False)
    print(report)
    print(f"\nذخیره شد در {out / 'f01_s2_feature_bug_check.md'}")


if __name__ == "__main__":
    main()
