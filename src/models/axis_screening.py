"""اسپرینت A — بند 7.9.1 بازنویسی‌شده (پیمایش محور-اول، ردیف ۳۷ decision_log).

به‌جای پیمایش «خانواده اول» (که در خ۱ به هزینه‌ی ۸.۷۶ ساعت-هسته و بی‌اثر ماندن روی
محورهای پرثمر تمام شد)، هر محور بند 7.9.1 با **دو کاوشگر ارزان** (یک خطی + یک
LightGBM-quantile پیش‌فرض) روی لنگر (flat/L1/global/rho/none) آزموده می‌شود. اگر هر دو
کاوشگر یک جهت را نشان دادند، نتیجه قابل‌تعمیم است (بند ۴ سند تصمیم ۳۷).

⚠️ **این ماژول فقط محورهایی را می‌پوشاند که با دیتاست L1 موجود قابل‌آزمایش‌اند:**
`architecture`، `weighting`، `target`، `scope`. محورهای `level` (نیازمند دیتاست‌های
L2/L3/L4/L5 که هنوز ساخته نشده‌اند) و `output_aggregation` (نیازمند تجمیع چندسطحی که
هنوز موجود نیست) عمداً اینجا نیستند — بند 7.9.1 آن‌ها را به اسپرینت‌های B/C (فیچر
کوهورت L5→L1، غربالگری خانواده‌های سری‌زمانی/سلسله‌مراتبی) موکول کرده؛ این خودش
«اندازه‌گیری‌نشده» نیست، «موکول مستند» است.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import StandardScaler

from src.baselines import pinball_loss
from src.cv import diebold_mariano
from src.models.axes import TUNING_TAU
from src.models.families import common

#: بند ۳ سند تصمیم ۳۷ — کمترین تفاوت pinball که به تفاوت مالی معنادار ترجمه می‌شود
EQUIVALENCE_MARGIN_DELTA = 0.0005

#: فیچرست فشرده و مقاوم برای کاوشگر خطی — نه FS_day کامل، چون بعضی محورها (per_restaurant)
#: fit را روی زیرمجموعه‌های ۳۰-۱۰۰ ردیفی انجام می‌دهند و فضای زیاد بی‌ثبات می‌شود
_LINEAR_FEATURES = ["log_res", "dow_sin1", "dow_cos1", "week_of_semester",
                    "is_holiday_any", "is_exam_period", "days_to_next_holiday", "is_khabgah"]

#: فقط فیچرهایی که واقعاً **سطح روز**اند (ثابت درون یک روز) — برای گام ۱ محور
#: `architecture=two_stage`. ⚠️ نباید فیچر سلولی (log_res, RestaurantName, …) اینجا
#: باشد چون `drop_duplicates("date_gregorian")` مقدارش را از یک سلول دلبخواه می‌گیرد،
#: نه میانگین/نماینده‌ی واقعی روز — دقیقاً همان دامی که نسخه‌ی اول این ماژول داشت.
_DAY_LEVEL_FEATURES = ["log_daily_total_res", "dow_sin1", "dow_cos1", "week_of_semester",
                       "is_holiday_any", "is_exam_period", "days_to_next_holiday",
                       "day_shock_lag1", "is_ramadan"]


def _load_fs_day() -> list[str]:
    import json

    from src.features.build import FEATURE_SETS_PATH
    return json.loads(FEATURE_SETS_PATH.read_text())["FS_day"]


def explorer_linear(train: pd.DataFrame, test: pd.DataFrame, tau: float, *,
                    target_col: str = "rho", sample_weight: np.ndarray | None = None,
                    feature_cols: list[str] | None = None) -> np.ndarray:
    """کاوشگر خطی — QuantileRegressor روی فیچرست فشرده. ``target_col`` امکان تنظیم روی
    یک ستون تبدیل‌شده (logit/arcsine) را می‌دهد؛ خروجی هنوز در **همان فضای target_col**
    است — بازگرداندن به فضای ρ به عهده‌ی صدازننده است (چون تبدیل عکس محورمحور است).
    ``feature_cols`` برای گام ۱ محور `architecture` (فیچرهای سطح روز، نه فشرده‌ی پیش‌فرض)."""
    cols = feature_cols or _LINEAR_FEATURES
    Xtr, Xte = common.design_matrix(train, test, cols)
    scaler = StandardScaler().fit(Xtr)
    Ztr, Zte = scaler.transform(Xtr), scaler.transform(Xte)
    model = QuantileRegressor(quantile=tau, alpha=0.01, solver="highs")
    model.fit(Ztr, train[target_col].to_numpy(), sample_weight=sample_weight)
    return model.predict(Zte)


def explorer_lgbm(train: pd.DataFrame, test: pd.DataFrame, tau: float, *,
                  target_col: str = "rho", sample_weight: np.ndarray | None = None,
                  feature_cols: list[str] | None = None) -> np.ndarray:
    """کاوشگر LightGBM-quantile — پارامتر پیش‌فرض (بند ۴ سند تصمیم ۳۷)."""
    import lightgbm as lgb

    cols = feature_cols or _load_fs_day()
    tr = train.copy()
    for c in cols:
        if tr[c].dtype == object:
            tr[c] = tr[c].astype("category")
    te = test.copy()
    for c in cols:
        if c in tr.columns and str(tr[c].dtype) == "category":
            te[c] = pd.Categorical(te[c], categories=tr[c].cat.categories)

    model = lgb.LGBMRegressor(objective="quantile", alpha=tau, n_estimators=200,
                              min_child_samples=10, verbosity=-1, random_state=42)
    model.fit(tr[cols], tr[target_col].to_numpy(), sample_weight=sample_weight)
    return model.predict(te[cols])


EXPLORERS = {"linear": explorer_linear, "lgbm": explorer_lgbm}


# ---------------------------------------------------------------------------
# محور «نگاشت هدف» — تبدیل باید یک‌به‌یک و یکنوا باشد تا کوانتایل جابه‌جا شود
# ---------------------------------------------------------------------------

def _logit(p: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _arcsine_sqrt(p: np.ndarray) -> np.ndarray:
    return np.arcsin(np.sqrt(np.clip(p, 0.0, 1.0)))


def _arcsine_sqrt_inv(x: np.ndarray) -> np.ndarray:
    return np.sin(np.clip(x, 0.0, np.pi / 2)) ** 2


TARGET_TRANSFORMS = {
    "rho": (lambda p: p, lambda x: x),
    "logit_rho": (_logit, _sigmoid),
    "arcsine_sqrt_rho": (_arcsine_sqrt, _arcsine_sqrt_inv),
}


def run_target_axis(explorer_fn, folds: list, tau: float) -> dict[str, np.ndarray]:
    """برمی‌گرداند: نگاشت هر مقدار محور → بردار پیش‌بینی رویِ **فضای ρ** (بعد از تبدیل عکس)."""
    out: dict[str, list[np.ndarray]] = {name: [] for name in TARGET_TRANSFORMS}
    for tr, te in folds:
        for name, (fwd, inv) in TARGET_TRANSFORMS.items():
            tr2 = tr.copy()
            tr2["_target"] = fwd(tr2["rho"].to_numpy())
            pred_t = explorer_fn(tr2, te, tau, target_col="_target")
            out[name].append(np.clip(inv(pred_t), 0.0, 1.0))
    return {k: np.concatenate(v) for k, v in out.items()}


# ---------------------------------------------------------------------------
# محور «وزن‌دهی»
# ---------------------------------------------------------------------------

WEIGHTING_FNS = {
    "none": lambda tr: None,
    "res": lambda tr: tr["Res"].to_numpy(float),
    "sqrt_res": lambda tr: np.sqrt(tr["Res"].to_numpy(float)),
}


def run_weighting_axis(explorer_fn, folds: list, tau: float) -> dict[str, np.ndarray]:
    out: dict[str, list[np.ndarray]] = {name: [] for name in WEIGHTING_FNS}
    for tr, te in folds:
        for name, wfn in WEIGHTING_FNS.items():
            pred = explorer_fn(tr, te, tau, sample_weight=wfn(tr))
            out[name].append(np.clip(pred, 0.0, 1.0))
    return {k: np.concatenate(v) for k, v in out.items()}


# ---------------------------------------------------------------------------
# محور «دامنه‌ی مدل» — global در برابر per_cluster/per_city/per_restaurant
# ---------------------------------------------------------------------------

SCOPE_KEYS = {
    "global": None,
    "per_cluster": "is_khabgah",   # F41: خوشه‌ی k=۲ بدون‌ناظر همان مرز خوابگاه/غیرتهران را می‌سازد
    "per_city": "is_tehran",
    "per_restaurant": "RestaurantName",
}
#: زیر این تعداد ردیف آموزشی در یک گروه، fit مجزا بی‌ثبات است ⇒ سقوط امن به fit سراسری
_MIN_GROUP_ROWS = 30


def _fit_scoped(explorer_fn, train: pd.DataFrame, test: pd.DataFrame, tau: float,
                key: str | None) -> np.ndarray:
    if key is None:
        return np.clip(explorer_fn(train, test, tau), 0.0, 1.0)
    pred = pd.Series(index=test.index, dtype=float)
    global_pred = None
    for val, te_g in test.groupby(key, observed=True):
        tr_g = train[train[key] == val]
        if len(tr_g) < _MIN_GROUP_ROWS:
            if global_pred is None:
                global_pred = pd.Series(np.clip(explorer_fn(train, test, tau), 0.0, 1.0), index=test.index)
            pred.loc[te_g.index] = global_pred.loc[te_g.index]
            continue
        pred.loc[te_g.index] = np.clip(explorer_fn(tr_g, te_g, tau), 0.0, 1.0)
    return pred.to_numpy()


def run_scope_axis(explorer_fn, folds: list, tau: float) -> dict[str, np.ndarray]:
    out: dict[str, list[np.ndarray]] = {name: [] for name in SCOPE_KEYS}
    for tr, te in folds:
        for name, key in SCOPE_KEYS.items():
            out[name].append(_fit_scoped(explorer_fn, tr, te, tau, key))
    return {k: np.concatenate(v) for k, v in out.items()}


# ---------------------------------------------------------------------------
# محور «معماری» — flat در برابر two_stage (عامل روز ← انحراف سلف، F59/F61)
# ---------------------------------------------------------------------------

def _two_stage_predict(explorer_fn, train: pd.DataFrame, test: pd.DataFrame, tau: float) -> np.ndarray:
    """گام ۱: کوانتایل τ نرخ **میانگین روز** از فیچرهای سطح روز. گام ۲: کوانتایل τ
    باقیمانده‌ی سلولی (ρ_سلول − میانگین ρ همان روز در train) از فیچرهای سلولی. جمع دو
    گام، تقریب صادقانه‌ی «کوانتایل جمع» است (نه دقیق آماری، ولی همان سطح تقریبی که
    Q3/باقیمانده در بقیه‌ی خ۱ استفاده می‌شود)."""
    day_mean_col = train.groupby("date_gregorian")["rho"].transform("mean")
    tr_resid = train.copy()
    tr_resid["_day_mean"] = day_mean_col
    tr_resid["_resid"] = tr_resid["rho"] - tr_resid["_day_mean"]

    # گام ۱ فقط فیچر سطح-روز می‌بیند — drop_duplicates روی این ستون‌ها معنادار است چون
    # همه‌شان درون یک روز ثابت‌اند (بند بالای _DAY_LEVEL_FEATURES)
    day_level = tr_resid.drop_duplicates("date_gregorian").copy()
    day_level["_target"] = day_level["_day_mean"]
    te_day = test.drop_duplicates("date_gregorian")
    stage1_test = explorer_fn(day_level, te_day, tau, target_col="_target",
                              feature_cols=_DAY_LEVEL_FEATURES)
    day_pred_map = dict(zip(te_day["date_gregorian"], stage1_test))
    stage1_pred = test["date_gregorian"].map(day_pred_map).to_numpy(dtype=float)

    # گام ۲: فیچر سلولی معمولی (فیچرست پیش‌فرض explorer_fn)، هدف = باقیمانده‌ی سلول از میانگین روز
    tr_resid["_target"] = tr_resid["_resid"]
    stage2_pred = explorer_fn(tr_resid, test, tau, target_col="_target")

    return np.clip(stage1_pred + stage2_pred, 0.0, 1.0)


def run_architecture_axis(explorer_fn, folds: list, tau: float) -> dict[str, np.ndarray]:
    out = {"flat": [], "two_stage": []}
    for tr, te in folds:
        out["flat"].append(np.clip(explorer_fn(tr, te, tau), 0.0, 1.0))
        out["two_stage"].append(_two_stage_predict(explorer_fn, tr, te, tau))
    return {k: np.concatenate(v) for k, v in out.items()}


AXIS_RUNNERS = {
    "architecture": (run_architecture_axis, "flat"),
    "target": (run_target_axis, "rho"),
    "weighting": (run_weighting_axis, "none"),
    "scope": (run_scope_axis, "global"),
}


# ---------------------------------------------------------------------------
# اجرا + گزارش
# ---------------------------------------------------------------------------

def _official_folds() -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    from src.cv import DATE_COL, load_cv_folds
    from src.features.build import FEATURES_A_PATH

    df = pd.read_parquet(FEATURES_A_PATH).sort_values(DATE_COL).reset_index(drop=True)
    fold_meta, _ = load_cv_folds()
    return [(df.loc[m1], df.loc[m2]) for f in fold_meta for m1, m2 in [f.masks(df[DATE_COL])]]


def _actual_rho(folds: list) -> np.ndarray:
    return np.concatenate([te["rho"].to_numpy() for _, te in folds])


def screen_axis(axis: str, tau: float = TUNING_TAU) -> pd.DataFrame:
    runner_fn, anchor = AXIS_RUNNERS[axis]
    folds = _official_folds()
    actual = _actual_rho(folds)
    rows = []
    for explorer_name, explorer_fn in EXPLORERS.items():
        preds = runner_fn(explorer_fn, folds, tau)
        losses = {name: pinball_loss(actual, p, tau) for name, p in preds.items()}
        anchor_loss = losses[anchor]
        for name, loss in losses.items():
            if name == anchor:
                continue
            dm, p = diebold_mariano(loss, anchor_loss)
            delta = float(loss.mean() - anchor_loss.mean())
            rows.append({
                "axis": axis, "explorer": explorer_name, "value": name, "anchor": anchor,
                "pinball_value": float(loss.mean()), "pinball_anchor": float(anchor_loss.mean()),
                "delta": delta, "dm_stat": dm, "p_value": p,
                "beats_anchor_significantly": bool(np.isfinite(p) and p < 0.05 and delta < -EQUIVALENCE_MARGIN_DELTA),
                "within_equivalence_margin": bool(abs(delta) < EQUIVALENCE_MARGIN_DELTA or not np.isfinite(p) or p >= 0.05),
            })
    return pd.DataFrame(rows)


def screen_all_axes(tau: float = TUNING_TAU) -> pd.DataFrame:
    return pd.concat([screen_axis(ax, tau) for ax in AXIS_RUNNERS], ignore_index=True)


def render_report(df: pd.DataFrame, tau: float) -> str:
    lines = [
        "# پیمایش محورها با کاوشگر ارزان — اسپرینت A",
        "",
        f"> بند 7.9.1 بازنویسی‌شده (ردیف ۳۷ decision_log). τ={tau}، دو کاوشگر (خطی +"
        " LightGBM-quantile پیش‌فرض)، هر ۵ fold رسمی. δ (حاشیه‌ی هم‌ارزی عملی) = "
        f"{EQUIVALENCE_MARGIN_DELTA}. محورهای `level`/`output_aggregation` عمداً پوشش داده "
        "نشدند — نیازمند دیتاست‌های L2-L5/تجمیع چندسطحی که هنوز ساخته نشده‌اند "
        "(اسپرینت‌های B/C).",
        "",
        "⚠️ **محدودیت پیاده‌سازی `architecture=two_stage` اینجا:** ترکیب دو مرحله با "
        "**جمع ساده‌ی دو کوانتایل** انجام شده (کوانتایل روز + کوانتایل باقیمانده)، که از "
        "نظر آماری تقریب است — کوانتایل مجموع، مجموع کوانتایل‌ها نیست مگر با فرض استقلال. "
        "نتیجه‌ی «بی‌اثر» زیر برای این محور باید با احتیاط خوانده شود: می‌تواند واقعاً "
        "بی‌اثر باشد **یا** می‌تواند اثر تقریب جمع ساده باشد. اگر خ۸ (سلسله‌مراتبی، "
        "اسپرینت C) هم دومرحله‌ای بود نتیجه‌ی متفاوت داد، این تفاوت را توضیح می‌دهد.",
        "",
        "| محور | کاوشگر | مقدار | Δ نسبت به لنگر | p-value | برد معنادار؟ | داخل δ (هم‌ارز)؟ |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        win = "🎯 بله" if r["beats_anchor_significantly"] else ""
        eq = "✅ هم‌ارز" if r["within_equivalence_margin"] else "❌ متفاوت"
        lines.append(
            f"| {r['axis']} | {r['explorer']} | `{r['value']}` (لنگر=`{r['anchor']}`) | "
            f"{r['delta']:+.5f} | {r['p_value']:.4f} | {win} | {eq} |"
        )

    lines += ["", "## توصیه‌ی قفل هر محور", "",
             "| محور | هر دو کاوشگر توافق دارند؟ | توصیه |", "|---|---|---|"]
    for axis in df["axis"].unique():
        sub = df[df["axis"] == axis]
        any_win = sub["beats_anchor_significantly"].any()
        per_explorer_win = sub.groupby("explorer")["beats_anchor_significantly"].any()
        agree = len(set(per_explorer_win)) == 1
        rec = "باز بماند — برد معنادار پیدا شد" if any_win else "قفل روی لنگر (بی‌اثر در این آزمون)"
        lines.append(f"| {axis} | {'بله' if agree else 'خیر (برهم‌کنش با نوع کاوشگر)'} | {rec} |")
    return "\n".join(lines)


def main() -> None:
    from src.config import REPORTS_DIR, set_global_seed

    set_global_seed()
    df = screen_all_axes()
    report = render_report(df, TUNING_TAU)
    out_dir = REPORTS_DIR / "phase7"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "axis_screening.md").write_text(report + "\n")
    df.to_json(out_dir / "axis_screening.json", orient="records", indent=2, force_ascii=False)
    print(report)
    print(f"\nذخیره شد در {out_dir / 'axis_screening.md'}")


if __name__ == "__main__":
    main()
