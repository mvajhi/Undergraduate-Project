"""بند 7.13.4 — معیار پایان خانواده‌ی ۴ (DFM سطح L4)، سنجش امکان‌سنجی روی هر ۵ fold رسمی.

⚠️ مستقل از `model_comparison.md` (سطح/دانه‌بندی متفاوت — بالای `f04_multivariate.py`).
مرجع اینجا **کوانتایل تجربی خودِ هر سری** (بدون در نظر گرفتن عامل مشترک) است — سؤالی که
این جدول جواب می‌دهد: «آیا مدل‌سازی عامل مشترک (F60) روی کوانتایل تک‌سری‌ای اضافه‌ارزش دارد؟»

اجرا: ``python -m src.models.run_f04_feasibility``
"""

import time
import warnings

import numpy as np
import pandas as pd

from src.baselines import pinball_loss
from src.cv import DATE_COL, load_cv_folds
from src.features.l4_series import build_l4_panel
from src.models.families.f04_multivariate import fit_predict_dfm


def _official_panel_folds(panel: pd.DataFrame) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    fold_meta, _ = load_cv_folds()
    dates = panel.reset_index()[DATE_COL]
    dates.index = panel.index
    return [(panel.loc[m1], panel.loc[m2]) for f in fold_meta for m1, m2 in [f.masks(dates)]]


def _naive_series_quantile(train_panel: pd.DataFrame, test_panel: pd.DataFrame, tau: float) -> pd.DataFrame:
    """مرجع: کوانتایل تجربی τ خودِ هر سری روی دوره‌ی آموزش، بدون هیچ ساختار مشترک."""
    q = train_panel.quantile(tau)
    return pd.DataFrame(np.tile(q.to_numpy(), (len(test_panel), 1)),
                        index=test_panel.index, columns=test_panel.columns)


def run_feasibility(tau: float = 0.2, k_factors: int = 1) -> pd.DataFrame:
    panel = build_l4_panel()
    folds = _official_panel_folds(panel)

    rows = []
    for i, (train, test) in enumerate(folds):
        actual_long = test.stack()
        if actual_long.empty:
            continue

        naive = _naive_series_quantile(train, test, tau)
        naive_long = naive.stack().reindex(actual_long.index)
        pb_naive = float(pinball_loss(actual_long.to_numpy(), naive_long.to_numpy(), tau).mean())

        t0 = time.time()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dfm_pred = fit_predict_dfm(train, test, tau, k_factors=k_factors)
            seconds = time.time() - t0
            dfm_long = dfm_pred.stack().reindex(actual_long.index)
            pb_dfm = float(pinball_loss(actual_long.to_numpy(), dfm_long.to_numpy(), tau).mean())
            status = "✅"
        except Exception as exc:  # DFM می‌تواند عددی ناهمگرا شود — شکست خودش داده است
            seconds = time.time() - t0
            pb_dfm = float("nan")
            status = f"❌ {type(exc).__name__}"

        rows.append({
            "fold": i, "n_series": train.shape[1], "n_train_days": len(train), "n_test_days": len(test),
            "status": status, "pinball_dfm": pb_dfm, "pinball_naive": pb_naive,
            "beats_naive": (pb_dfm < pb_naive) if np.isfinite(pb_dfm) else False,
            "seconds": seconds, "n_eval": len(actual_long.dropna()),
        })
    return pd.DataFrame(rows)


def render_report(df: pd.DataFrame, tau: float, k_factors: int) -> str:
    lines = [
        "# امکان‌سنجی خ۴ (DFM سطح L4) — بند 7.13.4",
        "",
        f"> τ={tau}، k_factors={k_factors}. هر ۵ fold رسمی، ۴۱ سری $(m,r)$ هم‌زمان. مرجع: "
        "کوانتایل تجربی خودِ هر سری (بدون عامل مشترک).",
        "",
        "| fold | n_train روز | n_test روز | وضعیت | pinball DFM | pinball تک‌سری | برد؟ | ثانیه |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        beats = "✅" if r["beats_naive"] else "—"
        pb_dfm = f"{r['pinball_dfm']:.5f}" if np.isfinite(r["pinball_dfm"]) else "—"
        lines.append(f"| {r['fold']} | {r['n_train_days']} | {r['n_test_days']} | {r['status']} | "
                    f"{pb_dfm} | {r['pinball_naive']:.5f} | {beats} | {r['seconds']:.1f} |")

    n_ok = int((df["status"] == "✅").sum())
    n_beat = int(df["beats_naive"].sum())
    lines += ["", f"**DFM در {n_ok} از {len(df)} fold با موفقیت برازش شد؛ در {n_beat} از آن‌ها از مرجع تک‌سری بهتر بود.**",
             "",
             "⚠️ **یادآوری دامنه:** سطح L4 (میانگین روی FoodName)، نه سلولی — مستقیماً با "
             "`model_comparison.md` قابل‌جمع‌بندی نیست (بند 7.24: آشتی سطوح، کار جداگانه)."]
    return "\n".join(lines)


def run_city_cluster_comparison(tau: float = 0.2, k_factors: int = 1) -> pd.DataFrame:
    """آزمایش هوشمند خوشه‌بندی (درخواست کاربر): آیا DFM محدود به سری‌های **تهران**
    (۳۱ از ۴۱ سری، همگن‌تر) از DFM سراسری (۴۱ سری، آمیخته با ۵ شهر تک‌سلفی) بهتر است؟

    شهر انتخاب شد نه خوشه‌ی F41، چون F12 (Cliff's δ=۰.۹۶۴) قوی‌ترین سیگنال گروه‌بندی
    مستندشده‌ی کل پروژه است، و بررسی ستون‌ها نشان داد ۵ شهر غیرتهرانی هرکدام فقط **۱**
    سلف دارند — یعنی DFM جداگانه برایشان اصلاً بی‌معناست (کمینه‌ی لازم ≥۲ سری). پس تنها
    مقایسه‌ی معنادار: DFM(تهران-تنها) در برابر DFM(سراسری)، هر دو روی همان سلول‌های
    تهران ارزیابی می‌شوند — نه یک grid روی همه‌ی شهرها.
    """
    from src.features.build import FEATURES_A_PATH
    from src.features.l4_series import SEP

    panel = build_l4_panel()
    fx = pd.read_parquet(FEATURES_A_PATH)
    tehran_restaurants = set(fx.loc[fx["is_tehran"], "RestaurantName"].unique())
    tehran_cols = [c for c in panel.columns if c.split(SEP)[0] in tehran_restaurants]

    folds = _official_panel_folds(panel)
    rows = []
    for i, (train, test) in enumerate(folds):
        actual_tehran = test[tehran_cols].stack()
        if actual_tehran.empty:
            continue

        result = {"fold": i, "n_tehran_series": len(tehran_cols)}
        for label, train_sub, test_sub in [
            ("پوششی (۴۱ سری)", train, test),
            ("فقط-تهران (۳۱ سری)", train[tehran_cols], test[tehran_cols]),
        ]:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pred = fit_predict_dfm(train_sub, test_sub, tau, k_factors=k_factors)
                pred_tehran = pred[tehran_cols].stack().reindex(actual_tehran.index)
                pb = float(pinball_loss(actual_tehran.to_numpy(), pred_tehran.to_numpy(), tau).mean())
            except Exception:
                pb = float("nan")
            result[label] = pb
        rows.append(result)
    return pd.DataFrame(rows)


def render_cluster_report(df: pd.DataFrame, tau: float) -> str:
    cols = [c for c in df.columns if c not in ("fold", "n_tehran_series")]
    lines = [
        "# آزمایش خوشه‌بندی خ۴ — DFM سراسری در برابر DFM فقط-تهران",
        "",
        f"> درخواست کاربر: «روی مدل‌های خوب، خوشه‌بندی و مدل جداگانه تست کن، هوشمندانه نه "
        "کامل». شهر (F12، Cliff's δ=۰.۹۶۴) انتخاب شد چون تنها سیگنال گروه‌بندی است که ≥۲ "
        "سری در هر گروه تضمین می‌کند (۵ شهر دیگر هرکدام ۱ سلف دارند). هر دو مدل روی **همان "
        f"سلول‌های تهران** ارزیابی می‌شوند — مقایسه‌ی منصفانه، τ={tau}.",
        "",
        "| fold | " + " | ".join(cols) + " |",
        "|---|" + "---|" * len(cols),
    ]
    for _, r in df.iterrows():
        vals = " | ".join(f"{r[c]:.5f}" if pd.notna(r[c]) else "—" for c in cols)
        lines.append(f"| {int(r['fold'])} | {vals} |")

    tehran_only_wins = int((df["فقط-تهران (۳۱ سری)"] < df["پوششی (۴۱ سری)"]).sum())
    lines += ["", f"**DFM فقط-تهران در {tehran_only_wins} از {len(df)} fold از DFM سراسری بهتر بود.**"]
    return "\n".join(lines)


def main() -> None:
    from src.config import REPORTS_DIR, set_global_seed

    set_global_seed()
    df = run_feasibility()
    report = render_report(df, 0.2, 1)
    out = REPORTS_DIR / "phase7"
    out.mkdir(parents=True, exist_ok=True)
    (out / "F04_feasibility.md").write_text(report + "\n")
    df.to_json(out / "F04_feasibility.json", orient="records", indent=2, force_ascii=False)
    print(report)

    df_cluster = run_city_cluster_comparison()
    report_cluster = render_cluster_report(df_cluster, 0.2)
    (out / "F04_city_cluster.md").write_text(report_cluster + "\n")
    df_cluster.to_json(out / "F04_city_cluster.json", orient="records", indent=2, force_ascii=False)
    print("\n" + report_cluster)
    print(f"\nذخیره شد در {out / 'F04_feasibility.md'} و {out / 'F04_city_cluster.md'}")


if __name__ == "__main__":
    main()
