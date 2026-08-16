"""آزمایش هوشمند خوشه‌بندی خ۳ (درخواست کاربر): آیا سری «عامل روز» محدود به سلف‌های
**تهران** از سری سراسری (همه‌ی شهرها) بهتر پیش‌بینی می‌شود؟

مدل `ma` انتخاب شد چون در امکان‌سنجی کامل (`F03_feasibility.md`) بهترین عملکرد کلی را
در هر دو وعده داشت — طبق درخواست کاربر («روی مدل‌های خوب... هوشمندانه، نه کامل»)، فقط
همین یک مدل و یک خوشه‌بندی (شهر، F12) آزموده می‌شود، نه یک grid.

اجرا: ``python -m src.models.run_f03_city_cluster``
"""

import warnings

import numpy as np
import pandas as pd

from src.baselines import pinball_loss
from src.cv import load_cv_folds
from src.features.build import FEATURES_A_PATH
from src.features.l3_series import build_l3_series
from src.models.families.f03_timeseries import fit_predict_ma

BEST_MODEL_ID = "ma"


def _official_l3_folds(series: pd.DataFrame) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    fold_meta, _ = load_cv_folds()
    return [(series.loc[m1], series.loc[m2]) for f in fold_meta for m1, m2 in [f.masks(series["date_gregorian"])]]


def run_comparison(tau: float = 0.2) -> pd.DataFrame:
    fx = pd.read_parquet(FEATURES_A_PATH)
    tehran_restaurants = set(fx.loc[fx["is_tehran"], "RestaurantName"].unique())

    national = build_l3_series()
    tehran_only = build_l3_series(restaurant_filter=tehran_restaurants)

    rows = []
    for meal in ("lunch", "dinner"):
        folds_national = _official_l3_folds(national[meal])
        folds_tehran = _official_l3_folds(tehran_only[meal])
        for i, ((tr_n, te_n), (tr_t, te_t)) in enumerate(zip(folds_national, folds_tehran)):
            result = {"meal": meal, "fold": i}
            for label, tr, te in [("سراسری", tr_n, te_n), ("فقط-تهران", tr_t, te_t)]:
                mask = te["day_shock"].notna().to_numpy()
                if mask.sum() == 0:
                    result[label] = float("nan")
                    continue
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        pred = np.asarray(fit_predict_ma(tr, te, tau), dtype=float)
                    pb = float(pinball_loss(te["day_shock"].to_numpy()[mask], pred[mask], tau).mean())
                except Exception:
                    pb = float("nan")
                result[label] = pb
            rows.append(result)
    return pd.DataFrame(rows)


def render_report(df: pd.DataFrame, tau: float) -> str:
    lines = [
        "# آزمایش خوشه‌بندی خ۳ — مدل `ma` سراسری در برابر فقط-تهران",
        "",
        f"> بهترین مدل خ۳ (`{BEST_MODEL_ID}`) روی سری «عامل روز» محدود به سلف‌های تهران "
        f"در برابر سری سراسری. τ={tau}، هر ۵ fold رسمی، جدا برای ناهار/شام.",
        "",
        "| وعده | fold | سراسری | فقط-تهران | برنده |",
        "|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        winner = "فقط-تهران ✅" if r["فقط-تهران"] < r["سراسری"] else "سراسری"
        lines.append(f"| {r['meal']} | {r['fold']} | {r['سراسری']:.5f} | {r['فقط-تهران']:.5f} | {winner} |")

    n_tehran_wins = int((df["فقط-تهران"] < df["سراسری"]).sum())
    lines += ["", f"**فقط-تهران در {n_tehran_wins} از {len(df)} ترکیب (وعده×fold) بهتر بود.**"]
    return "\n".join(lines)


def main() -> None:
    from src.config import REPORTS_DIR, set_global_seed

    set_global_seed()
    df = run_comparison()
    report = render_report(df, 0.2)
    out = REPORTS_DIR / "phase7"
    out.mkdir(parents=True, exist_ok=True)
    (out / "F03_city_cluster.md").write_text(report + "\n")
    df.to_json(out / "F03_city_cluster.json", orient="records", indent=2, force_ascii=False)
    print(report)
    print(f"\nذخیره شد در {out / 'F03_city_cluster.md'}")


if __name__ == "__main__":
    main()
