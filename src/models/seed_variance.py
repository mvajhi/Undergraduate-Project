"""دروازه‌ی M4 — واریانس سه-seed برای قهرمانان نهایی (قاعده‌ی A7، بند 7.29.2 محدودشده
طبق ردیف ۳۹ decision_log: فقط ۴ مدلی که وارد کالیبراسیون/ترکیب شدند، نه کل فهرست کوتاه).

seedها طبق بند ۷.۷ WBS: ``{42, 1337, 2026}``. برای هر قهرمان، با همان هایپرپارامتر S2
(بدون تنظیم مجدد)، مدل روی هر seed و هر ۵ fold رسمی برازش می‌شود؛ ``pinball@0.20`` به‌ازای
هر seed میانگین ۵ fold است، سپس میانگین/انحراف‌معیار روی ۳ seed گزارش می‌شود.

⚠️ ``knn_quantile`` هیچ مؤلفه‌ی تصادفی ندارد (k-NN دقیق + مقیاس‌بندی قطعی) — واریانس آن
باید دقیقاً صفر باشد؛ عدد غیرصفر نشانه‌ی باگ در تولید داده/ترتیب ردیف است، نه seed مدل.

اجرا: ``python -m src.models.seed_variance``
"""

import importlib

import numpy as np
import pandas as pd

from src.baselines import pinball_loss
from src.cv import DATE_COL, load_cv_folds
from src.features.build import FEATURES_A_PATH
from src.models.axes import TUNING_TAU
from src.models.card_writer import _FAMILY_MODULES, load_s2_result
from src.models.run_cqr_champions import CHAMPIONS

SEEDS = (42, 1337, 2026)


def _official_folds() -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    df = pd.read_parquet(FEATURES_A_PATH).sort_values(DATE_COL).reset_index(drop=True)
    fold_meta, _ = load_cv_folds()
    return [(df.loc[m1], df.loc[m2]) for f in fold_meta for m1, m2 in [f.masks(df[DATE_COL])]]


def evaluate_seed_variance(folds: list, tau: float = TUNING_TAU) -> pd.DataFrame:
    rows = []
    for family, model_id in CHAMPIONS:
        result = load_s2_result(model_id, family)
        hp = dict(result["best_hyperparams"])
        mod = importlib.import_module(_FAMILY_MODULES[family])
        fit_fn = mod.MODELS[model_id]

        seed_pinballs = []
        for seed in SEEDS:
            fold_pb = []
            for tr, te in folds:
                try:
                    pred = fit_fn(tr, te, tau, seed=seed, **hp)
                except TypeError:
                    pred = fit_fn(tr, te, tau, **hp)  # مدل بدون seed (مثلاً knn_quantile)
                fold_pb.append(float(pinball_loss(te["rho"].to_numpy(), np.asarray(pred), tau).mean()))
            seed_pinballs.append(float(np.mean(fold_pb)))

        rows.append({
            "model_id": model_id, "family": family,
            "pinball_mean": float(np.mean(seed_pinballs)),
            "pinball_std": float(np.std(seed_pinballs, ddof=1)),
            "pinball_per_seed": seed_pinballs,
        })
    return pd.DataFrame(rows)


def render_report(df: pd.DataFrame) -> str:
    lines = [
        "# دروازه‌ی M4 — واریانس سه-seed قهرمانان نهایی",
        "",
        f"> قاعده‌ی A7، محدودشده طبق ردیف ۳۹ decision_log به ۴ قهرمانی که وارد "
        f"کالیبراسیون/ترکیب (خ۱۲–۱۳) شدند. seedها: `{SEEDS}`. هایپرپارامتر هر مدل از S2 "
        f"(τ={TUNING_TAU}) بدون تنظیم مجدد.",
        "",
        "| مدل | میانگین pinball | std (۳ seed) | pinball هر seed |",
        "|---|---|---|---|",
    ]
    for _, r in df.sort_values("pinball_mean").iterrows():
        per_seed = ", ".join(f"{v:.5f}" for v in r["pinball_per_seed"])
        lines.append(f"| `{r['model_id']}` | {r['pinball_mean']:.5f} | {r['pinball_std']:.6f} | {per_seed} |")

    max_std = df["pinball_std"].max()
    lines += ["", f"**بیشینه‌ی std میان قهرمانان: {max_std:.6f}.**"]
    if max_std < 0.0005:
        lines.append("واریانس seed در همه‌ی قهرمانان ناچیز است (کوچک‌تر از حاشیه‌ی هم‌ارزی "
                    "δ=۰.۰۰۰۵) — رتبه‌بندی جدول اصلی به انتخاب seed حساس نیست.")
    else:
        lines.append("⚠️ واریانس seed در دست‌کم یک مدل از حاشیه‌ی هم‌ارزی δ=۰.۰۰۰۵ بزرگ‌تر است — "
                    "تفاوت‌های نزدیک به δ در جدول مقایسه‌ی نهایی باید با احتیاط خوانده شوند.")
    return "\n".join(lines)


def main() -> None:
    from src.config import REPORTS_DIR, set_global_seed

    set_global_seed()
    folds = _official_folds()
    df = evaluate_seed_variance(folds)
    report = render_report(df)
    out = REPORTS_DIR / "phase7"
    out.mkdir(parents=True, exist_ok=True)
    (out / "seed_variance_champions.md").write_text(report + "\n")
    df.to_json(out / "seed_variance_champions.json", orient="records", indent=2, force_ascii=False)
    print(report)
    print(f"\nذخیره شد در {out / 'seed_variance_champions.md'}")


if __name__ == "__main__":
    main()
