"""بند 7.5.4 — «انتخاب فیچر درون‌مدلی» برای خ۱: نمودار ضریب-در-برابر-λ برای
Lasso/ElasticNet + فیچرهای بازمانده روی هر ۵ fold رسمی (اجماع).

## چرا این آزمون با فیچرست S2 اجرا می‌شود، نه فیچرست باگ‌دار واقعاً-اجراشده

`src/models/run_f01_s2_feature_bug_check.py` کشف کرد S2 خ۱ هرگز `_feature_cols_s2()`
را واقعاً استفاده نکرد (یافته‌ی ۳۴، ردیف ۴۷ decision_log) — هر ۱۵ مدل روی `FS_day`
(۴۷ ستون) اجرا شدند. ولی **کل نکته‌ی این آزمون خاص** (بند 7.5.4) این بود که Lasso/
ElasticNet خوشه‌ی هم‌خط VIF ۲۰۰۰-۱۰۰۰۰ را (که فقط در `_feature_cols_s2()`/`FS_full_A`
حاضر است، نه در `FS_day`) با انتخاب فیچر خودشان حل کنند — پس این‌جا عمداً با فیچرست
**درست** (`_feature_cols_s2()`، ۱۵۴ ستون بعد از یک‌هات) اجرا می‌شود، وگرنه اصلاً
موضوعی برای آزمودن نیست.

## چه چیزی گزارش می‌شود

۱. مسیر کامل ضریب در برابر λ (`sklearn.linear_model.lasso_path`/`enet_path`) روی
   train فولد۰، با خط عمودی روی λ تنظیم‌شده‌ی S2.
۲. فیچرهای بازمانده (ضریب ≠ ۰) در λ تنظیم‌شده، به‌ازای هر ۵ fold رسمی جداگانه —
   اجماع = در چند fold از ۵ فیچر زنده ماند.
۳. آیا خوشه‌ی هم‌خط `cell_expanding_rate`/`cell_shrunk_rate`/`cell_dow_expanding_rate`/
   `cell_dow_shrunk_rate`/`food_expanding_rate`/`food_shrunk_rate` (بند 7.5.3، VIF
   ۲۰۰۰-۱۰۰۰۰) واقعاً هرس شد یا نه — سؤال مستقیم بند 7.5.4.

اجرا: ``python -m src.models.run_f01_coefficient_path``
"""

import matplotlib
import numpy as np
import pandas as pd
from sklearn.linear_model import enet_path, lasso_path
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.config import FIGURES_DIR, REPORTS_DIR
from src.cv import DATE_COL, load_cv_folds
from src.eda_lib.figio import save_fig
from src.features.build import FEATURES_A_PATH
from src.models.card_writer import load_s2_result
from src.models.families import common
from src.models.families.f01_linear import _feature_cols_s2
from src.viz_fa import fa
from src.viz_fa import setup as viz_setup

#: خوشه‌ی هم‌خط مستندشده در بند 7.5.3 — سؤال مستقیم این آزمون
COLLINEAR_CLUSTER = ("cell_expanding_rate", "cell_shrunk_rate", "cell_dow_expanding_rate",
                     "cell_dow_shrunk_rate", "food_expanding_rate", "food_shrunk_rate")

#: ⚠️ **باید نزولی باشد.** `lasso_path`/`enet_path` با warm-start از α قبلی کار
#: می‌کنند؛ اگر آرایه صعودی داده شود (α کوچک اول)، همگرایی نقاط ابتدایی (α کوچک،
#: باید ضریب بزرگ بدهد) در `max_iter` محدود می‌شکند و کاذباً صفر می‌ماند — آزموده و
#: تأیید شد (L1-norm مسیر با آرایه‌ی صعودی برای α کوچک اشتباه صفر بود، با نزولی درست).
ALPHA_GRID = np.logspace(2, -4, 100)
TOP_K_LABELED = 12


def _official_folds() -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    df = pd.read_parquet(FEATURES_A_PATH).sort_values(DATE_COL).reset_index(drop=True)
    fold_meta, _ = load_cv_folds()
    return [(df.loc[m1], df.loc[m2]) for f in fold_meta for m1, m2 in [f.masks(df[DATE_COL])]]


def _design_s2_matrix(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tr, te = train.copy(), test.copy()
    tr["log_res_sq"] = tr["log_res"] ** 2
    te["log_res_sq"] = te["log_res"] ** 2
    return common.design_matrix(tr, te, _feature_cols_s2())


def compute_path(train: pd.DataFrame, model: str, l1_ratio: float = 1.0) -> tuple[np.ndarray, list[str]]:
    Xtr, _ = _design_s2_matrix(train, train)
    Ztr = StandardScaler().fit_transform(Xtr)
    y = train["rho"].to_numpy()
    if model == "lasso":
        _, coefs, _ = lasso_path(Ztr, y, alphas=ALPHA_GRID, max_iter=5000)
    else:
        _, coefs, _ = enet_path(Ztr, y, alphas=ALPHA_GRID, l1_ratio=l1_ratio, max_iter=5000)
    return coefs, list(Xtr.columns)


def plot_path(coefs: np.ndarray, feature_names: list[str], tuned_alpha: float,
             title: str, fname: str) -> str:
    """۱۲ فیچر با بیشترین |ضریب| بیشینه رنگی و برچسب‌دار؛ بقیه خاکستری کم‌رنگ."""
    max_abs = np.abs(coefs).max(axis=1)
    top_idx = np.argsort(max_abs)[::-1][:TOP_K_LABELED]

    viz_setup()
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for i in range(coefs.shape[0]):
        if i in top_idx:
            continue
        ax.plot(ALPHA_GRID, coefs[i], color="lightgray", linewidth=0.6, alpha=0.5, zorder=1)
    cmap = plt.get_cmap("tab20")
    for rank, i in enumerate(top_idx):
        ax.plot(ALPHA_GRID, coefs[i], color=cmap(rank % 20), linewidth=1.8,
                label=fa(feature_names[i]), zorder=2)

    ax.axvline(tuned_alpha, color="black", linestyle="--", linewidth=1.2,
              label=fa(f"$\\lambda$ تنظیم‌شده‌ی S2 = {tuned_alpha:.5f}"))
    ax.set_xscale("log")
    ax.set_xlabel(fa("$\\lambda$ (alpha)")); ax.set_ylabel(fa("ضریب استانداردشده"))
    ax.set_title(fa(title))
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.grid(alpha=.3)
    fig.tight_layout()
    path = save_fig(fig, fname, FIGURES_DIR)
    plt.close(fig)
    return str(path)


def surviving_features_per_fold(model_id: str, folds: list, alpha: float,
                                l1_ratio: float | None = None) -> pd.DataFrame:
    from sklearn.linear_model import ElasticNet, Lasso

    rows = {}
    for fi, (tr, te) in enumerate(folds):
        Xtr, _ = _design_s2_matrix(tr, te)
        Ztr = StandardScaler().fit_transform(Xtr)
        if model_id == "lasso":
            model = Lasso(alpha=alpha, max_iter=5000).fit(Ztr, tr["rho"])
        else:
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000).fit(Ztr, tr["rho"])
        nonzero = set(np.array(Xtr.columns)[np.abs(model.coef_) > 1e-10])
        rows[fi] = nonzero
    all_features = sorted(set().union(*rows.values()))
    tab = pd.DataFrame({
        "feature": all_features,
        "n_folds_survived": [sum(f in rows[fi] for fi in rows) for f in all_features],
    }).sort_values("n_folds_survived", ascending=False).reset_index(drop=True)
    return tab


def render_report(model_id: str, survive_tab: pd.DataFrame, fig_path: str, alpha: float,
                  n_folds: int) -> str:
    n_total_candidates = 154  # بعد از یک‌هات _feature_cols_s2()
    n_survived_any = len(survive_tab)
    n_survived_all = int((survive_tab["n_folds_survived"] == n_folds).sum())
    cluster_status = {f: (int(survive_tab.loc[survive_tab["feature"] == f, "n_folds_survived"].iloc[0])
                          if f in survive_tab["feature"].values else 0)
                      for f in COLLINEAR_CLUSTER}
    lines = [
        f"## `{model_id}` — λ={alpha:.5f}",
        "",
        f"![{model_id} coefficient path]({fig_path})",
        "",
        f"از {n_total_candidates} ستون طراحی (بعد از یک‌هات `_feature_cols_s2()`): "
        f"**{n_survived_any}** حداقل در یک fold زنده ماندند، **{n_survived_all}** در هر ۵ fold "
        "(اجماع کامل).",
        "",
        "### وضعیت خوشه‌ی هم‌خط بند 7.5.3 (VIF ۲۰۰۰-۱۰۰۰۰)",
        "",
        "| فیچر | زنده در چند fold از ۵ |",
        "|---|---|",
    ]
    for f, n in cluster_status.items():
        lines.append(f"| `{f}` | {n} |")
    n_cluster_survived = sum(1 for n in cluster_status.values() if n > 0)
    lines += [
        "",
        f"**{n_cluster_survived} از {len(COLLINEAR_CLUSTER)} عضو خوشه در دست‌کم یک fold زنده ماندند.**",
        "",
        "### ۱۵ فیچر با بیشترین اجماع (زنده در بیشترین تعداد fold)",
        "",
        "| فیچر | زنده در چند fold |",
        "|---|---|",
    ]
    for _, r in survive_tab.head(15).iterrows():
        lines.append(f"| `{r['feature']}` | {r['n_folds_survived']} |")
    return "\n".join(lines)


def main() -> None:
    from src.config import set_global_seed

    set_global_seed()
    folds = _official_folds()
    train0 = folds[0][0]

    lasso_hp = load_s2_result("lasso", "F01")["best_hyperparams"]
    en_hp = load_s2_result("elasticnet", "F01")["best_hyperparams"]

    lasso_coefs, feat_names = compute_path(train0, "lasso")
    lasso_fig = plot_path(lasso_coefs, feat_names, lasso_hp["alpha"],
                          "مسیر ضریب Lasso در برابر $\\lambda$ (fold۰، فیچرست S2)",
                          "f01_lasso_coefficient_path")
    lasso_survive = surviving_features_per_fold("lasso", folds, lasso_hp["alpha"])
    lasso_report = render_report("lasso", lasso_survive, f"../figures/{lasso_fig.split('/')[-1]}",
                                 lasso_hp["alpha"], len(folds))

    en_coefs, _ = compute_path(train0, "elasticnet", l1_ratio=en_hp["l1_ratio"])
    en_fig = plot_path(en_coefs, feat_names, en_hp["alpha"],
                       f"مسیر ضریب ElasticNet در برابر $\\lambda$ (fold۰، l1_ratio={en_hp['l1_ratio']:.3f})",
                       "f01_elasticnet_coefficient_path")
    en_survive = surviving_features_per_fold("elasticnet", folds, en_hp["alpha"], en_hp["l1_ratio"])
    en_report = render_report("elasticnet", en_survive, f"../figures/{en_fig.split('/')[-1]}",
                              en_hp["alpha"], len(folds))

    report = "\n\n".join([
        "# مسیر ضریب Lasso/ElasticNet در برابر λ + فیچرهای بازمانده — بند 7.5.4",
        "",
        "> فیچرست: `_feature_cols_s2()` (۱۵۴ ستون بعد از یک‌هات) — نه فیچرست باگ‌دار "
        "واقعاً-اجراشده‌ی S2 (یافته‌ی ۳۴)؛ بند بالای ماژول توضیح می‌دهد چرا. λ از "
        "`S2_tuning_F01.json` (تنظیم‌شده روی همان فیچرست باگ‌دار — همچنان بهترین تخمین "
        "موجود از مقیاس منظم‌سازی معقول).",
        "",
        lasso_report, "", en_report,
    ])
    out = REPORTS_DIR / "phase7"
    out.mkdir(parents=True, exist_ok=True)
    (out / "f01_coefficient_path.md").write_text(report + "\n")
    lasso_survive.to_json(out / "f01_lasso_surviving_features.json", orient="records",
                          indent=2, force_ascii=False)
    en_survive.to_json(out / "f01_elasticnet_surviving_features.json", orient="records",
                       indent=2, force_ascii=False)
    print(report)
    print(f"\nذخیره شد در {out / 'f01_coefficient_path.md'}")


if __name__ == "__main__":
    main()
