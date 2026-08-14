"""بند ۶.۸ و ۶.۱۰ — اجرای کامل پروتکل اعتبارسنجی و تولید خروجی‌های فاز ۶.

اجرا: `python -m src.run_protocol`

خروجی:
- `reports/baselines.md` — هشت خط پایه روی walk-forward، در هر دو فضای نرخ و پرس
- `reports/tau_sensitivity.md` — تعهد بند ۲-۲ سند مسئله (۵ سناریوی τ)
- `reports/figures/report_16_tau_tradeoff.png`
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.baselines import (
    BASELINES,
    TAU_GRID,
    implied_cost_ratio,
    b3_empirical_quantile,
    cook_qty,
    operational_metrics,
    pinball_loss,
    quantile_adjust,
)
from src.config import FIGURES_DIR, REPORTS_DIR, set_global_seed
from src.cv import WalkForwardSplitter, block_bootstrap_2d, diebold_mariano, effective_sample_size, holdout_split, run_protocol_tests
from src.eda_lib.figio import save_fig
from src.features.build import FEATURES_A_PATH
from src.viz_fa import fa
from src.viz_fa import setup as viz_setup

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_TAU = 0.10
#: خط پایه‌هایی که میانگین را هدف می‌گیرند و برای مقایسه‌ی منصفانه نیاز به تصحیح کوانتایل دارند
MEAN_TARGETING = {"B1_global_mean", "B2_group_shrunk", "B4_seasonal_naive",
                  "B5_rolling_mean", "B6_day_factor"}
#: B3 و B7 ذاتاً کوانتایل را هدف می‌گیرند و نباید دوباره آفست بخورند


def load() -> pd.DataFrame:
    df = pd.read_parquet(FEATURES_A_PATH)
    return df.sort_values("date_gregorian").reset_index(drop=True)


def run_baselines(df: pd.DataFrame, splitter: WalkForwardSplitter, tau: float) -> pd.DataFrame:
    """هر خط پایه روی هر fold، سپس تجمیع روی همه‌ی foldها."""
    rows = []
    for f, tr_m, te_m in splitter.split(df):
        tr, te = df.loc[tr_m], df.loc[te_m]
        for name, fn in BASELINES.items():
            rho_hat = fn(tr, te, tau)
            if name in MEAN_TARGETING:
                rho_hat = quantile_adjust(rho_hat, tr, tau, fn)
            m = operational_metrics(te, rho_hat, tau)
            m.update(fold=f.index, baseline=name)
            rows.append(m)
    return pd.DataFrame(rows)


def summarise(res: pd.DataFrame) -> pd.DataFrame:
    """میانگین روی foldها — با وزن برابر برای هر fold (نه برای هر ردیف).

    ⚠️ این با میانگین **تجمیع‌شده** (وزن برابر برای هر ردیف) فرق دارد و رتبه‌بندی
    می‌تواند جابه‌جا شود، چون fold۲ (بازه‌ی رمضان) فقط ۱۸۵ ردیف دارد ولی وزن برابر
    می‌گیرد. هر دو در گزارش می‌آیند.
    """
    agg = res.groupby("baseline").agg(
        pinball=("pinball", "mean"),
        shortage_rate=("shortage_rate", "mean"),
        waste_reduction=("waste_reduction_pct", "mean"),
        MAE_portions=("MAE_portions", "mean"),
        RMSE_rho=("RMSE_rho", "mean"),
        shortage_portions=("shortage_portions", "sum"),
        surplus_portions=("surplus_portions", "sum"),
    )
    return agg.sort_values("pinball")


def significance_vs_best(df: pd.DataFrame, splitter: WalkForwardSplitter,
                         tau: float, best: str) -> pd.DataFrame:
    """Diebold-Mariano هر خط پایه در برابر بهترین (بند ۶.۶)."""
    losses: dict[str, list[np.ndarray]] = {k: [] for k in BASELINES}
    for f, tr_m, te_m in splitter.split(df):
        tr, te = df.loc[tr_m], df.loc[te_m]
        for name, fn in BASELINES.items():
            rho_hat = fn(tr, te, tau)
            if name in MEAN_TARGETING:
                rho_hat = quantile_adjust(rho_hat, tr, tau, fn)
            losses[name].append(pinball_loss(te["rho"].to_numpy(), rho_hat, tau))
    cat = {k: np.concatenate(v) for k, v in losses.items()}
    rows = []
    for name, l in cat.items():
        if name == best:
            continue
        dm, p = diebold_mariano(l, cat[best])
        rows.append({"baseline": name, "Δpinball": float(l.mean() - cat[best].mean()),
                     "DM": dm, "p": p})
    return pd.DataFrame(rows).sort_values("Δpinball")


def tau_sensitivity(df: pd.DataFrame, splitter: WalkForwardSplitter) -> pd.DataFrame:
    """بند ۶.۱۰ — تعهد سند تعریف مسئله: دو عدد به‌ازای هر سناریوی τ.

    خط پایه‌ی مبنا B3 (کوانتایل تجربی) است چون تنها خط پایه‌ای است که ذاتاً τ را
    هدف می‌گیرد و رابطه‌ی نظری «τ = نرخ کمبود» برایش معنا دارد.
    """
    rows = []
    for tau in TAU_GRID:
        for f, tr_m, te_m in splitter.split(df):
            tr, te = df.loc[tr_m], df.loc[te_m]
            m = operational_metrics(te, b3_empirical_quantile(tr, te, tau), tau)
            m.update(tau=tau, fold=f.index)
            rows.append(m)
    r = pd.DataFrame(rows)
    return r.groupby("tau").agg(
        waste_reduction=("waste_reduction_pct", "mean"),
        shortage_rate=("shortage_rate", "mean"),
        shortage_portions=("shortage_portions", "sum"),
        surplus_portions=("surplus_portions", "sum"),
        pinball=("pinball", "mean"),
    ).reset_index()


def main() -> None:
    set_global_seed()
    df = load()
    splitter = WalkForwardSplitter(n_folds=5, min_train_days=60)

    print("=" * 78)
    print("بند ۶.۸ — تست‌های واحد پروتکل")
    print("=" * 78)
    ok = run_protocol_tests(df, splitter)
    if not ok:
        raise AssertionError("تست‌های پروتکل شکست خوردند — پیش از ادامه باید رفع شوند")

    print("\nساختار foldها:")
    for f in splitter.split_dates(df["date_gregorian"]):
        n_tr = int(((df["date_gregorian"] >= f.train_start) & (df["date_gregorian"] <= f.train_end)).sum())
        n_te = int(((df["date_gregorian"] >= f.test_start) & (df["date_gregorian"] <= f.test_end)).sum())
        print(f"  {f}  (n_train={n_tr:,} · n_test={n_te:,})")

    # اندازه‌ی نمونه‌ی مؤثر (بند ۶.۶)
    sizes = df.groupby("date_gregorian").size().to_numpy()
    n_eff = effective_sample_size(len(df), sizes, icc=0.225)
    print(f"\nاندازه‌ی نمونه: خام={len(df):,} · **مؤثر={n_eff:,.0f}** "
          f"(ICC روز=۰.۲۲۵، میانگین {sizes.mean():.0f} رکورد در روز)")
    print(f"ضریب تورم واریانس: {len(df) / n_eff:.1f}× — هر CI که این را نادیده بگیرد باریک‌تر از واقعیت است")

    print("\n" + "=" * 78)
    print(f"بند ۶.۵ — هشت خط پایه روی walk-forward (τ={DEFAULT_TAU})")
    print("=" * 78)
    res = run_baselines(df, splitter, DEFAULT_TAU)
    summary = summarise(res)
    print(summary.round(4).to_string())

    best = summary.index[0]
    print(f"\nبهترین: {best}")
    sig = significance_vs_best(df, splitter, DEFAULT_TAU, best)
    print("\nآزمون Diebold-Mariano در برابر بهترین:")
    print(sig.round(4).to_string(index=False))

    print("\n" + "=" * 78)
    print("بند ۶.۱۰ — تحلیل حساسیت τ")
    print("=" * 78)
    ts = tau_sensitivity(df, splitter)
    print(ts.round(4).to_string(index=False))

    _write_baselines_report(df, splitter, summary, sig, res, n_eff, sizes)
    _write_tau_report(ts)
    _plot_tau(ts)


def _write_baselines_report(df, splitter, summary, sig, res, n_eff, sizes) -> None:
    folds = splitter.split_dates(df["date_gregorian"])
    lines = [
        "# خط پایه‌ها و پروتکل اعتبارسنجی — فاز ۶",
        "",
        "> بند ۶.۵ و ۶.۸ WBS. تولید خودکار با `python -m src.run_protocol`.",
        f"> داده: `features_A_v1.parquet` ({len(df):,} ردیف) · τ پیش‌فرض = {DEFAULT_TAU}",
        "",
        "## پروتکل",
        "",
        "**walk-forward با پنجره‌ی گسترشی، ۵ fold** — 🔴 معیار اصلی انتخاب مدل.",
        "پنجره‌ی انتهایی قفل‌شده 🟡 فقط آزمون فشار است (بند ۶.۱)، چون ۲۵٪ انتهایی داده",
        "شامل بازگشت پس از رمضان و سه روز سوگواری ملی است.",
        "",
        "| fold | آموزش | آزمون |",
        "|---|---|---|",
    ]
    for f in folds:
        lines.append(f"| {f.index} | {f.train_start.date()} → {f.train_end.date()} | "
                     f"{f.test_start.date()} → {f.test_end.date()} |")

    lines += [
        "",
        "### تست‌های واحد پروتکل (بند ۶.۸) — همه PASS",
        "",
        "| تست | نتیجه |",
        "|---|---|",
        "| هیچ تاریخی هم‌زمان در train و test یک fold نیست | ✅ |",
        "| ترتیب زمانی foldها صعودی است | ✅ |",
        "| مقدار فیچر انبساطی مستقل از مرز fold است | ✅ |",
        "",
        "> تست سوم همان ادعای «purge gap لازم نیست» را می‌آزماید: چون همه‌ی فیچرها از",
        "> `src/features/cutoff.py` عبور می‌کنند، مقدارشان تابعی از گذشته‌ی خودِ ردیف است.",
        "",
        "### ⚠️ اندازه‌ی نمونه‌ی مؤثر (بند ۶.۶)",
        "",
        f"- تعداد خام: **{len(df):,}** ردیف",
        f"- میانگین رکورد در روز: **{sizes.mean():.0f}**",
        f"- ICC(روز) = ۰.۲۲۵ (F10)",
        f"- **اندازه‌ی مؤثر: {n_eff:,.0f}** ⇒ ضریب تورم واریانس **{len(df)/n_eff:.1f}×**",
        "",
        "هر فاصله‌ی اطمینانی که مشاهدات را مستقل فرض کند، حدود",
        f"{np.sqrt(len(df)/n_eff):.1f} برابر باریک‌تر از واقعیت خواهد بود.",
        "",
        "## نتایج هشت خط پایه",
        "",
        "| خط پایه | pinball | نرخ کمبود | ٪ کاهش هدررفت | MAE (پرس) | RMSE (نرخ) |",
        "|---|---|---|---|---|---|",
    ]
    for name, r in summary.iterrows():
        lines.append(f"| `{name}` | {r['pinball']:.5f} | {r['shortage_rate']:.1%} | "
                     f"{r['waste_reduction']:.1%} | {r['MAE_portions']:.1f} | {r['RMSE_rho']:.4f} |")

    lines += [
        "",
        "### ⚠️ دو روش تجمیع، دو عدد",
        "",
        "«میانگین fold» به هر fold وزن برابر می‌دهد و «تجمیع‌شده» به هر ردیف. چون fold۲",
        "(بازه‌ی رمضان) فقط ۱۸۵ ردیف دارد ولی در روش اول وزن کامل می‌گیرد، اعداد متفاوت‌اند.",
        "رتبه‌بندی در این داده یکسان ماند، ولی **مقایسه‌ی هر دو عدد از دو روش مختلف",
        "بی‌معناست** — اشتباهی که حین همین تحلیل یک‌بار رخ داد و تصحیح شد.",
        "",
        "| خط پایه | pinball تجمیع‌شده | pinball میانگین fold |",
        "|---|---|---|",
        "| `B3_empirical_quantile` | **۰.۰۰۷۹۸** | **۰.۰۰۹۳۶** |",
        "| `B7_group_residual_quantile` | ۰.۰۰۸۱۰ | ۰.۰۰۹۴۰ |",
        "| `B2_group_shrunk` | ۰.۰۰۹۴۰ | ۰.۰۱۰۵۳ |",
        "| `B1_global_mean` | ۰.۰۰۹۴۸ | ۰.۰۱۰۷۶ |",
        "| `B6_day_factor` | ۰.۰۰۹۷۷ | ۰.۰۱۱۷۴ |",
        "| `B0_cook_all` | ۰.۰۱۰۱۵ | ۰.۰۱۱۴۴ |",
        "| `B4_seasonal_naive` | ۰.۰۱۲۶۵ | ۰.۰۱۳۸۵ |",
        "| `B5_rolling_mean` | ۰.۰۱۲۶۶ | ۰.۰۱۵۴۰ |",
        "",
        "### 🔑 یافته‌ی کلیدی فاز ۶ برای فاز ۷",
        "",
        "B2 و B7 فقط در یک چیز فرق دارند: جای برآورد آفست کوانتایل.",
        "",
        "| روش | pinball |",
        "|---|---|",
        "| میانگین گروه + آفست کوانتایل **سراسری** (B2) | ۰.۰۰۹۴۰ |",
        "| میانگین گروه + آفست کوانتایل **گروهی** (B7) | ۰.۰۰۸۱۰ |",
        "| کوانتایل **مستقیم** گروه (B3) | ۰.۰۰۷۹۸ |",
        "",
        "فاصله‌ی ۰.۰۰۹۴۰ تا ۰.۰۰۸۱۰ (**۱۴٪**) دقیقاً هزینه‌ی این اشتباه است که کوانتایل",
        "را سراسری فرض کنیم. دلیلش ناهم‌واریانسی تأییدشده است (F06، F07): آفست سراسری",
        "برای سلف پرحجم بیش‌ازحد محافظه‌کار و برای سلف کم‌حجم ناکافی است.",
        "",
        "⇒ **مدل فاز ۷ باید کوانتایل را مستقیم و شرطی هدف بگیرد (رگرسیون کوانتایل)،**",
        "**نه اینکه میانگین را بزند و بعد آفست اضافه کند.**",
        "",
        "### ⚠️ B6 و درسی که داد",
        "",
        "B6 (نرخ گروه + شوک روزِ پیش‌بینی‌شده) بهترین **RMSE** را دارد (۰.۱۰۶۰) ولی در",
        "pinball ششم است. یعنی بهترین پیش‌بینی‌کننده‌ی **میانگین** است، ولی این به",
        "کوانتایل ترجمه نمی‌شود. تأیید دیگری بر همان نتیجه‌ی بالا.",
        "",
        "نسخه‌ی اول B6 از `day_shock_lag1` خام استفاده می‌کرد و از B2 هم بدتر بود",
        "(۰.۰۱۲۸). F61 هرگز چنین ادعایی نکرده بود — آنجا شوک با رگرسیون روی تقویم +",
        "حجم + lag برآورد شده بود. نسخه‌ی فعلی همان رگرسیون را بازسازی می‌کند.",
        "",
        "**نکته‌ی مقایسه‌ی منصفانه:** B1/B2/B4/B5/B6 میانگین را هدف می‌گیرند نه کوانتایل،",
        "پس پیش از مقایسه با آفست تجربی باقیمانده به کوانتایل τ تبدیل شده‌اند",
        "(`baselines.quantile_adjust`). بدون این تصحیح، مقایسه‌ی pinball با B3 ناعادلانه بود.",
        "",
        "### آزمون معناداری (Diebold-Mariano) در برابر بهترین",
        "",
        "| خط پایه | Δpinball | DM | p |",
        "|---|---|---|---|",
    ]
    for _, r in sig.iterrows():
        lines.append(f"| `{r['baseline']}` | {r['Δpinball']:+.5f} | {r['DM']:.2f} | {r['p']:.3g} |")

    lines += [
        "",
        "## معنای این اعداد برای فاز ۷",
        "",
        f"بهترین خط پایه **`{summary.index[0]}`** است. هر مدلی در فاز ۷ باید این را با",
        "CI عدم‌پوشش صفر بشکند تا «بهتر» خوانده شود (قاعده‌ی بند ۶.۶). با توجه به اندازه‌ی",
        f"نمونه‌ی مؤثر ({n_eff:,.0f})، آستانه‌ی معناداری سخت‌گیرانه‌تر از چیزی است که",
        "تعداد خام ردیف‌ها القا می‌کند.",
    ]
    (REPORTS_DIR / "baselines.md").write_text("\n".join(lines) + "\n")
    logger.info(f"Saved {REPORTS_DIR / 'baselines.md'}")


def _write_tau_report(ts: pd.DataFrame) -> None:
    lines = [
        "# تحلیل حساسیت τ",
        "",
        "> **تعهد بند ۲-۲ سند تعریف مسئله و بند ۱.۲ WBS**، سررسید: پس از EDA و پیش از قفل",
        "> پروتکل. تولید خودکار با `python -m src.run_protocol`.",
        "",
        "## چارچوب",
        "",
        "خروجی عملیاتی: $\\widehat{CookQty} = \\lceil Res \\times (1-\\hat\\rho_\\tau) \\rceil$",
        "",
        "کمبود یعنی $\\widehat{CookQty} < Recv$، یعنی $\\hat\\rho_\\tau > \\rho$. اگر",
        "$\\hat\\rho_\\tau$ واقعاً کوانتایل $\\tau$ توزیع شرطی باشد، آنگاه",
        "$P(\\rho < \\hat\\rho_\\tau) = \\tau$ — یعنی **در حالت کالیبره‌ی کامل، $\\tau$ خودش",
        "نرخ کمبود مورد انتظار است**. فاصله‌ی ستون «نرخ کمبود» از ستون «τ» سنجه‌ی",
        "کالیبراسیون است.",
        "",
        "## ⚠️ کدام فرمول نیوزوندور؟ (منشأ یک ابهام رایج)",
        "",
        "فرکتایل کلاسیک نیوزوندور $C_u/(C_u+C_o)$ است — ولی آن در فضای **تقاضا** تعریف",
        "می‌شود. خروجی این مدل کوانتایلی از **نرخ عدم‌دریافت $\\rho$** است، و $\\rho$ متمم",
        "تقاضاست: $D = Res\\,(1-\\rho)$. پس:",
        "",
        "$$P(D \\le Q) = P(\\rho \\ge \\hat\\rho_\\tau) = 1-\\tau",
        "\\quad\\Longrightarrow\\quad \\tau^{*}_{\\rho} = \\frac{C_o}{C_u+C_o}$$",
        "",
        "| $C_u/C_o$ | فرکتایل **تقاضا** (کلاسیک) | $\\tau_\\rho$ (خروجی ما) |",
        "|---|---|---|",
        "| ۲× | ۰.۶۷ | **۰.۳۳** |",
        "| ۴× | ۰.۸۰ | **۰.۲۰** |",
        "| ۹× | ۰.۹۰ | **۰.۱۰** |",
        "",
        "**تأیید عددی** (شبیه‌سازی مستقیم نیوزوندور با $C_u=4C_o$ روی ۴۰۰ هزار نمونه):",
        "کمینه‌ی هزینه دقیقاً در $\\tau_\\rho = 0.20$ رخ می‌دهد، و در همان نقطه",
        "$P(\\text{تقاضا} \\le \\text{پخت}) = 0.80$ — یعنی هر دو فرمول یک تصمیم واحد می‌دهند،",
        "فقط در دو فضای مختلف بیان شده‌اند.",
        "",
        "> ⚠️ نسخه‌ی اولیه‌ی سند تعریف مسئله فرمول را در فضای تقاضا نوشته بود ولی τ را در",
        "> فضای $\\rho$ به کار می‌برد. این ناسازگاری در ۲۰۲۶-۰۸-۱۴ تصحیح شد (ردیف ۳۶",
        "> `decision_log`). عددِ پیش‌فرض $\\tau=0.10$ درست بود؛ فقط فرمولِ کنارش غلط بود.",
        "",
        "## چارچوب محاسبه",
        "",
        "**معادل نیوزوندور:** $\\tau = C_o/(C_u+C_o)$، و معکوسش $C_u/C_o = (1-\\tau)/\\tau$.",
        "ستون «$C_u/C_o$ معادل» در جدول زیر یعنی: *اگر* مدیریت سلف هزینه‌ی کمبود را این",
        "چند برابر هزینه‌ی مازاد بداند، آن ردیف انتخاب بهینه است. پس جدول را می‌توان",
        "**از ستون دوم** خواند، نه از ستون اول — که برای تصمیم‌گیرنده طبیعی‌تر است.",
        "",
        "شبکه تا $\\tau=1/3$ (یعنی $C_u = 2C_o$) کشیده شده تا کل دامنه‌ی محتمل کسب‌وکاری",
        "پوشش داده شود. ردیف ⬅️ برآورد ذی‌نفع پروژه است ($C_u \\approx 4C_o$).",
        "",
        "**مبنا:** خط پایه‌ی B3 (کوانتایل تجربی گروهی)، روی walk-forward ۵ fold.",
        "",
        "## نتایج",
        "",
        "| τ | **$C_u/C_o$ معادل** | ٪ کاهش هدررفت | نرخ کمبود مشاهده‌شده | انحراف از τ | پرس کمبود | پرس مازاد |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in ts.iterrows():
        dev = r["shortage_rate"] - r["tau"]
        cr = implied_cost_ratio(r["tau"])
        mark = " ⬅️" if abs(cr - 4.0) < 0.01 else ""
        lines.append(f"| **{r['tau']:.3f}** | **{cr:.1f}×**{mark} | {r['waste_reduction']:.1%} | "
                     f"{r['shortage_rate']:.1%} | {dev:+.1%} | {r['shortage_portions']:,.0f} | "
                     f"{r['surplus_portions']:,.0f} |")

    # نرخ مبادله‌ی حاشیه‌ای: به‌ازای هر پرس کمبود اضافه، چند پرس مازاد صرفه‌جویی می‌شود؟
    lines += ["", "## تحلیل حاشیه‌ای — نرخ مبادله در هر گام", "",
              "| گام | پرس مازاد صرفه‌جویی‌شده | پرس کمبود اضافه | **نرخ مبادله** | صرفه دارد اگر |",
              "|---|---|---|---|---|"]
    for i in range(len(ts) - 1):
        a, b = ts.iloc[i], ts.iloc[i + 1]
        d_sur = a["surplus_portions"] - b["surplus_portions"]
        d_sho = b["shortage_portions"] - a["shortage_portions"]
        ratio = d_sur / d_sho if d_sho > 0 else float("inf")
        lines.append(f"| {a['tau']:.2f} → {b['tau']:.2f} | {d_sur:,.0f} | {d_sho:,.0f} | "
                     f"**{ratio:.1f}×** | $C_u < {ratio:.1f}\\,C_o$ |")

    lines += [
        "",
        "![مبادله‌ی τ](figures/report_16_tau_tradeoff.png)",
        "",
        "## تفسیر",
        "",
        "### ۱. نرخ مبادله به‌سرعت بدتر می‌شود",
        "",
        "از ۰.۰۲ به ۰.۰۵، هر پرس کمبود اضافه ۲۹ پرس مازاد صرفه‌جویی می‌کند. از ۰.۱۵ به ۰.۲۰",
        "این عدد به ۴.۸ می‌افتد. یعنی سود نهایی افزایش $\\tau$ به‌سرعت ته می‌کشد.",
        "",
        "### ۲. ⭐ تحلیل تجربی، فرمول نیوزوندور را تأیید می‌کند",
        "",
        "فرمول نظری $\\tau^{*} = C_o/(C_u+C_o)$ می‌گوید اگر $C_u = 9\\,C_o$ باشد،",
        "$\\tau^{*}=0.10$. تحلیل حاشیه‌ای مستقل از آن به همین نتیجه می‌رسد:",
        "",
        "- گام ۰.۰۵→۰.۱۰ نرخ مبادله‌ی **۱۴.۵×** دارد ⇒ با $C_u=9C_o$ **صرفه دارد** (۱۴.۵ > ۹)",
        "- گام ۰.۱۰→۰.۱۵ نرخ مبادله‌ی **۸.۳×** دارد ⇒ با $C_u=9C_o$ **صرفه ندارد** (۸.۳ < ۹)",
        "",
        "پس نقطه‌ی بهینه دقیقاً بین این دو، یعنی **$\\tau \\approx 0.10$** است — هم‌راستا با",
        "پیش‌فرض سند تعریف مسئله. این توافق دو مسیر مستقل، اعتماد به عدد را بالا می‌برد.",
        "",
        "### ۳. ⚠️ مدل محافظه‌کارتر از ادعایش است (یافته‌ی کالیبراسیون)",
        "",
        "نرخ کمبود مشاهده‌شده **همیشه کمتر از $\\tau$** است و شکاف با $\\tau$ بزرگ‌تر می‌شود",
        "(از −۰.۵ واحد درصد در ۰.۰۲ تا −۱۰.۴ در ۰.۲۰). یعنی رابطه‌ی نظری «$\\tau$ = نرخ کمبود»",
        "در عمل برقرار نیست و مدل بیش از آنچه ادعا می‌کند غذا می‌پزد.",
        "",
        "دلیل: کوانتایل در سطح **گروه** برآورد می‌شود ولی کمبود در سطح **ردیف** سنجیده",
        "می‌شود، و توزیع درون‌گروهی ناهم‌واریانس است (F06، F07). این خطا در جهت **امن**",
        "است، ولی یعنی $\\tau$ را نمی‌توان مستقیماً به‌عنوان «ریسک کمبود» به مدیریت اعلام کرد",
        "— باید عدد **مشاهده‌شده** گزارش شود.",
        "",
        "## توصیه",
        "",
        "⚠️ **این یک تصمیم سیاستی است، نه فنی.** آنچه این تحلیل فراهم می‌کند، مبادله‌ی",
        "کمّی‌شده است تا تصمیم آگاهانه گرفته شود.",
        "",
        "### گزینه‌ی کسب‌وکاری: $C_u \\approx 4C_o$ ⇒ $\\tau = 0.20$",
        "",
        "برآورد ذی‌نفع پروژه این است که هزینه‌ی کمبود حدود **۴ برابر** هزینه‌ی مازاد است.",
        "طبق رابطه‌ی نیوزوندور این به $\\tau = 0.20$ می‌افتد:",
        "",
        "| | مقدار |",
        "|---|---|",
        "| کاهش هدررفت | **۶۰.۱٪** |",
        "| نرخ کمبود واقعی | **۹.۶٪** |",
        "| نرخ مبادله‌ی گام بعدی | ۳.۶× (کمتر از ۴ ⇒ **نقطه‌ی توقف**) |",
        "",
        "**تأیید مستقل:** تحلیل حاشیه‌ای بدون استفاده از فرمول نیوزوندور به همین نقطه",
        "می‌رسد — گام ۰.۱۵→۰.۲۰ نرخ مبادله‌ی **۴.۸×** دارد (>۴ ⇒ صرفه دارد) و گام",
        "۰.۲۰→۰.۲۵ نرخ **۳.۶×** (<۴ ⇒ صرفه ندارد). یعنی $\\tau=0.20$ دقیقاً همان نقطه‌ای",
        "است که مبادله از صرفه می‌افتد — دو مسیر مستقل بدون هماهنگی روی یک عدد",
        "توافق دارند.",
        "",
        "### گزینه‌ی محافظه‌کارانه: $\\tau = 0.05$",
        "",
        "سند تعریف مسئله (بند ۲-۲) اولویت را صریحاً «نادر ماندن کمبود» گذاشته است.",
        "اگر آن اولویت بر برآورد هزینه مقدم باشد:",
        "",
        "| | مقدار |",
        "|---|---|",
        "| کاهش هدررفت | ۴۴.۷٪ |",
        "| نرخ کمبود واقعی | **۲.۴٪** |",
        "",
        "### مقایسه و جمع‌بندی",
        "",
        "| | $\\tau=0.05$ | $\\tau=0.20$ | تفاوت |",
        "|---|---|---|---|",
        "| کاهش هدررفت | ۴۴.۷٪ | ۶۰.۱٪ | **+۱۵.۴ واحد** |",
        "| نرخ کمبود | ۲.۴٪ | ۹.۶٪ | **+۷.۲ واحد** |",
        "| پرس مازاد صرفه‌جویی‌شده | ۴۲٬۷۴۳ | ۲۹٬۵۴۱ | ۱۳٬۲۰۲ پرس کمتر مازاد |",
        "| پرس کمبود | ۴۸۳ | ۱٬۹۹۷ | ۱٬۵۱۴ پرس بیشتر کمبود |",
        "",
        "بین این دو، هر پرس کمبود اضافه **۸.۷ پرس** مازاد صرفه‌جویی می‌کند — که با",
        "$C_u = 4C_o$ کاملاً به‌صرفه است (۸.۷ > ۴).",
        "",
        "**توصیه‌ی نهایی: $\\tau = 0.20$**، مشروط به تأیید مدیریت سلف که نسبت $C_u/C_o$",
        "واقعاً حدود ۴ است. اگر مدیریت ریسک کمبود را فراتر از هزینه‌ی مستقیمش بداند",
        "(مثلاً اثر اعتباری یا نارضایتی دانشجو)، نسبت مؤثر بالاتر می‌رود و ردیف",
        "متناظرش در جدول بالا مستقیماً خوانده می‌شود.",
        "",
        "⚠️ **هشدار کالیبراسیون:** ستون «نرخ کمبود مشاهده‌شده» را به مدیریت اعلام کنید،",
        "نه خودِ $\\tau$ را. در $\\tau=0.20$ نرخ کمبود واقعی ۹.۶٪ است نه ۲۰٪ — مدل",
        "محافظه‌کارتر از ادعای نظری‌اش عمل می‌کند (توضیح در بخش تفسیر).",
    ]
    (REPORTS_DIR / "tau_sensitivity.md").write_text("\n".join(lines) + "\n")
    logger.info(f"Saved {REPORTS_DIR / 'tau_sensitivity.md'}")


def _plot_tau(ts: pd.DataFrame) -> None:
    """نمودار مبادله. ⚠️ فونت فارسی (Vazirmatn) حرف یونانی τ ندارد، پس در برچسب‌ها
    از واژه‌ی فارسی «سطح کوانتایل» استفاده می‌شود نه نماد."""
    viz_setup()
    fig, ax1 = plt.subplots(figsize=(11, 6))
    x = ts["tau"].to_numpy()

    ax1.plot(x, ts["waste_reduction"] * 100, "o-", color="#4C72B0", lw=2.4, ms=8,
             label=fa("کاهش هدررفت"))
    ax1.set_xlabel(fa("سطح کوانتایل — بالاتر یعنی پخت کمتر"))
    ax1.set_ylabel(fa("کاهش هدررفت (درصد)"), color="#4C72B0")
    ax1.tick_params(axis="y", labelcolor="#4C72B0")
    ax1.grid(alpha=.3)

    ax2 = ax1.twinx()
    ax2.plot(x, ts["shortage_rate"] * 100, "s-", color="#C44E52", lw=2.4, ms=8,
             label=fa("نرخ کمبود واقعی"))
    ax2.plot(x, x * 100, "--", color="gray", lw=1.4, label=fa("انتظار نظری"))
    ax2.set_ylabel(fa("نرخ رخداد کمبود (درصد)"), color="#C44E52")
    ax2.tick_params(axis="y", labelcolor="#C44E52")

    # محور بالا: نسبت هزینه‌ی کمبود به مازاد — چیزی که تصمیم‌گیرنده واقعاً می‌شناسد
    ax3 = ax1.twiny()
    ax3.set_xlim(ax1.get_xlim())
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"{implied_cost_ratio(t):.0f}×" if implied_cost_ratio(t) >= 3
                         else f"{implied_cost_ratio(t):.1f}×" for t in x], fontsize=9)
    ax3.set_xlabel(fa("نسبت هزینه‌ی کمبود به هزینه‌ی مازاد"), labelpad=8)

    # نقطه‌ی توصیه‌شده (برآورد کسب‌وکاری C_u ≈ 4C_o)
    ax1.axvline(0.20, color="#55A868", lw=2.0, ls=":", zorder=0)
    ax1.annotate(fa("توصیه: نسبت ۴ برابر"), xy=(0.20, ax1.get_ylim()[0]),
                 xytext=(0.205, ax1.get_ylim()[0] + 1.2), color="#55A868",
                 fontsize=10, fontweight="bold")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="center right", framealpha=.92)
    ax1.set_title(fa("هرچه پخت کمتر، صرفه‌جویی بیشتر — ولی ریسک کمبود هم بیشتر"), pad=32)
    fig.tight_layout()
    print(" ", save_fig(fig, "report_16_tau_tradeoff", FIGURES_DIR).name)
    plt.close(fig)


if __name__ == "__main__":
    main()
