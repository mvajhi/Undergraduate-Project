"""بند ۶.۴ و ۶.۵ — معیارها و هشت خط پایه.

هر خط پایه یک تابع است که $\\hat\\rho$ برمی‌گرداند و **فقط** از داده‌ی آموزش تغذیه
می‌شود. خروجی عملیاتی همه‌شان یکسان است:

$$\\widehat{CookQty} = \\lceil Res \\times (1 - \\hat\\rho_\\tau) \\rceil$$

⚠️ **رابطه‌ی $\\tau$ و کمبود.** کمبود یعنی $\\widehat{CookQty} < Recv$، یعنی
$\\hat\\rho_\\tau > \\rho$. اگر $\\hat\\rho_\\tau$ واقعاً کوانتایل $\\tau$ توزیع شرطی
$\\rho$ باشد، آنگاه $P(\\rho < \\hat\\rho_\\tau) = \\tau$، پس **$\\tau$ خودش نرخ کمبود
مورد انتظار است**. انحراف مشاهده‌شده از این تساوی، سنجه‌ی کالیبراسیون است (بند ۶.۱۰).
"""

import numpy as np
import pandas as pd

#: پارامترهای پیشین Beta از بند ۴.۸ (F8.3) — برای کوچک‌سازی نرخ گروه‌های کم‌حجم
BETA_ALPHA, BETA_BETA = 0.9758, 11.1506

#: شبکه‌ی سناریوهای τ. هر مقدار معادل یک نسبت هزینه است: $\\tau = C_o/(C_u+C_o)$
#: ⇒ $C_u/C_o = (1-\\tau)/\\tau$. شبکه عمداً تا $\\tau=1/3$ (یعنی $C_u = 2C_o$) کشیده
#: شده تا کل دامنه‌ی محتمل کسب‌وکاری را پوشش دهد؛ برآورد ذی‌نفع پروژه $C_u \\approx 4C_o$
#: است که به $\\tau=0.20$ می‌افتد و در میانه‌ی این شبکه قرار می‌گیرد.
TAU_GRID = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 1 / 3]


def implied_cost_ratio(tau: float) -> float:
    """نسبت $C_u/C_o$ که این $\\tau$ برایش بهینه است (معکوس رابطه‌ی نیوزوندور)."""
    return (1.0 - tau) / tau


# ---------------------------------------------------------------------------
# بند ۶.۴ — معیارها
# ---------------------------------------------------------------------------

def pinball_loss(y: np.ndarray, y_hat: np.ndarray, tau: float) -> np.ndarray:
    """زیان pinball به‌ازای هر مشاهده (بردار، نه میانگین — برای Diebold-Mariano لازم است)."""
    e = np.asarray(y, dtype=float) - np.asarray(y_hat, dtype=float)
    return np.maximum(tau * e, (tau - 1) * e)


def cook_qty(res: np.ndarray, rho_hat: np.ndarray) -> np.ndarray:
    return np.ceil(np.asarray(res, dtype=float) * (1.0 - np.clip(rho_hat, 0.0, 1.0)))


def operational_metrics(df: pd.DataFrame, rho_hat: np.ndarray, tau: float) -> dict:
    """معیارها در **هر دو** فضای نرخ و تعداد پرس (قاعده‌ی حیاتی بند ۶.۴).

    خروجی این تابع **قرارداد ثبت MLflow** است: `src/models/tracking.py::log_metrics_dict`
    هر کلید غیر از `n` را به‌عنوان metric ثبت می‌کند. پس افزودن کلید جدید اینجا
    خودکار یعنی ثبتش در هر سه مرحله‌ی S0/S1/S2 — و تست
    `test_all_operational_metrics_reach_mlflow` تضمین می‌کند هیچ کلیدی دور ریخته نشود.
    """
    res, recv, rho = df["Res"].to_numpy(float), df["Recv"].to_numpy(float), df["rho"].to_numpy(float)
    rho_hat = np.asarray(rho_hat, dtype=float)
    qty = cook_qty(res, rho_hat)

    shortage = qty < recv
    surplus = np.maximum(qty - recv, 0.0)
    surplus_b0 = np.maximum(res - recv, 0.0)  # B0: پخت = کل رزرو
    demand = recv  # تقاضای واقعی = تعداد دریافت‌کننده

    # پوشش تجربی: کسری از سلول‌ها که کوانتایل پیش‌بینی‌شده واقعیت را پوشانده است.
    # اگر $\hat\rho_\tau$ واقعاً کوانتایل τ باشد، این عدد باید ≈ τ شود (بند ۶.۱۰).
    coverage = float((rho <= rho_hat).mean())

    # $R^2$ در فضای نرخ. ⚠️ برای یک پیش‌بینی‌کننده‌ی **کوانتایل** (نه میانگین) این عدد
    # ذاتاً پایین است و **معیار انتخاب مدل نیست** — نقشش در بند 7.9.2 «سیم‌چین نشتی»
    # است: سقف واقع‌بینانه ۰.۴–۰.۵ (بند ۵.۱۳)، و هر مقدار >۰.۹ یعنی توقف و ممیزی نشت.
    ss_res = float(((rho - rho_hat) ** 2).sum())
    ss_tot = float(((rho - rho.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "pinball": float(pinball_loss(rho, rho_hat, tau).mean()),
        # همان زیان در فضای پرس — بند ۶.۴ صریحاً هر دو فضا را می‌خواهد، چون
        # ۰.۰۰۱ نرخ در سلولی با Res=۱۵ و سلولی با Res=۸۰۰ دو چیز کاملاً متفاوت است.
        "pinball_portions": float((pinball_loss(rho, rho_hat, tau) * res).mean()),
        "MAE_rho": float(np.abs(rho - rho_hat).mean()),
        "RMSE_rho": float(np.sqrt(((rho - rho_hat) ** 2).mean())),
        "R2_rho": float(r2),
        "MAE_portions": float(np.abs(qty - recv).mean()),
        "RMSE_portions": float(np.sqrt(((qty - demand) ** 2).mean())),
        "coverage": coverage,
        "coverage_gap": coverage - tau,
        "shortage_rate": float(shortage.mean()),
        "shortage_portions": float(np.maximum(recv - qty, 0.0).sum()),
        "surplus_portions": float(surplus.sum()),
        "waste_reduction_pct": float(1.0 - surplus.sum() / surplus_b0.sum()) if surplus_b0.sum() > 0 else np.nan,
        "n": int(len(df)),
    }


# ---------------------------------------------------------------------------
# بند ۶.۵ — هفت خط پایه
# ---------------------------------------------------------------------------

def _shrunk_group_rate(train: pd.DataFrame, keys: list[str]) -> pd.Series:
    """نرخ وزنی کوچک‌شده‌ی بیزی هر گروه. 🔄 تغییر نسبت به WBS ۲.۰ که میانه‌ی خام می‌گفت.

    دلیل: F11 نشان داد دُم توزیع عمدتاً از گروه‌های کم‌حجم می‌آید (میانه‌ی Res بالای
    صدک ۹۹ فقط ۱۴ است) و میانه/کوانتایل خام آن گروه‌ها بی‌معناست.
    """
    g = train.groupby(keys, observed=True).agg(k=("NoRecv", "sum"), n=("Res", "sum"))
    return (g["k"] + BETA_ALPHA) / (g["n"] + BETA_ALPHA + BETA_BETA)


def b0_cook_all(train, test, tau):
    """پخت = کل رزرو (وضع موجود). $\\hat\\rho = 0$."""
    return np.zeros(len(test))


def b1_global_mean(train, test, tau):
    """میانگین سراسری وزنی تاریخی."""
    r = train["NoRecv"].sum() / train["Res"].sum()
    return np.full(len(test), r)


def b2_group_shrunk(train, test, tau):
    """نرخ کوچک‌شده‌ی سلف×وعده×روزهفته — baseline رسمی سند مسئله (بند ۲.۳)."""
    keys = ["RestaurantName", "Meal", "dow"]
    rates = _shrunk_group_rate(train, keys)
    fallback = train["NoRecv"].sum() / train["Res"].sum()
    idx = pd.MultiIndex.from_frame(test[keys])
    return rates.reindex(idx).fillna(fallback).to_numpy()


def b3_empirical_quantile(train, test, tau):
    """کوانتایل تجربی $\\tau$ همان گروه — ✅ رقیب سرسخت (آگاه از عدم‌تقارن)."""
    keys = ["RestaurantName", "Meal", "dow"]
    q = train.groupby(keys, observed=True)["rho"].quantile(tau)
    fallback = train["rho"].quantile(tau)
    idx = pd.MultiIndex.from_frame(test[keys])
    return q.reindex(idx).fillna(fallback).to_numpy()


def b4_seasonal_naive(train, test, tau):
    """$\\hat\\rho_d = \\rho_{d-7}$ همان سلف×وعده — baseline سری زمانی."""
    col = "rho_cell_lag7"
    fallback = train["NoRecv"].sum() / train["Res"].sum()
    return test[col].fillna(fallback).to_numpy() if col in test else np.full(len(test), fallback)


def b5_rolling_mean(train, test, tau):
    """میانگین متحرک ۷ روزه‌ی همان سلف×وعده."""
    col = "rho_roll_mean_7"
    fallback = train["NoRecv"].sum() / train["Res"].sum()
    return test[col].fillna(fallback).to_numpy() if col in test else np.full(len(test), fallback)


def b6_day_factor(train, test, tau):
    """➕ نرخ تاریخی سلف×وعده **+ شوک روزِ پیش‌بینی‌شده** — رقیب جدی‌تر از B3.

    منطق: F59 نشان داد ۸۳٪ واریانس سلول شوک مشترک روزانه است و F61 نشان داد همین شوک
    با اطلاعات لحظه‌ی برش $R^2$ خارج‌نمونه ۰.۶۰ دارد. اگر مدل فاز ۷ این را نبرد،
    پیچیدگی اضافه‌اش توجیه ندارد.

    ⚠️ **شوک باید پیش‌بینی شود، نه اینکه lag خامش مستقیم جمع شود.** نسخه‌ی اول این
    خط پایه از `day_shock_lag1` خام استفاده می‌کرد و بدتر از B2 شد (pinball ۰.۰۱۲۸ در
    برابر ۰.۰۱۰۵) — چون lag خام نویزی است و F61 هرگز چنین ادعایی نکرده بود؛ آنجا شوک
    با یک رگرسیون روی تقویم + حجم رزرو + lag برآورد شده بود. اینجا همان رگرسیون
    (فقط روی داده‌ی آموزش) بازسازی می‌شود.
    """
    base = b2_group_shrunk(train, test, tau)
    feats = ["day_shock_lag1", "log_daily_total_res", "is_day_before_holiday",
             "is_exam_period", "dow"]
    if not all(c in train.columns for c in feats):
        return np.clip(base, 0.0, 1.0)

    # هدف رگرسیون: انحراف نرخ سلول از عادت همان سلف×وعده (یعنی همان «شوک»)
    tr = train.dropna(subset=feats + ["rho"])
    if len(tr) < 200:
        return np.clip(base, 0.0, 1.0)
    y = tr["rho"] - tr.groupby(["RestaurantName", "Meal"], observed=True)["rho"].transform("mean")
    X = pd.get_dummies(tr[feats], columns=["dow"], drop_first=True).astype(float)

    from sklearn.linear_model import Ridge
    model = Ridge(alpha=1.0).fit(X, y)

    te = test.copy()
    Xte = pd.get_dummies(te[feats].fillna({"day_shock_lag1": 0.0}), columns=["dow"], drop_first=True)
    Xte = Xte.reindex(columns=X.columns, fill_value=0.0).astype(float).fillna(0.0)
    return np.clip(base + model.predict(Xte), 0.0, 1.0)


def b7_group_residual_quantile(train, test, tau):
    """➕ نرخ کوچک‌شده‌ی گروه + کوانتایل باقیمانده‌ی **همان گروه** (نه آفست سراسری).

    **چه چیزی را نشان می‌دهد.** تفاوت این با B2 فقط یک چیز است: آفست کوانتایل به‌جای
    یک عدد سراسری، به تفکیک سلف×وعده برآورد می‌شود. همین تفاوت pinball را از ۰.۰۰۹۴۰
    به **۰.۰۰۸۱۰** می‌برد (تجمیع‌شده) — یعنی **جای درستِ آفست کوانتایل، سطح گروه است،
    نه سطح کل داده**. دلیلش ناهم‌واریانسی تأییدشده است (F06: نسبت std چارک کوچک به
    بزرگ ۳.۰۳؛ F07: بیش‌پراکندگی صعودی از ۳.۷× به ۱۵.۶×): یک آفست سراسری برای سلف
    پرحجم بیش‌ازحد محافظه‌کار و برای سلف کم‌حجم ناکافی است.

    ⚠️ **ولی B3 را نمی‌برد.** B3 = ۰.۰۰۷۹۸ در برابر B7 = ۰.۰۰۸۱۰ (تجمیع‌شده)؛ آزمون
    Diebold-Mariano تفاوت را معنادار نمی‌داند (p=۰.۱۴). یعنی «کوانتایل مستقیم گروه»
    و «میانگین گروه + کوانتایل باقیمانده‌ی گروه» عملاً هم‌ارزند — که منطقی است، چون
    هر دو در نهایت کوانتایل شرطی همان گروه را تخمین می‌زنند.

    **پیامد واقعی برای فاز ۷:** مدل باید کوانتایل را **مستقیم و شرطی** هدف بگیرد
    (رگرسیون کوانتایل)، نه اینکه میانگین را بزند و یک آفست سراسری اضافه کند —
    فاصله‌ی ۰.۰۰۹۴۰ تا ۰.۰۰۸۱۰ دقیقاً هزینه‌ی همین اشتباه است.
    """
    base = b2_group_shrunk(train, test, tau)
    keys = ["RestaurantName", "Meal"]
    resid = train["rho"].to_numpy() - b2_group_shrunk(train, train, tau)
    rq = pd.Series(resid, index=train.index).groupby([train[k] for k in keys]).quantile(tau)
    idx = pd.MultiIndex.from_frame(test[keys])
    offset = rq.reindex(idx).fillna(np.quantile(resid, tau)).to_numpy()
    return np.clip(base + offset, 0.0, 1.0)


BASELINES = {
    "B0_cook_all": b0_cook_all,
    "B1_global_mean": b1_global_mean,
    "B2_group_shrunk": b2_group_shrunk,
    "B3_empirical_quantile": b3_empirical_quantile,
    "B4_seasonal_naive": b4_seasonal_naive,
    "B5_rolling_mean": b5_rolling_mean,
    "B6_day_factor": b6_day_factor,
    "B7_group_residual_quantile": b7_group_residual_quantile,
}


def quantile_adjust(rho_hat: np.ndarray, train: pd.DataFrame, tau: float,
                    baseline_fn=None) -> np.ndarray:
    """تبدیل یک پیش‌بینی **میانگین‌محور** به کوانتایل $\\tau$ با افزودن آفست تجربی باقیمانده.

    B1/B2/B4/B5/B6 میانگین را هدف می‌گیرند نه کوانتایل. برای مقایسه‌ی منصفانه با B3
    (که ذاتاً کوانتایل است)، آفست $q_\\tau(\\rho - \\hat\\rho)$ روی **آموزش** برآورد و
    اضافه می‌شود. بدون این تصحیح، مقایسه‌ی pinball بین خط پایه‌ها ناعادلانه است.
    """
    if baseline_fn is None:
        return rho_hat
    resid = train["rho"].to_numpy() - baseline_fn(train, train, tau)
    return np.clip(rho_hat + np.quantile(resid, tau), 0.0, 1.0)
