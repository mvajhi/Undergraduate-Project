"""بند 7.10 سند فاز ۷ — خانواده‌ی ۱: خطی و تعمیم‌یافته (F01).

این نسخه سطح **L1** (سلول) را پوشش می‌دهد — عضو #۱۷ (Logistic/Probit/Cloglog روی L5) به
دلیل تفاوت اساسی منبع داده و هدف (رزرو فردی، هدف باینری) در ماژول جداگانه‌ای می‌آید؛
بند 7.9.1 «سطح داده × خانواده» را محور اول پیمایش می‌داند، پس L1×F01 ابتدا کامل می‌شود.

فیچرست S0/S1: ``FS_day`` (فیچرست پیش‌فرض بند 7.9.1)؛ فیچرست اختصاصی هر مدل (پایه‌های
فوریه + برهم‌کنش صریح، بند 7.5.3) در اسپرینت S2 اضافه می‌شود — اینجا فقط امکان‌سنجی است.

هر ``fit_predict_*`` امضای ``(train, test, tau, **hyperparams) -> np.ndarray`` دارد، دقیقاً
مثل ``src/baselines.py`` (بند ۶.۵)، تا هارنس S0/S1/S2 خط پایه‌ها و مدل‌های فاز ۷ را یکسان
صدا بزند.
"""

import json
from functools import lru_cache

import numpy as np
import optuna
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.optimize import linprog
from scipy.sparse import block_diag, csr_matrix, hstack, identity, vstack
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, LogisticRegression, QuantileRegressor, Ridge
from sklearn.preprocessing import StandardScaler

from src.models.families import common
from src.models.registry import ModelSpec, register
from src.models.spaces import register_space

FAMILY = "F01"
FEATURE_SET_S0 = "FS_day"
LEVEL = "L1"


@lru_cache(maxsize=1)
def _feature_cols() -> list[str]:
    from src.features.build import FEATURE_SETS_PATH
    return json.loads(FEATURE_SETS_PATH.read_text())[FEATURE_SET_S0]


def _design(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return common.design_matrix(train, test, _feature_cols())


def _add_const(Xtr: pd.DataFrame, Xte: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    Xtr_c = sm.add_constant(Xtr, has_constant="add")
    Xte_c = sm.add_constant(Xte, has_constant="add").reindex(columns=Xtr_c.columns, fill_value=0.0)
    return Xtr_c, Xte_c


# ---------------------------------------------------------------------------
# ۱–۶: OLS و منظم‌شده‌ها — همه از مسیر Q3 (کوانتایل باقیمانده به تفکیک چارک Res)
# ---------------------------------------------------------------------------

def fit_predict_ols(train: pd.DataFrame, test: pd.DataFrame, tau: float, **hp) -> np.ndarray:
    """⚠️ F02: چولگی ۴.۰۶، کشیدگی ۳۱.۹ — OLS نامناسب است، فقط به‌عنوان مرجع مطلق."""
    Xtr, Xte = _design(train, test)
    model = LinearRegression().fit(Xtr, train["rho"])
    return common.residual_quantile_by_res_quartile(train, test, model.predict(Xtr), model.predict(Xte), tau)


def fit_predict_ridge(train: pd.DataFrame, test: pd.DataFrame, tau: float, alpha: float = 1.0, **hp) -> np.ndarray:
    """پاسخ مستقیم به F44 (VIF=۱۸۲.۹) — مهار هم‌خطی."""
    Xtr, Xte = _design(train, test)
    scaler = StandardScaler().fit(Xtr)
    Ztr, Zte = scaler.transform(Xtr), scaler.transform(Xte)
    model = Ridge(alpha=alpha).fit(Ztr, train["rho"])
    return common.residual_quantile_by_res_quartile(train, test, model.predict(Ztr), model.predict(Zte), tau)


def fit_predict_lasso(train: pd.DataFrame, test: pd.DataFrame, tau: float, alpha: float = 1.0, **hp) -> np.ndarray:
    Xtr, Xte = _design(train, test)
    scaler = StandardScaler().fit(Xtr)
    Ztr, Zte = scaler.transform(Xtr), scaler.transform(Xte)
    model = Lasso(alpha=alpha, max_iter=5000).fit(Ztr, train["rho"])
    return common.residual_quantile_by_res_quartile(train, test, model.predict(Ztr), model.predict(Zte), tau)


def fit_predict_elasticnet(train: pd.DataFrame, test: pd.DataFrame, tau: float,
                           alpha: float = 1.0, l1_ratio: float = 0.5, **hp) -> np.ndarray:
    Xtr, Xte = _design(train, test)
    scaler = StandardScaler().fit(Xtr)
    Ztr, Zte = scaler.transform(Xtr), scaler.transform(Xte)
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000).fit(Ztr, train["rho"])
    return common.residual_quantile_by_res_quartile(train, test, model.predict(Ztr), model.predict(Zte), tau)


def fit_predict_adaptive_lasso(train: pd.DataFrame, test: pd.DataFrame, tau: float,
                               gamma: float = 1.0, alpha: float = 1.0, **hp) -> np.ndarray:
    """وزن‌دهی معکوس قدرمطلق ضرایب Ridge اولیه (oracle property). ترفند تبدیل:
    با تعریف $pw_j = 1/|\\hat\\beta_{ridge,j}|^\\gamma$ (وزن جریمه) و $w_j = 1/pw_j$، مسئله‌ی
    وزن‌دار روی $\\tilde X = X \\cdot w$ دقیقاً یک Lasso استاندارد می‌شود — و پیش‌بینی با
    $\\tilde X$ مستقیماً قابل‌محاسبه است، بدون نیاز به بازگرداندن ضرایب به مقیاس اصلی.
    """
    Xtr, Xte = _design(train, test)
    scaler = StandardScaler().fit(Xtr)
    Ztr, Zte = scaler.transform(Xtr), scaler.transform(Xte)

    ridge = Ridge(alpha=1.0).fit(Ztr, train["rho"])
    w = np.clip(np.abs(ridge.coef_), 1e-3, None) ** gamma
    Ztr_w, Zte_w = Ztr * w, Zte * w

    lasso = Lasso(alpha=alpha, max_iter=5000).fit(Ztr_w, train["rho"])
    return common.residual_quantile_by_res_quartile(train, test, lasso.predict(Ztr_w), lasso.predict(Zte_w), tau)


def fit_predict_group_lasso(train: pd.DataFrame, test: pd.DataFrame, tau: float,
                            group_reg: float = 0.05, l1_reg: float = 0.05, **hp) -> np.ndarray:
    """`restaurant_id`-گونه یک‌هات‌ها باید کامل وارد/خارج شوند (بند 7.10.1 عضو ۶)."""
    from group_lasso import GroupLasso

    Xtr, Xte = _design(train, test)
    groups = common.category_groups(_feature_cols(), train, Xtr.columns)
    scaler = StandardScaler().fit(Xtr)
    Ztr, Zte = scaler.transform(Xtr), scaler.transform(Xte)

    model = GroupLasso(groups=groups, group_reg=group_reg, l1_reg=l1_reg,
                       n_iter=200, supress_warning=True, random_state=42)
    model.fit(Ztr, train[["rho"]].to_numpy())
    mu_tr, mu_te = model.predict(Ztr).ravel(), model.predict(Zte).ravel()
    return common.residual_quantile_by_res_quartile(train, test, mu_tr, mu_te, tau)


# ---------------------------------------------------------------------------
# ۷–۹: رگرسیون کوانتایل — مسیر Q1 (بومی)
# ---------------------------------------------------------------------------

def fit_predict_quantile_regression(train: pd.DataFrame, test: pd.DataFrame, tau: float, **hp) -> np.ndarray:
    """⭐ کمینه‌سازی مستقیم معیار اصلی — `statsmodels.QuantReg` (بند 7.10.1 عضو ۷).

    مقیاس‌بندی پیش از برازش: حل‌کننده‌ی IRLS این مدل با فیچرهای هم‌مقیاس‌نشده به‌کندی
    همگرا می‌شود (مثل رگرسیون لجستیک در Hurdle).
    """
    Xtr, Xte = _design(train, test)
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = pd.DataFrame(scaler.transform(Xtr), index=Xtr.index, columns=Xtr.columns)
    Xte_s = pd.DataFrame(scaler.transform(Xte), index=Xte.index, columns=Xte.columns)
    Xtr_c, Xte_c = _add_const(Xtr_s, Xte_s)
    model = sm.QuantReg(train["rho"], Xtr_c).fit(q=tau, max_iter=5000)
    return np.clip(np.asarray(model.predict(Xte_c)), 0.0, 1.0)


def fit_predict_l1_quantile_regression(train: pd.DataFrame, test: pd.DataFrame, tau: float,
                                       alpha: float = 0.01, **hp) -> np.ndarray:
    """کوانتایل + انتخاب فیچر، با حل‌کننده‌ی LP (بند 7.10.1 عضو ۸)."""
    Xtr, Xte = _design(train, test)
    scaler = StandardScaler().fit(Xtr)
    model = QuantileRegressor(quantile=tau, alpha=alpha, solver="highs")
    model.fit(scaler.transform(Xtr), train["rho"])
    return np.clip(model.predict(scaler.transform(Xte)), 0.0, 1.0)


def fit_predict_composite_quantile_regression(train: pd.DataFrame, test: pd.DataFrame, tau: float,
                                              taus: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20),
                                              **hp) -> np.ndarray:
    """برآورد هم‌زمان چند τ با **شیب مشترک**، عرض‌ازمبدأ جدا — جلوگیری از تقاطع کوانتایل‌ها
    (بند 7.10.1 عضو ۹). LP: $\\min \\sum_k\\sum_i \\tau_k u_{ik} + (1-\\tau_k) v_{ik}$ با قید
    $y_i - b_{0k} - x_i\\beta = u_{ik} - v_{ik}$، $u,v\\ge0$، $\\beta$ مشترک بین همه‌ی $k$.
    """
    if tau not in taus:
        taus = tuple(sorted(set(taus) | {tau}))

    Xtr, Xte = _design(train, test)
    scaler = StandardScaler().fit(Xtr)
    Ztr, Zte = scaler.transform(Xtr), scaler.transform(Xte)
    y = train["rho"].to_numpy()
    n, p = Ztr.shape
    K = len(taus)

    Ztr_sp = csr_matrix(Ztr)
    Xp = vstack([Ztr_sp] * K)
    B0 = block_diag([np.ones((n, 1))] * K)
    # قید: x·β + b0ₖ + u − v = y  ⇒  y − pred = u − v، پس u باقیمانده‌ی **مثبت**
    # (کم‌برآورد) است و هزینه‌اش τ، و v باقیمانده‌ی منفی با هزینه‌ی ۱−τ. ⚠️ اگر علامت این دو
    # جابه‌جا شود، مسئله بی‌سروصدا کوانتایل ۱−τ را حل می‌کند نه τ.
    A_eq = hstack([Xp, -Xp, B0, identity(n * K), -identity(n * K)]).tocsr()
    b_eq = np.tile(y, K)

    n_beta, n_b0, n_slack = 2 * p, K, n * K
    c = np.zeros(n_beta + n_b0 + 2 * n_slack)
    for k, tk in enumerate(taus):
        c[n_beta + n_b0 + k * n: n_beta + n_b0 + (k + 1) * n] = tk
        c[n_beta + n_b0 + n_slack + k * n: n_beta + n_b0 + n_slack + (k + 1) * n] = 1 - tk

    bounds = [(0, None)] * n_beta + [(None, None)] * n_b0 + [(0, None)] * (2 * n_slack)
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"Composite QR LP همگرا نشد: {res.message}")

    beta = res.x[:p] - res.x[p:2 * p]
    b0 = dict(zip(taus, res.x[n_beta:n_beta + n_b0]))
    return np.clip(b0[tau] + Zte @ beta, 0.0, 1.0)


# ---------------------------------------------------------------------------
# ۱۰: Expectile Regression — جایگزین صاف کوانتایل (IRLS، Newey & Powell 1987)
# ---------------------------------------------------------------------------

def fit_predict_expectile_regression(train: pd.DataFrame, test: pd.DataFrame, tau: float,
                                     alpha: float = 1e-6, max_iter: int = 30,
                                     tol: float = 1e-6, **hp) -> np.ndarray:
    Xtr, Xte = _design(train, test)
    scaler = StandardScaler().fit(Xtr)
    Ztr, Zte = scaler.transform(Xtr), scaler.transform(Xte)
    y = train["rho"].to_numpy()

    beta, b0 = np.zeros(Ztr.shape[1]), float(y.mean())
    for _ in range(max_iter):
        resid = y - (b0 + Ztr @ beta)
        w = np.where(resid >= 0, tau, 1 - tau)
        model = Ridge(alpha=alpha).fit(Ztr, y, sample_weight=w)
        new_b0, new_beta = float(model.intercept_), model.coef_
        moved = max(abs(new_b0 - b0), np.max(np.abs(new_beta - beta)))
        b0, beta = new_b0, new_beta
        if moved < tol:
            break
    return np.clip(b0 + Zte @ beta, 0.0, 1.0)


# ---------------------------------------------------------------------------
# ۱۱–۱۴: خانواده‌ی GLM — مسیر Q2 (کوانتایل توزیع فرضی برازش‌شده)
# ---------------------------------------------------------------------------

#: پیوندهای مجاز GLM Gamma (بند 7.10.2). ``log`` پیش‌فرض است.
_GAMMA_LINKS = {"log": sm.families.links.Log, "inverse": sm.families.links.InversePower,
                "identity": sm.families.links.Identity}


def fit_predict_glm_gamma(train: pd.DataFrame, test: pd.DataFrame, tau: float,
                          link: str = "log", **hp) -> np.ndarray:
    """🔄 ارتقایافته — F04: Gamma بهترین برازش (KS=۰.۰۴۲۹) در برابر Beta (۰.۰۶۱۵)."""
    Xtr, Xte = _design(train, test)
    Xtr_c, Xte_c = _add_const(Xtr, Xte)
    y = np.clip(train["rho"].to_numpy(), 1e-4, None)  # Gamma>0؛ صفرهای F03 برش می‌خورند
    model = sm.GLM(y, Xtr_c, family=sm.families.Gamma(link=_GAMMA_LINKS[link]())).fit()
    mu_te = np.asarray(model.predict(Xte_c))
    return common.gamma_quantile(mu_te, float(model.scale), tau)


def fit_predict_glm_tweedie(train: pd.DataFrame, test: pd.DataFrame, tau: float,
                            power: float = 1.5, **hp) -> np.ndarray:
    """خانواده‌ی متغیر-واریانس — Tweedie با $p\\in(1,2)$ تورم صفر را هم می‌گیرد."""
    Xtr, Xte = _design(train, test)
    Xtr_c, Xte_c = _add_const(Xtr, Xte)
    y = np.clip(train["rho"].to_numpy(), 0.0, None)
    fam = sm.families.Tweedie(link=sm.families.links.Log(), var_power=power, eql=True)
    model = sm.GLM(y, Xtr_c, family=fam).fit()
    mu_te = np.asarray(model.predict(Xte_c))
    return common.tweedie_quantile_mc(mu_te, float(model.scale), power, tau)


#: پیوندهای مجاز میانگین در Beta regression (بند 7.10.2)
_BETA_LINKS = {"logit": sm.families.links.Logit, "probit": sm.families.links.Probit,
               "cloglog": sm.families.links.CLogLog}


def fit_predict_beta_regression(train: pd.DataFrame, test: pd.DataFrame, tau: float,
                                link_mu: str = "logit", **hp) -> np.ndarray:
    """🔄 تنزل‌یافته از انتخاب اول (F04) — نیاز به فشرده‌سازی $\\rho$ به $(0,1)$ باز."""
    from statsmodels.othermod.betareg import BetaModel

    Xtr, Xte = _design(train, test)
    Xtr_c, Xte_c = _add_const(Xtr, Xte)
    y = np.clip(train["rho"].to_numpy(), 1e-4, 1 - 1e-4)
    model = BetaModel(endog=y, exog=Xtr_c, link=_BETA_LINKS[link_mu]()).fit(disp=False)
    mu_te = np.asarray(model.predict(Xte_c))
    phi = float(np.exp(model.params.iloc[-1]))  # پارامتر precision (لینک log، آخرین پارامتر مدل)
    return common.beta_quantile(mu_te, phi, tau)


def fit_predict_glm_binomial(train: pd.DataFrame, test: pd.DataFrame, tau: float, **hp) -> np.ndarray:
    """❌ **رد‌شده** (F07: بیش‌پراکندگی صعودی ۳.۷×→۱۵.۶×) — یک‌بار اجرا **فقط** برای
    نشان‌دادن مقدار کم‌برآورد عدم‌قطعیت در گزارش (بند 7.10.1 عضو ۱۴)."""
    Xtr, Xte = _design(train, test)
    Xtr_c, Xte_c = _add_const(Xtr, Xte)
    y = train["rho"].to_numpy()
    model = sm.GLM(y, Xtr_c, family=sm.families.Binomial(), var_weights=train["Res"].to_numpy()).fit()
    mu_te = np.asarray(model.predict(Xte_c))
    return common.binomial_normal_quantile(mu_te, test["Res"].to_numpy(), tau)


# ---------------------------------------------------------------------------
# ۱۵: مدل دوبخشی (Hurdle) — مسیر Q2، پاسخ به تورم صفر (F03)
# ---------------------------------------------------------------------------

def fit_predict_hurdle(train: pd.DataFrame, test: pd.DataFrame, tau: float,
                       part1_link: str = "logit", part2_dist: str = "lognormal",
                       alpha: float = 1.0, **hp) -> np.ndarray:
    """F03: ۴.۹۰٪ رکوردها دقیقاً صفر، ولی تک‌مُدی (Sarle=۰.۴۶۱) ⇒ دوبخشی کافی است.

    دو مدل مستقل (بند 7.10.2): بخش ۱ احتمال $\\rho=0$، بخش ۲ توزیع $\\rho \\mid \\rho>0$.
    کوانتایل ترکیبی: اگر $\\tau \\le P(\\rho=0)$ آنگاه صفر، وگرنه کوانتایل
    $(\\tau - P_0)/(1 - P_0)$ اُمِ بخش دوم.
    """
    Xtr, Xte = _design(train, test)
    scaler = StandardScaler().fit(Xtr)  # ⚠️ بدون این، lbfgs روی فیچرهای هم‌مقیاس‌نشده به‌کندی همگرا می‌شود
    Ztr = pd.DataFrame(scaler.transform(Xtr), index=Xtr.index, columns=Xtr.columns)
    Zte = pd.DataFrame(scaler.transform(Xte), index=Xte.index, columns=Xte.columns)
    Ztr_c, Zte_c = _add_const(Ztr, Zte)

    # ---- بخش ۱: احتمال صفر بودن
    is_zero = (train["rho"] <= 0).astype(float).to_numpy()
    link = sm.families.links.Logit() if part1_link == "logit" else sm.families.links.Probit()
    zero_model = sm.GLM(is_zero, Ztr_c, family=sm.families.Binomial(link=link)).fit()
    p_zero_te = np.clip(np.asarray(zero_model.predict(Zte_c)), 1e-9, 1 - 1e-9)

    # ---- بخش ۲: توزیع مقادیر مثبت
    pos = (train["rho"] > 0).to_numpy()
    y_pos = train["rho"].to_numpy()[pos]
    Zpos = Ztr_c.loc[pos]

    # τ تعدیل‌شده‌ی هر ردیف: سهمی از توزیع بخش دوم که به کوانتایل کل می‌رسد
    tau_adj = np.clip((tau - p_zero_te) / (1.0 - p_zero_te), 0.0, 1.0)

    if part2_dist == "lognormal":
        log_y = np.log(y_pos)
        m = Ridge(alpha=alpha).fit(Zpos.to_numpy(), log_y)
        sigma = float((log_y - m.predict(Zpos.to_numpy())).std(ddof=1))
        q = np.exp(m.predict(Zte_c.to_numpy()) + sigma * stats.norm.ppf(np.clip(tau_adj, 1e-9, 1 - 1e-9)))
    elif part2_dist == "gamma":
        m = sm.GLM(y_pos, Zpos, family=sm.families.Gamma(link=sm.families.links.Log())).fit()
        mu = np.asarray(m.predict(Zte_c))
        phi = float(m.scale)
        q = stats.gamma.ppf(tau_adj, a=1.0 / phi, scale=np.clip(mu, 1e-9, None) * phi)
    elif part2_dist == "beta":
        from statsmodels.othermod.betareg import BetaModel

        m = BetaModel(endog=np.clip(y_pos, 1e-4, 1 - 1e-4), exog=Zpos).fit(disp=False)
        mu = np.clip(np.asarray(m.predict(Zte_c)), 1e-6, 1 - 1e-6)
        phi = float(np.exp(m.params.iloc[-1]))
        q = stats.beta.ppf(tau_adj, mu * phi, (1 - mu) * phi)
    else:
        raise ValueError(f"توزیع بخش دوم ناشناخته: {part2_dist!r}")

    return np.clip(np.where(tau <= p_zero_te, 0.0, q), 0.0, 1.0)


# ---------------------------------------------------------------------------
# ۱۶: GAM — غیرخطی صاف + تفسیرپذیر (بند 7.10.1 عضو ۱۶)
# ---------------------------------------------------------------------------

_GAM_SPLINE_FEATURES = ("log_res", "week_of_semester", "days_to_next_holiday", "day_of_month")


def fit_predict_gam(train: pd.DataFrame, test: pd.DataFrame, tau: float, n_splines: int = 10,
                    lam: float = 0.6, spline_order: int = 3, **hp) -> np.ndarray:
    """اسپلاین روی $\\log Res$، `week_of_semester`، `days_to_next_holiday` (بند 7.10.1).

    ⚠️ **مهار برون‌یابی (تصمیم پیش‌پردازش، گام ۶ کارت مدل).** اسپلاین بیرون از دامنه‌ی
    برازش قید ندارد و می‌تواند منفجر شود. این خطر در این پروتکل CV **واقعی و بزرگ** است:
    در fold صفر، `week_of_semester` آموزش [۹،۱۶] و آزمون [۱،۲] است — دو بازه‌ی کاملاً مجزا،
    چون fold آزمون ترم بعدی را می‌گیرد. بدون مهار، GAM میانگین ۰.۲۰ می‌داد در برابر ۰.۱۱
    واقعی (pinball ۰.۱۰۵ — ۱۲ برابر بدتر از خط پایه). فیچرهای آزمون به دامنه‌ی آموزش برش
    می‌خورند: «هفته‌ی ۱ مثل نزدیک‌ترین هفته‌ای که دیده‌ایم رفتار می‌کند» — محافظه‌کارانه و
    مستند، به‌جای برون‌یابی بی‌قید.
    """
    from pygam import LinearGAM, s

    cols = [c for c in _GAM_SPLINE_FEATURES if c in train.columns]
    medians = train[cols].median()
    Xtr = train[cols].fillna(medians).to_numpy()
    lo, hi = Xtr.min(axis=0), Xtr.max(axis=0)
    Xte = np.clip(test[cols].fillna(medians).to_numpy(), lo, hi)

    terms = s(0, n_splines=n_splines, lam=lam, spline_order=spline_order)
    for i in range(1, len(cols)):
        terms += s(i, n_splines=n_splines, lam=lam, spline_order=spline_order)
    model = LinearGAM(terms).fit(Xtr, train["rho"].to_numpy())

    mu_tr, mu_te = model.predict(Xtr), model.predict(Xte)
    return common.residual_quantile_by_res_quartile(train, test, mu_tr, mu_te, tau)


# ---------------------------------------------------------------------------
# رجیستری
# ---------------------------------------------------------------------------

#: model_id → تابع fit_predict. عضو #۱۷ (L5) اینجا نیست — بند بالای فایل.
MODELS = {
    "ols": fit_predict_ols,
    "ridge": fit_predict_ridge,
    "lasso": fit_predict_lasso,
    "elasticnet": fit_predict_elasticnet,
    "adaptive_lasso": fit_predict_adaptive_lasso,
    "group_lasso": fit_predict_group_lasso,
    "quantile_regression": fit_predict_quantile_regression,
    "l1_quantile_regression": fit_predict_l1_quantile_regression,
    "composite_quantile_regression": fit_predict_composite_quantile_regression,
    "expectile_regression": fit_predict_expectile_regression,
    "glm_gamma": fit_predict_glm_gamma,
    "glm_tweedie": fit_predict_glm_tweedie,
    "beta_regression": fit_predict_beta_regression,
    "glm_binomial": fit_predict_glm_binomial,
    "hurdle": fit_predict_hurdle,
    "gam": fit_predict_gam,
}

_QUANTILE_ROUTES = {
    "ols": "Q3", "ridge": "Q3", "lasso": "Q3", "elasticnet": "Q3", "adaptive_lasso": "Q3",
    "group_lasso": "Q3", "quantile_regression": "Q1", "l1_quantile_regression": "Q1",
    "composite_quantile_regression": "Q1", "expectile_regression": "Q1", "glm_gamma": "Q2",
    "glm_tweedie": "Q2", "beta_regression": "Q2", "glm_binomial": "Q2", "hurdle": "Q2", "gam": "Q3",
}

#: کلاس/کتابخانه‌ی واقعی هر مدل — به tag اختصاصی ``model_type`` هر MLflow run می‌رود
#: (بند 7.7.2، تفصیل در `doc/phase7-execution-standard.md`)، جدا از model_id که فقط
#: شناسه‌ی داخلی پروژه است.
_ALGORITHMS = {
    "ols": "sklearn.LinearRegression",
    "ridge": "sklearn.Ridge",
    "lasso": "sklearn.Lasso",
    "elasticnet": "sklearn.ElasticNet",
    "adaptive_lasso": "sklearn.Lasso (وزن‌دهی تطبیقی سفارشی)",
    "group_lasso": "group_lasso.GroupLasso",
    "quantile_regression": "statsmodels.QuantReg",
    "l1_quantile_regression": "sklearn.QuantileRegressor",
    "composite_quantile_regression": "scipy.optimize.linprog (LP سفارشی)",
    "expectile_regression": "sklearn.Ridge (IRLS سفارشی)",
    "glm_gamma": "statsmodels.GLM(Gamma)",
    "glm_tweedie": "statsmodels.GLM(Tweedie)",
    "beta_regression": "statsmodels.BetaModel",
    "glm_binomial": "statsmodels.GLM(Binomial)",
    "hurdle": "statsmodels.GLM(Binomial)+Ridge (دوبخشی سفارشی)",
    "gam": "pygam.LinearGAM",
}

for _model_id, _route in _QUANTILE_ROUTES.items():
    register(ModelSpec(model_id=_model_id, family=FAMILY, levels=(LEVEL,), quantile_route=_route,
                       algorithm=_ALGORITHMS[_model_id]))


# ---------------------------------------------------------------------------
# فضای هایپرپارامتر — بند 7.10.2. هر تابع یک optuna.Trial می‌گیرد و دیکشنری
# هایپرپارامتر متناظر با امضای fit_predict همان مدل برمی‌گرداند.
# ---------------------------------------------------------------------------

#: عضو #۱۴ (GLM Binomial) طبق بند 7.10.1 «رد‌شده — ولی یک‌بار اجرا و گزارش می‌شود» است؛
#: هیچ هایپرپارامتر آزاد ندارد و **وارد قیف S1/S2 نمی‌شود** — فقط اجرای پیش‌فرض S0 که
#: قبلاً برای کمّی‌سازی کم‌برآورد عدم‌قطعیت گزارش شد کافی است.
TUNING_EXCLUDED = frozenset({"glm_binomial"})


@register_space("ols", version=1, n_hyperparams=0)
def _space_ols(trial: optuna.Trial) -> dict:
    return {}  # بدون هایپرپارامتر — فقط به‌عنوان مرجع مطلق در جدول می‌ماند (بند 7.10.1 عضو ۱)


@register_space("ridge", version=1, n_hyperparams=1)
def _space_ridge(trial: optuna.Trial) -> dict:
    return {"alpha": trial.suggest_float("alpha", 1e-4, 1e2, log=True)}


@register_space("lasso", version=1, n_hyperparams=1)
def _space_lasso(trial: optuna.Trial) -> dict:
    return {"alpha": trial.suggest_float("alpha", 1e-4, 1e2, log=True)}


@register_space("elasticnet", version=1, n_hyperparams=2)
def _space_elasticnet(trial: optuna.Trial) -> dict:
    return {
        "alpha": trial.suggest_float("alpha", 1e-4, 1e2, log=True),
        "l1_ratio": trial.suggest_float("l1_ratio", 0.05, 0.95),
    }


@register_space("adaptive_lasso", version=1, n_hyperparams=2)
def _space_adaptive_lasso(trial: optuna.Trial) -> dict:
    """سند ردیف مجزایی برای Adaptive Lasso ندارد؛ فضای Ridge/Lasso/EN + توان وزن تطبیقی
    ($\\gamma$، بند 7.10.1 عضو ۵) تعمیم داده شده."""
    return {
        "alpha": trial.suggest_float("alpha", 1e-4, 1e2, log=True),
        "gamma": trial.suggest_float("gamma", 0.5, 2.0),
    }


@register_space("group_lasso", version=1, n_hyperparams=2)
def _space_group_lasso(trial: optuna.Trial) -> dict:
    return {
        "alpha": trial.suggest_float("alpha", 1e-4, 1e1, log=True),
        "group_reg": trial.suggest_float("group_reg", 1e-4, 1e1, log=True),
    }


@register_space("quantile_regression", version=1, n_hyperparams=0)
def _space_quantile_regression(trial: optuna.Trial) -> dict:
    """`q`=τ در سطح run ثابت می‌شود، نه هایپرپارامتر Optuna. statsmodels.QuantReg بدون
    منظم‌سازی هیچ هایپرپارامتر آزاد دیگری ندارد (سطر «QuantReg» جدول 7.10.2 عملاً برای
    نسخه‌ی L1-منظم‌شده‌ی زیر است)."""
    return {}


@register_space("l1_quantile_regression", version=1, n_hyperparams=1)
def _space_l1_quantile_regression(trial: optuna.Trial) -> dict:
    return {"alpha": trial.suggest_float("alpha", 1e-5, 1e0, log=True)}


@register_space("composite_quantile_regression", version=1, n_hyperparams=1)
def _space_composite_quantile_regression(trial: optuna.Trial) -> dict:
    """محور تنظیم این مدل مجموعه‌ی τهای هم‌زمان‌برازش‌شده است، نه یک عدد پیوسته
    (بند 7.10.1 عضو ۹) — یک هایپرپارامتر دسته‌ای طبق جدول 7.10.2."""
    grids = {"narrow": (0.05, 0.10, 0.15, 0.20), "wide": (0.02, 0.05, 0.10, 0.15, 0.20, 0.25)}
    return {"taus": grids[trial.suggest_categorical("taus_grid", list(grids))]}


@register_space("expectile_regression", version=1, n_hyperparams=1)
def _space_expectile_regression(trial: optuna.Trial) -> dict:
    """سند ردیف مجزا ندارد؛ تنها هایپرپارامتر آزاد پیاده‌سازی IRLS، ضریب منظم‌سازی Ridge
    داخلی است (پایداری عددی، نه انتخاب مدل)."""
    return {"alpha": trial.suggest_float("alpha", 1e-6, 1e1, log=True)}


@register_space("glm_gamma", version=1, n_hyperparams=1)
def _space_glm_gamma(trial: optuna.Trial) -> dict:
    return {"link": trial.suggest_categorical("link", ["log", "inverse", "identity"])}


@register_space("glm_tweedie", version=1, n_hyperparams=1)
def _space_glm_tweedie(trial: optuna.Trial) -> dict:
    return {"power": trial.suggest_float("power", 1.01, 1.99)}


@register_space("beta_regression", version=1, n_hyperparams=1)
def _space_beta_regression(trial: optuna.Trial) -> dict:
    return {"link_mu": trial.suggest_categorical("link_mu", ["logit", "probit", "cloglog"])}


@register_space("hurdle", version=1, n_hyperparams=2)
def _space_hurdle(trial: optuna.Trial) -> dict:
    return {
        "part1_link": trial.suggest_categorical("part1_link", ["logit", "probit"]),
        "part2_dist": trial.suggest_categorical("part2_dist", ["gamma", "beta", "lognormal"]),
    }


@register_space("gam", version=1, n_hyperparams=3)
def _space_gam(trial: optuna.Trial) -> dict:
    return {
        "n_splines": trial.suggest_int("n_splines", 5, 25),
        "lam": trial.suggest_float("lam", 1e-3, 1e3, log=True),
        "spline_order": trial.suggest_categorical("spline_order", [2, 3]),
    }


def main() -> None:
    """اجرای S0 (بند 7.3.1) برای هر ۱۶ عضو L1×F01 روی نخستین fold. اجرا: ``python -m
    src.models.families.f01_linear``. عضو #۱۷ (Logistic روی L5) اینجا نیست — بالای فایل."""
    from src.config import set_global_seed
    from src.cv import load_cv_folds, sha256_file
    from src.features.build import FEATURES_A_PATH
    from src.models.s0_runner import (
        RESULTS_MD,
        baseline_reference,
        print_summary,
        run_family_s0,
        save_results,
    )

    set_global_seed()
    df = pd.read_parquet(FEATURES_A_PATH).sort_values("date_gregorian").reset_index(drop=True)
    folds, cv_folds_hash = load_cv_folds()
    f = folds[0]
    tr_mask, te_mask = f.masks(df["date_gregorian"])
    train, test = df.loc[tr_mask], df.loc[te_mask]
    # ⚠️ features_A_v1.parquet هنوز در doc/data_manifest.md ثبت نشده (خلأ باقی‌مانده از فاز ۵)؛
    # هش زنده‌ی فایل محاسبه می‌شود تا run بدون آن هم ردیابی‌پذیر بماند.
    data_snapshot_hash = sha256_file(FEATURES_A_PATH)

    print(f"S0 — {FAMILY} ({LEVEL}) — {f} (train={len(train):,}, test={len(test):,})")
    print("⚠️ پارامتر پیش‌فرض، یک fold — نشانه‌ی اولیه است، نه رتبه‌بندی (بند 7.3.2)\n")
    results = run_family_s0(FAMILY, LEVEL, MODELS, train, test, feature_set=FEATURE_SET_S0,
                            data_snapshot_hash=data_snapshot_hash, cv_folds_hash=cv_folds_hash,
                            dataset_source=str(FEATURES_A_PATH))
    baseline = baseline_reference(train, test)
    ok = print_summary(results, baseline)
    save_results(results, baseline)
    print(f"\nذخیره شد در {RESULTS_MD}")
    print("MLflow: mlruns/ — با `make mlflow-ui` یا ابزار MLflow MCP قابل‌مشاهده است")
    if not ok:
        raise AssertionError("یک یا چند مدل L1×F01 در S0 شکست خوردند")


if __name__ == "__main__":
    main()
