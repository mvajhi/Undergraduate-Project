"""کمکی‌های مشترک بین خانواده‌های مدل فاز ۷ — بند 7.5 (پیش‌پردازش) و 7.23 (استخراج کوانتایل).

هر تابع ``fit_predict`` خانواده‌ها با امضای ``(train, test, tau, **hyperparams) -> np.ndarray``
نوشته می‌شود — دقیقاً همان قرارداد ``src/baselines.py`` (بند ۶.۵)، تا هارنس‌های S0/S1/S2
بتوانند خط پایه‌ها و مدل‌های فاز ۷ را یکسان صدا بزنند. ``train``/``test`` زیرمجموعه‌ی
سطرهای یک fold از ``features_A_v1.parquet``اند، نه ماتریس فیچر از پیش برش‌خورده.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import OneHotEncoder

#: چارک‌های Res برای مسیر Q3 — همان منطق ناهم‌واریانسی‌آگاه که در فاز ۶ (B7) اثبات شد
_N_RES_QUARTILES = 4
_MIN_BIN_SIZE_FOR_QUANTILE = 20


def design_matrix(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str]
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """میان‌گین‌گذاری عددی (میانه‌ی train) + یک‌هات دسته‌ای (fit روی train، دسته‌ی دیده‌نشده
    در test → همه صفر). خروجی float و بدون NaN؛ مقیاس‌بندی به عهده‌ی خودِ هر مدل است، چون
    بعضی (OLS/GLM) به آن حساس نیستند و بعضی (Lasso/Ridge/SVR) حتماً به آن نیاز دارند.
    """
    cat_cols = [c for c in feature_cols if train[c].dtype == object]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    medians = train[num_cols].median()
    tr_num = train[num_cols].fillna(medians).astype(float)
    te_num = test[num_cols].fillna(medians).astype(float)

    if not cat_cols:
        return tr_num, te_num

    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=float)
    enc.fit(train[cat_cols])
    names = enc.get_feature_names_out(cat_cols)
    tr_cat = pd.DataFrame(enc.transform(train[cat_cols]), index=train.index, columns=names)
    te_cat = pd.DataFrame(enc.transform(test[cat_cols]), index=test.index, columns=names)
    return pd.concat([tr_num, tr_cat], axis=1), pd.concat([te_num, te_cat], axis=1)


def category_groups(feature_cols: list[str], train: pd.DataFrame, encoded_columns: pd.Index) -> np.ndarray:
    """نگاشت هر ستون ماتریس طراحیِ یک‌هات‌شده به شناسه‌ی گروهش — برای Group Lasso (بند 7.10.1
    عضو ۶): همه‌ی ستون‌های یک‌هاتِ یک متغیر دسته‌ای باید یک گروه باشند تا باهم وارد/خارج شوند.
    """
    cat_cols = [c for c in feature_cols if train[c].dtype == object]
    groups = np.arange(len(encoded_columns))
    for i, col in enumerate(encoded_columns):
        for g, cat in enumerate(cat_cols):
            if col == cat or str(col).startswith(f"{cat}_"):
                groups[i] = len(encoded_columns) + g  # شناسه‌ی مشترک همه‌ی سطوح همان دسته
                break
    return groups


# ---------------------------------------------------------------------------
# بند 7.23 — مسیرهای استخراج کوانتایل
# ---------------------------------------------------------------------------

def residual_quantile_by_res_quartile(train: pd.DataFrame, test: pd.DataFrame,
                                      mu_hat_train: np.ndarray, mu_hat_test: np.ndarray,
                                      tau: float) -> np.ndarray:
    """مسیر Q3: کوانتایل تجربی باقیمانده‌ی out-of-sample، **به تفکیک چارک Res**.

    همان درسی که فاز ۶ داد (B2 در برابر B7، فاصله‌ی ۱۴٪ پینبال): یک آفست کوانتایل سراسری
    برای سلف پرحجم بیش‌ازحد محافظه‌کار و برای سلف کم‌حجم ناکافی است (F06 ناهم‌واریانسی).
    """
    resid = train["rho"].to_numpy() - mu_hat_train
    edges = train["Res"].quantile(np.linspace(0, 1, _N_RES_QUARTILES + 1)[1:-1]).to_numpy()
    tr_bin = np.digitize(train["Res"].to_numpy(), edges)
    te_bin = np.digitize(test["Res"].to_numpy(), edges)

    fallback = float(np.quantile(resid, tau))
    offsets = {}
    for b in range(_N_RES_QUARTILES):
        mask = tr_bin == b
        offsets[b] = float(np.quantile(resid[mask], tau)) if mask.sum() >= _MIN_BIN_SIZE_FOR_QUANTILE else fallback

    offset = np.array([offsets[b] for b in te_bin])
    return np.clip(mu_hat_test + offset, 0.0, 1.0)


def gamma_glm_regularized(y_train: np.ndarray, Xtr_const: pd.DataFrame, Xte_const: pd.DataFrame,
                          link, alpha: float = 0.01) -> tuple[np.ndarray, float]:
    """GLM Gamma با منظم‌سازی L2 خفیف به‌جای ``.fit()`` خام.

    ⚠️ **چرا لازم است.** یافته‌ی S1 خ۱ (بند 7.10): روی fold۲ با ۸۵ پارامتر (یک‌هات
    دسته‌ای‌های پرسطح) و برازش MLE بدون منظم‌سازی، یک ضریب به ۸۹ میلیارد واگرا شد
    (``converged=False``، شبه‌جدایی محتمل روی سطح کم‌داده‌ی یک دسته) و پیش‌بینی
    خارج‌نمونه به ``inf`` رسید — که پایین‌دست کوانتایل را به ۱٫۰ می‌چسباند (بند 7.23.2).
    ``alpha=0.01`` این واگرایی را کاملاً حذف می‌کند (بیشینه‌ی قدرمطلق ضریب از ~۸۹e9 به ~۲.۴).

    دیسپرسیون (φ) با باقیمانده‌ی پیرسون روی train محاسبه می‌شود چون ``fit_regularized``
    برخلاف ``fit()`` آن را برنمی‌گرداند.
    """
    import statsmodels.api as sm

    model = sm.GLM(y_train, Xtr_const, family=sm.families.Gamma(link=link))
    res = model.fit_regularized(alpha=alpha, L1_wt=0.0)
    mu_train = np.asarray(model.predict(res.params, Xtr_const))
    mu_test = np.asarray(model.predict(res.params, Xte_const))
    resid_pearson = (y_train - mu_train) / np.clip(mu_train, 1e-9, None)
    phi = float(np.sum(resid_pearson ** 2) / max(len(y_train) - Xtr_const.shape[1], 1))
    return mu_test, phi


def gamma_quantile(mu: np.ndarray, phi: float, tau: float) -> np.ndarray:
    """مسیر Q2 — GLM Gamma با پیوند log: میانگین=mu ⇒ shape=1/phi، scale=mu*phi."""
    shape = 1.0 / phi
    scale = np.clip(mu, 1e-9, None) * phi
    return np.clip(stats.gamma.ppf(tau, a=shape, scale=scale), 0.0, 1.0)


def beta_quantile(mu: np.ndarray, phi: float, tau: float) -> np.ndarray:
    """مسیر Q2 — Beta با پارامتر دقت phi: a=mu*phi، b=(1-mu)*phi."""
    mu = np.clip(mu, 1e-6, 1 - 1e-6)
    a = mu * phi
    b = (1 - mu) * phi
    return np.clip(stats.beta.ppf(tau, a, b), 0.0, 1.0)


def binomial_normal_quantile(mu: np.ndarray, res: np.ndarray, tau: float) -> np.ndarray:
    """مسیر Q2 — تقریب نرمال به نسبت دوجمله‌ای NoRecv/Res. **فقط برای عضو #14 (رد‌شده،
    بند 7.10.1)** که عمداً کم‌برآورد عدم‌قطعیت GLM دوجمله‌ای را نشان می‌دهد (پشتیبان F07)."""
    var = mu * (1 - mu) / np.clip(res, 1.0, None)
    z = stats.norm.ppf(tau)
    return np.clip(mu + z * np.sqrt(var), 0.0, 1.0)


def tweedie_quantile_mc(mu: np.ndarray, phi: float, power: float, tau: float,
                        seed: int = 42, n_sim: int = 500) -> np.ndarray:
    """مسیر Q2 — کوانتایل Tweedie ($1<p<2$) بدون فرم بسته، از نمایش پواسون-گامای فشرده:
    $N\\sim\\text{Poisson}(\\lambda)$، هر جهش $\\sim\\text{Gamma}(\\text{shape}, \\text{scale})$،
    مقدار Tweedie = مجموع جهش‌ها. برآورد مونت‌کارلو با n_sim تکرار به ازای هر ردیف.
    """
    rng = np.random.default_rng(seed)
    shape_const = (2 - power) / (power - 1)
    lam = mu ** (2 - power) / (phi * (2 - power))
    scale = phi * (power - 1) * mu ** (power - 1)

    out = np.empty(len(mu))
    for i in range(len(mu)):
        counts = rng.poisson(lam[i], size=n_sim)
        total = int(counts.sum())
        if total == 0:
            out[i] = 0.0
            continue
        draws = rng.gamma(shape_const, scale[i], size=total)
        sim_idx = np.repeat(np.arange(n_sim), counts)  # کدام جهش به کدام شبیه‌سازی تعلق دارد
        sums = np.bincount(sim_idx, weights=draws, minlength=n_sim)
        out[i] = np.quantile(sums, tau)
    return np.clip(out, 0.0, 1.0)
