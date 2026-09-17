"""برازش خانواده‌ی توزیعی هدف روی شبکه‌ی v4 (بند ۱۴-۴-۱).

## چرا این ماژول به‌جای یک هیستوگرام

$\\rho$ نسبت دو شمارش است ($NoRecv/Res$)، نه یک متغیر پیوسته. هیستوگرام $\\rho$
دو چیز کاملاً متفاوت را با هم قاطی می‌کند:

- **صفر نمونه‌گیری:** سلولی با $Res=۸$ و نرخ واقعی ۰.۰۸، با احتمال $0.92^8=۵۱٪$
  صفر مشاهده می‌شود. این صفر **خبری از پدیده نمی‌دهد**.
- **صفر ساختاری:** سلولی که نرخ واقعی‌اش صفر است (مثلاً خوابگاه متأهلین).

تفکیک این دو تنها با مدل‌کردن $(NoRecv, Res)$ ممکن است، نه با شکل $\\rho$. همین
تفکیک تعیین می‌کند خانواده‌ی ۹ (رگرسیون توزیعی) چه تابع پیوند و چه توزیعی بگیرد.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.special import betaln, expit, logit


def _binom_ll(k: np.ndarray, n: np.ndarray, p: float) -> float:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(np.sum(k * np.log(p) + (n - k) * np.log1p(-p)))


def _betabinom_ll(k: np.ndarray, n: np.ndarray, mu: float, phi: float) -> float:
    """لگاریتم درست‌نمایی بتا-دوجمله‌ای با پارامترسازی (میانگین، دقت).

    $a=\\mu\\phi$، $b=(1-\\mu)\\phi$. $\\phi$ بزرگ ⇒ نزدیک دوجمله‌ای؛ $\\phi$ کوچک ⇒
    بیش‌پراکندگی شدید. ضریب دوجمله‌ای ثابت است و در مقایسه‌ی AIC حذف می‌شود، ولی
    اینجا نگه داشته نمی‌شود چون در **هر سه** مدل یکسان است.
    """
    mu = np.clip(mu, 1e-9, 1 - 1e-9)
    phi = max(phi, 1e-9)
    a, b = mu * phi, (1 - mu) * phi
    return float(np.sum(betaln(k + a, n - k + b) - betaln(a, b)))


def _zibb_ll(k: np.ndarray, n: np.ndarray, mu: float, phi: float, pi0: float) -> float:
    """بتا-دوجمله‌ای با تورم صفر: با احتمال $\\pi_0$ سلول «ساختاراً صفر» است."""
    pi0 = np.clip(pi0, 1e-9, 1 - 1e-9)
    mu = np.clip(mu, 1e-9, 1 - 1e-9)
    phi = max(phi, 1e-9)
    a, b = mu * phi, (1 - mu) * phi
    ll_bb = betaln(k + a, n - k + b) - betaln(a, b)
    is0 = k == 0
    out = np.where(is0, np.logaddexp(np.log(pi0), np.log1p(-pi0) + ll_bb),
                   np.log1p(-pi0) + ll_bb)
    return float(np.sum(out))


def fit_families(k: np.ndarray, n: np.ndarray) -> pd.DataFrame:
    """سه خانواده را با MLE برازش و با AIC/BIC مقایسه می‌کند."""
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    m = n > 0
    k, n = k[m], n[m]
    N = len(k)
    p0 = float(k.sum() / n.sum())

    ll_bin = _binom_ll(k, n, p0)

    r = optimize.minimize(lambda t: -_betabinom_ll(k, n, expit(t[0]), np.exp(t[1])),
                          x0=[logit(p0), np.log(10.0)], method="Nelder-Mead",
                          options={"maxiter": 4000, "xatol": 1e-8, "fatol": 1e-8})
    mu_bb, phi_bb = float(expit(r.x[0])), float(np.exp(r.x[1]))
    ll_bb = -float(r.fun)

    r2 = optimize.minimize(lambda t: -_zibb_ll(k, n, expit(t[0]), np.exp(t[1]), expit(t[2])),
                           x0=[r.x[0], r.x[1], logit(0.05)], method="Nelder-Mead",
                           options={"maxiter": 8000, "xatol": 1e-8, "fatol": 1e-8})
    mu_zi, phi_zi, pi0 = float(expit(r2.x[0])), float(np.exp(r2.x[1])), float(expit(r2.x[2]))
    ll_zi = -float(r2.fun)

    rows = [
        {"خانواده": "دوجمله‌ای", "k_par": 1, "logL": ll_bin, "μ": p0, "φ": np.inf, "π₀": 0.0},
        {"خانواده": "بتا-دوجمله‌ای", "k_par": 2, "logL": ll_bb, "μ": mu_bb, "φ": phi_bb, "π₀": 0.0},
        {"خانواده": "ZI بتا-دوجمله‌ای", "k_par": 3, "logL": ll_zi, "μ": mu_zi, "φ": phi_zi, "π₀": pi0},
    ]
    out = pd.DataFrame(rows)
    out["AIC"] = 2 * out.k_par - 2 * out.logL
    out["BIC"] = out.k_par * np.log(N) - 2 * out.logL
    out["ΔAIC"] = out.AIC - out.AIC.min()
    out["N"] = N
    return out


def zero_excess(k: np.ndarray, n: np.ndarray, mu: float, phi: float) -> dict[str, float]:
    """صفرِ مشاهده‌شده در برابر صفرِ موردانتظارِ نمونه‌گیری.

    زیر بتا-دوجمله‌ای، $P(k=0) = B(a,\\,n+b)/B(a,b)$. تفاوت با نسبت مشاهده‌شده،
    همان «تورم صفر»ی است که مدل شمارشی نمی‌تواند توضیح دهد.
    """
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    a, b = mu * phi, (1 - mu) * phi
    p_zero = np.exp(betaln(a, n + b) - betaln(a, b))
    obs = float((k == 0).mean())
    exp = float(p_zero.mean())
    return {"مشاهده‌شده": obs, "موردانتظار (BB)": exp, "مازاد": obs - exp,
            "نسبت": obs / exp if exp > 0 else np.inf}


def zero_profile(df: pd.DataFrame, res_col: str = "Res", k_col: str = "NoRecv",
                 mu: float | None = None, phi: float | None = None,
                 bins=(0, 5, 10, 20, 50, 100, 250, 10 ** 9)) -> pd.DataFrame:
    """پروفایل صفر بر حسب اندازه‌ی سلول — جایی که صفر نمونه‌گیری از ساختاری جدا می‌شود."""
    d = df.copy()
    labels = [f"{lo}–{hi if hi < 10 ** 8 else '∞'}" for lo, hi in zip(bins[:-1], bins[1:])]
    d["سطل"] = pd.cut(d[res_col], bins=bins, labels=labels, right=False)
    a, b = mu * phi, (1 - mu) * phi
    d["p_zero_bb"] = np.exp(betaln(a, d[res_col] + b) - betaln(a, b))
    g = d.groupby("سطل", observed=True).agg(
        سلول=(k_col, "size"), پرس=(res_col, "sum"),
        صفر_مشاهده=(k_col, lambda s: float((s == 0).mean())),
        صفر_موردانتظار=("p_zero_bb", "mean"))
    g["مازاد"] = g.صفر_مشاهده - g.صفر_موردانتظار
    g["سهم سلول"] = g.سلول / g.سلول.sum()
    return g
