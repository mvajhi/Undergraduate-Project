"""خوشه‌بندی چندنمایی رفتاری فرد — زیرساخت بندهای ۹-۷ تا ۹-۹.

## چرا «چندنمایی» و نه یک خوشه‌بندی روی همه‌ی فیچرها

یک خوشه‌بندی روی مجموع فیچرها، بُعدی را که واریانس بیشتری دارد غالب می‌کند.
`n_res` دامنه‌ی ۶۰ برابری بین دهک یکم و نهم دارد (بند ۹-۳-۲) و اگر با
`burstiness` که دامنه‌اش ۰.۶ واحد است در یک ماتریس بنشیند، هر فاصله‌ای عملاً
فاصله‌ی حجم مصرف است. شش نمای جدا این را نمی‌پوشاند؛ **آشکارش** می‌کند —
هم‌پوشانی یا استقلال نماها در بند ۹-۸-۳ خودش یافته است.

## سه تصمیم محاسباتی که عدد را عوض می‌کنند و باید اعلام شوند

۱. **Ward روی زیرنمونه.** پیوند وارد ماتریس فاصله‌ی کامل می‌خواهد: ۳۷٬۷۳۶ نقطه
   یعنی ~۵.۶ گیگابایت. پس روی زیرنمونه‌ی `WARD_SUBSAMPLE` برازش می‌شود و بقیه با
   نزدیک‌ترین مرکز برچسب می‌گیرند. این **تقریب** است و در خروجی علامت می‌خورد.
۲. **سیلوئت روی زیرنمونه.** سیلوئت هم $O(n^2)$ است؛ با `SILHOUETTE_SAMPLE` و
   بذر ثابت حساب می‌شود.
۳. **تبدیل رتبه‌ای پیش از هر روش فاصله‌محور.** بدون آن، خوشه‌ها دنباله‌ی توزیع
   را پیدا می‌کنند نه ساختار را.

## قاعده‌ی خ — جمعیت‌شناسی هرگز وارد ماتریس نمی‌شود

`build_view_matrix` هر ستونی با پیشوند `demo_` را **صریحاً حذف می‌کند** و اگر
نمایی چنین ستونی بخواهد `ValueError` می‌دهد. این تنها راه تضمین اینکه پاسخ
بند ۹-۸-۵ («آیا خوشه فقط دانشکده را کشف کرده؟») معنادار بماند.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WARD_SUBSAMPLE = 8000
SILHOUETTE_SAMPLE = 10000
RANDOM_STATE = 42

#: شش نمای بند ۹-۷-۲. هر نما = زیرمجموعه‌ای از بلوک‌های بند ۹-۳-۲.
VIEWS: dict[str, dict] = {
    "ن۱ شدت×هدررفت": {
        "cols": ["n_res", "n_active_days", "res_per_active_month", "density",
                 "rho_eb", "n_waste", "rho_slope_seq", "lunch_share"],
        "desc": "نمای سیاستی — چقدر مصرف می‌کند و چقدر هدر می‌دهد",
    },
    "ن۲ ریتم زمانی": {
        "cols": ["dow0_share", "dow1_share", "dow2_share", "dow3_share",
                 "dow4_share", "dow5_share", "dow_entropy", "gap_mean", "gap_cv"],
        "compositional": ["dow0_share", "dow1_share", "dow2_share", "dow3_share",
                          "dow4_share", "dow5_share"],
        "desc": "الگوی هفتگی و منظم‌بودن حضور",
    },
    "ن۳ سبد مکان و منو": {
        "cols": ["n_venues", "venue_hhi", "venue_top_share", "venue_switch_rate",
                 "dorm_venue_share", "n_dishes", "dish_hhi", "dish_entropy",
                 "mainstream_index"],
        "desc": "تنوع‌طلبی در برابر وفاداری — مکان و غذا",
    },
    "ن۴ پویایی و شوک": {
        "cols": ["burstiness", "lag1_autocorr", "spell_len_mean", "spell_n",
                 "waste_in_spells_share", "shock_gap", "shock_slope"],
        "desc": "ساختار توالی غیبت و واکنش به روز بد",
    },
    "ن۵ چرخه‌ی عمر": {
        "cols": ["tenure_days", "days_since_last", "n_gaps_gt14", "max_gap",
                 "volume_trend", "n_active_months"],
        "desc": "ورود، ماندن، و خروج",
    },
    "ن۶ جامع": {
        "cols": ["n_res", "res_per_active_month", "density", "rho_eb", "n_waste",
                 "lunch_share", "dow_entropy", "gap_cv", "burstiness",
                 "lag1_autocorr", "waste_in_spells_share", "venue_hhi",
                 "venue_switch_rate", "dorm_venue_share", "dish_entropy",
                 "mainstream_index", "shock_gap", "tenure_days", "volume_trend",
                 "vote_rate", "forgot_card_rate"],
        "desc": "بخش‌بندی سرفصل — ورودی گزارش ذی‌نفع",
    },
}


def _clr(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """تبدیل لگاریتم-نسبت مرکزی برای بردارهای ترکیبی (سهم‌هایی که جمعشان ۱ است).

    بدون این، شش ستون سهم روزهفته یک قید خطی دقیق دارند و ماتریس کوواریانس
    منفرد می‌شود؛ PCA آخرین مؤلفه‌اش را از نویز عددی می‌سازد.
    """
    Xp = np.clip(X, eps, None)
    g = np.exp(np.log(Xp).mean(axis=1, keepdims=True))
    return np.log(Xp / g)


def build_view_matrix(P: pd.DataFrame, view: str, *, pca_var: float = 0.90
                      ) -> tuple[np.ndarray, pd.Index, list[str]]:
    """ماتریس آماده‌ی خوشه‌بندی یک نما: تبدیل، استانداردسازی، و PCA.

    برمی‌گرداند: (ماتریس، اندیس `PersonId`، نام ستون‌های ورودی).
    ردیف‌هایی که در ستون‌های نما مقدار تهی دارند **حذف می‌شوند** و تعدادشان در
    `attrs` ثبت می‌شود — جایگزینی با میانه در خوشه‌بندی، خوشه‌ی مصنوعی می‌سازد.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import QuantileTransformer, StandardScaler

    spec = VIEWS[view]
    cols = [c for c in spec["cols"] if c in P.columns]
    bad = [c for c in cols if c.startswith("demo_")]
    if bad:
        raise ValueError(f"قاعده‌ی خ نقض شد — ستون جمعیتی در نمای {view}: {bad}")

    sub = P[["PersonId"] + cols].dropna()
    idx = sub.PersonId
    X = sub[cols].to_numpy(dtype="float64")

    comp = [c for c in spec.get("compositional", []) if c in cols]
    if comp:
        pos = [cols.index(c) for c in comp]
        X[:, pos] = _clr(X[:, pos])

    # تبدیل رتبه‌ای به نرمال: چولگی و دنباله را می‌خواباند بدون حذف ترتیب
    X = QuantileTransformer(output_distribution="normal", n_quantiles=1000,
                            random_state=RANDOM_STATE).fit_transform(X)
    X = StandardScaler().fit_transform(X)
    p = PCA(n_components=pca_var, random_state=RANDOM_STATE)
    Xp = p.fit_transform(X)
    return Xp, idx, cols


def k_scan(X: np.ndarray, k_range=range(2, 13), *, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """اسکن $k$ با چهار معیار. هیچ‌کدام تنها معیار نیستند — همه گزارش می‌شوند."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                                 silhouette_score)

    rng = np.random.default_rng(seed)
    samp = rng.choice(len(X), size=min(SILHOUETTE_SAMPLE, len(X)), replace=False)
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(X)
        lab = km.labels_
        rows.append({
            "k": k,
            "inertia": float(km.inertia_),
            "silhouette": float(silhouette_score(X[samp], lab[samp])),
            "davies_bouldin": float(davies_bouldin_score(X, lab)),
            "calinski": float(calinski_harabasz_score(X, lab)),
            "min_cluster_share": float(np.bincount(lab, minlength=k).min() / len(lab)),
        })
    return pd.DataFrame(rows)


def fit_algorithms(X: np.ndarray, k: int, *, seed: int = RANDOM_STATE) -> dict[str, np.ndarray]:
    """چهار الگوریتم روی همان ماتریس و همان $k$.

    HDBSCAN $k$ نمی‌گیرد — تعداد خوشه را خودش تعیین می‌کند و نقطه‌ی بی‌خوشه
    (برچسب ‎-۱) هم تولید می‌کند. همین تفاوت، ارزشش را می‌سازد: اگر ساختار
    چگالی‌محور نباشد، HDBSCAN بخش بزرگی را «نویز» اعلام می‌کند و این خودش
    پاسخ است.
    """
    from sklearn.cluster import HDBSCAN, AgglomerativeClustering, KMeans
    from sklearn.mixture import GaussianMixture

    out: dict[str, np.ndarray] = {}
    out["K-Means"] = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(X)
    out["مخلوط گاوسی"] = GaussianMixture(n_components=k, covariance_type="full",
                                         random_state=seed, n_init=3).fit_predict(X)
    out["Ward (زیرنمونه)"] = _ward_subsample(X, k, seed=seed)
    out["HDBSCAN"] = HDBSCAN(min_cluster_size=max(50, len(X) // 200),
                             min_samples=10).fit_predict(X)
    return out


def _ward_subsample(X: np.ndarray, k: int, *, seed: int = RANDOM_STATE) -> np.ndarray:
    """Ward روی زیرنمونه، سپس تخصیص بقیه با نزدیک‌ترین مرکز خوشه.

    ⚠️ تقریب اعلام‌شده (بند «سه تصمیم محاسباتی»). ماتریس فاصله‌ی کامل ۳۷ هزار
    نقطه ~۵.۶ گیگابایت است و روی ماشین ۱۶ گیگابایتی پروژه جا نمی‌شود.
    """
    from sklearn.cluster import AgglomerativeClustering

    rng = np.random.default_rng(seed)
    n = len(X)
    if n <= WARD_SUBSAMPLE:
        return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
    samp = rng.choice(n, size=WARD_SUBSAMPLE, replace=False)
    lab_s = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X[samp])
    cent = np.vstack([X[samp][lab_s == c].mean(axis=0) for c in range(k)])
    d = ((X[:, None, :] - cent[None, :, :]) ** 2).sum(axis=2) if n * k < 4e7 else None
    if d is None:
        lab = np.empty(n, dtype=int)
        for s in range(0, n, 20000):
            e = min(s + 20000, n)
            lab[s:e] = ((X[s:e, None, :] - cent[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
        return lab
    return d.argmin(axis=1)


def bootstrap_stability(X: np.ndarray, k: int, *, n_boot: int = 100,
                        seed: int = RANDOM_STATE) -> pd.DataFrame:
    """پایداری هر خوشه با بوت‌استرپ — ژاکارِ بیشینه‌ی بازیابی (روش clusterboot).

    برای هر تکرار: نمونه‌گیری با جایگذاری، برازش دوباره، و برای هر خوشه‌ی مرجع
    بیشینه‌ی ژاکارِ با خوشه‌های تازه. میانه‌ی این مقادیر، پایداری آن خوشه است.

    آستانه‌های قاعده‌ی ج دور ۹: <۰.۶ تفسیر نمی‌شود · ۰.۶–۰.۷۵ با احتیاط ·
    >۰.۷۵ پایدار.
    """
    from sklearn.cluster import KMeans

    rng = np.random.default_rng(seed)
    base = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(X)
    ref = [set(np.flatnonzero(base.labels_ == c)) for c in range(k)]
    jac = np.zeros((n_boot, k))
    n = len(X)
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        lab = KMeans(n_clusters=k, random_state=int(rng.integers(1e6)),
                     n_init=3).fit_predict(X[idx])
        new = [set(idx[lab == c]) for c in range(k)]
        for c in range(k):
            jac[b, c] = max((len(ref[c] & s) / len(ref[c] | s)) if (ref[c] | s) else 0.0
                            for s in new)
    return pd.DataFrame({
        "خوشه": range(k),
        "اندازه": [len(s) for s in ref],
        "سهم": [len(s) / n for s in ref],
        "ژاکارِ میانه": np.median(jac, axis=0),
        "ژاکارِ صدک ۵": np.quantile(jac, 0.05, axis=0),
        "وضعیت": ["پایدار" if m > 0.75 else ("با احتیاط" if m > 0.60 else "ناپایدار")
                  for m in np.median(jac, axis=0)],
    })
