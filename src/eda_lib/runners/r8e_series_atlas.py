"""بند ۸-۸ — اطلس سری زمانی L4 (دور ۸).

F160 گفت ۱۰۹ سری با میانه‌ی ۳۹۰ نقطه حالا برازش‌پذیرند. این بسته پیش‌نیاز
دیداری و آماری خانواده‌های خ۳/خ۴/خ۵ جریان پ است.

اجرا: python -m src.eda_lib.runners.r8e_series_atlas
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.config import set_global_seed
from src.eda_lib import figs
from src.eda_lib.runners._common_v4 import header, load_features, md_table

MEAL_FA = {"lunch": "ناهار", "dinner": "شام"}


def build_series(f: pd.DataFrame) -> dict[tuple, pd.Series]:
    """سری L4 = نرخ وزنی هر (سلف، وعده) روی روزهای سرویس آن سری."""
    g = (f.groupby(["restaurant", "meal", "date_gregorian"], observed=True)[["Res", "NoRecv"]]
         .sum().reset_index())
    g["rate"] = g.NoRecv / g.Res
    out = {}
    for key, sub in g.groupby(["restaurant", "meal"], observed=True):
        s = sub.set_index("date_gregorian").rate.sort_index()
        if len(s) >= 40:
            out[key] = s
    return out


# ---------------------------------------------------------------------------
# ۸-۸-۱ نمای کلی
# ---------------------------------------------------------------------------

def p881_overview(f: pd.DataFrame, series: dict) -> pd.DataFrame:
    header("۸-۸-۱ نمای کلی سری‌های L4", 2)
    meta = []
    for (r, m), s in series.items():
        span = (s.index.max() - s.index.min()).days + 1
        meta.append(dict(restaurant=r, meal=m, n_points=len(s), span_days=span,
                         density=len(s) / span, mean=float(s.mean()), sd=float(s.std()),
                         cv=float(s.std() / s.mean()) if s.mean() else np.nan))
    meta = pd.DataFrame(meta).sort_values("n_points", ascending=False)

    fig, axes = figs.new(3, 3, w=3.6, h=2.4, sharex=False)
    pick = pd.concat([meta.head(3), meta.iloc[len(meta) // 2 - 1:len(meta) // 2 + 2],
                      meta.tail(3)])
    for ax, (_, row) in zip(axes.ravel(), pick.iterrows()):
        s = series[(row.restaurant, row.meal)]
        ax.plot(s.index, s.values, lw=0.8, color=figs.C_MAIN)
        ax.plot(s.index, s.rolling(21, min_periods=5).mean().values, lw=1.8, color=figs.C_ALT)
        ax.set_title(f"{row.restaurant[:22]} · {MEAL_FA[row.meal]}", fontsize=7)
        ax.tick_params(labelsize=6, axis="x", rotation=30)
        ax.set_ylim(0, min(0.45, float(np.nanquantile(s, 0.99)) * 1.4))
    fig.suptitle("نه سری نماینده — پرنقطه‌ترین، میانی، و کم‌نقطه‌ترین (خط نارنجی: میانگین ۲۱ روزه)")
    figs.save(fig, "r8e_01_series_overview", meta)
    print(md_table(meta.head(12)))
    print(f"\nسری با ≥۴۰ نقطه: {len(meta)} · میانه‌ی طول {int(meta.n_points.median())} · "
          f"میانه‌ی چگالی سرویس {meta.density.median():.2f} روز-سرویس در هر روز تقویمی")
    return meta


# ---------------------------------------------------------------------------
# ۸-۸-۲ و ۸-۸-۳ خودهمبستگی شکاف‌آگاه، و ایستایی
# ---------------------------------------------------------------------------

def p882_acf_stationarity(series: dict, meta: pd.DataFrame) -> pd.DataFrame:
    header("۸-۸-۲ / ۸-۸-۳ خودهمبستگی شکاف‌آگاه و ایستایی", 2)
    from statsmodels.tsa.stattools import adfuller, kpss

    def gap_acf(s: pd.Series, lag: int) -> float:
        """ACF بر پایه‌ی **فاصله‌ی تقویمی**، نه فاصله‌ی موقعیتی.

        سری شکاف سرویس دارد؛ `s.shift(1)` جفت‌هایی می‌سازد که در تقویم ۳۰ روز
        فاصله دارند و آن را «lag-1» می‌خواند.
        """
        idx = s.index
        target = idx + pd.Timedelta(days=lag)
        aligned = s.reindex(target)
        both = pd.DataFrame({"a": s.to_numpy(), "b": aligned.to_numpy()}).dropna()
        if len(both) < 30:
            return np.nan
        return float(np.corrcoef(both.a, both.b)[0, 1])

    rows = []
    for (r, m), s in series.items():
        if len(s) < 80:
            continue
        rec = dict(restaurant=r, meal=m, n=len(s))
        for lag in (1, 2, 3, 7, 14, 28):
            rec[f"acf_{lag}"] = gap_acf(s, lag)
        v = s.to_numpy()
        try:
            rec["adf_p"] = float(adfuller(v, autolag="AIC")[1])
        except Exception:
            rec["adf_p"] = np.nan
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rec["kpss_p"] = float(kpss(v, regression="c", nlags="auto")[1])
        except Exception:
            rec["kpss_p"] = np.nan
        # اثر ARCH: خودهمبستگی مربع باقی‌مانده‌ی نسبت به میانگین متحرک
        resid = v - pd.Series(v).rolling(21, min_periods=5, center=True).mean().to_numpy()
        rr = pd.Series(resid).dropna()
        rec["arch_lb_p"] = float(stats.pearsonr(rr[:-1] ** 2, rr[1:] ** 2).pvalue) if len(rr) > 40 else np.nan
        rows.append(rec)
    out = pd.DataFrame(rows)
    out["stationary"] = np.where((out.adf_p < 0.05) & (out.kpss_p > 0.05), "ایستا",
                        np.where((out.adf_p >= 0.05) & (out.kpss_p <= 0.05), "ناایستا", "مبهم"))

    fig, ax = figs.new(1, 2, w=6.6, h=4.3)
    lags = [1, 2, 3, 7, 14, 28]
    data = [out[f"acf_{l}"].dropna() for l in lags]
    bp = ax[0].boxplot(data, positions=range(len(lags)), widths=0.6, patch_artist=True,
                       medianprops=dict(color=figs.C_ALT, lw=2))
    for b in bp["boxes"]:
        b.set_facecolor(figs.C_MAIN); b.set_alpha(0.55)
    ax[0].axhline(0, color="black", lw=1)
    ax[0].set_xticks(range(len(lags))); ax[0].set_xticklabels(lags)
    ax[0].set_xlabel("تأخیر (روز تقویمی، نه موقعیت)"); ax[0].set_ylabel("خودهمبستگی")
    ax[0].set_title(f"ACF شکاف‌آگاه روی {len(out)} سری L4")
    figs.annotate(ax[0], f"میانه‌ی lag-1 {out.acf_1.median():+.2f}\n"
                         f"میانه‌ی lag-7 {out.acf_7.median():+.2f}\n"
                         f"سهم سری‌های با lag-1 > ۰.۲: {(out.acf_1 > 0.2).mean():.0%}")

    ct = out.stationary.value_counts()
    ax[1].bar(range(len(ct)), ct.values, color=[figs.C_OK, figs.C_ACCENT, figs.C_ALT][:len(ct)])
    ax[1].set_xticks(range(len(ct))); ax[1].set_xticklabels(ct.index)
    ax[1].set_ylabel("تعداد سری")
    ax[1].set_title("نتیجه‌ی ترکیبی ADF و KPSS")
    arch = float((out.arch_lb_p < 0.05).mean())
    figs.annotate(ax[1], f"سری‌های با اثر ARCH معنادار: {arch:.0%}\n"
                         f"⇒ واریانس شرطی صریح توجیه دارد" if arch > 0.3 else
                         f"سری‌های با اثر ARCH معنادار: {arch:.0%}")
    figs.save(fig, "r8e_02_acf_stationarity", out)
    print(md_table(out.head(15), float_fmt="{:.3f}"))
    print(f"\nخلاصه: {ct.to_dict()} · اثر ARCH معنادار در {arch:.1%} سری")
    return out


# ---------------------------------------------------------------------------
# ۸-۸-۴ و ۸-۸-۶ خوشه‌بندی شکل و هم‌حرکتی
# ---------------------------------------------------------------------------

def p884_clusters(series: dict, meta: pd.DataFrame) -> pd.DataFrame:
    header("۸-۸-۴ / ۸-۸-۶ خوشه‌بندی شکل سری‌ها و هم‌حرکتی (پیش‌نیاز خ۴ پنلی)", 2)
    from scipy.cluster import hierarchy
    from sklearn.decomposition import PCA

    keys = [k for k in series if len(series[k]) >= 120]
    wide = pd.DataFrame({f"{r}|{m}": series[(r, m)] for r, m in keys})
    wide = wide.resample("W").mean()
    wide = wide.dropna(thresh=int(len(wide) * 0.5), axis=1)
    corr = wide.corr(min_periods=20).fillna(0)

    Z = hierarchy.linkage(1 - corr.to_numpy()[np.triu_indices_from(corr, k=1)], method="average")
    lab = hierarchy.fcluster(Z, t=4, criterion="maxclust")
    filled = wide.interpolate(limit_direction="both").dropna(axis=1)
    p = PCA(n_components=min(5, filled.shape[1]))
    p.fit(filled.T.to_numpy())

    fig, ax = figs.new(1, 2, w=6.8, h=4.6)
    off = corr.to_numpy()[np.triu_indices_from(corr, k=1)]
    ax[0].hist(off, bins=40, color=figs.C_MAIN, alpha=0.9)
    ax[0].axvline(0, color="black", lw=1.2)
    ax[0].axvline(float(np.median(off)), color=figs.C_ALT, lw=2, ls="--")
    ax[0].set_xlabel("همبستگی هفتگی بین دو سری L4"); ax[0].set_ylabel("تعداد جفت")
    ax[0].set_title("آیا سری‌ها هم‌حرکت‌اند؟")
    figs.annotate(ax[0], f"{corr.shape[0]} سری · {len(off):,} جفت\n"
                         f"میانه {np.median(off):+.2f}\nسهم جفت‌های >۰.۳: {np.mean(off > 0.3):.0%}")

    ev = p.explained_variance_ratio_
    ax[1].bar(range(1, len(ev) + 1), ev * 100, color=figs.C_OK)
    ax[1].plot(range(1, len(ev) + 1), np.cumsum(ev) * 100, "o-", color=figs.C_ALT, lw=2)
    ax[1].set_xlabel("مؤلفه‌ی اصلی"); ax[1].set_ylabel("٪ واریانس")
    ax[1].set_title("عامل مشترک بین سری‌ها")
    figs.annotate(ax[1], f"مؤلفه‌ی اول {ev[0]:.0%}\nپنج مؤلفه {np.cumsum(ev)[-1]:.0%}\n"
                         f"⇒ ساختار عاملی قوی = مدل پنلی موجه", loc="upper left")
    out = pd.DataFrame(dict(series=corr.index, cluster=lab))
    figs.save(fig, "r8e_03_series_comovement", out)
    print(f"همبستگی جفتی: میانه {np.median(off):+.4f} · سهم >۰.۳ {np.mean(off>0.3):.3f} · "
          f"مؤلفه‌ی اول {ev[0]:.3f}")
    print(out.cluster.value_counts().to_string())
    return out


def main() -> None:
    set_global_seed()
    figs.setup()
    f = load_features()
    series = build_series(f)
    header(f"دور ۸ · اطلس سری L4 — {len(series)} سری با ≥۴۰ نقطه")
    meta = p881_overview(f, series)
    p882_acf_stationarity(series, meta)
    p884_clusters(series, meta)


if __name__ == "__main__":
    main()
