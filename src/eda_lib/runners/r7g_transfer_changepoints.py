"""۱۴-۵-۷ پایداری فصلی بین سال‌ها · ۱۴-۵-۸ نقطه‌ی تغییر و رانش.

اجرا: ``python -m src.eda_lib.runners.r7g_transfer_changepoints``

⚠️ قاعده‌ی ۱۴-۱-ب: هر مقایسه با **بوت‌استرپ بلوکی دوبعدی** می‌آید. پس ارزیابی
انتقال‌پذیری در سطح **سلول** (L1) انجام می‌شود نه L3 — چون L3 بُعد سلف ندارد و
بلوک دوبعدی روی آن تعریف نمی‌شود.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import set_global_seed
from src.cv import block_bootstrap_2d
from src.eda_lib.runners._common_v4 import EVENTS, header, load_features, md_table, show
from src.eda_lib.seasonal_v4 import calendar_positions, fourier_terms

COLS = ["date_gregorian", "meal", "restaurant", "dish", "Res", "NoRecv", "academic_year",
        "day_of_week", "month_jalali", "day_of_month_jalali", "is_ramadan", "days_into_ramadan",
        "is_holiday_any", "is_bridge_day", "holiday_block_length", "is_exam_period",
        "is_in_session", "week_of_semester", "semester", "days_since_semester_start",
        "days_to_next_holiday", "days_since_last_holiday", "venue_type", "campus"]


def _wfit(X, y, w, alpha=1e-3):
    sw = np.sqrt(w)
    A, b = X * sw[:, None], y * sw
    P = np.eye(X.shape[1]) * alpha * float(np.trace(A.T @ A)) / X.shape[1]
    P[0, 0] = 0
    return np.linalg.solve(A.T @ A + P, A.T @ b)


def component_blocks(f: pd.DataFrame, pos: pd.DataFrame) -> dict[str, np.ndarray]:
    """هر مؤلفه‌ی فصلی، جدا — تا انتقال‌پذیری‌اش مستقل سنجیده شود."""
    hol_abs = np.minimum(f.days_to_next_holiday.fillna(99), f.days_since_last_holiday.fillna(99))
    return {
        "هفتگی": pd.get_dummies(f.day_of_week, drop_first=True).to_numpy(dtype=float),
        "سالانه (هارمونیک شمسی)": fourier_terms(pos.pos_jalali.to_numpy(), 4, "j").to_numpy(),
        "رمضان": np.column_stack([
            f.is_ramadan.astype(float),
            (f.is_ramadan.astype(float) * f.days_into_ramadan.fillna(0)),
            (f.is_ramadan.astype(float) * (f.meal == "dinner").astype(float))]),
        "تعطیلات": np.column_stack([
            f.is_holiday_any.astype(float), f.is_bridge_day.astype(float),
            f.holiday_block_length.fillna(0).astype(float),
            (hol_abs <= 1).astype(float), (hol_abs == 2).astype(float), (hol_abs == 3).astype(float)]),
        "نیمسال": np.column_stack([
            f.is_in_session.astype(float), f.is_exam_period.astype(float),
            pd.get_dummies(f.week_of_semester.fillna(0).astype(int), drop_first=True).to_numpy(dtype=float),
            np.clip(7 - f.days_since_semester_start.fillna(99), 0, 7)]),
    }


def _oos_mse(f, X, w, y, ytr, yte):
    m = (f.academic_year == ytr).to_numpy()
    t = (f.academic_year == yte).to_numpy()
    beta = _wfit(X[m], y[m], w[m])
    return t, (y[t] - X[t] @ beta) ** 2


def main() -> None:
    set_global_seed()
    f = load_features(columns=COLS)
    f["rho"] = f.NoRecv / f.Res
    pos = calendar_positions(f)
    f["academic_year"] = pos.academic_year_filled.to_numpy()
    y = f.rho.to_numpy()
    w = f.Res.to_numpy().astype(float)
    base = np.column_stack([np.ones(len(f)),
                            (f.meal == "dinner").astype(float).to_numpy(),
                            pd.get_dummies(f.restaurant, drop_first=True).to_numpy(dtype=float)])
    yrs = sorted(f.academic_year.unique())
    blocks = component_blocks(f, pos)

    header("۱۴-۵-۷ · ۱ · ⭐ انتقال‌پذیری هر مؤلفه‌ی فصلی — سطح سلول")
    print(f"  پایه: عرض از مبدأ + وعده + اثر ثابت سلف ({base.shape[1]} ستون)")
    print(f"  {len(f):,} سلول · آموزش روی سال t، ارزیابی روی t+1 و t+2 · وزن‌دار با Res\n")

    rows = []
    for name, B in blocks.items():
        X = np.hstack([base, B])
        for i, ytr in enumerate(yrs[:-1]):
            for yte in yrs[i + 1:]:
                t, e_full = _oos_mse(f, X, w, y, ytr, yte)
                _, e_base = _oos_mse(f, base, w, y, ytr, yte)
                # سهم درون‌نمونه‌ای روی خودِ سال آموزش
                m = (f.academic_year == ytr).to_numpy()
                bb = _wfit(X[m], y[m], w[m])
                b0 = _wfit(base[m], y[m], w[m])
                in_gain = (np.average((y[m] - base[m] @ b0) ** 2, weights=w[m])
                           - np.average((y[m] - X[m] @ bb) ** 2, weights=w[m]))
                oos_gain = (np.average(e_base, weights=w[t]) - np.average(e_full, weights=w[t]))
                rows.append({"مؤلفه": name, "ستون": B.shape[1], "آموزش": ytr, "آزمون": yte,
                             "فاصله": yrs.index(yte) - i,
                             "سود درون‌نمونه": in_gain, "سود برون‌نمونه": oos_gain,
                             "نسبت انتقال": oos_gain / in_gain if in_gain > 0 else np.nan})
    tr = pd.DataFrame(rows)
    show(tr, n=40)

    header("۲ · رتبه‌بندی انتقال‌پذیری (ورودی اجباری بند ۱۴-۱۰)")
    rank = (tr.groupby("مؤلفه")
              .agg(ستون=("ستون", "first"), سود_درون=("سود درون‌نمونه", "mean"),
                   سود_برون=("سود برون‌نمونه", "mean"), نسبت=("نسبت انتقال", "mean"),
                   کمینه_نسبت=("نسبت انتقال", "min"))
              .sort_values("نسبت", ascending=False))
    show(rank)
    print("\n--- Markdown ---\n" + md_table(rank.reset_index(), "{:.6f}"))

    header("۳ · CI بلوکی دوبعدی برای سود برون‌نمونه‌ای هر مؤلفه (بدترین جفت سال)")
    worst = yrs[0], yrs[-1]
    for name, B in blocks.items():
        X = np.hstack([base, B])
        t, e_full = _oos_mse(f, X, w, y, *worst)
        _, e_base = _oos_mse(f, base, w, y, *worst)
        d = pd.DataFrame({"date_gregorian": f.date_gregorian.to_numpy()[t],
                          "restaurant": f.restaurant.to_numpy()[t],
                          "diff": e_base - e_full, "w": w[t]})

        def stat(s):
            return float(np.average(s["diff"], weights=s["w"]))

        est, lo, hi = block_bootstrap_2d(d, stat, day_col="date_gregorian",
                                         unit_col="restaurant", n_boot=200, seed=42)
        sig = "✅" if lo > 0 else ("❌" if hi < 0 else "⚠️ شامل صفر")
        print(f"  {name:24s} سود={est:+.7f}  CI=[{lo:+.7f}, {hi:+.7f}]  {sig}")

    header("۱۴-۵-۸ · ۴ · نقطه‌ی تغییر در **رفتار** — با شاهد مثبت")
    import ruptures as rpt

    def _daily_adjusted(df):
        """سری روزانه‌ی $\\rho$ پس از حذف اثر هفتگی — تا نقطه‌ی تغییر «رژیم» باشد نه «ترکیب»."""
        d = (df.groupby("date_gregorian", observed=True)
               .apply(lambda g: g.NoRecv.sum() / g.Res.sum(), include_groups=False)
               .rename("rho").reset_index())
        dow = df.groupby("date_gregorian").day_of_week.first().reindex(d.date_gregorian).to_numpy()
        r = d.rho.to_numpy() - pd.Series(d.rho.to_numpy()).groupby(dow).transform("mean").to_numpy()
        return d, (r - r.mean()) / r.std()

    def _detect(sig, dates, pens=(4, 6, 8, 12), min_size=5):
        """⚠️ `min_size` باید از کوتاه‌ترین رژیمی که دنبالش هستیم کوچک‌تر باشد.

        نسخه‌ی نخست این بند با `min_size=15` اجرا شد و هیچ نقطه‌ای پیدا نکرد — از جمله
        پنجره‌ی جنگ که **۱۳ روز** است و ساختاراً از فیلتر رد نمی‌شد. «چیزی پیدا نشد»
        در آن پیکربندی یک گزاره‌ی تهی بود، نه یافته.
        """
        out = {}
        for pen in pens:
            bk = rpt.Pelt(model="rbf", min_size=min_size).fit(sig).predict(pen=pen)
            out[pen] = [str(dates.iloc[b - 1].date()) for b in bk[:-1]]
        return out

    print("  ⚠️ شاهد مثبت اجباری: اگر روش نتواند پنجره‌ی جنگ را — که می‌دانیم هست —")
    print("     پیدا کند، آنگاه «نقطه‌ی تغییری یافت نشد» هیچ معنایی ندارد.\n")
    anom = load_features(with_anomalies=True,
                         columns=["date_gregorian", "meal", "Res", "NoRecv", "day_of_week"])
    d_a, sig_a = _daily_adjusted(anom)
    print("  الف) شاهد مثبت — ماتریس **با** پنجره‌ی شوک:")
    for pen, pts in _detect(sig_a, d_a.date_gregorian).items():
        print(f"     pen={pen:2d}: {len(pts)} نقطه — {pts}")

    d_c, sig_c = _daily_adjusted(f)
    print("\n  ب) ماتریس تحلیلی (پنجره‌ی شوک حذف‌شده، ردیف ۸۰):")
    for pen, pts in _detect(sig_c, d_c.date_gregorian).items():
        print(f"     pen={pen:2d}: {len(pts)} نقطه — {pts}")

    header("۵ · نقطه‌ی تغییر در **ثبت** — سهم کدهای ReserveStatus")
    from src.data.load_v4 import load_clean_v4
    raw = load_clean_v4(columns=["DateReserve", "ReserveStatus", "Count", "mealid"])
    raw["p"] = raw.jyear.astype(str) + "-" + raw.jmonth.astype(int).astype(str).str.zfill(2)
    tab = raw.pivot_table(index="p", columns="ReserveStatus", values="Count",
                          aggfunc="sum", observed=True).fillna(0)
    share = (tab.T / tab.sum(axis=1)).T * 100
    union_cols = [c for c in share.columns if c in ("ارسال نشده", "دریافت نشده", "منقضی شده")]
    share["اجتماع"] = share[union_cols].sum(axis=1)
    show(share.round(2), n=40)
    print()
    for col in union_cols + ["اجتماع"]:
        v = share[col].to_numpy()
        if v.std() == 0:
            continue
        bk = rpt.Pelt(model="l2", min_size=4).fit((v - v.mean()) / v.std()).predict(pen=6)
        pts = [str(share.index[b - 1]) for b in bk[:-1]]
        print(f"  {col:12s}: دامنه {v.min():5.2f}–{v.max():5.2f}٪ · نسبت {v.max()/max(v.min(),0.01):8.1f}× "
              f"· نقطه‌ی تغییر: {pts}")
    print("\n  ⇒ مقایسه‌ی دامنه‌ی تک‌کدها با دامنه‌ی «اجتماع» همان آزمون F90 است.")


if __name__ == "__main__":
    main()
