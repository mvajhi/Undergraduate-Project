"""بند ۴.۵ — داده‌ی پرت: شناسایی چندروشی + **تحلیل ریشه‌ای**، دور ۱.

قاعده‌ی صریح بند ۴.۵.۲ WBS: «هرگز پرت را کورکورانه حذف نکنید.» پس خروجی اصلی این
اسکریپت یک *عدد* نیست، یک **جدول ریشه‌یابی** است: هر پرت با `events.csv` و
`calendar_tehran.csv` تطبیق داده می‌شود و یک طبقه‌بندی (رویداد واقعی / خطای ثبت /
نویز کم‌نمونگی) و یک تصمیم می‌گیرد.

⚠️ درسی از بند ۴.۳: باقیمانده‌های بزرگ STL در روزهای **درون‌یابی‌شده** (بدون سرو)
مصنوع‌اند و نباید پرت شمرده شوند — اینجا صریحاً فیلتر می‌شوند.

اجرا: `python -m src.eda_lib.runners.s05_outliers`
"""

import numpy as np
import pandas as pd

from src.eda_lib.outlier_helpers import (
    build_outlier_flag_table,
    calendar_flags,
    iqr_outlier_mask,
    isolation_forest_mask,
    lof_mask,
    modified_zscore_outlier_mask,
    nearest_event,
    pairwise_jaccard,
    zscore_outlier_mask,
)
from src.eda_lib.runners._common import CALENDAR_PATH, EVENTS_PATH, header, kv, load_dataset, pct, setup


def run_detection(df: pd.DataFrame) -> pd.DataFrame:
    header("۴.۵.۱ شناسایی چندروشی")
    rho = df["rho"]
    masks = {}
    m_iqr, bounds = iqr_outlier_mask(rho)
    masks["IQR"] = m_iqr.values
    kv("IQR کران‌ها", f"[{bounds[0]:.4f}, {bounds[1]:.4f}]")
    m_z, z = zscore_outlier_mask(rho)
    masks["Z>3"] = m_z.values
    m_mz, mz = modified_zscore_outlier_mask(rho)
    masks["ModZ>3.5"] = m_mz.values

    feats = df[["rho", "Res", "gender_ratio", "DayOfWeek"]].copy()
    feats["log_res"] = np.log1p(feats["Res"])
    feats = feats.drop(columns="Res").fillna(feats.median(numeric_only=True))
    masks["IsolationForest"] = isolation_forest_mask(feats) if callable(isolation_forest_mask) else None
    masks["LOF"] = lof_mask(feats)

    flags = build_outlier_flag_table({k: v for k, v in masks.items() if v is not None}, df.index)
    methods = [c for c in flags.columns if c != "consensus"]
    print("\nتعداد پرت شناسایی‌شده به روش:")
    for m in methods:
        kv(f"  {m}", f"{int(flags[m].sum())} ({pct(flags[m].mean())})")
    print("\nتوزیع اجماع (چند روش هم‌زمان یک رکورد را پرت می‌دانند):")
    print(flags["consensus"].value_counts().sort_index().to_string())
    print("\nهم‌پوشانی Jaccard بین روش‌ها:")
    print(pairwise_jaccard(flags, methods).round(3).to_string())
    return flags


def run_root_cause(df: pd.DataFrame, flags: pd.DataFrame) -> None:
    header("۴.۵.۲ تحلیل ریشه‌ای — جدول پرت‌های با اجماع بالا")
    events = pd.read_csv(EVENTS_PATH, parse_dates=["date_start", "date_end"])
    cal = pd.read_csv(CALENDAR_PATH, parse_dates=["date_gregorian"])

    d = df.copy()
    d["consensus"] = flags["consensus"].values
    top = d[d["consensus"] >= 3].sort_values("rho", ascending=False)
    kv("رکوردهای با اجماع ≥۳ روش", len(top))

    print("\n--- بالاترین ۲۰ رکورد پرت (بر اساس ρ) ---")
    rows = []
    for _, r in top.head(20).iterrows():
        ev = nearest_event(r["date_gregorian"], events, window_days=3)
        cf = calendar_flags(r["date_gregorian"], cal)
        rows.append({
            "تاریخ": r["DateReserve"], "سلف": r["RestaurantName"], "وعده": r["Meal"],
            "شهر": r["city"], "Res": int(r["Res"]), "ρ": round(r["rho"], 3),
            "رویداد/تقویم": (ev or "—")[:44] + (" | " + cf[:34] if cf else ""),
        })
    print(pd.DataFrame(rows).to_string(index=False))

    header("پرت‌های مرزی: ρ=۱ (هیچ‌کس دریافت نکرد) و ρ=۰", 2)
    ones = df[df["rho"] == 1.0].sort_values("Res", ascending=False)
    print(f"ρ=۱ : {len(ones)} رکورد")
    for _, r in ones.iterrows():
        ev = nearest_event(r["date_gregorian"], events, window_days=3)
        print(f"  {r['DateReserve']} {r['RestaurantName']:<16s} {r['Meal']:<7s} "
              f"Res={int(r['Res']):>4d}  {(ev or '—')[:60]}")
    zeros = df[df["rho"] == 0.0]
    print(f"\nρ=۰ : {len(zeros)} رکورد — توزیع Res:")
    print(zeros["Res"].describe().round(1).to_string())
    print(f"سهم رکوردهای ρ=۰ که Res<20 دارند: {(zeros['Res'] < 20).mean():.1%} "
          f"(در برابر {(df['Res'] < 20).mean():.1%} در کل داده)")

    header("آیا پرت‌ها در سلف/شهر خاصی متمرکزند؟", 2)
    d["is_out"] = d["consensus"] >= 3
    by_r = d.groupby(["RestaurantName", "city"]).agg(
        n=("is_out", "size"), n_out=("is_out", "sum"), Res_median=("Res", "median"))
    by_r["نرخ پرت"] = (by_r["n_out"] / by_r["n"]).round(3)
    print(by_r.sort_values("نرخ پرت", ascending=False).head(12).to_string())
    print("\nنرخ پرت به تفکیک چارک Res:")
    q = pd.qcut(d["Res"], 4, labels=["Q1 کوچک", "Q2", "Q3", "Q4 بزرگ"])
    print(d.groupby(q, observed=True)["is_out"].agg(["size", "sum", "mean"]).round(3).to_string())

    header("تجمیع روزانه: کدام روزها در کل دانشگاه پرت‌اند؟", 2)
    daily = df.groupby(["date_gregorian", "DateReserve"]).apply(
        lambda x: pd.Series({"rho_w": x["NoRecv"].sum() / x["Res"].sum(),
                             "Res": x["Res"].sum(), "n_rows": len(x)}), include_groups=False).reset_index()
    mz = (daily["rho_w"] - daily["rho_w"].median()) / (
        1.4826 * np.median(np.abs(daily["rho_w"] - daily["rho_w"].median())))
    daily["modz"] = mz
    out_days = daily[daily["modz"].abs() > 3.5].sort_values("modz", ascending=False)
    print(f"روزهای پرت (|ModZ|>3.5): {len(out_days)}")
    for _, r in out_days.iterrows():
        ev = nearest_event(r["date_gregorian"], events, window_days=2)
        cf = calendar_flags(r["date_gregorian"], cal)
        print(f"  {r['DateReserve']}  ρ={r['rho_w']:.4f}  ModZ={r['modz']:+.1f}  "
              f"Res={int(r['Res']):>6d}  {(ev or '—')[:40]}  {cf[:40]}")


def main() -> None:
    setup()
    df = load_dataset()
    flags = run_detection(df)
    run_root_cause(df, flags)


if __name__ == "__main__":
    main()
