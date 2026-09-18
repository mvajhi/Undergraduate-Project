"""بند ۹-۱۲ — مرز تصمیم سیاستی: چه کاری روی چه کسی، و با چه سقفی.

اجرا: python -m src.eda_lib.runners.r9j_policy

## قاعده‌ی ح در عمل

بند ۹-۱۲-۱ نسخه‌ی **درون‌نمونه‌ای** است (رتبه و اثر روی یک پنجره) و بازآزمون
مستقیم F86. بند ۹-۱۲-۲ نسخه‌ی **آینده‌نگر** است: رتبه‌بندی فقط با داده‌ی سال $t$،
اندازه‌گیری روی هدررفت سال $t+1$. فاصله‌ی این دو، هزینه‌ی واقعی سیاست است — و
F86 فقط ستون اول را داشت و هرگز اعلام نکرد که کران بالاست.

## تجزیه‌ی افت

افت از درون‌نمونه‌ای به آینده‌نگر دو منبع دارد و سیاست‌گذار باید تفکیکشان را
ببیند، چون درمانشان فرق می‌کند:

- **خطای رتبه‌بندی** — فرد می‌ماند ولی رفتارش عوض می‌شود. درمان: مدل بهتر.
- **خروج** — فرد اصلاً سال بعد نیست. درمان: هیچ مدلی حلش نمی‌کند؛ باید بودجه‌ی
  مداخله را به فراوانی خروج (F204: ۸۱.۳٪) تعدیل کرد.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import set_global_seed
from src.eda_lib import figs
from src.eda_lib.person_profile_v4 import build_person_year_v4
from src.eda_lib.runners._common_r9 import (
    N_MIN_MAIN, at_threshold, header, load_profile, md_table, setup,
)

#: کسرهای هدف‌گیری که منحنی روی آن‌ها ارزیابی می‌شود.
FRACTIONS = np.array([0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50])

V2_ANCHORS_F86 = {"اوراکل": 0.3810, "تاریخچه×حجم": 0.3803, "فقط نرخ تاریخچه": 0.2365,
                  "جمعیت×حجم": 0.1837, "تصادفی": 0.1000, "فقط جمعیت‌شناسی": 0.0872}


def _coverage(score: np.ndarray, waste: np.ndarray, fracs=FRACTIONS) -> np.ndarray:
    """سهم هدررفتِ پوشش‌داده‌شده وقتی بالاترین `frac` افراد بر اساس `score` هدف شوند."""
    order = np.argsort(-score, kind="stable")
    w = waste[order]
    cum = np.cumsum(w) / waste.sum()
    n = len(score)
    return np.array([cum[max(0, int(round(f * n)) - 1)] for f in fracs])


def _demo_score(train: pd.DataFrame, apply_to: pd.DataFrame, target: str) -> np.ndarray:
    """نرخ پیش‌بینی‌شده از جمعیت‌شناسیِ تنها — بازسازی قاعده‌ی F86."""
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import OneHotEncoder

    cols = ["demo_college", "demo_degree", "demo_session", "demo_is_dorm", "demo_gender"]
    enc = OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False)
    Xtr = enc.fit_transform(train[cols].astype(str))
    m = LinearRegression().fit(Xtr, train[target])
    return m.predict(enc.transform(apply_to[cols].astype(str)))


def p9121_in_sample(P: pd.DataFrame) -> pd.DataFrame:
    """۹-۱۲-۱ مرز هدف‌گیری درون‌نمونه‌ای (بازآزمون F86)."""
    header(f"۹-۱۲-۱ مرز هدف‌گیری — درون‌نمونه‌ای (آستانه n_min={N_MIN_MAIN})", 2)
    S = at_threshold(P, N_MIN_MAIN).reset_index(drop=True)
    waste = S.n_waste.to_numpy(dtype="float64")
    rng = np.random.default_rng(42)
    demo_rate = _demo_score(S, S, "rho_eb")

    rules = {
        "اوراکل": waste,
        "تاریخچه×حجم": S.rho_eb.to_numpy() * S.n_res.to_numpy(),
        "فقط نرخ تاریخچه": S.rho_eb.to_numpy(),
        "فقط حجم مصرف": S.n_res.to_numpy(dtype="float64"),
        "جمعیت×حجم": demo_rate * S.n_res.to_numpy(),
        "فقط جمعیت‌شناسی": demo_rate,
        "تصادفی": rng.random(len(S)),
    }
    rows = []
    for name, sc in rules.items():
        cov = _coverage(sc, waste)
        r = {"قاعده": name}
        r.update({f"{int(f * 100)}٪": c for f, c in zip(FRACTIONS, cov)})
        r["لنگر v2 در ۱۰٪"] = V2_ANCHORS_F86.get(name, np.nan)
        rows.append(r)
    t = pd.DataFrame(rows)
    print(md_table(t.round(4)))

    fig, ax = figs.new(w=7.6, h=4.8)
    for name, sc in rules.items():
        f = np.linspace(0.01, 1.0, 80)
        ax.plot(f * 100, _coverage(sc, waste, f) * 100, lw=1.9, label=name)
    ax.set_xlabel("درصد افراد هدف‌گیری‌شده")
    ax.set_ylabel("درصد هدررفت پوشش‌داده‌شده")
    ax.set_title("مرز هدف‌گیری — درون‌نمونه‌ای (کران بالا)")
    ax.legend(fontsize=8, loc="lower right")
    ax.axvline(10, color=figs.C_GREY, ls=":", lw=1.2)
    figs.save(fig, "r9j_01_targeting_frontier_insample", t,
              caption="بازآزمون F86 — و یادآوری اینکه این منحنی کران بالاست، نه انتظار")
    return t


def p9122_prospective() -> tuple[pd.DataFrame, pd.DataFrame]:
    """⭐⭐ ۹-۱۲-۲ همان مرز، ولی آینده‌نگر — عدد سرفصل دور ۹."""
    header("۹-۱۲-۲ مرز هدف‌گیری — آینده‌نگر (رتبه با سال t، اثر در سال t+1)", 2)
    Y = build_person_year_v4()
    P = load_profile()
    years = list(Y.year_block.cat.categories) if hasattr(Y.year_block, "cat") \
        else sorted(Y.year_block.unique())
    rng = np.random.default_rng(42)

    rows, decomp = [], []
    for y0, y1 in zip(years[:-1], years[1:]):
        a = Y[(Y.year_block == y0) & (Y.n_res >= N_MIN_MAIN)].set_index("PersonId")
        b = Y[Y.year_block == y1].set_index("PersonId")
        # جامعه‌ی تصمیم: هرکس که در پایان سال t قابل‌رتبه‌بندی است. کسی که سال بعد
        # نمی‌آید هدررفت صفر می‌آورد — این واقعیتِ بودجه است، نه نقص داده.
        d = a.join(b[["n_waste", "n_res"]].rename(
            columns={"n_waste": "waste_next", "n_res": "res_next"}), how="left")
        d[["waste_next", "res_next"]] = d[["waste_next", "res_next"]].fillna(0.0)
        d = d.join(P.set_index("PersonId")[["demo_college", "demo_degree",
                                            "demo_session", "demo_is_dorm",
                                            "demo_gender"]], how="left")
        churn = float((d.res_next == 0).mean())
        demo_rate = _demo_score(d.reset_index(), d.reset_index(), "rho_eb")
        w_next = d.waste_next.to_numpy(dtype="float64")

        rules = {
            "اوراکل سال بعد (دست‌نیافتنی)": w_next,
            "تاریخچه×حجم سال t": d.rho_eb.to_numpy() * d.n_res.to_numpy(),
            "فقط هدررفت مطلق سال t": d.n_waste.to_numpy(dtype="float64"),
            "فقط نرخ EB سال t": d.rho_eb.to_numpy(),
            "فقط حجم سال t": d.n_res.to_numpy(dtype="float64"),
            "فقط جمعیت‌شناسی": demo_rate,
            "تصادفی": rng.random(len(d)),
        }
        for name, sc in rules.items():
            cov = _coverage(sc, w_next)
            r = {"جفت سال": f"{y0} → {y1}", "قاعده": name, "n": len(d),
                 "نرخ خروج": churn}
            r.update({f"{int(f * 100)}٪": c for f, c in zip(FRACTIONS, cov)})
            rows.append(r)

        # تجزیه‌ی افت: ماندگارها در برابر کل جامعه
        stay = d[d.res_next > 0]
        sc_stay = stay.rho_eb.to_numpy() * stay.n_res.to_numpy()
        decomp.append({
            "جفت سال": f"{y0} → {y1}",
            "پوشش ۱۰٪ — کل جامعه": _coverage(
                d.rho_eb.to_numpy() * d.n_res.to_numpy(), w_next)[3],
            "پوشش ۱۰٪ — فقط ماندگارها": _coverage(
                sc_stay, stay.waste_next.to_numpy(dtype="float64"))[3],
            "پوشش ۱۰٪ — اوراکل سال بعد": _coverage(w_next, w_next)[3],
            "نرخ خروج": churn,
        })

    t = pd.DataFrame(rows)
    dc = pd.DataFrame(decomp)
    print(md_table(t.drop(columns=["n"]).round(4)))
    print("\n▸ تجزیه‌ی افت (در نقطه‌ی هدف‌گیری ۱۰٪):")
    print(md_table(dc.round(4)))

    fig, ax = figs.new(1, 2, w=5.8, h=4.6)
    pair = t["جفت سال"].iloc[0]
    sub = t[t["جفت سال"] == pair]
    fcols = [f"{int(f * 100)}٪" for f in FRACTIONS]
    for _, r in sub.iterrows():
        ax[0].plot(FRACTIONS * 100, [r[c] * 100 for c in fcols], "o-", lw=1.7,
                   label=r["قاعده"], ms=4)
    ax[0].set_xlabel("درصد افراد هدف‌گیری‌شده")
    ax[0].set_ylabel("درصد هدررفت **سال بعد**")
    ax[0].set_title(f"مرز آینده‌نگر\n{pair}", fontsize=10)
    ax[0].legend(fontsize=7, loc="lower right")

    x = np.arange(len(dc))
    ax[1].bar(x - 0.27, dc["پوشش ۱۰٪ — اوراکل سال بعد"] * 100, 0.27,
              color=figs.C_GREY, label="اوراکل سال بعد")
    ax[1].bar(x, dc["پوشش ۱۰٪ — فقط ماندگارها"] * 100, 0.27,
              color=figs.C_OK, label="قاعده، فقط ماندگارها")
    ax[1].bar(x + 0.27, dc["پوشش ۱۰٪ — کل جامعه"] * 100, 0.27,
              color=figs.C_MAIN, label="قاعده، کل جامعه")
    ax[1].set_xticks(x, dc["جفت سال"], fontsize=7)
    ax[1].set_ylabel("پوشش هدررفت سال بعد در ۱۰٪ (٪)")
    ax[1].set_title("تجزیه‌ی افت: خطای رتبه در برابر خروج", fontsize=10)
    ax[1].legend(fontsize=7)
    figs.save(fig, "r9j_02_targeting_frontier_prospective", t,
              caption="عدد سرفصل دور ۹ — آنچه یک سیاست واقعی سال بعد به دست می‌آورد")
    return t, dc


def p9124_sensitivity(P: pd.DataFrame) -> pd.DataFrame:
    """۹-۱۲-۴ اگر مداخله نرخ گروه هدف را x٪ کم کند، کل هدررفت چقدر کم می‌شود؟"""
    header("۹-۱۲-۴ جدول حساسیت مداخله (بدون ادعای علّی)", 2)
    S = at_threshold(P, N_MIN_MAIN)
    total = S.n_waste.sum()
    rows = []
    for frac in (0.05, 0.10, 0.20):
        k = int(round(frac * len(S)))
        score = S.rho_eb.to_numpy() * S.n_res.to_numpy()
        top = S.iloc[np.argsort(-score, kind="stable")[:k]]
        share = top.n_waste.sum() / total
        # با ماندگاری واقعی دهک (بند ۹-۵-۳) تعدیل می‌شود: نیمی از فهرست سال بعد
        # همان‌جا نیست، پس اثر مؤثر نصف اثر اسمی است.
        for eff in (0.10, 0.20, 0.30):
            rows.append({"سهم هدف‌گیری": frac, "افراد": k,
                         "سهم هدررفت گروه": share, "کاهش فرضی نرخ گروه": eff,
                         "کاهش کل — اسمی": share * eff,
                         "کاهش کل — با تعدیل ماندگاری ۰.۵": share * eff * 0.5})
    t = pd.DataFrame(rows)
    print(md_table(t.round(4)))
    print("\n⚠️ این جدول «اگر» است، نه «خواهد شد». F64 صریح ثبت کرده که علیت معکوس")
    print("   (پرغیبت‌ها رزرو را رها می‌کنند) رد نشده. ستون تعدیل‌شده از ماندگاری")
    print("   دهک بند ۹-۵-۳ (~۰.۵) می‌آید، نه از اثر مداخله.")

    fig, ax = figs.new(w=7.4, h=4.4)
    for frac, grp in t.groupby("سهم هدف‌گیری"):
        ax.plot(grp["کاهش فرضی نرخ گروه"] * 100, grp["کاهش کل — اسمی"] * 100,
                "o-", label=f"هدف‌گیری {frac:.0%} — اسمی")
        ax.plot(grp["کاهش فرضی نرخ گروه"] * 100,
                grp["کاهش کل — با تعدیل ماندگاری ۰.۵"] * 100, "s--", alpha=0.6,
                label=f"هدف‌گیری {frac:.0%} — تعدیل‌شده")
    ax.set_xlabel("کاهش فرضی نرخ گروه هدف (٪)")
    ax.set_ylabel("کاهش کل هدررفت (٪)")
    ax.set_title("سقف هر مداخله — پیش از هر ادعای اثربخشی")
    ax.legend(fontsize=7, ncol=2)
    figs.save(fig, "r9j_03_intervention_sensitivity", t,
              caption="قاعده‌ی چ: هر توصیه با سقفش می‌آید")
    return t


def p9125_segment_to_cell(P: pd.DataFrame) -> pd.DataFrame:
    """۹-۱۲-۵ پل به سطح تصمیم: هدررفت گروه هدف در کدام سلف×وعده می‌نشیند؟"""
    header("۹-۱۲-۵ پل از سطح فرد به سطح تصمیم (سلف×وعده)", 2)
    from src.eda_lib.person_profile_v4 import build_person_rows_v4

    S = at_threshold(P, N_MIN_MAIN)
    k = int(round(0.10 * len(S)))
    score = S.rho_eb.to_numpy() * S.n_res.to_numpy()
    target = set(S.iloc[np.argsort(-score, kind="stable")[:k]].PersonId)

    rows_df = build_person_rows_v4(columns=["PersonId", "RestaurantName", "meal",
                                            "dont_receive"])
    rows_df["is_target"] = rows_df.PersonId.isin(target)
    g = (rows_df.groupby(["RestaurantName", "meal"], observed=True)
         .agg(هدررفت_کل=("dont_receive", "sum"), رزرو=("dont_receive", "size")))
    gt = (rows_df[rows_df.is_target].groupby(["RestaurantName", "meal"], observed=True)
          .agg(هدررفت_هدف=("dont_receive", "sum")))
    t = g.join(gt, how="left").fillna(0).reset_index()
    t["سهم هدف از هدررفت"] = t.هدررفت_هدف / t.هدررفت_کل.replace(0, np.nan)
    t["سهم از کل هدررفت"] = t.هدررفت_کل / t.هدررفت_کل.sum()
    t = t.sort_values("هدررفت_کل", ascending=False)
    print("ده سلف×وعده‌ی پرهدررفت:")
    print(md_table(t.head(10).round(4)))
    print(f"\nپوشش کلی گروه هدف (۱۰٪ افراد): "
          f"{t.هدررفت_هدف.sum() / t.هدررفت_کل.sum():.4f} از کل هدررفت")
    print(f"دامنه‌ی سهم هدف بین سلف×وعده‌ها: "
          f"{t['سهم هدف از هدررفت'].min():.3f} تا {t['سهم هدف از هدررفت'].max():.3f}")

    top = t.head(18)
    fig, ax = figs.new(w=8.0, h=5.6)
    y = np.arange(len(top))
    ax.barh(y, top["سهم از کل هدررفت"] * 100, color=figs.C_GREY, label="کل هدررفت")
    ax.barh(y, top["سهم از کل هدررفت"] * top["سهم هدف از هدررفت"] * 100,
            color=figs.C_ALT, label="سهم گروه هدف")
    ax.set_yticks(y, [f"{r.RestaurantName[:26]}·{r.meal}" for r in top.itertuples()],
                  fontsize=6)
    ax.set_xlabel("سهم از کل هدررفت پنجره (٪)")
    ax.set_title("هدررفت گروه هدف کجا می‌نشیند")
    ax.legend(fontsize=8)
    figs.save(fig, "r9j_04_segment_to_cell", t,
              caption="بخش‌بندی فردی که به سلف×وعده ترجمه نشود، برای تصمیم پخت بی‌مصرف است")
    return t


def main() -> None:
    set_global_seed()
    setup()
    P = load_profile()
    header("۹-۱۲ مرز تصمیم سیاستی")
    p9121_in_sample(P)
    p9122_prospective()
    p9124_sensitivity(P)
    p9125_segment_to_cell(P)
    header("پایان بند ۹-۱۲")


if __name__ == "__main__":
    main()
