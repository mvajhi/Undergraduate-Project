"""بند ۵.۱۹ — پل مدل A↔B: تجمیع تاریخچه‌ی رزروکنندگان به فیچر سطح سلول.

**چرا مجاز است.** رزرو ۷۲ ساعت پیش از سرو بسته می‌شود، پس در لحظه‌ی برش **هویت
رزروکنندگان کاملاً معلوم است**. آنچه معلوم نیست، *نتیجه*ی آن‌هاست — و ما فقط از
تاریخچه‌ی پیش‌از‌برشِ همان افراد استفاده می‌کنیم.

**چرا میانگین ساده کافی نیست.** فاز ۴ نشان داد میانگینِ تاریخچه‌ها در سطح سلول تقریباً
هیچ اضافه نمی‌کند (ΔR²=+۰.۰۰۲۴، F59) — چون اثر فرد و روز **ضرب‌شونده** است نه
جمع‌شونده (F62). پس علاوه بر میانگین، **پراکندگی** توزیع تاریخچه‌ها هم استخراج می‌شود
(std، صدک ۹۰، سهم افراد پرریسک): اگر اثر ضربی باشد، دُم توزیع مهم‌تر از مرکز آن است.

⚠️ **و مهم‌تر:** این فیچر فقط برای **شام** ارزش دارد — درون سلف×وعده، شام r=+۰.۵۱۹ و
ناهار r=−۰.۰۲۰ (F63). دلیل: شام خوابگاهی جمعیت کوچک و متغیر دارد، ناهار دانشکده
جمعیت بزرگ و همگن.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: آستانه‌ی «سابقه‌ی کافی» برای اینکه تاریخچه‌ی فرد وارد تجمیع شود
MIN_HIST = 10
#: آستانه‌ی «فرد پرریسک» برای فیچر سهم دُم
HIGH_RISK_RATE = 0.20


def build_composition(person_feat: pd.DataFrame) -> pd.DataFrame:
    """از جدول فیچر سطح فرد، فیچرهای ترکیب در سطح (روز، وعده، سلف) می‌سازد.

    ورودی باید ستون‌های `date_gregorian`, `Meal`, `restaurant_canonical` و
    `person_shrunk_norecv_rate` / `person_n_prior_reservations` داشته باشد.
    """
    p = person_feat.copy()
    p["_has_hist"] = (p["person_n_prior_reservations"] >= MIN_HIST).astype(float)
    # برای تجمیع از نرخ **کوچک‌شده** استفاده می‌کنیم تا افراد کم‌سابقه نویز تزریق نکنند
    rate = p["person_shrunk_norecv_rate"]
    p["_rate"] = rate
    p["_high_risk"] = (rate >= HIGH_RISK_RATE).astype(float)

    key = ["date_gregorian", "Meal", "restaurant_canonical"]
    g = p.groupby(key, observed=True)
    comp = g.agg(
        composition_n=("_rate", "size"),
        composition_mean=("_rate", "mean"),
        composition_std=("_rate", "std"),
        composition_p90=("_rate", lambda s: s.quantile(0.90)),
        composition_high_risk_share=("_high_risk", "mean"),
        composition_coverage=("_has_hist", "mean"),
    ).reset_index()
    comp = comp.rename(columns={"restaurant_canonical": "RestaurantName"})
    return comp


def add_composition(df: pd.DataFrame, comp: pd.DataFrame) -> pd.DataFrame:
    """الحاق به دیتاست تجمیعی + ساخت برهم‌کنش ضرب‌شونده‌ی بند ۵.۱۸."""
    out = df.merge(comp, on=["date_gregorian", "Meal", "RestaurantName"],
                   how="left", validate="many_to_one")
    # ⭐ برهم‌کنش ضرب‌شونده (F62): در روزهای بد، بی‌ثبات‌ها ۳.۳۶ برابر بیشتر می‌ریزند
    for c in ["composition_mean", "composition_p90", "composition_high_risk_share"]:
        out[f"{c}_x_dayshock"] = out[c] * out["day_shock_lag1"]
    # فیچر ترکیب فقط برای شام معنا دارد (F63) — نسخه‌ی ماسک‌شده هم ساخته می‌شود
    is_dinner = (out["Meal"] == "dinner").astype(float)
    out["composition_mean_dinner_only"] = out["composition_mean"] * is_dinner
    return out
