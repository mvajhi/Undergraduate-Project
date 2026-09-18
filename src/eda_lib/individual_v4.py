"""سطح فرد روی پنجره‌ی v4 — ساخت، دوره‌ی غیبت، و بازآزمون صفت/حالت (بند ۱۴-۶).

## چرا یک ماژول جدا

سطح فرد v4 هشت میلیون رزرو و ۴۶ هزار فرد دارد و ساختش از خام ~۹۰ ثانیه می‌گیرد.
نتیجه یک‌بار در `data/interim/person_level_v4.parquet` کش می‌شود تا هر بسته‌کاری
بند ۱۴-۶ آن را بخواند، نه بسازد.

## ⚠️ `meal_seq` و نه `date`

تحلیل دوره‌ی غیبت باید روی **توالی وعده** باشد نه توالی تاریخ. دو دلیل:

۱. هر روز **دو** فرصت دارد (ناهار و شام) و فردی که ناهار را می‌گیرد و شام را نه،
   «یک روز غایب» نیست.
۲. جمعه و تعطیلات سرویس ندارند؛ اگر با تاریخ بشماریم، یک آخر هفته‌ی عادی به‌عنوان
   «دوره‌ی غیبت» شمرده می‌شود.

پس یک شاخص سراسری `meal_seq` ساخته می‌شود: رتبه‌ی هر (تاریخ، وعده) در فهرست مرتب
همه‌ی فرصت‌های سرویسِ موجود در داده.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import DATA_INTERIM

PERSON_V4 = DATA_INTERIM / "person_level_v4.parquet"

_KEEP = ["PersonId", "miladi", "DateReserve", "mealid", "ReserveStatus", "Count",
         "RestaurantName", "FoodName", "Gender", "CollegeName", "DegreeName",
         "GroupName", "PersonType", "FieldName"]


def build_person_level_v4(force: bool = False) -> pd.DataFrame:
    """ساخت (یا خواندن کشِ) جدول سطح فرد v4، محدود به رده‌ی `core` و ناهار/شام."""
    if PERSON_V4.exists() and not force:
        return pd.read_parquet(PERSON_V4)

    from src.data.load_v4 import load_clean_v4
    from src.eda_lib.restaurant_taxonomy import CORE_CLASS, add_restaurant_class

    df = load_clean_v4(columns=_KEEP)
    df = add_restaurant_class(df)
    df = df[df.restaurant_class == CORE_CLASS].copy()

    # شاخص سراسری فرصتِ سرویس: (تاریخ، وعده) مرتب‌شده
    slots = (df[["date_gregorian", "meal"]].drop_duplicates()
               .sort_values(["date_gregorian", "meal"], ascending=[True, True])
               .reset_index(drop=True))
    slots["meal_seq"] = np.arange(len(slots))
    df = df.merge(slots, on=["date_gregorian", "meal"], how="left")

    keep = ["PersonId", "date_gregorian", "meal", "meal_seq", "dont_receive", "Count",
            "RestaurantName", "FoodName", "Gender", "CollegeName", "DegreeName",
            "GroupName", "PersonType", "jyear", "jmonth"]
    out = df[[c for c in keep if c in df.columns]].copy()
    out["dont_receive"] = out.dont_receive.astype(bool)
    PERSON_V4.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(PERSON_V4, index=False)
    return out


# ---------------------------------------------------------------------------
# دوره‌ی غیبت (بند ۱۴-۶-۲)
# ---------------------------------------------------------------------------

def spells(df: pd.DataFrame, min_reservations: int = 20) -> pd.DataFrame:
    """دوره‌های پیوسته‌ی عدم‌دریافت هر فرد، روی توالی `meal_seq`.

    خروجی: یک ردیف به‌ازای هر دوره، با طول، موقعیت شروع، و نرخ پایه‌ی فرد.
    ⚠️ فقط افرادی که دست‌کم `min_reservations` رزرو دارند؛ برای فرد با ۳ رزرو
    «دوره» معنا ندارد.
    """
    d = df[["PersonId", "meal_seq", "dont_receive"]].sort_values(["PersonId", "meal_seq"])
    n = d.groupby("PersonId").PersonId.transform("size")
    d = d[n >= min_reservations]
    if d.empty:
        return pd.DataFrame(columns=["PersonId", "start", "length", "base_rate", "n"])

    v = d.dont_receive.to_numpy()
    pid = d.PersonId.to_numpy()
    new_person = np.empty(len(d), dtype=bool)
    new_person[0] = True
    new_person[1:] = pid[1:] != pid[:-1]
    # شروع دوره = مقدار True که قبلش True نیست (یا فرد عوض شده)
    prev = np.empty(len(d), dtype=bool)
    prev[0] = False
    prev[1:] = v[:-1]
    start = v & (~prev | new_person)
    gid = np.cumsum(start) - 1
    mask = v
    sp = (pd.DataFrame({"PersonId": pid[mask], "gid": gid[mask],
                        "seq": d.meal_seq.to_numpy()[mask]})
            .groupby("gid").agg(PersonId=("PersonId", "first"), start=("seq", "min"),
                                length=("seq", "size")))
    stats = d.groupby("PersonId").dont_receive.agg(["mean", "size"])
    sp = sp.join(stats.rename(columns={"mean": "base_rate", "size": "n"}), on="PersonId")
    return sp.reset_index(drop=True)


def simulate_null_spells(base_rates: np.ndarray, ns: np.ndarray, seed: int = 42,
                         reps: int = 1) -> pd.DataFrame:
    """توزیع طول دوره **تحت استقلال**، با همان نرخ پایه‌ی هر فرد.

    فرض تحلیلی (توزیع هندسی) استفاده نمی‌شود چون طول دنباله‌ی هر فرد متفاوت و
    متناهی است و دُم را می‌بُرد. شبیه‌سازی مستقیم این را درست نگه می‌دارد.
    """
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(reps):
        for p, n in zip(base_rates, ns):
            v = rng.random(int(n)) < p
            if not v.any():
                continue
            idx = np.flatnonzero(v)
            brk = np.flatnonzero(np.diff(idx) > 1)
            lens = np.diff(np.concatenate([[-1], brk, [len(idx) - 1]]))
            out.append(lens)
    if not out:
        return pd.DataFrame({"length": []})
    return pd.DataFrame({"length": np.concatenate(out)})


def hazard_of_ending(sp: pd.DataFrame, max_len: int = 12) -> pd.DataFrame:
    """تابع مخاطره‌ی پایان دوره بر حسب طول — آیا دوره‌ی بلندتر دیرتر تمام می‌شود؟"""
    rows = []
    for k in range(1, max_len + 1):
        at_risk = int((sp.length >= k).sum())
        ends = int((sp.length == k).sum())
        rows.append({"طول": k, "در معرض": at_risk, "پایان‌یافته": ends,
                     "مخاطره": ends / at_risk if at_risk else np.nan})
    return pd.DataFrame(rows)
