"""تست‌های خودکار snapshot v4 — بند ۱۳-۸ و دروازه‌های بند ۱۳-۵.

**چرا تست و نه تحلیل یک‌باره.** بند ۱۳-۸ صریح می‌گوید آزمون پایداری هدف باید «تست
خودکار» شود نه «تحلیل یک‌باره»، چون تغییر رژیم باید **دیده** شود نه اینکه منتظر بمانیم
کسی دوباره همان نوت‌بوک را اجرا کند. یافته F90 نشان داد سهم تک‌کدهای `ReserveStatus`
تا ۱۰ برابر نوسان می‌کند و مدلی که روی تک‌کد ساخته شود روی سال بعد هدفش تقریباً صفر
می‌شود **بدون اینکه هیچ خطایی بدهد** — دقیقاً همان دسته‌ی شکستی که تست برای آن است.

قرارداد همان `src/models/tests.py` است: هر تست `(bool, str)` برمی‌گرداند.

اجرا: `python -m src.data.tests_v4`
"""

from __future__ import annotations

import pandas as pd

from src.data.food_dim_v4 import OUTPUT_PATH as FOOD_DIM_PATH
from src.data.load_v4 import (
    DONT_RECEIVE_STATUSES,
    load_clean_v4,
    status_stability_table,
)
from src.data.restaurant_dim_v4 import OUTPUT_PATH as RESTAURANT_DIM_PATH

#: بیشینه‌ی نوسان مجاز نرخ اجتماع بین سال‌های شمسی، بر حسب واحد درصد. F102 کل جابه‌جایی
#: را «زیر یک واحد درصد» ثبت کرده (۷.۷۰/۸.۵۴/۸.۶۱)؛ آستانه‌ی ۲ فضای تنفس می‌دهد بدون
#: اینکه یک تغییر رژیم واقعی از آن رد شود.
MAX_UNION_DRIFT_PP = 2.0

#: نرخ اجتماع باید در این بازه بماند. بیرون‌زدن یعنی تعریف هدف عوض شده است.
UNION_RATE_BOUNDS = (0.05, 0.12)

_CACHE: dict[str, pd.DataFrame] = {}


def _data() -> pd.DataFrame:
    if "df" not in _CACHE:
        _CACHE["df"] = load_clean_v4(["miladi", "DateReserve", "mealid", "ReserveStatus",
                                      "Count", "RestaurantName", "FoodName"])
    return _CACHE["df"]


# ---------------------------------------------------------------------------
# بند ۱۳-۸ — پایداری هدف
# ---------------------------------------------------------------------------


def academic_year(jyear: pd.Series, jmonth: pd.Series) -> pd.Series:
    """سال تحصیلی از تاریخ شمسی: از شهریور (ماه ۶) شروع می‌شود.

    مبنای مرز، خودِ تقویم آموزشی است: نیمسال اول ۱۴۰۴-۱۴۰۵ در ۲۹ شهریور ۱۴۰۴ شروع
    می‌شود و نیمسال دوم ۱۴۰۳-۱۴۰۴ در ۱۲ تیر ۱۴۰۴ تمام (F105).
    """
    start = jyear.where(jmonth >= 6, jyear - 1)
    return start.astype(str) + "-" + (start + 1).astype(str)


def test_union_rate_stable_across_years() -> tuple[bool, str]:
    """نرخ **اجتماع** باید بین سال‌های **تحصیلی** پایدار بماند (F90/F102).

    ⚠️ هر سه تعریف «سال» عدد متفاوتی می‌دهند و این تست عمداً همانی را می‌گیرد که
    F102 روی آن ثبت شده — **سال تحصیلی**:

    | تعریف | اول | دوم | سوم | نوسان |
    |---|---|---|---|---|
    | سال میلادی | ۷.۲۰ | ۸.۱۶ | ۸.۸۴ (+۲۰۲۶ ناقص: ۱۰.۶۶) | **۳.۴۵** |
    | سال شمسی | ۷.۷۰ | ۸.۲۶ | ۸.۸۹ | ۱.۱۹ |
    | **سال تحصیلی** (این تست) | **۷.۷۴** | **۸.۶۷** | **۸.۷۷** | **۱.۰۴** |
    | سال تحصیلی، شبکه‌ی فیلترشده | ۷.۷۱ | ۸.۴۹ | ۸.۶۵ | ۰.۹۴ |

    سال میلادی بدترین انتخاب است چون ۲۰۲۳ و ۲۰۲۶ در این snapshot ناقص‌اند (از
    ۲۰۲۳-۰۹-۲۵ و تا ۲۰۲۶-۰۲-۲۰) و هرکدام فقط نیمی از یک سال تحصیلی‌اند؛ نسخه‌ی اول
    همین تست با آن گروه‌بندی نوسان ۳.۴۵ واحد درصد می‌داد و شکست — نه به‌خاطر تغییر
    رژیم، بلکه به‌خاطر مقایسه‌ی نیم‌سال با سال کامل.

    ⚠️ دو سطر آخر هر دو «سال تحصیلی»اند و باز هم فرق دارند: این تست روی **همه‌ی
    رده‌ها با وزن ردیف** حساب می‌کند و شبکه روی **`core` + `is_dish` با وزن `Count`**.
    عدد ثبت‌شده در F102 (۷.۷۰/۸.۵۴/۸.۶۱) به سطر آخر نزدیک است. هر نقل‌قولی از این
    اعداد باید بگوید از کدام مسیر آمده — اختلافشان کوچک است ولی صفر نیست.
    """
    df = _data()
    ay = academic_year(df["jyear"], df["jmonth"])
    by_year = df.groupby(ay)["dont_receive"].mean() * 100
    drift = float(by_year.max() - by_year.min())
    ok = drift <= MAX_UNION_DRIFT_PP
    return ok, f"نوسان نرخ اجتماع {drift:.2f} واحد درصد (حد {MAX_UNION_DRIFT_PP}) · " + \
               " / ".join(f"{y}:{v:.2f}٪" for y, v in by_year.items())


def test_union_rate_within_bounds() -> tuple[bool, str]:
    """نرخ کل باید در بازه‌ی مورد انتظار بماند — پرچم تغییر تعریف هدف."""
    rate = float(_data()["dont_receive"].mean())
    lo, hi = UNION_RATE_BOUNDS
    return lo <= rate <= hi, f"نرخ اجتماع {rate:.4f} (بازه‌ی مجاز {lo}–{hi})"


def test_single_codes_are_unstable_as_documented() -> tuple[bool, str]:
    """⚠️ این تست **عمداً بی‌ثباتی را تأیید می‌کند**، نه ثبات را.

    F90 گفت تک‌کدها ناپایدارند و فقط اجتماعشان پایدار است. اگر روزی تک‌کدها هم پایدار
    شوند، آن هم یک **تغییر** است و باید دیده شود: یعنی یا سامانه رفتارش را عوض کرده یا
    داده فیلتر شده. در هر دو حالت فرضِ زیربنای قاعده‌ی پایداری هدف باید بازبینی شود.
    """
    table = status_stability_table(_data())
    singles = [c for c in DONT_RECEIVE_STATUSES if c in table.columns]
    if not singles:
        return False, "هیچ‌کدام از سه کد عدم‌دریافت در داده نیست — پرچم قرمز"
    spreads = {c: float(table[c].max() - table[c].min()) for c in singles}
    union_spread = float(table["union"].max() - table["union"].min())
    max_single = max(spreads.values())
    ok = max_single > union_spread
    return ok, (f"بیشینه‌ی نوسان تک‌کد {max_single:.1f} واحد درصد در برابر اجتماع "
                f"{union_spread:.1f} — " + ("طبق F90" if ok else "⚠️ الگو عوض شده"))


def test_all_expected_status_codes_present() -> tuple[bool, str]:
    """سخت‌کردن بند ۱۳-۸: نبودِ یک کد مورد انتظار الان **بی‌صدا** رد می‌شود.

    `clean_individual` فقط کدهای **ناشناخته** را خطا می‌دهد؛ اگر یک کدِ شناخته‌شده
    (مثلاً `منقضی شده`) از داده **غایب** شود، همه‌چیز بی‌خطا کار می‌کند و هدف بی‌صدا
    کوچک می‌شود. این تست همان حالت را می‌گیرد.
    """
    present = set(_data()["ReserveStatus"].dropna().unique())
    missing = DONT_RECEIVE_STATUSES - present
    return not missing, ("هر سه کد عدم‌دریافت حاضرند" if not missing
                         else f"⚠️ کد غایب: {sorted(missing)}")


def test_no_status_code_dominates_alone() -> tuple[bool, str]:
    """هیچ تک‌کدی نباید بیش از ۹۰٪ اجتماع را بسازد.

    اگر بسازد، یعنی عملاً هدف به یک تک‌کد فروکاسته شده و F90 دیگر محافظت نمی‌کند.
    """
    df = _data()
    dont = df[df["dont_receive"]]
    shares = dont["ReserveStatus"].value_counts(normalize=True)
    top, share = shares.index[0], float(shares.iloc[0])
    return share <= 0.90, f"بیشترین سهم تک‌کد: «{top}» = {share:.1%} از اجتماع"


# ---------------------------------------------------------------------------
# دروازه‌های بند ۱۳-۵ — پوشش نگاشت
# ---------------------------------------------------------------------------


def test_every_restaurant_has_dim_row() -> tuple[bool, str]:
    """هیچ نام سلفی نباید بی‌ویژگی بماند — «نام بدون نگاشت نباید بی‌صدا حذف شود»."""
    names = set(_data()["RestaurantName"].astype(str).unique())
    known = set(pd.read_csv(RESTAURANT_DIM_PATH)["raw_name"].astype(str))
    missing = names - known
    return not missing, (f"هر {len(names)} نام سلف پوشش دارد" if not missing
                         else f"{len(missing)} نام بی‌ردیف: {sorted(missing)[:3]}")


def test_every_food_has_dim_row() -> tuple[bool, str]:
    names = set(_data()["FoodName"].astype(str).unique())
    known = set(pd.read_csv(FOOD_DIM_PATH)["raw_food_name"].astype(str))
    missing = names - known
    return not missing, (f"هر {len(names)} نام غذا پوشش دارد" if not missing
                         else f"{len(missing)} نام بی‌ردیف: {sorted(missing)[:3]}")


def test_restaurant_dim_has_no_unknown_campus() -> tuple[bool, str]:
    """هر سلف باید شهری داشته باشد که در داده‌ی هواشناسی هست، وگرنه الحاق بی‌صدا
    ردیف‌ها را `NaN` می‌کند."""
    from src.config import DATA_EXTERNAL

    dim = pd.read_csv(RESTAURANT_DIM_PATH)
    weather_cities = set(pd.read_csv(
        DATA_EXTERNAL / "weather_aqi_by_city_v4.csv", usecols=["city"]
    )["city"].unique())
    bad = set(dim["city"].dropna().unique()) - weather_cities
    return not bad and dim["city"].notna().all(), (
        f"هر {dim['city'].nunique()} شهر در داده‌ی هواشناسی هست" if not bad else f"شهر بی‌هواشناسی: {bad}"
    )


def test_count_is_not_binary() -> tuple[bool, str]:
    """⚠️ تأیید F94 به‌عنوان تست: اگر روزی `Count` دوباره باینری شود، فرضِ «جمع بزن،
    ردیف نشمار» بی‌خطر می‌شود ولی **کدی که ردیف می‌شمارد هم بی‌صدا درست می‌شود** —
    و آن وقت باگ بعدی دیرتر پیدا می‌شود."""
    counts = _data()["Count"]
    n_multi = int((counts > 1).sum())
    return n_multi > 0, f"{n_multi:,} ردیف با Count>1 (F94) — جمع‌زدن الزامی است"


_ALL_TESTS = [
    test_union_rate_stable_across_years,
    test_union_rate_within_bounds,
    test_single_codes_are_unstable_as_documented,
    test_all_expected_status_codes_present,
    test_no_status_code_dominates_alone,
    test_every_restaurant_has_dim_row,
    test_every_food_has_dim_row,
    test_restaurant_dim_has_no_unknown_campus,
    test_count_is_not_binary,
]


def run_all() -> bool:
    all_ok = True
    for fn in _ALL_TESTS:
        ok, msg = fn()
        print(f"  {'✅' if ok else '❌'} {fn.__name__:<46s} {msg}")
        all_ok &= ok
    return all_ok


if __name__ == "__main__":
    print("=" * 100)
    print("تست‌های خودکار snapshot v4 — پایداری هدف و پوشش نگاشت (بندهای ۱۳-۵ و ۱۳-۸)")
    print("=" * 100)
    if not run_all():
        raise AssertionError("یک یا چند تست v4 شکست خورد")
    print("\nهمه‌ی تست‌ها PASS")
