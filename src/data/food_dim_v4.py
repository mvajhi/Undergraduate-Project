"""جدول بعد غذاها برای snapshot نسخه‌ی ۴ — جایگزین `food_mapping.csv`.

**چرا بازسازی (یافته F91).** `food_mapping.csv` هر نام خام را به نام غذاهای **فایل
تجمیعی** می‌برد و آن مرجع فقط ۴۹۶ از ۱٬۷۶۲ نام v4 را می‌شناسد. ۱٬۲۶۶ نام بی‌نگاشت
می‌مانند و در تجمیع بی‌صدا حذف می‌شوند.

**سه لایه‌ی نام، نه یکی.** ستون `FoodName` سه چیز متفاوت را قاطی کرده و هر تحلیلی باید
بداند با کدام‌شان کار می‌کند:

| لایه | تعداد | نمونه | کِی به‌کار می‌آید |
|---|---|---|---|
| `raw_food_name` | ۱٬۷۳۰ | `چلوکباب کوبیده+دوغ+کره (تکنفره)+گوجه فرنگی کبابی` | ردیابی به داده‌ی خام |
| `main_food` | ۱۸۳ | `چلوکباب کوبیده` | — |
| `canonical_dish` | ۱۶۱ | `چلوکباب کوبیده` | **محور $f$ در شبکه‌ی L1** |

فاصله‌ی ۱۸۳ به ۱۶۱ رسم‌الخط است نه معنا: `چلوخورشت قورمه سبزی` و
`چلو خورش قورمه سبزی` یک غذایند، `خوراک شینسل مرغ` غلط تایپی `خوراک شنیسل مرغ` است، و
`خوراک کتلت با دورچین` همان `خوراک کتلت` است با دورچین. بدون این لایه، یک غذا در شبکه
دو سلول کم‌نمونه می‌شود به‌جای یک سلول سالم.

**۵۰ غذا ۹۵.۳٪ پرس‌ها را می‌پوشانند** — محور $f$ عملاً کوچک است، با یک دم بلند که باید
دیده شود ولی مدل را بزرگ نمی‌کند.

⚠️ **همه‌ی `FoodName`ها غذا نیستند.** ستون `is_dish` ردیف‌هایی مثل
`رستوران کلاسیک -ظرفیت محدود-پیش پرداخت:` (توصیف قرارداد مکمل، F99)، `بسته ارزاق`
(بسته‌ی خواربار)، `میوه و تره بار روزانه` و `رستوران تعطیل است` (وضعیت سرویس، نه غذا)
را جدا می‌کند. اگر این‌ها غذا شمرده شوند، «تنوع منو»ی یک سلف با تعداد قراردادهایش زیاد
می‌شود.
"""

from __future__ import annotations

import logging
import re

import pandas as pd

from src.config import DATA_EXTERNAL
from src.data.load_v4 import _normalize_categories, load_raw_v4
from src.data.mapping import parse_food_name

logger = logging.getLogger(__name__)

OUTPUT_PATH = DATA_EXTERNAL / "food_dim_v4.csv"


# ---------------------------------------------------------------------------
# یکسان‌سازی رسم‌الخط
# ---------------------------------------------------------------------------

#: هر قاعده با نمونه‌ی واقعی‌اش. هیچ‌کدام حدس نیست — همه از فهرست ۱۸۳ غذای اصلیِ
#: خودِ داده بیرون آمده‌اند و ادغام‌هایشان در `merge_report` قابل‌بازبینی است.
_DISH_RULES: list[tuple[str, str, str]] = [
    # «۲ خوراک چیکن استروگانوف» → همان غذا، شمارش پرس است نه نام
    (r"^\s*\d+\s+(?=\S)", "", "شمارنده‌ی ابتدای نام"),
    # «چلوخورشت قورمه سبزی» ≡ «چلو خورش قورمه سبزی»
    (r"خورشت", "خورش", "خورشت ← خورش"),
    # واژه‌هایی که گاهی به واژه‌ی بعدی می‌چسبند: «چلوکباب»، «سالادماکارونی»، «کوکوسبزی»
    (r"(چلو|سالاد|کوکو)(?=\S)", r"\1 ", "جداکردن واژه‌ی چسبیده"),
    (r"پلوبا", "پلو با ", "جداکردن پلو-با"),
    # «خوراک کتلت با دورچین» ≡ «خوراک کتلت» — دورچین مخلفات است نه غذا
    (r"\s*با\s*دورچین\s*$", "", "حذف پسوند دورچین"),
    # غلط تایپی مستند: «شینسل» ≡ «شنیسل»
    (r"شینسل", "شنیسل", "غلط تایپی شنیسل"),
    (r"میرزا\s*قاسمی", "میرزاقاسمی", "میرزاقاسمی"),
    # «با» اختیاری است و سه رسم‌الخط دارد: «سبزی پلو با گوشت»، «سبزی پلو گوشت»،
    # «سبزی پلوباگوشت». حذفش هر سه را یکی می‌کند.
    # ⚠️ مرز واژه لازم است وگرنه «مسما بادمجان» به «مسما دمجان» تبدیل می‌شود.
    (r"\bبا\b\s*", "", "حذف «با»ی اختیاری"),
    (r"\bبا(?=گوشت|مرغ|سویا|ماهی)", "", "حذف «با»ی چسبیده"),
]


def dish_key(main_food: str) -> str:
    """کلید گروه‌بندی غذا — فقط برای یکی‌کردن رسم‌الخط، نه برای نمایش.

    عمداً تهاجمی‌تر از نام نمایشی است (مثلاً «با» را حذف می‌کند و
    `سبزی پلو گوشت` می‌سازد). خواناییِ خروجی از `canonical_dish` می‌آید که همیشه
    یکی از نام‌های **واقعاً مشاهده‌شده** است — پس هیچ نام ساختگی وارد گزارش نمی‌شود.
    """
    out = main_food
    for pattern, repl, _ in _DISH_RULES:
        out = re.sub(pattern, repl, out)
    return re.sub(r"\s+", " ", out).strip()


# ---------------------------------------------------------------------------
# رده‌ی غذا و تشخیص «غذا نبودن»
# ---------------------------------------------------------------------------

#: ردیف‌هایی که اصلاً یک وعده‌ی پخته‌شده نیستند. هر کدام یک ماهیت متفاوت دارد و
#: به دلیل خودش بیرون است، نه به‌خاطر کم‌حجم‌بودن.
NON_DISH_PATTERNS: list[tuple[str, str]] = [
    ("contract", r"ظرفیت محدود|پیش پرداخت|رستوران مکمل"),  # توصیف قرارداد مکمل (F99)
    ("closed", r"رستوران تعطیل"),                            # وضعیت سرویس، نه غذا
    ("grocery", r"بسته ارزاق|تره بار"),                      # بسته‌ی خواربار
    ("buffet_purchase", r"خرید از بوفه"),                    # تراکنش خرید (F101)
    # «ناهار رستوران آزاد …» نام یک وعده است نه یک غذا — چه بپزند چه نپزند همین ثبت می‌شود
    ("event_meal", r"اردو علمی|غذای ویژه|صبحانه هفتگی|بخش دانشجویی رستوران باشگاه|رستوران آزاد"),
]

#: رده‌ی آشپزی. **ترتیب معنا دارد** — هر غذا به نخستین الگویی می‌افتد که می‌خواند:
#: `چلو خورش قلیه ماهی` پیش از آنکه `khoresh` شود `fish` می‌شود، چون ماده‌ی اصلی
#: اطلاعات بیشتری از روش پخت دارد.
DISH_GROUP_PATTERNS: list[tuple[str, str]] = [
    ("fish", r"ماهی|میگو"),
    ("khoresh", r"خورش"),
    ("kebab", r"کباب|کوبیده|جوجه|بریان"),
    ("sandwich", r"ساندویچ|برگر|هات داگ|ژامبون"),
    ("pizza_pasta", r"پیتزا|لازانیا|پاستا|ماکارونی|نودل|نودالیت"),
    ("polo", r"پلو|دمی|باقلا|شوید|کلم پلو"),
    ("khorak", r"^خوراک|^چلو |دلمه"),
]


def classify_dish(canonical: str) -> str:
    for label, pattern in NON_DISH_PATTERNS:
        if re.search(pattern, canonical):
            return label
    for label, pattern in DISH_GROUP_PATTERNS:
        if re.search(pattern, canonical):
            return label
    return "other"


def is_dish(food_group: str) -> bool:
    return food_group not in {label for label, _ in NON_DISH_PATTERNS}


# ---------------------------------------------------------------------------
# ساخت
# ---------------------------------------------------------------------------


def build_food_dim(output_path=OUTPUT_PATH) -> pd.DataFrame:
    df = load_raw_v4(["FoodName", "Count", "miladi"], meals=(1, 3))
    df["FoodName"] = _normalize_categories(df["FoodName"])

    agg = df.groupby("FoodName", observed=True).agg(
        rows=("Count", "size"),
        portions=("Count", "sum"),
        first_day=("miladi", "min"),
        last_day=("miladi", "max"),
    )
    agg.index.name = "raw_food_name"
    out = agg.reset_index()
    out["raw_food_name"] = out["raw_food_name"].astype(str)

    parsed = out["raw_food_name"].map(parse_food_name)
    out["main_food"] = parsed.map(lambda t: t[0])
    out["has_extras"] = parsed.map(lambda t: t[1])
    out["extras_list"] = parsed.map(lambda t: "|".join(t[2]))
    out["n_extras"] = parsed.map(lambda t: len(t[2]))

    out["dish_key"] = out["main_food"].map(dish_key)
    # نام نمایشی = پرحجم‌ترین رسم‌الخطِ **واقعاً مشاهده‌شده** در هر گروه. هیچ نام
    # ساختگی وارد گزارش نمی‌شود، و انتخاب قطعی است (بیشترین پرس، سپس الفبایی).
    display = (
        out.groupby(["dish_key", "main_food"], as_index=False)["portions"].sum()
        .sort_values(["dish_key", "portions", "main_food"], ascending=[True, False, True])
        .drop_duplicates("dish_key").set_index("dish_key")["main_food"]
    )
    out["canonical_dish"] = out["dish_key"].map(display)
    out["food_group"] = out["dish_key"].map(classify_dish)
    out["is_dish"] = out["food_group"].map(is_dish)

    for col in ("first_day", "last_day"):
        out[col] = pd.to_datetime(out[col]).dt.strftime("%Y-%m-%d")

    out = out.sort_values("portions", ascending=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    logger.info(
        "جدول بعد غذاها ذخیره شد: %s (%d نام خام → %d غذای کانونیک)",
        output_path, len(out), out["dish_key"].nunique(),
    )
    return out


def merge_report(dim: pd.DataFrame) -> pd.DataFrame:
    """هر ادغامی که `canonical_dish` انجام داده، برای بازبینی چشمی.

    دروازه‌ی بند ۱۳-۵ می‌گوید «نام بدون نگاشت نباید بی‌صدا حذف شود»؛ قرینه‌اش هم
    درست است — نام‌ها هم نباید بی‌صدا **با هم جمع شوند**. این تابع دقیقاً همان
    مواردی را نشان می‌دهد که باید یک‌بار با چشم تأیید شوند.
    """
    g = dim.groupby("dish_key")["main_food"].unique()
    merged = g[g.map(len) > 1]
    rows = [
        {
            "canonical_dish": dim.loc[dim.dish_key == key, "canonical_dish"].iloc[0],
            "n_variants": len(variants),
            "merged_from": " | ".join(sorted(variants)),
            "portions": int(dim.loc[dim.dish_key == key, "portions"].sum()),
        }
        for key, variants in merged.items()
    ]
    return pd.DataFrame(rows).sort_values("portions", ascending=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    from src.config import set_global_seed

    set_global_seed()
    dim = build_food_dim()

    pd.set_option("display.width", 250, "display.max_colwidth", 95)
    print(f"\nنام خام {len(dim)} → غذای اصلی {dim.main_food.nunique()} → کانونیک {dim.dish_key.nunique()}")

    print("\n=== رده‌ی غذا (بر حسب پرس) ===")
    grp = dim.groupby("food_group").agg(names=("raw_food_name", "size"), portions=("portions", "sum"))
    grp["pct"] = (100 * grp.portions / grp.portions.sum()).round(2)
    print(grp.sort_values("portions", ascending=False).to_string())

    print("\n=== ردیف‌های «غذا نیست» ===")
    nd = dim[~dim.is_dish]
    print(f"{len(nd)} نام خام · {nd.portions.sum():,.0f} پرس "
          f"({100 * nd.portions.sum() / dim.portions.sum():.2f}٪)")

    print("\n=== ادغام‌های رسم‌الخطی (نیازمند تأیید چشمی) ===")
    print(merge_report(dim).to_string(index=False))
