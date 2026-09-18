"""بارگذاری و پاکسازی پایه‌ی snapshot نسخه‌ی ۴ (استخراج parquet سطح فرد).

این ماژول **تنها نقطه‌ی ورود** داده‌ی خام v4 است. دلیل وجودش جدا از
`clean_individual.py` این است که آن ماژول حول `pd.read_excel` و فهرست فایل‌های ماهانه
ساخته شده؛ v4 یک فایل parquet یکپارچه‌ی ۸.۵ میلیون‌ردیفی است و الگوی خواندنش فرق می‌کند
(ستون‌گزین، row-group به row-group). منطق پاکسازی — نرمال‌سازی متن، نگاشت وضعیت — همان
قواعد `clean_individual.py` را دنبال می‌کند تا دو نسخه‌ی واگرا به‌وجود نیاید.

⚠️ قاعده‌ی پایداری هدف (بند «قاعده‌ی پایداری هدف» `AGENTS.md`، یافته F90): هدف همیشه از
**اجتماع** سه کد عدم‌دریافت ساخته می‌شود. هیچ کد `ReserveStatus`ی به‌تنهایی معتبر نیست.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.config import DATA_RAW
from src.data.text_normalize import normalize_persian_text

RAW_V4_PATH = DATA_RAW / "StudentReserve.parquet"

#: هش ثبت‌شده در `doc/data_manifest.md` (بخش snapshot نسخه‌ی ۴).
EXPECTED_SHA256 = "91508f9e3d787b26be9f6737d24c97259c217b874a89bfdfba7cda3408a2ee4a"

#: نگاشت `mealid` → برچسب انگلیسی. با مقایسه‌ی `value_counts` ستون `Name` و `mealid`
#: روی کل ۸٬۵۶۷٬۲۶۵ ردیف تأیید شد (هر شش زوج دقیقاً منطبق‌اند).
MEAL_BY_ID = {
    1: "lunch",          # ناهار — ۵٬۳۳۲٬۷۹۹
    2: "sahari",         # سحر — ۶۹٬۰۹۳
    3: "dinner",         # شام — ۳٬۰۳۸٬۶۹۰
    4: "iftar",          # افطار — ۷۱۰
    5: "breakfast",      # صبحانه — ۱۲۴٬۲۶۵
    8: "dinner_summer",  # شام (تابستان) — ۱٬۷۰۸
}

#: وعده‌هایی که دامنه‌ی مسئله‌اند (تصمیم فاز ۳، بازتأییدشده در بند ۱۳-۷).
MAIN_MEAL_IDS = (1, 3)

#: نگاشت آماری `ReserveStatus` → عدم‌دریافت. عیناً همان مجموعه‌ی
#: `clean_individual._DONT_RECEIVE_STATUSES`؛ اینجا تکرار نشده، import می‌شود تا
#: واگرایی ممکن نباشد.
from src.data.clean_individual import (  # noqa: E402
    _DONT_RECEIVE_STATUSES as DONT_RECEIVE_STATUSES,
)

_RECEIVE_STATUSES = {"دریافت شده"}

#: ستون‌هایی که برای اغلب تحلیل‌ها کافی‌اند — خواندن همه‌ی ۲۹ ستون حدود دو برابر
#: حافظه می‌گیرد و معمولاً لازم نیست.
CORE_COLUMNS = [
    "PersonId", "Reserveid", "miladi", "DateReserve", "Name", "mealid",
    "FoodName", "Count", "ReserveStatus", "Price",
    "RestaurantName", "resturantId", "GroupName",
    "CollegeName", "FieldName", "DegreeName", "EducationSession", "Gender",
]

#: ستون‌های متنی که آلودگی `_x000D_\n` یا رسم‌الخط عربی دارند.
_TEXT_COLUMNS = [
    "Name", "FoodName", "ReserveStatus", "RestaurantName", "GroupName",
    "CollegeName", "FieldName", "DegreeName", "EducationSession", "Gender",
    "PersonType", "Comment", "Reception",
]

#: ستون‌هایی که **فقط** dictionary-encode می‌شوند و نرمال‌سازی متنی **نمی‌گیرند**.
#: `DateReserve` ~۸۸۰ مقدار یکتا روی ۸.۴ میلیون ردیف دارد و به‌صورت رشته‌ی آزاد
#: ۴۸۵ مگابایت می‌گیرد؛ ولی `normalize_persian_text` روی آن **ممنوع** است چون
#: می‌تواند ارقام را دستکاری کند.
_DICT_ONLY_COLUMNS = ["DateReserve"]


def verify_snapshot(path: Path | str = RAW_V4_PATH, *, strict: bool = True) -> str:
    """هش فایل خام را با مقدار ثبت‌شده در manifest مقایسه می‌کند.

    همان نقشی را دارد که `assert` سلول ۳ نوت‌بوک‌های GPU (بند ۷-۸-۲) دارد: اثبات اینکه
    تحلیل روی همان بایت‌هایی اجرا شده که در `doc/data_manifest.md` ثبت است.
    """
    path = Path(path)
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    digest = h.hexdigest()
    if strict and digest != EXPECTED_SHA256:
        raise ValueError(
            f"data_snapshot_hash mismatch for {path}\n"
            f"  expected: {EXPECTED_SHA256}\n  actual:   {digest}"
        )
    return digest


def load_raw_v4(
    columns: list[str] | None = None,
    *,
    meals: tuple[int, ...] | None = None,
    path: Path | str = RAW_V4_PATH,
) -> pd.DataFrame:
    """خواندن ستون‌گزین snapshot خام v4.

    `meals` اگر داده شود، فیلتر `mealid` **حین خواندن** اعمال می‌شود (نه بعدش) تا
    ردیف‌های بی‌مصرف اصلاً وارد حافظه نشوند.

    ⚠️ ستون‌های متنی با `dictionary` encoding خوانده و به `category` تبدیل می‌شوند.
    این تزئین نیست، شرط اجراست: بیشترین کاردینالیتی متنی ۱٬۷۶۲ (`FoodName`) در برابر
    ۸.۴ میلیون ردیف است، و خواندن به‌صورت رشته‌ی آزاد ~۸ گیگابایت می‌گیرد و روی ماشین
    ۱۶ گیگابایتی پروژه با OOM کشته می‌شود.
    """
    columns = list(columns) if columns is not None else list(CORE_COLUMNS)
    if meals is not None and "mealid" not in columns:
        columns.append("mealid")

    filters = [("mealid", "in", list(meals))] if meals is not None else None
    dict_cols = [c for c in (*_TEXT_COLUMNS, *_DICT_ONLY_COLUMNS) if c in columns]
    # ⚠️ `read_dictionary` ستون‌ها را **حین خواندن** dictionary می‌کند، نه با
    # `table.cast()` پس از آن. تفاوتش حافظه است نه سلیقه: `cast` جدول تازه‌ای
    # می‌سازد در حالی که جدول رشته‌ی آزاد هنوز زنده است ⇒ اوج مصرف دو برابر.
    # روی ۱۴ ستون، ۵٬۳۶۰ مگابایت در برابر **۲٬۰۰۱**.
    table = pq.read_table(path, columns=columns, filters=filters,
                          read_dictionary=dict_cols or None)
    # ⚠️ `self_destruct` بافرهای Arrow را حین تبدیل آزاد می‌کند. بدون آن، در لحظه‌ی
    # `to_pandas` هم جدول Arrow و هم قاب pandas هم‌زمان در حافظه‌اند و اوج مصرف
    # تقریباً **دو برابر** می‌شود. ⚠️ پیامدش این است که `table` پس از فراخوانی
    # دیگر قابل‌استفاده نیست — به همین دلیل بلافاصله برگردانده می‌شود.
    return table.to_pandas(self_destruct=True, split_blocks=True)


def _normalize_categories(s: pd.Series) -> pd.Series:
    """نرمال‌سازی متن فارسی روی *دسته‌ها*ی یک ستون categorical، نه روی ردیف‌هایش.

    ۱٬۷۶۲ فراخوانی به‌جای ۸٬۳۷۱٬۴۸۹ تا. نرمال‌سازی می‌تواند چند نام خام را به یک نام
    برساند (مثلاً `خوابگاه- فاطمیه -خواهران` و `خوابگاه فاطمیه-خواهران` — همان ریشه‌ی
    یافته F91)، پس دسته‌های هم‌نام باید ادغام شوند؛ `rename_categories` این حالت را
    نمی‌پذیرد و باید از راه کدها رفت.
    """
    import numpy as np

    cats = np.asarray(
        [normalize_persian_text(c) for c in s.cat.categories], dtype=object
    )
    uniq, inverse = np.unique(cats, return_inverse=True)
    codes = s.cat.codes.to_numpy()
    new_codes = np.where(codes >= 0, inverse[codes], -1)
    return pd.Series(
        pd.Categorical.from_codes(new_codes, categories=uniq), index=s.index, name=s.name
    )


def normalization_collisions(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """نام‌های خامی که پس از نرمال‌سازی به یک نام می‌رسند.

    خروجی این تابع ورودی مستقیم بازسازی نگاشت‌ها (بند ۱۳-۵) است: هر ردیفش یعنی سامانه‌ی
    مبدأ یک موجودیت را با چند رسم‌الخط نوشته و شمارش‌های خام آن را چندتکه کرده‌اند.
    """
    raw = df[column].astype("string").dropna().unique()
    norm = pd.Series({r: normalize_persian_text(r) for r in raw}, name="normalized")
    g = norm.reset_index().rename(columns={"index": "raw"}).groupby("normalized")["raw"]
    merged = g.agg(list)[g.size() > 1]
    return merged.reset_index().rename(columns={"raw": "raw_variants"})


def _codes_and_categories(s: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """کدها و دسته‌های یک ستون — چه categorical باشد چه نباشد."""
    cat = s if isinstance(s.dtype, pd.CategoricalDtype) else s.astype("category")
    return cat.cat.codes.to_numpy(), np.asarray(cat.cat.categories, dtype=object)


def _from_category_map(codes: np.ndarray, values: np.ndarray) -> pd.Categorical:
    """ستون مشتق را از روی **دسته‌ها** می‌سازد، نه از روی ردیف‌ها.

    همان ترفند `_normalize_categories`: به‌جای ۸.۴ میلیون فراخوانی، ~۸۸۰ تا. و چون
    نگاشت می‌تواند دو دسته را به یکی برساند (`1402/9/1` و `1402/09/01` هر دو
    `1402-09-01`)، ادغام باید از راه کدها انجام شود.
    """
    uniq, inverse = np.unique(values, return_inverse=True)
    new_codes = np.where(codes >= 0, inverse[np.maximum(codes, 0)], -1)
    return pd.Categorical.from_codes(new_codes, categories=uniq)


def clean_v4(df: pd.DataFrame, *, normalize_text: bool = True) -> pd.DataFrame:
    """پاکسازی پایه: نرمال‌سازی متن، ستون‌های مشتق تاریخ/وعده، و هدف `dont_receive`.

    چیزی حذف نمی‌کند — تصمیم‌های دامنه (سلف موقت، رستوران مکمل، وعده‌های فرعی) عمداً
    اینجا نیستند و در بندهای ۱۳-۵ و ۱۳-۷ جداگانه گرفته و مستند می‌شوند.

    ## ⚠️ قرارداد حافظه — چرا این تابع این‌قدر محتاط نوشته شده

    نسخه‌ی نخست این تابع روی ماشین ۱۶ گیگابایتی پروژه با **OOM کشته می‌شد**. سه علت
    داشت و هر سه اینجا رفع شده‌اند:

    - `df.copy()` عمیق، کل قاب را دوباره می‌ساخت ⇒ حالا کپی **کم‌عمق** است.
    - `DateReserve.str.split(expand=True)` سه ستون رشته‌ی ۸.۴ میلیونی می‌ساخت
      (~۱.۵ گیگابایت) ⇒ حالا تجزیه روی **دسته‌ها** انجام می‌شود (~۸۸۰ تا).
    - `date_jalali` و `meal` به‌صورت رشته‌ی آزاد ساخته می‌شدند (۴۹۴ و ۴۵۵ مگابایت
      برای ستون‌هایی با ۸۸۰ و **۲** مقدار یکتا) ⇒ حالا `category`اند.

    اوج حافظه از **۵.۵ گیگابایت** به زیر **۰.۷ گیگابایت** رسید. مقادیر ستون‌ها
    دقیقاً همان‌اند؛ فقط `dtype` سه ستون از `object` به `category` رفته.
    """
    out = df.copy(deep=False)

    if normalize_text:
        for col in _TEXT_COLUMNS:
            if col not in out.columns:
                continue
            if isinstance(out[col].dtype, pd.CategoricalDtype):
                out[col] = _normalize_categories(out[col])
            else:
                out[col] = _normalize_categories(out[col].astype("category"))

    if "miladi" in out.columns:
        out["date_gregorian"] = pd.to_datetime(out["miladi"]).dt.normalize()

    if "DateReserve" in out.columns:
        # «1402/9/8» → «1402-09-08» تا با قرارداد فایل تجمیعی یکی شود (بند ۳-۵ WBS).
        codes, cats = _codes_and_categories(out["DateReserve"])
        parts = [str(c).split("/") for c in cats]
        jalali = np.array([f"{p[0]}-{int(p[1]):02d}-{int(p[2]):02d}" for p in parts],
                          dtype=object)
        out["date_jalali"] = _from_category_map(codes, jalali)
        jy = np.array([int(p[0]) for p in parts], dtype="int16")
        jm = np.array([int(p[1]) for p in parts], dtype="int8")
        safe = np.maximum(codes, 0)
        out["jyear"] = pd.Series(np.where(codes >= 0, jy[safe], -1), index=out.index,
                                 dtype="int16")
        out["jmonth"] = pd.Series(np.where(codes >= 0, jm[safe], -1), index=out.index,
                                  dtype="int8")

    if "mealid" in out.columns:
        ids = out["mealid"].to_numpy()
        uniq_ids, inverse = np.unique(ids, return_inverse=True)
        labels = np.array([MEAL_BY_ID.get(int(i)) for i in uniq_ids], dtype=object)
        known = np.array([lbl is not None for lbl in labels])
        # وعده‌ی ناشناخته ⇒ کد ‎-۱ (NaN)، نه یک دسته‌ی ساختگی
        lut = np.full(len(uniq_ids), -1, dtype="int64")
        kept = np.asarray(labels[known], dtype=object)
        uniq_lbl, lbl_inv = np.unique(kept, return_inverse=True)
        lut[known] = lbl_inv
        out["meal"] = pd.Categorical.from_codes(lut[inverse], categories=uniq_lbl)

    if "ReserveStatus" in out.columns:
        unexpected = set(out["ReserveStatus"].dropna().unique()) - DONT_RECEIVE_STATUSES - _RECEIVE_STATUSES
        if unexpected:
            raise ValueError(f"Unmapped ReserveStatus values (data drift?): {unexpected}")
        out["dont_receive"] = out["ReserveStatus"].isin(DONT_RECEIVE_STATUSES)

    return out


def load_clean_v4(
    columns: list[str] | None = None,
    *,
    meals: tuple[int, ...] | None = MAIN_MEAL_IDS,
    verify: bool = False,
) -> pd.DataFrame:
    """میان‌بر رایج: خواندن + پاکسازی، به‌طور پیش‌فرض فقط ناهار و شام."""
    if verify:
        verify_snapshot()
    return clean_v4(load_raw_v4(columns, meals=meals))


def status_stability_table(df: pd.DataFrame, freq: str = "ME") -> pd.DataFrame:
    """جدول پایداری هدف: سهم هر کد `ReserveStatus` و اجتماعشان، به تفکیک دوره.

    این تابع **آزمون** یافته F90 است، نه یک گزارش توصیفی: اگر سهم تک‌کدها جابه‌جا شود
    ولی ستون `union` پایدار بماند، نگاشت اجتماع معتبر است. هر جهش در `union` یعنی
    تعریف هدف بین دوره‌ها قابل‌مقایسه نیست.
    """
    period = df["date_gregorian"].dt.to_period("M" if freq == "ME" else freq)
    ct = pd.crosstab(period, df["ReserveStatus"], normalize="index") * 100
    ct["union"] = 100 * df.groupby(period)["dont_receive"].mean()
    ct["n"] = df.groupby(period).size()
    return ct.round(2)
