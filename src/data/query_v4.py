"""لایه‌ی پرس‌وجوی تجمیعی روی snapshot خام v4 — بدون بارگذاری در حافظه.

## چرا این ماژول وجود دارد

`load_clean_v4()` کل ۸٬۳۷۱٬۴۸۹ ردیف را وارد pandas می‌کند. حتی پس از بهینه‌سازی
`clean_v4` (کپی کم‌عمق، تجزیه روی دسته‌ها، ستون‌های categorical) اوج مصرف **~۳.۱
گیگابایت** است — چون `pq.read_table` و `to_pandas` هر دو باید کل ستون‌ها را
مادی کنند. روی ماشین ۱۶ گیگابایتی پروژه، هم‌زمانی آن با هر کار دیگری OOM می‌دهد؛
تست‌های `src/data/tests_v4.py` دقیقاً به همین دلیل کشته می‌شدند.

ولی **هیچ‌کدام از آن تست‌ها به ردیف‌ها نیاز ندارند** — همه‌شان `GROUP BY` یا
`DISTINCT`اند. DuckDB همان پاسخ را با projection/predicate pushdown و اجرای
جریانی می‌دهد:

| پرسش | pandas | DuckDB |
|---|---|---|
| سهم هر کد `ReserveStatus` | ~۳٬۱۰۰ مگابایت | **~۱۸۰ مگابایت** |

## قاعده‌ی استفاده

- **پرسش تجمیعی** (سهم، شمارش، یکتا، بازه) → این ماژول.
- **تحلیل سطح ردیف** (توالی فردی، دوره‌ی غیبت، ساخت شبکه) → `load_clean_v4`.

⚠️ این ماژول نرمال‌سازی متن فارسی **انجام نمی‌دهد**. برای ستون‌هایی که مقایسه‌ی
متنی لازم دارند، مقادیر **یکتا** را بگیرید و نرمال‌سازی را در پایتون روی همان
~۱٬۷۶۲ مقدار انجام دهید — نه روی ۸.۴ میلیون ردیف. `distinct_values()` همین کار
را می‌کند.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.load_v4 import (
    DONT_RECEIVE_STATUSES,
    MAIN_MEAL_IDS,
    RAW_V4_PATH,
    _RECEIVE_STATUSES,
)

#: همان مرز سال تحصیلی که `tests_v4.academic_year` استفاده می‌کند (از شهریور).
_ACADEMIC_YEAR_SQL = """
    CASE WHEN CAST(split_part(DateReserve, '/', 2) AS INT) >= 6
         THEN CAST(split_part(DateReserve, '/', 1) AS INT)
         ELSE CAST(split_part(DateReserve, '/', 1) AS INT) - 1 END
"""


def _quoted(values) -> str:
    return ", ".join("'" + str(v).replace("'", "''") + "'" for v in values)


#: ⚠️ **وزن، یک انتخاب است نه جزئیات پیاده‌سازی.** `rows` هر رکورد رزرو را یک واحد
#: می‌شمارد؛ `count` ستون `Count` را جمع می‌زند (یعنی پرس). از F94 به بعد این دو
#: یکی نیستند — ۲۳٬۹۴۲ ردیف `Count=۲` دارند. تست‌های `tests_v4` تاریخاً **سطری**اند
#: و شبکه‌ی L1/L2 **پرسی**. هر عددی که نقل می‌شود باید وزنش معلوم باشد.
_WEIGHT_SQL = {"rows": "1", "count": "Count"}


def _w(weight: str) -> str:
    try:
        return _WEIGHT_SQL[weight]
    except KeyError:
        raise ValueError(f"weight باید 'rows' یا 'count' باشد، نه {weight!r}") from None


def connect(path: Path | str = RAW_V4_PATH, *, meals: tuple[int, ...] | None = MAIN_MEAL_IDS):
    """اتصال DuckDB با یک view به نام `reserve` روی فایل خام.

    `meals` پیش‌فرض ناهار و شام است — همان پیش‌فرض `load_clean_v4`، تا دو مسیر
    پاسخ‌های ناسازگار ندهند.
    """
    import duckdb

    con = duckdb.connect()
    where = f" WHERE mealid IN ({', '.join(str(m) for m in meals)})" if meals else ""
    con.execute(
        f"CREATE VIEW reserve AS SELECT * FROM read_parquet('{Path(path).as_posix()}'){where}"
    )
    return con


def assert_known_statuses(con=None) -> set[str]:
    """همان دروازه‌ی رانشِ `clean_v4`، ولی بدون بارگذاری هیچ ردیفی.

    اگر کد تازه‌ای در `ReserveStatus` ظاهر شود، تعریف هدف بی‌صدا عوض شده است.
    """
    own = con is None
    con = con or connect()
    try:
        vals = {r[0] for r in con.execute(
            "SELECT DISTINCT ReserveStatus FROM reserve WHERE ReserveStatus IS NOT NULL"
        ).fetchall()}
    finally:
        if own:
            con.close()
    unexpected = vals - DONT_RECEIVE_STATUSES - _RECEIVE_STATUSES
    if unexpected:
        raise ValueError(f"Unmapped ReserveStatus values (data drift?): {unexpected}")
    return vals


def status_shares(con=None, *, by: str = "month", weight: str = "rows") -> pd.DataFrame:
    """سهم درصدی هر کد `ReserveStatus` و ستون `union`، به تفکیک دوره.

    `by`: `month` (سال-ماه شمسی) · `jyear` · `academic_year`.
    `weight`: `rows` (پیش‌فرض، هم‌ارز `status_stability_table`) یا `count` (پرس).
    """
    own = con is None
    con = con or connect()
    # ⚠️ `greg_month` عمداً وجود دارد: `status_stability_table` تاریخاً با ماه
    # **میلادی** دوره‌بندی می‌کند، و تستی که به آن تکیه دارد نباید بی‌صدا مبنایش
    # عوض شود. `month` (شمسی) انتخاب درست‌تری برای این پروژه است (F128) ولی
    # جایگزین‌کردنش یک تغییر است، نه یک اصلاح.
    period = {
        "month": "CAST(split_part(DateReserve,'/',1) AS INT) || '-' || "
                 "lpad(split_part(DateReserve,'/',2), 2, '0')",
        "greg_month": "strftime(miladi, '%Y-%m')",
        "jyear": "CAST(split_part(DateReserve,'/',1) AS INT)",
        "academic_year": f"({_ACADEMIC_YEAR_SQL}) || '-' || ({_ACADEMIC_YEAR_SQL} + 1)",
    }[by]
    try:
        df = con.execute(f"""
            SELECT {period} AS period, ReserveStatus AS status, SUM({_w(weight)}) AS n
            FROM reserve GROUP BY 1, 2
        """).fetch_df()
    finally:
        if own:
            con.close()
    tab = df.pivot_table(index="period", columns="status", values="n",
                         aggfunc="sum").fillna(0.0)
    share = tab.div(tab.sum(axis=1), axis=0) * 100
    present = [c for c in tab.columns if c in DONT_RECEIVE_STATUSES]
    share["union"] = share[present].sum(axis=1)
    share["n"] = tab.sum(axis=1)
    return share.sort_index()


def union_rate(con=None, *, by: str | None = None,
               weight: str = "rows") -> pd.DataFrame | float:
    """نرخ اجتماع سه کد — کلی (`by=None`) یا به تفکیک دوره.

    `weight="rows"` پیش‌فرض است تا با تست‌های تاریخی `tests_v4` هم‌خوان بماند؛
    `weight="count"` عددِ هم‌ارز شبکه‌ی L1/L2 را می‌دهد.
    """
    own = con is None
    con = con or connect()
    codes = _quoted(DONT_RECEIVE_STATUSES)
    try:
        if by is None:
            v = con.execute(f"""
                SELECT SUM(CASE WHEN ReserveStatus IN ({codes}) THEN {_w(weight)} ELSE 0 END)
                       / SUM({_w(weight)}) FROM reserve
            """).fetchone()[0]
            return float(v)
        period = {
            "jyear": "CAST(split_part(DateReserve,'/',1) AS INT)",
            "academic_year": f"({_ACADEMIC_YEAR_SQL}) || '-' || ({_ACADEMIC_YEAR_SQL} + 1)",
        }[by]
        return con.execute(f"""
            SELECT {period} AS period,
                   SUM(CASE WHEN ReserveStatus IN ({codes}) THEN {_w(weight)} ELSE 0 END)
                   / SUM({_w(weight)}) AS rate,
                   SUM({_w(weight)}) AS n
            FROM reserve GROUP BY 1 ORDER BY 1
        """).fetch_df().set_index("period")
    finally:
        if own:
            con.close()


def distinct_values(column: str, con=None, *, with_counts: bool = False) -> pd.DataFrame:
    """مقادیر یکتای یک ستون — ورودی هر بررسی پوشش نگاشت.

    نرمال‌سازی متن **اینجا انجام نمی‌شود**؛ روی همین خروجی کوچک در پایتون انجامش دهید.
    """
    own = con is None
    con = con or connect()
    try:
        if with_counts:
            return con.execute(
                f'SELECT "{column}" AS value, SUM(Count) AS n FROM reserve '
                f'WHERE "{column}" IS NOT NULL GROUP BY 1 ORDER BY 2 DESC'
            ).fetch_df()
        return con.execute(
            f'SELECT DISTINCT "{column}" AS value FROM reserve WHERE "{column}" IS NOT NULL'
        ).fetch_df()
    finally:
        if own:
            con.close()


def count_distribution(con=None) -> pd.DataFrame:
    """توزیع ستون `Count` — پایه‌ی F94 («`Count` دیگر باینری نیست»)."""
    own = con is None
    con = con or connect()
    try:
        return con.execute(
            "SELECT Count AS value, count(*) AS rows FROM reserve GROUP BY 1 ORDER BY 1"
        ).fetch_df()
    finally:
        if own:
            con.close()


def row_count(con=None) -> int:
    own = con is None
    con = con or connect()
    try:
        return int(con.execute("SELECT count(*) FROM reserve").fetchone()[0])
    finally:
        if own:
            con.close()
