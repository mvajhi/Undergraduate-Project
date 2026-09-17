"""رده‌بندی موجودیت‌های ستون `RestaurantName` در snapshot نسخه‌ی ۴.

ستون `RestaurantName` در داده‌ی تازه **یک‌جنس نیست**. کنار ۷۳ سلف واقعی، سه رده‌ی
دیگر هم با همان ستون ثبت می‌شوند که هیچ‌کدام «سلفی که دانشگاه برایش غذا می‌پزد»
نیستند. اگر این‌ها تفکیک نشوند، هر تجمیعی روی $(d,m,r,f)$ چهار نوع موجودیت متفاوت را
در یک سلول می‌ریزد.

رده‌ها (شمارش روی ناهار و شام، ۸٬۳۷۱٬۴۸۹ ردیف):

| رده | تعداد | سهم ردیف | میانه‌ی $\\rho$ | ماهیت |
|---|---|---|---|---|
| `core` | ۷۳ | ۹۸.۲۲٪ | ۰.۰۷۲ | سلف واقعی — دامنه‌ی مسئله |
| `mokammel` | ۲۵ | ۱.۳۴٪ | ۰.۱۵۴ | رستوران طرف قرارداد بخش خصوصی |
| `guest` | ۲۴ | ۰.۴۴٪ | ۰.۱۷۶ | سهمیه‌ی مهمان (نام خودش می‌گوید «روزانه ۲۰ نفر») |
| `howzeh` | ۱ | ۰.۰۱٪ | ۰.۵۶۶ | دوره‌ی موقت — همتای موردی که در v2 حذف شد |
| `test` | ۱ | ~۰٪ | ۱.۰۰۰ | رکورد آزمایشی سامانه |

تفکیک از روی **الگوی نام** انجام می‌شود، نه فهرست دستی: نام‌ها خودشان رده را اعلام
می‌کنند و الگو در برابر افزوده‌شدن شعبه‌ی تازه پایدار است (بین ۲۰۲۴-۱۱ و ۲۰۲۶-۰۲ تعداد
شعب مکمل از ۳ به ۲۰ رسید؛ فهرست دستی تا امروز سه بار کهنه می‌شد).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: الگوی نام هر رده. ترتیب مهم است — نخستین تطابق برنده است.
CLASS_PATTERNS: list[tuple[str, str]] = [
    ("test", "تست اپ"),
    ("guest", r"^مهمان"),
    ("mokammel", "رستوران مکمل"),
    ("howzeh", "حوزه"),
]

CORE_CLASS = "core"

#: رده‌هایی که هدف مدل از آن‌ها ساخته **نمی‌شود**. تصمیم و دلیلش در
#: `doc/eda/round6/R6a_restaurant_taxonomy.md` و ردیف متناظر `decision_log`.
NON_TARGET_CLASSES = ("test", "guest", "howzeh", "mokammel")


def classify_restaurants(names: pd.Series) -> pd.Series:
    """برچسب رده برای هر مقدار `RestaurantName`."""
    s = names.astype("string")
    conds = [s.str.contains(pat, regex=True, na=False) for _, pat in CLASS_PATTERNS]
    labels = [cls for cls, _ in CLASS_PATTERNS]
    return pd.Series(
        np.select(conds, labels, default=CORE_CLASS), index=names.index, dtype="object"
    )


def add_restaurant_class(df: pd.DataFrame, column: str = "RestaurantName") -> pd.DataFrame:
    out = df.copy()
    out["restaurant_class"] = classify_restaurants(out[column])
    return out


def lifecycle(df: pd.DataFrame, data_end: pd.Timestamp | None = None) -> pd.DataFrame:
    """چرخه‌ی عمر هر سلف: بازه‌ی فعالیت، چگالی، حجم، و نرخ عدم‌دریافت.

    `density` = روزهای فعال ÷ طول بازه. مقدار پایین یعنی سلف در بازه‌اش پیوسته کار
    نکرده — نشانه‌ی یک دوره‌ی موقت، نه یک سلف دائمی کم‌مشتری. این دو را فقط با حجم
    نمی‌شود از هم تشخیص داد و به همین دلیل هر دو ستون لازم‌اند.

    `dead_days` = فاصله‌ی آخرین روز فعال تا پایان داده. مقدار بزرگ یعنی سلف تعطیل شده
    (یا دست‌کم در پنجره‌ی پایانی داده غایب است) و نگه‌داشتنش در آموزش، مدل را روی
    موجودیتی تنظیم می‌کند که دیگر وجود ندارد.
    """
    if data_end is None:
        data_end = df["date_gregorian"].max()

    g = df.groupby("RestaurantName", observed=True).agg(
        rows=("PersonId", "size"),
        people=("PersonId", "nunique"),
        days=("date_gregorian", "nunique"),
        first=("date_gregorian", "min"),
        last=("date_gregorian", "max"),
        rho=("dont_receive", "mean"),
        price_med=("Price", "median"),
        foods=("FoodName", "nunique"),
    )
    g["span"] = (g["last"] - g["first"]).dt.days + 1
    g["density"] = g["days"] / g["span"]
    g["rows_per_day"] = g["rows"] / g["days"]
    g["dead_days"] = (data_end - g["last"]).dt.days
    g["restaurant_class"] = classify_restaurants(g.index.to_series())
    return g.sort_values("rows", ascending=False)


def flag_suspect_restaurants(
    life: pd.DataFrame,
    *,
    min_rows: int = 1_000,
    min_days: int = 30,
    min_density: float = 0.25,
    dead_after: int = 120,
) -> pd.DataFrame:
    """سلف‌هایی که پیش از ورود به آموزش باید تعیین تکلیف شوند، با دلیل هر پرچم.

    آستانه‌ها عمداً شل‌اند: خروجی این تابع **فهرست بررسی** است، نه فهرست حذف. حذف
    خودکار روی آستانه، همان اشتباهی است که یک سلف کوچکِ واقعی را بی‌صدا از دامنه بیرون
    می‌اندازد.
    """
    f = life.copy()
    reasons = {
        "low_volume": f["rows"] < min_rows,
        "few_days": f["days"] < min_days,
        "sparse": f["density"] < min_density,
        "discontinued": f["dead_days"] > dead_after,
        "non_core_class": f["restaurant_class"] != CORE_CLASS,
    }
    for name, mask in reasons.items():
        f[name] = mask
    f["flags"] = pd.Series(
        [
            ", ".join(n for n, m in reasons.items() if m.loc[idx])
            for idx in f.index
        ],
        index=f.index,
    )
    return f[f["flags"] != ""].sort_values(["restaurant_class", "rows"])


def mokammel_substitution_did(
    df: pd.DataFrame,
    *,
    min_side_days: int = 60,
    seed: int = 42,
    n_boot: int = 2000,
) -> dict:
    """آیا باز شدن رستوران مکمل، تقاضای سلف اصلی را می‌دزدد؟ — تفاضل‌در‌تفاضل.

    مقایسه‌ی ساده‌ی «قبل و بعد» برای این پرسش کافی نیست: مصرف سلف در کل داده روند
    نزولی دارد، پس هر گروهی بعد از هر تاریخی کمتر غذا می‌گیرد. گروه کنترل
    (افرادی که هرگز از مکمل استفاده نکرده‌اند) یک $t_0$ ساختگی از همان توزیع
    $t_0$ پذیرندگان می‌گیرد تا هر دو گروه روی **پنجره‌های تقویمی یکسان** سنجیده شوند
    و روند مشترک از اثر جدا شود.

    ⚠️ قید تفسیری: پذیرندگان پایه‌ی مصرف بسیار بالاتری دارند (~۲۲.۸ در برابر ~۱۲.۷
    وعده در ۳۰ روز)، پس فرض «روند موازی» کاملاً بی‌خطر نیست و بخشی از اثر می‌تواند
    بازگشت به میانگین باشد. عدد را به‌عنوان کران بالای اندازه‌ی جانشینی بخوانید، نه
    برآورد نقطه‌ای علّی.
    """
    rng = np.random.default_rng(seed)
    cls = classify_restaurants(df["RestaurantName"])
    core = df[cls == CORE_CLASS]
    mk = df[cls == "mokammel"]

    first = mk.groupby("PersonId")["date_gregorian"].min().rename("t0")
    adopters = set(first.index)
    controls = np.array(sorted(set(core["PersonId"].unique()) - adopters))
    placebo = pd.Series(rng.choice(first.values, size=len(controls)), index=controls, name="t0")
    t0 = pd.concat([first, placebo])
    t0.index.name = "PersonId"

    c = core[["PersonId", "date_gregorian"]].merge(t0, on="PersonId", how="inner")
    cnt = c.assign(after=c.date_gregorian >= c.t0).groupby(["PersonId", "after"]).size().unstack(fill_value=0)
    span = core.groupby("PersonId")["date_gregorian"].agg(["min", "max"]).join(t0, how="inner")
    span["d_before"] = (span.t0 - span["min"]).dt.days
    span["d_after"] = (span["max"] - span.t0).dt.days

    d = cnt.join(span[["d_before", "d_after"]], how="inner")
    d = d[(d.d_before >= min_side_days) & (d.d_after >= min_side_days)]
    d["before"] = d[False] / d.d_before * 30
    d["after"] = d[True] / d.d_after * 30
    d["delta"] = d["after"] - d["before"]
    d["treated"] = d.index.isin(adopters)

    tr, ct = d.loc[d.treated, "delta"].to_numpy(), d.loc[~d.treated, "delta"].to_numpy()
    did = tr.mean() - ct.mean()
    boot = np.array([
        rng.choice(tr, len(tr), replace=True).mean() - rng.choice(ct, len(ct), replace=True).mean()
        for _ in range(n_boot)
    ])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    base = d.loc[d.treated, "before"].mean()
    return {
        "n_treated": int(d.treated.sum()),
        "n_control": int((~d.treated).sum()),
        "treated_before": float(base),
        "treated_after": float(d.loc[d.treated, "after"].mean()),
        "control_before": float(d.loc[~d.treated, "before"].mean()),
        "control_after": float(d.loc[~d.treated, "after"].mean()),
        "did": float(did),
        "did_ci": (float(lo), float(hi)),
        "did_relative": float(did / base),
    }
