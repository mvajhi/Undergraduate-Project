r"""قاعده‌ی برش اطلاعاتی — بنیادی‌ترین ماژول فاز ۵.

**ایده‌ی مرکزی.** قاعده‌ی برش پروژه به تفکیک وعده تعریف شده (ناهار روز $d$ ← ساعت ۱۵
روز $d-1$؛ شام روز $d$ ← ساعت ۲۳ روز $d-1$) و همین باعث شده رایج‌ترین منبع باگ نشتی
باشد. اما اگر به هر (روز، وعده) یک **شماره‌ی سراسری** بدهیم — ناهار روز $t$ → $2t$،
شام روز $t$ → $2t+1$ — کل قاعده به **یک نامساوی واحد** فرو می‌ریزد:

```
هدف ناهار روز d  (seq=2d)    آخرین در دسترس: ناهار روز d−1 (seq=2d−2) = seq−2
هدف شام  روز d  (seq=2d+1)   آخرین در دسترس: شام  روز d−1 (seq=2d−1) = seq−2
```

$$\boxed{\text{مشاهده مجاز است} \iff \text{meal\_seq}_{\text{obs}} \le \text{meal\_seq}_{\text{target}} - 2}$$

چون در هر دو حالت عدد ۲ درمی‌آید، همه‌ی فیچرهای تاریخی (سطح سلول و سطح فرد) با یک
تابع مشترک ساخته می‌شوند و با **یک** تست خودکار ممیزی می‌شوند. جدول «آخرین وعده‌ی در
دسترس» بند ۴-۲ سند مسئله دقیقاً همین است.

⚠️ **نکته‌ای که این فرمول‌بندی آشکار کرد:** تحلیل اکتشافی دور ۳ فیچر تاریخچه‌ی فرد را
با `groupby(PersonId).cumsum().shift()` ساخته بود — یعنی «همه‌ی رزروهای قبلی». برای
*شام* روز $d$، این تعریف **ناهار همان روز** را هم شامل می‌شود، در حالی که در ساعت ۲۳
روز $d-1$ هنوز سرو نشده. اندازه‌ی این خوش‌بینی در `doc/leakage_audit.md` کمّی شده است.
"""

import numpy as np
import pandas as pd

MEAL_ORDER = {"lunch": 0, "dinner": 1}
N_MEALS = 2
#: فاصله‌ی مجاز بین وعده‌ی هدف و آخرین وعده‌ی قابل‌استفاده (بر حسب شماره‌ی سراسری وعده)
CUTOFF_LAG = 2


def meal_seq(dates: pd.Series, meals: pd.Series, origin: pd.Timestamp | None = None) -> pd.Series:
    """شماره‌ی سراسری وعده: ``2 * (روز نسبت به مبدأ) + رتبه‌ی وعده``.

    ``origin`` اگر داده نشود، کمینه‌ی ``dates`` است. نتیجه یک عدد صحیح یکنواخت صعودی
    در زمان است که ناهار هر روز را قبل از شام همان روز قرار می‌دهد.
    """
    d = pd.to_datetime(dates)
    if origin is None:
        origin = d.min()
    day_idx = (d - origin).dt.days.astype("int64")
    rank = meals.astype(str).map(MEAL_ORDER)
    if rank.isna().any():
        bad = sorted(set(meals.astype(str)[rank.isna()]))
        raise ValueError(f"وعده‌ی ناشناخته (فقط lunch/dinner مجاز است): {bad}")
    return day_idx * N_MEALS + rank.astype("int64")


def last_available_seq(target_seq: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    """بزرگ‌ترین ``meal_seq``ی که در لحظه‌ی برشِ وعده‌ی هدف واقعاً سرو شده است."""
    return target_seq - CUTOFF_LAG


def is_available(obs_seq, target_seq) -> np.ndarray:
    """آیا مشاهده‌ای با ``obs_seq`` در لحظه‌ی برشِ ``target_seq`` در دسترس است؟"""
    return np.asarray(obs_seq) <= np.asarray(target_seq) - CUTOFF_LAG


def expanding_stat_at_cutoff(
    df: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
    seq_col: str = "meal_seq",
    out_prefix: str = "hist",
) -> pd.DataFrame:
    """میانگین/تعداد/جمعِ انبساطیِ ``value_col`` در هر گروه، **فقط تا لحظه‌ی برش**.

    برخلاف ``groupby().shift().expanding()`` که «همه‌ی ردیف‌های قبلی» را می‌گیرد، اینجا
    مرز، مقدار ``seq_col`` است: فقط ردیف‌هایی شمرده می‌شوند که
    ``seq_obs <= seq_target - CUTOFF_LAG``. پس چند رزرو از یک وعده (یا وعده‌ی
    سرو‌نشده‌ی همان روز) هرگز وارد تاریخچه نمی‌شوند.

    روش: در هر گروه، ردیف‌ها بر اساس ``seq`` مرتب می‌شوند؛ جمع تجمعی روی مقادیر یکتا‌ی
    seq ساخته می‌شود؛ سپس برای هر ردیف با ``searchsorted`` مرزِ ``seq-2`` پیدا می‌شود.
    خروجی: ستون‌های ``{prefix}_sum``, ``{prefix}_count``, ``{prefix}_mean``.
    """
    work = df[group_cols + [value_col, seq_col]].copy()
    work["_row"] = np.arange(len(work))
    out_sum = np.full(len(work), np.nan)
    out_cnt = np.zeros(len(work), dtype="int64")

    for _, g in work.groupby(group_cols, observed=True, sort=False):
        g = g.sort_values(seq_col, kind="stable")
        seqs = g[seq_col].to_numpy()
        vals = g[value_col].to_numpy(dtype=float)
        csum = np.concatenate([[0.0], np.cumsum(vals)])
        ccnt = np.arange(len(seqs) + 1)
        # تعداد مشاهداتی که seq آن‌ها ≤ seq_target − CUTOFF_LAG است
        k = np.searchsorted(seqs, seqs - CUTOFF_LAG, side="right")
        out_sum[g["_row"].to_numpy()] = csum[k]
        out_cnt[g["_row"].to_numpy()] = ccnt[k]

    res = pd.DataFrame(index=df.index)
    res[f"{out_prefix}_sum"] = out_sum
    res[f"{out_prefix}_count"] = out_cnt
    with np.errstate(invalid="ignore", divide="ignore"):
        res[f"{out_prefix}_mean"] = np.where(out_cnt > 0, out_sum / out_cnt, np.nan)
    return res


def usage_rate(count, first_seen_day, current_day, min_days: float = 7.0):
    """شمارنده‌ی تجمعی را به **نرخ استفاده در واحد زمان** تبدیل می‌کند.

    **مسئله‌ای که این تابع حل می‌کند (کشف‌شده در ممیزی فاز ۵).** هر شمارنده‌ی «تعداد
    مشاهدات قبلی» یکنواخت با زمان رشد می‌کند و عملاً یک **شاخص تقویم** است
    (Spearman با روزِ تقویم: ۰.۹۵ برای سلول، ۰.۷۶ برای فرد). در یک تقسیم زمانی،
    دامنه‌ی این فیچر در مجموعه‌ی آزمون بیرون از دامنه‌ی آموزش می‌افتد؛ مدل درختی
    برون‌یابی نمی‌کند و همه‌ی آن ردیف‌ها را در آخرین سطلِ آموزش می‌گذارد — سطلی که
    متناظر با انتهای دوره‌ی آموزش است، نه با «کاربر پرسابقه». در این پروژه همین یک
    فیچر $R^2$ خارج‌نمونه را از $+0.08$ به $-1.09$ برد.

    ⚠️ **تبدیل اشباع‌شونده ($n/(n+k)$) این را حل نمی‌کند.** چون یکنواخت است و مدل‌های
    درختی نسبت به تبدیل یکنواخت **ناوردا**ند، نتیجه بیت‌به‌بیت یکسان می‌ماند. (این
    مسیر آزموده و رد شد.) راه‌حل باید ترتیب را عوض کند، نه مقیاس را.

    **نرخ** این کار را می‌کند: «چند رزرو در هفته» بین افرادِ هم‌زمان تفاوت دارد ولی با
    گذشت زمان به‌طور سیستماتیک رشد نمی‌کند، پس در آموزش و آزمون هم‌دامنه است.
    """
    days = pd.Series(current_day).astype(float) - pd.Series(first_seen_day).astype(float)
    days = days.clip(lower=min_days)
    return pd.Series(count).astype(float).fillna(0).to_numpy() / days.to_numpy() * 7.0


def shrunk_rate(sum_col: pd.Series, count_col: pd.Series, alpha: float, beta: float) -> pd.Series:
    """کوچک‌سازی بیزی تجربی: ``(k + α) / (n + α + β)``.

    α و β از برازش Beta با تصحیح بیش‌پراکندگی در بند ۴.۸ می‌آیند (F8.3). برای گروه‌های
    کم‌نمونه مقدار را به سمت میانگین کل می‌کشد و از نرخ‌های ۰٪/۱۰۰٪ کاذب جلوگیری می‌کند.
    """
    return (sum_col.fillna(0) + alpha) / (count_col.fillna(0) + alpha + beta)


def same_meal_lag(
    df: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
    lags: list[int],
    date_col: str = "date_gregorian",
    gap_buffer_days: int = 6,
) -> pd.DataFrame:
    """lag در **توالی هم‌وعده**: «همین وعده در همین سلف، k روز پیش».

    چون گروه‌بندی روی وعده انجام می‌شود، هر lag≥۱ خودبه‌خود قاعده‌ی برش را رعایت می‌کند
    (وعده‌ی هم‌نوعِ روز قبل همیشه پیش از لحظه‌ی برش سرو شده است).

    ⚠️ **آستانه‌ی فاصله باید به‌ازای هر lag متفاوت باشد، نه یک عدد سراسری.** یک آستانه‌ی
    ثابت (مثلاً ۳۵ برای همه‌ی lagها) باعث می‌شود ``lag_1`` بتواند از شکاف ۲۹روزه‌ی رمضان
    (F38) عبور کند و در عمل ``lag_34`` را به اسم ``lag_1`` برچسب بزند — دقیقاً همان تله‌ای
    که این پارامتر قرار بود جلویش را بگیرد. اینجا آستانه‌ی هر ``k`` برابر
    ``k + gap_buffer_days`` است: کمی بیش از خودِ ``k`` تا چند روز سرونشده‌ی پراکنده
    (تعطیلی معمولی) مجاز باشد، ولی نه یک شکاف ساختاری کامل.
    """
    work = df.sort_values(group_cols + [date_col], kind="stable")
    out = pd.DataFrame(index=work.index)
    grp = work.groupby(group_cols, observed=True, sort=False)
    for k in lags:
        col = f"{value_col}_lag{k}"
        out[col] = grp[value_col].shift(k)
        gap = (pd.to_datetime(work[date_col]) - grp[date_col].shift(k)).dt.days
        out.loc[gap > k + gap_buffer_days, col] = np.nan
    return out.reindex(df.index)
