"""ماتریس پروفایل سطح فرد روی پنجره‌ی v4 — زیرساخت دور ۹ (بند ۹-۳).

## چرا یک کش تازه به‌جای گسترش کش موجود

`data/interim/person_level_v4.parquet` (بند ۱۴-۶) پنج ستونی را که دور ۹ لازم دارد
ندارد: `EducationSession`، `Comment`، `ReceptionType`، `Price`، `FieldName`.
گسترش همان فایل وسوسه‌انگیز است ولی هر اجراکننده‌ی `r7*`ی که ستون‌ها را کامل
می‌خواند خروجی متفاوتی می‌گرفت — تغییری بی‌صدا در تحلیلی که قبلاً بسته شده.
پس فایل تازه‌ای ساخته می‌شود و قدیمی دست‌نخورده می‌ماند.

## ⚠️ قرارداد حافظه

هر پنج ستون افزوده `category` می‌شوند. `FieldName` به‌تنهایی ۸۵۹ مقدار یکتا روی
۸.۲ میلیون ردیف دارد و به‌صورت `object` نزدیک نیم گیگابایت می‌گیرد. ماتریس پروفایل
**بلوک‌به‌بلوک** ساخته می‌شود: هر بلوک فقط ستون‌های خودش را می‌خواند، تجمیع می‌کند،
و قاب ردیفی را آزاد می‌کند.

## قواعد دور ۹ که در همین ماژول اجرا می‌شوند

- **قاعده‌ی پ (انقباض اجباری):** `rho_eb` از پیشین بتا-دوجمله‌ای برآوردشده روی همین
  پنجره می‌آید. `rho_raw` ساخته می‌شود ولی مصرفش فقط گزارشی است.
- **قاعده‌ی ث (سانسور صریح):** `left_censored` و `right_censored` ستون‌های اجباری‌اند.
- **قاعده‌ی خ (جمعیت‌شناسی بیرون از خوشه‌بندی):** ستون‌های بلوک ب۱۰ با پیشوند
  `demo_` می‌آیند تا انتخاب ماتریس خوشه‌بندی خطاپذیر نباشد.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import DATA_EXTERNAL, DATA_INTERIM

PERSON_ROWS_V4 = DATA_INTERIM / "person_rows_v4.parquet"
PERSON_PROFILE_V4 = DATA_INTERIM / "person_profile_v4.parquet"

#: ستون‌های خام لازم برای کش ردیفی دور ۹.
_RAW_KEEP = [
    "PersonId", "miladi", "DateReserve", "mealid", "ReserveStatus", "Count", "Price",
    "RestaurantName", "FoodName",
    "Gender", "CollegeName", "DegreeName", "GroupName", "FieldName", "EducationSession",
    "Comment", "ReceptionType",
]

#: مقدار پیش‌فرض `Comment` — یعنی «رأی نداد»، نه «رأی خنثی داد».
NO_VOTE = "بدون نظر"

#: `ReceptionType=2` ⇔ «بارکد فراموشی» (دیکشنری داده‌ی فردی، بند ۲.۱.۴).
FORGOT_CARD_TYPES = (2, 3)


def _map_categories(s: pd.Series, predicate) -> np.ndarray:
    """یک محمول بولی را روی **دسته‌های** یک ستون رده‌ای اجرا و به ردیف‌ها پخش می‌کند.

    ستون `GroupName` ‏۷۱ مقدار یکتا روی ۸.۲ میلیون ردیف دارد. اجرای `str.contains`
    روی ردیف‌ها یعنی ۸.۲ میلیون بار regex؛ روی دسته‌ها یعنی ۷۱ بار. کد ‎-۱
    (مقدار گمشده) مقدار `False` می‌گیرد.
    """
    if not isinstance(s.dtype, pd.CategoricalDtype):
        s = s.astype("category")
    flags = np.asarray(predicate(s.cat.categories.to_series()), dtype=bool)
    codes = s.cat.codes.to_numpy()
    return np.where(codes >= 0, flags[np.maximum(codes, 0)], False)


# ---------------------------------------------------------------------------
# ۹-۳-۱ کش ردیفی نسخه‌ی ۲
# ---------------------------------------------------------------------------

def build_person_rows_v4(force: bool = False,
                         columns: list[str] | None = None) -> pd.DataFrame:
    """کش ردیفی سطح فرد با پنج ستون افزوده، محدود به رده‌ی `core` و ناهار/شام.

    خروجی همان دامنه‌ی `build_person_level_v4` را دارد؛ دروازه‌ی بند ۹-۳-۱ همین را
    آزمون می‌کند (شمار ردیف و `PersonId` یکتا باید یکی باشند).
    """
    if PERSON_ROWS_V4.exists() and not force:
        return pd.read_parquet(PERSON_ROWS_V4, columns=columns)

    from src.data.load_v4 import load_clean_v4
    from src.eda_lib.restaurant_taxonomy import CORE_CLASS, add_restaurant_class

    df = load_clean_v4(columns=_RAW_KEEP)
    df = add_restaurant_class(df)
    df = df[df.restaurant_class == CORE_CLASS].copy(deep=False)

    # شاخص سراسری فرصتِ سرویس — همان تعریف `individual_v4` تا دو ماژول واگرا نشوند.
    slots = (df[["date_gregorian", "meal"]].drop_duplicates()
               .sort_values(["date_gregorian", "meal"])
               .reset_index(drop=True))
    slots["meal_seq"] = np.arange(len(slots), dtype="int32")
    df = df.merge(slots, on=["date_gregorian", "meal"], how="left")

    # روز هفته‌ی شمسی (۰=شنبه) از تقویم پروژه، نه بازمحاسبه‌ی محلی.
    cal = pd.read_csv(DATA_EXTERNAL / "calendar_tehran_v4.csv",
                      usecols=["date_gregorian", "day_of_week"],
                      parse_dates=["date_gregorian"])
    df = df.merge(cal, on="date_gregorian", how="left")
    df["day_of_week"] = df["day_of_week"].astype("int8")

    # ⚠️ هر سه پرچم روی **دسته‌ها** ساخته می‌شوند، نه ردیف‌ها (قاعده‌ی ۱ بخش حافظه).
    # نسخه‌ی نخست همین سه خط با `.astype(str)` نوشته شده بود و اوج حافظه را به
    # ۶.۷ گیگابایت می‌برد: ۸.۲ میلیون شیء رشته‌ای پایتون برای ستونی با ۷۱ مقدار یکتا.
    df["is_dorm_resident"] = _map_categories(
        df.GroupName, lambda c: c.str.contains("خوابگاه", na=False)
        | c.str.contains("متاهلین", na=False))
    df["voted"] = _map_categories(df.Comment, lambda c: c != NO_VOTE)
    df["forgot_card"] = df.ReceptionType.isin(FORGOT_CARD_TYPES).to_numpy()

    keep = ["PersonId", "date_gregorian", "meal", "meal_seq", "day_of_week",
            "dont_receive", "Count", "Price", "RestaurantName", "FoodName",
            "Comment", "voted", "forgot_card", "is_dorm_resident",
            "Gender", "CollegeName", "DegreeName", "GroupName", "FieldName",
            "EducationSession", "jyear", "jmonth"]
    out = df[[c for c in keep if c in df.columns]].copy(deep=False)
    out["dont_receive"] = out.dont_receive.astype(bool)
    out["Count"] = out.Count.astype("int8")
    out["Price"] = out.Price.astype("float32")
    PERSON_ROWS_V4.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(PERSON_ROWS_V4, index=False)
    return out[columns] if columns else out


# ---------------------------------------------------------------------------
# ۹-۳-۳ پیشین بتا-دوجمله‌ای و انقباض تجربی-بیز
# ---------------------------------------------------------------------------

def fit_beta_binomial(k: np.ndarray, n: np.ndarray) -> tuple[float, float]:
    """بیشینه‌ی درست‌نمایی حاشیه‌ای پیشین $\\text{Beta}(\\alpha,\\beta)$.

    ضریب دوجمله‌ای در تابع هدف نمی‌آید چون به پارامترها وابسته نیست. بهینه‌سازی
    روی $\\log\\alpha, \\log\\beta$ انجام می‌شود تا قید مثبت‌بودن خودکار برقرار باشد.
    """
    from scipy.optimize import minimize
    from scipy.special import betaln

    k = np.asarray(k, dtype="float64")
    n = np.asarray(n, dtype="float64")

    def nll(theta):
        a, b = np.exp(theta)
        return -np.sum(betaln(a + k, b + n - k) - betaln(a, b))

    m = float(np.sum(k) / np.sum(n))
    res = minimize(nll, x0=np.log([max(m * 10, 1e-3), max((1 - m) * 10, 1e-3)]),
                   method="Nelder-Mead",
                   options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 5000})
    a, b = np.exp(res.x)
    return float(a), float(b)


def eb_shrink(k: np.ndarray, n: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """میانگین پسین $\\frac{k+\\alpha}{n+\\alpha+\\beta}$ — نرخ منقبض‌شده‌ی فرد."""
    return (np.asarray(k) + alpha) / (np.asarray(n) + alpha + beta)


# ---------------------------------------------------------------------------
# کمکی‌های آماری برداری (روی گروه، بدون حلقه‌ی پایتون)
# ---------------------------------------------------------------------------

def _group_slope(df: pd.DataFrame, key: str, x: str, y: str) -> pd.Series:
    """شیب رگرسیون $y \\sim x$ داخل هر گروه، از مجموع‌ها.

    حلقه‌ی پایتون روی ۴۶ هزار گروه چند دقیقه می‌گیرد؛ این نسخه چند ثانیه.
    گروهی که واریانس $x$ ندارد `NaN` می‌گیرد، نه صفر — صفر یعنی «شیب افقی» و
    این دو یکی نیستند.
    """
    xv = df[x].to_numpy(dtype="float64")
    yv = df[y].to_numpy(dtype="float64")
    tmp = pd.DataFrame({key: df[key].to_numpy(), "_x": xv, "_y": yv,
                        "_xx": xv * xv, "_xy": xv * yv})
    g = tmp.groupby(key, observed=True)
    n = g["_x"].size()
    sx, sy, sxx, sxy = g["_x"].sum(), g["_y"].sum(), g["_xx"].sum(), g["_xy"].sum()
    denom = sxx - sx * sx / n
    return ((sxy - sx * sy / n) / denom.where(denom > 0)).rename("slope")


def _shannon(shares: np.ndarray) -> float:
    p = shares[shares > 0]
    return float(-(p * np.log(p)).sum())


def _entropy_by_group(df: pd.DataFrame, key: str, cat: str) -> pd.Series:
    """آنتروپی شانون توزیع `cat` داخل هر گروه `key`."""
    cnt = df.groupby([key, cat], observed=True).size().rename("n").reset_index()
    tot = cnt.groupby(key, observed=True)["n"].transform("sum")
    cnt["p"] = cnt["n"] / tot
    cnt["plogp"] = -cnt["p"] * np.log(cnt["p"])
    return cnt.groupby(key, observed=True)["plogp"].sum()


def _hhi_by_group(df: pd.DataFrame, key: str, cat: str) -> pd.DataFrame:
    """شاخص تمرکز هرفیندال و سهم پرتکرارترین رده، داخل هر گروه."""
    cnt = df.groupby([key, cat], observed=True).size().rename("n").reset_index()
    tot = cnt.groupby(key, observed=True)["n"].transform("sum")
    cnt["p"] = cnt["n"] / tot
    cnt["p2"] = cnt["p"] ** 2
    g = cnt.groupby(key, observed=True)
    return pd.DataFrame({"hhi": g["p2"].sum(), "top_share": g["p"].max(),
                         "n_distinct": g.size()})


def _group_corr(df: pd.DataFrame, key: str, x: str, y: str) -> pd.Series:
    """همبستگی پیرسون داخل هر گروه، از مجموع‌ها (بدون حلقه‌ی پایتون)."""
    xv = df[x].to_numpy(dtype="float64")
    yv = df[y].to_numpy(dtype="float64")
    tmp = pd.DataFrame({key: df[key].to_numpy(), "_x": xv, "_y": yv,
                        "_xx": xv * xv, "_yy": yv * yv, "_xy": xv * yv})
    g = tmp.groupby(key, observed=True)
    n = g["_x"].size()
    sx, sy = g["_x"].sum(), g["_y"].sum()
    vx = g["_xx"].sum() - sx * sx / n
    vy = g["_yy"].sum() - sy * sy / n
    cov = g["_xy"].sum() - sx * sy / n
    denom = np.sqrt(vx.clip(lower=0) * vy.clip(lower=0))
    return (cov / denom.where(denom > 0)).rename("corr")


# ---------------------------------------------------------------------------
# سال تحصیلی بدون شکاف
# ---------------------------------------------------------------------------

def add_year_block(df: pd.DataFrame, col: str = "date_gregorian") -> pd.DataFrame:
    """سال تحصیلی **بدون شکاف** (اول سپتامبر تا آخر اوت) به‌عنوان ستون `year_block`.

    ⚠️ چرا ستون رسمی `academic_year` تقویم استفاده نمی‌شود: آن ستون ۲۱۳ روز از ۹۴۳
    روز را `NaN` می‌گذارد (همان شکافی که F243 ثبتش کرده). برای تحلیل تجمیعی سلول
    این حذف بی‌صداست ولی قابل‌تحمل؛ برای تحلیل **سال‌به‌سال فرد** نیست — فردی که
    نیمی از رزروهایش در روزهای بی‌سال افتاده، بی‌صدا از یک سال حذف و در سال دیگر
    شمرده می‌شود و ماتریس گذار را خراب می‌کند.

    هر ادعای عددی که باید با دورهای قبل مقایسه شود همچنان `academic_year` رسمی را
    می‌گیرد؛ `year_block` فقط برای تقسیم‌های سال‌به‌سال این دور است.
    """
    d = pd.to_datetime(df[col])
    start_year = np.where(d.dt.month >= 9, d.dt.year, d.dt.year - 1)
    out = df.copy(deep=False)
    out["year_block"] = pd.Categorical(
        [f"{y}-{y + 1}" for y in start_year],
        categories=[f"{y}-{y + 1}" for y in range(int(start_year.min()),
                                                  int(start_year.max()) + 1)],
        ordered=True)
    return out



# ---------------------------------------------------------------------------
# ۹-۳-۲ ماتریس پروفایل — ده بلوک
# ---------------------------------------------------------------------------
#
# ⚠️ چرا این تابع بلوک‌به‌بلوک نوشته شده و نه یک‌جا
#
# نسخه‌ی نخست کل قاب ۸.۲ میلیون×۲۲ را یک‌بار می‌خواند و همه‌ی بلوک‌ها را رویش
# حساب می‌کرد. با **SIGKILL** کشته شد. سه علت داشت و هر سه اینجا رفع شده‌اند:
#
# ۱. هر `sort_values` روی قاب کامل یک کپی‌ی کامل می‌سازد، و سه بلوک مرتب‌سازی
#    لازم دارند ⇒ حالا فقط **تصویر دو-سه ستونی** مرتب می‌شود.
# ۲. `merge` روی چهار کلید که دوتاشان رده‌ای‌اند، جدول میانی بزرگی می‌سازد ⇒
#    حالا کلید مرکب **factorize** می‌شود و نگاشت با اندیس‌گذاری numpy برمی‌گردد.
# ۳. قاب کامل بین بلوک‌ها زنده می‌ماند ⇒ حالا هر بلوک ستون‌های خودش را از
#    parquet می‌خواند و در پایانش آزاد می‌شود.
#
# قاعده‌ی نگهداری: هر بلوک تازه **باید** امضای `(...) -> pd.DataFrame` با اندیس
# `PersonId` داشته باشد و ستون‌هایش را خودش بخواند. بلوکی که قاب مشترک بگیرد،
# همان تله را برمی‌گرداند.

KEY = "PersonId"


def _read(cols: list[str]) -> pd.DataFrame:
    return build_person_rows_v4(columns=[KEY] + cols)


def _block_volume() -> pd.DataFrame:
    d = _read(["date_gregorian"])
    g = d.groupby(KEY, observed=True)
    P = pd.DataFrame({"n_res": g.size(),
                      "n_active_days": g["date_gregorian"].nunique(),
                      "first_seen": g["date_gregorian"].min(),
                      "last_seen": g["date_gregorian"].max()})
    ym = (d.date_gregorian.dt.year * 12 + d.date_gregorian.dt.month).astype("int32")
    P["n_active_months"] = d.assign(_ym=ym).groupby(KEY, observed=True)["_ym"].nunique()
    P["span_days"] = (P.last_seen - P.first_seen).dt.days + 1
    P["res_per_active_month"] = P.n_res / P.n_active_months
    P["res_per_active_day"] = P.n_res / P.n_active_days
    P["density"] = P.n_res / P.span_days
    return P


def _block_waste() -> pd.DataFrame:
    d = _read(["dont_receive", "meal", "meal_seq", "Count"])
    g = d.groupby(KEY, observed=True)
    P = pd.DataFrame({"n_waste": g["dont_receive"].sum(),
                      "n_portions": g["Count"].sum()})
    n_res = g.size()
    P["rho_raw"] = P.n_waste / n_res
    P["waste_share"] = P.n_waste / P.n_waste.sum()
    bm = d.groupby([KEY, "meal"], observed=True)["dont_receive"].agg(["sum", "size"]).unstack("meal")
    for m in ("lunch", "dinner"):
        if ("size", m) in bm.columns:
            P[f"n_res_{m}"] = bm[("size", m)].reindex(P.index).fillna(0)
            P[f"rho_{m}"] = (bm[("sum", m)] / bm[("size", m)]).reindex(P.index)
    P["lunch_share"] = P.get("n_res_lunch", 0) / n_res
    P["rho_slope_seq"] = _group_slope(d, KEY, "meal_seq", "dont_receive").reindex(P.index)
    return P


def _block_rhythm() -> pd.DataFrame:
    d = _read(["day_of_week"])
    dow = d.groupby([KEY, "day_of_week"], observed=True).size().unstack("day_of_week", fill_value=0)
    share = dow.div(dow.sum(axis=1), axis=0)
    P = pd.DataFrame({f"dow{int(c)}_share": share[c] for c in share.columns})
    p = share.to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        P["dow_entropy"] = -np.nansum(np.where(p > 0, p * np.log(p), 0.0), axis=1)
    del d, dow, share

    s = _read(["meal_seq"]).sort_values([KEY, "meal_seq"])
    s["_gap"] = s.groupby(KEY, observed=True)["meal_seq"].diff()
    gs = s.groupby(KEY, observed=True)["_gap"].agg(["mean", "std"])
    gp = s.groupby(KEY, observed=True)["_gap"]
    P["gap_mean"] = gs["mean"].reindex(P.index)
    P["gap_cv"] = (gs["std"] / gs["mean"]).reindex(P.index)
    # شاخص انفجاری‌بودن گوه–بارابشی: ‎+۱ کاملاً خوشه‌ای، ‎-۱ کاملاً منظم
    P["burstiness"] = ((gs["std"] - gs["mean"]) / (gs["std"] + gs["mean"])).reindex(P.index)
    P["n_gaps_gt14"] = s.assign(_b=s._gap.gt(14)).groupby(KEY, observed=True)["_b"].sum().reindex(P.index)
    P["max_gap"] = gp.max().reindex(P.index)
    return P


def _block_sequence() -> pd.DataFrame:
    from src.eda_lib.individual_v4 import spells

    d = _read(["meal_seq", "dont_receive"])
    P = pd.DataFrame(index=d.groupby(KEY, observed=True).size().index)
    n_waste = d.groupby(KEY, observed=True)["dont_receive"].sum()

    sp = spells(d, min_reservations=1)
    if len(sp):
        sg = sp.groupby(KEY)["length"]
        P["spell_n"] = sg.size().reindex(P.index).fillna(0)
        P["spell_len_mean"] = sg.mean().reindex(P.index)
        P["spell_len_max"] = sg.max().reindex(P.index)
        long_sum = sp.loc[sp.length >= 2].groupby(KEY)["length"].sum()
        P["waste_in_spells_share"] = (long_sum.reindex(P.index).fillna(0)
                                      / n_waste.where(n_waste > 0).reindex(P.index))
    del sp

    d = d.sort_values([KEY, "meal_seq"])
    d["_lag"] = d.groupby(KEY, observed=True)["dont_receive"].shift()
    d = d.dropna(subset=["_lag"])
    P["lag1_autocorr"] = _group_corr(d, KEY, "_lag", "dont_receive").reindex(P.index)
    return P


def _block_venue() -> pd.DataFrame:
    d = _read(["RestaurantName", "is_dorm_resident", "meal_seq"])
    P = _hhi_by_group(d, KEY, "RestaurantName").rename(
        columns={"n_distinct": "n_venues", "hhi": "venue_hhi", "top_share": "venue_top_share"})
    P["dorm_venue_share"] = d.groupby(KEY, observed=True)["is_dorm_resident"].mean()
    s = d[[KEY, "RestaurantName", "meal_seq"]].sort_values([KEY, "meal_seq"])
    code = s.RestaurantName.cat.codes.to_numpy() if isinstance(
        s.RestaurantName.dtype, pd.CategoricalDtype) else pd.factorize(s.RestaurantName)[0]
    sw = pd.DataFrame({KEY: s[KEY].to_numpy(), "_c": code})
    sw["_switch"] = sw.groupby(KEY, observed=True)["_c"].diff().ne(0)
    sw = sw[sw.groupby(KEY, observed=True).cumcount() > 0]
    P["venue_switch_rate"] = sw.groupby(KEY, observed=True)["_switch"].mean().reindex(P.index)
    return P


def _block_menu() -> pd.DataFrame:
    d = _read(["FoodName", "RestaurantName", "meal", "date_gregorian"])
    P = _hhi_by_group(d, KEY, "FoodName").rename(
        columns={"n_distinct": "n_dishes", "hhi": "dish_hhi", "top_share": "dish_top_share"})
    P["dish_entropy"] = _entropy_by_group(d, KEY, "FoodName").reindex(P.index)

    # «همراهی با جمع»: سهم همان غذا داخل مجموعه‌ی انتخابِ همان (روز، وعده، سلف).
    # با factorize، نه merge — merge روی چهار کلیدِ رده‌ای همان چیزی بود که OOM داد.
    slot = pd.factorize(pd.MultiIndex.from_arrays(
        [d.date_gregorian.to_numpy(), d.meal.to_numpy(), d.RestaurantName.to_numpy()]))[0]
    combo = pd.factorize(pd.MultiIndex.from_arrays([slot, d.FoodName.to_numpy()]))[0]
    n_combo = np.bincount(combo)
    slot_of_combo = np.zeros(n_combo.size, dtype="int64")
    slot_of_combo[combo] = slot
    n_slot = np.bincount(slot)
    popshare = n_combo / n_slot[slot_of_combo]
    P["mainstream_index"] = (pd.DataFrame({KEY: d[KEY].to_numpy(), "_p": popshare[combo]})
                             .groupby(KEY, observed=True)["_p"].mean().reindex(P.index))
    return P


def _block_shock() -> pd.DataFrame:
    d = _read(["date_gregorian", "meal", "dont_receive"])
    slot_id = pd.factorize(pd.MultiIndex.from_arrays(
        [d.date_gregorian.to_numpy(), d.meal.to_numpy()]))[0]
    y = d.dont_receive.to_numpy()
    N = np.bincount(slot_id).astype("float64")
    S = np.bincount(slot_id, weights=y)
    rho_slot = S / N
    lo, hi = np.quantile(rho_slot, [0.2, 0.8])

    # فرصتِ سرویسی که فقط یک رزرو دارد، «نرخ روزِ بدون خود فرد» ندارد ⇒ NaN صریح،
    # نه صفر. (میانه‌ی اندازه‌ی فرصت ۷٬۰۵۹ است؛ چنین فرصتی نادر ولی موجود است.)
    n_minus_1 = N[slot_id] - 1
    loo = np.divide(S[slot_id] - y, n_minus_1, out=np.full(y.shape, np.nan),
                    where=n_minus_1 > 0)
    t = pd.DataFrame({KEY: d[KEY].to_numpy(), "_y": y, "_loo": loo,
                      "_bad": rho_slot[slot_id] >= hi,
                      "_good": rho_slot[slot_id] <= lo})
    P = pd.DataFrame(index=t.groupby(KEY, observed=True).size().index)
    P["rho_on_bad_days"] = t[t._bad].groupby(KEY, observed=True)["_y"].mean().reindex(P.index)
    P["rho_on_good_days"] = t[t._good].groupby(KEY, observed=True)["_y"].mean().reindex(P.index)
    P["shock_gap"] = P.rho_on_bad_days - P.rho_on_good_days
    P["shock_slope"] = _group_slope(t, KEY, "_loo", "_y").reindex(P.index)
    P.attrs["median_slot_size"] = float(np.median(N))
    P.attrs["n_slots"] = int(N.size)
    return P


def _block_satisfaction() -> pd.DataFrame:
    d = _read(["voted", "forgot_card", "Comment"])
    g = d.groupby(KEY, observed=True)
    P = pd.DataFrame({"vote_rate": g["voted"].mean(),
                      "forgot_card_rate": g["forgot_card"].mean()})
    cm = d.groupby([KEY, "Comment"], observed=True).size().unstack(fill_value=0)
    share = cm.div(cm.sum(axis=1), axis=0)
    for lbl, out in (("خوب", "vote_good_share"), ("بد", "vote_bad_share"),
                     ("ممتنع", "vote_abstain_share")):
        if lbl in share.columns:
            P[out] = share[lbl].reindex(P.index)
    return P


def _block_demographics() -> pd.DataFrame:
    """⚠️ قاعده‌ی خ — هر ستون این بلوک پیشوند `demo_` می‌گیرد و وارد خوشه‌بندی نمی‌شود."""
    cols = ["Gender", "CollegeName", "DegreeName", "EducationSession", "FieldName",
            "GroupName", "is_dorm_resident", "Price"]
    d = _read(cols)
    g = d.groupby(KEY, observed=True)
    # پرتکرارترین مقدار هر فرد، روی **کدهای رده‌ای** — `value_counts` روی ۴۶ هزار
    # گروهِ رشته‌ای چند دقیقه می‌گیرد.
    P = pd.DataFrame(index=g.size().index)
    for col, out in (("Gender", "demo_gender"), ("CollegeName", "demo_college"),
                     ("DegreeName", "demo_degree"), ("EducationSession", "demo_session"),
                     ("FieldName", "demo_field"), ("GroupName", "demo_group")):
        s = d[col]
        cats = s.cat.categories if isinstance(s.dtype, pd.CategoricalDtype) else None
        codes = s.cat.codes.to_numpy() if cats is not None else pd.factorize(s)[0]
        mode = (pd.DataFrame({KEY: d[KEY].to_numpy(), "_c": codes})
                .groupby([KEY, "_c"], observed=True).size().rename("n").reset_index()
                .sort_values("n").groupby(KEY, observed=True)["_c"].last())
        P[out] = pd.Categorical.from_codes(mode.reindex(P.index).fillna(-1).astype(int),
                                           categories=cats) if cats is not None else mode
    P["demo_is_dorm"] = g["is_dorm_resident"].mean() > 0.5
    P["demo_price_median"] = g["Price"].median()
    P["demo_free_share"] = d.assign(_f=d.Price.le(0)).groupby(KEY, observed=True)["_f"].mean()
    return P


def _block_lifecycle(P: pd.DataFrame) -> pd.DataFrame:
    """چرخه‌ی عمر از ستون‌های بلوک ب۱ مشتق می‌شود و خواندن دوباره لازم ندارد."""
    d = _read(["date_gregorian"])
    data_start, data_end = d.date_gregorian.min(), d.date_gregorian.max()
    del d
    out = pd.DataFrame(index=P.index)
    # پنجره‌ی ۳۰ روزه: کسی که در ماه اول داده دیده شده، احتمالاً پیش از آن هم فعال
    # بوده و تِنورش کوتاه‌تر از واقع ثبت می‌شود (F202: ۴۰.۸٪ سانسور چپ).
    out["left_censored"] = P.first_seen <= data_start + pd.Timedelta(days=30)
    out["right_censored"] = P.last_seen >= data_end - pd.Timedelta(days=30)
    out["tenure_days"] = P.span_days
    out["days_since_last"] = (data_end - P.last_seen).dt.days
    return out


def _block_volume_trend() -> pd.DataFrame:
    """نسبت حجم نیمه‌ی دوم به نیمه‌ی اولِ **دوره‌ی حضور خود فرد**."""
    d = _read(["date_gregorian"])
    g = d.groupby(KEY, observed=True)["date_gregorian"]
    lo, hi = g.transform("min"), g.transform("max")
    second = d.date_gregorian > (lo + (hi - lo) / 2)
    t = (pd.DataFrame({KEY: d[KEY].to_numpy(), "_h": second.to_numpy()})
         .groupby([KEY, "_h"], observed=True).size().unstack(fill_value=0))
    out = pd.DataFrame(index=t.index)
    if True in t.columns and False in t.columns:
        out["volume_trend"] = t[True] / t[False].replace(0, np.nan)
    return out


def build_person_profile_v4(force: bool = False) -> pd.DataFrame:
    """یک ردیف به‌ازای هر فرد، ده بلوک فیچر (جدول بند ۹-۳-۲)."""
    import gc

    if PERSON_PROFILE_V4.exists() and not force:
        return pd.read_parquet(PERSON_PROFILE_V4)

    meta: dict = {}
    parts = []
    for name, fn in (("ب۱ حجم", _block_volume), ("ب۲ هدررفت", _block_waste),
                     ("ب۳ ریتم", _block_rhythm), ("ب۴ توالی", _block_sequence),
                     ("ب۵ مکان", _block_venue), ("ب۶ منو", _block_menu),
                     ("ب۸ شوک", _block_shock), ("ب۹ رضایت", _block_satisfaction),
                     ("ب۱۰ جمعیتی", _block_demographics),
                     ("روند حجم", _block_volume_trend)):
        b = fn()
        meta.update(b.attrs)
        parts.append(b)
        print(f"  {name}: {b.shape[1]} ستون", flush=True)
        gc.collect()

    P = pd.concat(parts, axis=1)
    del parts
    gc.collect()
    P = pd.concat([P, _block_lifecycle(P)], axis=1)

    # ۹-۳-۳ انقباض تجربی-بیز (قاعده‌ی پ)
    a, b = fit_beta_binomial(P.n_waste.to_numpy(), P.n_res.to_numpy())
    P["rho_eb"] = eb_shrink(P.n_waste.to_numpy(), P.n_res.to_numpy(), a, b)
    meta.update(eb_alpha=a, eb_beta=b, eb_prior_strength=a + b)

    P = P.reset_index().rename(columns={"index": KEY})
    P.attrs.update(meta)
    PERSON_PROFILE_V4.parent.mkdir(parents=True, exist_ok=True)
    P.to_parquet(PERSON_PROFILE_V4, index=False)
    import json
    (DATA_INTERIM / "person_profile_v4_meta.json").write_text(
        json.dumps({k: float(v) for k, v in meta.items()}, indent=2), encoding="utf-8")
    return P
