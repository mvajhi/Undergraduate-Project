"""تبارشناسی اختلال سرویس — پایه‌ی مشترک دور ۴ کاوش داده.

**چرا این ماژول لازم شد.** آرشیو `events.csv` می‌گوید چه رویدادی *رخ داده*، ولی نمی‌گوید
سرویس غذا در آن روز *چه شد*. سه منبع باید کنار هم بنشینند تا «روز اختلال» تعریف عملیاتی
پیدا کند:

1. `dataset_v2` — چه (روز، وعده، سلف)ی واقعاً سرو شده،
2. `calendar_tehran.csv` — کدام غیاب از پیش برنامه‌ریزی‌شده بوده (جمعه، تعطیل رسمی، نوروز،
   میان‌ترم)،
3. `events.csv` — چه رویداد بافتاری‌ای روی آن روز نشسته.

تفکیک کلیدی که این ماژول اعمال می‌کند: **غیاب برنامه‌ریزی‌شده** در برابر **غیاب
غیرمنتظره**. غیاب غیرمنتظره یعنی جفت (سلف، وعده) در همان روزِ هفته معمولاً فعال است ولی
آن روز خاص هیچ سروی نداشته. آستانه‌ها (`P_ACTIVE_MIN`, `N_DOW_MIN`) صریح‌اند تا نتیجه
بازتولیدپذیر بماند.

⚠️ خروجی این ماژول **فیچر یادگیری نیست** (بند ۲.۴ WBS: هر رویداد یکتا یک مشاهده دارد).
نقشش توصیف و آزمون رفتار حول اختلال است.

اجرا: `python -m src.eda_lib.disruption`
"""

import numpy as np
import pandas as pd

from src.config import DATA_EXTERNAL, DATA_INTERIM, DATA_PROCESSED

COVERAGE_GRID_PATH = DATA_INTERIM / "coverage_grid_v2.csv"
DATASET_PATH = DATA_PROCESSED / "dataset_v2.csv"
CALENDAR_PATH = DATA_EXTERNAL / "calendar_tehran.csv"
EVENTS_PATH = DATA_EXTERNAL / "events.csv"

# یک جفت (سلف، وعده) در یک روزِ هفته «معمولاً فعال» است اگر دست‌کم در ۸۰٪ روزهای
# غیرتعطیلِ آن روزِ هفته سرو داشته باشد، و دست‌کم ۴ بار فرصت سرو داشته باشد. آستانه‌ی
# ۰.۸ سلف‌های نیمه‌فعال را کنار می‌گذارد تا «تعطیلی غیرمنتظره» با «برنامه‌ی نامنظم» خلط نشود.
P_ACTIVE_MIN = 0.80
N_DOW_MIN = 4


def load_calendar() -> pd.DataFrame:
    return pd.read_csv(CALENDAR_PATH, parse_dates=["date_gregorian"])


def load_events() -> pd.DataFrame:
    ev = pd.read_csv(EVENTS_PATH, parse_dates=["date_start", "date_end"])
    return ev


def expand_events(ev: pd.DataFrame) -> pd.DataFrame:
    """آرشیو رویداد را از سطح «بازه» به سطح «روز» باز می‌کند (یک ردیف به‌ازای هر روزِ رویداد)."""
    rows = []
    for r in ev.itertuples(index=False):
        for day in pd.date_range(r.date_start, r.date_end, freq="D"):
            rows.append({"date_gregorian": day, "event_id": r.event_id,
                         "event_type": r.event_type, "service_status": r.service_status,
                         "scope": r.scope})
    return pd.DataFrame(rows)


def unit_service_matrix() -> pd.DataFrame:
    """سطح (روز، سلف، وعده): فعال بود یا نه، و آیا غیابش غیرمنتظره است.

    فقط جفت‌های (سلف، وعده)ای که در کل بازه دست‌کم یک‌بار سرو داشته‌اند وارد می‌شوند —
    بقیه ساختاری‌اند (`restaurant_meal_not_offered`) و غیابشان اختلال نیست.
    """
    grid = pd.read_csv(COVERAGE_GRID_PATH, parse_dates=["date_gregorian"])
    cal = load_calendar()

    rm = grid.groupby(["date_gregorian", "RestaurantName", "Meal"], as_index=False).agg(
        n_served=("is_served", "sum"),
        structural=("absence_reason", lambda s: bool((s == "restaurant_meal_not_offered").all())),
    )
    rm = rm[~rm["structural"]].drop(columns="structural")
    rm["active"] = rm["n_served"] > 0

    rm = rm.merge(cal[["date_gregorian", "date_jalali", "day_of_week", "is_holiday_any"]],
                  on="date_gregorian", how="left")

    open_days = rm[~rm["is_holiday_any"]]
    prop = (open_days.groupby(["RestaurantName", "Meal", "day_of_week"])["active"]
            .agg(p_active="mean", n_dow="size").reset_index())
    rm = rm.merge(prop, on=["RestaurantName", "Meal", "day_of_week"], how="left")

    rm["expected_active"] = (~rm["is_holiday_any"]) & (rm["p_active"] >= P_ACTIVE_MIN) & (rm["n_dow"] >= N_DOW_MIN)
    rm["unexpected_closure"] = rm["expected_active"] & (~rm["active"])
    return rm


def day_disruption_table() -> pd.DataFrame:
    """سطح روز: شدت اختلال + برچسب رده + الحاق رویداد.

    ستون `disruption_class` پنج مقدار می‌گیرد:
      - `تعطیل_تقویمی`      روز در تقویم تعطیل است (جمعه/رسمی/نوروز/میان‌ترم)
      - `قطع_کامل`          روز غیرتعطیل، ولی هیچ واحدِ انتظارروَنده‌ای سرو نکرده
      - `قطع_گسترده`        ≥۵۰٪ واحدهای انتظارروَنده بسته
      - `قطع_موردی`         ۱ تا <۵۰٪ واحدها بسته
      - `عادی`              هیچ غیاب غیرمنتظره‌ای ندارد
    """
    rm = unit_service_matrix()
    cal = load_calendar()

    exp = rm[rm["expected_active"]]
    agg = exp.groupby("date_gregorian").agg(
        n_expected=("expected_active", "size"),
        n_closed=("unexpected_closure", "sum")).reset_index()

    served_days = rm.groupby("date_gregorian")["active"].sum().rename("n_active").reset_index()

    day = cal[["date_gregorian", "date_jalali", "day_of_week", "is_holiday_any",
               "is_nowruz_block", "is_inter_semester_break", "is_exam_period", "holiday_name"]].copy()
    day = day.merge(agg, on="date_gregorian", how="left").merge(served_days, on="date_gregorian", how="left")
    day = day[day["n_active"].notna()].copy()  # فقط بازه‌ی داده
    day[["n_expected", "n_closed"]] = day[["n_expected", "n_closed"]].fillna(0)
    day["frac_closed"] = np.where(day["n_expected"] > 0, day["n_closed"] / day["n_expected"], 0.0)

    def classify(r):
        if r["is_holiday_any"]:
            return "تعطیل_تقویمی"
        if r["n_expected"] > 0 and r["frac_closed"] >= 0.999:
            return "قطع_کامل"
        if r["frac_closed"] >= 0.50:
            return "قطع_گسترده"
        if r["n_closed"] > 0:
            return "قطع_موردی"
        return "عادی"

    day["disruption_class"] = day.apply(classify, axis=1)

    ev_days = expand_events(load_events())
    ev_roll = (ev_days.groupby("date_gregorian")
               .agg(event_ids=("event_id", lambda s: "؛ ".join(sorted(s))),
                    event_types=("event_type", lambda s: "؛ ".join(sorted(set(s)))),
                    n_events=("event_id", "size")).reset_index())
    day = day.merge(ev_roll, on="date_gregorian", how="left")
    day["n_events"] = day["n_events"].fillna(0).astype(int)
    return day.sort_values("date_gregorian").reset_index(drop=True)


def meal_gap_table(cell: pd.DataFrame) -> pd.DataFrame:
    """برای هر وعده، فاصله‌ی هر روزِ سرو تا روزِ سروِ قبلیِ همان وعده در سطح دانشگاه.

    `run_pos` = چندمین روزِ سرو پس از آخرین شکافِ ≥`GAP_MIN` روزه. این ستون است که
    «رفتار روزهای بعد» را قابل‌اندازه‌گیری می‌کند.
    """
    out = []
    for meal, s in cell.groupby("Meal"):
        days = (s.groupby("date_gregorian")["Res"].sum().reset_index()
                .sort_values("date_gregorian"))
        days["gap_days"] = days["date_gregorian"].diff().dt.days
        out.append(days.assign(Meal=meal))
    return pd.concat(out, ignore_index=True)[["date_gregorian", "Meal", "gap_days"]]


def add_run_position(gaps: pd.DataFrame, gap_min: int = 4) -> pd.DataFrame:
    """شماره‌ی روزِ سرو پس از هر شکافِ ≥`gap_min` روزه (۱ = اولین روز بازگشت)."""
    out = []
    for meal, s in gaps.groupby("Meal"):
        s = s.sort_values("date_gregorian").copy()
        s["is_return"] = s["gap_days"].fillna(0) >= gap_min
        s["run_id"] = s["is_return"].cumsum()
        s["run_pos"] = s.groupby("run_id").cumcount() + 1
        s.loc[s["run_id"] == 0, "run_pos"] = np.nan
        s["gap_len"] = s.groupby("run_id")["gap_days"].transform("first")
        s.loc[s["run_id"] == 0, "gap_len"] = np.nan
        out.append(s)
    return pd.concat(out, ignore_index=True)


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    day = day_disruption_table()
    print(day["disruption_class"].value_counts().to_string())
    print()
    print(day[day["disruption_class"].isin(["قطع_کامل", "قطع_گسترده"])]
          [["date_gregorian", "date_jalali", "day_of_week", "n_expected", "n_closed",
            "frac_closed", "disruption_class", "event_ids"]].to_string(index=False))
