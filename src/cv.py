"""بند ۶.۱، ۶.۲، ۶.۸ — پروتکل اعتبارسنجی زمانی.

**قاعده‌ی حاکم:** تقسیم بر اساس **تاریخ**، نه ردیف. تمام رکوردهای یک روز در یک بخش
می‌مانند، چون ICC(روز)=۰.۲۲۵ (F10) — دو رکورد از یک روز مشاهدات مستقل نیستند و
تقسیم تصادفی ردیفی، نشتی مؤثر ایجاد می‌کند.

**دو لایه‌ی ارزیابی (بند ۶.۱):**

1. ``WalkForwardSplitter`` — ۵ fold با پنجره‌ی گسترشی. 🔴 **معیار اصلی انتخاب مدل.**
2. ``holdout_split`` — پنجره‌ی انتهایی قفل‌شده. 🟡 **آزمون فشار**، نه معیار انتخاب.

چرا لایه‌ی دوم تنزل داده شد: ۲۵٪ انتهایی این داده (۱۴۰۳-۰۱-۲۸ به بعد) شامل بازگشت
پس از شکاف ۲۹ روزه‌ی رمضان و سه روز سوگواری ملی است. ممیزی فاز ۵ نشان داد اتکا به آن
به‌تنهایی، رتبه‌بندی فیچرست‌ها را گمراه می‌کند.

**چرا purge gap لازم نیست:** همه‌ی فیچرهای تاریخی از `src/features/cutoff.py` عبور
می‌کنند (مرز `meal_seq ≤ target − 2`)، پس هر فیچر فقط از گذشته‌ی خودِ آن ردیف ساخته
شده و هیچ ردیف آموزشی از داده‌ی fold آزمون تغذیه نمی‌شود. این ادعا در
`test_expanding_features_independent_of_fold` آزموده می‌شود.
"""

import hashlib
import json
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED

DATE_COL = "date_gregorian"

#: بند 7.7.3 WBS فاز ۷ — دروازه‌ی انصاف A1. `src/cv.py` این فایل را **یک‌بار** می‌سازد؛
#: هر run آموزش مدل فاز ۷ باید فقط از `load_cv_folds()` بخواند، نه `WalkForwardSplitter`
#: مستقیم، تا هش (`cv_folds_hash`) با آنچه در `doc/data_manifest.md` ثبت شده مطابق بماند.
CV_FOLDS_PATH = DATA_PROCESSED / "cv_folds.json"

#: سه روز پایانی تحت تأثیر سوگواری ملی (ردیف ۲۰ decision_log). حذف نمی‌شوند؛
#: فقط در گزارش فاز ۸ به‌عنوان برش جداگانه می‌آیند.
NATIONAL_MOURNING_START = pd.Timestamp("2024-05-19")


@dataclass(frozen=True)
class Fold:
    """یک fold از walk-forward. مرزها بر حسب **تاریخ**‌اند، نه اندیس ردیف."""

    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def masks(self, dates: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        d = pd.to_datetime(dates)
        tr = ((d >= self.train_start) & (d <= self.train_end)).to_numpy()
        te = ((d >= self.test_start) & (d <= self.test_end)).to_numpy()
        return tr, te

    def __str__(self) -> str:
        return (f"fold{self.index}: train {self.train_start.date()}→{self.train_end.date()} | "
                f"test {self.test_start.date()}→{self.test_end.date()}")


class WalkForwardSplitter:
    """پنجره‌ی گسترشی (بند ۶.۲ گزینه A).

    با ۱۴۲ روز سرو، دورریختن داده‌ی قدیمی (پنجره‌ی لغزان) قابل‌توجیه نیست. گزینه‌ی
    لغزان فقط در صورت کشف شکست رژیمی در فاز ۸ بازبینی می‌شود.
    """

    def __init__(self, n_folds: int = 5, min_train_days: int = 60,
                 test_days: int | None = None, embargo_days: int = 0):
        self.n_folds = n_folds
        self.min_train_days = min_train_days
        self.test_days = test_days
        self.embargo_days = embargo_days

    def split_dates(self, dates: pd.Series) -> list[Fold]:
        uniq = pd.Index(sorted(pd.to_datetime(dates).unique()))
        n = len(uniq)
        if n <= self.min_train_days + self.n_folds:
            raise ValueError(f"داده‌ی کافی نیست: {n} روز یکتا")

        remaining = n - self.min_train_days
        test_days = self.test_days or max(1, remaining // self.n_folds)

        folds = []
        for i in range(self.n_folds):
            tr_end_idx = self.min_train_days + i * test_days - 1
            te_start_idx = tr_end_idx + 1 + self.embargo_days
            te_end_idx = min(te_start_idx + test_days - 1, n - 1)
            if te_start_idx > n - 1:
                break
            folds.append(Fold(
                index=i,
                train_start=uniq[0],
                train_end=uniq[tr_end_idx],
                test_start=uniq[te_start_idx],
                test_end=uniq[te_end_idx],
            ))
        return folds

    def split(self, df: pd.DataFrame, date_col: str = DATE_COL) -> Iterator[tuple[Fold, np.ndarray, np.ndarray]]:
        for f in self.split_dates(df[date_col]):
            tr, te = f.masks(df[date_col])
            yield f, tr, te


def holdout_split(df: pd.DataFrame, test_days: int = 25,
                  date_col: str = DATE_COL) -> tuple[np.ndarray, np.ndarray, Fold]:
    """پنجره‌ی انتهایی قفل‌شده — 🟡 **آزمون فشار**، نه معیار انتخاب مدل (بند ۶.۱)."""
    uniq = pd.Index(sorted(pd.to_datetime(df[date_col]).unique()))
    cut = uniq[-test_days]
    f = Fold(index=-1, train_start=uniq[0], train_end=uniq[-test_days - 1],
             test_start=cut, test_end=uniq[-1])
    tr, te = f.masks(df[date_col])
    return tr, te, f


# ---------------------------------------------------------------------------
# بند 7.7.3 سند فاز ۷ — انجماد foldها در فایل + هش، دروازه‌ی انصاف A1
# ---------------------------------------------------------------------------

def _folds_to_records(folds: list[Fold]) -> list[dict]:
    return [
        {
            "index": f.index,
            "train_start": f.train_start.date().isoformat(),
            "train_end": f.train_end.date().isoformat(),
            "test_start": f.test_start.date().isoformat(),
            "test_end": f.test_end.date().isoformat(),
        }
        for f in folds
    ]


def _records_to_folds(records: list[dict]) -> list[Fold]:
    return [
        Fold(
            index=r["index"],
            train_start=pd.Timestamp(r["train_start"]),
            train_end=pd.Timestamp(r["train_end"]),
            test_start=pd.Timestamp(r["test_start"]),
            test_end=pd.Timestamp(r["test_end"]),
        )
        for r in records
    ]


def _folds_payload(df: pd.DataFrame, splitter: WalkForwardSplitter, source: str) -> dict:
    """محتوای قطعی (deterministic) فایل — بدون timestamp، تا هش فقط تابع داده و تنظیمات splitter باشد."""
    folds = splitter.split_dates(df[DATE_COL])
    return {
        "date_col": DATE_COL,
        "source": source,
        "splitter": {
            "n_folds": splitter.n_folds,
            "min_train_days": splitter.min_train_days,
            "test_days": splitter.test_days,
            "embargo_days": splitter.embargo_days,
        },
        "n_unique_dates": int(df[DATE_COL].nunique()),
        "folds": _folds_to_records(folds),
    }


def sha256_file(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_cv_folds(df: pd.DataFrame, splitter: WalkForwardSplitter, source: str,
                   path=CV_FOLDS_PATH) -> str:
    """می‌سازد و می‌نویسد. **فقط یک‌بار** اجرا شود (`python -m src.cv`) — نتیجه در git/DVC می‌ماند
    و هرگز بی‌سروصدا بازنویسی نمی‌شود، وگرنه هش تغییر می‌کند و run‌های قبلی از دروازه‌ی A1 رد می‌شوند.

    برمی‌گرداند: هش SHA-256 فایل نوشته‌شده، برای ثبت در `doc/data_manifest.md`.
    """
    payload = _folds_payload(df, splitter, source)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return sha256_file(path)


def load_cv_folds(path=CV_FOLDS_PATH) -> tuple[list[Fold], str]:
    """بارگذاری foldهای منجمدشده + هش فایل. هر اسکریپت/نوت‌بوک آموزش مدل فاز ۷ باید از این
    تابع بخواند و `cv_folds_hash` بازگشتی را به‌عنوان param هر MLflow run ثبت کند (بند 7.7.2).
    """
    payload = json.loads(path.read_text())
    return _records_to_folds(payload["folds"]), sha256_file(path)


# ---------------------------------------------------------------------------
# بند ۶.۶ — اندازه‌ی نمونه‌ی مؤثر و بوت‌استرپ بلوکی دوبعدی
# ---------------------------------------------------------------------------

def effective_sample_size(n: int, cluster_sizes: np.ndarray, icc: float) -> float:
    """$n_{eff} = n / (1 + (\\bar m - 1)\\,\\rho_{ICC})$ — تصحیح اثر طرح خوشه‌ای.

    با ICC(روز)=۰.۲۲۵ و ~۵۳ رکورد در روز، ضریب تورم واریانس بزرگ است. هر فاصله‌ی
    اطمینانی که این را نادیده بگیرد، **باریک‌تر از واقعیت** است (F10، F59).
    """
    m_bar = float(np.mean(cluster_sizes))
    return n / (1.0 + (m_bar - 1.0) * icc)


def block_bootstrap_2d(df: pd.DataFrame, stat_fn, day_col: str = DATE_COL,
                       unit_col: str = "RestaurantName", n_boot: int = 1000,
                       alpha: float = 0.05, seed: int = 42) -> tuple[float, float, float]:
    """بوت‌استرپ بلوکی **دوبعدی**: نمونه‌گیری هم‌زمان روی روز و سلف (بند ۶.۶).

    بوت‌استرپ تک‌بعدی کافی نیست: ICC(روز)=۰.۲۲۵ و ICC(سلف)=۰.۲۰۵ تقریباً برابرند
    (F10)، پس نادیده‌گرفتن هرکدام، CI را به‌طور غیرقابل‌قبولی باریک می‌کند.

    روش: در هر تکرار، هم مجموعه‌ی روزها و هم مجموعه‌ی سلف‌ها با جایگذاری نمونه‌گیری
    می‌شوند و حاصل‌ضرب دکارتی آن‌ها نمونه‌ی بوت‌استرپ را می‌سازد.
    """
    rng = np.random.default_rng(seed)
    days = df[day_col].unique()
    units = df[unit_col].unique()
    idx = {(d, u): g.index.to_numpy() for (d, u), g in df.groupby([day_col, unit_col], observed=True)}

    stats = []
    for _ in range(n_boot):
        bd = rng.choice(days, size=len(days), replace=True)
        bu = rng.choice(units, size=len(units), replace=True)
        parts = [idx[(d, u)] for d in bd for u in bu if (d, u) in idx]
        if not parts:
            continue
        sample = df.loc[np.concatenate(parts)]
        try:
            stats.append(float(stat_fn(sample)))
        except Exception:
            continue
    stats = np.asarray(stats, dtype=float)
    stats = stats[np.isfinite(stats)]
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(stat_fn(df)), float(lo), float(hi)


def diebold_mariano(err_a: np.ndarray, err_b: np.ndarray, h: int = 1) -> tuple[float, float]:
    """آزمون Diebold-Mariano برای مقایسه‌ی دقت دو پیش‌بینی (بند ۶.۶).

    ورودی‌ها **زیان** هر مشاهده‌اند (نه خطای خام)، تا با هر تابع زیانی — از جمله
    pinball نامتقارن — کار کند. آماره با تصحیح خودهمبستگی تا وقفه‌ی $h-1$ محاسبه می‌شود.
    """
    from scipy import stats as st

    d = np.asarray(err_a, dtype=float) - np.asarray(err_b, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    d_bar = d.mean()
    gamma0 = np.var(d, ddof=0)
    gammas = [np.cov(d[:-k], d[k:], ddof=0)[0, 1] for k in range(1, h)] if h > 1 else []
    var_d = (gamma0 + 2 * sum(gammas)) / n
    if var_d <= 0:
        return float("nan"), float("nan")
    dm = d_bar / np.sqrt(var_d)
    p = 2 * (1 - st.norm.cdf(abs(dm)))
    return float(dm), float(p)


# ---------------------------------------------------------------------------
# بند ۶.۸ — تست‌های واحد پروتکل
# ---------------------------------------------------------------------------

def test_no_date_overlap(df: pd.DataFrame, splitter: WalkForwardSplitter) -> tuple[bool, str]:
    """هیچ تاریخی نباید هم‌زمان در train و test یک fold باشد."""
    for f, tr, te in splitter.split(df):
        d_tr = set(pd.to_datetime(df.loc[tr, DATE_COL]).unique())
        d_te = set(pd.to_datetime(df.loc[te, DATE_COL]).unique())
        if d_tr & d_te:
            return False, f"{f}: {len(d_tr & d_te)} تاریخ مشترک"
    return True, "هیچ تاریخ مشترکی بین train و test هیچ fold نیست"


def test_folds_chronological(df: pd.DataFrame, splitter: WalkForwardSplitter) -> tuple[bool, str]:
    """آزمون هر fold باید کاملاً پس از آموزش همان fold و پس از آزمون fold قبلی باشد."""
    folds = splitter.split_dates(df[DATE_COL])
    for i, f in enumerate(folds):
        if f.test_start <= f.train_end:
            return False, f"fold{i}: آزمون پیش از پایان آموزش شروع می‌شود"
        if i > 0 and f.test_start <= folds[i - 1].test_start:
            return False, f"fold{i}: ترتیب زمانی fold‌ها صعودی نیست"
    return True, f"{len(folds)} fold، همه با ترتیب زمانی صحیح"


def test_expanding_features_independent_of_fold(df: pd.DataFrame, feature: str,
                                                splitter: WalkForwardSplitter) -> tuple[bool, str]:
    """مقدار فیچر انبساطی یک ردیف نباید به اینکه در کدام fold است بستگی داشته باشد.

    این همان ادعای «purge gap لازم نیست» است (بند ۶.۲): چون فیچرها با قاعده‌ی برش
    ساخته شده‌اند، مقدارشان تابعی از گذشته‌ی خودِ ردیف است و بازمحاسبه روی زیرمجموعه‌ی
    آموزشِ هر fold باید همان عدد را بدهد.
    """
    from src.features.cutoff import expanding_stat_at_cutoff, meal_seq

    d = df.copy()
    d["meal_seq"] = meal_seq(d[DATE_COL], d["Meal"])
    folds = splitter.split_dates(d[DATE_COL])
    f = folds[len(folds) // 2]
    tr, _ = f.masks(d[DATE_COL])

    key = ["RestaurantName", "Meal"]
    full = expanding_stat_at_cutoff(d, key, "NoRecv", out_prefix="x")["x_sum"]
    sub = expanding_stat_at_cutoff(d.loc[tr], key, "NoRecv", out_prefix="x")["x_sum"]
    diff = (full.loc[sub.index] - sub).abs()
    bad = int((diff > 1e-9).sum())
    ok = bad == 0
    return ok, f"{len(sub):,} ردیف آموزشی بازمحاسبه شد · {bad} اختلاف"


def run_protocol_tests(df: pd.DataFrame, splitter: WalkForwardSplitter) -> bool:
    checks = [
        ("عدم هم‌پوشانی تاریخ train/test", test_no_date_overlap(df, splitter)),
        ("ترتیب زمانی fold‌ها", test_folds_chronological(df, splitter)),
        ("استقلال فیچر انبساطی از مرز fold", test_expanding_features_independent_of_fold(
            df, "cell_expanding_rate", splitter)),
    ]
    all_ok = True
    for name, (ok, msg) in checks:
        print(f"  {'✅' if ok else '❌'} {name:<38s} {msg}")
        all_ok &= ok
    return all_ok


def main() -> None:
    """تولید یک‌بارِ `data/processed/cv_folds.json` (بند 7.7.3 سند فاز ۷).

    اجرا: ``python -m src.cv``. اگر فایل از قبل وجود دارد، بازنویسی نمی‌کند — چون بازنویسی
    یعنی هش عوض می‌شود و run‌های فاز ۷ که قبلاً با هش قدیمی ثبت شده‌اند نامعتبر می‌شوند.
    """
    from src.config import ROOT_DIR
    from src.features.build import FEATURES_A_PATH

    if CV_FOLDS_PATH.exists():
        _, h = load_cv_folds()
        print(f"⚠️  {CV_FOLDS_PATH} از قبل وجود دارد — بازنویسی نمی‌شود.")
        print(f"    هش فعلی: {h}")
        print("    برای بازتولید عمدی، ابتدا فایل را حذف کنید (و پیامدش روی run‌های قبلی را بسنجید).")
        return

    df = pd.read_parquet(FEATURES_A_PATH)
    splitter = WalkForwardSplitter(n_folds=5, min_train_days=60)
    h = write_cv_folds(df, splitter, source=str(FEATURES_A_PATH.relative_to(ROOT_DIR)))

    folds, _ = load_cv_folds()
    print(f"✅ {CV_FOLDS_PATH} ساخته شد — {len(folds)} fold از {df[DATE_COL].nunique()} تاریخ یکتا")
    for f in folds:
        print(f"    {f}")
    print(f"\ncv_folds_hash = {h}")
    print("\n⚠️ این هش را در doc/data_manifest.md ثبت کنید — دروازه‌ی انصاف A1 فاز ۷.")


if __name__ == "__main__":
    main()
