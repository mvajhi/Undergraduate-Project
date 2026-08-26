"""بند 7.16 + بند 7.24.3 — خانواده‌ی ۷ روی **سطح فرد (L5)**، با تجمیع پواسون-دوجمله‌ای.

## سؤال محوری این ماژول

بند 7.16 صریح می‌گوید انتظار از شبکه‌ی عصبی روی L1 منفی است (۷٬۵۷۹ رکورد) ولی روی
**L5 جدی‌ترین شانس خانواده** است: ۲٬۰۴۹٬۳۲۲ رزرو، ۲۶٬۷۶۸ فرد، و F58 که می‌گوید
تاریخچه‌ی شخصی ~۸ برابر قوی‌تر از جمعیت‌شناسی است. embedding می‌تواند هر دو را
هم‌زمان و با تعامل یاد بگیرد — کاری که فیچرهای دست‌ساز نمی‌کنند.

⚠️ **ولی اسپرینت B همین فرضیه را از یک مسیر دیگر رد کرد** (یافته‌ی ۲۱): فیچرهای
کوهورت L5→L1 هیچ کمکی نکردند چون سیگنال فردی از قبل، به شکل تجمیع‌شده، در فیچرهای
rolling سلولی حاضر بود. این ماژول همان فرضیه را در **قوی‌ترین شکل ممکنش** می‌آزماید:
نه فیچر تجمیع‌شده، بلکه یک مدل کامل سطح فرد با embedding خودِ `PersonId`. اگر این هم
نبرد، رد فرضیه دیگر «شاید تجمیع بد بود» ندارد.

## تجمیع L5 → سلول (بند 7.24.3) — نکته‌ی فنی محوری

مدل به‌ازای هر رزرو $i$ یک احتمال $p_i = P(\\text{عدم دریافت})$ می‌دهد. تعداد
عدم‌دریافت یک سلول مجموع برنولی‌های **ناهم‌توزیع** است ⇒ توزیع **پواسون-دوجمله‌ای**:

$$\\mu = \\sum_i p_i, \\qquad \\sigma^2 = \\sum_i p_i(1-p_i), \\qquad
\\gamma = \\frac{\\sum_i p_i(1-p_i)(1-2p_i)}{\\sigma^3}$$

کوانتایل با بسط **کورنیش-فیشر** (تصحیح چولگی روی تقریب نرمال) گرفته می‌شود، نه با
DP دقیق: DP دقیق $O(n^2)$ به‌ازای هر سلول است و با ~۴٬۱۰۰ سلول × ۵ fold از بودجه‌ی
کل نوت‌بوک بیشتر می‌شود، در حالی که با $n$ معمول این داده خطای تقریب ناچیز است.

⚠️⚠️ **و این‌جا دقیقاً همان تله‌ای است که بند 7.24.3 هشدار می‌دهد:** پواسون-دوجمله‌ای
**استقلال افراد** را فرض می‌کند، ولی F59 می‌گوید ۸۳٪ واریانس سلول شوک **مشترک**
روزانه است — یعنی خطاها همبسته‌اند و $\\sigma^2$ بالا **کم‌برآورد** می‌شود. به همین
دلیل یک هایپرپارامتر صریح `overdispersion` (ضریب تورم انحراف معیار) گذاشته شده که
Optuna تنظیمش می‌کند: اگر مقدار بهینه‌اش به‌طور معنادار بزرگ‌تر از ۱ درآید، این
**خودش یک اندازه‌گیری از همبستگی درون‌روزی** است، نه یک وصله.

## سطح آزمایش در برابر سطح ارزیابی

مدل روی **L5** یاد می‌گیرد ولی ارزیابی روی همان ردیف‌های **L1** انجام می‌شود که
همه‌ی خانواده‌های دیگر با آن سنجیده شده‌اند (بند 7.1.2 قاعده‌ی مقایسه‌پذیری). چون
داده‌ی فردی `FoodType` ندارد، پیش‌بینی هر سلول $(d,m,r)$ روی همه‌ی سطرهای هم‌غذای
آن پخش می‌شود — همان محدودیتی که `src/features/cohort_features.py` هم دارد و
این‌جا هم صریحاً در گزارش می‌آید (محور `output_aggregation="aggregated"`).
"""

import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from scipy import stats

from src.cv import DATE_COL
from src.models.gpu_runner import FamilyFitter, models_from_fitters
from src.models.registry import ModelSpec, register
from src.models.spaces import register_space

FAMILY = "F07"
LEVEL = "L5"
FEATURE_SET = "FS_F07_l5_v1"

PERSON_PATH = Path("data/processed/person_features_v1.parquet")
CELL_KEYS = ["date_gregorian", "Meal", "RestaurantName"]

#: فیچرهای عددی سطح فرد (همه از قبل با قاعده‌ی برش ساخته شده‌اند — `person_features.py`)
PERSON_NUMERIC = [
    "person_n_prior_reservations", "person_n_prior_norecv", "person_reservations_per_week",
    "person_expanding_norecv_rate", "person_shrunk_norecv_rate", "person_ewm_norecv_rate",
    "person_meal_share", "person_restaurant_share", "person_dow_share",
    "college_freq", "field_freq", "is_honeymoon", "is_cold_start", "is_dorm_resident",
    "is_grad", "is_female", "is_evening_session", "is_tehran",
]
#: ستون‌های دسته‌ای که embedding می‌گیرند — `PersonId` مهم‌ترینشان (بند 7.16.1 عضو ۲)
PERSON_CATEGORICAL = ["PersonId", "Meal", "restaurant_canonical", "city"]

#: ⭐ زمینه‌ی روز، از جدول L1 (که ممیزی برش را پاس کرده) به هر رزرو merge می‌شود.
#: بدون این‌ها مدل L5 اصلاً نمی‌تواند شوک مشترک روزانه را ببیند — و F59 می‌گوید همان
#: شوک ۸۳٪ واریانس سلول است. حذفشان یعنی آزمودن فرضیه با یک دست بسته.
DAY_CONTEXT = [
    "dow", "is_holiday_any", "is_day_before_holiday", "is_exam_period", "is_final_exam_period",
    "is_ramadan", "week_of_semester", "days_to_next_holiday", "temp_min",
    "log_daily_total_res", "day_shock_lag1", "day_shock_lag7", "day_shock_roll_mean_7",
]

_PERSON_CACHE: dict = {}


def load_person_frame(path: Path | str = PERSON_PATH, l1: pd.DataFrame | None = None) -> pd.DataFrame:
    """جدول رزرو فردی + زمینه‌ی روز، یک‌بار خوانده و در حافظه کش می‌شود (۲ میلیون ردیف)."""
    key = str(path)
    if key in _PERSON_CACHE:
        return _PERSON_CACHE[key]

    person = pd.read_parquet(path)
    person = person.rename(columns={"restaurant_canonical": "RestaurantName"})
    person["RestaurantName"] = person["RestaurantName"].astype(str)
    person["Meal"] = person["Meal"].astype(str)

    if l1 is not None:
        day_cols = [c for c in DAY_CONTEXT if c in l1.columns]
        ctx = (l1.groupby([DATE_COL, "Meal"], observed=True)[day_cols].first().reset_index())
        ctx["Meal"] = ctx["Meal"].astype(str)
        person = person.merge(ctx, on=[DATE_COL, "Meal"], how="left")

    _PERSON_CACHE[key] = person
    return person


def load_l5_bridge(person_path: Path | str = PERSON_PATH):
    """`LevelData` ای که **ارزیابی‌اش L1 است ولی محور `level` را L5 اعلام می‌کند**.

    عمدی و صریح: مدل روی رزرو فردی یاد می‌گیرد (محور سطح داده = L5)، ولی هر معیاری
    که گزارش می‌شود روی همان سطرهای L1 محاسبه می‌شود که خ۱/خ۲/… با آن سنجیده شدند —
    وگرنه عدد pinball این خانواده با هیچ ردیف دیگری از جدول مقایسه قابل‌قیاس نبود.
    """
    from dataclasses import replace

    from src.cv import sha256_file
    from src.models.gpu_runner import load_l1

    data = load_l1()
    load_person_frame(person_path, l1=data.df)     # کش‌کردن + merge زمینه‌ی روز
    return replace(data, level="L5", data_snapshot_hash=sha256_file(Path(person_path)),
                   source=str(person_path))


# ---------------------------------------------------------------------------
# پیش‌پردازش سطح فرد
# ---------------------------------------------------------------------------

class PersonPrep:
    def __init__(self):
        self.num_cols: list[str] = []
        self.cat_cols: list[str] = []
        self.medians: dict = {}
        self.center: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.vocab: dict[str, dict] = {}

    def fit(self, train_person: pd.DataFrame) -> "PersonPrep":
        self.num_cols = [c for c in PERSON_NUMERIC + DAY_CONTEXT if c in train_person.columns]
        self.cat_cols = [c for c in PERSON_CATEGORICAL if c in train_person.columns]
        num = train_person[self.num_cols].astype(float)
        self.medians = num.median().to_dict()
        num = num.fillna(pd.Series(self.medians))
        q25, q75 = num.quantile(0.25).to_numpy(), num.quantile(0.75).to_numpy()
        self.center = num.median().to_numpy()
        self.scale = np.where((q75 - q25) > 1e-9, q75 - q25, 1.0)
        for c in self.cat_cols:
            levels = pd.unique(train_person[c].astype(str))
            self.vocab[c] = {v: i + 1 for i, v in enumerate(sorted(map(str, levels)))}
        return self

    @property
    def cardinalities(self) -> list[int]:
        return [len(self.vocab[c]) + 1 for c in self.cat_cols]

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        num = df[self.num_cols].astype(float).fillna(pd.Series(self.medians))
        dense = ((num.to_numpy() - self.center) / self.scale).astype(np.float32)
        cats = np.zeros((len(df), len(self.cat_cols)), dtype=np.int64)
        for j, c in enumerate(self.cat_cols):
            vocab = self.vocab[c]
            cats[:, j] = df[c].astype(str).map(vocab).fillna(0).to_numpy(dtype=np.int64)
        return dense, cats


def _build_net_cls():
    import torch
    from torch import nn

    class PersonNet(nn.Module):
        """MLP با entity embedding روی سطح رزرو — خروجی: لوجیت احتمال عدم‌دریافت."""

        def __init__(self, n_dense: int, cardinalities: list[int], emb_dim_k: float,
                     hidden_dim: int, n_layers: int, dropout: float):
            super().__init__()
            self.embeddings = nn.ModuleList()
            emb_total = 0
            for card in cardinalities:
                dim = max(2, min(64, int(np.ceil(card ** 0.25) * emb_dim_k)))
                self.embeddings.append(nn.Embedding(card, dim))
                emb_total += dim
            layers: list[nn.Module] = []
            in_dim = n_dense + emb_total
            for _ in range(n_layers):
                layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
                in_dim = hidden_dim
            self.body = nn.Sequential(*layers)
            self.head = nn.Linear(in_dim, 1)

        def forward(self, dense, cats):
            parts = [dense] + [emb(cats[:, j]) for j, emb in enumerate(self.embeddings)]
            return self.head(self.body(torch.cat(parts, dim=1))).squeeze(-1)

    return PersonNet


class L5Model:
    def __init__(self, net, prep: PersonPrep, arch: dict, history: list, n_parameters: int,
                 person_path: str):
        self.net = net
        self.prep = prep
        self.arch = arch
        self.history = history
        self.n_parameters = n_parameters
        self.person_path = person_path

    def person_probabilities(self, person_rows: pd.DataFrame, batch: int = 65536) -> np.ndarray:
        import torch

        from src.models.gpu_runner import torch_device

        device = torch_device()
        self.net.eval().to(device)
        dense, cats = self.prep.transform(person_rows)
        out = np.empty(len(person_rows), dtype=np.float64)
        with torch.no_grad():
            for s in range(0, len(person_rows), batch):
                e = slice(s, min(s + batch, len(person_rows)))
                logits = self.net(torch.as_tensor(dense[e], dtype=torch.float32, device=device),
                                  torch.as_tensor(cats[e], dtype=torch.long, device=device))
                out[e] = torch.sigmoid(logits).cpu().numpy()
        return out


# ---------------------------------------------------------------------------
# تجمیع پواسون-دوجمله‌ای (بند 7.24.3)
# ---------------------------------------------------------------------------

def poisson_binomial_quantile_rate(p: np.ndarray, tau: float, overdispersion: float = 1.0
                                   ) -> float:
    """کوانتایل $\\tau$ **نرخ** $\\rho=K/n$ برای $K\\sim\\text{PoissonBinomial}(p)$،
    با بسط کورنیش-فیشر (تصحیح چولگی) و تورم اختیاری انحراف معیار."""
    n = len(p)
    if n == 0:
        return float("nan")
    mu = float(p.sum())
    var = float((p * (1 - p)).sum())
    if var <= 1e-12:
        return float(np.clip(mu / n, 0.0, 1.0))
    sd = np.sqrt(var) * overdispersion
    gamma = float((p * (1 - p) * (1 - 2 * p)).sum()) / (var ** 1.5)
    z = stats.norm.ppf(tau)
    z_cf = z + (z * z - 1.0) * gamma / 6.0
    return float(np.clip((mu + z_cf * sd) / n, 0.0, 1.0))


def aggregate_to_cells(person_rows: pd.DataFrame, probs: np.ndarray, tau: float,
                       overdispersion: float = 1.0, mode: str = "cornish_fisher") -> pd.DataFrame:
    """از احتمال هر رزرو به کوانتایل نرخ هر سلول $(d,m,r)$."""
    df = person_rows[CELL_KEYS].copy()
    df["p"] = probs
    rows = []
    for keys, g in df.groupby(CELL_KEYS, observed=True):
        p = g["p"].to_numpy()
        if mode == "mean_only":
            q = float(p.mean())
        elif mode == "normal":
            q = poisson_binomial_quantile_rate(p, tau, overdispersion) if len(p) else float("nan")
        else:
            q = poisson_binomial_quantile_rate(p, tau, overdispersion)
        rows.append((*keys, q, len(p), float(p.mean())))
    return pd.DataFrame(rows, columns=[*CELL_KEYS, "rho_q", "n_reservations", "p_mean"])


# ---------------------------------------------------------------------------
# برازش / پیش‌بینی
# ---------------------------------------------------------------------------

def _person_slice(l1_frame: pd.DataFrame, person_path: str) -> pd.DataFrame:
    """رزروهای همان بازه‌ی تاریخی که این fold از L1 دارد — مرز زمانی fold حفظ می‌شود
    (بند 7.16.3: «در L5 تقسیم بر اساس تاریخ نه فرد»)."""
    person = load_person_frame(person_path)
    dates = pd.unique(l1_frame[DATE_COL])
    return person[person[DATE_COL].isin(dates)]


def fit_l5(train: pd.DataFrame, tau: float, *, seed: int = 42,
           person_path: str = str(PERSON_PATH), **hp) -> L5Model:
    import torch
    from torch import nn

    from src.models.gpu_runner import torch_device

    torch.manual_seed(seed)
    device = torch_device()
    rng = np.random.default_rng(seed)

    person = _person_slice(train, person_path)
    prep = PersonPrep().fit(person)
    dense, cats = prep.transform(person)
    y = person["dont_receive"].to_numpy(dtype=np.float32)

    PersonNet = _build_net_cls()
    net = PersonNet(dense.shape[1], prep.cardinalities,
                    emb_dim_k=float(hp.get("emb_dim_k", 2.0)),
                    hidden_dim=int(hp.get("hidden_dim", 128)),
                    n_layers=int(hp.get("n_layers", 2)),
                    dropout=float(hp.get("dropout", 0.1))).to(device)
    n_params = sum(p.numel() for p in net.parameters())

    opt = torch.optim.AdamW(net.parameters(), lr=float(hp.get("learning_rate", 1e-3)),
                            weight_decay=float(hp.get("weight_decay", 1e-5)))
    # ⚠️ داده به‌شدت نامتوازن است (نرخ عدم‌دریافت ~۹٪) — `pos_weight` بدون تنظیم،
    # احتمال‌ها را سیستماتیک بالا می‌برد و تجمیع را خراب می‌کند؛ پیش‌فرض ۱ (بدون
    # وزن‌دهی) و Optuna اجازه دارد آن را تا ۴ ببرد اگر واقعاً کمک کند.
    pos_weight = torch.tensor(float(hp.get("pos_weight", 1.0)), device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    Xd = torch.as_tensor(dense, dtype=torch.float32, device=device)
    Xc = torch.as_tensor(cats, dtype=torch.long, device=device)
    Y = torch.as_tensor(y, dtype=torch.float32, device=device)

    # اعتبارسنجی زمانی داخلی برای زودایست (قاعده‌ی ۳ بند 7.16.3)
    dates = np.array(sorted(pd.unique(person[DATE_COL])))
    cut = max(1, int(len(dates) * 0.8))
    tr_mask = person[DATE_COL].to_numpy() <= dates[cut - 1]
    tr_idx = np.flatnonzero(tr_mask)
    va_idx = np.flatnonzero(~tr_mask)
    if len(va_idx) == 0:
        va_idx = tr_idx[-max(1, len(tr_idx) // 10):]

    batch_size = int(hp.get("batch_size", 8192))
    epochs = int(hp.get("epochs", 12))
    patience = int(hp.get("patience", 3))
    best_val, best_state, bad, history = float("inf"), None, 0, []

    va_t = torch.as_tensor(va_idx, dtype=torch.long, device=device)
    for _ in range(epochs):
        net.train()
        perm = rng.permutation(len(tr_idx))
        for s in range(0, len(perm), batch_size):
            idx = torch.as_tensor(tr_idx[perm[s:s + batch_size]], dtype=torch.long, device=device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(net(Xd[idx], Xc[idx]), Y[idx])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vals = []
            for s in range(0, len(va_idx), 262144):
                vi = va_t[s:s + 262144]
                vals.append(float(loss_fn(net(Xd[vi], Xc[vi]), Y[vi]).item()) * len(vi))
            val = sum(vals) / max(1, len(va_idx))
        history.append(val)
        if val < best_val - 1e-6:
            best_val, bad = val, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)

    arch = {"hidden_dim": int(hp.get("hidden_dim", 128)), "n_layers": int(hp.get("n_layers", 2)),
            "dropout": float(hp.get("dropout", 0.1)), "emb_dim_k": float(hp.get("emb_dim_k", 2.0)),
            "dense_dim": dense.shape[1], "cardinalities": prep.cardinalities,
            "overdispersion": float(hp.get("overdispersion", 1.0)),
            "aggregation_mode": str(hp.get("aggregation_mode", "cornish_fisher")),
            "n_person_rows_train": int(len(person))}
    return L5Model(net, prep, arch, history, n_params, person_path)


def predict_l5(model: L5Model, test: pd.DataFrame, tau: float) -> np.ndarray:
    """احتمال هر رزرو ← کوانتایل نرخ هر سلول ← پخش روی سطرهای L1 همان سلول."""
    person = _person_slice(test, model.person_path)
    if len(person) == 0:
        return np.full(len(test), float(np.nan))
    probs = model.person_probabilities(person)
    cells = aggregate_to_cells(person, probs, tau,
                               overdispersion=model.arch.get("overdispersion", 1.0),
                               mode=model.arch.get("aggregation_mode", "cornish_fisher"))
    left = test[CELL_KEYS].copy()
    left["Meal"] = left["Meal"].astype(str)
    left["RestaurantName"] = left["RestaurantName"].astype(str)
    merged = left.merge(cells, on=CELL_KEYS, how="left")
    out = merged["rho_q"].to_numpy(dtype=float)
    # سلول L1 بدون رزرو فردی ثبت‌شده (باید نادر باشد — F53) ⇒ میانه‌ی همان پیش‌بینی‌ها
    fallback = float(np.nanmedian(out)) if np.isfinite(out).any() else 0.0
    return np.clip(np.where(np.isfinite(out), out, fallback), 0.0, 1.0)


def _save(model: L5Model, stem) -> list:
    import torch

    path = Path(f"{stem}.pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.net.state_dict(), "prep": model.prep, "arch": model.arch,
                "history": model.history, "n_parameters": model.n_parameters,
                "person_path": model.person_path}, path)
    meta = Path(f"{stem}.json")
    meta.write_text(json.dumps({"arch": model.arch, "n_parameters": model.n_parameters,
                                "val_curve": model.history}, ensure_ascii=False, indent=2) + "\n")
    return [path, meta]


def _load(stem) -> L5Model:
    import torch

    blob = torch.load(Path(f"{stem}.pt"), map_location="cpu", weights_only=False)
    arch = blob["arch"]
    PersonNet = _build_net_cls()
    net = PersonNet(arch["dense_dim"], arch["cardinalities"], arch["emb_dim_k"],
                    arch["hidden_dim"], arch["n_layers"], arch["dropout"])
    net.load_state_dict(blob["state_dict"])
    return L5Model(net, blob["prep"], arch, blob["history"], blob["n_parameters"],
                   blob["person_path"])


FITTERS: dict[str, FamilyFitter] = {
    "mlp_embedding_l5": FamilyFitter(
        "mlp_embedding_l5", fit=fit_l5, predict=predict_l5, save=_save, load=_load,
        defaults={"hidden_dim": 128, "n_layers": 2, "dropout": 0.1, "learning_rate": 1e-3,
                  "batch_size": 8192, "epochs": 12, "patience": 3, "emb_dim_k": 2.0,
                  "overdispersion": 1.0, "aggregation_mode": "cornish_fisher"}),
}

MODELS = models_from_fitters(FITTERS)

register(ModelSpec(model_id="mlp_embedding_l5", family=FAMILY, levels=("L5",),
                   quantile_route="Q4",
                   algorithm="torch.nn MLP + nn.Embedding(PersonId) → تجمیع پواسون-دوجمله‌ای"))


@register_space("mlp_embedding_l5", version=1, n_hyperparams=9)
def _space_l5(trial: optuna.Trial) -> dict:
    return {
        "hidden_dim": trial.suggest_int("hidden_dim", 32, 512, log=True),
        "n_layers": trial.suggest_int("n_layers", 1, 4),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "emb_dim_k": trial.suggest_float("emb_dim_k", 1.0, 4.0),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [4096, 8192, 16384]),
        "epochs": trial.suggest_int("epochs", 6, 20),
        # ⭐ اندازه‌گیری همبستگی درون‌روزی، نه یک وصله — بند بالای ماژول
        "overdispersion": trial.suggest_float("overdispersion", 0.5, 6.0, log=True),
    }


QUANTREG_MODEL_IDS: frozenset[str] = frozenset()
TUNING_EXCLUDED: frozenset[str] = frozenset()
