"""بند 7.16 سند فاز ۷ — خانواده‌ی ۷: شبکه‌ی عصبی، **گروه ۷-الف (جدولی) روی L1**.

⚠️ **این ماژول کل بند 7.16 (۲۷ مدل) را پیاده نمی‌کند.** طبق فهرست کوتاه اسپرینت C
(`doc/decisions/37-phase7-rescope.md` بند ۵: «خ۷ شبکه‌ی عصبی: حداکثر ۲ مدل») این‌جا
سه معماری با ذات متفاوت پیاده شده‌اند تا محور «معماری شبکه» واقعاً پیموده شود، نه
یک نقطه‌اش:

| مدل | چه چیزی را می‌آزماید |
|---|---|
| `mlp_quantile` | خط پایه‌ی عصبی روی همان ماتریس یک‌هاتِ خانواده‌های دیگر — «آیا شبکه‌ی ساده چیزی اضافه می‌کند؟» |
| `mlp_entity_embedding` | ⭐ بازنمایی آموخته‌شده به‌جای یک‌هات برای سلف(۳۰)/غذا/وعده/روزهفته — عضو ۲ بند 7.16.1 |
| `ft_transformer` | توجه بین فیچرها (عضو ۴) — گران‌ترین، و صادقانه‌ترین آزمون «آیا پیچیدگی می‌ارزد؟» |

**انتظار پیش از اجرا صریحاً منفی است** (جدول بند 7.16): با ۷٬۵۷۹ رکورد و ۱۴۲ روز،
L1 رژیم شبکه‌ی عصبی نیست و سه شاهد ادبیات (Grinsztajn 2022، Chin 2023، Malefors 2022)
می‌گویند درخت می‌برد. **این اجرا برای رد کردن یک فرضیه است، نه برای بردن** — و طبق
بند 7.16.4 نتیجه هرچه باشد در جدول «پیچیدگی در برابر بهره» با تعداد پارامتر و زمان
آموزش گزارش می‌شود، نه پنهان.

## سه قاعده‌ی بند 7.16.3 که این‌جا در کد اجبار شده‌اند

1. **دسته‌بندی بلوکی روزانه** (`_day_blocks`) — ICC(روز)=۰.۲۲۵ (F10)؛ دسته‌ی تصادفی
   ردیفی یک روز را بین چند دسته پخش می‌کند و گرادیان را با فرض استقلالِ غلط می‌سازد.
2. **نرمال‌سازی فقط با آماره‌ی train fold** (`TabularPrep.fit`) — هیچ آماره‌ای از
   test دیده نمی‌شود، حتی میانه‌ی جای‌گذاری.
3. **اعتبارسنجی زودایست، زمانی نه تصادفی** (`_time_split_inner`) — انتهای پنجره‌ی
   train، هم‌راستا با منطق `conformal._time_split`.

## سر چند-کوانتایلی (بند 7.16.2 ردیف `loss`)

هر شبکه هم‌زمان کل `TAU_GRID` را برمی‌گرداند و زیانش میانگین pinball روی همه‌ی
τهاست. دو سود: (۱) یک برازش، پنج کوانتایل — همان مزیتی که خ۹ داشت؛ (۲) با
`cummax` روی محور τ، **تقاطع کوانتایل ساختاراً ناممکن می‌شود** (بند 7.23.1) نه
اینکه بعداً پس‌پردازش شود.
"""

import json
from functools import lru_cache

import numpy as np
import optuna
import pandas as pd

from src.models.axes import TAU_GRID, TUNING_TAU
from src.models.gpu_runner import FamilyFitter, models_from_fitters
from src.models.registry import ModelSpec, register
from src.models.spaces import register_space

FAMILY = "F07"
LEVEL = "L1"
FEATURE_SET = "FS_F07_nn_v1"

#: τهایی که سر چند-کوانتایلی هم‌زمان یاد می‌گیرد (بند 7.16.2: «τ∈{۰.۰۵,۰.۱,۰.۱۵,۰.۲} هم‌زمان»)
TRAIN_TAUS: tuple[float, ...] = TAU_GRID

#: ستون‌های دسته‌ای که به‌جای یک‌هات، embedding می‌گیرند (بند 7.16.1 عضو ۲)
EMBEDDING_COLS = ("RestaurantName", "FoodType", "Meal", "dow_name", "city", "RestaurantType")


@lru_cache(maxsize=1)
def _feature_cols() -> list[str]:
    """`FS_full_A` — همان فیچرستی که خ۲ (درختی) با آن برنده شد، تا مقایسه‌ی
    «معماری» باشد نه «فیچرست» (بند 7.1.2 قاعده‌ی مقایسه‌پذیری)."""
    from src.features.build import FEATURE_SETS_PATH

    return json.loads(FEATURE_SETS_PATH.read_text())["FS_full_A"]


# ---------------------------------------------------------------------------
# پیش‌پردازش — برازش فقط روی train (قاعده‌ی ۲ بند 7.16.3)
# ---------------------------------------------------------------------------

class TabularPrep:
    """میانه‌گذاری + `RobustScaler` دستی برای عددی‌ها، واژگان صحیح برای دسته‌ای‌ها.

    عمداً دستی و نه `sklearn.Pipeline`: باید کنار وزن‌های شبکه در یک فایل ذخیره و
    بازخوانی شود، و باید هم حالت یک‌هات و هم حالت اندیس-برای-embedding را از یک
    برازش بدهد (`transform_dense` / `transform_split`).

    `RobustScaler` نه `StandardScaler`: پرت‌های تأییدشده‌ی F05 (بند 7.15.5 همین
    استدلال را برای کرنل هم دارد).
    """

    def __init__(self, feature_cols: list[str], embedding_cols: tuple[str, ...] = ()):
        self.feature_cols = list(feature_cols)
        self.embedding_cols = [c for c in embedding_cols if c in self.feature_cols]
        self.cat_cols = [c for c in self.feature_cols if c not in self.embedding_cols]
        self.num_cols: list[str] = []
        self.onehot_cols: list[str] = []
        self.medians: dict = {}
        self.center: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.vocab: dict[str, dict] = {}
        self.onehot_levels: dict[str, list] = {}

    def fit(self, train: pd.DataFrame) -> "TabularPrep":
        obj_cols = [c for c in self.feature_cols if str(train[c].dtype) in ("object", "category")]
        self.embedding_cols = [c for c in self.embedding_cols if c in obj_cols]
        self.onehot_cols = [c for c in obj_cols if c not in self.embedding_cols]
        self.num_cols = [c for c in self.feature_cols if c not in obj_cols]

        num = train[self.num_cols].astype(float)
        self.medians = num.median().to_dict()
        num = num.fillna(pd.Series(self.medians))
        q25, q75 = num.quantile(0.25).to_numpy(), num.quantile(0.75).to_numpy()
        iqr = q75 - q25
        self.center = num.median().to_numpy()
        self.scale = np.where(iqr > 1e-9, iqr, 1.0)

        for c in self.embedding_cols:
            levels = sorted(map(str, pd.unique(train[c].astype(str))))
            self.vocab[c] = {v: i + 1 for i, v in enumerate(levels)}  # ۰ = دسته‌ی دیده‌نشده
        for c in self.onehot_cols:
            self.onehot_levels[c] = sorted(map(str, pd.unique(train[c].astype(str))))
        return self

    @property
    def cardinalities(self) -> list[int]:
        return [len(self.vocab[c]) + 1 for c in self.embedding_cols]

    def _numeric(self, df: pd.DataFrame) -> np.ndarray:
        num = df[self.num_cols].astype(float).fillna(pd.Series(self.medians))
        return ((num.to_numpy() - self.center) / self.scale).astype(np.float32)

    def _onehot(self, df: pd.DataFrame) -> np.ndarray:
        if not self.onehot_cols:
            return np.zeros((len(df), 0), dtype=np.float32)
        blocks = []
        for c in self.onehot_cols:
            vals = df[c].astype(str).to_numpy()
            levels = self.onehot_levels[c]
            block = np.zeros((len(df), len(levels)), dtype=np.float32)
            index = {v: i for i, v in enumerate(levels)}
            for r, v in enumerate(vals):
                j = index.get(v)
                if j is not None:      # دسته‌ی دیده‌نشده ⇒ همه صفر (مثل `common.design_matrix`)
                    block[r, j] = 1.0
            blocks.append(block)
        return np.concatenate(blocks, axis=1)

    def transform_dense(self, df: pd.DataFrame) -> np.ndarray:
        """همه‌چیز عددی/یک‌هات — برای `mlp_quantile` (بدون embedding)."""
        return np.concatenate([self._numeric(df), self._onehot(df)], axis=1)

    def transform_split(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """(عددی+یک‌هات، اندیس دسته‌ای) — برای معماری‌های embedding‌دار."""
        dense = self.transform_dense(df)
        if not self.embedding_cols:
            return dense, np.zeros((len(df), 0), dtype=np.int64)
        cats = np.zeros((len(df), len(self.embedding_cols)), dtype=np.int64)
        for j, c in enumerate(self.embedding_cols):
            vocab = self.vocab[c]
            cats[:, j] = [vocab.get(v, 0) for v in df[c].astype(str)]
        return dense, cats


def _time_split_inner(train: pd.DataFrame, val_frac: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """اعتبارسنجی زودایست از **انتهای زمانی** پنجره‌ی train (قاعده‌ی ۳ بند 7.16.3)."""
    from src.cv import DATE_COL

    dates = np.array(sorted(pd.unique(train[DATE_COL])))
    cut = max(1, int(len(dates) * (1 - val_frac)))
    mask = (train[DATE_COL].to_numpy() <= dates[cut - 1])
    if mask.all() or not mask.any():
        mask = np.ones(len(train), dtype=bool)
        mask[int(len(train) * (1 - val_frac)):] = False
    return mask, ~mask


def _day_blocks(dates: np.ndarray, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    """دسته‌بندی **بلوکی روزانه** (قاعده‌ی ۱ بند 7.16.3): هر روز کامل داخل یک دسته
    می‌ماند؛ روزها به‌هم می‌چسبند تا اندازه‌ی دسته پر شود."""
    uniq = pd.unique(dates)
    order = rng.permutation(len(uniq))
    by_day = {d: np.flatnonzero(dates == d) for d in uniq}
    batches, cur = [], []
    for k in order:
        cur.append(by_day[uniq[k]])
        if sum(len(c) for c in cur) >= batch_size:
            batches.append(np.concatenate(cur))
            cur = []
    if cur:
        batches.append(np.concatenate(cur))
    return batches


# ---------------------------------------------------------------------------
# معماری‌ها
# ---------------------------------------------------------------------------

def _build_modules():
    """ساخت تنبل کلاس‌های torch — تا import این ماژول بدون torch (محیط CPU محلی) نشکند."""
    import torch
    from torch import nn

    class MultiQuantileMLP(nn.Module):
        """MLP با سر چند-کوانتایلی و (اختیاری) entity embedding."""

        def __init__(self, n_dense: int, cardinalities: list[int], emb_dim_k: float,
                     hidden_dim: int, n_layers: int, dropout: float, activation: str,
                     batch_norm: bool, n_taus: int):
            super().__init__()
            self.embeddings = nn.ModuleList()
            emb_total = 0
            for card in cardinalities:
                dim = max(2, min(50, int(np.ceil(card ** 0.25) * emb_dim_k)))
                self.embeddings.append(nn.Embedding(card, dim))
                emb_total += dim

            act = {"relu": nn.ReLU, "gelu": nn.GELU, "selu": nn.SELU}[activation]
            layers: list[nn.Module] = []
            in_dim = n_dense + emb_total
            for _ in range(n_layers):
                layers.append(nn.Linear(in_dim, hidden_dim))
                if batch_norm:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers += [act(), nn.Dropout(dropout)]
                in_dim = hidden_dim
            self.body = nn.Sequential(*layers)
            self.head = nn.Linear(in_dim, n_taus)

        def forward(self, dense, cats):
            parts = [dense]
            for j, emb in enumerate(self.embeddings):
                parts.append(emb(cats[:, j]))
            h = self.body(torch.cat(parts, dim=1) if len(parts) > 1 else dense)
            # ⭐ بند 7.23.1 — تقاطع کوانتایل ساختاراً ناممکن: خروجی = پایه + جمع تجمعی افزایش‌های نامنفی
            raw = self.head(h)
            base = raw[:, :1]
            steps = torch.nn.functional.softplus(raw[:, 1:])
            return torch.cat([base, base + torch.cumsum(steps, dim=1)], dim=1)

    class FTTransformer(nn.Module):
        """FT-Transformer سبک (بند 7.16.1 عضو ۴): هر فیچر یک توکن، یک توکن [CLS]."""

        def __init__(self, n_dense: int, cardinalities: list[int], d_model: int, n_heads: int,
                     n_layers: int, ff_dim: int, dropout: float, n_taus: int):
            super().__init__()
            self.n_dense = n_dense
            self.num_weight = nn.Parameter(torch.randn(n_dense, d_model) * 0.02)
            self.num_bias = nn.Parameter(torch.zeros(n_dense, d_model))
            self.cat_embeddings = nn.ModuleList([nn.Embedding(c, d_model) for c in cardinalities])
            self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                               dim_feedforward=ff_dim, dropout=dropout,
                                               batch_first=True, norm_first=True,
                                               activation="gelu")
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, n_taus)

        def forward(self, dense, cats):
            tokens = [dense.unsqueeze(-1) * self.num_weight + self.num_bias]
            for j, emb in enumerate(self.cat_embeddings):
                tokens.append(emb(cats[:, j]).unsqueeze(1))
            x = torch.cat([self.cls.expand(dense.shape[0], -1, -1)] + tokens, dim=1)
            h = self.norm(self.encoder(x)[:, 0])
            raw = self.head(h)
            base = raw[:, :1]
            steps = torch.nn.functional.softplus(raw[:, 1:])
            return torch.cat([base, base + torch.cumsum(steps, dim=1)], dim=1)

    return MultiQuantileMLP, FTTransformer


def multi_pinball_loss(pred, target, taus, weights=None):
    """میانگین pinball روی همه‌ی τهای سر چند-کوانتایلی."""
    import torch

    e = target.unsqueeze(1) - pred
    loss = torch.maximum(taus.unsqueeze(0) * e, (taus.unsqueeze(0) - 1.0) * e)
    if weights is not None:
        loss = loss * weights.unsqueeze(1)
        return loss.sum() / (weights.sum() * pred.shape[1])
    return loss.mean()


# ---------------------------------------------------------------------------
# برازش/پیش‌بینی مشترک
# ---------------------------------------------------------------------------

class NeuralModel:
    """مدل آموزش‌دیده + پیش‌پردازش + فراداده — همان چیزی که ذخیره و بازخوانی می‌شود."""

    def __init__(self, net, prep: TabularPrep, arch: dict, history: list, n_parameters: int):
        self.net = net
        self.prep = prep
        self.arch = arch
        self.history = history
        self.n_parameters = n_parameters

    def predict_all_taus(self, df: pd.DataFrame) -> np.ndarray:
        import torch

        from src.models.gpu_runner import torch_device

        device = torch_device()
        self.net.eval().to(device)
        dense, cats = self.prep.transform_split(df)
        with torch.no_grad():
            out = self.net(torch.as_tensor(dense, dtype=torch.float32, device=device),
                           torch.as_tensor(cats, dtype=torch.long, device=device))
        return out.cpu().numpy()

    def predict(self, df: pd.DataFrame, tau: float) -> np.ndarray:
        taus = list(self.arch["train_taus"])
        if tau not in taus:
            raise ValueError(f"τ={tau} در سر چند-کوانتایلی این مدل نیست ({taus})")
        return np.clip(self.predict_all_taus(df)[:, taus.index(tau)], 0.0, 1.0)


def _train(net, prep: TabularPrep, train: pd.DataFrame, *, train_taus, learning_rate: float,
           batch_size: int, epochs: int, patience: int, weight_decay: float, optimizer: str,
           scheduler: str, grad_clip: float, weighting: str, seed: int):
    import torch
    from torch import nn  # noqa: F401  — لازم برای ثبت پارامترها روی دستگاه

    from src.cv import DATE_COL
    from src.models.gpu_runner import torch_device

    device = torch_device()
    net = net.to(device)
    rng = np.random.default_rng(seed)

    tr_mask, va_mask = _time_split_inner(train)
    dense, cats = prep.transform_split(train)
    y = train["rho"].to_numpy(dtype=np.float32)
    w = (train["Res"].to_numpy(dtype=np.float32) if weighting == "res"
         else np.sqrt(train["Res"].to_numpy(dtype=np.float32)) if weighting == "sqrt_res"
         else None)

    to_t = lambda a, dt: torch.as_tensor(a, dtype=dt, device=device)  # noqa: E731
    Xd, Xc, Y = to_t(dense, torch.float32), to_t(cats, torch.long), to_t(y, torch.float32)
    W = to_t(w, torch.float32) if w is not None else None
    taus_t = to_t(np.asarray(train_taus, dtype=np.float32), torch.float32)

    opt_cls = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW,
               "rmsprop": torch.optim.RMSprop}[optimizer]
    opt = opt_cls(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    sched = None
    if scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    elif scheduler == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=max(2, patience // 3))

    tr_idx = np.flatnonzero(tr_mask)
    va_idx = np.flatnonzero(va_mask)
    tr_dates = train[DATE_COL].to_numpy()[tr_idx]
    best_val, best_state, bad, history = float("inf"), None, 0, []

    for epoch in range(epochs):
        net.train()
        for block in _day_blocks(tr_dates, batch_size, rng):
            idx = to_t(tr_idx[block], torch.long)
            opt.zero_grad(set_to_none=True)
            pred = net(Xd[idx], Xc[idx])
            loss = multi_pinball_loss(pred, Y[idx], taus_t, W[idx] if W is not None else None)
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
            opt.step()

        net.eval()
        with torch.no_grad():
            vi = to_t(va_idx, torch.long)
            val = float(multi_pinball_loss(net(Xd[vi], Xc[vi]), Y[vi], taus_t).item()) \
                if len(va_idx) else float(loss.item())
        history.append(val)
        if sched is not None:
            sched.step(val) if scheduler == "plateau" else sched.step()

        if val < best_val - 1e-7:
            best_val, bad = val, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        net.load_state_dict(best_state)
    return net, history


def _fit_generic(kind: str, train: pd.DataFrame, tau: float, *, seed: int = 42,
                 weighting: str = "none", **hp) -> NeuralModel:
    import torch

    MultiQuantileMLP, FTTransformer = _build_modules()
    torch.manual_seed(seed)

    train_taus = tuple(TRAIN_TAUS) if tau in TRAIN_TAUS else (tau,)
    if kind == "ft":
        # FT-Transformer هر فیچر را یک توکن می‌بیند؛ یک‌هات‌کردن دسته‌ای‌ها آن‌ها را به
        # ده‌ها توکن صفر/یک می‌شکند و کل ایده‌ی «توکن فیچر» را خراب می‌کند ⇒ همه‌ی
        # دسته‌ای‌ها embedding می‌گیرند، نه فقط شش‌تای فهرست EMBEDDING_COLS.
        emb_cols = tuple(c for c in _feature_cols()
                         if str(train[c].dtype) in ("object", "category"))
    elif kind == "embedding":
        emb_cols = EMBEDDING_COLS
    else:
        emb_cols = ()
    prep = TabularPrep(_feature_cols(), emb_cols).fit(train)
    dense_dim = prep.transform_dense(train.head(1)).shape[1]

    if kind == "ft":
        net = FTTransformer(dense_dim, prep.cardinalities,
                            d_model=int(hp.get("d_model", 64)), n_heads=int(hp.get("n_heads", 4)),
                            n_layers=int(hp.get("n_blocks", 2)), ff_dim=int(hp.get("ff_dim", 128)),
                            dropout=float(hp.get("dropout", 0.1)), n_taus=len(train_taus))
        arch = {"kind": kind, "d_model": int(hp.get("d_model", 64)),
                "n_heads": int(hp.get("n_heads", 4)), "n_blocks": int(hp.get("n_blocks", 2)),
                "ff_dim": int(hp.get("ff_dim", 128))}
    else:
        net = MultiQuantileMLP(dense_dim, prep.cardinalities,
                               emb_dim_k=float(hp.get("emb_dim_k", 2.0)),
                               hidden_dim=int(hp.get("hidden_dim", 128)),
                               n_layers=int(hp.get("n_layers", 2)),
                               dropout=float(hp.get("dropout", 0.1)),
                               activation=str(hp.get("activation", "relu")),
                               batch_norm=bool(hp.get("batch_norm", True)),
                               n_taus=len(train_taus))
        arch = {"kind": kind, "hidden_dim": int(hp.get("hidden_dim", 128)),
                "n_layers": int(hp.get("n_layers", 2)),
                "activation": str(hp.get("activation", "relu")),
                "batch_norm": bool(hp.get("batch_norm", True)),
                "emb_dim_k": float(hp.get("emb_dim_k", 2.0))}

    arch.update({"train_taus": list(train_taus), "dropout": float(hp.get("dropout", 0.1)),
                 "dense_dim": dense_dim, "cardinalities": prep.cardinalities})
    n_params = sum(p.numel() for p in net.parameters())

    net, history = _train(net, prep, train, train_taus=train_taus,
                          learning_rate=float(hp.get("learning_rate", 1e-3)),
                          batch_size=int(hp.get("batch_size", 256)),
                          epochs=int(hp.get("epochs", 120)),
                          patience=int(hp.get("patience", 15)),
                          weight_decay=float(hp.get("weight_decay", 1e-4)),
                          optimizer=str(hp.get("optimizer", "adamw")),
                          scheduler=str(hp.get("scheduler", "cosine")),
                          grad_clip=float(hp.get("grad_clip", 1.0)),
                          weighting=weighting, seed=seed)
    return NeuralModel(net, prep, arch, history, n_params)


def _predict(model: NeuralModel, test: pd.DataFrame, tau: float) -> np.ndarray:
    return model.predict(test, tau)


def _save(model: NeuralModel, stem) -> list:
    """وزن‌ها + پیش‌پردازش + معماری در **یک** فایل `.pt` — بدون پیش‌پردازش، وزن‌ها
    بی‌فایده‌اند (مقیاس‌بندی و واژگان دسته‌ای بخشی از مدل‌اند، نه جانبی)."""
    import torch

    from pathlib import Path

    path = Path(f"{stem}.pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.net.state_dict(), "prep": model.prep, "arch": model.arch,
                "history": model.history, "n_parameters": model.n_parameters}, path)
    meta = Path(f"{stem}.json")
    meta.write_text(json.dumps({"arch": model.arch, "n_parameters": model.n_parameters,
                                "val_curve": model.history}, ensure_ascii=False, indent=2) + "\n")
    return [path, meta]


def _load(stem) -> NeuralModel:
    """بازخوانی کامل — همان API `predict` را می‌دهد، پس مدل ذخیره‌شده واقعاً قابل استفاده است."""
    import torch

    from pathlib import Path

    blob = torch.load(Path(f"{stem}.pt"), map_location="cpu", weights_only=False)
    MultiQuantileMLP, FTTransformer = _build_modules()
    arch, prep = blob["arch"], blob["prep"]
    n_taus = len(arch["train_taus"])
    if arch["kind"] == "ft":
        net = FTTransformer(arch["dense_dim"], arch["cardinalities"], arch["d_model"],
                            arch["n_heads"], arch["n_blocks"], arch["ff_dim"],
                            arch["dropout"], n_taus)
    else:
        net = MultiQuantileMLP(arch["dense_dim"], arch["cardinalities"], arch["emb_dim_k"],
                               arch["hidden_dim"], arch["n_layers"], arch["dropout"],
                               arch["activation"], arch["batch_norm"], n_taus)
    net.load_state_dict(blob["state_dict"])
    return NeuralModel(net, prep, arch, blob["history"], blob["n_parameters"])


# ---------------------------------------------------------------------------
# رجیستری مدل‌ها
# ---------------------------------------------------------------------------

FITTERS: dict[str, FamilyFitter] = {
    "mlp_quantile": FamilyFitter(
        "mlp_quantile",
        fit=lambda tr, tau, **hp: _fit_generic("plain", tr, tau, **hp),
        predict=_predict, save=_save, load=_load,
        defaults={"hidden_dim": 128, "n_layers": 2, "dropout": 0.1, "learning_rate": 1e-3,
                  "batch_size": 256, "epochs": 120, "patience": 15}),
    "mlp_entity_embedding": FamilyFitter(
        "mlp_entity_embedding",
        fit=lambda tr, tau, **hp: _fit_generic("embedding", tr, tau, **hp),
        predict=_predict, save=_save, load=_load,
        defaults={"hidden_dim": 128, "n_layers": 2, "dropout": 0.1, "learning_rate": 1e-3,
                  "batch_size": 256, "epochs": 120, "patience": 15, "emb_dim_k": 2.0}),
    "ft_transformer": FamilyFitter(
        "ft_transformer",
        fit=lambda tr, tau, **hp: _fit_generic("ft", tr, tau, **hp),
        predict=_predict, save=_save, load=_load,
        defaults={"d_model": 64, "n_heads": 4, "n_blocks": 2, "ff_dim": 128, "dropout": 0.1,
                  "learning_rate": 1e-3, "batch_size": 256, "epochs": 100, "patience": 12}),
}

MODELS = models_from_fitters(FITTERS)

_ALGORITHMS = {
    "mlp_quantile": "torch.nn MLP (multi-quantile head)",
    "mlp_entity_embedding": "torch.nn MLP + nn.Embedding (entity embedding)",
    "ft_transformer": "torch.nn FT-Transformer (feature tokenizer + encoder)",
}

for _mid, _algo in _ALGORITHMS.items():
    register(ModelSpec(model_id=_mid, family=FAMILY, levels=(LEVEL,), quantile_route="Q1",
                       algorithm=_algo))


# ---------------------------------------------------------------------------
# فضای هایپرپارامتر — بند 7.16.2
# ---------------------------------------------------------------------------

def _common_space(trial: optuna.Trial) -> dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        # ⚠️ دامنه‌ی دسته عمداً از ۶۴ شروع می‌شود نه ۳۲: دسته **بلوکی روزانه** است و
        # یک روز L1 حدود ۵۳ ردیف دارد (F10) — دسته‌ی کوچک‌تر از یک روز بی‌معناست.
        "batch_size": trial.suggest_int("batch_size", 64, 1024, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True),
        "dropout": trial.suggest_float("dropout", 0.0, 0.6),
        "optimizer": trial.suggest_categorical("optimizer", ["adam", "adamw", "rmsprop"]),
        "scheduler": trial.suggest_categorical("scheduler", ["none", "cosine", "plateau"]),
        "grad_clip": trial.suggest_float("grad_clip", 0.1, 5.0, log=True),
        "epochs": trial.suggest_int("epochs", 40, 250),
        "patience": trial.suggest_int("patience", 5, 30),
    }


@register_space("mlp_quantile", version=1, n_hyperparams=13)
def _space_mlp(trial: optuna.Trial) -> dict:
    return {**_common_space(trial),
            "n_layers": trial.suggest_int("n_layers", 1, 5),
            "hidden_dim": trial.suggest_int("hidden_dim", 16, 512, log=True),
            "activation": trial.suggest_categorical("activation", ["relu", "gelu", "selu"]),
            "batch_norm": trial.suggest_categorical("batch_norm", [True, False])}


@register_space("mlp_entity_embedding", version=1, n_hyperparams=14)
def _space_mlp_emb(trial: optuna.Trial) -> dict:
    return {**_space_mlp(trial),
            "emb_dim_k": trial.suggest_float("emb_dim_k", 1.0, 4.0)}


@register_space("ft_transformer", version=1, n_hyperparams=14)
def _space_ft(trial: optuna.Trial) -> dict:
    d_model = trial.suggest_categorical("d_model", [16, 32, 64, 128, 256])
    n_heads = trial.suggest_categorical("n_heads", [1, 2, 4, 8])
    while d_model % n_heads != 0:          # محدودیت ساختاری MultiheadAttention
        n_heads = max(1, n_heads // 2)
    return {**_common_space(trial), "d_model": d_model, "n_heads": n_heads,
            "n_blocks": trial.suggest_int("n_blocks", 1, 4),
            "ff_dim": trial.suggest_int("ff_dim", 32, 512, log=True)}


QUANTREG_MODEL_IDS: frozenset[str] = frozenset()
TUNING_EXCLUDED: frozenset[str] = frozenset()


def _design_s2(train, test, quantreg: bool = False):
    """سازگاری با `s2_runner` (که پیش‌طراحی ماتریس می‌کند). این خانواده پیش‌پردازشش را
    **داخل** برازش و به‌ازای هر fold انجام می‌دهد (چون آماره‌ها باید فقط از train همان
    fold بیایند)، پس این‌جا فقط عبور می‌دهد."""
    return train, test
