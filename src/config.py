"""Project-wide paths, constants, and the single source of truth for random seeding."""

import random
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_EXTERNAL = DATA_DIR / "external"
DATA_INTERIM = DATA_DIR / "interim"
DATA_PROCESSED = DATA_DIR / "processed"

NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
DOCS_DIR = ROOT_DIR / "doc"

MLFLOW_TRACKING_URI = str(ROOT_DIR / "mlruns")

RANDOM_SEED = 42


def set_global_seed(seed: int = RANDOM_SEED) -> None:
    """Seed every RNG this project may use. Call this first, in every entrypoint (notebook or script)."""
    random.seed(seed)

    import numpy as np

    np.random.seed(seed)

    try:
        import sklearn  # noqa: F401  -- sklearn reads randomness from numpy's global state, already seeded above
    except ImportError:
        pass

    try:
        import lightgbm  # noqa: F401
    except ImportError:
        pass  # lightgbm takes random_state per-call; nothing global to seed here

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    try:
        from src.viz_fa import setup as setup_viz_fa

        setup_viz_fa()
    except Exception:
        pass

