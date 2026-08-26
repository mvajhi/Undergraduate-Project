"""بند 7.8.3 گام ۲ — ساخت بسته‌ی آپلود برای کولب/کگل.

قالب `_template_gpu.ipynb` فرض می‌کند مخزن با ``git clone`` داخل کولب کشیده می‌شود.
این مخزن **remote ندارد** (`git remote -v` خالی است)، پس آن مسیر عملی نیست و
جایگزینش این است: یک zip حاوی `src/` + دقیقاً همان فایل‌های `data/processed/` که
دروازه‌ی انصاف A1 رویشان assert می‌زند، یک‌بار ساخته و روی Drive هر اکانت گذاشته
می‌شود.

قاعده‌ی «`notebooks/` = روایت، `src/` = حقیقت» (`AGENTS.md`) دست‌نخورده می‌ماند: کد
همچنان از `src/` می‌آید، فقط مسیر رسیدنش به کولب zip است نه git.

اجرا:

    python -m src.models.gpu_bundle

خروجی: `gpu_bundle.zip` در ریشه‌ی مخزن + چاپ هش‌هایی که باید در سلول ۳ نوت‌بوک‌ها
باشند (و از قبل در `doc/data_manifest.md` ثبت شده‌اند).
"""

import json
import zipfile
from pathlib import Path

from src.config import ROOT_DIR
from src.cv import sha256_file

BUNDLE_PATH = ROOT_DIR / "gpu_bundle.zip"

#: فایل‌های داده‌ای که هر چهار نوت‌بوک GPU ممکن است بخواهند. `person_features_v1.parquet`
#: (۴۰ مگابایت) فقط برای نوت‌بوک L5 لازم است ولی چون بسته یکی است و برای هر چهار اکانت
#: یکسان، جدا کردنش فقط دستورالعمل را پیچیده می‌کند.
DATA_FILES = (
    "data/processed/features_A_v1.parquet",
    "data/processed/person_features_v1.parquet",
    "data/processed/feature_sets_v1.json",
    "data/processed/cv_folds.json",
)

#: بند رگرسور آینده‌ی `f07_neural_l4.py` — پوشش تقویمی کامل، مستقل از هش دروازه‌ی
#: انصاف (کوچک و ثابت، اسپلیتر رویش assert نمی‌زند)، پس در ``DATA_FILES`` هش‌دار
#: نمی‌رود، فقط باید در بسته حاضر باشد.
CALENDAR_FILE = "data/external/calendar_tehran.csv"

#: ⭐ نتایج CPU خانواده‌های قبلی هم داخل بسته می‌روند — تا جدول «پیچیدگی در برابر بهره»
#: (بند 7.16.4) و مقایسه با قهرمان فعلی (`lightgbm_quantile`) داخل خودِ نوت‌بوک با
#: **عدد واقعی** ساخته شود، نه با عددی که دستی در متن نوت‌بوک کپی شده و کهنه می‌شود.
EXTRA_FILES = (
    "requirements-gpu.txt", "doc/data_manifest.md", "AGENTS.md",
    "reports/phase7/S2_tuning_F02.json", "reports/phase7/dm_test_F02.json",
    "reports/phase7/cqr_champions.json", "reports/phase7/ensemble_champions.json",
)

_SKIP_DIR_PARTS = {"__pycache__", ".ipynb_checkpoints", ".pytest_cache"}


def _iter_src_files(root: Path):
    for p in sorted((root / "src").rglob("*.py")):
        if _SKIP_DIR_PARTS & set(p.parts):
            continue
        yield p


def build_bundle(out: Path = BUNDLE_PATH, root: Path = ROOT_DIR) -> dict:
    hashes = {}
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        n_src = 0
        for p in _iter_src_files(root):
            zf.write(p, arcname=str(p.relative_to(root)))
            n_src += 1
        for rel in DATA_FILES:
            p = root / rel
            if not p.exists():
                raise FileNotFoundError(f"فایل داده‌ی لازم پیدا نشد: {rel} — ابتدا `make data-pull`")
            zf.write(p, arcname=rel)
            hashes[rel] = sha256_file(p)
        for rel in EXTRA_FILES:
            p = root / rel
            if p.exists():
                zf.write(p, arcname=rel)
        cal_p = root / CALENDAR_FILE
        if not cal_p.exists():
            raise FileNotFoundError(f"فایل تقویم لازم پیدا نشد: {CALENDAR_FILE}")
        zf.write(cal_p, arcname=CALENDAR_FILE)
        info = {
            "bundle_for": "phase7 GPU notebooks (بند 7.8)",
            "n_src_files": n_src,
            "data_hashes": hashes,
            "note": "سلول ۳ هر نوت‌بوک این هش‌ها را assert می‌کند — دروازه‌ی انصاف A1، بند 7.7.3",
        }
        zf.writestr("BUNDLE_INFO.json", json.dumps(info, ensure_ascii=False, indent=2) + "\n")

    size_mb = out.stat().st_size / 1e6
    print(f"✅ {out.name} ساخته شد — {n_src} فایل پایتون + {len(DATA_FILES)} فایل داده "
          f"({size_mb:.1f} MB)\n")
    print("هش‌هایی که سلول ۳ هر نوت‌بوک assert می‌کند (همان مقادیر `doc/data_manifest.md`):")
    for rel, h in hashes.items():
        print(f"  {Path(rel).name:<32s} {h}")
    print("\nگام بعد: این فایل را در Google Drive **هر چهار اکانت** بگذارید "
          "(مسیر پیشنهادی: MyDrive/phase7/gpu_bundle.zip)")
    return {"path": str(out), "size_mb": size_mb, **info}


def main() -> None:
    build_bundle()


if __name__ == "__main__":
    main()
