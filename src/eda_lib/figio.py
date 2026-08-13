"""ذخیره‌ی نمودار — یک پیاده‌سازی واحد برای کل فاز ۴.

پیش از این، چهار ماژول کمکی (`univariate_`, `correlation_`, `timeseries_`,
`outlier_helpers`) هرکدام نسخه‌ی خودشان از `save_fig` را داشتند و هر چهار نسخه یک باگ
مشترک داشتند: نام‌گذاری این پروژه با شماره‌ی بند WBS شروع می‌شود (`4.3_acf`)، پس
`Path.suffix` مقدار `.3_acf` می‌دهد و matplotlib آن را «فرمت تصویر» می‌فهمد و
`ValueError: Format '3_acf' is not supported` می‌دهد. اینجا یک‌بار درست شده و بقیه
فقط همین را re-export می‌کنند.
"""

from pathlib import Path

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".pdf", ".svg", ".webp", ".tif", ".tiff"}


def save_fig(fig, name: str, figures_dir: Path | str, dpi: int = 150) -> Path:
    """شکل را در `figures_dir` ذخیره می‌کند؛ اگر نام پسوند تصویری نداشته باشد `.png` می‌گیرد."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    if Path(name).suffix.lower() not in _IMAGE_SUFFIXES:
        name = f"{name}.png"
    out_path = figures_dir / name
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    return out_path
