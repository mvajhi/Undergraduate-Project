"""Persian (Farsi) text support for matplotlib figures used in EDA/report plots.

matplotlib does not perform Arabic/Persian glyph shaping on its own.
Importing this module or calling `setup()` configures the font and auto-patches
matplotlib so that any Persian text (titles, labels, legends) renders correctly.
"""

import re
from pathlib import Path

import arabic_reshaper
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.text as mtext

_FONT_PATH = Path.home() / ".local/share/fonts/shamsiCalendarFonts/Vazirmatn.ttf"


class ShapedText(str):
    """Custom str subclass to mark text that has already been reshaped."""

    pass


def reshape_persian(text: str) -> str:
    """Safely reshape Persian text using arabic_reshaper, preserving LaTeX math expressions like $\\rho$."""
    if not isinstance(text, str) or isinstance(text, ShapedText):
        return text

    # If no Persian/Arabic characters, return as is
    if not re.search(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]', text):
        return text

    # Handle LaTeX math blocks like $...$ separately so they aren't corrupted
    parts = re.split(r'(\$.*?\$)', text)
    processed = []
    for part in parts:
        if part.startswith('$') and part.endswith('$'):
            processed.append(part)
        elif re.search(r'[\u0600-\u06FF]', part):
            processed.append(arabic_reshaper.reshape(part))
        else:
            processed.append(part)

    return ShapedText("".join(processed))


_original_set_text = mtext.Text.set_text
_is_patched = False


def _patched_set_text(self, s):
    if s is not None and isinstance(s, str) and not isinstance(s, ShapedText):
        s = reshape_persian(s)
    _original_set_text(self, s)


def setup() -> None:
    """Register the Vazirmatn font with matplotlib, make it default with fallback, and patch text rendering."""
    global _is_patched

    font_name = "Vazirmatn"
    if _FONT_PATH.exists():
        try:
            fm.fontManager.addfont(str(_FONT_PATH))
            font_prop = fm.FontProperties(fname=str(_FONT_PATH))
            font_name = font_prop.get_name()
        except Exception:
            pass

    # Use 'sans-serif' family so matplotlib falls back to DejaVu Sans for missing glyphs (like Greek rho)
    plt.rcParams["font.family"] = "sans-serif"
    current_sans = list(plt.rcParams.get("font.sans-serif", []))
    if font_name in current_sans:
        current_sans.remove(font_name)
    plt.rcParams["font.sans-serif"] = [font_name] + current_sans
    plt.rcParams["axes.unicode_minus"] = False

    if not _is_patched:
        mtext.Text.set_text = _patched_set_text
        _is_patched = True


def fa(text: str) -> str:
    """Reshape a Persian/Arabic string for correct matplotlib rendering."""
    return reshape_persian(text)


# Auto-apply setup on module import
try:
    setup()
except Exception:
    pass


