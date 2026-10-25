"""Palette and type of the paper's figures, used to draw the README copies of Figs. 1, 4 and 5.

FIG_THEME=dark switches only the background, text and axis colours. The data colours stay the
paper's, and words drawn on the light event boxes stay dark in both themes.

FIG_LANG=zh draws the Chinese copy used by the Chinese README (assets/<asset>-zh.png and
-zh-dark.png): every label goes through L(en, zh), Latin text stays in DejaVu Sans and Chinese
falls back to the first installed CJK font of _style.CJK."""
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.offsetbox import AnnotationBbox, DrawingArea, HPacker, TextArea, VPacker  # noqa: E402

from _style import CJK, _bold_cjk  # noqa: E402

THEME = os.environ.get("FIG_THEME", "light")
DARK = THEME == "dark"
LANG = os.environ.get("FIG_LANG", "en")
if LANG not in ("en", "zh"):
    raise SystemExit(f"FIG_LANG must be en or zh, not {LANG!r}")
ZH = LANG == "zh"

# ---- data palette (the paper's; the same in both themes) ----
C_KV = "#2166AC"     # KV / candidate-logit / primary blue
C_TX = "#7A828E"     # text / free-generation grey
C_POS = "#14866D"    # donor follow / positive teal
C_NEG = "#C44E3F"    # anti / chance / deficit vermillion
C_SRC = "#F6C66E"    # source event amber
C_CAR = "#CFE3F4"    # carrier / root event light blue
C_FILL = "#F1F3F5"   # filler event grey
C_ARM = {
    "pas": "#B8C2CC",
    "bind": "#9CC7E4",
    "dir": "#5FA0CF",
    "nld": C_KV,
    "exp": C_POS,
}

# ---- theme: background, text and axes ----
C_BG = "#0d1117" if DARK else "#ffffff"
C_INK = "#e6edf3" if DARK else "#243447"      # primary text / strokes
C_MUTED = "#9198a1" if DARK else "#667085"    # secondary labels
C_GRID = "#3d444d" if DARK else "#D9E2EC"     # subtle guides
C_BOX_INK = "#243447"                         # words on the light event fills
C_KV_TEXT = "#79b8ff" if DARK else C_KV       # blue words, lightened on the dark background
C_POS_TEXT = "#3fc1a0" if DARK else C_POS     # teal words, lightened on the dark background
C_NEG_TEXT = "#f0806f" if DARK else C_NEG     # vermillion words, lightened on the dark background

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
    "font.size": 8.2,
    "axes.titlesize": 9.2,
    "axes.titleweight": "semibold",
    "axes.labelsize": 8.5,
    "axes.labelcolor": C_INK,
    "axes.edgecolor": C_INK,
    "axes.linewidth": 0.7,
    "axes.facecolor": C_BG,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "xtick.color": C_INK,
    "ytick.color": C_INK,
    "xtick.major.width": 0.65,
    "ytick.major.width": 0.65,
    "legend.fontsize": 7.0,
    "text.color": C_INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": C_BG,
    "savefig.facecolor": C_BG,
    "savefig.pad_inches": 0.06,
    "figure.dpi": 150,
})
if ZH:
    # DejaVu Sans first, so Latin text, digits and model names look as in the English copy; Chinese
    # glyphs fall back to the CJK fonts. _bold_cjk registers the bold face for semibold/bold titles.
    _bold_cjk()
    _have = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams["font.family"] = ["DejaVu Sans"] + [f for f in CJK if f in _have]

HERE = os.path.dirname(os.path.abspath(__file__))
DATADIR = os.path.join(HERE, "data")
ASSETS = os.path.join(HERE, "..", "assets")
# script figure name -> README asset name
ASSET_NAME = {"fig1_concept": "concept", "fig4_access": "envelope", "fig5_active": "active"}


def L(en, zh):
    """The label in the language FIG_LANG selects."""
    return zh if ZH else en


def save(fig, name):
    """Write assets/<asset>[-zh][-dark].png, as FIG_LANG and FIG_THEME select."""
    os.makedirs(ASSETS, exist_ok=True)
    path = os.path.join(ASSETS, ASSET_NAME.get(name, name) + ("-zh" if ZH else "") + ("-dark" if DARK else "")
                        + ".png")
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("wrote", os.path.relpath(path, os.path.join(HERE, "..")))


_HAN = re.compile(r"[\u2e80-\u9fff\uf900-\ufaff\uff00-\uffef]")   # CJK characters and full-width punctuation


def mixed_text(ax, x, y, s, ha="left", va="baseline", **props):
    """ax.text, except for a line that mixes Chinese with $math$. Mathtext has no font fallback, so
    Chinese inside a math line would come out as boxes; such a line is drawn as pieces (Chinese text,
    mathtext) packed side by side on one baseline. English text never takes that path."""
    lines = s.split("\n")
    if not any("$" in ln and _HAN.search(ln) for ln in lines):
        return ax.text(x, y, s, ha=ha, va=va, **props)
    fs = props.get("fontsize", plt.rcParams["font.size"])
    space = 0.32 * fs                                   # a DejaVu Sans space, in points
    rows = []
    for ln in lines:
        # Chinese runs stand alone; everything else (math and the Latin words between) stays one mathtext.
        pieces = []
        for tok in (t for t in re.split(r"(\$[^$]*\$)", ln) if t):
            if _HAN.search(tok) or not pieces or _HAN.search(pieces[-1]):
                pieces.append(tok)
            else:
                pieces[-1] += tok
        children = []
        for i, p in enumerate(pieces):
            # the spaces where two pieces meet become a fixed gap
            gap = (len(pieces[i - 1]) - len(pieces[i - 1].rstrip(" ")) + len(p) - len(p.lstrip(" "))) if i else 0
            if gap:
                children.append(DrawingArea(gap * space, 0))
            children.append(TextArea(p.strip(" "), textprops=props))
        rows.append(HPacker(children=children, align="baseline", pad=0, sep=0))
    align = {"left": "left", "center": "center", "right": "right"}[ha]
    box = VPacker(children=rows, align=align, pad=0, sep=0.16 * fs)
    ab = AnnotationBbox(box, (x, y), xycoords="data", frameon=False, pad=0,
                        box_alignment=({"left": 0, "center": 0.5, "right": 1}[ha],
                                       {"bottom": 0, "baseline": 0, "center": 0.5, "top": 1}[va]))
    ax.add_artist(ab)
    return ab


def panel_title(ax, letter, title, pad=7):
    """Consistent left-aligned panel heading."""
    ax.set_title(f"{letter}  {title}", loc="left", pad=pad, fontweight="semibold")


def light_ygrid(ax):
    """A quiet horizontal grid that stays behind marks."""
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=C_GRID, lw=0.55, alpha=0.8)


def light_xgrid(ax):
    """A quiet vertical grid for horizontal effect plots."""
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=C_GRID, lw=0.55, alpha=0.8)
