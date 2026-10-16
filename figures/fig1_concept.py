"""Paper Fig. 1 (intro) -- Semantic materialization and the donor comparison. README: assets/concept.png.

Pure schematic (no data). All prose lives in the paper caption, not the image.
  Panel A  write time: the root event attends the source at prefill, so its KV rows
           absorb S's state -- it becomes a materialized view of S.
  Panel B  serve time: drop the source, serve only the root + query; a donor pair
           holds every served token/position fixed and flips only the deleted source.
  Panel C  the reference chain: source -> root -> edge.

Run:  python fig1_concept.py              ->  ../assets/concept.png
      FIG_THEME=dark python fig1_concept.py  ->  ../assets/concept-dark.png"""
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.pyplot as plt
from _paper_style import (
    save, panel_title, C_BG, C_INK, C_MUTED, C_KV, C_KV_TEXT, C_TX, C_POS, C_NEG,
    C_POS_TEXT, C_NEG_TEXT, C_SRC, C_CAR, C_FILL, C_BOX_INK,
)


def box(ax, x, y, w, h, fc, ec=C_INK, lw=0.8, text="", fs=7.4, tc=None, radius=0.08):
    """A rounded event box. White boxes take the page background, so they turn dark in the
    dark theme; words on the light event fills stay dark in both themes."""
    if fc == "white":
        fc, ink = C_BG, C_INK
    else:
        ink = C_BOX_INK
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0.03,rounding_size={radius}",
                                fc=fc, ec=ec, lw=lw))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                color=tc or ink)


def arrow(ax, p0, p1, color=C_INK, lw=0.9, rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=9, lw=lw,
                                 color=color, connectionstyle=f"arc3,rad={rad}"))


fig = plt.figure(figsize=(7.35, 3.15))
gs = fig.add_gridspec(2, 2, width_ratios=[0.94, 1.40],
                      height_ratios=[2.35, 0.95], wspace=0.10, hspace=0.42)
fig.subplots_adjust(left=0.015, right=0.995, top=0.87, bottom=0.04)

# ---------------- Panel A: write time ----------------
ax = fig.add_subplot(gs[0, 0]); ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
panel_title(ax, "A", "Write: one global prefill", pad=5)

# Event stream: the root attends backward to the source.
specs = [
    (0.18, 1.10, C_FILL, C_MUTED, "$e_1$"),
    (1.48, 1.90, C_SRC, C_INK, "source\n$S = \\mathrm{ON}$"),
    (3.58, 1.10, C_FILL, C_MUTED, "$e_3$"),
    (4.88, 2.35, C_CAR, C_KV, "root\n$M$ mirrors $S$"),
    (7.43, 1.10, C_FILL, C_MUTED, "$e_5$"),
]
y0, bh = 2.35, 1.45
for x, w, fc, ec, label in specs:
    box(ax, x, y0, w, bh, fc, ec=ec, text=label, fs=7.5)
xs = 1.48 + 1.90 / 2
xr = 4.88 + 2.35 / 2
arrow(ax, (xr, y0 + bh + 0.02), (xs, y0 + bh + 0.02), color=C_KV, lw=1.15, rad=0.35)
ax.text((xs + xr) / 2, 4.95, "write-time attention", ha="center", fontsize=7.0,
        color=C_KV_TEXT, fontweight="semibold")

# The resulting memory row is the object reused later.
arrow(ax, (xr, y0 - 0.02), (xr, 1.38), color=C_KV, lw=0.95)
box(ax, 4.30, 0.24, 3.50, 0.88, "white", ec=C_KV, lw=1.1,
    text="root KV:  $M = \\mathrm{ON}$", fs=7.6, tc=C_KV_TEXT)
ax.text(xr + 0.22, 1.47, "materializes", ha="left", va="bottom", fontsize=6.6,
        color=C_MUTED)

# ---------------- Panel B: serve time + donor pair ----------------
ax = fig.add_subplot(gs[0, 1]); ax.set_xlim(0, 13.7); ax.set_ylim(0, 6); ax.axis("off")
panel_title(ax, "B", "Read: source omitted; only donor KV changes", pad=5)

ax.text(5.40, 5.02, "same served tokens + positions", ha="center", fontsize=6.8,
        color=C_MUTED)


def served_row(ax, y, srcstate, memory, ans, ans_col, ans_text):
    # Omitted source (shown only to identify the paired donor).
    box(ax, 0.25, y, 1.60, 1.22, "white", ec=C_SRC, lw=0.9,
        text="$S = \\mathrm{" + srcstate + "}$", fs=6.9, tc=C_MUTED)
    ax.plot([0.40, 1.70], [y + 1.05, y + 0.17], color=C_NEG, lw=1.25)
    ax.text(1.05, y - 0.32, "omitted", ha="center", fontsize=6.0, color=C_NEG_TEXT)

    # Root text is fixed; only its harvested KV differs between donors.
    box(ax, 2.65, y, 4.25, 1.22, C_CAR, ec=C_KV, lw=0.85)
    ax.text(4.78, y + 0.82, "root text:  $M$ mirrors $S$", ha="center", va="center",
            fontsize=6.9, color=C_BOX_INK)
    ax.text(4.78, y + 0.34, "KV:  $M = \\mathrm{" + memory + "}$", ha="center",
            va="center", fontsize=7.0, color=C_KV, fontweight="semibold")

    box(ax, 7.45, y, 1.45, 1.22, "white", ec=C_INK, lw=0.8, text="$M$?", fs=7.8)
    arrow(ax, (9.08, y + 0.61), (9.83, y + 0.61), color=C_INK, lw=0.9)
    box(ax, 10.00, y, 2.72, 1.22, "white", ec=ans_col, lw=1.25,
        text=ans, fs=7.7, tc=ans_text)


served_row(ax, 3.35, "ON", "ON", "ONLINE", C_POS, C_POS_TEXT)
served_row(ax, 1.00, "OFF", "OFF", "OFFLINE", C_NEG, C_NEG_TEXT)
ax.plot([2.28, 2.28], [1.00, 4.57], color=C_TX, lw=0.9)
ax.plot([2.10, 2.28], [1.00, 1.00], color=C_TX, lw=0.9)
ax.plot([2.10, 2.28], [4.57, 4.57], color=C_TX, lw=0.9)
ax.text(2.43, 2.78, "donor pair", ha="center", va="center", rotation=90,
        fontsize=6.2, color=C_MUTED)

# ---------------- Panel C: the reference chain (source -> root -> edge) ----------------
axc = fig.add_subplot(gs[1, :]); axc.set_xlim(0, 14); axc.set_ylim(0, 2); axc.axis("off")
panel_title(axc, "C", "The reference chain: how far each event sits from the source", pad=3)
ccy, cch = 0.50, 1.10
box(axc, 0.55, ccy, 2.75, cch, C_SRC, ec=C_INK, text="source  $e_{\\mathrm{src}}$\nsets the value", fs=6.9)
box(axc, 5.30, ccy, 2.75, cch, C_CAR, ec=C_KV, text="root  $e_{\\mathrm{root}}$\n$M$ mirrors $S$", fs=6.9)
box(axc, 10.05, ccy, 2.75, cch, "white", ec=C_INK, text="edge  $e_{\\mathrm{edge}}$\n$T$ mirrors $M$", fs=6.9)
ymid = ccy + cch / 2
arrow(axc, (5.30, ymid), (3.36, ymid), color=C_MUTED, lw=1.0, rad=-0.22)
arrow(axc, (10.05, ymid), (8.11, ymid), color=C_MUTED, lw=1.0, rad=-0.22)
axc.text(4.32, ymid + 0.28, "mirrors", ha="center", fontsize=6.3, color=C_MUTED)
axc.text(9.07, ymid + 0.28, "mirrors", ha="center", fontsize=6.3, color=C_MUTED)
# Labels hang from just under the boxes (va="top"), so their descenders stay inside the frame.
axc.text(6.675, ccy - 0.08, "carries the value", ha="center", va="top", fontsize=6.4, color=C_POS_TEXT)
axc.text(11.425, ccy - 0.08, "routes to the root", ha="center", va="top", fontsize=6.4, color=C_MUTED)

save(fig, "fig1_concept")
