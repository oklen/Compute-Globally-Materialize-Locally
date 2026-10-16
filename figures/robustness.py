"""README figure: the donor-pair direction never reverses.

For every serving cell, model and readout the paper (v2) reports, the share of donor-sensitive
pairs whose answer follows the omitted source's value, with a Wilson 95% interval, and the
follow:anti count. Sources are line numbers in the v2 LaTeX source (materialize.tex).

Run:  python robustness.py   ->  ../assets/robustness.png and robustness-dark.png"""
import os

import matplotlib.pyplot as plt

from _style import BLUE, save, use

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets")

# (model, cell, follow, anti, source in materialize.tex v2)
ROWS = [
    ("Qwen3-8B", "root only (headline)", 99, 0, "Table 1 / L208, L221"),
    ("Qwen3-8B", "256-pair replication", 130, 0, "L222-223, L1897-1899"),
    ("Qwen3-8B", "mention absent", 52, 1, "Table 1 / L209"),
    ("Qwen3-8B", "mention co-served (narrowest cell)", 39, 18, "L224-225, L1905-1906"),
    ("Qwen3-8B", "menu readout", 88, 0, "L225-226, L1908"),
    ("Qwen3-8B", "asked through a reference edge (menu)", 84, 0, "tab:gen / L1040"),
    ("Ministral-3-8B", "mention absent", 90, 1, "Table 1 / L210"),
    ("Ministral-3-8B", "asked through a reference edge", 298, 0, "L342, L1041"),
    ("Gemma-4-12B", "mention co-served", 80, 0, "Table 1 / L211"),
    ("Gemma-4-12B", "mention absent", 17, 0, "Fig. 2A (fig3_composition.pdf, included at L323)"),
    ("Gemma-4-12B", "asked through a reference edge", 41, 0, "L332, L1042"),
]


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def draw(theme):
    t = use(theme)
    # One y slot per row, plus a header slot above each model's group.
    labels, ys, header_y, y = [], [], [], 0.0
    layout = []
    last = None
    for model, cell, f, a, _ in ROWS:
        if model != last:
            if last is not None:
                y += 0.5
            header_y.append((y, model))
            y += 1
            last = model
        layout.append((y, cell, f, a))
        y += 1
    total = y

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    fig.subplots_adjust(left=0.36, right=0.87, top=0.82, bottom=0.12)
    for yy, cell, f, a in layout:
        p, lo, hi = wilson(f, f + a)
        ax.plot([lo, hi], [yy, yy], color=BLUE, lw=1.6, solid_capstyle="round", zorder=2)
        ax.scatter([p], [yy], s=42, color=BLUE, edgecolor=t["bg"], linewidth=1.5, zorder=3)
        bold = cell.endswith("(headline)")
        ax.text(1.03, yy, f"{f} : {a}", transform=ax.get_yaxis_transform(), ha="left",
                va="center", fontsize=10, color=t["fg"], fontweight="bold" if bold else "normal")
        ax.text(-0.02, yy, cell, transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=9.5, color=t["fg"])
    for yy, model in header_y:
        ax.text(-0.02, yy, model, transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=10, fontweight="bold", color=t["fg"])
    ax.text(1.03, -0.9, "follow : anti", transform=ax.get_yaxis_transform(), ha="left",
            va="center", fontsize=9, color=t["muted"])

    ax.axvline(0.5, color=t["muted"], lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.text(0.5, -0.9, "no preference", ha="center", va="center", fontsize=9, color=t["muted"],
            bbox={"facecolor": t["bg"], "edgecolor": "none", "pad": 1.5})
    ax.set_xlim(0, 1.0)
    ax.set_ylim(total + 0.2, -1.4)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("share of donor-sensitive pairs that follow the omitted value")
    ax.grid(axis="x", lw=0.6)
    ax.set_axisbelow(True)

    fig.text(0.02, 0.965, "Donor-sensitive answers follow the omitted value in every cell we measured",
             fontsize=12, fontweight="bold", color=t["fg"], va="top")
    fig.text(0.02, 0.91, "Source omitted in every cell. Dot: follow share. Whiskers: Wilson 95% interval.\n"
             "Readout: menu for Gemma-4 and the two Qwen3 menu rows, free generation otherwise.",
             fontsize=9, color=t["muted"], va="top", linespacing=1.4)
    save(fig, ASSETS, "robustness", theme)


for theme in ("light", "dark"):
    draw(theme)
