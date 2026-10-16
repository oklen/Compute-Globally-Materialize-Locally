"""README figure for Sec. 9: two deployed heuristics that assume the converse.

A  247 tool-using HotpotQA trajectories (Qwen3-8B): what dropping each region's cache rows costs
   on follow-ups about values the agent read but never wrote down (gold-vs-decoy log-odds, nats).
B  418 cited multi-hop answers: the two citation failures, each over its own denominator.

Every value is from the paper (v2 LaTeX source, materialize.tex): A from L493-495 and L2432-2434,
B from L528-533 and L2485-2489.

Run:  python consequences.py   ->  ../assets/consequences.png and consequences-dark.png"""
import os

import matplotlib.pyplot as plt

from _style import BLUE, GREY, RED, save, use

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets")

# Region dropped -> cost on unstated follow-ups (nats). The retrieved documents are the tool output.
REGIONS = [("retrieved documents", 21.99), ("plan", 1.52), ("scratchpad", 0.81),
           ("tool input", 0.09), ("answer", -2.24)]
# (failure, share in %, denominator)
CITES = [("used, not cited", 20.7, "of load-bearing passages"),
         ("cited, not used", 1.5, "of citations (decorative)")]


def num(v, fmt):
    return fmt.format(v).replace("-", "−")


def draw(theme):
    t = use(theme)
    fig = plt.figure(figsize=(8.6, 3.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.55)
    fig.subplots_adjust(left=0.17, right=0.97, top=0.64, bottom=0.16)

    # ---- A: region ablation ----
    ax = fig.add_subplot(gs[0])
    ys = list(range(len(REGIONS)))
    for y, (name, cost) in zip(ys, REGIONS):
        ax.barh(y, cost, height=0.62, color=BLUE if name == "retrieved documents" else GREY)
        # Every value label sits right of the bar end (right of zero for the one negative bar),
        # so none runs into the region names on the left.
        ax.text(max(cost, 0) + 0.4, y, num(cost, "{:.2f}"), va="center", ha="left", fontsize=9.5,
                color=t["fg"])
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in REGIONS])
    ax.invert_yaxis()
    ax.axvline(0, color=t["muted"], lw=0.9)
    ax.set_xlim(-6.5, 27)
    ax.set_xticks([0, 10, 20])
    ax.set_xlabel("cost of dropping the region (nats)")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_title("A  Only the retrieved documents\n    are load-bearing", loc="left", fontsize=11,
                 fontweight="bold", pad=8)

    # ---- B: citation failures ----
    ax = fig.add_subplot(gs[1])
    ys = list(range(len(CITES)))
    for y, (name, share, denom) in zip(ys, CITES):
        ax.barh(y, share, height=0.55, color=RED if share > 10 else GREY)
        ax.text(share + 0.5, y, f"{share:.1f}%", va="center", ha="left", fontsize=10,
                color=t["fg"], fontweight="bold")
        ax.text(share + 0.5, y + 0.33, denom, va="center", ha="left", fontsize=9, color=t["muted"])
    ax.set_yticks(ys)
    ax.set_yticklabels([c[0] for c in CITES])
    ax.invert_yaxis()
    ax.set_ylim(1.6, -0.6)
    ax.set_xlim(0, 44)   # room for the denominators inside the panel, so the figure stays ~8.6 in wide
    ax.set_xticks([0, 10, 20])
    ax.set_xticklabels(["0%", "10%", "20%"])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_title("B  Citations fail by omission,\n    not decoration", loc="left", fontsize=11,
                 fontweight="bold", pad=8)

    fig.text(0.02, 0.975, "Two deployed heuristics assume a downstream product stands in for its evidence",
             fontsize=12, fontweight="bold", color=t["fg"], va="top")
    fig.text(0.02, 0.905, "A: 247 HotpotQA agent trajectories (Qwen3-8B); follow-ups about values the agent "
             "read but never wrote down.\nB: 418 cited multi-hop answers. Bridge passages are load-bearing "
             "on 76% of items and cited on 30%.",
             fontsize=9, color=t["muted"], va="top", linespacing=1.4)
    save(fig, ASSETS, "consequences", theme)


for theme in ("light", "dark"):
    draw(theme)
