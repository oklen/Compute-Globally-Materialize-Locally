"""Paper Fig. 4 (Sec. 8, Programming) -- passive harvest fails; deliberate carriers restore it. README: assets/active.png.

  A  real dialogs (REALTALK / LoCoMo): harvested - isolated recognition accuracy, with
     95% CIs; passive harvesting of natural mentions shows no detected advantage.
  B  controlled trajectories: donor-follow rate across five carrier arms, worst -> best
     (passive -> ... -> explicit text upper bound), Qwen3-8B.

Data: data/fig_data.json (fig5A) and data/x9_rates.json (arm -> [mean, lo, hi]).
Run:  python fig5_active.py   ->  ../assets/active.png  (FIG_THEME=dark -> active-dark.png)"""
import json, os
import numpy as np
import matplotlib.pyplot as plt
from _paper_style import (
    save, DATADIR, panel_title, light_ygrid, light_xgrid,
    C_INK, C_MUTED, C_KV, C_NEG, C_ARM,
)

D = json.load(open(os.path.join(DATADIR, "fig_data.json")))
cells = [tuple(c) for c in D["fig5A"]["cells"]]        # (label, mean, lo, hi)
x9 = json.load(open(os.path.join(DATADIR, "x9_rates.json")))  # model -> arm -> [m,lo,hi]
MODELS = [("qwen", "Qwen3-8B", C_KV)]


def signed(v, fmt="{:+.3f}"):
    """A signed number with a real minus sign."""
    return fmt.format(v).replace("-", "−")


fig = plt.figure(figsize=(7.35, 2.75))
gs = fig.add_gridspec(1, 2, wspace=0.40, width_ratios=[1.08, 1.55])
fig.subplots_adjust(left=0.13, right=0.99, top=0.82, bottom=0.23)

# --- A: real-dialog harvest deficit ---
ax = fig.add_subplot(gs[0])
panel_title(ax, "A", "Passive harvest: no detected advantage")
y = np.arange(len(cells))[::-1]
for yy, (lb, m, lo, hi) in zip(y, cells):
    ax.errorbar(m, yy, xerr=[[m - lo], [hi - m]], fmt="o", color=C_NEG,
                ecolor=C_NEG, elinewidth=1.5, capsize=2.2, markersize=4.3,
                markeredgecolor="white", markeredgewidth=.45, zorder=3)
    ax.text(lo - 0.006, yy, signed(m), ha="right", va="center", fontsize=6.2,
            color=C_MUTED)
ax.axvline(0, color=C_INK, lw=0.8)
ax.set_yticks(y); ax.set_yticklabels([c[0].replace("Gemma4", "Gemma-4") for c in cells],
                                     fontsize=6.6)
ax.set_xlabel("harvested − isolated accuracy"); ax.set_xlim(-0.185, 0.075)
ax.set_xticks([-.15, -.1, -.05, 0, .05])
ax.set_ylim(-0.6, 3.55)
light_xgrid(ax)

# --- B: active-materialization arms ---
ax = fig.add_subplot(gs[1])
panel_title(ax, "B", "Active carriers restore the channel")
arms = ["pas", "bind", "dir", "nld", "exp"]
armlab = ["passive\nmention", "grounded\nbinding", "structured\ndirective",
          "NL compute\ndirective", "explicit\ntext\n(upper bound)"]
x = np.arange(5)
mods = [(m, lab, col) for m, lab, col in MODELS if m in x9]
w = 0.78 / max(len(mods), 1)
for i, (mod, lab, col) in enumerate(mods):
    off = -0.39 + w / 2 + i * w
    vals = [x9[mod][a] for a in arms]
    means = [v[0] for v in vals]
    low = [max(0.0, v[0] - v[1]) for v in vals]
    high = [max(0.0, v[2] - v[0]) for v in vals]
    colours = [C_ARM[a] for a in arms] if len(mods) == 1 else [col] * len(arms)
    bars = ax.bar(x + off, means, width=w * 0.88, color=colours, label=lab,
                  yerr=[low, high], error_kw={"ecolor": C_INK, "elinewidth": .8,
                                              "capsize": 2, "capthick": .8})
    # Value labels sit above the upper end of each interval, clear of the error bar.
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, min(max(v[0], v[2]) + .035, 1.06),
                f"{v[0]:.2f}", ha="center", va="bottom", fontsize=6.4)
ax.axhline(0.0, color=C_INK, lw=0.75)
ax.set_xticks(x); ax.set_xticklabels(armlab, fontsize=6.2)
ax.set_ylabel("donor-follow rate"); ax.set_ylim(0, 1.14)
ax.set_yticks([0, .25, .5, .75, 1.0])
ax.text(0.01, 0.98, "Qwen3-8B · 95% CI", transform=ax.transAxes, ha="left",
        va="bottom", fontsize=6.7, color=C_MUTED)
light_ygrid(ax)

save(fig, "fig5_active")
