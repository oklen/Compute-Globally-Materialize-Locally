"""Paper Fig. 3 (Sec. 6, Access) -- the native accessibility envelope (Qwen3-8B). README: assets/envelope.png.

  A  native recovery vs payload cardinality (menu readout, independent decoys):
     strong for binary, collapses toward chance for 4-way / 8-way, 0 for 3-digit.
  B  a write-time compute note shifts the copied->derived frontier but not numeric.
  C  free generation under-reports the store vs a candidate-logit probe (legacy checkpoints).

Data: data/fig_data.json  keys fig4 (cardinality, nk2) and fig4C (probe pairs).
Run:  python fig4_access.py   ->  ../assets/envelope.png  (FIG_THEME=dark -> envelope-dark.png)"""
import json, os
import numpy as np
import matplotlib.pyplot as plt
from _paper_style import (
    save, DATADIR, light_ygrid, light_xgrid,
    C_KV, C_TX, C_NEG,
)

D = json.load(open(os.path.join(DATADIR, "fig_data.json")))
harv = D["fig4"]["cardinality_sameprotocol"]         # bin2/dir4/name8/num -> acc
nkr = {k: tuple(v) for k, v in D["fig4"]["nk2"].items()}  # setting -> (filler, note, n)
pairs = [tuple(p) for p in D["fig4C"]["pairs"]]      # (label, free-gen, logit-probe)
chance = {"bin2": .5, "dir4": .25, "name8": .125, "num": 1 / 900}

fig = plt.figure(figsize=(7.45, 2.95))
gs = fig.add_gridspec(1, 3, wspace=0.55, width_ratios=[1.32, 1.10, 1.14])
fig.subplots_adjust(left=0.075, right=0.99, top=0.76, bottom=0.19)

# Panel titles are two lines each, left-aligned over their own panel, so they cannot run together.
# --- A: cardinality curve ---
ax = fig.add_subplot(gs[0])
ax.set_title("A  Recovery falls to chance\n    above binary", loc="left", pad=6)
labs = ["binary", "4-way", "8-way", "3-digit"]; setts = ["bin2", "dir4", "name8", "num"]
vals = [harv[s] for s in setts]; x = np.arange(4)
ax.bar(x, vals, width=0.56, color=C_KV, label="harvested KV")
for xx, s, v in zip(x, setts, vals):
    ax.plot([xx - 0.35, xx + 0.35], [chance[s]] * 2, color=C_NEG, lw=1.25, ls="--")
    ax.text(xx, max(v, chance[s]) + 0.035, f"{v:.2f}", ha="center", fontsize=7.0)
ax.plot([], [], color=C_NEG, lw=1.25, ls="--", label="chance")
ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=6.9)
ax.set_ylabel("accuracy after source drop"); ax.set_ylim(0, 1.16)
ax.set_yticks([0, .25, .5, .75, 1.0])
ax.legend(frameon=False, fontsize=6.5, loc="upper right", handletextpad=0.45,
          labelspacing=0.25)
light_ygrid(ax)

# --- B: compute-note knob ---
ax = fig.add_subplot(gs[1])
ax.set_title("B  Compute note\n    shifts access", loc="left", pad=6)
ks = ["verdict", "num", "bin2"]; x = np.arange(3)
ax.bar(x - 0.18, [nkr[k][0] for k in ks], width=0.34, color="#B9C0C8", label="neutral filler")
ax.bar(x + 0.18, [nkr[k][1] for k in ks], width=0.34, color=C_KV, label="compute note")
for xx, k in zip(x, ks):
    if nkr[k][0] < .005 and nkr[k][1] < .005:
        ax.text(xx, 0.035, "0 / 0", ha="center", fontsize=6.7); continue
    ax.text(xx - 0.18, nkr[k][0] + 0.03, f"{nkr[k][0]:.2f}", ha="center", fontsize=6.7)
    ax.text(xx + 0.18, nkr[k][1] + 0.03, f"{nkr[k][1]:.2f}", ha="center", fontsize=6.7)
ax.set_xticks(x); ax.set_xticklabels(["derived\nbinary", "copied\n3-digit", "copied\nbinary"], fontsize=6.8)
ax.set_ylim(0, 1.22); ax.set_ylabel("accuracy after source drop")
ax.set_yticks([0, .25, .5, .75, 1.0])
ax.legend(frameon=False, fontsize=6.3, loc="upper center", bbox_to_anchor=(0.5, 1.00),
          ncol=2, columnspacing=0.75, handletextpad=0.35)
light_ygrid(ax)

# --- C: generation vs candidate-logit probe ---
ax = fig.add_subplot(gs[2])
ax.set_title("C  Logit probe reveals\n    hidden access", loc="left", pad=6)
y = np.arange(len(pairs))[::-1]
for yy, (lb, g, p) in zip(y, pairs):
    ax.plot([g, p], [yy, yy], color="#B8C0CA", lw=1.4, zorder=1)
    ax.scatter([g], [yy], color=C_TX, s=30, zorder=3, edgecolor="white", linewidth=.4)
    ax.scatter([p], [yy], color=C_KV, s=30, zorder=3, edgecolor="white", linewidth=.4)
ax.scatter([], [], color=C_TX, s=26, label="free generation")
ax.scatter([], [], color=C_KV, s=26, label="candidate logit")
_clab = lambda s: s.replace("Llama ", "Llama-3.1 / ").replace("Gemma ", "Gemma-2 / ")
ax.set_yticks(y); ax.set_yticklabels([_clab(p[0]) for p in pairs], fontsize=6.4)
ax.set_xlabel("gold rate"); ax.set_xlim(0, 1.05); ax.set_ylim(-0.62, 4.25)
ax.set_xticks([0, .25, .5, .75, 1.0])
ax.legend(frameon=False, fontsize=6.2, loc="upper left", bbox_to_anchor=(0.01, 1.0),
          handletextpad=0.25, labelspacing=0.20, ncol=1)
light_xgrid(ax)

save(fig, "fig4_access")
