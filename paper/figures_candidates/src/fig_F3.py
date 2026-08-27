"""Figure F3: registers are retrieved through middle-layer attention.

(a) Blocking attention to the edited value token collapses edit-following
on four models; blocking matched control tokens does not.
(b) A single-layer state patch suffices at any layer up to 22 and is inert
from layer 24 on. (c) Attention knockout collapses following only when the
middle third of layers is blocked. Panels (b) and (c) share the layer axis.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib.pyplot as plt
from style import (apply, tag, save, BLUE, SKY, GRAY, MUT, T_SKY,
                   MODEL_COLORS, MODEL_MARKERS)

apply()

# recomputed from results/raw/r2_*.jsonl + r2fix_*.jsonl (means, n=600/cond)
KNOCK = {
    "Qwen3-4B": [0.900, 0.903, 0.890, 0.878, 0.000],
    "Phi-3-medium": [0.880, 0.863, 0.865, 0.850, 0.000],
    "Qwen2.5-7B": [0.643, 0.640, 0.670, 0.658, 0.000],
    "OLMo-2-7B": [0.333, 0.297, 0.342, 0.305, 0.005],
}
CONDS = ["No\nblock", "Neighbor\ntoken", "Random\nprompt\ntoken",
         "Random\ntrace\ntoken", "Value\ntoken"]
LABEL_Y = {"Qwen3-4B": 0.955, "Phi-3-medium": 0.785,
           "Qwen2.5-7B": 0.565, "OLMo-2-7B": 0.24}

# from results/raw/lsweep_qwen3-4b.jsonl and lknock_qwen3-4b.jsonl
P_LAYERS = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34]
P_SINGLE = [0.903, 0.913, 0.917, 0.933, 0.930, 0.933, 0.930, 0.930, 0.937,
            0.937, 0.940, 0.927, 0.007, 0.020, 0.007, 0.003, 0.010, 0.010]
K_NONE = 0.857
K_MID = 0.025
K_SINGLE = [0.860, 0.878, 0.865, 0.857, 0.872, 0.863, 0.885, 0.882, 0.860,
            0.868, 0.855, 0.875, 0.855, 0.872, 0.843, 0.855, 0.850, 0.880]

fig = plt.figure(figsize=(5.5, 2.55))
gsL = fig.add_gridspec(1, 1, left=0.095, right=0.48, top=0.95, bottom=0.24)
gsR = fig.add_gridspec(2, 1, hspace=0.28, left=0.60, right=0.985,
                       top=0.95, bottom=0.185)
axA = fig.add_subplot(gsL[0])
axB = fig.add_subplot(gsR[0])
axC = fig.add_subplot(gsR[1], sharex=axB)

# (a) knockout slopegraph
x = np.arange(5)
for m, vals in KNOCK.items():
    axA.plot(x, vals, "-", marker=MODEL_MARKERS[m], color=MODEL_COLORS[m],
             ms=3.4, lw=1.1, mec="white", mew=0.5, zorder=3)
    axA.text(-0.18, LABEL_Y[m], m, color=MODEL_COLORS[m], fontsize=6.4,
             ha="left")
axA.set_xticks(x)
axA.set_xticklabels(CONDS, fontsize=6.2)
axA.set_xlabel("Token to which attention is blocked", fontsize=7)
axA.set_ylabel("Fraction of runs where the final\nanswer uses the edited value")
axA.set_ylim(-0.05, 1.02)
axA.set_yticks([0, 0.5, 1.0])
axA.set_xlim(-0.25, 4.25)
axA.grid(axis="y", color="#efefef", lw=0.6)
axA.set_axisbelow(True)
tag(axA, "a", x=-0.21)

# (b) single-layer patch sufficiency
for ax in (axB, axC):
    ax.axvspan(11.5, 23.5, color=T_SKY, zorder=0)
axB.plot(P_LAYERS, P_SINGLE, "-o", color=BLUE, ms=2.6, lw=1.0, zorder=3)
axB.axvline(23, color=MUT, lw=0.6, ls=":", zorder=1)
axB.text(24.2, 0.55, "Boundary\nnear layer 23", fontsize=6.0, color=MUT)
axB.set_ylabel("Answer uses the\ninjected value", fontsize=7)
axB.set_ylim(-0.06, 1.06)
axB.set_yticks([0, 0.5, 1.0])
axB.tick_params(labelbottom=False)
tag(axB, "b", x=-0.20)

# (c) knockout necessity
axC.bar((11.5 + 23.5) / 2, K_NONE - K_MID, width=12, color=SKY,
        edgecolor="none", zorder=2)
axC.plot(P_LAYERS, [max(0.0, K_NONE - v) for v in K_SINGLE], "o",
         color=GRAY, ms=2.4, zorder=3)
axC.text(17.5, 0.66, "Middle third\nblocked", fontsize=6.0, ha="center",
         color="white", va="top", zorder=4)
axC.text(28.5, 0.38, "Single-layer\nblocks:\nno effect", fontsize=6.0,
         color=MUT, ha="center")
axC.set_ylabel("Drop in edit use\nwhen blocked", fontsize=7)
axC.set_ylim(-0.06, 1.0)
axC.set_yticks([0, 0.5, 1.0])
axC.set_xlabel("Layer (Qwen3-4B, 36 layers)")
axC.set_xticks([0, 12, 24, 35])
axC.set_xlim(-1, 36)
tag(axC, "c", x=-0.20)

save(fig, "F3_retrieval")
