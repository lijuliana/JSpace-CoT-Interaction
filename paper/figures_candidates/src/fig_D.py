"""Figure D: the edit-and-continue protocol with the three interventions."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from analysis.figstyle import apply

apply()
BLUE, VERM, GREEN, SKY = "#0072B2", "#D55E00", "#009E73", "#56B4E9"
INK, MUT = "#1a1a1a", "#666666"
T_GRAY, T_VERM, T_GREEN, T_BLUE = "#f0f0ee", "#fdeee6", "#e4f4ec", "#e8f1f8"

fig, ax = plt.subplots(figsize=(5.6, 2.3))
ax.axis("off")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

segs = [
    (0.015, 0.155, "prompt:\nstart 481,\n10 ops", T_GRAY, INK),
    (0.180, 0.185, "After step 4\nthe value is 390.", T_GRAY, INK),
    (0.375, 0.185, "After step 5\nthe value is 457.", T_VERM, VERM),
    (0.570, 0.185, "model continues\nsteps 6..10", T_GRAY, INK),
    (0.765, 0.180, "#### answer", T_BLUE, BLUE),
]
y0, h = 0.40, 0.20
for x, w, text, fc, tc in segs:
    ax.add_patch(Rectangle((x, y0), w, h, fc=fc, ec="none"))
    ax.text(x + w / 2, y0 + h / 2, text, ha="center", va="center",
            fontsize=5.9, family="monospace", color=tc)

# 1: text edit
ax.annotate("1  edit the visible text  (452 to 457)",
            xy=(0.4675, y0 + h + 0.02), xytext=(0.30, 0.93),
            fontsize=6.2, color=INK,
            arrowprops=dict(arrowstyle="->", color=MUT, lw=0.8))

# 2: state overwrite
ax.add_patch(Rectangle((0.335, 0.03), 0.265, 0.17, fc=T_GREEN, ec="none"))
ax.text(0.4675, 0.115, "hidden state at '457'\ngets a value\nnever written",
        ha="center", va="center", fontsize=5.6, family="monospace",
        color=GREEN)
ax.add_patch(FancyArrowPatch((0.4675, 0.21), (0.4675, y0 - 0.015),
                             arrowstyle="->", color=GREEN, lw=1.3,
                             mutation_scale=9))
ax.text(0.615, 0.115, "2  overwrite the state,\n    text unchanged",
        fontsize=6.2, color=GREEN, va="center")

# 3: attention knockout
ax.add_patch(FancyArrowPatch((0.665, y0 + h + 0.015), (0.49, y0 + h + 0.015),
                             arrowstyle="->", color=SKY, lw=1.3,
                             mutation_scale=9,
                             connectionstyle="arc3,rad=0.35"))
ax.text(0.578, 0.685, "x", ha="center", va="center", fontsize=8, color=VERM,
        fontweight="bold")
ax.text(0.635, 0.86, "3  block attention from later\n    positions to the value token",
        fontsize=6.2, color=SKY)

ax.text(0.855, 0.24, "readout: does the answer\nfollow the planted value?",
        ha="center", fontsize=5.8, color=MUT)

out = os.path.join(os.path.dirname(__file__), "..")
fig.tight_layout()
fig.savefig(os.path.join(out, "D_protocol.pdf"))
fig.savefig("/tmp/cur_D_protocol.png", dpi=170)
print("D done")
