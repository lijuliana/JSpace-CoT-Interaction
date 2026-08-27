"""Figure D: the edit-and-continue protocol with the three interventions.
Monospace text marks token content; serif text marks annotations."""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from style import (apply, save, BLUE, VERM, GREEN, SKY, INK, MUT,
                   T_GRAY, T_VERM, T_GREEN, T_BLUE)

apply()

fig, ax = plt.subplots(figsize=(5.5, 2.2))
ax.axis("off")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

GAP = 0.010
segs = [
    (0.015, 0.150, "prompt:\nstart 481,\n10 ops", T_GRAY, INK),
    (0.175, 0.185, "After step 4\nthe value is 390.", T_GRAY, INK),
    (0.370, 0.185, "After step 5\nthe value is 457.", T_VERM, VERM),
    (0.565, 0.185, "model continues\nsteps 6..10", T_GRAY, INK),
    (0.760, 0.180, "#### answer", T_BLUE, BLUE),
]
y0, h = 0.40, 0.20
for x, w, text, fc, tc in segs:
    ax.add_patch(Rectangle((x, y0), w - GAP, h, fc=fc, ec="none"))
    ax.text(x + (w - GAP) / 2, y0 + h / 2, text, ha="center", va="center",
            fontsize=5.9, family="monospace", color=tc)
MID = 0.370 + (0.185 - GAP) / 2  # center of the edited segment

ax.annotate("1  Edit the visible text (452 to 457)",
            xy=(MID, y0 + h + 0.02), xytext=(0.28, 0.93),
            fontsize=6.4, color=INK,
            arrowprops=dict(arrowstyle="->", color=MUT, lw=0.8))

ax.add_patch(Rectangle((0.33, 0.03), 0.265, 0.17, fc=T_GREEN, ec="none"))
ax.text(0.4625, 0.115, "hidden state at '457'\ngets a value\nnever written",
        ha="center", va="center", fontsize=5.6, family="monospace",
        color=GREEN)
ax.add_patch(FancyArrowPatch((MID, 0.21), (MID, y0 - 0.015),
                             arrowstyle="->", color=GREEN, lw=1.2,
                             mutation_scale=9))
ax.text(0.615, 0.115, "2  Overwrite the state,\n    text unchanged",
        fontsize=6.4, color=GREEN, va="center")

ax.add_patch(FancyArrowPatch((0.655, y0 + h + 0.015), (0.485, y0 + h + 0.015),
                             arrowstyle="->", color=SKY, lw=1.2,
                             mutation_scale=9,
                             connectionstyle="arc3,rad=0.35"))
ax.text(0.570, 0.675, "x", ha="center", va="center", fontsize=8,
        color=VERM, fontweight="bold")
ax.text(0.625, 0.86, "3  Block attention from later\n    positions to the value token",
        fontsize=6.4, color=SKY)

ax.text(0.850, 0.25, "Readout: does the answer\nfollow the planted value?",
        ha="center", fontsize=6.2, color=MUT)

fig.tight_layout()
save(fig, "D_protocol")
