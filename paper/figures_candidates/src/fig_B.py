"""Figure B: two-register composition, schematic plus rates.
(a) Two counters run in the same text; primed values are planted into the
hidden state only. (b) Rate of the expected answer per condition."""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from style import (apply, tag, save, BLUE, VERM, GREEN, SKY, INK, MUT,
                   T_GRAY, T_VERM, T_GREEN, T_BLUE)

apply()

fig, (axL, axR) = plt.subplots(1, 2, figsize=(5.5, 2.3),
                               gridspec_kw={"width_ratios": [1.15, 1],
                                            "wspace": 0.32, "left": 0.02,
                                            "right": 0.99, "top": 0.92,
                                            "bottom": 0.22})
axL.axis("off")


def band(ax, x, y, w, h, text, fc=T_GRAY, tc=INK):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec="none"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc,
            fontsize=6.3, family="monospace")


band(axL, 0.02, 0.80, 0.55, 0.13, "A after op 4: 512.")
band(axL, 0.02, 0.62, 0.55, 0.13, "B after op 4: 347.")
band(axL, 0.02, 0.40, 0.55, 0.13, "A + B = ?")
band(axL, 0.02, 0.10, 0.55, 0.15, "#### 641", fc=T_BLUE, tc=BLUE)
band(axL, 0.70, 0.80, 0.26, 0.13, "a' = 529", fc=T_VERM, tc=VERM)
band(axL, 0.70, 0.62, 0.26, 0.13, "b' = 112", fc=T_GREEN, tc=GREEN)
axL.add_patch(FancyArrowPatch((0.70, 0.865), (0.585, 0.865), arrowstyle="->",
                              color=VERM, lw=1.2, mutation_scale=9))
axL.add_patch(FancyArrowPatch((0.70, 0.685), (0.585, 0.685), arrowstyle="->",
                              color=GREEN, lw=1.2, mutation_scale=9))
axL.text(0.83, 0.42, "Planted into the\nhidden state only;\ntext unchanged",
         fontsize=6.2, color=MUT, ha="center", va="center")
axL.text(0.295, 0.30, "Answer $= a' + b' = 641$ (written nowhere)",
         fontsize=6.4, color=BLUE, ha="center")
axL.set_xlim(0, 1)
axL.set_ylim(0.05, 0.97)
tag(axL, "a", x=0.0, y=0.94)

q25 = [0.938, 0.945, 0.903, 0.045]
q3 = [0.888, 0.930, 0.880, 0.048]
x = np.arange(4)
w = 0.34
axR.bar(x - w / 2, q25, w, color=BLUE, edgecolor="none", label="Qwen2.5-7B")
axR.bar(x + w / 2, q3, w, color=SKY, edgecolor="none", label="Qwen3-4B")
for xi, (a, b) in enumerate(zip(q25, q3)):
    axR.text(xi - w / 2, a + 0.025, f"{a:.2f}", ha="center", fontsize=5.9,
             color=INK)
    axR.text(xi + w / 2, b + 0.025, f"{b:.2f}", ha="center", fontsize=5.9,
             color=INK)
axR.set_xticks(x)
axR.set_xticklabels(["Plant $a'$\n(get $a'{+}B$)", "Plant $b'$\n(get $A{+}b'$)",
                     "Plant both\n(get $a'{+}b'$)", "Random\n(answer\nmoves)"],
                    fontsize=6.0)
axR.set_ylabel("Rate of the expected answer")
axR.set_ylim(0, 1.14)
axR.set_yticks([0, 0.5, 1.0])
axR.legend(fontsize=6.0, loc="upper right", bbox_to_anchor=(1.02, 1.04))
axR.text(0.97, 0.50, "Answers using only\none planted value: 0.01",
         transform=axR.transAxes, fontsize=6.0, color=MUT, ha="right")
axR.grid(axis="y", color="#efefef", lw=0.6)
axR.set_axisbelow(True)
tag(axR, "b", x=-0.20)

save(fig, "B_composition")
