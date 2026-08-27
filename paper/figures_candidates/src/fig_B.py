"""Figure B: two-register composition, schematic plus rates."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import numpy as np
from analysis.figstyle import apply

apply()
BLUE, VERM, GREEN, SKY = "#0072B2", "#D55E00", "#009E73", "#56B4E9"
INK, MUT = "#1a1a1a", "#666666"
T_GRAY, T_VERM, T_GREEN, T_BLUE = "#f0f0ee", "#fdeee6", "#e4f4ec", "#e8f1f8"

fig, (axL, axR) = plt.subplots(1, 2, figsize=(5.6, 2.35),
                               gridspec_kw={"width_ratios": [1.15, 1]})
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
                              color=VERM, lw=1.3, mutation_scale=9))
axL.add_patch(FancyArrowPatch((0.70, 0.685), (0.585, 0.685), arrowstyle="->",
                              color=GREEN, lw=1.3, mutation_scale=9))
axL.text(0.83, 0.42, "planted into the\nhidden state only;\ntext unchanged",
         fontsize=6, color=MUT, ha="center", va="center")
axL.text(0.295, 0.30, "answer = a' + b' = 641  (written nowhere)",
         fontsize=6.2, color=BLUE, ha="center")
axL.set_xlim(0, 1)
axL.set_ylim(0.05, 0.97)

q25 = [0.938, 0.945, 0.903, 0.045]
q3 = [0.888, 0.930, 0.880, 0.048]
x = np.arange(4)
w = 0.34
axR.bar(x - w / 2, q25, w, color=BLUE, edgecolor="none", label="Qwen2.5-7B")
axR.bar(x + w / 2, q3, w, color=SKY, edgecolor="none", label="Qwen3-4B")
for xi, (a, b) in enumerate(zip(q25, q3)):
    axR.text(xi - w / 2, a + 0.02, f"{a:.2f}", ha="center", fontsize=5.8,
             color=INK)
    axR.text(xi + w / 2, b + 0.02, f"{b:.2f}", ha="center", fontsize=5.8,
             color=INK)
axR.set_xticks(x)
axR.set_xticklabels(["plant A\n(get a′+B)", "plant B\n(get A+b′)",
                     "plant both\n(get a′+b′)", "random\n(answer\nmoves)"],
                    fontsize=5.8)
axR.set_ylabel("rate of expected answer")
axR.set_ylim(0, 1.12)
axR.set_yticks([0, 0.5, 1.0])
axR.legend(fontsize=5.6, loc="upper right", frameon=False)
axR.set_title("answers using only one planted value: 0.01", fontsize=6.0,
              color=MUT)
axR.grid(axis="y", color="#e8e8e8", lw=0.8)
axR.set_axisbelow(True)

out = os.path.join(os.path.dirname(__file__), "..")
fig.tight_layout()
fig.savefig(os.path.join(out, "B_composition.pdf"))
fig.savefig("/tmp/cur_B_composition.png", dpi=170)
print("B done")
