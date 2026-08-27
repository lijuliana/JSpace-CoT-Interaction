"""Figure C: per-model rates for the residual-stream state patch
(standalone version of F2b)."""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib.pyplot as plt
from style import apply, save, BLUE, GREEN, GRAY

apply()

MODELS = ["Qwen3-4B", "Qwen2.5-7B", "Phi-3-medium", "OLMo-2-7B",
          "R1-distill-7B", "R1-distill-14B"]
RESTORE = [(1.00, 1.00, 1.00), (0.97, 0.92, 1.00), (0.98, 0.95, 1.00),
           (0.88, 0.82, 0.94), (1.00, 0.99, 1.00), (0.99, 0.98, 1.00)]
PLANT = [(1.00, 0.99, 1.00), (0.74, 0.67, 0.81), (0.94, 0.91, 0.97),
         (0.93, 0.88, 0.97), (0.76, 0.68, 0.84), (0.70, 0.61, 0.80)]
RAND = [(0.00, 0.00, 0.00), (0.00, 0.00, 0.01), (0.00, 0.00, 0.00),
        (0.00, 0.00, 0.00), (0.30, 0.22, 0.39), (0.21, 0.13, 0.29)]

fig, ax = plt.subplots(figsize=(3.4, 2.25))
y = np.arange(6)[::-1]
for data, color, mk, label, o in [
        (PLANT, BLUE, "o", "Inject a new value", 0.0),
        (RESTORE, GREEN, "s", "Restore the correct value", 0.24),
        (RAND, GRAY, "^", "Random direction (control)", -0.24)]:
    m = [d[0] for d in data]
    lo = [d[0] - d[1] for d in data]
    hi = [d[2] - d[0] for d in data]
    ax.errorbar(m, y + o, xerr=[lo, hi], fmt=mk, color=color, ms=4.4,
                mec="white", mew=0.8, elinewidth=1.0, label=label, zorder=3)
ax.set_yticks(y)
ax.set_yticklabels(MODELS, fontsize=6.6)
ax.set_xlabel("Fraction of runs where the final answer uses the injected value")
ax.set_xlim(-0.04, 1.06)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xticklabels(["0", "", "0.5", "", "1"])
ax.grid(axis="x", color="#efefef", lw=0.6)
ax.set_axisbelow(True)
ax.legend(fontsize=6.0, loc="center left", bbox_to_anchor=(0.10, 0.44))
fig.tight_layout()
save(fig, "C_register_dots")
