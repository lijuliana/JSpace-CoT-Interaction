"""Figure A: single-layer patch sufficiency ends at the layer where the
attention read completes (standalone version of F3b/c)."""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
import matplotlib.pyplot as plt
from style import apply, save, BLUE, MUT, T_SKY, SKY

apply()

LAYERS = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34]
SINGLE = [0.903, 0.913, 0.917, 0.933, 0.930, 0.933, 0.930, 0.930, 0.937,
          0.937, 0.940, 0.927, 0.007, 0.020, 0.007, 0.003, 0.010, 0.010]

fig, ax = plt.subplots(figsize=(3.4, 2.1))
ax.axvspan(11.5, 23.5, color=T_SKY, zorder=0)
ax.plot(LAYERS, SINGLE, "-o", color=BLUE, ms=3.0, lw=1.2, zorder=3)
ax.axvline(23, color=MUT, lw=0.6, ls=":", zorder=1)
ax.text(17.5, 0.13, "Attention knockout\ncollapses following\nonly here (12–23)",
        fontsize=6.2, color="#31708f", ha="center")
ax.annotate("0.93 to 0.01\nbetween layers 22 and 24",
            xy=(23.2, 0.47), xytext=(26.0, 0.68), fontsize=6.2, color=MUT,
            arrowprops=dict(arrowstyle="-", color=MUT, lw=0.5))
ax.set_xlabel("Layer patched (single layer, Qwen3-4B)")
ax.set_ylabel("Answer follows the planted value")
ax.set_xticks([0, 12, 24, 35])
ax.set_xlim(-1, 36)
ax.set_ylim(-0.05, 1.05)
ax.set_yticks([0, 0.5, 1.0])
ax.grid(axis="y", color="#efefef", lw=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
save(fig, "A_layer_cliff")
