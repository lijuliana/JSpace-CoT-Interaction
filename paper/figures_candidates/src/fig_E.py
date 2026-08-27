"""Figure E: rate of reverting an edited value to correct, by the evidence
available in the prompt (heatmap with colorbar)."""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib.pyplot as plt
from style import apply, save, MUT

apply()

ROWS = ["DeepSeek V3.2", "Claude Sonnet 4.5", "Llama 3.3 70B", "Qwen3-4B"]
COLS = ["Only implied\nby the prompt", "Stated; also\nderivable\nelsewhere",
        "Stated; the\nonly source", "Prompt states\nhow to derive it"]
M = np.array([[0.01, 0.01, 0.05, 0.43],
              [0.26, 0.13, 0.78, 0.32],
              [0.01, 0.03, np.nan, 0.01],
              [0.00, np.nan, np.nan, 0.00]])

fig, ax = plt.subplots(figsize=(3.4, 2.35))
masked = np.ma.masked_invalid(M)
cmap = plt.get_cmap("Blues").copy()
cmap.set_bad("#f2f2f2")
im = ax.pcolormesh(masked, cmap=cmap, vmin=0, vmax=0.8,
                   edgecolors="white", linewidth=1.5)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        if np.isnan(M[i, j]):
            ax.text(j + 0.5, i + 0.5, "—", ha="center", va="center",
                    fontsize=6.6, color=MUT)
        else:
            dark = M[i, j] > 0.45
            ax.text(j + 0.5, i + 0.5, f"{M[i, j]:.2f}", ha="center",
                    va="center", fontsize=6.6,
                    color="white" if dark else "#1a1a1a")
ax.set_xticks(np.arange(4) + 0.5)
ax.set_xticklabels(COLS, fontsize=5.9)
ax.set_yticks(np.arange(4) + 0.5)
ax.set_yticklabels(ROWS, fontsize=6.6)
ax.invert_yaxis()
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
cb.set_label("Fraction of runs reverting an edited\nvalue to the correct one", fontsize=6.4)
cb.ax.tick_params(labelsize=6)
cb.outline.set_visible(False)
fig.tight_layout()
save(fig, "E_checking_heatmap")
