"""Figure F4: when models recompute instead of reading back, by model and
evidence condition (small multiples). Blue: the answer follows the edited
value. Vermilion: the model reverts to the correct value instead."""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib.pyplot as plt
from style import apply, save, BLUE, VERM, INK

apply()

PANELS = [
    ("DeepSeek V3.2", [
        ("Value implied\nby the prompt", 0.93, BLUE),
        ("Prompt defines\nthe value", 0.57, VERM),
        ("Defined, edit\ntwo steps away", 0.96, BLUE)]),
    ("Claude Sonnet 4.5", [
        ("No checkpoint\n(bridges 4 ops)", 0.81, BLUE),
        ("Checkpoint is\nsole source", 0.22, VERM)]),
    ("Llama 3.3 70B", [
        ("Value implied\nby the prompt", 0.97, BLUE),
        ("Prompt defines\nthe value", 0.97, BLUE)]),
    ("GSM8K corruption", [
        ("Qwen3-4B", 0.87, BLUE),
        ("DeepSeek V3.2", 0.10, VERM)]),
]

fig, axes = plt.subplots(1, 4, figsize=(5.5, 1.75),
                         gridspec_kw={"wspace": 1.15, "left": 0.115,
                                      "right": 0.985, "top": 0.86,
                                      "bottom": 0.26})
for ax, (name, rows), letter in zip(axes, PANELS, "abcd"):
    labs = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    cols = [r[2] for r in rows]
    y = np.arange(len(rows))[::-1]
    ax.barh(y, vals, 0.55, color=cols, edgecolor="none")
    for yy, v in zip(y, vals):
        ax.text(v + 0.05, yy, f"{v:.2f}", va="center", fontsize=6.2,
                color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(labs, fontsize=6.0)
    ax.set_xlim(0, 1.22)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xticklabels(["0", ".5", "1"])
    ax.set_ylim(-0.7, len(rows) - 0.3 + 0.4)
    ax.text(0.5, 1.06, name, transform=ax.transAxes, ha="center",
            fontsize=7.0, style="italic")
    ax.grid(axis="x", color="#efefef", lw=0.6)
    ax.set_axisbelow(True)
axes[0].set_xlabel("Answer follows the edited value")
axes[0].xaxis.set_label_coords(2.9, -0.22)
save(fig, "F4_read_vs_recompute")
