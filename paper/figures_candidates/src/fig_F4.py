"""Figure F4: when models recompute an edited written value instead of
reading it back, by model and by the evidence the prompt provides.
Blue: the final answer builds on the edited value. Vermilion: the model
reverts to the correct value it recomputed from the prompt."""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib.pyplot as plt
from style import apply, save, BLUE, VERM, INK

apply()

PANELS = [
    ("DeepSeek V3.2", [
        ("Prompt only implies\nthe correct value", 0.93, BLUE),
        ("Prompt states how\nto derive the value", 0.57, VERM),
        ("Stated, but edit is\ntwo steps later", 0.96, BLUE)]),
    ("Claude Sonnet 4.5", [
        ("Edited step derivable\nfrom nearby steps", 0.81, BLUE),
        ("Edited step is the\nonly source", 0.22, VERM)]),
    ("Llama 3.3 70B", [
        ("Prompt only implies\nthe correct value", 0.97, BLUE),
        ("Prompt states how\nto derive the value", 0.97, BLUE)]),
    ("GSM8K, corrupted\nworked solutions", [
        ("Qwen3-4B", 0.87, BLUE),
        ("DeepSeek V3.2", 0.10, VERM)]),
]

fig, axes = plt.subplots(1, 4, figsize=(5.5, 1.75),
                         gridspec_kw={"wspace": 1.50, "left": 0.135,
                                      "right": 0.985, "top": 0.82,
                                      "bottom": 0.28})
for ax, (name, rows) in zip(axes, PANELS):
    labs = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    cols = [r[2] for r in rows]
    y = np.arange(len(rows))[::-1]
    ax.barh(y, vals, 0.55, color=cols, edgecolor="none")
    for yy, v in zip(y, vals):
        ax.text(v + 0.05, yy, f"{v:.2f}", va="center", fontsize=6.2,
                color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(labs, fontsize=5.9)
    ax.set_xlim(0, 1.22)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xticklabels(["0", ".5", "1"])
    ax.set_ylim(-0.55, len(rows) - 0.45)
    ax.text(0.5, 1.07, name, transform=ax.transAxes, ha="center",
            fontsize=7.0, style="italic")
    ax.grid(axis="x", color="#efefef", lw=0.6)
    ax.set_axisbelow(True)
fig.text((0.135 + 0.985) / 2, 0.03,
         "Fraction of runs where the final answer uses the edited value",
         ha="center", fontsize=8)
save(fig, "F4_read_vs_recompute")
