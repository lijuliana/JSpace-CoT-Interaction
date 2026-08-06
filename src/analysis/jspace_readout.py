"""Readout for the J-space protection run.

Three questions from one file. The anchor asymmetry: does J-space ablation
hurt direct answering more than cot answering, against the random-subspace
arm at the same frozen dose. The N2 redo: does the targeted squeeze change
cot externalization or length. Bookkeeping: cap hits and unparseable rates
per cell, and direct-cell token counts proving the channel was closed.
"""

import argparse
import json
from collections import defaultdict

import numpy as np


def ci(v, n_boot=2000, seed=0):
    v = np.asarray(v, float)
    if len(v) == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    m = rng.choice(v, (n_boot, len(v))).mean(1)
    return v.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    args = ap.parse_args()
    rows = []
    for p in args.inputs:
        rows += [json.loads(l) for l in open(p)]
    c = defaultdict(list)
    for r in rows:
        c[(r["arm"], r["condition"], r["difficulty"])].append(r)
    arms = ["clean", "jlens", "random"]
    diffs = sorted({d for _, _, d in c})

    print("accuracy by cell (cap-hit / unparseable rates in brackets):")
    print(f"{'d':>3} {'cond':>7} " + " ".join(f"{a:>22}" for a in arms))
    for d in diffs:
        for cond in ["direct", "cot"]:
            cells = []
            for a in arms:
                rs = c.get((a, cond, d), [])
                if rs:
                    acc = np.mean([r["correct"] for r in rs])
                    cap = np.mean([r["hit_cap"] for r in rs])
                    up = np.mean([r["unparseable"] for r in rs])
                    cells.append(f"{acc:.2f} [{cap:.2f}/{up:.2f}]")
                else:
                    cells.append("-")
            print(f"{d:>3} {cond:>7} " + " ".join(f"{x:>22}" for x in cells))

    print("\nanchor asymmetry: ablation-minus-clean accuracy drop")
    print(f"{'d':>3} {'arm':>7} {'direct_drop':>12} {'cot_drop':>9} "
          f"{'asym(dir-cot)':>14}")
    for d in diffs:
        for a in ["jlens", "random"]:
            dd = (np.mean([r["correct"] for r in c[("clean", "direct", d)]])
                  - np.mean([r["correct"] for r in c[(a, "direct", d)]])
                  ) if c.get((a, "direct", d)) else float("nan")
            cd = (np.mean([r["correct"] for r in c[("clean", "cot", d)]])
                  - np.mean([r["correct"] for r in c[(a, "cot", d)]])
                  ) if c.get((a, "cot", d)) else float("nan")
            print(f"{d:>3} {a:>7} {dd:>12.2f} {cd:>9.2f} {dd - cd:>14.2f}")

    print("\nN2 redo: cot externalization and length under the squeeze")
    print(f"{'d':>3} " + " ".join(f"{a:>16}" for a in arms))
    for d in diffs:
        cells = []
        for a in arms:
            rs = [r for r in c.get((a, "cot", d), [])
                  if r.get("ext_frac") is not None]
            if rs:
                e = np.mean([r["ext_frac"] for r in rs])
                t = np.mean([r["gen_tokens"] for r in rs])
                cells.append(f"ext {e:.2f} tok {t:4.0f}")
            else:
                cells.append("-")
        print(f"{d:>3} " + " ".join(f"{x:>16}" for x in cells))

    print("\ndirect-channel verification: generated tokens in direct cells")
    for a in arms:
        toks = [r["gen_tokens"] for (aa, cond, d), rs in c.items()
                if aa == a and cond == "direct" for r in rs]
        if toks:
            print(f"  {a}: median {int(np.median(toks))}, "
                  f"max {max(toks)}")


if __name__ == "__main__":
    main()
