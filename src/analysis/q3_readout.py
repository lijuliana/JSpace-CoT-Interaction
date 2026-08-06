"""Readout for the assignment-core runs (Qwen3-4B x three datasets).

Sweep: accuracy per dataset x condition, overall and split by token-cap
hit, with unparseable rates and direct-cell token medians (channel
closure). Read-back: answer-change rate under corruption vs the resample
noise floor. The 2x2 is read by jspace_readout.py.
"""

import argparse
import json
from collections import defaultdict

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep")
    ap.add_argument("--readback")
    args = ap.parse_args()

    if args.sweep:
        rows = [json.loads(l) for l in open(args.sweep)]
        c = defaultdict(list)
        for r in rows:
            c[(r["dataset"], r["condition"])].append(r)
        print("sweep: accuracy overall | non-cap-only (n_noncap), "
              "cap rate, unparseable rate, median gen tokens")
        for ds in ["gsm8k", "math500", "aime2024"]:
            for co in ["direct", "free"]:
                rs = c.get((ds, co), [])
                if not rs:
                    continue
                acc = np.mean([r["correct"] for r in rs])
                nc = [r for r in rs if not r["hit_cap"]]
                acc_nc = np.mean([r["correct"] for r in nc]) if nc else float("nan")
                cap = np.mean([r["hit_cap"] for r in rs])
                up = np.mean([r["unparseable"] for r in rs])
                mt = int(np.median([r["gen_tokens"] for r in rs]))
                print(f"  {ds:>9} {co:>7} acc {acc:.2f} | {acc_nc:.2f} "
                      f"(n={len(nc)}), cap {cap:.2f}, unpar {up:.2f}, "
                      f"tok {mt}")

    if args.readback:
        rows = [json.loads(l) for l in open(args.readback)]
        ch = np.mean([r["answer_changed"] for r in rows])
        fl = np.mean([r["resample_changed"] for r in rows])
        up = np.mean([r["unparseable"] for r in rows])
        print(f"\nread-back (gsm8k worked solutions, n={len(rows)}): "
              f"answer changed under corruption {ch:.2f}, "
              f"resample noise floor {fl:.2f}, unparseable {up:.2f}")


if __name__ == "__main__":
    main()
