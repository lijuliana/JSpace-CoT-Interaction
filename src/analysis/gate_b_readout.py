"""Gate B readout: corruption-follow rates by difficulty with bootstrap CIs.

The number that matters: of continuations after a corrupted written value,
what fraction follow the corruption (external copy is live) vs the clean
value (internal copy rules) vs neither. Restated-clean cases are split out
since noticing the edit also requires reading the token.
"""

import argparse
import json
from collections import defaultdict

import numpy as np


def ci(flags, n_boot=2000, seed=0):
    v = np.asarray(flags, dtype=float)
    if len(v) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    m = rng.choice(v, (n_boot, len(v))).mean(axis=1)
    return (v.mean(), *np.percentile(m, [2.5, 97.5]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--csv")
    args = ap.parse_args()

    rows = []
    for p in args.inputs:
        rows += [json.loads(l) for l in open(p)]
    by_d = defaultdict(list)
    for r in rows:
        by_d[r["difficulty"]].append(r)

    out = []
    for d in sorted(by_d):
        rs = by_d[d]
        fc = ci([r["follows_corruption"] for r in rs])
        fl = ci([r["follows_clean"] for r in rs])
        nt = ci([not (r["follows_corruption"] or r["follows_clean"])
                 for r in rs])
        rc = ci([r["restates_clean_early"] for r in rs])
        out.append({"difficulty": d, "n": len(rs),
                    "follows_corruption": fc[0], "fc_lo": fc[1], "fc_hi": fc[2],
                    "follows_clean": fl[0], "fl_lo": fl[1], "fl_hi": fl[2],
                    "neither": nt[0], "restates_clean": rc[0]})
        print(f"d={d:3d} n={len(rs):4d}  "
              f"follows_corruption={fc[0]:.2f} [{fc[1]:.2f},{fc[2]:.2f}]  "
              f"follows_clean={fl[0]:.2f}  neither={nt[0]:.2f}  "
              f"restates_clean={rc[0]:.2f}")

    if args.csv:
        import csv as csvmod
        with open(args.csv, "w", newline="") as f:
            w = csvmod.DictWriter(f, fieldnames=list(out[0].keys()))
            w.writeheader()
            w.writerows(out)


if __name__ == "__main__":
    main()
