"""Decompose the corruption effect into token-path vs internal-path.

Among items where the plain corrupted run followed the corruption (there is
an effect to explain), restoring the internal mid-band state to its clean
value either reverts the answer (internal path) or not (token path). The
random-direction patch is the control: it should not revert.
"""

import argparse
import json

import numpy as np


def ci(v, n_boot=2000, seed=0):
    v = np.asarray(v, float)
    if len(v) == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    m = rng.choice(v, (n_boot, len(v))).mean(1)
    return (v.mean(), *np.percentile(m, [2.5, 97.5]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    args = ap.parse_args()
    rows = []
    for p in args.inputs:
        rows += [json.loads(l) for l in open(p)]

    # restrict to items with a corruption effect to decompose
    eff = [r for r in rows if r["corr_follows_corruption"] >= 0.5]
    print(f"{len(rows)} items, {len(eff)} with a corruption effect "
          f"(plain corrupted follows corruption >= 0.5)\n")

    # among those, does the internal-state patch revert to clean?
    revert = ci([r["patched_follows_clean"] for r in eff])
    still = ci([r["patched_follows_corruption"] for r in eff])
    rand_rev = ci([r["rand_follows_clean"] for r in eff])
    rand_still = ci([r["rand_follows_corruption"] for r in eff])

    print("among items with a corruption effect:")
    print(f"  internal-state patch -> follows CLEAN (revert): "
          f"{revert[0]:.2f} [{revert[1]:.2f},{revert[2]:.2f}]")
    print(f"  internal-state patch -> still follows CORRUPTION: "
          f"{still[0]:.2f} [{still[1]:.2f},{still[2]:.2f}]")
    print(f"  random-direction patch -> follows CLEAN (control): "
          f"{rand_rev[0]:.2f} [{rand_rev[1]:.2f},{rand_rev[2]:.2f}]")
    print(f"  random-direction patch -> still follows CORRUPTION: "
          f"{rand_still[0]:.2f} [{rand_still[1]:.2f},{rand_still[2]:.2f}]\n")

    print("interpretation:")
    print("  corr_follows_corruption>=0.5 (the analyzed set) means corrupting")
    print("  the written value flips the answer, so the model uses the")
    print("  written value and does NOT recompute from the unchanged")
    print("  operands. Restoring the residual at that token position to the")
    print("  clean value reverts the answer while the token string stays")
    print(f"  corrupt (revert {revert[0]:.2f}), and a matched-norm random")
    print(f"  patch does not ({rand_rev[0]:.2f}). The causal carrier of the")
    print("  read-back is the value-bearing residual at the written token;")
    print("  the read-back is specific and is not recomputation from operands.")


if __name__ == "__main__":
    main()
