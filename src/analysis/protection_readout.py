"""Read out the protection experiment.

Two comparisons, both now valid after the lesion-symmetry fix (the lesion
fires during prefill, so direct and cot face the same internal squeeze):

  within-family: protection = cot accuracy minus direct accuracy, and the
  cot accuracy drop under the target lesion vs the matched-damage control.

  cross-family (the dissociation): the target-lesion cot accuracy drop for
  entity tracking (internally stored) vs variable chains (externally stored),
  at matched neutral-text KL. The thesis predicts entity is hurt more.

Damage is metered on neutral text (kl_neutral); kl_task is reported too but
not used for matching, since on-task KL folds the targeted effect into the
meter.
"""

import argparse
import json
from collections import defaultdict

import numpy as np


def ci(flags, n_boot=2000, seed=0):
    v = np.asarray(flags, dtype=float)
    if len(v) == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    m = rng.choice(v, (n_boot, len(v))).mean(axis=1)
    return (v.mean(), *np.percentile(m, [2.5, 97.5]))


def load(path):
    rows = [json.loads(l) for l in open(path)]
    cells = defaultdict(list)
    kln, klt = {}, {}
    for r in rows:
        cells[(r["arm"], r["difficulty"], r["condition"])].append(r["correct"])
        kln[r["arm"]] = r.get("kl_neutral", r.get("kl", 0.0))
        klt[r["arm"]] = r.get("kl_task", 0.0)
    return cells, kln, klt


def family_report(path, name):
    cells, kln, klt = load(path)
    diffs = sorted({d for _, d, _ in cells})
    print(f"### {name}")
    print("kl_neutral by arm:", {a: round(v, 3) for a, v in kln.items()})
    print("kl_task by arm:   ", {a: round(v, 3) for a, v in klt.items()})
    print(f"{'d':>3} {'arm':>8} {'cot':>6} {'direct':>7} {'protect':>8} "
          f"{'cot_drop_vs_clean':>18}")
    drops = {}
    for d in diffs:
        base = np.mean(cells.get(("clean", d, "cot"), [np.nan]))
        for arm in ["clean", "target", "control"]:
            cot = np.mean(cells.get((arm, d, "cot"), [np.nan]))
            dr = np.mean(cells.get((arm, d, "direct"), [np.nan]))
            drop = base - cot
            if arm == "target":
                drops[d] = drop
            print(f"{d:>3} {arm:>8} {cot:>6.2f} {dr:>7.2f} {cot-dr:>8.2f} "
                  f"{drop:>18.2f}")
        print()
    return drops, kln.get("target", np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity")
    ap.add_argument("--chain")
    args = ap.parse_args()

    reports = {}
    if args.entity:
        reports["entity"] = family_report(args.entity, "entity tracking")
    if args.chain:
        reports["chain"] = family_report(args.chain, "variable chains")

    if "entity" in reports and "chain" in reports:
        ed, ekl = reports["entity"]
        cd, ckl = reports["chain"]
        print("### dissociation: target-lesion cot accuracy drop, "
              "entity vs chain")
        print(f"(entity target kl_neutral={ekl:.3f}, "
              f"chain target kl_neutral={ckl:.3f})")
        print(f"{'d':>3} {'entity_drop':>12} {'chain_drop':>12} "
              f"{'entity-chain':>13}")
        for d in sorted(set(ed) & set(cd)):
            print(f"{d:>3} {ed[d]:>12.2f} {cd[d]:>12.2f} "
                  f"{ed[d]-cd[d]:>13.2f}")
        print("\npositive entity-minus-chain means the internal lesion hurts "
              "parallel storage more than serial chains, the predicted "
              "dissociation.")


if __name__ == "__main__":
    main()
