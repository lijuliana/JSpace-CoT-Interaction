"""Phase 2b eviction test with exogenous writing.

Teacher-forced traces make writing a controlled variable: for each
variable-chain instance we construct two trace texts, identical except that
one target step's computed value is written and the other omits it
(the step line keeps the operation but not the result). A forward pass over
prompt+trace captures residual streams at the end of each subsequent step.
Linear probes then measure how decodable the target value is from the
current position's hidden state, as a function of steps elapsed since
computation, separately for written and suppressed variants.

Eviction predicts written < suppressed at matched distance (the model can
drop what it wrote). Redundant-cache predicts written >= suppressed.
Both are visible in the same measurement.

Design notes:
- the target step is mid-chain (step index depth//2), same position in both
  variants, so token alignment differences are local to one number.
- probes are ridge regressions to the scalar value, trained across
  instances, evaluated on held-out instances by R^2. Control probe: same
  regression against a random other instance's target value (should be at
  chance) guards against probes keying on position artifacts.
- hidden states from every 4th layer; probe per (layer, elapsed-steps,
  variant).
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tasks.generators import variable_chain  # noqa: E402
from tasks.traces import build_trace  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--layer-step", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=device)
    model.eval()
    n_layers = model.config.num_hidden_layers
    layers = list(range(args.layer_step, n_layers + 1, args.layer_step))

    # feature collection: X[variant][layer][elapsed] -> list of vectors,
    # y -> target values, yc -> control values (another instance's target)
    feats = {v: {l: {} for l in layers} for v in ("written", "suppressed")}
    targets, controls = [], []

    insts = [variable_chain(args.depth, 50_000 + s) for s in range(args.n)]
    for k, inst in enumerate(insts):
        tgt_val = int(inst.intermediates[len(inst.meta["ops"]) // 2 + 1][1])
        ctrl_val = int(insts[(k + 1) % args.n].intermediates[
            len(inst.meta["ops"]) // 2 + 1][1])
        targets.append(tgt_val)
        controls.append(ctrl_val)
        for variant, write in (("written", True), ("suppressed", False)):
            text, ends, tgt = build_trace(inst, write)
            full = inst.prompt + "\n" + text
            ids = tok(full, return_tensors="pt").to(device)
            out = model(**ids, output_hidden_states=True)
            offset = len(tok(inst.prompt + "\n")["input_ids"])
            # map char ends to token positions
            enc = tok(full, return_offsets_mapping=True)
            offmap = enc["offset_mapping"]
            base = len(inst.prompt) + 1
            for elapsed, ei in enumerate(range(tgt, len(ends))):
                char_end = base + ends[ei]
                pos = max(i for i, (a, b) in enumerate(offmap)
                          if a < char_end)
                for l in layers:
                    vec = out.hidden_states[l][0, pos].float().cpu().numpy()
                    feats[variant][l].setdefault(elapsed, []).append(vec)
        if (k + 1) % 50 == 0:
            print(f"{k+1}/{args.n} instances", flush=True)

    # probes
    from numpy.linalg import lstsq
    y = np.array(targets, dtype=np.float64)
    yc = np.array(controls, dtype=np.float64)
    split = int(0.8 * args.n)
    results = []

    def r2(X, yy):
        Xtr, Xte = X[:split], X[split:]
        ytr, yte = yy[:split], yy[split:]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        Xtr = (Xtr - mu) / sd
        Xte = (Xte - mu) / sd
        lam = 10.0
        A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
        w = np.linalg.solve(A, Xtr.T @ (ytr - ytr.mean()))
        pred = Xte @ w + ytr.mean()
        ss = ((yte - pred) ** 2).sum()
        st = ((yte - yte.mean()) ** 2).sum()
        return 1 - ss / st

    for variant in feats:
        for l in layers:
            for elapsed, vecs in sorted(feats[variant][l].items()):
                X = np.stack(vecs)
                results.append({
                    "variant": variant, "layer": l, "elapsed": elapsed,
                    "r2": float(r2(X, y)),
                    "r2_control": float(r2(X, yc)),
                    "n": len(vecs)})
                print(f"{variant} l={l} elapsed={elapsed} "
                      f"r2={results[-1]['r2']:.3f} "
                      f"ctrl={results[-1]['r2_control']:.3f}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
