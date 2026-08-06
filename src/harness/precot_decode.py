"""Replication A1: is the answer linearly decodable from the residual stream,
and when.

Prior work (Reasoning Theater 2603.05488, pre-CoT decoding 2603.01437)
reports the final answer is linearly decodable from hidden states, often
before it is written. We test this on our variable chains, which doubles as
validation of the probing machinery the read-back patch relies on.

For each instance we run the model on prompt + a deterministic worked trace,
extract the residual at the end of each step, and train a ridge probe to
predict the final answer from that residual. We report decodability (R^2 on
held-out instances) vs position in the trace, at the best layer, with a
control probe against a different instance's answer (must be at chance).

Two things this establishes. Validation: if late-position decodability is
high and the control is at chance, the probe and extraction work. Science:
the shape (does decodability rise through the trace, or is the answer there
early) tests whether the pre-CoT-decoding phenomenon holds on tasks deep
enough to exceed internal capacity.
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


def ridge_r2(X, y, split, lam=10.0):
    Xtr, Xte = X[:split], X[split:]
    ytr, yte = y[:split], y[split:]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
    w = np.linalg.solve(A, Xtr.T @ (ytr - ytr.mean()))
    pred = Xte @ w + ytr.mean()
    ss = ((yte - pred) ** 2).sum()
    st = ((yte - yte.mean()) ** 2).sum() + 1e-9
    return 1 - ss / st


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--layer-step", type=int, default=4)
    ap.add_argument("--target", default="answer",
                    choices=["answer", "answer_minus_start"],
                    help="answer_minus_start removes the start-value confound, "
                    "isolating the genuinely computed part of the answer")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=device)
    model.eval()
    n_layers = model.config.num_hidden_layers
    layers = list(range(args.layer_step, n_layers + 1, args.layer_step))

    # collect residuals at each step-end, for each layer
    feats = {l: {} for l in layers}   # layer -> stepfrac_bucket -> list vec
    answers, controls = [], []
    insts = [variable_chain(args.depth, 90_000 + s) for s in range(args.n)]
    nbuckets = 6
    def target_val(inst):
        a = int(inst.answer)
        if args.target == "answer_minus_start":
            # the chain's start value is the first intermediate
            return a - int(inst.intermediates[0][1])
        return a

    for k, inst in enumerate(insts):
        answers.append(target_val(inst))
        controls.append(target_val(insts[(k + 1) % args.n]))
        text, ends, _ = build_trace(inst, write_target=True)
        full = inst.prompt + "\n" + text
        enc = tok(full, return_offsets_mapping=True)
        offmap = enc["offset_mapping"]
        ids = torch.tensor([enc["input_ids"]]).to(device)
        out = model(ids, output_hidden_states=True)
        base = len(inst.prompt) + 1
        nsteps = len(ends)
        for si, ce in enumerate(ends):
            char_end = base + ce
            pos = max(i for i, (a, b) in enumerate(offmap) if a < char_end)
            bucket = min(nbuckets - 1, int(nbuckets * (si + 1) / nsteps))
            for l in layers:
                v = out.hidden_states[l][0, pos].float().cpu().numpy()
                feats[l].setdefault(bucket, []).append((k, v))
        if (k + 1) % 100 == 0:
            print(f"{k+1}/{args.n}", flush=True)

    y = np.array(answers, float)
    yc = np.array(controls, float)
    split = int(0.8 * args.n)
    results = []
    for l in layers:
        for bucket, items in sorted(feats[l].items()):
            # align to instance index so train/test split is by instance
            idx = np.array([k for k, _ in items])
            X = np.stack([v for _, v in items])
            order = np.argsort(idx)
            X = X[order]
            yy = y[idx[order]]
            yyc = yc[idx[order]]
            sp = int(0.8 * len(X))
            results.append({
                "layer": l, "bucket": bucket,
                "step_frac": round((bucket + 1) / nbuckets, 2),
                "r2_answer": float(ridge_r2(X, yy, sp)),
                "r2_control": float(ridge_r2(X, yyc, sp)),
                "n": len(X)})

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    # print best-layer curve
    best = max(layers, key=lambda l: np.mean(
        [r["r2_answer"] for r in results
         if r["layer"] == l and r["bucket"] == max(
             rr["bucket"] for rr in results)]))
    print(f"best layer {best}, answer decodability vs step fraction:")
    for r in sorted([r for r in results if r["layer"] == best],
                    key=lambda r: r["bucket"]):
        print(f"  step_frac={r['step_frac']:.2f} "
              f"r2_answer={r['r2_answer']:.2f} "
              f"r2_control={r['r2_control']:.2f}")


if __name__ == "__main__":
    main()
