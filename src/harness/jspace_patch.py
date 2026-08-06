"""Patch decomposition: does the readable value register live in J-space?

The earlier read-back patch overwrote the full residual at the written
token, which shows the register exists but not where. This decomposes it.
At the patch band we split the clean-minus-corrupt delta into its J-space
component (the span of preimage directions for the most active lens
concepts at each position, union over the clean and corrupt states) and the
complement, and restore each separately:

  arms: corrupt (no patch) / full delta / jspace component only /
        complement only / random rank-matched projection of the delta

If the answer reverts under the J-space component alone, the value register
is the lens-readable concept workspace; if only the complement reverts, the
register lives outside what the lens reads; full-residual stays as the
sanity baseline. The random rank-matched arm guards against any-projection
effects. Termination and parse failures are logged separately throughout.
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tasks.generators import variable_chain  # noqa: E402
from tasks.traces import build_trace  # noqa: E402
from harness.generate import build_prompt, extract_answer  # noqa: E402
from harness.readback_patch import (ResidualPatch, capture_states,  # noqa: E402
                                    continue_from, value_token_positions)


def jspace_basis(h_states, lens, layer, U, m):
    """Per-position orthonormal basis for the union of top-m active lens
    concepts of each state in h_states (list of [seq,d] tensors)."""
    J = lens.jacobians[layer].to(h_states[0].device)
    Uf = U.to(h_states[0].device).float()
    seq = h_states[0].shape[0]
    tops = []
    for h in h_states:
        logits = (h.float() @ J.T) @ Uf.T  # [seq, V]
        tops.append(logits.topk(m, dim=-1).indices)  # [seq, m]
    cat = torch.cat(tops, dim=-1)  # [seq, 2m]
    Q = []
    for p in range(seq):
        toks = torch.unique(cat[p])
        v = Uf[toks] @ J  # [t, d] preimages
        q, _ = torch.linalg.qr(v.T)  # [d, t]
        Q.append(q)
    return Q  # list of [d, rank] per position


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--lens-repo", default="neuronpedia/jacobian-lens")
    ap.add_argument("--lens-file",
                    default="qwen2.5-7b-it/jlens/Salesforce-wikitext/"
                            "Qwen2.5-7B-Instruct_jacobian_lens.pt")
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--m", type=int, default=16,
                    help="top-m active concepts per state; union over "
                    "clean+corrupt gives the J-space basis")
    ap.add_argument("--layers", default="10-19")
    ap.add_argument("--delta", type=int, default=40)
    ap.add_argument("--max-new", type=int, default=300)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from jlens.lens import JacobianLens

    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=device)
    model.eval()
    lens = JacobianLens.from_pretrained(args.lens_repo,
                                        filename=args.lens_file)
    a, b = args.layers.split("-")
    band = [l for l in range(int(a), int(b) + 1)
            if l in lens.source_layers]
    print(f"patch band (fitted layers only): {band}", flush=True)
    U = model.lm_head.weight

    def forward_answer(inst, idx, val):
        v = val
        for op, arg in inst.meta["ops"][idx:]:
            v = v + arg if op == "+" else v - arg
        return str(v)

    g = torch.Generator().manual_seed(0)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out = open(args.out, "w")
    kept = 0
    for s in range(args.n):
        inst = variable_chain(args.depth, 60_000 + s)
        tgt = len(inst.meta["ops"]) // 2
        clean_val = int(inst.intermediates[tgt + 1][1])
        corr_val = clean_val + args.delta
        clean_ans = forward_answer(inst, tgt + 1, clean_val)
        corr_ans = forward_answer(inst, tgt + 1, corr_val)
        if clean_ans == corr_ans:
            continue
        prompt = build_prompt(inst, "cot", tok, False)
        text, ends, _ = build_trace(inst, write_target=True)
        prefix = text[:ends[tgt]]
        clean_text = prompt + "\n" + prefix
        marker = f"= {clean_val}."
        pos = clean_text.rfind(marker)
        if pos != len(clean_text) - len(marker):
            continue
        corr_text = clean_text[:pos] + f"= {corr_val}."
        ci = tok(clean_text, return_tensors="pt").to(device)
        xi = tok(corr_text, return_tensors="pt").to(device)
        if ci.input_ids.shape[1] != xi.input_ids.shape[1]:
            continue
        plen = tok(prompt + "\n", return_tensors="pt").input_ids.shape[1]
        cs = capture_states(model, ci.input_ids, band)
        xs = capture_states(model, xi.input_ids, band)
        vpos = value_token_positions(tok, xi.input_ids[0].tolist(),
                                     plen, str(corr_val))
        start = vpos[-1] if vpos else plen
        end = xi.input_ids.shape[1]

        # build per-arm target states
        arm_states = {"full": {li: cs[li] for li in band}}
        js, comp, rnd = {}, {}, {}
        for li in band:
            delta = (cs[li] - xs[li])  # [seq, d]
            dev = delta.device
            Qs = jspace_basis([cs[li].to("cuda"), xs[li].to("cuda")],
                              lens, li, U, args.m)
            tj = xs[li].clone()
            tc = xs[li].clone()
            tr = xs[li].clone()
            for p in range(start, end):
                Q = Qs[p].cpu().float()
                dp = delta[p].float()
                proj = Q @ (Q.T @ dp)
                tj[p] = xs[li][p] + proj
                tc[p] = xs[li][p] + (dp - proj)
                # random rank-matched projection of the same delta
                rmat = torch.randn(dp.shape[0], Q.shape[1], generator=g)
                qr, _ = torch.linalg.qr(rmat)
                tr[p] = xs[li][p] + qr @ (qr.T @ dp)
            js[li], comp[li], rnd[li] = tj, tc, tr
        arm_states["jspace"] = js
        arm_states["complement"] = comp
        arm_states["random_rank"] = rnd

        # generate: corrupt baseline + each patch arm
        base_ans, _ = continue_from(model, tok, xi.input_ids, None,
                                    args.max_new, args.samples)
        rec = {"seed": inst.seed, "depth": args.depth,
               "clean_val": clean_val, "corr_val": corr_val,
               "clean_ans": clean_ans, "corr_ans": corr_ans,
               "corr_follows_corruption":
                   sum(x == corr_ans for x in base_ans) / len(base_ans),
               "unparseable_base":
                   sum(not x.strip() for x in base_ans) / len(base_ans)}
        for arm, states in arm_states.items():
            patch = ResidualPatch(model, band, states, start, end)
            with patch:
                ans, _ = continue_from(model, tok, xi.input_ids, patch,
                                       args.max_new, args.samples)
            rec[f"{arm}_follows_clean"] = (
                sum(x == clean_ans for x in ans) / len(ans))
            rec[f"{arm}_follows_corruption"] = (
                sum(x == corr_ans for x in ans) / len(ans))
            rec[f"{arm}_unparseable"] = (
                sum(not x.strip() for x in ans) / len(ans))
        out.write(json.dumps(rec) + "\n")
        out.flush()
        kept += 1
        if kept % 10 == 0:
            print(f"{kept} items", flush=True)
    out.close()
    print(f"done, {kept} items", flush=True)


if __name__ == "__main__":
    main()
