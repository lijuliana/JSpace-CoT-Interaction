"""S1: slot algebra. Two independent counters computed in one trace, the
answer is their sum. We patch the internal state at counter A's final
written value to a never-written value a', at counter B's to b', separately
and jointly, leaving all visible text clean. If written-token states are
independent addressable slots, the joint patch should yield a' + b', and
each single patch the corresponding mixed sum.

Cells per item: none (clean), A-only, B-only, joint. Expected answers:
A+B, a'+B, A+b', a'+b'. A random-norm control on the joint cell guards the
usual objection. Composition uses two disjoint-range ResidualPatch hooks:
[posA, posB) takes states captured from a text with a' written, [posB, L)
from a text with b' written; each capture differs from clean text at one
value only, so the never-written content enters through state alone.
"""

import argparse
import json
import random
import re
import sys
import os

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from harness.readback_patch import (  # noqa: E402
    ResidualPatch, capture_states, continue_from)
from harness.r1_recompute import final_answer  # noqa: E402


def make_item(rng, d=4):
    def chain(start):
        vals = [start]
        ops = []
        for _ in range(d):
            c = rng.choice([x for x in range(-99, 100) if abs(x) >= 10])
            ops.append(c)
            vals.append(vals[-1] + c)
        return ops, vals
    a0, b0 = rng.randint(100, 999), rng.randint(100, 999)
    aops, avals = chain(a0)
    bops, bvals = chain(b0)
    def optxt(ops):
        return "; ".join(f"({i+1}) {'add' if c>=0 else 'subtract'} {abs(c)}"
                         for i, c in enumerate(ops))
    prompt = (
        f"Counter A starts at {a0} and gets, in order: {optxt(aops)}. "
        f"Counter B starts at {b0} and gets, in order: {optxt(bops)}. "
        "Track each counter after every operation, then give the sum of "
        "the two final values on its own line as '#### <number>'.")
    lines = [f"A after op {i+1}: {avals[i+1]}." for i in range(d)] + \
            [f"B after op {i+1}: {bvals[i+1]}." for i in range(d)]
    return {"prompt": prompt, "lines": lines,
            "afinal": avals[-1], "bfinal": bvals[-1]}


def build_ids(tok, model_name, prompt, lines):
    msgs = [{"role": "user", "content": prompt}]
    kw = {"tokenize": False, "add_generation_prompt": True}
    if "qwen3" in model_name.lower():
        kw["enable_thinking"] = False
    return tok.apply_chat_template(msgs, **kw) + "\n".join(lines) + "\n"


def last_span(tok, text, sub):
    cs = text.rfind(sub)
    if cs < 0:
        return None
    enc = tok(text, return_offsets_mapping=True)
    ixs = [i for i, (a, b) in enumerate(enc.offset_mapping)
           if a < cs + len(sub) and b > cs]
    return ixs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--layers", default="10-19")
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    lo, hi = args.layers.split("-")
    layers = list(range(int(lo), int(hi) + 1))
    rng = random.Random(6000)
    fout = open(args.out, "a")
    done = 0

    while done < args.n:
        it = make_item(rng)
        A, B = it["afinal"], it["bfinal"]
        ap_, bp_ = A + rng.choice([17, 23, -19, 31]), \
            B + rng.choice([13, 29, -23, 37])
        clean = build_ids(tok, args.model, it["prompt"], it["lines"])
        atext = clean.replace(f"A after op 4: {A}.",
                              f"A after op 4: {ap_}.")
        btext = clean.replace(f"B after op 4: {B}.",
                              f"B after op 4: {bp_}.")
        ids = tok(clean, return_tensors="pt").input_ids.to("cuda")
        aids = tok(atext, return_tensors="pt").input_ids.to("cuda")
        bids = tok(btext, return_tensors="pt").input_ids.to("cuda")
        if not (ids.shape == aids.shape == bids.shape):
            continue
        spanA = last_span(tok, clean, str(A))
        spanB = last_span(tok, clean, str(B))
        if not spanA or not spanB or max(spanA) >= min(spanB):
            continue
        L = ids.shape[1]
        sA = capture_states(model, aids, layers)
        sB = capture_states(model, bids, layers)
        posA, posB = min(spanA), min(spanB)
        cells = {
            "none": [],
            "aonly": [ResidualPatch(model, layers, sA, posA, L)],
            "bonly": [ResidualPatch(model, layers, sB, posB, L)],
            "joint": [ResidualPatch(model, layers, sA, posA, posB),
                      ResidualPatch(model, layers, sB, posB, L)],
        }
        # random-norm control: clean states plus noise matched per layer to
        # the swap deltas' norm, applied over the same joint range
        sC = capture_states(model, ids, layers)
        rstates = {}
        for li in layers:
            noise = torch.randn_like(sC[li])
            scale = ((sA[li] - sC[li]).norm() + (sB[li] - sC[li]).norm()) / 2
            rstates[li] = sC[li] + noise * scale / noise.norm()
        cells["randctl"] = [ResidualPatch(model, layers, rstates, posA, L)]
        expected = {"none": A + B, "aonly": ap_ + B,
                    "bonly": A + bp_, "joint": ap_ + bp_,
                    "randctl": A + B}
        rec = {"exp": "s1", "model": args.model, "item": done,
               "A": A, "B": B, "ap": ap_, "bp": bp_}
        ok = True
        for cell, patches in cells.items():
            for p in patches:
                p.__enter__()
                p.enabled = True
            try:
                out = model.generate(
                    ids, max_new_tokens=args.max_new, do_sample=True,
                    temperature=0.6, top_p=0.95,
                    num_return_sequences=args.samples,
                    pad_token_id=tok.eos_token_id)
            finally:
                for p in patches:
                    p.enabled = False
                    p.__exit__()
            answers = [final_answer(tok.decode(row[L:],
                                               skip_special_tokens=True))
                       for row in out]
            rec[cell] = {"answers": answers,
                         "expected": expected[cell],
                         "hit": sum(a == str(expected[cell])
                                    for a in answers) / len(answers)}
        fout.write(json.dumps(rec) + "\n")
        fout.flush()
        done += 1
        if done % 10 == 0:
            print(f"{done}/{args.n}", flush=True)
    fout.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
