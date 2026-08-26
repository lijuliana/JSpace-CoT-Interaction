"""Layer-localization sweep for the state patch (sufficiency).

For each item we corrupt the written value at step j and continue. We then
plant the never-written third value into the residual stream at that token,
restricting the patch to a chosen set of layers, and measure how often the
answer follows the planted value. Sweeping the layer set answers whether a
narrow band suffices (compact read) or the full band is needed (distributed).

Layer sets:
  single-<L>   patch only layer L (sweep L across depth)
  early/mid/late  the three thirds of the network
  full         all layers (reference, matches the main-paper patch)
Each cell also carries a same-layers norm-matched random control.
"""

import argparse
import json
import random
import sys
import os

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from harness.readback_patch import ResidualPatch, capture_states  # noqa: E402
from harness.r1_recompute import (  # noqa: E402
    make_item, forward_from, final_answer)


def build_text(tok, model_name, prompt, prefill):
    msgs = [{"role": "user", "content": prompt}]
    kw = {"tokenize": False, "add_generation_prompt": True}
    if "qwen3" in model_name.lower():
        kw["enable_thinking"] = False
    return tok.apply_chat_template(msgs, **kw) + prefill


def span(tok, text, sub):
    cs = text.rfind(sub)
    if cs < 0:
        return None
    enc = tok(text, return_offsets_mapping=True)
    return [i for i, (a, b) in enumerate(enc.offset_mapping)
            if a < cs + len(sub) and b > cs]


@torch.no_grad()
def gen(model, tok, ids, patch, samples, max_new):
    if patch is not None:
        patch.enabled = True
    try:
        out = model.generate(ids, max_new_tokens=max_new, do_sample=True,
                             temperature=0.6, top_p=0.95,
                             num_return_sequences=samples,
                             pad_token_id=tok.eos_token_id)
    finally:
        if patch is not None:
            patch.enabled = False
    return [final_answer(tok.decode(r[ids.shape[1]:], skip_special_tokens=True))
            for r in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--d", type=int, default=10)
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--stride", type=int, default=2,
                    help="single-layer sweep stride")
    ap.add_argument("--coarse", action="store_true",
                    help="thirds + full only, skip per-layer sweep")
    ap.add_argument("--max-new", type=int, default=300)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    nlayers = model.config.num_hidden_layers
    third = nlayers // 3
    all_layers = list(range(nlayers))
    layer_sets = {
        "early": list(range(0, third)),
        "mid": list(range(third, 2 * third)),
        "late": list(range(2 * third, nlayers)),
        "full": all_layers,
    }
    if not args.coarse:
        for L in range(0, nlayers, args.stride):
            layer_sets[f"single-{L}"] = [L]

    rng = random.Random(9000)
    fout = open(args.out, "a")
    done = tried = 0
    while done < args.n and tried < args.n * 4:
        tried += 1
        it = make_item(rng, args.d, rng.randint(4, 7))
        j = it["j"]
        V = it["vals"][j]
        delta = rng.choice(range(2, 10)) * rng.choice([-1, 1])
        Vc, Vs = V + delta, V + 2 * delta
        clean_ans = it["answer"]
        swap_ans = forward_from(it, j, Vs)
        corr_ans = forward_from(it, j, Vc)
        if len({clean_ans, corr_ans, swap_ans}) < 3:
            continue
        lines = it["lines"][:j]
        lines[-1] = f"After step {j} the value is {Vc}."
        corr_text = build_text(tok, args.model, it["deep_prompt"],
                               "\n".join(lines) + "\n")
        lines[-1] = f"After step {j} the value is {Vs}."
        swap_text = build_text(tok, args.model, it["deep_prompt"],
                               "\n".join(lines) + "\n")
        ids_c = tok(corr_text, return_tensors="pt").input_ids.to("cuda")
        ids_s = tok(swap_text, return_tensors="pt").input_ids.to("cuda")
        if ids_c.shape != ids_s.shape:
            continue
        vspan = span(tok, corr_text, str(Vc))
        if not vspan:
            continue
        start, L = min(vspan), ids_c.shape[1]
        # capture swap states at every layer once
        sS_all = capture_states(model, ids_s, all_layers)

        rec = {"exp": "lsweep", "model": args.model, "item": done, "j": j,
               "nlayers": nlayers, "swap": str(swap_ans),
               "clean": str(clean_ans), "corr": str(corr_ans), "cells": {}}
        for name, layers in layer_sets.items():
            sS = {li: sS_all[li] for li in layers}
            patch = ResidualPatch(model, layers, sS, start, L)
            patch.__enter__()
            try:
                ans = gen(model, tok, ids_c, patch, args.samples, args.max_new)
            finally:
                patch.__exit__()
            rec["cells"][name] = {
                "follows_swap": sum(a == str(swap_ans) for a in ans) / len(ans),
                "follows_clean": sum(a == str(clean_ans) for a in ans) / len(ans),
                "follows_corr": sum(a == str(corr_ans) for a in ans) / len(ans),
            }
        fout.write(json.dumps(rec) + "\n")
        fout.flush()
        done += 1
        if done % 10 == 0:
            print(f"{done}/{args.n}", flush=True)
    fout.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
