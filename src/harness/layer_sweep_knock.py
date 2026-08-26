"""Layer-localization sweep for the attention knockout (necessity).

Blocks attention from post-edit positions to the edited value's token at a
restricted set of layers, and measures edit-following. Sweeping the layer
set answers whether cutting the read at a narrow band is enough to collapse
following (compact read) or whether all layers must be cut (distributed).

A forward pre-hook on each decoder layer records the current layer index in
a contextvar; the patched attention only masks when the current layer is in
the target set.
"""

import argparse
import contextvars
import json
import random
import sys
import os

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from harness.r1_recompute import make_item, forward_from, final_answer  # noqa: E402

CUR_LAYER = contextvars.ContextVar("cur_layer", default=-1)
KNOCK = contextvars.ContextVar("knock", default=None)  # (key_ixs, from_q, layerset)
_orig_sdpa = F.scaled_dot_product_attention


def patched_sdpa(q, k, v, attn_mask=None, **kw):
    cfg = KNOCK.get()
    if cfg is None or CUR_LAYER.get() not in cfg[2]:
        return _orig_sdpa(q, k, v, attn_mask=attn_mask, **kw)
    key_ixs, from_q, _ = cfg
    qlen, klen = q.shape[-2], k.shape[-2]
    ixs = [i for i in key_ixs if i < klen]
    if ixs:
        if attn_mask is None:
            attn_mask = torch.zeros(qlen, klen, dtype=q.dtype, device=q.device)
        elif attn_mask.dtype == torch.bool:
            attn_mask = torch.where(
                attn_mask, torch.zeros((), dtype=q.dtype, device=q.device),
                torch.full((), float("-inf"), dtype=q.dtype, device=q.device))
        else:
            attn_mask = attn_mask.clone()
        if kw.pop("is_causal", False):
            causal = torch.full((qlen, klen), float("-inf"),
                                dtype=q.dtype, device=q.device)
            attn_mask = attn_mask + torch.triu(causal, diagonal=klen - qlen + 1)
        first_row = max(0, from_q - (klen - qlen))
        if first_row < qlen:
            attn_mask[..., first_row:, ixs] = float("-inf")
    return _orig_sdpa(q, k, v, attn_mask=attn_mask, **kw)


F.scaled_dot_product_attention = patched_sdpa


def install_layer_hooks(model):
    def mk(i):
        def hook(mod, inp):
            CUR_LAYER.set(i)
            return None
        return hook
    for i, layer in enumerate(model.model.layers):
        layer.register_forward_pre_hook(mk(i))


def build_text(tok, model_name, prompt, prefill):
    msgs = [{"role": "user", "content": prompt}]
    kw = {"tokenize": False, "add_generation_prompt": True}
    if "qwen3" in model_name.lower():
        kw["enable_thinking"] = False
    return tok.apply_chat_template(msgs, **kw) + prefill


def span(tok, text, sub):
    cs = text.rfind(sub)
    enc = tok(text, return_offsets_mapping=True)
    return [i for i, (a, b) in enumerate(enc.offset_mapping)
            if cs >= 0 and a < cs + len(sub) and b > cs]


@torch.no_grad()
def gen(model, tok, ids, key_ixs, from_q, layerset, samples, max_new):
    KNOCK.set((key_ixs, from_q, layerset) if key_ixs else None)
    try:
        out = model.generate(ids, max_new_tokens=max_new, do_sample=True,
                             temperature=0.6, top_p=0.95,
                             num_return_sequences=samples,
                             pad_token_id=tok.eos_token_id)
    finally:
        KNOCK.set(None)
    return [final_answer(tok.decode(r[ids.shape[1]:], skip_special_tokens=True))
            for r in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--d", type=int, default=10)
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--max-new", type=int, default=300)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa")
    model.eval()
    install_layer_hooks(model)
    nlayers = model.config.num_hidden_layers
    third = nlayers // 3
    allL = set(range(nlayers))
    sets = {"early": set(range(0, third)), "mid": set(range(third, 2 * third)),
            "late": set(range(2 * third, nlayers)), "full": allL, "none": set()}
    for Lx in range(0, nlayers, args.stride):
        sets[f"single-{Lx}"] = {Lx}

    rng = random.Random(9100)
    fout = open(args.out, "a")
    done = tried = 0
    while done < args.n and tried < args.n * 4:
        tried += 1
        it = make_item(rng, args.d, rng.randint(4, 7))
        j = it["j"]
        delta = rng.choice([x for x in range(-9, 10) if x != 0])
        vj_bad = it["vals"][j] + delta
        tgt = forward_from(it, j, vj_bad)
        lines = it["lines"][:j]
        lines[-1] = f"After step {j} the value is {vj_bad}."
        text = build_text(tok, args.model, it["deep_prompt"],
                          "\n".join(lines) + "\n")
        vspan = span(tok, text, str(vj_bad))
        if not vspan:
            continue
        ids = tok(text, return_tensors="pt").input_ids.to("cuda")
        from_q = max(vspan) + 1
        rec = {"exp": "lknock", "model": args.model, "item": done, "j": j,
               "nlayers": nlayers, "target": str(tgt),
               "clean": str(it["answer"]), "cells": {}}
        for name, layerset in sets.items():
            kx = vspan if layerset else None
            ans = gen(model, tok, ids, kx, from_q, layerset,
                      args.samples, args.max_new)
            rec["cells"][name] = {
                "follows_edit": sum(a == str(tgt) for a in ans) / len(ans),
                "follows_clean": sum(a == str(it["answer"]) for a in ans) / len(ans),
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
