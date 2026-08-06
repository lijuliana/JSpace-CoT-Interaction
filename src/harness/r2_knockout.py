"""R2: read-back localization by attention knockout.

Deep-rendering chain items with an edited value v_j (as in R1). During the
continuation we block attention from every post-edit position to a chosen
set of key positions, and measure whether edit-following collapses.

Conditions (knockout target):
  none      - baseline, no knockout
  value     - the digit tokens of the edited value in the prefill
  neighbor  - the tokens immediately before the value on the same line
  operand   - the tokens of the next operation's amount in the prompt
  random    - a random prompt token span matched in length and distance
  dots      - no knockout, but the edited value is replaced by dots in the
              text (separates content from position-only compute)

If following collapses only under 'value' (and 'dots' behaves like value
removal rather than like baseline), read-back runs through attention to the
value token specifically.

Mechanism: F.scaled_dot_product_attention is monkeypatched; a contextvar
holds the set of blocked absolute key positions and the first query
position allowed to see them (queries before the edit are unaffected).
Layer range restriction comes from wrapping decoder layers to toggle the
contextvar. This first pass knocks out all layers.
"""

import argparse
import contextvars
import json
import random
import re
import sys
import os

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from harness.r1_recompute import make_item, forward_from, final_answer  # noqa: E402

KNOCK = contextvars.ContextVar("knock", default=None)  # (key_ixs, from_q)
_orig_sdpa = F.scaled_dot_product_attention


def patched_sdpa(q, k, v, attn_mask=None, **kw):
    cfg = KNOCK.get()
    if cfg is None:
        return _orig_sdpa(q, k, v, attn_mask=attn_mask, **kw)
    key_ixs, from_q = cfg
    qlen, klen = q.shape[-2], k.shape[-2]
    ixs = [i for i in key_ixs if i < klen]
    if ixs:
        if attn_mask is None:
            attn_mask = torch.zeros(qlen, klen, dtype=q.dtype,
                                    device=q.device)
        elif attn_mask.dtype == torch.bool:
            attn_mask = torch.where(
                attn_mask, torch.zeros((), dtype=q.dtype, device=q.device),
                torch.full((), float("-inf"), dtype=q.dtype,
                           device=q.device))
        else:
            attn_mask = attn_mask.clone()
        # absolute position of query row r is klen - qlen + r
        first_row = max(0, from_q - (klen - qlen))
        if first_row < qlen:
            attn_mask[..., first_row:, ixs] = float("-inf")
    return _orig_sdpa(q, k, v, attn_mask=attn_mask, **kw)


F.scaled_dot_product_attention = patched_sdpa


def token_span(tok, text, sub, occurrence=-1):
    """Token index range covering the given substring occurrence."""
    starts = [m.start() for m in re.finditer(re.escape(sub), text)]
    if not starts:
        return None
    cs = starts[occurrence]
    enc = tok(text, return_offsets_mapping=True)
    ixs = [i for i, (a, b) in enumerate(enc.offset_mapping)
           if a < cs + len(sub) and b > cs]
    return ixs


def build_text(tok, model_name, prompt, prefill):
    msgs = [{"role": "user", "content": prompt}]
    kw = {"tokenize": False, "add_generation_prompt": True}
    if "qwen3" in model_name.lower():
        kw["enable_thinking"] = False
    return tok.apply_chat_template(msgs, **kw) + prefill


@torch.no_grad()
def gen_one(model, tok, text, knock_ixs, from_q, max_new, temp):
    enc = tok(text, return_tensors="pt").to(model.device)
    KNOCK.set((knock_ixs, from_q) if knock_ixs else None)
    try:
        out = model.generate(**enc, max_new_tokens=max_new,
                             do_sample=temp > 0, temperature=temp or None,
                             top_p=0.95 if temp > 0 else None,
                             pad_token_id=tok.pad_token_id
                             or tok.eos_token_id)
    finally:
        KNOCK.set(None)
    return tok.decode(out[0, enc.input_ids.shape[1]:],
                      skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--d", type=int, default=10)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--max-new", type=int, default=300)
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--conditions", nargs="+",
                    default=["none", "value", "neighbor", "operand",
                             "random", "dots"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa")
    model.eval()
    fout = open(args.out, "a")

    for seed in args.seeds:
        rng = random.Random(2000 + seed)
        for ix in range(args.n):
            it = make_item(rng, args.d, rng.randint(4, 7))
            delta = rng.choice([x for x in range(-9, 10) if x != 0])
            j = it["j"]
            vj_bad = it["vals"][j] + delta
            tgt = forward_from(it, j, vj_bad)
            lines = it["lines"][:j]
            lines[-1] = f"After step {j} the value is {vj_bad}."
            prefill = "\n".join(lines) + "\n"
            text = build_text(tok, args.model, it["deep_prompt"], prefill)
            val_span = token_span(tok, text, str(vj_bad))
            if not val_span:
                continue
            from_q = max(val_span) + 1
            for cond in args.conditions:
                ktext, kixs = text, None
                if cond == "value":
                    kixs = val_span
                elif cond == "neighbor":
                    kixs = token_span(tok, text, "value is", -1)
                elif cond == "operand":
                    op_amt = str(abs(it["deltas"][j])) \
                        if j < len(it["deltas"]) else str(abs(it["deltas"][-1]))
                    kixs = token_span(tok, text, op_amt, 0)
                elif cond == "random":
                    width = len(val_span)
                    lo = max(0, min(val_span) - 120)
                    start = rng.randint(lo, max(lo + 1, min(val_span) - 40))
                    kixs = list(range(start, start + width))
                elif cond == "dots":
                    cs = text.rfind(str(vj_bad))
                    ktext = text[:cs] + "." * len(str(vj_bad)) \
                        + text[cs + len(str(vj_bad)):]
                out = gen_one(model, tok, ktext, kixs, from_q,
                              args.max_new, args.temp)
                ans = final_answer(out)
                fout.write(json.dumps({
                    "exp": "r2", "model": args.model, "seed": seed,
                    "item": ix, "j": j, "cond": cond, "answer": ans,
                    "clean": str(it["answer"]), "edit_target": str(tgt),
                    "follows_edit": ans == str(tgt),
                    "follows_clean": ans == str(it["answer"]),
                }) + "\n")
            fout.flush()
            if ix % 20 == 0:
                print(f"seed {seed}: item {ix}/{args.n}", flush=True)
    fout.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
