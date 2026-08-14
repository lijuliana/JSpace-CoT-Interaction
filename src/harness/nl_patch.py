"""Third-value residual patch on the natural-language state task.

Same logic as readback_patch.py but on nl_state items: teacher-force the
narrative trace through step j, and per item run four cells: text edit
only, restore correct state under corrupted text, plant a never-written
third value, norm-matched random control. Also logs the first 300 chars
of each continuation so reversion behavior is classifiable.
"""

import argparse
import json
import random
import sys
import os

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from harness.readback_patch import (  # noqa: E402
    ResidualPatch, capture_states)
from harness.r1_recompute import final_answer  # noqa: E402
from tasks import nl_state  # noqa: E402


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
    texts = [tok.decode(r[ids.shape[1]:], skip_special_tokens=True)
             for r in out]
    return [final_answer(t) for t in texts], texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--d", type=int, default=10)
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--layers", default="10-19")
    ap.add_argument("--max-new", type=int, default=300)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    lo, hi = args.layers.split("-")
    layers = list(range(int(lo), int(hi) + 1))
    rng = random.Random(7000)
    fout = open(args.out, "a")
    done = tried = 0

    while done < args.n and tried < args.n * 3:
        tried += 1
        it = nl_state.make_item(rng, args.d, rng.randint(4, 7))
        j = it["j"]
        V = it["vals"][j]
        delta = rng.choice([x for x in range(2, 10)]) * rng.choice([-1, 1])
        Vc = V + delta                      # corrupted (written) value
        Vs = V + 2 * delta                  # never-written third value
        clean_ans = it["answer"]
        corr_ans = nl_state.forward_from(it, j, Vc)
        swap_ans = nl_state.forward_from(it, j, Vs)
        if len({clean_ans, corr_ans, swap_ans}) < 3:
            continue
        corr_text = build_text(tok, args.model, it["deep_prompt"],
                               nl_state.prefix(it, Vc))
        clean_text = build_text(tok, args.model, it["deep_prompt"],
                                nl_state.prefix(it, V))
        swap_text = build_text(tok, args.model, it["deep_prompt"],
                               nl_state.prefix(it, Vs))
        ids_c = tok(corr_text, return_tensors="pt").input_ids.to("cuda")
        ids_k = tok(clean_text, return_tensors="pt").input_ids.to("cuda")
        ids_s = tok(swap_text, return_tensors="pt").input_ids.to("cuda")
        if not (ids_c.shape == ids_k.shape == ids_s.shape):
            continue
        vspan = span(tok, corr_text, str(Vc))
        if not vspan:
            continue
        L = ids_c.shape[1]
        start = min(vspan)
        sK = capture_states(model, ids_k, layers)
        sS = capture_states(model, ids_s, layers)
        sC = capture_states(model, ids_c, layers)
        rstates = {}
        for li in layers:
            noise = torch.randn_like(sC[li])
            scale = (sK[li] - sC[li]).norm()
            rstates[li] = sC[li] + noise * scale / noise.norm()

        rec = {"exp": "nlpatch", "model": args.model, "item": done, "j": j,
               "clean": str(clean_ans), "corr": str(corr_ans),
               "swap": str(swap_ans)}
        for cell, patch in (
                ("edit", None),
                ("restore", ResidualPatch(model, layers, sK, start, L)),
                ("third", ResidualPatch(model, layers, sS, start, L)),
                ("randctl", ResidualPatch(model, layers, rstates, start, L))):
            if patch is not None:
                patch.__enter__()
            try:
                answers, texts = gen(model, tok, ids_c, patch,
                                     args.samples, args.max_new)
            finally:
                if patch is not None:
                    patch.__exit__()
            rec[cell] = {
                "answers": answers,
                "follows_corr": sum(a == str(corr_ans)
                                    for a in answers) / len(answers),
                "follows_clean": sum(a == str(clean_ans)
                                     for a in answers) / len(answers),
                "follows_swap": sum(a == str(swap_ans)
                                    for a in answers) / len(answers),
                "text0": texts[0][:300],
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
