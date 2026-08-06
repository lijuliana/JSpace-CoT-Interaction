"""R1: within-task recomputability manipulation.

Two renderings of the same arithmetic item, matched op count and identical
final answer, differing only in the rederivation depth of one target value:

  deep    - sequential chain, v_j = v_{j-1} + c_j; rederiving v_j from the
            prompt requires replaying j operations.
  shallow - the same v_j is introduced as the sum of two fresh prompt givens
            (a_j + b_j = v_j numerically); rederivation depth is 1. The chain
            then continues from v_j exactly as in deep, so everything after
            the target is identical between renderings.

Protocol per item (mirrors gate_b / gsm8k_readback): build the worked
solution up to and including the line stating v_j, corrupt v_j by a small
delta, and let the model continue from the corrupted line. Cells:

  edit    - corrupted v_j
  floor   - same truncation, no corruption (resample noise floor)

Outcome: follows_edit (answer = forward-computed from corrupted v_j),
follows_clean (answer = true answer), neither. Prediction from the plan:
follows_edit high in deep, near floor in shallow; clean accuracy in the
floor cells must match across renderings or the item set is confounded.

Trace lines state values only ("After step i the value is X."), never
restating operands, so within-trace 1-step rederivation is equally available
in both renderings and the only asymmetry is the prompt's dependency graph.
"""

import argparse
import json
import random
import re

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # item logic is importable API-side without torch
    torch = None


def make_item(rng, d, j):
    """One item; returns dict with both renderings."""
    v0 = rng.randint(100, 999)
    deltas = []
    vals = [v0]
    for _ in range(d):
        c = rng.choice([x for x in range(-99, 100)
                        if abs(x) >= 10 and abs(x) <= 99])
        deltas.append(c)
        vals.append(vals[-1] + c)
    vj = vals[j]
    # shallow decomposition of vj into two fresh givens
    a = rng.randint(100, min(999, abs(vj) + 900))
    b = vj - a
    deep_ops = [f"{'add' if c >= 0 else 'subtract'} {abs(c)}"
                for c in deltas]
    deep_prompt = (
        f"Start with the value {v0}. Apply these operations in order: "
        + "; ".join(f"({i+1}) {op}" for i, op in enumerate(deep_ops))
        + ". Track the running value after every operation. After the last "
        "operation, give the final value on its own line as '#### <number>'.")
    shallow_prompt = (
        f"Start with the value {v0}. Apply these operations in order: "
        + "; ".join(f"({i+1}) {op}" for i, op in enumerate(deep_ops[:j-1]))
        + f". Separately, Maya has {a} tokens and "
        + (f"gains {b} more" if b >= 0 else f"loses {abs(b)}")
        + f", and her total replaces the running value at step {j}. "
        "Then continue with: "
        + "; ".join(f"({i+1}) {op}"
                    for i, op in enumerate(deep_ops[j:], start=j))
        + ". Track the running value after every operation. After the last "
        "operation, give the final value on its own line as '#### <number>'.")
    lines = [f"After step {i+1} the value is {vals[i+1]}."
             for i in range(d)]
    return {"v0": v0, "deltas": deltas, "vals": vals, "j": j,
            "a": a, "b": b, "answer": vals[-1],
            "deep_prompt": deep_prompt, "shallow_prompt": shallow_prompt,
            "lines": lines}


def forward_from(item, j, vj_new):
    val = vj_new
    for c in item["deltas"][j:]:
        val = val + c
    return val


def prefix(item, vj_shown):
    """Worked-solution prefix through step j, with the step-j value as given."""
    lines = item["lines"][:item["j"]]
    lines[-1] = f"After step {item['j']} the value is {vj_shown}."
    return "\n".join(lines) + "\n"


def final_answer(text):
    m = re.findall(r"####\s*(-?[\d,]+)", text)
    if m:
        return m[-1].replace(",", "")
    return None


def continue_batch(model, tok, prompts, prefills, max_new, temp):
    torch.set_grad_enabled(False)
    texts = []
    for p, pre in zip(prompts, prefills):
        msgs = [{"role": "user", "content": p}]
        s = tok.apply_chat_template(msgs, tokenize=False,
                                    add_generation_prompt=True,
                                    enable_thinking=False) \
            if "qwen3" in model.name_or_path.lower() else \
            tok.apply_chat_template(msgs, tokenize=False,
                                    add_generation_prompt=True)
        texts.append(s + pre)
    tok.padding_side = "left"
    enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
    out = model.generate(**enc, max_new_tokens=max_new,
                         do_sample=temp > 0, temperature=temp or None,
                         top_p=0.95 if temp > 0 else None,
                         pad_token_id=tok.pad_token_id or tok.eos_token_id)
    gen = out[:, enc.input_ids.shape[1]:]
    return tok.batch_decode(gen, skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--d", type=int, default=10)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-new", type=int, default=300)
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    fout = open(args.out, "a")
    for seed in args.seeds:
        rng = random.Random(1000 + seed)
        items = [make_item(rng, args.d, rng.randint(4, 7))
                 for _ in range(args.n)]
        jobs = []  # (item_idx, rendering, cell, prompt, prefill, target)
        for ix, it in enumerate(items):
            delta = rng.choice([x for x in range(-9, 10) if x != 0])
            vj_bad = it["vals"][it["j"]] + delta
            for rend in ("deep", "shallow"):
                p = it[f"{rend}_prompt"]
                jobs.append((ix, rend, "edit", p, prefix(it, vj_bad),
                             forward_from(it, it["j"], vj_bad)))
                jobs.append((ix, rend, "floor", p,
                             prefix(it, it["vals"][it["j"]]),
                             forward_from(it, it["j"], vj_bad)))
        for k in range(0, len(jobs), args.batch):
            chunk = jobs[k:k + args.batch]
            outs = continue_batch(model, tok,
                                  [c[3] for c in chunk],
                                  [c[4] for c in chunk],
                                  args.max_new, args.temp)
            for (ix, rend, cell, _p, _pre, tgt), text in zip(chunk, outs):
                it = items[ix]
                ans = final_answer(text)
                rec = {"exp": "r1", "model": args.model, "seed": seed,
                       "item": ix, "d": args.d, "j": it["j"],
                       "rendering": rend, "cell": cell,
                       "answer": ans,
                       "clean": str(it["answer"]),
                       "edit_target": str(tgt),
                       "follows_edit": ans == str(tgt),
                       "follows_clean": ans == str(it["answer"]),
                       "n_tokens": len(tok(text).input_ids)}
                fout.write(json.dumps(rec) + "\n")
            fout.flush()
            if k % (args.batch * 10) == 0:
                print(f"seed {seed}: {k}/{len(jobs)}", flush=True)
    fout.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
