"""Gate B: are written intermediate values read back?

Procedure per instance: generate a trace, find the last clean mention of a
mid-chain intermediate value, corrupt it (perturb the number), truncate the
trace right after the corrupted mention, and let the model continue. If the
external copy is load-bearing, the continuation should follow the corrupted
value (final answer consistent with corruption); if the internal copy rules,
the continuation ignores it.

Three outcomes are distinguished per item:
  follows_corruption  - answer matches recomputing forward from the corrupted value
  follows_clean       - answer matches the original ground truth
  neither             - anything else (model notices and re-derives, or errors)
A model that notices the inconsistency and rewrites the value is evidence of
read-back too (it read the token to notice); those get flagged separately by
checking whether the continuation restates the clean value early.

Also runs the paired control from plan.md: patching is not needed at gate
stage, but we log everything needed to pick instances for phase 2c.
"""

import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tasks.generators import mod_arithmetic, variable_chain  # noqa: E402
from analysis.externalization import find_value_mentions  # noqa: E402
from harness.generate import build_prompt, extract_answer  # noqa: E402


def forward_from(inst, step_idx, new_value):
    """Recompute the chain answer assuming the intermediate at index
    step_idx took new_value. Uses the generator's own op list from meta,
    never re-parses the prompt."""
    ops = inst.meta["ops"]
    if inst.family == "mod_arithmetic":
        p = inst.meta["modulus"]
        val = new_value % p
        for op, a in ops[step_idx + 1:]:  # inters[i] is the result of ops[i]
            val = (val + a) % p if op == "+" else \
                  (val - a) % p if op == "-" else (val * a) % p
        return str(val)
    if inst.family == "variable_chain":
        val = new_value
        for op, a in ops[step_idx:]:  # inters[0] is the start; ops[i]: i->i+1
            val = val + a if op == "+" else val - a
        return str(val)
    raise ValueError(inst.family)


def corrupt_value(v, rng):
    delta = rng.choice([d for d in range(-9, 10) if d != 0])
    return int(v) + delta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--family", default="mod_arithmetic",
                    choices=["mod_arithmetic", "variable_chain"])
    ap.add_argument("--difficulties", default="4,8,16,32")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, gpu_memory_utilization=0.92,
              max_model_len=16384)
    gen = mod_arithmetic if args.family == "mod_arithmetic" else variable_chain
    sp = SamplingParams(temperature=args.temperature, top_p=0.95,
                        max_tokens=args.max_tokens)
    rng = random.Random(0)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out = open(args.out, "w")
    for d in [int(x) for x in args.difficulties.split(",")]:
        insts = [gen(d, s) for s in range(args.n)]
        prompts = [build_prompt(i, "cot", tok, True) for i in insts]
        gens = llm.generate(prompts, sp)

        cont_prompts, metas = [], []
        for inst, prompt, g in zip(insts, prompts, gens):
            trace = g.outputs[0].text
            pred = extract_answer(trace, "cot")
            if pred.strip().lower() != inst.answer.strip().lower():
                continue  # only corrupt correct traces
            # values that appear in the prompt or repeat across steps are
            # ambiguous corruption targets: skip them entirely
            pvals = {v for v in re.findall(r"-?\d+", inst.prompt)}
            all_vals = [v for _, v in inst.intermediates]
            candidates = []
            for idx, (name, value) in enumerate(inst.intermediates):
                if idx < 1 or idx > len(inst.intermediates) - 2:
                    continue  # skip first and last steps
                if str(value) in pvals or all_vals.count(value) > 1:
                    continue
                spans = find_value_mentions(trace, value)
                if spans:
                    # corrupt the LAST clean mention: it is the copy the
                    # continuation is most plausibly reading
                    candidates.append((idx, value, spans[-1]))
            if not candidates:
                continue
            idx, value, (a, b) = rng.choice(candidates)
            # reject corruptions colliding with any prompt value or other
            # intermediate, so downstream matching stays unambiguous
            newv = None
            for _ in range(20):
                cand = corrupt_value(value, rng)
                if str(cand) not in pvals and str(cand) not in all_vals:
                    newv = cand
                    break
            if newv is None:
                continue
            corrupted = trace[:a] + str(newv) + trace[b:]
            cut = a + len(str(newv))
            cont_prompts.append(prompt + corrupted[:cut])
            metas.append((inst, idx, int(value), newv, trace))

        if not cont_prompts:
            continue
        conts = llm.generate(cont_prompts, sp)
        for (inst, idx, clean_v, corr_v, orig_trace), c in zip(metas, conts):
            cont_text = c.outputs[0].text
            pred = extract_answer(cont_text, "cot")
            expect_corr = forward_from(inst, idx, corr_v)
            restated_clean = bool(
                find_value_mentions(cont_text[:200], clean_v))
            out.write(json.dumps({
                "difficulty": d, "seed": inst.seed, "step": idx,
                "clean_value": clean_v, "corrupt_value": corr_v,
                "answer_clean": inst.answer, "answer_if_corrupt": expect_corr,
                "pred": pred,
                "follows_corruption": pred == expect_corr,
                "follows_clean": pred == inst.answer,
                "restates_clean_early": restated_clean,
                "continuation": cont_text[:2000],
            }) + "\n")
        out.flush()
        print(f"gate_b d={d}: {len(metas)} corrupted items", flush=True)
    out.close()


if __name__ == "__main__":
    main()
