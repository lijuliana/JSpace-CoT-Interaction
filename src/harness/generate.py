"""Generation runner: sweep task instances through a model under the three
phase 1 conditions and log everything needed downstream.

Runs on the GPU box with vLLM. Output is one jsonl per (model, family,
condition) with the instance, the full trace, and the externalization record.
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tasks.generators import FAMILIES  # noqa: E402
from analysis.externalization import externalization_record  # noqa: E402

DIRECT_SUFFIX = ("\nAnswer with only the final answer, nothing else. "
                 "Do not show any working.")
COT_SUFFIX = ("\nThink step by step, then give the final answer on its own "
              "line as 'Answer: X'.")

CONDITIONS = ["direct", "free", "cot"]


def build_prompt(inst, condition, tokenizer, is_reasoning_model):
    text = inst.prompt
    if condition == "direct":
        text += DIRECT_SUFFIX
    elif condition == "cot":
        text += COT_SUFFIX
    msgs = [{"role": "user", "content": text}]
    out = tokenizer.apply_chat_template(msgs, tokenize=False,
                                        add_generation_prompt=True)
    if condition == "direct" and is_reasoning_model:
        # suppress the think block; reported as secondary per plan.md.
        # some R1-distill template versions already open <think> in the
        # generation prompt, so only add what is missing
        if out.rstrip().endswith("<think>"):
            out += "\n\n</think>\n\n"
        else:
            out += "<think>\n\n</think>\n\n"
    return out


def extract_answer(text, condition):
    body = re.split(r"</think>", text)[-1]
    boxed = re.findall(r"\\boxed\{(-?[\w.,]+)\}", body)
    if boxed:
        return boxed[-1].replace(",", "").rstrip(".")
    m = re.findall(r"[Aa]nswer\s*(?:is)?\s*[:=]?\s*\$?(-?[\d][\d.,]*|\w+)",
                   body)
    m = [x for x in m if x.lower() not in ("is", "the", "a")]
    if m:
        return m[-1].replace(",", "").rstrip(".")
    tokens = re.findall(r"-?\d+|\b\w+\b", body.strip())
    return tokens[-1] if tokens else ""


def prompt_values(inst):
    vals = re.findall(r"-?\d+", inst.prompt)
    vals += [w for w in re.findall(r"\b[a-z]+\b", inst.prompt.lower())]
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--family", required=True, choices=list(FAMILIES))
    ap.add_argument("--difficulties", default="1,2,3,4,6,8,12,16,24,32")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conditions", default="direct,free,cot")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, gpu_memory_utilization=0.92,
              max_model_len=16384)
    is_reasoning = "r1" in args.model.lower() or "distill" in args.model.lower()
    gen = FAMILIES[args.family]
    diffs = [int(d) for d in args.difficulties.split(",")]
    conditions = args.conditions.split(",")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for cond in conditions:
            temp = 0.0 if cond == "direct" else args.temperature
            for d in diffs:
                insts = [gen(d, s) for s in range(args.n)]
                for rep in range(args.seeds if cond != "direct" else 1):
                    sp = SamplingParams(
                        temperature=temp, top_p=args.top_p,
                        max_tokens=64 if cond == "direct"
                        else args.max_tokens,
                        seed=rep)  # reproducible sampling per rep
                    prompts = [build_prompt(i, cond, tok, is_reasoning)
                               for i in insts]
                    outs = llm.generate(prompts, sp)
                    for inst, o in zip(insts, outs):
                        trace = o.outputs[0].text
                        pred = extract_answer(trace, cond)
                        ext = externalization_record(
                            trace, inst.intermediates,
                            prompt_values(inst)) if cond != "direct" else None
                        f.write(json.dumps({
                            "model": args.model, "condition": cond,
                            "rep": rep, "difficulty": d,
                            "seed": inst.seed,
                            "answer": inst.answer, "pred": pred,
                            "correct": pred.strip().lower()
                                == inst.answer.strip().lower(),
                            "trace_tokens": len(o.outputs[0].token_ids),
                            "externalization": ext,
                            "trace": trace,
                        }) + "\n")
                    f.flush()
                print(f"done {cond} d={d}", flush=True)


if __name__ == "__main__":
    main()
