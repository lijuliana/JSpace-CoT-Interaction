"""Phase 1.5 protection experiment: accuracy under internal lesion x answer
condition (direct vs cot) x difficulty.

If written tokens hold state that would otherwise live in the residual
stream, the cot condition should degrade less than direct under the same
lesion, and the protection gap should vary with difficulty in one of the
three patterns written down in hypotheses.md. Per-instance version: within
each (arm, d) cell we also store the clean-trace externalization fraction
for the same instance seed, so lesioned accuracy can be regressed on it.

Reuses the WindowLesion/bank machinery from gate A. Entity tracking is the
family of choice: its free-generation accuracy cliff (0.97 at d=1 to 0.18
at d=24 on 7B) leaves room for both protection and its absence, and direct
accuracy is near floor but nonzero.
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tasks.generators import FAMILIES  # noqa: E402
from harness.generate import build_prompt, extract_answer  # noqa: E402
from harness.gate_a_lesion import (WindowLesion, build_bank,  # noqa: E402
                                   generate_with_lesion, measure_damage)


NEUTRAL_TEXTS = [
    "The city grew slowly over several centuries, and its streets still "
    "follow the paths that farmers once walked between their fields.",
    "Water expands as it freezes, which is why pipes can burst in winter "
    "when the temperature drops below the freezing point for long enough.",
    "She opened the window to let in the morning air, then sat down at the "
    "desk and began to read the letters that had arrived the day before.",
    "The orchestra tuned their instruments while the audience found their "
    "seats, and a hush settled over the hall as the conductor walked out.",
    "Trade routes across the desert depended on wells that were spaced a "
    "day's travel apart, and caravans planned their journeys around them.",
    "The recipe called for folding the egg whites gently into the batter "
    "so that the air stayed trapped and the cake would rise in the oven.",
    "After the storm passed, the villagers walked down to the shore to see "
    "what the waves had left behind on the sand and among the rocks.",
    "He kept a notebook of the birds he saw each spring, noting the date "
    "and the weather so he could compare the seasons from year to year.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--family", default="entity_tracking")
    ap.add_argument("--difficulties", default="2,4,8,16")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--alpha", type=float, default=0.08,
                    help="single sub-cliff dose chosen from the fine gate a "
                    "sweep; rerun at a second dose as robustness")
    ap.add_argument("--target-layers", default="12-17")
    ap.add_argument("--control-layers", default="4-9")
    ap.add_argument("--max-new", type=int, default=3072)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=device)
    model.eval()

    def parse_win(s):
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))

    arms = {"clean": None, "target": parse_win(args.target_layers),
            "control": parse_win(args.control_layers)}
    gen = FAMILIES[args.family]
    diffs = [int(x) for x in args.difficulties.split(",")]
    all_layers = sorted(set(arms["target"] + arms["control"]))
    bank_prompts = [build_prompt(gen(max(diffs), 10_000 + s), "cot",
                                 tok, True) for s in range(16)]
    bank = build_bank(model, tok, bank_prompts, all_layers, device)
    # damage meter on NEUTRAL text, not task prompts: on-task KL folds the
    # targeted effect into the meter (the target window is more task-critical)
    # and would under-dose the target arm at matched KL. We report both.
    neutral_texts = NEUTRAL_TEXTS
    task_texts = bank_prompts[:8]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for arm, layers in arms.items():
            lesion = WindowLesion(model, layers or [],
                                  0.0 if layers is None else args.alpha,
                                  bank, lesion_prefill=True)
            with lesion:
                kl_neutral = 0.0 if layers is None else measure_damage(
                    model, tok, neutral_texts, lesion, device)
                kl_task = 0.0 if layers is None else measure_damage(
                    model, tok, task_texts, lesion, device)
                kl = kl_neutral
                print(f"arm={arm} kl_neutral={kl_neutral:.4f} "
                      f"kl_task={kl_task:.4f}", flush=True)
                for d in diffs:
                    for cond in ["direct", "cot"]:
                        for s in range(args.n):
                            inst = gen(d, s)
                            prompt = build_prompt(inst, cond, tok, True)
                            trace = generate_with_lesion(
                                model, tok, prompt, lesion,
                                64 if cond == "direct" else args.max_new,
                                device)
                            pred = extract_answer(trace, cond)
                            f.write(json.dumps({
                                "arm": arm, "alpha": lesion.alpha,
                                "kl": kl, "kl_neutral": kl_neutral,
                                "kl_task": kl_task,
                                "condition": cond, "difficulty": d,
                                "seed": s,
                                "correct": pred.strip().lower()
                                    == inst.answer.strip().lower(),
                                "trace_tokens":
                                    len(tok(trace)["input_ids"]),
                            }) + "\n")
                            f.flush()
                    print(f"arm={arm} d={d} done", flush=True)


if __name__ == "__main__":
    main()
