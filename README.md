# J-Space and the Chain of Thought: Where Reasoning State Lives

CS 2881R HW0. The report is `report.pdf` in this directory (source: `paper/main.md`).

We ask where a model keeps intermediate results during multi-step reasoning: in J-space, the internal concept workspace read out by the Jacobian lens, or in the written chain of thought. The assignment core runs on Qwen3-4B across GSM8K, MATH-500, and AIME 2024 (the 30 problems of AIME 2024 I and II, `HuggingFaceH4/aime_2024`), with mechanism experiments on synthetic tasks where every intermediate value has exact ground truth.

## Main findings

1. **Models write out every intermediate step even on trivial problems.** The fraction of known intermediate values appearing in the trace is 0.93 to 1.00 from one-step problems up, at every scale tested (1.5B to 671B). There is no difficulty threshold where writing switches on, so the planned onset scaling law had nothing to fit.
2. **Forbidden from writing, models fail after one or two dependent steps** (channel closure verified from token counts; Llama-70B reaches about five). On Qwen3-4B this shows as a large thinking-vs-bare-answer gap on all three benchmarks, and the model overruns its answer-only budget more often as problems harden (0.29 of GSM8K attempts, 0.80 of AIME attempts): told not to write, it tries anyway.
3. **Change one written value in the model's own work and the final answer usually follows the change** (0.84 of items on the causal testbed).
4. **Overwriting the internal state at the edited token controls the answer, whatever value we write into it**, including values the model never wrote (followed 0.76 of the time; a same-size random perturbation does nothing). The state at a written token works like a memory slot that later computation reads.
5. **Stronger models follow the edited value more often, not less** (0.42 at 7B, 0.78 at 671B, 0.97 for Claude Sonnet 4.5), so the written trace stays load-bearing at the frontier.
6. **Edits only matter when the model cannot recompute the value, and recomputability is relative to the model.** The 671B model ignores GSM8K edits (0.10, near floor) because it recomputes those intermediates; Qwen3-4B follows the same edits 0.87 of the time because it cannot. On synthetic chains, where no model can reconstruct the values, edits dominate at every depth.
7. **A J-space ablation at a dose verified harmless on neutral text changes nothing**, in accuracy or in how much the model writes, against a random-directions control at identical dose.
8. **Under token budgets models shorten wording but never drop values** (failure floor near 12 tokens per step), and **parallel state lives in activations while chained state lives on the page** (box tracking succeeds with a fifth of its state written; an internal lesion hurts the internally-stored task about 3x more).

## Repository structure

- `report.pdf`, `paper/main.md`: the report and its markdown source.
- `notes/`: research process. `findings.md` (assignment-scoped findings summary), `plan.md` (design and part-2 plan), `hypotheses.md` (predictions written before running, with what the opposite result would look like), `results-log.md` (dated lab notebook, includes interim readings later revised), `lit-review.md`.
- `src/tasks/`: synthetic task generators (`generators.py`) with self-tests (`test_generators.py`) and deterministic worked-trace builder (`traces.py`).
- `src/harness/`: experiment code. Prompts and experimental settings are in these files. Key ones: `qwen3_bench.py` (Qwen3-4B benchmark sweep and GSM8K edit test), `jspace_ablate.py` (J-space ablation with neutral-text dose calibration), `jspace_patch.py` (J-space patch decomposition, planned), `readback_patch.py` (residual patch and third-value swap), `gate_b_corruption.py` (trace edits), `protection.py` (residual lesion), `api_sweep.py` / `api_readback.py` / `truncation_faithfulness.py` / `gsm8k_readback.py` (API experiments), `precot_decode.py` (answer decodability probe).
- `src/analysis/`: readouts and figures. `figures.py` regenerates all figures; `q3_readout.py`, `jspace_readout.py`, `readback_patch_readout.py`, `gate_b_readout.py`, `protection_readout.py`, `curves.py`, `capacity_law.py`.
- `results/raw/`: key result files (jsonl, one row per generation or item) for every table in the report. `results/figures/`: generated figures. `results/*.csv`: summary tables.
- `experiments/`: one directory per experiment with the exact command and result summary.
- `data/` (gitignored, fetched by instructions below): GSM8K, MATH-500, AIME 2024 local copies.

## Setup

```
python3 -m venv .venv && .venv/bin/pip install vllm transformers accelerate torch numpy pandas matplotlib huggingface_hub boto3
git clone https://github.com/anthropics/jacobian-lens && pip install -e jacobian-lens   # lens library
```

GPU experiments ran on single NVIDIA A10G/L40S instances (24-48 GB). API experiments used AWS Bedrock (DeepSeek V3.2, R1-671B, Llama 3.x) and the Anthropic API (Claude Sonnet 4.5); set credentials in the environment or `.env`.

Datasets:

```
curl -sL https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl -o data/gsm8k_test.jsonl
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('HuggingFaceH4/MATH-500','test.jsonl',repo_type='dataset',revision='6e4ed1a2a79a',local_dir='data')"
# AIME 2024: HuggingFaceH4/aime_2024 revision 2fe88a2f1091, converted to data/aime2024.jsonl (see src/harness/qwen3_bench.py header)
```

Pinned revisions (models, lens, datasets) are listed in the report's Reproducibility section.

## Reproducing the principal tables and figures

Assignment core (Qwen3-4B):

```
python src/harness/qwen3_bench.py --mode sweep --out results/raw/q3_sweep.jsonl        # benchmark ladder table
python src/harness/qwen3_bench.py --mode readback --out results/raw/q3_readback.jsonl  # GSM8K edit test
python src/harness/jspace_ablate.py --model Qwen/Qwen3-4B --calibrate --out results/raw/q3_jcal.jsonl
python src/harness/jspace_ablate.py --model Qwen/Qwen3-4B --family gsm8k --k <K> --alpha <A> --out results/raw/q3_jspace_2x2.jsonl
python src/analysis/q3_readout.py --sweep results/raw/q3_sweep.jsonl --readback results/raw/q3_readback.jsonl
python src/analysis/jspace_readout.py results/raw/q3_jspace_2x2.jsonl
```

Mechanism experiments (tables in the report, in order of appearance): closed-channel table from `src/harness/generate.py` + `api_sweep.py` outputs via `src/analysis/curves.py` and `capacity_law.py`; edit/patch table from `gate_b_corruption.py` and `readback_patch.py` via `readback_patch_readout.py`; cross-model edit table from `api_readback.py`; budget table from `api_sweep.py --conditions budget:N` via `format_readout.py`-style summaries; format table from `format_sweep.py` via `format_readout.py`; lesion comparison from `protection.py` via `protection_readout.py`; J-space calibration and ablation from `jspace_ablate.py` via `jspace_readout.py`.

All figures: `python src/analysis/figures.py` (reads `results/raw/`, writes `results/figures/`).

Report PDF: `pandoc paper/main.md -s --css=<any simple css> -o report.html` then print to PDF (we used headless Chrome).

## External links

- Jacobian lens library: https://github.com/anthropics/jacobian-lens
- Fitted lens artifacts: https://huggingface.co/neuronpedia/jacobian-lens (interactive: https://www.neuronpedia.org/jlens)
- Qwen3-4B: https://huggingface.co/Qwen/Qwen3-4B
- GSM8K: https://github.com/openai/grade-school-math
- MATH-500: https://huggingface.co/datasets/HuggingFaceH4/MATH-500
- AIME 2024: https://huggingface.co/datasets/HuggingFaceH4/aime_2024
- Qwen2.5-7B-Instruct: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
- DeepSeek-R1 distills: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
