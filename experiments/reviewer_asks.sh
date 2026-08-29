#!/bin/bash
# two archival-reviewer asks: (1) filler-token control for the no-fallback
# claim, (2) j-space dose sweep bracketing the coherence-safe dose.
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python

# (1) filler control: direct vs dot-prefilled direct, two models
$PY src/harness/generate.py --model Qwen/Qwen2.5-7B-Instruct \
  --family variable_chain --difficulties 1,2,4,8 --n 30 \
  --conditions direct,filler --out results/raw/filler_var_7b.jsonl
$PY src/harness/generate.py --model Qwen/Qwen3-4B \
  --family variable_chain --difficulties 1,2,4,8 --n 30 \
  --conditions direct,filler --out results/raw/filler_var_q3.jsonl

# (2) dose sweep at k=16, alphas bracketing the tested 0.5
for A in 0.125 0.25 0.5 0.75 1.0; do
  $PY src/harness/jspace_ablate.py --model Qwen/Qwen2.5-7B-Instruct \
    --k 16 --alpha $A --family variable_chain --difficulties 1,2 --n 40 \
    --out results/raw/jsweep_a${A}.jsonl
done
echo ALL DONE
