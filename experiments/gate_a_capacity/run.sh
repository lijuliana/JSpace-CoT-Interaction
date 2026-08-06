#!/usr/bin/env bash
# gate a (dropped, kept for the record): coarse dose sweep that overshot the cliff
set -euo pipefail
cd "$(dirname "$0")/../.."
python src/harness/gate_a_lesion.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --family variable_chain --difficulty 8 --n 200 --alphas 0.15,0.3,0.5 \
  --out results/raw/gate_a_var_7b.jsonl
