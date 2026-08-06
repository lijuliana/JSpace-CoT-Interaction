#!/usr/bin/env bash
# gate b: corrupt the last written mention of a mid-chain value, continue, score
set -euo pipefail
cd "$(dirname "$0")/../.."
MODEL="${1:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"
M=$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')
python src/harness/gate_b_corruption.py --model "$MODEL" --family variable_chain \
  --difficulties 4,8,16,32 --n 150 --out "results/raw/gate_b_var_${M}.jsonl"
python src/analysis/gate_b_readout.py "results/raw/gate_b_var_${M}.jsonl"
