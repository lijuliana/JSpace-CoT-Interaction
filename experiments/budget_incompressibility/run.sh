#!/usr/bin/env bash
# token budgets vs free generation on v3.2 (bedrock)
set -euo pipefail
cd "$(dirname "$0")/../.."
python src/harness/api_sweep.py --model-id deepseek.v3.2 --family variable_chain \
  --difficulties 8,16,32,64 --conditions budget:64,budget:128,budget:256,budget:512,free \
  --n 30 --out results/raw/budget_var_v32.jsonl
