#!/usr/bin/env bash
# capacity law: direct-condition depth sweeps across the llama ladder;
# distill and deepseek direct cells come from the phase 1 sweeps
set -euo pipefail
cd "$(dirname "$0")/../.."
for MODEL in meta-llama/Llama-3.2-1B-Instruct meta-llama/Llama-3.2-3B-Instruct \
             meta-llama/Llama-3.1-8B-Instruct meta-llama/Llama-3.2-11B-Vision-Instruct \
             meta-llama/Llama-3.1-70B-Instruct; do
  M=$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')
  python src/harness/generate.py --model "$MODEL" --family variable_chain \
    --difficulties 1,2,3,4,6,8 --n 100 --conditions direct \
    --out "results/raw/direct_var_${M}.jsonl"
done
python src/analysis/capacity_law.py --family var
