# Internal serial capacity

Question: how deep a serial chain does a model complete with no writing (direct condition), and does that depth track model depth or parameters?

Command: direct-condition sweeps across the Llama depth ladder (1B/3B/8B/11B/70B), plus the distill and deepseek data.

Read: `python src/analysis/capacity_law.py --family var`

Result (2026-08-02, partial): d_int correlates with layer count, not log-params, on the non-reasoning models. Full Llama ladder fit in progress.
