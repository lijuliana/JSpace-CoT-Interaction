# Paper skeleton (structure only, no prose yet)

## One-sentence claim

A written intermediate value in a chain of thought is stored at its token
position as a causally addressable memory: later computation reads it back
through attention to that token, answers track whatever value is planted in
that internal state (including values the model never wrote), and models
consult this memory by default, overriding it only when the prompt itself
defines the value and the model is both able and trained to check.

## Title candidates

- Written values are read back: the chain of thought as addressable memory
- The chain of thought is the model's source of record
- Reading what was written: causal memory in the chain of thought

## Sections and the one thing each proves

1. **Introduction.** Claim, headline numbers (third-value 0.74 to 0.94
   across three families with random control at 0.00; knockout 0.90 to
   0.00; verification dissociation 0.96 / 0.95 / 0.54), why it matters:
   when a trace is read back, trace monitoring observes the computation.

2. **Setup.** Arithmetic chain tasks with exact ground truth for every
   intermediate; edit-and-continue protocol; the three interventions
   (text edit, residual-state patch, attention block). One figure:
   protocol diagram.

3. **The state at a written token is an addressable memory** (was R3).
   Table: text-follow / restore / third-value / random for Qwen2.5-7B,
   Phi-3-medium, OLMo-2-7B (+ distills and Qwen3-4B after redos).
   Figure: third-value patch schematic with rates. Proves: the value in
   the state at that position, not the visible token and not correctness,
   determines the answer downstream.

4. **Read-back runs through attention to the value token** (was R2).
   Figure (screenshot candidate): follow rate under each knockout target,
   two models; value 0.000, neighbor at baseline, dots reverting to clean.
   Proves: a specific attention edge carries the value; when the token is
   unreadable the model rederives from the previous line, so read-back is
   the preferred path, not the only one.

5. **Models read the trace instead of checking it** (was R1/R4/R4b).
   Figure: three-condition dissociation per model (value implicitly
   derivable / stated as checkpoint / defined by the prompt), follow rate
   with CIs. Table: the six-model R1 grid. Proves: the trace is the
   default source of record; only a generative definition in the prompt
   triggers verification, verification is targeted at the defined value,
   capacity is necessary for it (absent in 4B and 7B models) and training
   policy decides its use (absent in Llama-70B despite capacity).

6. **Scope and boundary results.** Closed-channel capacity (models fail
   at one to two dependent steps without writing); J-space ablation null
   at calibrated dose; box-tracking dissociation (parallel state stays
   internal). One figure: capacity table or small multiples. Proves the
   claim's scope: the mechanism covers serially dependent state the model
   cannot regenerate from the prompt.

7. **Related work.** Differentiate: 2606.29522 (fine-tuned synthetic
   registers), 2505.04955 (program-variable interventions, no
   localization), 2602.15868 (position paper, no experiments); bounds:
   2404.15758, 2412.01113, 2603.01437.

8. **Discussion and limitations.** Monitoring consequence stated plainly;
   arithmetic-task scope; white-box results at 4B to 14B; behavioral only
   at frontier; verification-policy finding is two families deep, not a
   law.

## Figures (all matplotlib, one style module, column-width readable)

- F1 protocol diagram (edit / patch / knockout on one trace)
- F2 third-value patch rates, three families, with random control
- F3 knockout bar chart, two models, six conditions
- F4 three-condition verification dissociation across models
- F5 (appendix or main) R4 depth curves and R4b anchored/unanchored
- Style: colorblind-safe palette, no chartjunk, direct labels over
  legends where possible, reference look: NeurIPS 2022-2024 orals.

## Appendices

- A: all prompts and templates, verbatim
- B: full per-cell tables with bootstrap CIs (every run in results/raw)
- C: per-run configs (model revision, seeds, sampling, layer bands)
- D: harness details: sdpa masking, patch mechanics, parsing rules
- E: negative and superseded results (thinking-mode confound, truncation
  confound, R4b v1 redundancy, J-space ablation dose rule)

## Writing rules (from CLAUDE_WRITING.md, binding)

Declarative sentences; no em dashes; no "not X but Y"; no rhetorical
questions; no coined metaphors carried across sentences; no lab shorthand
(cell, arm, harness, floor) without plain-word definition; strongest
number first; every phrase readable at first glance.
