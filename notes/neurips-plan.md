# neurips submission plan

Status: phase 1, approved direction; runs ordered by information per GPU-hour. R1 to R3 are decision-relevant and run first.

## 1. Audit of assignment-phase results

Graded on three axes: surprising (does it change a prior), defensible (causal, controlled, powered), and taken (already in the literature).

### Strong

**Memory-slot patch (third-value swap).** Overwriting the residual state at a written-value token with a never-written third value makes the answer track that value (0.76 instruct / 0.74 reasoning), random perturbation control at 0.00, correct-restore 0.97, redundant across layer bands, probe shows the answer builds as steps are written. Surprising: the state at a written token is an addressable slot whose content, not just its correctness, is causally read forward. Defensible: the third-value design kills the answer-injection objection; the random control kills the disruption objection. Weaknesses: one task family (variable chains), two closely related models (Qwen 7B instruct/reasoning), n=141/91, d=10/20 only, slot not localized (which positions, which layers minimally, which attention edges read it), not decomposed into J-space vs residual remainder.

**Recomputability-relative reliance.** Same GSM8K edit protocol: V3.2 (671B) ignores edits (0.10, floor 0.05) while Qwen3-4B follows them (0.87, floor 0.00); on synthetic chains no model can recompute, and following instead rises with scale (distill-7B 0.42, V3.2 0.78, Sonnet 4.5 0.97). Surprising twice: reliance on the trace is set by model capacity relative to step cost, not by the task; and scale increases trace-reliance on hard state. Directly relevant to CoT monitoring. Weaknesses: the capacity comparison is across models that differ in everything, recomputability was never manipulated within a task, GSM8K cells are n=63/84, no CIs published, frontier points are single-model observations.

### Sound but confirmatory

**Closed-channel capacity (1-2 dependent steps, ~5 for Llama-70B).** Clean, well-controlled (token-count verification), but it confirms expressivity theory in vivo rather than revealing a mechanism. Supporting-material grade.

**Externalization saturated at ~1.0 everywhere.** Kills our own prior, but reviewers will attribute it to formatting and RLHF norms, and the confound is real (format asks for step results). Framing material, not evidence.

**Budget compression and format results.** Good color, small n, one model. Appendix.

### Suggestive but underpowered

**Serial vs parallel dissociation (box tracking vs chains).** The externalization dissociation (writing predicts correctness for chains, not boxes) is clean; the 3x lesion-fragility ratio rests on one dose, one model, and a lesion that also damages read-back. Needs the targeted version to be defensible as a headline.

### Null

**J-space ablation (top-k active concepts, coherence-safe dose): indistinguishable from random-directions control on everything.** Honest and useful as a boundary on lens-readable state, but a null with one dose and one k cannot carry anything. Needs a dose-response curve and a positive control (a task the ablation does break) to be interpretable even as a secondary result.

## 2. Center of gravity

The paper is the mechanism plus its scope law. One claim, worded to be defensible and to occupy the gap the prior-art check found open:

> In unmodified pretrained reasoning models, a written intermediate value is stored at its token position as a causally addressable memory slot and read back through identifiable attention edges, and the model consults this external memory exactly for values whose rederivation depth from the prompt exceeds its internal capacity; that boundary is model-relative, so larger models rely more, not less, on their written trace for state they cannot recompute.

Differentiation is built into the wording. Unmodified pretrained models separates us from 2606.29522, whose register edits live on a fine-tuned synthetic-task model. Attention edges and the capacity law separate us from 2505.04955, which shows value interventions without localization or scope. The scope clause (exactly for values beyond rederivation capacity) is what no one owns, it is the surprising part, and it is also the honest part: it concedes up front what the bounding papers show, that recomputable state and filler compute are not covered. The monitoring implication is the downstream use the calibration says spotlight papers need: on state beyond a model's internal capacity the trace is faithful by necessity, and that is the regime that matters for oversight.

Honest assessment: current evidence is poster-grade. Two closely related models for the causal core, no within-task manipulation of the boundary, no localization, no cross-family causal replication. Each gap maps to a specific run below.

## 3. Runs

Ordered by information per GPU-hour, not by tier label. The first three are decision-relevant: they determine whether the boundary claim or only the mechanism claim survives, so they run first and everything later inherits their outcome.

### R1. Within-task recomputability manipulation (was A2; highest information, lowest cost)

Same items, two renderings that hold difficulty fixed and vary only rederivation depth from the prompt: operands visible (the edited value is one op from prompt numbers) versus operands consumed into a running state the prompt never restates (rederiving the value requires replaying the chain). Both endpoints are already observed across models (0.87 at high depth on Qwen3-4B, 0.10 at depth one on V3.2); the run converts the contrast into a within-model, within-item experiment. Models: Qwen3-4B, Qwen2.5-7B-it, Llama-3.1-8B. n=300 per cell, 3 seeds. Controls: verify clean accuracy matches across renderings (difficulty confound), edit a restated operand as a non-computed-token control, resample floor per cell. Prediction: follow rate jumps from near floor to above 0.6 within the same items. Kill criterion: if the jump is under 0.2, the boundary claim demotes to an across-model observation and the paper leans on the mechanism arm.

### R2. Read-back localization by attention knockout (was B1; the screenshot figure)

Block attention from post-edit positions to the edited value token and measure edit-following; controls knock out a neighboring non-value token, an operand token, a random token at matched distance, and (responding to the filler-token literature) a variant where the value token is replaced by filler dots so position-only compute is separated from content. If following collapses only when the value token is cut, read-back is localized to identifiable edges. Precedent (attention-flow hubs whose perturbation flips answers, 2606.10646) plus our own effect sizes make this the safest mechanism bet, and the figure (trace with one attention edge cut, answer reverts) is the one people screenshot. Models: Qwen2.5-7B-it and Qwen3-4B first, then the R1 distills. Sweep knockout by layer range and by lag between value token and reading position. n=200 per condition, 3 seeds.

### R3. Patch generality and cross-family replication (was A1, sharpened)

Third-value patch on Qwen3-4B, Qwen2.5-7B-it, R1-distill-1.5B/7B/14B, Llama-3.1-8B, and Gemma-2-9B-it, on variable chains and an entity-state narrative task (natural-language state with exact ground truth; mod-arithmetic retained for edit-following only, since its externalization detector is unreliable). d in {5, 10, 20, 40}, n=300 per cell, 3 seeds, bootstrap CIs. New controls a reviewer will ask for: patch at a non-value position between written values (slot should be specific to value tokens), and third values matched to the original in token count and digit distribution. The minimum that defuses the Qwen-family risk: third-value result with random control on Llama-3.1-8B and Gemma-2-9B-it at n of at least 200 each. Kill criterion: if the patch fails on both non-Qwen families, the mechanism claim scopes to a model family and the paper pivots per the risk section.

### R4. Capacity-boundary law (was A3, now operationalized)

The draft's c(s) minus r(m) was not measurable as written; this is the measurable version. Define rederivation depth delta of an edited value as the length of the minimal dependency chain from prompt-visible numbers to that value: exact by construction on synthetics (delta equals edit position), annotatable on GSM8K from the calculation graph (delta is one or two for most intermediates, which is why edits die there). Define internal capacity r(m) as the closed-channel depth at which bare-answer accuracy crosses 0.5, which our existing instrument already measures (one to two for most models, about five for Llama-70B). The law: edit-following is low for delta at or below r(m) and high above it, with the crossover at r(m). Run edit-following versus delta curves for five white-box models plus API models, overlay the independently measured r(m) per model. If the crossovers line up, this is the title figure: two independent measurements predicting a third across a model ladder. Kill criterion: crossovers exist but do not track r(m); then the paper reports the boundary without the capacity identification and says so.

### R5. Mechanism depth on the lens (was B2, B3, B4)

J-space patch decomposition (split the effective patch into lens-readable component and remainder, apply each alone), slot protection (restore only the slot subspace during an effective lesion), and the ablation dose-response (k in {8, 16, 32, 64} crossed with strength past the coherence rule, neutral-text damage on the same axis, plus a positive control task the ablation demonstrably breaks). These locate the slot relative to J-space and turn the null into a curve. They run after R1 to R3 because their interpretation depends on where the slot is confirmed to be.

### R6. Scale and completeness (was Tier C)

Full benchmark sets (GSM8K 1319, MATH-500 complete, AIME 30) crossed with thinking and bare modes, 3 seeds, on Qwen3-4B and R1-distill-14B. Frontier edit ladder through APIs, n of at least 200 per model, Claude, DeepSeek, and Llama families, recomputable and non-recomputable variants with CIs; this doubles as the frontier arm of R4. Everything currently under n=150 rerun at n of at least 300.

### Surprise budget

Stretch experiments, each foreshadowed by an existing result, each with a prediction and a kill criterion. Budgeted after R1 to R4, before R6 completes.

- **S1. Slot algebra.** Patch two different never-written values into two different slots in one trace and check the answer equals the correct function of both. Foreshadowed by single-slot third-value at 0.76 and by layer-band redundancy. Prediction: joint follow at or above the product of single-slot rates. Kill: joint follow well below the product; then slots interact and we report single-slot addressability only, which is still the core claim.
- **S2. Live mid-generation edit.** Edit a value in the KV-visible trace while the model is still generating and watch the answer flip in-flight, including the streaming probe showing the answer representation updating step-locked to the edit. Foreshadowed by text-edit following at 0.84 and the probe build-up from 0.27 to 0.83. Prediction: post-edit trajectories converge to the edited value at near the static rate. Kill: in-flight edits are ignored; that itself is reportable as a commitment effect, at lower prominence.
- **S3. The boundary moves but never closes.** Extend the R4 curve to frontier APIs: measure follow versus delta for Sonnet-class and V3.2 and locate their crossover. Foreshadowed at both ends (0.10 at delta one or two, 0.78 and 0.97 on deep chains). Prediction: frontier crossover sits at higher delta but exists, meaning no model escapes trace-dependence, it only moves the boundary. Kill: no crossover found within measurable delta for frontier models; then the monitoring claim scopes to open-weight scales.

### Cost sketch

R1 and R2 are days on the current single-GPU instance class; R3 to R5 want two or three 48GB-plus GPUs for two to four days (the 14B and Gemma runs fit in 48GB bf16); R6 API spend is modest. The binding resource is harness engineering, not compute.

### Risks and pivot

If R1 fails, the boundary claim demotes and the paper is the memory-slot mechanism with localization (R2, R3, R5), a solid mechanistic paper with a harder spotlight case. If R3 fails outside the Qwen family, the honest outcome is a family-scoped claim and the pivot target is the serial versus parallel dissociation with the targeted instruments from R5. Decision point after R1 to R3, before R6 spend. R2 failing (no localized edge) is informative rather than fatal: read-back would be distributed, the claim drops the attention-edge clause, and the patch evidence still stands.

## 4. Calibration against recent spotlight-level work

Verified against venue pages (ICLR/NeurIPS/ICML 2024-2026 orals and spotlights). What separates spotlight from poster in this subfield: multiple converging lines of causal evidence (single ablation reads as poster), a scale or frontier-model existence proof, a sharp and explicitly scoped claim (overclaiming is punished even in accepted papers), a downstream use for the mechanism, and honest negative results, which reviewers reward. Figure quality is table stakes, not a differentiator. Reference points: SAE scaling laws (ICLR25 oral), sparse feature circuits with the SHIFT debiasing application (ICLR25 oral), entity-recognition directions gating hallucination (ICLR25 oral), CoT verification via attribution graphs (ICLR26 oral). Notably, the most famous CoT-faithfulness results are poster or preprint tier.

Prior-art check on our claim: no published paper claims the full external-memory-with-causal-read-back thesis on natural models. Must differentiate from 2606.29522 (causal register edits, 80-91 percent follow, but a fine-tuned Qwen-7B on synthetic data only) and 2505.04955 (CoT tokens as program variables, partial read-back). The framing sentence itself is owned by a no-experiments position paper (2602.15868), and the claim is bounded by filler-token compute (2404.15758), pre-CoT answer decodability (2412.01113, 2603.01437), and hidden-state patching (2604.23351). The open combination, which this plan targets: natural pretrained reasoning models, attention-level causal read-back evidence, and a capacity law tying closed-channel working-memory limits to trace reliance, with CoT monitoring as the downstream use.

## Phase 2 (after runs)

Rewrite from scratch. Structure before prose: one-sentence claim, one figure per section each proving one thing, appendix holds full tables, prompts, and per-run configs. Real matplotlib figures, consistent style, readable at column width. Writing rules unchanged: no em dashes, no llm-isms, nothing that reads generated.

## Revisions after the first result wave (2026-08-06)

R1 and R2 landed as planned and the mechanism arm is ahead of schedule (two-model knockout localization at 0.000 vs baseline with neighbor intact). The boundary arm changed shape: the R4 depth sweep is non-monotone at low depth and R4b shows stated checkpoints are ignored at every distance, so the scope law is not follow-versus-depth with a capacity crossover. The defensible law is categorical: models verify a written value only against a generative definition in the prompt, never against implicit derivability or stated equalities, with capacity necessary for the override (absent in 4B/7B) and training policy deciding its use (absent in Llama-70B despite capacity). R4's planned r(m)-crossover figure is replaced by a three-condition dissociation figure (derivable / stated / generative) per model. The kill criterion in R4 (crossover does not track r(m)) fired in the informative direction: the paper reports the dissociation and says so plainly. Remaining runs unchanged: R3 generality, R5 lens arm, R6 scale, S1 slot algebra (running), S2 optional.
