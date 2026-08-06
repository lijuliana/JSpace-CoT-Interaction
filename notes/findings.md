# Findings summary

Research question: When does a model's reasoning live in J-space, the concept workspace read out by the Jacobian lens, and when does it live in the written chain of thought? Does the split move as problems get harder or as either side is constrained?

Setup under the assignment constraints: model Qwen3-4B (HF revision 1cfa9a720891) with the fitted public lens (neuronpedia/jacobian-lens revision a4114d7752d1, qwen3-4b artifact; library commit 581d398613e5). Datasets: GSM8K test (openai/grade-school-math, test.jsonl sha1 4a3eef48d603), MATH-500 (HuggingFaceH4/MATH-500 revision 6e4ed1a2a79a), and AIME 2024, by which we mean the 30 problems of AIME 2024 I and II as distributed in HuggingFaceH4/aime_2024 revision 2fe88a2f1091. The three sets are the difficulty ladder, easy to hard. Ablation doses are calibrated and frozen on neutral text before any task cell; every cell logs token-cap hits and unparseable answers separately from wrong answers, and direct cells log generated token counts to verify the channel was closed.

Terms used throughout:

* **J-space**: the model's active concept workspace, read and intervened on through a fitted Jacobian lens (per-layer maps whose readout is the concepts currently loaded; ablation projects out the preimages of the top active concepts). Our targeted internal instrument.
* **Internal workspace**: the broader construct J-space samples from, the activations of a forward pass. Always present, cannot be switched off; coarser interventions damage it (residual lesion) or overwrite parts of it (residual patch). Findings established with the coarse instruments generalize the J-space picture from concept directions to the full residual stream.
* **Token channel**: the chain-of-thought text written before the answer. Can be closed (force a bare answer), capped (token budget), or edited (corrupt a written value).
* **d**: difficulty, the number of dependent steps in a synthetic problem (chained arithmetic, variable chains, box tracking). Every intermediate value has known ground truth.
* **Externalization**: fraction of those ground-truth values that appear in the written trace.

The argument runs J-space outward, in two parts. Part I is the assignment core on Qwen3-4B and the three benchmark datasets: the free-vs-direct ladder, the calibrated J-space ablation 2x2, and the corruption test on GSM8K worked solutions. Part II is the mechanism supplement on synthetic families, which exist because the benchmark datasets cannot supply what causal work needs, exact ground truth for every intermediate value and a difficulty knob decoupled from output length; the synthetic results explain why the Part I benchmark numbers look the way they do. Coarse residual-level instruments (lesion, full-residual patch) appear as the precursors that motivated the J-space versions and as robustness checks.

Each finding states the manipulated variable, the measurement, and the result.

## Part I: assignment core (Qwen3-4B, GSM8K / MATH-500 / AIME 2024)

### I.1 Free vs direct across the difficulty ladder

Thinking vs bare-answer accuracy (non-truncated): GSM8K 0.99 vs 0.12, MATH-500 0.76 vs 0.19, AIME 2024 0.90 vs 0.00 (n=150/150/30). Bare-answer overrun of the 32-token budget climbs 0.29 / 0.47 / 0.80 down the ladder (told not to write, the model tries anyway), and median thinking length grows 1952 / 3503 / 11819 tokens.

### I.2 J-space ablation 2x2 (ablation on/off x direct/cot), GSM8K

Frozen dose k=8, alpha=0.5. Cot accuracy 0.97 clean, 0.93 jlens, 0.93 random control: no targeted effect. Direct is floored for a 4B on GSM8K (0.00 to 0.03 in every arm), so no asymmetry could appear there; on synthetic chains with direct headroom the same calibrated ablation was also null. The anchor asymmetry does not replicate at a coherence-safe dose.

### I.3 Corrupting written intermediates in GSM8K worked solutions

The answer follows a corrupted intermediate 0.87 of the time (n=84, resample floor 0.00). Together with V3.2's 0.10 on the same protocol, recomputability is model-relative: the 671B recomputes GSM8K intermediates and ignores edits, the 4B cannot and reads them back.

## Part II: mechanism supplement (synthetic families)


### 1. Externalization is saturated from the easiest problems onward

**Variable:** d from 1 to 64, model size 1.5B to 671B. **Measured:** externalization in free generation.

**Result:** 0.93 to 1.00 everywhere, including d=1. No model waits for difficulty before writing intermediate values; there is no spill threshold where internal reasoning gives way to writing. The write policy is saturated from the start.

### 2. Models cannot replace the token channel with internal computation

**Variable:** token channel open vs closed (closed = instructed to output only the final answer, zero reasoning tokens). **Measured:** accuracy vs serial depth.

**Result:** accuracy collapses past one dependent operation. DeepSeek V3.2 (671B) gets 23 percent at d=1 and near zero above. With finding 1, this rules out the overflow picture: writing is not what happens when a sufficient workspace fills up, because a single forward pass never fit a chain in the first place. Read behaviorally, not as proof of zero latent serial capacity; the direct condition also carries prompting and training-distribution effects.

### 3. Subsequent computation follows edited written values

**Variable:** one written intermediate value in a valid trace replaced with a counterfactual (417 to 457, say), model continues from the edit. **Measured:** whether the answer matches the clean value, the edited value, or neither.

**Result:** the answer follows the edit on 84 percent of items (Qwen2.5-7B-Instruct, d=10, n=141). Most of the rest follow neither (later arithmetic slips). Almost none recover the original from the unchanged earlier steps, which are still in context. The written value is not a record of computation done elsewhere; later computation depends on it.

### 4. The residual at the written token is a readable value register

**Variable:** token text stays corrupted, but the residual-stream state at that position is overwritten, either with the clean state from the uncorrupted run or with the state for an arbitrary third value the model never wrote; control arm gets a matched-norm random perturbation. **Measured:** which value the continuation's answer follows.

**Result:** restoring the clean state reverts the answer on 97 percent of affected items (CI 94 to 99), replicates at d=20 (93 percent), and the random control reverts 0 percent. The stronger condition: overwriting the residual with an arbitrary third value makes the answer follow that value 76 percent of the time (CI 70 to 82) and the clean value 0 percent. So the residual at the written token is a readable value register, set it to any value and the downstream computation reads and propagates that value. This rules out "the patch injects the answer," since the injected quantity is a mid-chain intermediate the model never produced and the answer follows it. It holds on the reasoning model too (R1-Distill-Qwen-7B: clean 100 percent, third value 74 percent), where the swap condition is confound-free because the post-think re-solve produces the clean answer, never the arbitrary third value. Caveat: the patch still overwrites the whole residual at that position, but because different injected values produce answers following those values, the value is the causal quantity.

### 5. Reliance on written values increases with model capability

**Variable:** model, same corruption test via API prefill. **Measured:** fraction of answers following the edit.

**Result:** 42 percent (7B distill), 78 percent (DeepSeek V3.2), 97 percent (Claude Sonnet 4.5). One might expect stronger models to hold more in activations and depend less on the trace; the trend runs the other way, and the trace stays load-bearing at the frontier, which matters for monitoring. Not a causal effect of scale, since these models differ in architecture, training, and serving.

### 6. Under token budgets, models cut prose but keep every value

**Variable:** hard output budgets 64 to 512 tokens vs unrestricted. **Measured:** trace length, externalization, accuracy.

**Result:** V3.2 at d=16 compresses traces 466 to 184 tokens with accuracy intact (0.93 to 0.97) and externalization still 0.98 to 1.00. Below roughly twelve tokens per step the model does not drop values and hold them internally; it truncates and accuracy falls to zero. The values themselves are apparently incompressible; only the wording around them is negotiable. The floor is task- and model-specific, not a universal limit.

### 7. Serial and parallel memory demands externalize differently

**Variable:** task structure, serial (each value depends on the previous) vs parallel (five boxes, contents repeatedly swapped). **Measured:** externalization among correct traces.

**Result:** serial chains at d of 16 and up, externalization is exactly 1.00 in every correct trace (n=1935 pooled across three model sizes); box tracking, the model handles sixteen swaps correctly while writing about a fifth of the state. What stays internal is set by the structure of the memory demand, not the amount: serial state lives on the page, parallel state can live in activations. The families differ in more than seriality, so this is evidence for a structural split, not a separation theorem.

### 8. Activation lesions hurt the parallel task about three times more

**Variable:** matched-dose residual-stream lesion during generation, both task types, chain of thought available. **Measured:** accuracy drop.

**Result:** box tracking falls 0.34, variable chains 0.11 (d=2 to 4, n=40 per cell). The task whose state is less externalized is the one fragile to internal damage, the causal counterpart of finding 7. The lesion is blunt (it also disrupts re-reading of written values), so this is supporting evidence, not a clean separation of the two channels.

## Negative and null results

### N1. The starting hypothesis is not supported

The initial hypothesis was that models reason internally at low difficulty and start externalizing once capacity is exceeded. Findings 1 and 2 contradict it: writing is saturated at d=1, and closing the token channel reveals no low-difficulty internal regime to fall back on.

### N2. Damaging the workspace does not induce more writing

If externalization were an online response to internal scarcity, damaging the workspace should push the model to write more. It does not: as lesion strength crosses the accuracy cliff (0.99 clean to 0.01 lesioned), externalization falls and traces disintegrate. No sign the write policy adapts.

### N3. Restricting the token channel does not induce internalization

The mirror test of N2. Under hard budgets the model compresses prose while keeping every value; below the floor it truncates and fails rather than holding values internally. Together with N2: no evidence of adaptive movement of state between channels at inference time.

### N4. Naive corruption tests are confounded in reasoning models

Corrupt a value inside a think block and it propagates through the reasoning, but the answer section may then re-solve the problem from the prompt, so the final answer returns to clean even though the corrupted value was used. This masks read-back when only the answer is measured, and is why findings 3 and 4 use a non-reasoning model as the primary testbed. The swap control in finding 4 recovers a clean reasoning-model result despite this: re-solving lands on the clean answer, never on an arbitrary injected value, so the answer following the injected value is not maskable by re-solving.

### N6. Read-back does not fire on GSM8K, because its intermediates are recomputable

Running the corruption test on GSM8K is essentially null: a corrupted intermediate changes the answer 0.10 of the time against a 0.05 resample floor. This is not a depth effect (read-back is 0.64 to 0.80 across d=3 to 16 on synthetic chains); it is recomputability. A GSM8K intermediate is a shallow function of the problem's givens, recomputable in about one operation, so the model recomputes it and ignores the edit, whereas a chain intermediate requires re-deriving the whole chain, which exceeds the one-step internal ceiling. Read-back carries genuinely serial, non-recomputable state; on problems with shallow-recomputable intermediates it does not appear. This scopes the mechanism's footprint honestly.

### N5. Returns to the clean answer do not establish a persistent internal copy

An earlier reading treated a 0.48 to 0.68 clean-return rate after corruption as evidence of an uncorrupted internal copy. Not identified: the full problem stays in context, so re-derivation from the prompt predicts the same behavior. Whether an internal copy survives writing remains undetermined, and no internal-copy claim is made.

## Known limitations

* Lesion result (finding 8): one model, one dose, n=40 per cell; the lesion also damages re-reading of written values, so the comparison is only clean at low-to-mid difficulty.
* Budget result (finding 6): one model family, and the model is told the cap, so failure conflates cannot-compress with does-not-plan-for-the-cap.
* Frontier read-back (finding 5) is behavioral only; the residual patch needs white-box access and is 7B-scale.
* The patch (finding 4) overwrites the full residual at the position; value-specificity is shown by the swap result (different injected values yield answers following those values) but the co-located representation is not literally isolated.
* Read-back is recomputability-gated (N6), so the mechanism applies to genuinely serial, non-shortcuttable computation and not to problems whose intermediates the model can cheaply recompute; this is the main limit on generality.
* Synthetic tasks buy exact ground truth and controlled depth at the cost of generality; the one real-benchmark test (GSM8K) came back null for the recomputability reason above, so the same allocation pattern is not established for open-ended reasoning.

## Validation

Before extending prior work we reproduced two known results on our setup. Truncation faithfulness (Lanham et al. 2023): forcing an answer after a fraction of the model's own chain of thought gives accuracy monotonic in that fraction, collapsing when late steps are removed (V3.2 near zero to 0.97, Sonnet 0.32 to 1.00), which confirms the chain of thought is causally load-bearing on our tasks. And a linear probe reads the answer from the residual at R-squared up to 0.96 with a control probe at chance, validating the probing machinery the patch relies on.
