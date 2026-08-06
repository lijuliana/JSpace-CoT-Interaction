# Results log

Append-only. Dated entries, newest last. Interpretations here are working notes, not conclusions.

## 2026-08-01, Frontier behavioral sweeps (bedrock)

Models: DeepSeek R1-671B (us.deepseek.r1-v1:0), DeepSeek V3.2 (non-reasoning counterpart). Families: mod arithmetic (p=97), variable chains. n=30 per cell, free and direct conditions, full data in results/raw/p1_*_r1_671b.jsonl and p1_mod_v32.jsonl.

Accuracy: R1-671B free holds ~100 percent through d=48 (mod) and d=64 (var); V3.2 free the same through d=48 with traces roughly 40 percent shorter at matched d. V3.2 direct is near zero at every d including d=1 (23 percent). Raw inspection of d=1 direct outputs shows genuine failures, not harness artifacts: bare numeric answers, with errors like reporting -18 where the subtraction was done but the mod reduction skipped. One forward pass reliably executes one arithmetic op here but not two.

Externalization: at ceiling (fraction ~1.0) in free generation for both models at every difficulty measured. No load-dependent onset is visible in free generation on these families; the write policy is saturated from d=1. This was one of the anticipated outcome patterns (hypotheses.md, phase 1 outcome a). Consequences: the onset axis must come from constrained regimes (token budgets), interventions, and the habit-vs-necessity gap, defined as the distance between where models start writing (immediately) and where writing becomes necessary (the direct cliff, which for V3.2 is d=1).

Matcher calibration (permutation control, matching each trace against a different instance's intermediates at the same difficulty): variable chains are clean, false-positive fraction 0.00 to 0.08 across d=2 to 64 against a true fraction of 1.00. Mod-97 is not clean at high d: false-positive fraction climbs from 0.10 (d=1) to 0.72 (d=48) because long traces mention most of 0..96 by chance. Mod-97 externalization numbers therefore need chance-correction, and variable chains are the primary externalization family going forward. Candidate fix for future mod runs: larger modulus.

Caveat logged: the surface matcher says written; whether written values are read back is gate B's question, and the causal definition of externalization (corrupt the written value, see if the answer moves) remains the ground truth to reconcile against on a subsample.

## 2026-08-01, Gate b (read-back) and phase 1 on r1-distill-7b

Gate B, variable chains (the collision-free family), n=128 to 149 per difficulty: corrupting the last written mention of a mid-chain value flips the final answer in 31 percent of continuations at d=4, 42 at d=8, 50 at d=16, 36 at d=32. At d=32 the drop coincides with restates_clean jumping to 0.45: the model increasingly notices the edit and reasserts the clean value early in the continuation. Reading: written values are causally live (the CoT-as-projection position predicts near zero and is refuted at this scale), but the internal copy persists alongside (follows_clean 0.48 to 0.68 throughout), and at high d an active cross-check between tiers appears. The picture so far is redundant storage with verification, not strict write-then-evict. The eviction probe (2b) now has a sharper question: not whether the internal copy disappears, but whether its precision degrades once the written copy exists.

Gate B, mod-97: follows_corruption collapses from 0.44 (d=6) to 0.01 (d=12). Not interpreted: the matcher's false-positive rate at these difficulties means many corruptions likely hit coincidental number mentions rather than the step value. Mod-97 gate B needs collision-validated targeting before it counts.

Phase 1, 7B, free condition: mod arithmetic externalization runs 0.78 to 0.91 below d=6 and reaches 0.99 by d=12 (chance-correction pending given the mod-97 matcher issue); variable chains sit at 0.93 to 1.00 everywhere. Direct condition collapses by d=2 on all families (7B cannot do two serial ops without writing, matching the frontier result). Entity tracking gives the best difficulty cliff in free generation (0.97 at d=1 down to 0.18 at d=24) and is the designated family for the protection experiment; its externalization fraction is unmeasurable with the current matcher because every object name also appears in the prompt, so it needs a pattern-based measure (planned fix, not blocking).

## 2026-08-01, Token-budget sweep, v3.2, variable chains

Budgets 64/128/256/512 tokens vs free, d=8/16/32/64, n=30 per cell (results/raw/budget_var_v32.jsonl). Three regularities. First, prose compression: under a binding but sufficient budget the model cuts trace length up to ~2.5x versus free (466 to 184 tokens at d=16) with accuracy intact (0.93 to 0.97). Second, content incompressibility: externalization fraction stays at 0.98 to 1.00 through that compression; the model removes filler words, never intermediate values. Third, a floor near 11 to 12 tokens per step: once the budget falls below roughly 12 x d, the model does not skip values, summarize, or shift computation internally; it writes at its floor rate until truncated and accuracy drops to exactly zero (externalization fraction then tracks budget/need: 0.72, 0.36, 0.17 down the difficulty column at budget 128). Accuracy per cell is essentially binary.

Extraction check (from review): at the wall (budget 64, d>=32, all 60 items) every trace is truncated with no answer marker, the extractor's fallback grabs garbage tokens, and none of them accidentally match the correct answer (0 of 60), so accuracy exactly zero there is genuine truncation failure, not lucky matching or a grading artifact.

Reading: the write policy is elastic in verbosity and inelastic in content. Written values behave like incompressible cargo, consistent with load-bearing external memory rather than narration. Under external-channel pressure this model truncates rather than internalizes, meaning no internal substitution capacity is available at these depths, the sharpest behavioral statement yet of the hierarchy's lower tier being mandatory. Caveats: single model, one family, instruction-based budgets (the model was told the cap, so floor behavior conflates cannot-compress with does-not-plan-for-cap); a reasoning-model version and a local 7B version with hard caps and no instruction are the natural follow-ups.

## 2026-08-02, Gate a first pass, ladder sweeps, and the necessity result

Gate A, coarse doses (alphas 0.15/0.3/0.5, 7B, variable chains d=8): all doses landed past the accuracy cliff (clean 0.99; target 0.01 and control 0.22 at the lowest dose). Two observations survive the overshoot: the mid-stack target window is much more task-critical than the shallow control window at similar KL (0.632 vs 0.538), and past the cliff externalization drops rather than rises, with traces ballooning into incoherence, the expected signature of a destroyed workspace being unable to organize writing at all. The compensation question is undecided at sub-cliff doses; a fine sweep (alphas 0.02 to 0.12) is running.

Ladder (1.5B/7B/14B, variable chains, free): accuracy at d=48 is 0.16 / 0.81 / 0.87; externalization among all traces declines with d for 1.5B only (0.54 at d=48). Conditioning on correctness dissolves that decline: externalization among correct traces is exactly 1.00 for every model at every d of 16 and above (n=1125 correct traces pooled), against 0.42 to 0.62 among wrong traces. No correct deep trace anywhere in the ladder omits a single intermediate value. Complete externalization behaves as a necessary condition for depth, and the apparent internalization of the small model is trace disintegration among its failures.

## 2026-08-02, Serial vs parallel dissociation (entity tracking rescored)

With the pattern-based matcher (box-object binding statements), entity tracking on 7B shows externalization among correct traces falling from 0.83 (d=4) to 0.20 (d=16), and ext among correct is statistically indistinguishable from ext among wrong at every d (0.20 vs 0.17 at d=16). Contrast with variable chains, where ext among correct is exactly 1.00 at d of 16 and above and wrong traces sit near 0.5. So externalization predicts success on the serial family and not at all on the parallel-storage family, and models demonstrably succeed at 16-move entity tracking while writing a fifth of the state bindings. Reading: internal capacity is wide but shallow. Parallel state (many simultaneous bindings) lives comfortably in activations; chained serial updates do not survive a forward pass, which is what forces the external tier. This is the transformer-expressivity prediction showing up behaviorally in one model with matched task surface. Caveat: matcher recall differs between families, so only the within-family predictiveness contrast is claimed, not absolute levels. The lesion experiments can now target this dissociation directly: internal lesions should hurt entity tracking (internal storage) more than variable chains (external storage) at matched difficulty, which the protection experiment measures.

## 2026-08-02, Answer extraction fix and cross-family results

Found via Llama 70B spot-check: traces ending "The final answer is: $\boxed{X}$" were graded by capturing "is". extract_answer now handles boxed and comma-grouped answers; every stored file was rescored in place (backups kept). Impact: Llama free-condition numbers were badly understated (70B at d=16 was 0.00, is 1.00); ladder numbers moved by at most a few points; the necessity result is unchanged and now stands at ext|correct exactly 1.000 over 1,935 correct traces at d of 16 and above across 1.5B/7B/14B.

Cross-family (Llama 3.1-8B, 3.3-70B, variable chains): 70B free is perfect through d=48; 8B holds 0.93 at d=48. The notable number is the direct cliff: 70B holds 0.93 direct accuracy at d=4 and collapses at d=8, while every other model measured (including R1-671B and V3.2) collapses at d=1 to 2. First clear cross-model variance in internal serial capacity, giving the onset law real variance to fit. Externalization in free generation is at ceiling for both Llamas, matching the saturation picture; the models that need less writing do not write less.

## 2026-08-02, Internal serial capacity, Llama depth ladder

Direct-condition d_int (largest depth with >=0.5 no-CoT accuracy) across Llama 3.x: 1B=0, 3B=0, 8B=1.9, 11B=0, 70B=5.4. Within this family depth and log-params correlate with d_int at 0.90 and 0.84, so they are collinear and the ladder cannot separate them. Two honest caveats: the 11B is the vision-augmented model and reads as an anomaly (0, below the 8B), and small models failing at d=1 means one in-context +/- step already exceeds their reliable no-CoT capacity.

Correction to the earlier depth-not-params note: the only evidence pointing to depth over params is the cross-family contrast, Llama-70B (80 layers, 70B) reaches d_int~5 while DeepSeek V3.2 and R1-671B (61 layers, 671B) sit near the floor. That is suggestive but confounded by architecture family and training regime, so the capacity law is downgraded from a claim to an open question. What would settle it: a set matched on parameter count but varying depth (looped or depth-scaled transformers), which we do not have. Recorded so the writeup does not overclaim.

## 2026-08-02, External representation geometry (format sweep, V3.2, variable chains)

Four scratchpad formats requested by instruction, same instances (format_var_v32.jsonl, n=30 per format-difficulty). Result is a clean ordering by accuracy per token and a within-format dose-response.

- code_eval (code-style assignments with the evaluated value written as a comment, "b = a + 52  # 399") is the efficient optimum: 100 percent accuracy through d=48 at 555 tokens, versus prose 1105 and state 3848 at the same depth. Accuracy per 100 tokens is 2 to 10 times prose.
- code (same style but values not evaluated, symbolic only) suppresses value-writing at low depth and fails in proportion. Per-instance correlation of externalization fraction with correctness is +0.74 (n=150): when the model writes under half the needed values accuracy is 0.58, when it writes 90 percent or more accuracy is 1.00. The model's compliance with "no evaluation" erodes as depth rises (ext climbs 0.57 at d=4 to 0.97 at d=32) and accuracy climbs with it. A format that nudges the model away from writing values makes it fail exactly to the degree it complies.
- state (verbose running state dump after every step) is the inefficient extreme and collapses at d=48 (accuracy 0.07, 3848 tokens): restating all prior values each step wastes the budget and degrades. Over-externalization hurts as much as under.

Reading: the payload that matters is the evaluated value, not the operation and not the prose around it. The efficient external memory is a compact value store (code_eval); writing operations without values (code) reverts toward internal-only failure, and the within-code dose-response is behavioral causal evidence that value-externalization produces correctness. This is the external-geometry section and it has a direct design implication: a compact value-carrying scratchpad format raises effective external capacity several fold over prose, and a well-used external memory is a readable one, which is the monitorability upside.

## 2026-08-02, Readback patch, teacher-forcing diagnostic (methodological, with a finding)

The residual patch-back experiment on R1-distill-7B hit the teacher-forcing distribution problem the internal review predicted, and the diagnostic traces show something worth keeping. Fed a corrupted worked trace as its own prior output, the reasoning model continues the chain inside the think block and reaches the corruption-consistent value (for instance a trace with a corrupted step reaches l = z - 91 = 391, exactly the corrupt-forward answer), so within the ongoing computation it does read back the corrupted written value. But at the close of the think block the model restarts and re-solves the problem from scratch in its answer section, and that fresh solve, not the corrupted continuation, sets the final answer. So a reasoning model's think block propagates a corrupted written value while its post-think answer self-corrects. This is a real observation about how the two phases of a reasoning model relate, and it also means the clean read-back decomposition should run on a non-reasoning model whose native output is a worked trace with no restart. Moving the patch experiment to Qwen2.5-7B-Instruct; the gate B read-back result stays on the reasoning model and this diagnostic explains why its follows-clean share is substantial (the post-think self-correction).

## 2026-08-02, Readback patch on Qwen2.5-7B-Instruct, d=10 (the causal centerpiece)

n=141 items (variable chains, mid-chain value corrupted by +40, 5 samples each, residual patch on layers 10-19 at the corrupted token position and onward). Results:

- Corrupting the written value flips the final answer on 84 percent of items (corr_follows_corruption mean 0.83; 119 of 141 at >=0.5). The 22 items without the effect follow neither answer (follows_clean 0.03), so they are arithmetic slips in the continuation, not recomputation from operands. Essentially nothing recomputes the intermediate from the unchanged operands; the model uses the written value.
- On the 119 items with an effect, overwriting the residual at the corrupted token position with the clean state (token string still corrupt) reverts the answer to clean 0.97 of the time [0.94, 0.99].
- A matched-norm random-direction patch at the same positions reverts 0.00 [0.00, 0.01] and leaves the answer following the corruption 0.92.

Reading: the downstream computation reads the intermediate value from the value-bearing residual at the written token, and this pathway is causal and specific. Two alternatives are ruled out in the same design: recomputation from operands (corruption would then have no effect, but it flips 84 percent) and a generic "any perturbation resets it" account (the matched-norm random patch does nothing). This is the clean single-pass demonstration of token read-back the reasoning model could not give because of its post-think re-solve.

Replication and localization. At d=20 the result holds (103 of 141 with an effect, restore reverts 0.93 [0.88, 0.97], random 0.00). The layer-band sweep does not localize to a narrow depth: restoring an early band (layers 0-9), the mid band (10-19), or a late band (20-27) each reverts the answer about equally (0.96, 0.97, 0.94), with the random control at 0.00 throughout. The honest reading is that the written value is redundantly carried in the residual at the token position across the whole stack, so overwriting any contiguous band to clean is sufficient because the cumulative residual reconstructs the clean value downstream. This is robustness of the representation, not a sharp circuit localization, and we report it as such.

## 2026-08-02, Dag necessity test (inconclusive, recorded honestly)

The plan was to de-circularize the necessity result on DAG reachability, where the answer is a node label and the hop nodes are not accumulated into it. Result: hop-node externalization is at ceiling (1.00) for both correct and wrong traces at every difficulty (d=2 to 10, accuracy 0.94 down to 0.78). This does not give the clean contrast variable chains gave (where wrong traces sat near 0.5). The reason is structural: every node label already appears in the prompt edge list, and a reasoning model mentions many node labels while searching the graph, so "the hop node appears in the trace" is true regardless of correctness and does not measure externalization of a computed quantity. The DAG task cannot de-circularize the necessity claim.

What actually addresses the circularity is the read-back experiment, not this test. Whether the written value appears "by construction" is beside the point once we have shown, causally, that corrupting it changes the answer 84 percent of the time and the model does not recompute from operands. The value is used, which is the load-bearing claim. The position decomposition (early intermediates written at 1.00, not only the format-forced late ones) is the supporting behavioral evidence. We drop the DAG necessity test and let read-back plus position decomposition carry the point.

## 2026-08-03, Corrections to the gate B reading (from review)

Two interpretation fixes to the 2026-08-01 gate B entry, which over-read the follows_clean number.

First, follows_clean (0.48 to 0.68) was called evidence that an internal copy persists. It is not. The full problem statement is still in context during the corrupted continuation, so the model can re-derive the clean value from the prompt with no internal copy at all. The behavioral data cannot separate an internal copy from prompt re-derivation. The reasoning-model diagnostic makes this concrete: the post-think section re-solves the problem, which is re-derivation from the prompt. So follows_clean is consistent with re-derivation and is not evidence for a stored internal copy. The earlier "redundant storage with verification" phrasing is withdrawn (already dropped from synthesis and paper). Separating internal copy from prompt re-derivation would need an attention-knockout that blocks the prompt region versus the written-value tokens; we did not run it, and we do not make the internal-copy claim.

Second, the d=32 restates_clean jump was read as verification. Same problem: it may be re-derivation from the prompt, not a cross-check against internal state, and the detector used a 200-character window that at high step density catches the clean value sooner by chance. Dropped.

Also logged for completeness. Gate B yield falls with difficulty (corruptible items from 150 attempted: 149 at d=4, 146 at d=8, 138 at d=16, 128 at d=32), because only correct traces are corrupted, so higher-d cells are selected toward the easier instances of that difficulty. The flip-rate curve should be read with that selection in mind. Corruptions also mix two cases we did not separate: corrupting the value on its own computation line (which partly probes local arithmetic consistency) versus on a later re-reference (cleaner read-back); the clean causal separation comes from the read-back patch, not gate B.

Net effect on the argument: none of the load-bearing claims depended on the internal-copy reading. The read-back patch (which corrupts a written value and shows the answer follows the residual at that position, on a model that does not re-derive) is the causal result, and it is unaffected.

## 2026-08-03, Protection reworked after review, KL meter subtlety

After fixing the lesion to fire during prefill (so direct and cot face the same squeeze) and moving the damage meter to neutral text, the alpha=0.10 target lesion gives kl_task=0.145 (a real dose, versus the old 0.03 that moved nothing) and kl_neutral=1.30. The large gap between the two meters is itself worth noting: the lesion resamples from a bank of reasoning-state vectors, so applying them to neutral prose is heavily off-distribution and inflates neutral KL, while applying them to task context is milder. So neither meter is a clean "generic damage" number: task KL folds the targeted effect in (the reviewer's point), neutral KL is inflated by bank-versus-text mismatch. We report both and lean on the matched-damage control arm (same alpha, different layer window) rather than on either KL alone for interpretation, and match cross-family on neutral KL since it uses the same neutral text for both families. Full dissociation result to follow when both families finish.

## 2026-08-03, Protection dissociation result (corrected lesion, alpha 0.10)

Both families, clean/target/control arms, n=40 per cell, lesion firing during prefill and decode, dose kl_task ~0.15 (target and control matched within ~0.01). The comparison that matters is at difficulties where both families still have accuracy headroom, since entity-tracking clean accuracy floors quickly (0.97, 0.60, 0.12, 0.03 at d=2,4,8,16) while variable-chain clean stays high (0.97 through d=8). In the headroom cells (d=2, 4):

- entity-tracking CoT accuracy drops by 0.34 under the target lesion and 0.33 under the matched-damage control.
- variable-chain CoT accuracy drops by 0.11 under target and 0.04 under control.

So an internal residual lesion hurts entity tracking about three times as much as variable chains (target), and the family gap holds for both lesioned layer windows, which is the point: entity tracking (parallel state living in the residual stream) is far more fragile to an internal lesion than variable chains (serial state redundantly written on the page). This is the causal side of the serial/parallel dissociation. The lack of target-versus-control specificity is expected and not damning here, because the hypothesis is about the residual stream broadly, not a single circuit, and both windows sit in that stream.

Honest caveats. First, the lesion is blunt: it damages the re-reading of written values as well as the internal workspace, so it is not silent on chains either. This shows at d=8, where chain CoT drops 0.38 under the target lesion (the chain is now long enough that corrupting the re-read of eight written values accumulates), while entity at d=8 is already floored and cannot drop further. So the clean family comparison lives at low-to-mid difficulty; at high difficulty the blunt lesion hits both tiers and the floor effect on entity muddies it. Second, n=40 per cell, single model, one dose. The result is supporting evidence for the dissociation, consistent with the read-back patch and the behavioral dissociation, not a stand-alone clean causal claim, and the paper frames it that way.

Net: the dissociation prediction (internal lesion hurts parallel storage more than serial chains) holds where it can be cleanly measured, at roughly a three to one ratio, robust to the lesioned window.

## 2026-08-03, Frontier read-back (behavioral, via API assistant prefill)

We cannot patch residuals through an API, but the behavioral half of the read-back test runs at frontier scale: ask the model to solve a variable chain writing each value, corrupt one written value, prefill the assistant turn with the corrupted partial trace, and let it continue. DeepSeek V3.2 (671B), d=10, n=60: corrupting a written value flips the final answer 0.78 of the time (follows_clean 0.12, neither 0.10). So a frontier model reads its own written value back rather than recomputing from the operands, at a higher rate than the 7B distill (which was ~0.42 at d=8). The read-back mechanism is not a small-model artifact. Claude Sonnet 4.5 (same protocol) flips 0.97 of the time (follows_clean 0.03, neither 0.00). So read-back reliability increases with model capability: 7B ~0.42, V3.2 0.78, Sonnet 4.5 0.97. More capable models depend on their written intermediates more, not less, which cuts against the intuition that stronger models would keep more in their heads and matters for monitorability (the trace stays load-bearing at the frontier). Caveat: this is the behavioral half only (answer follows the corrupted token); the residual-patch causal isolation is the 7B result, since APIs give no white-box access.

## 2026-08-04, Hardening round: replication and real-benchmark generality

Closing the gaps flagged in hardening.md.

Replication A2 (truncation faithfulness, Lanham et al.). Forcing the model to answer after only a fraction f of its own CoT, on variable chains d=12. V3.2 accuracy: 0.00 up to f=0.6, 0.03 at 0.8, 0.28 at 0.9, 0.97 at 1.0. Sonnet 4.5: 0.32 at f=0 (it solves a third of d=12 chains with no CoT, real internal capacity), dips through the middle, then 0.86, 0.98, 1.00 at f=0.8, 0.9, 1.0. Both curves are monotonic in the load-bearing region and collapse when the late steps are removed. This reproduces the Lanham qualitative result (CoT is causally load-bearing, truncation before the decisive steps destroys accuracy) on our setup, which is the harness anchor we were missing. On strictly serial chains the curve is close to a step function, the extreme-faithful end of the spectrum.

Real-benchmark read-back (fix C, GSM8K) came back NEGATIVE, and the negative is the interesting part. Corrupting a written intermediate on GSM8K (>=5 calculation steps, n=63) changes the final answer only 0.10 of the time, against a 0.05 no-corruption resample floor. Essentially no read-back. The edit lands (verified) but the model recomputes the intermediate from the question and ignores the corruption.

Why: it is not depth. A synthetic depth sweep shows read-back stays high across the board, 0.80 at d=3, 0.64 at d=5, 0.78 at d=8, 0.76 at d=12, 0.70 at d=16. Even a three-step chain is read back. The difference between the chains and GSM8K is recomputability. A variable-chain intermediate is the running total, recomputable only by re-deriving the whole chain from the start, which the model cannot do internally (the one-serial-step ceiling), so it reads the written value back. A GSM8K intermediate is a shallow function of the problem's givens, recomputable in about one operation from numbers still in the prompt, so the model recomputes it and the corruption does nothing.

Finding (new, and a scoping of generality): read-back is gated by recomputability, not by raw depth. The model reads a written value back precisely when that value cannot be cheaply recomputed from what is still in context. It is the mechanism for carrying genuinely serial, non-shortcuttable state, not a universal habit. Honest consequence: on real problems whose intermediates are shallow functions of the inputs, a large share of GSM8K among them, read-back does not fire, so the mechanism's practical footprint is narrower than "all chain-of-thought." Caveat: a controlled recomputability manipulation (matched depth, intermediate recomputable vs not) would nail the threshold; we have the chains-vs-GSM8K contrast plus the structural argument, not a within-task manipulation.

## 2026-08-04, A1 probe: machinery validated, pre-CoT shape confounded

Ridge probe from the residual to the final answer, Qwen2.5-7B-Instruct, variable chains d=12, n=400, best layer 4. Decodability (R^2 on held-out instances) rises from 0.80 at step-fraction 0.17 to 0.96 at 1.0; the control probe (predicting a different instance's answer) is strongly negative (-0.8 to -1.3), well below chance, so there is no leakage.

What this validates: the probing and activation-extraction machinery works. The answer is linearly decodable from the residual and the control is at or below chance, which is the machinery-validation the read-back patch relies on. This is the white-box half of the replication anchor (the behavioral half is the truncation curve).

What this does NOT show: early answer computation. R^2 is already 0.80 two steps into a twelve-step chain, which cannot be the answer being computed early, because steps three through twelve have not happened. It is the start-value confound: the final answer is start plus a sum of smaller signed args, so the 3-digit start dominates the answer's variance and is decodable from the first position, inflating early decodability. A clean pre-CoT-decoding test would regress out the start (decode answer-minus-start, the genuinely computed part); we did not, so we do not claim early computation from this curve. Recorded as a validation success and an honest non-claim on the pre-CoT-decoding phenomenon.

## 2026-08-04, Swap control: the residual is a readable value register (fix #3)

The read-back patch invited the objection that overwriting the value token's residual to clean and getting the clean answer is injecting the answer. The swap control refutes it. In addition to patching the residual to the clean value, we patch it to an arbitrary third value (clean plus 2 delta, distinct from both clean and corrupt) and ask which answer the continuation reaches. Qwen2.5-7B-Instruct, d=10, n=135 (113 with a corruption effect):

- patch to clean -> answer follows clean 0.97 [0.94, 1.00]
- patch to the third value -> answer follows the third value 0.76 [0.70, 0.82], follows clean 0.00
- random-direction patch -> follows clean 0.00 [0.00, 0.01]

So overwriting that one residual with any value makes the final answer follow forward-from-that-value. It is a genuine value register, read and then propagated through the remaining steps, not an injected answer: the patched value is a mid-chain intermediate, and setting it to a third value the model never wrote produces the answer that value implies. This closes the position/value-specificity gap with a positive read-write demonstration rather than a restore-to-clean that could be read as circular.

## 2026-08-04, reasoning-model causal patch (fix #2, closes positively)

We had moved the clean patch to a non-reasoning model because the reasoning distill re-solves after its think block. The swap control lets us run it on the reasoning model after all. R1-Distill-Qwen-7B, d=10, n=91 (57 with a corruption effect):

- corr_follows_corruption (behavioral read-back): 0.59
- patch to clean -> follows clean 1.00 [1.00, 1.00]
- patch to the third value -> follows the third value 0.74 [0.66, 0.82]
- random patch -> follows clean 0.29 [0.20, 0.38]

The random control is elevated here (0.29 vs 0.00 on the instruct model), and that is the post-think re-solve: about a third of the time the model re-derives the clean answer regardless of the patch. But the re-solve produces the clean answer, never the arbitrary third value, so the swap condition cuts through the confound. The answer follows the injected third value 0.74 of the time, which re-solving cannot produce, so it is confound-free evidence that the reasoning model reads the injected value out of the residual and propagates it. The causal read-back holds on the reasoning model, not only the instruct model, which closes the off-scope weakness: the paper's causal claim now covers reasoning models directly, with the swap condition as the clean measurement.

## 2026-08-04, start-controlled probe: the computed answer emerges through the trace

Re-ran A1 decoding answer-minus-start (the start value removed, so the target is the genuinely computed part of the answer), Qwen2.5-7B-Instruct, d=12, n=300, best layer. Decodability rises from 0.27 at step-fraction 0.17 to 0.83 at 1.0, control at chance throughout. Compared with the confounded version (0.80 early to 0.96 late), removing the start confirms the start was doing the early work: the computed part is not decodable early (0.27 two steps in) and builds up through the written steps.

Two clean conclusions. The probing machinery is validated (0.83 late decodability, control at chance). And the pre-CoT-decoding phenomenon does not hold for the genuinely computed quantity on our tasks: the answer is not sitting in the activations before the model writes the steps, it emerges as the steps are written, which is what the thesis predicts (the computation happens on the page). This is the honest, start-controlled version of the earlier confounded curve.

## 2026-08-06, assignment core: Qwen3-4B ladder complete, GSM8K edit test overturns a prediction

Ladder (n=150/150/30, thinking vs bare-answer): non-truncated thinking accuracy 0.99 / 0.76 / 0.90 down GSM8K, MATH-500, AIME 2024; bare-answer 0.12 / 0.19 / 0.00; bare-answer overrun rate climbs 0.29 / 0.47 / 0.80 (told not to write, the model tries anyway, more as problems harden); median thinking length 1952 / 3503 / 11819 tokens.

Edit test on GSM8K worked solutions (n=84): the answer follows a corrupted intermediate 0.87 of the time against a 0.00 resample floor. We predicted near-floor from the V3.2 result (0.10). The prediction failed, and the two results together revise the recomputability claim into its model-relative form: the 671B recomputes GSM8K intermediates and ignores edits, the 4B cannot and reads them back. Read-back fires when recomputing a value exceeds the model's internal capacity; what a frontier model recomputes, a small model must reread. Consistent with the chains capability trend (stronger models follow non-recomputable values more faithfully).

## 2026-08-06, assignment core complete: J-space 2x2 on Qwen3-4B GSM8K

Frozen dose by the pre-stated rule on Qwen3-4B neutral text: k=8, alpha=0.5. Result: cot accuracy 0.97 clean / 0.93 jlens / 0.93 random (n=30 per cell), so the targeted ablation does no more than matched random damage; direct is floored on GSM8K for a 4B (0.00 to 0.03 in every arm, 0.37 to 0.60 of direct attempts overrun the answer budget), so no differential could appear in that dimension. Combined with the synthetic-chain 2x2 (which does have direct headroom and was also null), the anchor ablation-by-condition asymmetry does not appear anywhere at doses verified harmless on neutral text. Remaining follow-up: dose escalation with neutral-text damage reported alongside.

Note on this file: it is the lab notebook, in dated order, including interim readings that were later revised. For the settled claims see notes/synthesis.md, for what did not work see notes/negative-results.md, and for the writeup see paper/main.md.
