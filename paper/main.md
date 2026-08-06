# J-Space and the Chain of Thought: Where Reasoning State Lives

## Research question and hypothesis

When a language model solves a multi-step problem, where do the intermediate results live: in J-space, the internal concept workspace that the Jacobian lens reads out of the model's activations, or in the chain of thought the model writes? Before running anything, I predicted that models use the internal workspace first and only write intermediate results down once a problem is hard enough to overflow it. This implies two things: easy problems should show little writing, and ablating J-space should hurt answers produced without chain of thought more than answers produced with it.

**Neither prediction held.** Models write out every intermediate step even on trivial problems, fail after one or two dependent steps when writing is forbidden, and are unaffected by a J-space ablation calibrated to leave normal text generation intact. The editing and patching experiments below support a different picture: the written chain of thought is where the serial computation happens. The model reads its own written values back through the internal state at each written token, and the state that stays internal is either parallel (many values needed at once, none depending on another) or cheap to recompute from the problem statement. I show this on the assignment model and datasets first, then use synthetic tasks to isolate the mechanism.

## Experiment design

The design has three axes, each with its own controls.

- **Difficulty**: the benchmark ladder GSM8K, MATH-500, AIME 2024 (easy to hard), and synthetic problems where difficulty d is the exact number of dependent steps.
- **The written channel**: open (normal chain of thought or thinking), closed (the prompt demands only the final answer; token counts verify the model really wrote nothing), capped (a hard token budget), or edited (one written value in the model's own work is changed and the model continues).
- **The internal side**: J-space ablation (remove the currently active lens concepts from the activations, with a random-directions ablation of identical size as the control), a coarser residual-stream lesion, and a residual-stream patch (overwrite the internal state at one written token with a chosen state).

Intervention rules were fixed in advance. Ablation strength is chosen on neutral text before any task runs: the strongest dose keeping next-token agreement with the clean model at 0.80 or above and the perplexity ratio at 1.30 or below. Task effects therefore cannot come from a dose that already breaks the model in general. Every cell logs wrong answers, token exhaustion, and unparseable answers separately, so accuracy effects are not confused with truncation. I wrote down a prediction for each experiment before it ran, with the opposite outcome stated as detectable.

## Experiment details

- **Assignment model and data**: Qwen3-4B, with the fitted public Jacobian lens for this exact model (neuronpedia/jacobian-lens). GSM8K test set; MATH-500; AIME 2024, meaning the 30 problems of AIME 2024 I and II as distributed in HuggingFaceH4/aime_2024. All revisions pinned in Reproducibility.
- **Synthetic tasks**, used where causal work needs exact ground truth for every intermediate value: variable chains (start with a 3-digit number, apply signed 2-digit additions and subtractions, report the final value; d = chain length), modular arithmetic chains, box tracking (5 boxes whose contents get swapped; state is parallel, not chained), and DAG path-following. Generators are seeded, and replay tests confirm the corruption arithmetic matches the generators.
- **Supporting models** for the mechanism experiments: Qwen2.5-7B-Instruct (also lens-fitted) and DeepSeek-R1-Distill-Qwen 1.5B/7B/14B white-box; DeepSeek V3.2, R1-671B, Llama 3.x up to 70B, and Claude Sonnet 4.5 through APIs, behavioral only.
- **Sampling**: temperature 0.6, top-p 0.95 for chain of thought and thinking; temperature 0 for direct answers. Fixed in advance.
- **Measurement calibration**: the externalization detector (does a known intermediate value appear in the trace) was checked by scoring each trace against a different problem's values; false positives are 0.00 to 0.08 on variable chains but reach 0.72 on mod-97 chains at high difficulty, so mod-97 is excluded from externalization claims.
- **Replication before extension**: two known results were reproduced on this setup first. Truncating a chain of thought degrades accuracy as the faithfulness literature reports (keep 60 percent of the trace and V3.2 scores 0.00; keep all of it, 0.97; Sonnet 4.5 goes 0.32 to 1.00), and a linear probe reads the final answer from activations at R-squared 0.96 while a control probe stays at chance, validating the probe and patch machinery.

## Experimental results and analysis

### Accuracy with and without written reasoning

**Models write nearly every intermediate value, even on the easiest problems.** On synthetic chains, the fraction of known intermediate values appearing in the written trace is 0.93 to 1.00 at every difficulty from 1 step to 64, for every model from 1.5B to 671B. Among correct traces at 16 or more steps it is exactly 1.000 (n=1935 across three model sizes), versus about 0.5 among wrong ones, with the same ceiling for early, middle, and late steps. If writing were only output formatting, models would sometimes skip early steps they could hold in their heads; they never do. Since the solution format asks for step results, I treat the 1.000 as a strong association rather than proof of necessity; the causal role of the written values is established in the editing experiments.

**With writing forbidden, models fail after one or two dependent steps.** The prompt demands the bare answer, and token counts verify nothing else was generated:

| Model | d=1 | d=2 | d=4 | d=8 | Really wrote nothing? |
|---|---|---|---|---|---|
| Qwen2.5-7B-it | 1.00 | 0.96 | 0.04 | 0.00 | Yes (median 4 tokens) |
| Llama-3.1-8B | 1.00 | 0.47 | 0.00 | 0.00 | Yes (2-4 tokens) |
| Llama-3.3-70B | 1.00 | 1.00 | 0.93 | 0.03 | Yes (2-4 tokens) |
| V3.2 (671B) | 0.23 | 0.03 | 0.00 | 0.00 | Yes (2 tokens) |
| R1 distills | - | - | - | - | No (74-89 percent overran) |

The R1 distills kept generating reasoning despite the no-think template, so their cells are excluded: their channel was never actually closed. Internal capacity for chained computation exists but is small, one to two steps for most models and about five for Llama-70B; whether depth or parameter count sets it cannot be separated, because the two grow together within model families. One scope note: these chains use random values that cannot be memorized. The same V3.2 that fails 2-step chains answers 38 percent of GSM8K correctly with one-word replies, because parts of natural problems can be recalled or recomputed in one step. This distinction between values a model can and cannot recompute recurs in the editing experiments.

**Qwen3-4B on the assignment benchmarks shows the same dependence on writing, growing with difficulty.** Thinking-mode accuracy stays high while bare-answer accuracy sits near the floor on all three datasets. Told not to write, the model also overruns its 32-token answer budget more often the harder the problem (0.29 of GSM8K attempts, 0.80 of AIME attempts), and the amount of thinking it writes when allowed grows steeply down the ladder (median 1952 tokens on GSM8K, 3503 on MATH-500, 11819 on AIME).

| Dataset | Bare answer acc (non-truncated) | Thinking acc (non-truncated) | Bare-answer overrun rate | Thinking cap rate |
|---|---|---|---|---|
| GSM8K (n=150) | 0.10 (0.12) | 0.84 (0.99) | 0.29 | 0.46 |
| MATH-500 (n=150) | 0.11 (0.19) | 0.65 (0.76) | 0.47 | 0.23 |
| AIME 2024 (n=30) | 0.00 (0.00) | 0.67 (0.90) | 0.80 | 0.30 |

So writing is not overflow from a full workspace; the workspace never held a chain to begin with. These results are still consistent with two readings, a trace the model computes with or a commentary beside computation happening elsewhere. The editing experiments distinguish them.

### Editing the model's written values

**Changing one written value in the model's own work usually changes the final answer to match.** I take a correct trace, replace one mid-chain value (say 417 with 457), and let the model continue. On the 7B reasoning model the answer follows the edit 31, 42, 50, and 36 percent of the time at d = 4, 8, 16, 32 (n = 149, 146, 138, 128 editable items of 150; yield falls with difficulty because only correct traces are edited). If the trace were commentary, edits would do nothing. Two ambiguities remain: the edited text also feeds any fresh recomputation, and reasoning models re-solve the problem after their think block, which pulls answers back to the correct value without any internal memory being involved. The decisive version therefore runs on a model that continues a worked solution directly.

**Overwriting the internal state at the edited token controls the answer.** I overwrite the residual-stream activations at the edited token (middle layers) while leaving the visible text unchanged. On Qwen2.5-7B-Instruct at 10 steps (n=141):

| Intervention | Outcome |
|---|---|
| Edit the text only | Answer follows the edit on 0.84 of items |
| Also restore internal state to the correct value | Answer returns to correct: 0.97 [0.94, 0.99]; at d=20: 0.93 |
| Instead write a third, never-written value into the state | Answer follows that third value: 0.76 [0.70, 0.82] |
| Write a random perturbation of the same size | Answer returns to correct: 0.00 [0.00, 0.01] |
| Same experiment on the reasoning model (n=91) | Correct-restore 1.00; third value 0.74 [0.66, 0.82] |

The third-value row matters most: the answer tracks whatever value sits in the internal state at that token, including values the model never wrote, computed forward through the remaining steps. The state at a written token functions as a memory slot that later computation reads, and the result rules out the objection that restoring the correct state merely injects the correct answer. On the reasoning model the random-perturbation control is elevated (0.29 versus 0.00 on the instruct model) because of post-think re-solving, but re-solving can only produce the correct answer, never the arbitrary third value, so the 0.74 is clean. Restoring an early, middle, or late band of layers works about equally (0.96 / 0.97 / 0.94), so the value is stored redundantly across depth and I claim no specific circuit. A probe trained to decode the answer minus the known start value (the start inflates naive decodability) reads almost nothing two steps in (0.27) and 0.83 by the end: the answer comes into existence as the steps get written, not before.

**Larger models follow the edited value more often, not less.** The same text-editing test through API prefill gives R1-distill-7B 0.42, DeepSeek V3.2 0.78, Claude Sonnet 4.5 0.97. This is the opposite of what one would expect if stronger models held more in their heads, and it means the trace remains causally load-bearing at the largest scales tested, though the comparison is observational since these models differ in more than size.

**Edits matter only when the model cannot recompute the value from the problem.** On synthetic chains the follow-the-edit rate is high at every depth (0.80, 0.64, 0.78, 0.76, 0.70 at d = 3, 5, 8, 12, 16). On GSM8K it collapses to 0.10 against a 0.05 do-nothing floor (problems with 5 or more calculation steps, n=63): a GSM8K intermediate is usually one operation away from the given numbers, so the model recomputes it and ignores the edit. I predicted the assignment-core edit test on Qwen3-4B would likewise sit near its floor. That prediction failed in an informative way: on Qwen3-4B the answer follows the edit on 0.87 of GSM8K worked solutions (n=84, resample floor 0.00, no unparseable answers). Side by side, the 671B model recomputes these intermediates and ignores edits (0.10) while the 4B reads them back (0.87). Recomputability is therefore relative to the model's internal capacity, not a property of the task alone: what a frontier model recomputes, a small model must reread. This is consistent with the scale trend above, since on chain values no model can recompute, larger models follow the trace more faithfully, while on values they can recompute they stop depending on it.

### Token budgets and scratchpad formats

**Under a token budget, models shorten their wording but keep writing every value.** With hard output caps (V3.2, chains at d=16), traces compress up to 2.5x with accuracy intact, then fail completely once the cap cannot fit the values themselves. Computation does not shift inward:

| Budget (d=16) | Tokens used | Accuracy | Values written |
|---|---|---|---|
| Unlimited | 466 | 1.00 | 1.00 |
| 256 | 184 | 0.93 | 0.98 |
| 128 | 128 (hit cap) | 0.00 | 0.72 |

The failure floor sits near 12 tokens per step; below it, the fraction of values written tracks budget over need (0.72, 0.36, 0.17 as difficulty rises at budget 128). All wall failures are genuine truncations rather than grading artifacts (0 of 60 lucky matches). One caveat: the model is told its cap, so failing to compress and failing to plan are confounded.

**Only the computed values matter; writing operations without their results fails.** Asking for the same chains in different formats: code-style lines with each computed value succeed at d=48 in 555 tokens; ordinary prose succeeds in 1105; a verbose format restating all values every step collapses (0.07 accuracy at d=48); and a code format instructed to omit computed values succeeds only where the model disobeys and writes them anyway, with per-item value-writing correlating with correctness at +0.74 (0.58 accuracy when under half the values get written, 1.00 when nearly all do). The compact value-carrying format is at once the most token-efficient and the most legible.

### Serial versus parallel tasks, and the J-space ablation

**A task whose state is many independent values fits in the model's head; a long dependency chain does not.** Box tracking (5 boxes, repeatedly swapped contents) needs as much state as a chain but with no step-to-step dependency, and it behaves in the opposite way: the model tracks 16 swaps correctly while writing only about a fifth of the state, and writing does not predict correctness (0.20 among correct versus 0.17 among wrong, where chains sit at 1.00 versus 0.5). The causal check agrees: a residual-stream lesion at matched dose cuts box-tracking accuracy by 0.34 (target window) and 0.33 (control window), against 0.11 and 0.04 for chains, at difficulties where both tasks have room to fall. The internally-stored task is roughly three times more fragile to internal damage, whichever window is hit. The lesion is blunt, however; it also damages re-reading of written values, visible at longer chains (a 0.38 drop at d=8), which is why the targeted J-space version is needed.

**Ablating the active J-space concepts, at a dose verified harmless on neutral text, changes nothing on these tasks.** The ablation removes, at every token position, the top-16 concepts the lens reads as currently active (strength 0.5, the maximum passing the pre-frozen harmlessness rule; next-token agreement 0.87 and perplexity ratio 1.12 on neutral text; the random-directions control at the same size is actually more disruptive on neutral text, ratio 1.29, which works against finding a spurious targeted effect). Under this ablation, bare-answer and chain-of-thought accuracy both match the unablated model at every difficulty, and the amount the model writes does not change. The assignment-core 2x2 (ablation by answer-mode on GSM8K, Qwen3-4B, frozen dose k=8, alpha=0.5) extends the null to the benchmark where the original asymmetry was reported. Chain-of-thought accuracy is 0.97 clean, 0.93 under J-space ablation, and 0.93 under the random control (n=30 per cell; this condition is a no-think worked solution on the first 30 test problems with a 512-token cap, so its clean baseline is not comparable to the thinking-mode ladder cell above). The ablation does no more than matched random damage. The bare-answer condition sits at floor for a 4B on GSM8K (0.00 to 0.03 in every arm, with 0.37 to 0.60 of attempts overrunning the answer budget), leaving no headroom for a differential. So no ablation-by-condition asymmetry appears in either testbed: not in the synthetic chains that have bare-answer headroom, and not on the benchmark that lacks it. Either the decision of what to write is fixed by training rather than responsive to internal load, or chain state lives outside the top lens concepts; a dose escalation with neutral-text damage reported alongside is the natural follow-up.

## Positioning

Expressivity theory says fixed-depth transformers can do only limited serial computation in one forward pass and that chain-of-thought length buys serial power (Merrill and Sabharwal 2023, 2024; Li et al. 2024; Feng et al. 2023); it predicts the behavioral results here and motivates the mechanism experiments, which it does not itself describe. Faithfulness studies (Turpin 2023; Lanham 2023; Bentham 2024) and recent work decoding reasoning state from activations (2603.05488, 2603.01437, 2604.18307, 2606.13603) establish that traces and internal state can disagree. The two closest papers stop short of the causal test run here: 2604.15726 calls for a token-corruption experiment it never runs, and 2605.30343 builds latent memory blocks without manipulating capacity or measuring read-back. The J-space construct and the lens are from Gurnee et al. 2026.

## Negative Results

**The writing threshold the project was designed around does not exist.** The plan was to find the difficulty at which models start externalizing and fit a scaling law to that onset across model sizes; a whole project phase was budgeted for it. There is no onset: writing is saturated from one-step problems at every scale. What survives is the quantity on the other side of the boundary, the depth a model can handle with no writing at all, and there the data supports only an ordinal statement (one to two steps for most models, about five for Llama-70B), because depth and parameter count cannot be separated with public checkpoints. This null redirected the project from measuring an onset to asking what the writing is for.

**Making the internal side scarcer does not make the model write more.** If externalization responded to internal pressure, induced scarcity should induce writing. I tested this twice. A heavy residual lesion pushes the model past an accuracy cliff, where writing degrades along with everything else. Because that lesion is arguably too crude, the calibrated J-space ablation served as the fairer instrument, and it produces no change either: not in accuracy, not in the amount written, not in the fraction of values externalized. Two readings survive: the write policy is set by training and does not respond to inference-time load, or chain state is carried outside the top lens concepts. Either way, the adaptive-hierarchy picture, in which the model manages a live trade-off between workspace and page, finds no support in either direction; capping the page also fails to push computation inward, since budget failures are truncations with no sign of internalization.

**The GSM8K edit test was null, which bounds the mechanism's scope.** Corrupting a written intermediate on GSM8K (V3.2) moves the answer at roughly the sampling noise floor, where the same edit on synthetic chains flips the answer at every depth; as discussed above, the difference is recomputability rather than depth, since the chain flip rate is flat from three to sixteen steps. Written values function as memory precisely for state the model cannot cheaply reconstruct, and much of benchmark math does not have that character. Any claim about chain of thought as working memory that omits this qualifier is overstated, including mine before this test.

**Two early claims were withdrawn after a confound was found.** A clean-return rate of 0.48 to 0.68 after edits, and its rise at high difficulty, were initially read as the model verifying written values against a persistent internal copy. Both readings collapsed on the observation that the full problem statement stays in context, so returning to the correct answer requires only re-derivation from the prompt; the reasoning-model experiments then showed exactly that post-think re-solving. An early-decodability claim fell the same way: the answer seemed readable from activations two steps into a chain, but only because the chain's start value dominates the answer's variance; regressing it out shows the computed part is absent early and builds as steps are written. No internal-copy or verification claim survives, and the corrected probe supports the tape picture rather than opposing it.

Smaller instrument failures, recorded for completeness: the DAG task cannot measure externalization because every node name already appears in the prompt; mod-97 chains were dropped when the permutation control showed match false positives up to 0.72; and the first lesion experiment was invalid as coded (it fired only during generation, leaving the bare-answer condition nearly unlesioned) and was rerun after the fix.

**Limitations.** The central mechanism applies to state the model cannot recompute, which excludes much of GSM8K-style math. Synthetic tasks trade realism for exact ground truth. White-box causal results are at 4B to 7B scale; frontier evidence is behavioral. The patch overwrites the full residual at a position, and although the third-value result shows the value is what matters, the value's own subspace is not isolated; splitting the patch into its J-space component and the remainder is the planned experiment that would locate the memory slot relative to the lens-readable workspace. The lesion cannot separate workspace damage from read-back damage. Remaining open: a within-task recomputability manipulation, that patch decomposition, and a dose escalation of the J-space ablation with neutral-text damage reported alongside.

## Reproducibility

Pinned revisions:

- Qwen/Qwen3-4B 1cfa9a720891; Qwen/Qwen2.5-7B-Instruct a09a35458c70
- DeepSeek-R1-Distill-Qwen-1.5B ad9f0ae0864d; 7B 916b56a44061; 14B 1df8507178af
- neuronpedia/jacobian-lens a4114d7752d1 (qwen3-4b and qwen2.5-7b-it artifacts); lens library commit 581d398613e5
- GSM8K test.jsonl sha1 4a3eef48d603; HuggingFaceH4/MATH-500 6e4ed1a2a79a; HuggingFaceH4/aime_2024 2fe88a2f1091

Dose rules were frozen before task cells ran. Every cell logs cap hits, unparseable answers, and bare-answer token counts. Raw data regenerate from src/harness; figures from src/analysis/figures.py; the dated evidence trail is notes/results-log.md.
