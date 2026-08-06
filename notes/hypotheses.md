# Initial hypotheses

What we expected before running anything, and why. Each entry also says what the opposite result would look like in the same measurement; if a result could not show us the opposite, the experiment gets redesigned before it runs. Kept as a running record: updates append, nothing gets rewritten after its experiment starts.

## Gate A, does the write policy respond online

Expected: honestly close to a coin flip, prior around 40 to 50 percent that a layer-window lesion moves externalization fraction relative to random-site lesions at matched damage. The distills were trained by imitation, so the write policy may be open-loop. Opposite visible by construction: flat externalization under targeted lesions, which reroutes the paper to the fallback headline. Either outcome is decision-relevant, which is why this runs first.

## Gate B, is anything read back

Expected: corruption flip rate above 50 percent at high d for arithmetic-style families, low at small d. Why: at high d the internal copy should be stale or absent, so downstream steps have to use the written token. Opposite visible: flip rate near zero everywhere, which supports the CoT-as-projection antagonist position and would make read-back a null result to report while shifting weight to 2b and phase 4.

## Phase 1, externalization curves

Expected: for each model, externalization fraction stays near zero up to some d*, then rises steeply; d* grows with model size; CoT length grows superlinearly past d*. Why: theory says single-pass serial capacity scales with depth, so bigger models can hold longer chains internally; Lanham-style inverse scaling of CoT reliance points the same way.

Opposite results that would be visible: (a) externalization fraction is high at all d, including trivially easy problems, meaning reasoning-trained models write everything down by habit and there is no load-dependent onset, only a trained policy. This is a live possibility given the overthinking literature, and it would push the project toward "the policy is miscalibrated relative to the capacity boundary," measured as the gap between d* and the direct-answer failure point. (b) No dependence on model size, which would undercut the capacity account entirely. (c) Externalization tracks output-format habits per family rather than difficulty, which the cross-family comparisons will expose.

Also expected: the direct-answer accuracy cliff sits above d* for reasoning models (they write before they must). If instead d* coincides exactly with the cliff, the cost-benefit story is cleaner than we assumed; if d* sits far below the cliff, habit dominates capacity.

## Phase 1.5, protection

Expected: CoT protects accuracy against internal lesions relative to the direct condition, with protection shrinking at high d. Why: at low-to-mid d the trace holds most intermediates, so lesioning the workspace leaves the external copies intact; at high d traces degrade and some load returns to internal state. Weaker secondary expectation: per-instance protection tracks the clean-trace externalization fraction, positive slope, absent in the random-window arm at matched KL.

Opposite results visible in the same design: (a) no protection anywhere, CoT and direct degrade alike, meaning written tokens are not a usable store under damage, a hard blow to the hierarchy reading; (b) protection constant in d, meaning externalization coverage does not thin out with difficulty in the measured range; (c) protection growing with d, the reasonable pattern if easy problems are answered without writing anything, so there is no external fallback. Pattern (c) and our expected pattern differ only in where on the d axis the trace has content, so the externalization curves from phase 1 are read alongside this result and the interpretation rule is fixed in advance: whichever pattern appears, it is interpreted jointly with the measured externalization fraction at that d, not by itself.

## Phase 2b, eviction

Expected: probe decodability of an intermediate value at current-position residual streams drops faster after the value is written to tokens than in matched unwritten cases at equal distance from computation. Why: keeping a copy is expensive under superposition interference; a written value is retrievable by attention, so the workspace should reclaim the space. Effect size guess: modest, 10 to 30 relative percent drop, not to zero, since breadcrumbs persist.

Opposite visible: decodability equal or higher after writing (broadcast copy, redundant cache, no eviction). That result would be worth publishing on its own since it says the "hierarchy" has no writeback discipline, and it would predict that CoT corruption fails to change answers on those traces, a cross-check we run either way.

## Phase 2c, read-back

Expected: corrupting a written value flips downstream computation increasingly often as d rises; attention knockout to the value's tokens reproduces most of the corruption effect; a small set of heads carries most of it. Why: at high d the internal copy is gone (2b) so the token is the live copy; receiver-head concentration matches thought-anchors findings at sentence level.

Opposite visible: answers track the internal value under token corruption (patching the clean internal state back has no effect, corrupted token ignored), meaning CoT is a write-only log, decorative at value level even when accurate. Also possible and visible: read-back fraction high at low d and lower at high d, which would invert the hierarchy story.

## Phase 3a, squeeze internal

Expected: workspace-subspace ablation at moderate dose increases CoT length and externalization fraction and partially restores accuracy relative to random-subspace controls at matched KL; control tasks flat. Why: if writing is a response to workspace scarcity, induced scarcity should induce writing. This is the riskiest prediction in the project and the one we care most about. Honest prior: maybe 40 percent it works as stated, because the write policy may be fixed by training rather than load-sensitive at inference time.

Opposite visible: ablation degrades accuracy with no change in externalization (policy is static; the trade-off is set during training, not adapted online), or externalization rises equally under random-subspace damage of matched KL (the response is generic distress, not memory management). Both are distinguishable in the design because we log externalization against the damage meter for targeted and random interventions separately.

## Phase 3b, squeeze external

Expected: under token budgets, probe decodability of intermediate values at late positions rises (the model holds more internally) and accuracy holds until a d-dependent ceiling, then breaks; paraphrase hurts little (semantic memory) while filler substitution hurts a lot at high d (content matters, not just slots). Why: Pfau et al. show fillers only buy parallel compute; our high-d tasks are serial.

Opposite visible: filler tokens rescue performance as well as real CoT at high d, meaning the channel is compute slots and the memory framing is wrong for that family; or internal decodability does not rise under budgets, meaning there is no compensatory internal storage and the two tiers do not trade.

## Phase 4, onset law

Expected: d* approximately linear in effective depth within a family (serial budget), with reasoning-trained models showing lower d* than matched base models at equal size (RL teaches earlier writing). Cross-family transfer of the fitted law: genuinely uncertain, no confident prediction, and we say so; a family-specific intercept with shared slope is our weak guess.

Opposite visible: log-depth fits better (parallel-scan regime), or d* tracks parameters rather than depth (capacity is width/superposition, not serial), or no stable d* exists across seeds (onset is stochastic, fit distributions instead, per the random-emergence literature). The model comparison is set up before fitting so any of these outcomes is a result rather than a failure.

## Phase 5, formats

Expected: structured formats raise accuracy per written token and raise d* at matched budgets; read-back heads transfer across formats (addressing is general). Opposite visible: prose wins (models read their own prose better than tables, training distribution dominates), or format effects vanish at scale.
