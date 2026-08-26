# Meta-review

Four independent reviewers (alfa, bravo, charlie, delta). **Unanimous verdict: Major revision.** All four judge the core causal result (planting a never-written value propagates; norm-matched noise does not; the effect is cross-family and cross-task) credible and well-controlled. The revision requests concentrate on four themes.

## Consensus concerns (raised by 3 or 4 reviewers)

1. **The mechanistic claim outruns the localization.** (charlie #7, alfa #6, bravo #6, delta #6) The state patch overwrites an unspecified "middle band" of layers and the attention block masks every layer. Neither is swept or localized. Until a layer-resolved ablation shows the effect concentrates in a specific range, "addressable memory read through attention" is not distinguishable from "deleting a token's entire contribution to the forward pass, which unsurprisingly matters." This is the single most-cited blocker and it targets the title claim. Fixing it requires a new experiment (single-layer and early/mid/late-third patches and masks).

2. **The strength-scaling claim does not survive the paper's own data.** (charlie #1, bravo #4, delta #1) "Trace-dependence holds more, not less, for stronger models" is built from three protocol-mismatched points (R1-distill-7B 0.42, V3.2 0.78, Sonnet 0.97) and omits Llama 3.3 70B, which the paper reports at 0.97 elsewhere and which breaks monotonicity. The mechanistic evidence is also confined to open 4B-14B models while this claim rests on behavioral frontier numbers. Either put all four points on one axis and report the correlation honestly, or scope the claim to "policy differs by model; not found monotonic in capability."

3. **Novelty framing overstates priority.** (charlie #2, alfa #1) arXiv:2606.29522 and arXiv:2505.04955 already ran causal interventions on written CoT tokens at comparable effect sizes. Section 7 concedes this; the abstract and intro do not. Concede the causal question was answered at smaller scale and reframe the contribution as generalizes / localizes / composes / taxonomizes.

4. **Reproducibility and disclosure gaps.** (charlie #6, bravo #6, delta #2, delta #7) No code/data-availability statement, no decoding parameters in the main text, patch-band selection deferred to appendix. Add an availability statement and put decoding settings and the layer band in the main text.

## Recurring specific errors (2 reviewers each, all fixable in writing)

5. **Composition numbers are unverifiable and inconsistent** (alfa #3, delta #4): abstract says 88 percent, body says 0.88-0.90, and "at or above the product of the single-slot rates" cannot be checked because the single-slot rates are not reported next to it. Report the single-slot rates, the product, and the joint rate together.

6. **R1-distill-7B 0.60 vs 0.42** (bravo #7, delta #1, charlie minor): two different quantities sharing a bare model name. Disambiguate.

7. **The Jacobian lens is never defined or cited** (charlie #5, alfa #2, delta #5): the only scope-boundary result rests on an undefined method. Define it in one sentence or cite the source, and report the dose-response rather than asserting the endpoint.

8. **Section 6 scope is entirely unquantified** (bravo #1, charlie minor): capacity and box-tracking claims have no n or interval, unlike the rest of the paper. Add numbers or cut.

9. **The distill 24/24 explanation is a convenience sample** (charlie #3, bravo #2): the denominator is unstated and appears to be a subsample of ~30. Run an automated classifier over the full reverting population and report the rate with an interval.

10. **Conditioned-rate comparability** (charlie minor, delta #3): rates are conditioned on "edit followed," which itself ranges 0.44-0.98 across models, so cross-model comparison is not clean. Report the unconditioned rate alongside, or state the caveat.

11. **Statistical detail** (bravo #2, bravo #3, delta #7): no multiple-comparison correction; unclear whether the bootstrap resamples items, seeds, or both. Clarify the resampling unit and note that the paper reports intervals rather than significance tests.

## Single-reviewer concerns worth addressing

- Knockout coherence confound (charlie #4): report the unparseable/incoherent-output rate under the attention block to rule out "the model can no longer speak" rather than "the model can no longer retrieve the value."
- Structured-but-wrong control (bravo #5): the only patch control is norm-matched noise; a valid-activation-for-a-different-value control would rule out "any arithmetic-shaped activation moves the answer."
- Bare-ID citations (alfa #4): the closest-prior-art papers are cited by arXiv ID only, unlike the author-year citations elsewhere. Give them full entries.

## Priority for revision

Blocking the headline claim: 1 (layer localization) and 2 (drop or scope the strength-scaling claim).
Fixable in writing with data in hand: 3, 4, 5, 6, 7, 10, 11, and the bare-ID citations.
Cheap analyses from logged data: 8 (report capacity/box numbers), 9 (automated distill classification), and charlie #4 (knockout coherence rate).
New experiments: 1 (layer sweep, blocking) and bravo #5 (structured control, optional).
