---
name: trajectory-shuffled
description: Predict the single most likely next decision in an ongoing research project from its ordered history of prior decisions (typed as method/experiment/ablation/hypothesis, with retained/abandoned/superseded status). Use when asked to forecast what a research team does next, to rank candidate next steps, or to continue a decision trajectory.
---

# Predicting the next research decision

## 1. Reconstruct the active frontier before guessing

Read the trajectory backwards. For the last two to four decisions, record (a) which *object* each touched — a method component, a data/degradation/annotation pipeline, an evaluation protocol, a reported metric, or a free knob; (b) whether it was a replacement of an earlier decision or a new front; (c) what obligation it left open.

Open obligations are the strongest predictor. Recurring ones:

- a formulation was introduced but is still conceptual, non-executable, or defined only in prose;
- a new free parameter, threshold, mixture coefficient, or taxonomy category was introduced and not yet swept or dropped;
- a method was stabilized but applied to only one data source or one backbone;
- a component was abandoned/superseded, leaving a functional hole that must be refilled;
- one arm of a comparison changed, so the counterpart arm is no longer matched;
- a pipeline is applied to inputs it cannot handle (overlong, degenerate, out-of-domain, zero-baseline conditions), so eligibility rules are missing.

Also treat the project's title/claim as an attractor. Trajectories converge toward the minimal configuration that the claim asserts (e.g., a claim of a single control parameter predicts collapsing a multi-branch formulation; a claim of post-hoc/training-free operation predicts abandoning arms that retrain; a claim of exactness predicts replacing an estimator with a closed-form computation). Decisions that contradict the claim are the ones most likely to be replaced next.

## 2. Candidate generators, in rough order of observed frequency

**Replace the most recent component in place, along one axis.** Most method decisions are replacements, not additions. Observed axes, any of which can fire:

- idealized/conceptual criterion → executable proxy computed from quantities the system already produces (likelihoods, distances, frequencies, counts);
- trained sub-module → directly fitted or closed-form substitute that removes a training loop;
- soft penalty/regularizer → hard constraint applied at a fixed point in the loop;
- stochastic estimator → exact computation (or exact → estimator when exactness is intractable), with a validation-against-exact step;
- judge/annotator/teacher model swapped for a stronger or deterministic one, and/or given more context: reference answer, preceding steps, several sampled responses with their grades, category metadata;
- whole-object processing → per-unit or chunked processing with explicit labels/scores per unit and an exclusion rule;
- multi-parameter or multi-branch formulation → single-parameter reparameterization with a named endpoint that recovers the unmodified baseline;
- free-form rewriting/generation → classification or selection over existing units (safer, more auditable).

**Apply the stabilized method to another source.** Annotate/process a second corpus, model family, or benchmark, produce the modified counterpart, and compare it against its unmodified version on the project's existing benchmark set.

**Ablate exactly what the previous decision introduced.** Sweep the new knob over a short explicit grid, compare its endpoints against the default, or drop-one/keep-one over the new categories or branches.

**Add a matched baseline or control.** Unmodified counterpart under identical accounting; oracle or upper bound; the standard-practice alternative the paper argues against; a reordering of pipeline stages; an end-to-end-trained arm held fixed for post-hoc comparison.

**Harden the protocol.** Fix seed lists, number of repetitions, mean±std aggregation, deterministic splits with stated sizes, per-condition exclusions, subsample counts.

**Extend the reported quantities.** Add the complementary axis to quality — cost, runtime, token count, efficiency ratio — or add threshold-free plus thresholded summaries, or per-item plus aggregate views.

**Scale an axis already parameterized.** More model sizes, more modalities/domains/resolutions, larger generation or privacy budgets, a wider or finer sweep range.

## 3. Choosing one candidate

- Prefer the minimal edit to the most recently touched object over opening a new front. New fronts appear, but usually after the current object has been replaced two or three times and settled.
- Type follows type: after a method change, expect an experiment applying it or an ablation of its new knob; after an experiment that exposes a measurement weakness, expect a metric/grading/protocol fix; after an ablation that enumerates variants, expect commitment to one variant as the default, often via a reparameterization.
- After an abandonment or supersession, predict the replacement that preserves the goal while removing the cost or ambiguity that killed it: cheaper, more standard, more deterministic, or narrower in scope.
- If the project has repeatedly grown coverage, predict one more coverage step rather than a new mechanism. If it has repeatedly reformulated one component, predict another reformulation of that same component.
- Inherit the project's protocol vocabulary verbatim: the same seed list, sweep endpoints and step size, dataset triple, epoch count, number of repetitions, judge model, aggregation rule. Predicted decisions in these trajectories almost always reuse existing settings rather than inventing new ones.
- Exhaustive cross-matrices, very wide parameter sweeps, and extra side baselines are plausible next decisions when the project is already enumerating conditions — even though such decisions are often later abandoned. Do not suppress them, but do not propose them in projects that have been narrowing.

## 4. Form of the answer

State one decision as a single scientific action, at the granularity of the records: name the object being changed or evaluated, the concrete arms or settings, the comparison target, and the reported quantity. Label it method, experiment, ablation, or hypothesis. Do not include justification, expected outcome, or retrospective evaluation.

## 5. Failure modes to avoid

- Generic steps the trajectories never contain: broad hyperparameter tuning, writing, literature review, "verify results."
- Proposing a brand-new mechanism when the recent history moves by replacement of one component.
- Wrong granularity: "evaluate the method on more data" instead of naming the source, the arms, the sweep, and the metric.
- Ignoring the claim in the title and predicting a step that reintroduces a property the paper disclaims.
- Predicting a step that duplicates an already-retained decision, or that re-adds something already abandoned without changing what made it costly.
- Treating record numbering as chronology when it conflicts with trajectory order; the trajectory order is the timeline, and supersession links tell you which decisions are still live.
