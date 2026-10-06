# Research Decision Prediction

## What these trajectories look like

A project is a set of interleaved *threads*, each thread being one component (a scoring rule, a data-generation pipeline, an annotation/judge protocol, an evaluation protocol, an architecture slot). Decisions arrive one per thread turn, each labeled method / experiment / ablation / hypothesis, each ending as retained, superseded-by-X, or abandoned. Prediction is mostly a matter of identifying which thread is currently *open* and applying that thread's characteristic direction of travel.

## Step 1 — Reconstruct the state

Before predicting, write down briefly:

1. **Open substitution chains.** Which components have already been replaced once or twice and have no retained terminus yet. These are the highest-probability sites of the next decision.
2. **The retained core so far.** The components marked retained define a protocol template (sweep ranges, seeds, metrics, optimizer settings, benchmark list) that later decisions copy nearly verbatim.
3. **Free parameters just introduced.** Any coefficient, threshold, count, or stage ordering created by the most recent retained method.
4. **Coverage axes already in play.** Datasets, model families, model scales, task types, resolutions, modality counts, noise types. Note which axes have been expanded once — they tend to be expanded again or pruned.
5. **Side branches.** Threads that use data, baselines, or machinery outside the core claim.

## Step 2 — Read the phase

- **Exploration phase** (several method variants of the same component, few retained experiments): the next decision is almost certainly another *method* decision on that same component, not a new experiment.
- **Consolidation phase** (core method retained, evaluations accumulating): the next decision is an *experiment* that reuses the retained protocol on a new axis, an *ablation* of the newest free parameter, or a reporting/aggregation rule.
- **Closing phase** (matched final evaluation exists, many abandonments): expect narrowing — abandoning a sweep, dropping the largest/most expensive suite, replacing a broad comparison with a focused one, or fixing how results are averaged and which conditions are excluded.

## Step 3 — Directions of travel (the predictive core)

Substitutions in these records are strongly directional. When a chain is open, continue it in the same direction rather than reverting.

- **Conceptual → operational.** Formulas or criteria stated in idealized terms get replaced by something directly computable from an available quantity, with eligibility carve-outs for degenerate cases. Multi-term normalized scores collapse to one measurable ratio or difference.
- **Indirect/trained → direct/closed-form.** Gradient-trained auxiliary heads, grid-plus-interpolation constructions, and soft penalty regularizers get replaced by fitted closed-form estimators, exact evaluation, and hard post-step constraints.
- **Single-stage → explicitly staged.** A monolithic rewrite or annotation becomes: first partition/structure, then label, then link. Structure once retained is rarely revisited; further refinement hits the *scoring* stage attached to it.
- **Free parameter → automatic or principled selection.** A swept radius/branching factor/count becomes chosen by minimizing a bound or derived from an intermediate quantity; a heuristic becomes an explicit admissibility condition.
- **Removal-based edits → preserve-the-backbone edits.** Methods that delete or rewrite whole units are replaced by ones that classify units by role and only touch the auxiliary ones; the deletion variants reappear as controls.
- **Absolute claim → parameterized claim.** A sign/all-pairs hypothesis is abandoned and replaced by a continuous quantity plus an explicit control parameter, after which ablations sweep that parameter.
- **Judge/annotator protocols** move toward richer input (add sampled candidate outputs, their labels/grades, reference keys), a stronger backend model, and narrower scope (the one benchmark with trustworthy ground truth). Extending such a judge to many benchmarks, or distilling it into a small model, is typically proposed and then abandoned.
- **Edge cases**: first excluded by a length/validity filter, then a restructuring that handles them is attempted; the restructuring is a plausible abandonment.
- **Coverage axes** expand within a family (depths, sizes), then add a different family, then the odd family is replaced by a more current one rather than kept alongside.

## Step 4 — Compose the prediction

Predict exactly one decision. Make it as specific as the retained protocol permits, because new decisions inherit prior detail:

- Name the component or axis, the new form it takes, and the protocol it reuses (same metric set, same seeds/repetitions, same sweep grid, same baseline comparison).
- If a method was just retained, the strongest candidates are: (a) its first application to one further dataset/backbone/task, (b) an ablation of its newest coefficient at its endpoint values, (c) a matched control isolating it from a trivially simpler variant.
- If an experiment was just run in a weak form (one run, one dataset, a subset, a normalized aggregate metric, a permissive grader), predict its strengthening: repetitions with mean and standard deviation, the full evaluation set, a matched original-vs-modified comparison across all model families in play, or an unnormalized per-instance metric.
- If two stages exist whose order is arbitrary, predict an ablation comparing the orders.
- If a claim depends on a training choice (initialization, schedule, duration), predict a robustness check with repeated seeds or a contaminated/degraded variant used as a deliberately worse condition.
- If the project has just finished broad evaluation, predict an explicit aggregation rule: which conditions count, how to average across conditions versus runs, and an efficiency-normalized or cost-aware summary.

## Signals that the next decision is a removal

Favor predicting abandonment/removal when the candidate thread: relies on datasets or tasks outside the project's stated domain; duplicates a signal already obtained by a retained component; sweeps the decision threshold or component subsets of an already-fixed ensemble; sweeps a test-time budget or granularity orthogonal to the claim; introduces a baseline that reframes the contribution (e.g., a no-mechanism control or a grader swap) after a cleaner protocol has been adopted; or transfers a retained procedure into a secondary regime for which no ground truth exists.

## Prediction failure modes to avoid

- Do not invent a new component when an open substitution chain exists; refinement of the open chain outranks novelty.
- Do not predict a broad, multi-dimensional program; individual decisions change one component or add one axis.
- Do not predict reversion to an already-superseded form, or re-tuning a parameter that a later decision fixed by default.
- Do not upgrade the claim's ambition; late decisions tighten, control, and report rather than extend.
- When two threads are open, prefer the one whose most recent decision is nearest in the record and whose replacement has not yet been exercised in any experiment.
