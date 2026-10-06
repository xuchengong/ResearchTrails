# Predicting the Next Research Decision

Input is an ordered list of prior decisions, each carrying a type (method / experiment / ablation / hypothesis) and a fate (retained / abandoned / superseded by a later decision). Output one concrete next decision. The trajectories show that next decisions are almost never new topics: they are local moves on the component or protocol that the most recent decisions were already touching.

## Step 1 — Locate the active thread

Scan backwards for the most recent decisions and group them by the *component* they act on (a scoring rule, an annotation stage, a calibration routine, a judge, a data-generation pipeline, an evaluation matrix, a readout head).

- The thread containing the latest decisions is the prediction target. Weight recency far above breadth.
- Read fates as stability signals. A component with two or more supersessions in its chain is unstable: predict yet another replacement of *that same component* rather than a move elsewhere. A component whose latest decision is retained is stable: predict a move that *uses* it (new condition, ablation of its knob, or downstream evaluation).
- Note which knobs, thresholds, taxonomies, exclusion rules, and stage orders the latest decision just exposed. These are the raw material for the next decision.

## Step 2 — Choose the move type

Four move types cover nearly all observed transitions. Choose by the signals below.

1. **Replace/reformulate the same component.** Triggered by an unstable chain, by a component still stated conceptually rather than computably, or by a component whose current form requires an expensive inner loop (training a probe, rewriting whole artifacts, multi-stage generation). Phrase the prediction as "replace X with Y" and name what Y removes.
2. **Apply the working recipe to one more condition.** Triggered by a retained end-to-end pipeline. Conditions recur in a fixed vocabulary: another data source or benchmark family, another backbone or model family, another model scale, another split or supervision protocol, another modality count, another task type (classification → regression → structured prediction). A common variant substitutes one condition for another (drop one out-of-family backbone, add a different one) rather than adding both.
3. **Ablate a knob the method just introduced.** Triggered by a retained method with a free coefficient, threshold, radius, ratio, component set, category set, or granularity. Ablations most often compare a small set of named values including the degenerate endpoints (mixture coefficient at 0 and 1; keep-one vs drop-one vs remove-all of a taxonomy's categories; component subsets of an ensemble).
4. **Strengthen evaluation scaffolding.** Triggered by results being compared across arms. Recurring forms: replace brittle automatic grading with a model judge; enrich judge inputs (reference answer, sampled responses with their grades, category metadata); upgrade to a stronger or deterministic judge; add repetitions/seeds with mean and standard deviation; add an oracle or upper-bound reference; add a theoretical bound to compare the empirical curve against; change how per-condition gains are normalized and aggregated.

If two move types are plausible, prefer the one that operates on the *most recently introduced* object.

## Step 3 — Apply drift directions inside a thread

Chains of replacements move in consistent directions. Use these to fill in the content of a predicted replacement.

- Conceptual → executable: a formula becomes a concrete computation with explicit inputs, plus omission rules for cases where it is undefined.
- Learned/gradient-fitted → closed-form or deterministic; soft penalty → hard constraint applied after each step; stochastic judge → deterministic judge.
- Whole-artifact rewriting → per-unit labeling or scoring with a small discrete label set, keeping only confidently flagged units.
- Implicit parameterization → explicit parameterization that isolates the quantity of interest and recovers the baseline exactly at a distinguished parameter value.
- Generic mechanism-agnostic routine → specialized exact routine for the specific mechanism, with the generic one retained as a comparison arm.
- Fewer moving parts: branches, mixture terms, auxiliary losses, and extra annotation stages added early are often dropped later.
- Granularity shifts one level at a time (token → step → group → trace; per-axis → one shared scalar per module; per-run flattening → per-condition then across-condition aggregation).
- Length/size limits are first handled by exclusion, later by chunk-then-merge processing (or the reverse).
- Taxonomies gain one category and a structured output format; typed links between units get added.

## Step 4 — Inherit the protocol

When predicting a new arm, condition, or ablation, assume it reuses the established protocol rather than inventing one: the same seed set, the same parameter sweep range and step, the same repetition count, the same metrics, and the same baseline endpoint as a comparison condition. Predicted decisions that silently invent new constants are usually wrong. Two exceptions appear: learning rates are revised when model scale changes, and a newly added condition may bring one condition-specific input adaptation (resizing, channel handling, activation conversion, dataset-specific prompt and answer parsing).

## Step 5 — Also predict pruning

Some next decisions remove rather than add. Predict abandonment or narrowing when the trajectory contains:

- a broad sweep over a threshold's full range (next: fix a single value, or ablate a narrower named set);
- an extra baseline or diagnostic outside the core claim (direct-answer baselines, simple agreement statistics, exploratory geometry checks on off-the-shelf artifacts);
- an early pilot evaluation on a small condition set (next: a fuller matched benchmark supersedes it);
- a cost-saving shortcut such as single-run reporting or subsampling observations (these are typically reverted toward repeated, fuller evaluation).

## Writing the prediction

State one action at the grain of the records: verb (replace / add / evaluate / ablate / apply), the named component, the specific new form or condition, and the inherited protocol elements. Include the comparison arm explicitly when the move is an evaluation or ablation, and include eligibility or validity guards when the move exposes a method for use — exposing a method conditionally, only where its own error bound or confidence criterion holds, is itself a recurring decision.

Avoid predicting: several moves at once, a change of research question, or an evaluation whose conditions and metrics do not already appear somewhere in the trajectory. When a retained method has just been finalized and its knobs already ablated, the highest-probability next decision is transfer of the whole recipe to a second data source or a larger model scale, keeping every other protocol element fixed.
