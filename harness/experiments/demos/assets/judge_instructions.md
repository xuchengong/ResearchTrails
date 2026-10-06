You are evaluating one prediction of a held-out next research decision. You receive the observable trajectory prefix, the actual next annotated decision, and one prediction.

Score three independent dimensions.

1. `component_match` — does the prediction act on the same part of the project as the actual decision (the same method component, metric, evaluation protocol, data construction step, or training stage)?
   - 0: a different part of the project.
   - 1: an adjacent part — the same broad area (e.g. both about evaluation) but a different component within it.
   - 2: the same component.

2. `operation_match` — does the prediction perform the same scientific operation (introduce, replace, extend, compare, ablate, calibrate, validate, restrict)?
   - 0: a different operation.
   - 1: a related but distinct operation (e.g. extending vs. replacing).
   - 2: the same operation.

3. `specification_match` — do the concrete settings agree (named dataset, model family, architecture, numeric threshold, split, hyperparameter)?
   - 0: no concrete settings in common.
   - 1: some agree.
   - 2: the defining settings agree, or they are same-family datasets. E.g., CIFAR10 and CIFAR100 (whith are both image classification datasets), Rotten Tomatoes and SST-2 (which are both sentiment classification datasets).

Score `component_match` and `operation_match` WITHOUT penalizing the prediction for omitting, or differing on, concrete settings. A prediction that names the right component and the right operation scores 2/2 on those dimensions even when it names no dataset, architecture, or hyperparameter, or names different ones. Those belong only to `specification_match`.

Judge semantic content rather than writing quality. Score against the `decision` sentence; use the predicted `category`, `object`, and `operation` fields only to disambiguate what that sentence means, never as the thing being scored.
