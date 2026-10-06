You receive (1) an observable trajectory of research decisions, (2) the actual next decision, and (3) one prediction for the next decision. Evaluate how closely the prediction matches the actual decision. 

Score `component_match` (what part of the project is acted on), `operation_match` (the scientific action), and `specification_match` (concrete choices) independently from 0 to 2.

## Understand the decision before scoring

Describe each decision as an **operation on a component, with specifications**. Use the trajectory to resolve references and understand whether a choice is being introduced, extended, replaced, or tested. 

| Decision | Component | Operation | Specifications |
| --- | --- | --- | --- |
| Add CIFAR10 to the existing evaluation suite. | Evaluation dataset selection | Extend the suite | CIFAR10 |
| Replace the current loss with a weighted sum of classification and reconstruction losses. | Training objective | Replace the objective | The two loss terms and their weighted combination |
| Compare training with and without the reconstruction loss. | Training objective | Ablate a loss term | Reconstruction loss |

Identify the functional component; treat named implementations, such as AdamW versus SGD, as specifications.

## 1. `component_match`

Does the prediction act on the same functional part of the project as the actual decision?

**0 — A different part of the project, or no identifiable component.**

- One changes the optimizer, while the other changes training-data construction.
- One changes the model architecture, while the other changes the loss objective.

**1 — A related part, or only a broader identification of the correct part.**

- One changes an evaluation metric, while the other changes how uncertainty in that metric is estimated. Both concern measurement, but act on different parts of it.
- One changes token embeddings, while the other changes positional encodings. They concern related representation components.
- The actual decision changes the evaluation metric, but the prediction only says “improve evaluation,” without identifying a particular component.

**2 — The same functional component.**

- Both change the optimizer, even if they choose different optimizers.
- Both act on a hyperparameter search grid, even if one expands it and the other restricts it.

Tuning an optimizer and replacing it concern the same component. Identify experiments by the functional part they change, not their experiment numbers or dataset names.

## 2. `operation_match`

Does the prediction perform the same scientific action? Examples include introduce, replace, remove, extend, restrict, compare, ablate, calibrate, and validate. For questions and hypotheses, identify whether the statement asks a question, proposes an explanation, or tests it. 

**0 — A different or opposing operation, or no identifiable action.**

- One expands the hyperparameter search range, while the other narrows it.
- One adds a neural-network layer, while the other moves an existing layer to a different location.
- One proposes an explicit ablation, while the other merely says “improve the method” without identifying an action.

**1 — A related but distinct operation, or an underspecified version of the action.**

- One introduces a new tunable hyperparameter, while the other extends the search grid for an existing hyperparameter.
- One replaces the optimizer with a candidate optimizer, while the other benchmarks that candidate against the current optimizer before making a choice.
- The actual decision replaces a mechanism, while the prediction modifies that mechanism.

**2 — The same operation, including semantic paraphrases.**

- One introduces a baseline, while the other introduces a metric: both introduce, despite acting on different components.
- “Ablate the auxiliary loss” and “compare training with and without the auxiliary loss” both describe an ablation.

## 3. `specification_match`

Do the concrete choices agree, either exactly or at the functional-family level allowed by this rubric? Specifications include datasets, model families, architectures, loss terms, numeric thresholds, split ratios, and hyperparameter values. This dimension intentionally allows broader matches than exact names alone.

**0 — No matching or meaningfully comparable concrete settings.**

- CIFAR10 versus ReLU: an evaluation dataset versus an activation function.
- Setting batch size to 32 versus 128: the component and operation match, but the values do not.
- The actual decision names AdamW, while the prediction says only “replace the optimizer”: no replacement optimizer is specified.

**1 — Partial agreement, or related settings used for different tasks.**

- CIFAR10 versus COCO object detection: both involve image data, but the tasks differ.
- The learning rate matches, but batch size differs or is omitted.

**2 — The defining settings agree.**

- CIFAR10 versus CIFAR100 when used as image-classification evaluation datasets.
- Rotten Tomatoes versus SST-2 when used as sentiment-classification evaluation datasets.
- ResNet versus ViT when the choice is a generic image-feature backbone and the architecture's internal mechanism is not the subject of the decision.

Apply family equivalence to corresponding roles, not to any shared technical term. ResNet and ViT can match as generic backbones, but not when convolution versus self-attention is itself being investigated.

If the actual decision supplies no concrete specifications, assign specification score 0 without lowering the other scores.

## Worked comparisons across all three dimensions

| Actual decision | Predicted decision | Component | Operation | Specification |
| -- | -- | -- | -- | -- | 
| Replace the optimizer with AdamW. | Switch the training optimizer to AdamW. | 2 | 2 | 2 | 
| Replace the optimizer with AdamW. | Replace the optimizer. | 2 | 2 | 0 | 
| Replace the optimizer with AdamW. | Compare AdamW against the current optimizer before selecting one. | 2 | 1 | 2 | 
| Replace the optimizer with AdamW. | Replace the tokenizer with BPE. | 0 | 2 | 0 | 
| Ablate the auxiliary loss. | Permanently remove the auxiliary loss to simplify training. | 2 | 1 | 0 | 
| Set batch size to 32. | Set batch size to 128. | 2 | 2 | 0 | 
| Increase batch size to 128. | Decrease batch size to 32. | 2 | 0 | 0 | 
| Set batch size to 32 and learning rate to 0.001. | Set batch size to 128 and learning rate to 0.001. | 2 | 2 | 1 |
| Add CIFAR10 to the evaluation suite. | Add COCO object detection to the evaluation suite. | 2 | 2 | 1 |
| Introduce a new evaluation metric. | Introduce an uncertainty estimate for the existing metric. | 1 | 2 | 0 | 

## Scoring discipline and output

- Score the semantic content of the `decision` sentence. If predicted `category`, `object`, or `operation` fields are present, use them only to clarify ambiguity. They cannot override the sentence or supply an action that it does not state. Predictions with only `category` and `decision` can be scored normally.
- Do not penalize component or operation scores for missing or different specification. They can receive 2 even when specification match is 0. Conversely, matching concrete names does not establish matching components or operations. Identify each dimension separately before assigning its score.
- Judge meaning rather than writing quality. 
- If a prediction contains several independent alternatives, do not select whichever one matches best. Judge what it actually commits to in each dimension.