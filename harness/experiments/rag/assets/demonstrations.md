# Natural early-prefix next-decision demonstrations

Each solved example shows the complete one-, two-, or three-decision history available near the beginning of a project, followed by the immediate next decision.

## Training trajectory 001
### Prefix length 1

Observed decisions, from oldest to newest:

- T000 | method
  Compress GAIR/LIMO mathematical reasoning by prompting DeepSeek-R1 to rewrite each complete solution, removing redundant reasoning while preserving essential logical steps and returning only the modified solution.

Actual immediate next decision:
Category: method
Decision: Replace complete-solution rewriting with period-level redundancy classification: give QwQ-32B-Preview each candidate step together with the question, correct answer, and preceding solution steps; label redundant as 2, nonredundant as 1, and uncertain as 0, and remove only steps labeled 2.

## Training trajectory 001
### Prefix length 2

Observed decisions, from oldest to newest:

- T000 | method
  Compress GAIR/LIMO mathematical reasoning by prompting DeepSeek-R1 to rewrite each complete solution, removing redundant reasoning while preserving essential logical steps and returning only the modified solution.

- T001 | method
  Replace complete-solution rewriting with period-level redundancy classification: give QwQ-32B-Preview each candidate step together with the question, correct answer, and preceding solution steps; label redundant as 2, nonredundant as 1, and uncertain as 0, and remove only steps labeled 2.

Actual immediate next decision:
Category: method
Decision: Select reasoning steps by self-consistency: sample answer-matching DeepSeek-R1 solutions, use Qwen2.5-72B-Instruct to find semantically recurring steps, keep steps appearing at least six times, and reconstruct them into a coherent solution.

## Training trajectory 001
### Prefix length 3

Observed decisions, from oldest to newest:

- T000 | method
  Compress GAIR/LIMO mathematical reasoning by prompting DeepSeek-R1 to rewrite each complete solution, removing redundant reasoning while preserving essential logical steps and returning only the modified solution.

- T001 | method
  Replace complete-solution rewriting with period-level redundancy classification: give QwQ-32B-Preview each candidate step together with the question, correct answer, and preceding solution steps; label redundant as 2, nonredundant as 1, and uncertain as 0, and remove only steps labeled 2.

- T002 | method
  Select reasoning steps by self-consistency: sample answer-matching DeepSeek-R1 solutions, use Qwen2.5-72B-Instruct to find semantically recurring steps, keep steps appearing at least six times, and reconstruct them into a coherent solution.

Actual immediate next decision:
Category: method
Decision: Replace the Qwen reconstruction stage with a prompt that preserves every selected reasoning process, including double checks, using DeepSeek-R1-Distill-Qwen-32B or Claude-3.7-Sonnet.

## Training trajectory 002
### Prefix length 1

Observed decisions, from oldest to newest:

- T000 | method
  Introduce SmoothReLU as a sigmoid-gated smooth replacement for ReLU, controlled by a smoothness parameter, and provide a mapping that replaces ReLU modules with this activation.

Actual immediate next decision:
Category: experiment
Decision: Run a pre-training toy binary-classification ablation in which networks are trained with the replacement activation across a 0-to-1 parameter sweep.

## Training trajectory 002
### Prefix length 2

Observed decisions, from oldest to newest:

- T000 | method
  Introduce SmoothReLU as a sigmoid-gated smooth replacement for ReLU, controlled by a smoothness parameter, and provide a mapping that replaces ReLU modules with this activation.

- T001 | ablation
  Run a pre-training toy binary-classification ablation in which networks are trained with the replacement activation across a 0-to-1 parameter sweep.

Actual immediate next decision:
Category: method
Decision: Reparameterize the smooth ReLU replacement as BetaReLU, using beta in a sigmoid-gated activation so that beta controls the transition toward an unreplaced ReLU endpoint at beta=1.

## Training trajectory 002
### Prefix length 3

Observed decisions, from oldest to newest:

- T000 | method
  Introduce SmoothReLU as a sigmoid-gated smooth replacement for ReLU, controlled by a smoothness parameter, and provide a mapping that replaces ReLU modules with this activation.

- T001 | ablation
  Run a pre-training toy binary-classification ablation in which networks are trained with the replacement activation across a 0-to-1 parameter sweep.

- T002 | method
  Reparameterize the smooth ReLU replacement as BetaReLU, using beta in a sigmoid-gated activation so that beta controls the transition toward an unreplaced ReLU endpoint at beta=1.

Actual immediate next decision:
Category: experiment
Decision: Evaluate post-training activation replacement by training one ReLU toy classifier, independently copying it for each replacement setting, and comparing the resulting decision boundaries against the original ReLU boundary. The spiral-data variant uses 1,024 points with noise 0.7 and a 0.16 label-flip fraction, trains without weight decay using a StepLR decay factor of 0.1, and compares the fixed ReLU classifier with BetaAgg at beta values 0.9 and 0.5.

## Training trajectory 003
### Prefix length 1

Observed decisions, from oldest to newest:

- T000 | method
  Score item validity from inter-item dependency patterns, comparing tetrachoric correlation, L1 distance correlation, adjusted mutual information, and Chatterjee xi with mean, median, or upper-quantile aggregation and explicit thresholds.

Actual immediate next decision:
Category: method
Decision: Use a deterministic Gemini judge given each question and official answer to classify incorrect, ambiguous, or biased items, and evaluate its binary predictions against platinum labels.

## Training trajectory 003
### Prefix length 2

Observed decisions, from oldest to newest:

- T000 | method
  Score item validity from inter-item dependency patterns, comparing tetrachoric correlation, L1 distance correlation, adjusted mutual information, and Chatterjee xi with mean, median, or upper-quantile aggregation and explicit thresholds.

- T001 | method
  Use a deterministic Gemini judge given each question and official answer to classify incorrect, ambiguous, or biased items, and evaluate its binary predictions against platinum labels.

Actual immediate next decision:
Category: ablation
Decision: Ablate McDonald's omega latent-factor item scores across multiple factor structures, using general- and first-factor loadings as validity signals.

## Training trajectory 003
### Prefix length 3

Observed decisions, from oldest to newest:

- T000 | method
  Score item validity from inter-item dependency patterns, comparing tetrachoric correlation, L1 distance correlation, adjusted mutual information, and Chatterjee xi with mean, median, or upper-quantile aggregation and explicit thresholds.

- T001 | method
  Use a deterministic Gemini judge given each question and official answer to classify incorrect, ambiguous, or biased items, and evaluate its binary predictions against platinum labels.

- T002 | ablation
  Ablate McDonald's omega latent-factor item scores across multiple factor structures, using general- and first-factor loadings as validity signals.

Actual immediate next decision:
Category: experiment
Decision: Evaluate invalid-item detectors on LLM-by-item binary response matrices, using platinum labels for GSM8K and MMLU High-School Mathematics and applying the pipeline across AIR-Bench, Thai Exam, MedQA, LegalBench, WikiFact, OpenBookQA, BoolQ, BBQ, medical MMLU subsets, and full HELM Lite MMLU.

## Training trajectory 004
### Prefix length 1

Observed decisions, from oldest to newest:

- T000 | method
  Calibrate an Opacus-compatible DP mechanism directly to a target attack advantage by searching for the noise multiplier satisfying epsilon = 0 and delta = the advantage bound.

Actual immediate next decision:
Category: method
Decision: Calibrate noise to target attack FPR alpha and FNR beta by mapping candidate delta values to the required epsilon and selecting the delta/noise pair that minimizes noise; use bounded scalar minimization over delta as the sole optimization route, with bounded monotone inversion for the inner noise search.

## Training trajectory 004
### Prefix length 2

Observed decisions, from oldest to newest:

- T000 | method
  Calibrate an Opacus-compatible DP mechanism directly to a target attack advantage by searching for the noise multiplier satisfying epsilon = 0 and delta = the advantage bound.

- T001 | method
  Calibrate noise to target attack FPR alpha and FNR beta by mapping candidate delta values to the required epsilon and selecting the delta/noise pair that minimizes noise; use bounded scalar minimization over delta as the sole optimization route, with bounded monotone inversion for the inner noise search.

Actual immediate next decision:
Category: experiment
Decision: Evaluate DP-SGD noise calibration for asymmetric attack errors by sweeping TPR from 0.05 to 0.5 and TNR over 0.9, 0.95, and 0.99, comparing fixed-delta calibration at delta 1e-5 under Connect-the-Dots PLD accounting with direct PLD FPR/FNR calibration.

## Training trajectory 004
### Prefix length 3

Observed decisions, from oldest to newest:

- T000 | method
  Calibrate an Opacus-compatible DP mechanism directly to a target attack advantage by searching for the noise multiplier satisfying epsilon = 0 and delta = the advantage bound.

- T001 | method
  Calibrate noise to target attack FPR alpha and FNR beta by mapping candidate delta values to the required epsilon and selecting the delta/noise pair that minimizes noise; use bounded scalar minimization over delta as the sole optimization route, with bounded monotone inversion for the inner noise search.

- T002 | experiment
  Evaluate DP-SGD noise calibration for asymmetric attack errors by sweeping TPR from 0.05 to 0.5 and TNR over 0.9, 0.95, and 0.99, comparing fixed-delta calibration at delta 1e-5 under Connect-the-Dots PLD accounting with direct PLD FPR/FNR calibration.

Actual immediate next decision:
Category: experiment
Decision: Evaluate DP-SGD noise calibration over attack advantage values from 0.004 to 0.25 using five logarithmic points from 0.004 to 0.05 and five linear points from 0.05 to 0.25, comparing fixed-delta calibration at delta 1e-5 under Connect-the-Dots PLD accounting with direct PLD advantage calibration; use sample rate 0.001 and 10,000 steps.

## Training trajectory 005
### Prefix length 1

Observed decisions, from oldest to newest:

- T000 | method
  For trainable non-project networks, construct an undirected adjacency with a prescribed edge density by applying a symmetric Gumbel softmax, selecting hard top-k upper-triangular entries, symmetrizing them, and using a straight-through gradient.

Actual immediate next decision:
Category: method
Decision: Learn functional brain-network structures at four selectable resolutions by maintaining one network per sample, one per subject, one per class/group, or one shared across the entire project dataset.

## Training trajectory 005
### Prefix length 2

Observed decisions, from oldest to newest:

- T000 | method
  For trainable non-project networks, construct an undirected adjacency with a prescribed edge density by applying a symmetric Gumbel softmax, selecting hard top-k upper-triangular entries, symmetrizing them, and using a straight-through gradient.

- T001 | method
  Learn functional brain-network structures at four selectable resolutions by maintaining one network per sample, one per subject, one per class/group, or one shared across the entire project dataset.

Actual immediate next decision:
Category: method
Decision: Train the prediction model with task cross-entropy plus an optional subject-identity contrastive loss, weighted 0.5 relative to classification loss, that makes embeddings from homologous samples of the same person more alike, based on the stated assumption that an individual's brain state is stable.

## Training trajectory 005
### Prefix length 3

Observed decisions, from oldest to newest:

- T000 | method
  For trainable non-project networks, construct an undirected adjacency with a prescribed edge density by applying a symmetric Gumbel softmax, selecting hard top-k upper-triangular entries, symmetrizing them, and using a straight-through gradient.

- T001 | method
  Learn functional brain-network structures at four selectable resolutions by maintaining one network per sample, one per subject, one per class/group, or one shared across the entire project dataset.

- T002 | method
  Train the prediction model with task cross-entropy plus an optional subject-identity contrastive loss, weighted 0.5 relative to classification loss, that makes embeddings from homologous samples of the same person more alike, based on the stated assumption that an individual's brain state is stable.

Actual immediate next decision:
Category: experiment
Decision: Configure the evaluation across Cog State, SLIM, and DynHCP data, with DynHCP subtypes for Age, Gender, and Activity.

## Training trajectory 006
### Prefix length 1

Observed decisions, from oldest to newest:

- T000 | experiment
  Use a two-view numerical simulation in which view 0 observes latent indices [0,1] and view 1 observes [0,2], making z0 shared and z1 and z2 view-specific.

Actual immediate next decision:
Category: method
Decision: Evaluate learned shared representations with PCM and GCM conditional-independence diagnostics against both private latents jointly given the true shared latent z0, and include reverse tests that condition on each learned proxy instead of z0.

## Training trajectory 006
### Prefix length 2

Observed decisions, from oldest to newest:

- T000 | experiment
  Use a two-view numerical simulation in which view 0 observes latent indices [0,1] and view 1 observes [0,2], making z0 shared and z1 and z2 view-specific.

- T001 | experiment
  Evaluate learned shared representations with PCM and GCM conditional-independence diagnostics against both private latents jointly given the true shared latent z0, and include reverse tests that condition on each learned proxy instead of z0.

Actual immediate next decision:
Category: experiment
Decision: Generate dependent simulated latents from a three-variable linear-Gaussian structure encoded by B=[[0,0,0],[1,0,0],[1,1,0]], deriving the Gaussian covariance from the inverse of I-B.

## Training trajectory 006
### Prefix length 3

Observed decisions, from oldest to newest:

- T000 | experiment
  Use a two-view numerical simulation in which view 0 observes latent indices [0,1] and view 1 observes [0,2], making z0 shared and z1 and z2 view-specific.

- T001 | experiment
  Evaluate learned shared representations with PCM and GCM conditional-independence diagnostics against both private latents jointly given the true shared latent z0, and include reverse tests that condition on each learned proxy instead of z0.

- T002 | experiment
  Generate dependent simulated latents from a three-variable linear-Gaussian structure encoded by B=[[0,0,0],[1,0,0],[1,1,0]], deriving the Gaussian covariance from the inverse of I-B.

Actual immediate next decision:
Category: experiment
Decision: Use per-latent squared Pearson correlation (R²) between a recovered shared representation and each of z0, z1, and z2 as a representation-quality baseline across the model conditions.

## Training trajectory 007
### Prefix length 1

Observed decisions, from oldest to newest:

- T000 | experiment
  Run an early pilot comparison on synthetic Gaussian or Gaussian-mixture data using equally spaced quantiles and normalized maximum and mean rank error, while varying the number of quantiles and privacy settings.

Actual immediate next decision:
Category: method
Decision: Estimate multiple quantiles by adding correlated continual-counting noise to target ranks, taking bounded slices around the noisy ranks, and applying exponential-mechanism quantile estimation while enforcing ordered outputs.

## Training trajectory 007
### Prefix length 2

Observed decisions, from oldest to newest:

- T000 | experiment
  Run an early pilot comparison on synthetic Gaussian or Gaussian-mixture data using equally spaced quantiles and normalized maximum and mean rank error, while varying the number of quantiles and privacy settings.

- T001 | method
  Estimate multiple quantiles by adding correlated continual-counting noise to target ranks, taking bounded slices around the noisy ranks, and applying exponential-mechanism quantile estimation while enforcing ordered outputs.

Actual immediate next decision:
Category: ablation
Decision: Ablate the exponential-mechanism slice radius or its logarithmic scale factor over broad ranges and multiple datasets or target quantiles, measuring repeated rank error and comparing it with a computed upper bound.

## Training trajectory 007
### Prefix length 3

Observed decisions, from oldest to newest:

- T000 | experiment
  Run an early pilot comparison on synthetic Gaussian or Gaussian-mixture data using equally spaced quantiles and normalized maximum and mean rank error, while varying the number of quantiles and privacy settings.

- T001 | method
  Estimate multiple quantiles by adding correlated continual-counting noise to target ranks, taking bounded slices around the noisy ranks, and applying exponential-mechanism quantile estimation while enforcing ordered outputs.

- T002 | ablation
  Ablate the exponential-mechanism slice radius or its logarithmic scale factor over broad ranges and multiple datasets or target quantiles, measuring repeated rank error and comparing it with a computed upper bound.

Actual immediate next decision:
Category: method
Decision: Use a k-ary continual-counting tree with two-sided geometric node noise, automatically select its branching factor by minimizing a worst-case variance bound, and characterize its simultaneous high-probability error with a Chernoff/union bound.

## Training trajectory 008
### Prefix length 1

Observed decisions, from oldest to newest:

- T000 | experiment
  Use a multimodal reasoning evaluation suite that began with ScienceQA and MathVista and expanded to repository runners/loaders for M3CoT, MMMU-Pro, MMStar, MM-Vet, MathVision, and WeMath, with dataset-specific prompt and answer handling.

Actual immediate next decision:
Category: method
Decision: Train the initial multimodal reward model as an unbounded scalar linear head over a vision-language model's final-token vocabulary logits, using mean-squared error against correctness or rollout-derived targets.

## Training trajectory 008
### Prefix length 2

Observed decisions, from oldest to newest:

- T000 | experiment
  Use a multimodal reasoning evaluation suite that began with ScienceQA and MathVista and expanded to repository runners/loaders for M3CoT, MMMU-Pro, MMStar, MM-Vet, MathVision, and WeMath, with dataset-specific prompt and answer handling.

- T001 | method
  Train the initial multimodal reward model as an unbounded scalar linear head over a vision-language model's final-token vocabulary logits, using mean-squared error against correctness or rollout-derived targets.

Actual immediate next decision:
Category: method
Decision: Elicit initial reasoning traces with a two-shot Phi-3.5 prompt that prepends worked examples and requests step-by-step answers.

## Training trajectory 008
### Prefix length 3

Observed decisions, from oldest to newest:

- T000 | experiment
  Use a multimodal reasoning evaluation suite that began with ScienceQA and MathVista and expanded to repository runners/loaders for M3CoT, MMMU-Pro, MMStar, MM-Vet, MathVision, and WeMath, with dataset-specific prompt and answer handling.

- T001 | method
  Train the initial multimodal reward model as an unbounded scalar linear head over a vision-language model's final-token vocabulary logits, using mean-squared error against correctness or rollout-derived targets.

- T002 | method
  Elicit initial reasoning traces with a two-shot Phi-3.5 prompt that prepends worked examples and requests step-by-step answers.

Actual immediate next decision:
Category: method
Decision: Estimate each reasoning prefix's process target by repeatedly sampling continuations and setting its reward to the fraction judged correct.

## Training trajectory 009
### Prefix length 1

Observed decisions, from oldest to newest:

- T000 | ablation
  Provide a z-loss ablation for pretraining by selecting LOSS_TYPE=ZLOSS and using fused linear cross entropy with an LSE-square scale of 1.0e-4.

Actual immediate next decision:
Category: experiment
Decision: For the separately selectable PROPOSED training path, constrain every output-embedding row to unit L2 norm by normalizing the language-model head weights before each forward pass.

## Training trajectory 009
### Prefix length 2

Observed decisions, from oldest to newest:

- T000 | ablation
  Provide a z-loss ablation for pretraining by selecting LOSS_TYPE=ZLOSS and using fused linear cross entropy with an LSE-square scale of 1.0e-4.

- T001 | method
  For the separately selectable PROPOSED training path, constrain every output-embedding row to unit L2 norm by normalizing the language-model head weights before each forward pass.

Actual immediate next decision:
Category: experiment
Decision: Run vocabulary-size experiments at 24K, 49K, 98K, and 196K using the corresponding FineWeb-Edu tokenizer and tokenized dataset.

## Training trajectory 009
### Prefix length 3

Observed decisions, from oldest to newest:

- T000 | ablation
  Provide a z-loss ablation for pretraining by selecting LOSS_TYPE=ZLOSS and using fused linear cross entropy with an LSE-square scale of 1.0e-4.

- T001 | method
  For the separately selectable PROPOSED training path, constrain every output-embedding row to unit L2 norm by normalizing the language-model head weights before each forward pass.

- T002 | experiment
  Run vocabulary-size experiments at 24K, 49K, 98K, and 196K using the corresponding FineWeb-Edu tokenizer and tokenized dataset.

Actual immediate next decision:
Category: ablation
Decision: Run an embedding-tying ablation for the 12-layer Llama configuration by setting tie_word_embeddings to true and training tied variants at 24K, 49K, 98K, and 196K with the PROPOSED path and learning rate 6.0e-4.

## Training trajectory 010
### Prefix length 1

Observed decisions, from oldest to newest:

- T000 | ablation
  Ablate fixed relative bias from -1 to 1 while training temperature in synthetic SigLIP, and measure loss and the gap between the worst matching and best non-matching similarities across both dense and five-point bias sweeps.

Actual immediate next decision:
Category: ablation
Decision: Compare fixed temperature 100 with a trainable temperature initialized at 10 across five fixed relative biases in two-modality synthetic SigLIP, measuring loss, learned scale, and matching/non-matching geometry.

## Training trajectory 010
### Prefix length 2

Observed decisions, from oldest to newest:

- T000 | ablation
  Ablate fixed relative bias from -1 to 1 while training temperature in synthetic SigLIP, and measure loss and the gap between the worst matching and best non-matching similarities across both dense and five-point bias sweeps.

- T001 | ablation
  Compare fixed temperature 100 with a trainable temperature initialized at 10 across five fixed relative biases in two-modality synthetic SigLIP, measuring loss, learned scale, and matching/non-matching geometry.

Actual immediate next decision:
Category: experiment
Decision: Extend synthetic SigLIP from two to three and four modalities and then sweep 4, 6, 8, and 10 modalities, measuring loss and pooled matching/non-matching similarities across every modality pair.

## Training trajectory 010
### Prefix length 3

Observed decisions, from oldest to newest:

- T000 | ablation
  Ablate fixed relative bias from -1 to 1 while training temperature in synthetic SigLIP, and measure loss and the gap between the worst matching and best non-matching similarities across both dense and five-point bias sweeps.

- T001 | ablation
  Compare fixed temperature 100 with a trainable temperature initialized at 10 across five fixed relative biases in two-modality synthetic SigLIP, measuring loss, learned scale, and matching/non-matching geometry.

- T002 | experiment
  Extend synthetic SigLIP from two to three and four modalities and then sweep 4, 6, 8, and 10 modalities, measuring loss and pooled matching/non-matching similarities across every modality pair.

Actual immediate next decision:
Category: experiment
Decision: Evaluate modality separation in pretrained SigLIP B/16-384 on ImageNet validation images and class-label text using matching/mismatching similarities and a perceptron probe.

## Training trajectory 011
### Prefix length 1

Observed decisions, from oldest to newest:

- T000 | method
  Augment masked L1 reconstruction training with a source-focused auxiliary loss: detect and perform elliptical photometry on HR patches, form a flux-weighted Gaussian attention map from source position, shape, orientation, and absolute flux, and add attention-weighted absolute reconstruction error, generally with coefficient 0.01, to the image loss.

Actual immediate next decision:
Category: method
Decision: Construct the HST DRC dataset by selecting products with NCOMBINE equal to 4, spatially de-duplicating and partitioning observations using WCS/RA boundaries (train below RA 250 and evaluation above RA 255), checking train/test polygons at a 3-arcsec threshold, and extracting spatially aligned 256×256 HR and 128×128 LR half-stride pairs with more than 80% valid pixels after continuous zero regions are treated as missing.

## Training trajectory 011
### Prefix length 2

Observed decisions, from oldest to newest:

- T000 | method
  Augment masked L1 reconstruction training with a source-focused auxiliary loss: detect and perform elliptical photometry on HR patches, form a flux-weighted Gaussian attention map from source position, shape, orientation, and absolute flux, and add attention-weighted absolute reconstruction error, generally with coefficient 0.01, to the image loss.

- T001 | experiment
  Construct the HST DRC dataset by selecting products with NCOMBINE equal to 4, spatially de-duplicating and partitioning observations using WCS/RA boundaries (train below RA 250 and evaluation above RA 255), checking train/test polygons at a 3-arcsec threshold, and extracting spatially aligned 256×256 HR and 128×128 LR half-stride pairs with more than 80% valid pixels after continuous zero regions are treated as missing.

Actual immediate next decision:
Category: method
Decision: Create each model input on the fly from a standardized target by convolving it with a fixed normalized PSF formed as the product of Airy and Gaussian atmospheric-turbulence components, with resolution reduction left inactive.

## Training trajectory 011
### Prefix length 3

Observed decisions, from oldest to newest:

- T000 | method
  Augment masked L1 reconstruction training with a source-focused auxiliary loss: detect and perform elliptical photometry on HR patches, form a flux-weighted Gaussian attention map from source position, shape, orientation, and absolute flux, and add attention-weighted absolute reconstruction error, generally with coefficient 0.01, to the image loss.

- T001 | experiment
  Construct the HST DRC dataset by selecting products with NCOMBINE equal to 4, spatially de-duplicating and partitioning observations using WCS/RA boundaries (train below RA 250 and evaluation above RA 255), checking train/test polygons at a 3-arcsec threshold, and extracting spatially aligned 256×256 HR and 128×128 LR half-stride pairs with more than 80% valid pixels after continuous zero regions are treated as missing.

- T002 | experiment
  Create each model input on the fly from a standardized target by convolving it with a fixed normalized PSF formed as the product of Airy and Gaussian atmospheric-turbulence components, with resolution reduction left inactive.

Actual immediate next decision:
Category: method
Decision: Evaluate astronomical reconstruction with a normalized flux-error measure computed after background subtraction and star detection, comparing detected-region or aperture flux between the prediction and ground truth and reporting it as TFE alongside conventional image metrics where enabled.

## Training trajectory 012
### Prefix length 1

Observed decisions, from oldest to newest:

- T000 | method
  At the generated score-token position, project every transformer layer's hidden state through the language-model head, restrict to allowed numeric score tokens, softmax those logits, and compute both the most likely score and the probability-weighted expected score for each layer.

Actual immediate next decision:
Category: ablation
Decision: Ablate output-score granularity by generating rubric-free Flask prompts whose maximum allowed score is swept over 5, 9, 19, 29, 39, 49, 59, 69, 79, 89, and 99.

## Training trajectory 012
### Prefix length 2

Observed decisions, from oldest to newest:

- T000 | method
  At the generated score-token position, project every transformer layer's hidden state through the language-model head, restrict to allowed numeric score tokens, softmax those logits, and compute both the most likely score and the probability-weighted expected score for each layer.

- T001 | ablation
  Ablate output-score granularity by generating rubric-free Flask prompts whose maximum allowed score is swept over 5, 9, 19, 29, 39, 49, 59, 69, 79, 89, and 99.

Actual immediate next decision:
Category: experiment
Decision: Evaluate point-wise judge scores against human ratings on Flask, HelpSteer, and BIGGen, with model-specific result/validation handling including Qwen alongside the previously supported model families.

## Training trajectory 012
### Prefix length 3

Observed decisions, from oldest to newest:

- T000 | method
  At the generated score-token position, project every transformer layer's hidden state through the language-model head, restrict to allowed numeric score tokens, softmax those logits, and compute both the most likely score and the probability-weighted expected score for each layer.

- T001 | ablation
  Ablate output-score granularity by generating rubric-free Flask prompts whose maximum allowed score is swept over 5, 9, 19, 29, 39, 49, 59, 69, 79, 89, and 99.

- T002 | experiment
  Evaluate point-wise judge scores against human ratings on Flask, HelpSteer, and BIGGen, with model-specific result/validation handling including Qwen alongside the previously supported model families.

Actual immediate next decision:
Category: ablation
Decision: Ablate evaluator prompting by comparing score-only prompts with prompts that require written feedback before the numeric score.

## Training trajectory 013
### Prefix length 1

Observed decisions, from oldest to newest:

- T000 | method
  Formulate verified-programming benchmark tasks in Lean by pairing candidate implementations with logical specifications and proof obligations, and bind the existential result witness to the candidate output with conjunction (`∃ result, impl ... = result ∧ spec result`) so that the returned result itself must satisfy the specification.

Actual immediate next decision:
Category: experiment
Decision: Evaluate each configured Lean task on an all-or-nothing point basis, awarding its assigned score only when the build log contains no `declaration uses 'sorry'` diagnostic for that task.

## Training trajectory 013
### Prefix length 2

Observed decisions, from oldest to newest:

- T000 | method
  Formulate verified-programming benchmark tasks in Lean by pairing candidate implementations with logical specifications and proof obligations, and bind the existential result witness to the candidate output with conjunction (`∃ result, impl ... = result ∧ spec result`) so that the returned result itself must satisfy the specification.

- T001 | experiment
  Evaluate each configured Lean task on an all-or-nothing point basis, awarding its assigned score only when the build log contains no `declaration uses 'sorry'` diagnostic for that task.

Actual immediate next decision:
Category: method
Decision: Use the indexed HumanEval problem corpus—retaining Python task descriptions and reference implementations alongside the earlier Rust material—as source material for tasks translated into Lean specifications.

## Training trajectory 013
### Prefix length 3

Observed decisions, from oldest to newest:

- T000 | method
  Formulate verified-programming benchmark tasks in Lean by pairing candidate implementations with logical specifications and proof obligations, and bind the existential result witness to the candidate output with conjunction (`∃ result, impl ... = result ∧ spec result`) so that the returned result itself must satisfy the specification.

- T001 | experiment
  Evaluate each configured Lean task on an all-or-nothing point basis, awarding its assigned score only when the build log contains no `declaration uses 'sorry'` diagnostic for that task.

- T002 | experiment
  Use the indexed HumanEval problem corpus—retaining Python task descriptions and reference implementations alongside the earlier Rust material—as source material for tasks translated into Lean specifications.

Actual immediate next decision:
Category: method
Decision: Represent machine-level failure or out-of-domain behavior explicitly with `Option` outputs and preconditions, requiring `some` results to satisfy the specification and allowing `none` only when the corresponding precondition fails.

## Training trajectory 014
### Prefix length 1

Observed decisions, from oldest to newest:

- T000 | method
  Use a DeepSVDD hypersphere over learned text embeddings for anomaly detection, with squared distance to a center and support for one-class or soft-boundary objectives.

Actual immediate next decision:
Category: experiment
Decision: Evaluate continuous detector scores with ROC-AUC, including DeepSVDD distances, KNN confidence, and HRN outputs.

## Training trajectory 014
### Prefix length 2

Observed decisions, from oldest to newest:

- T000 | method
  Use a DeepSVDD hypersphere over learned text embeddings for anomaly detection, with squared distance to a center and support for one-class or soft-boundary objectives.

- T001 | experiment
  Evaluate continuous detector scores with ROC-AUC, including DeepSVDD distances, KNN confidence, and HRN outputs.

Actual immediate next decision:
Category: method
Decision: Estimate the DeepSVDD hypersphere center using only machine-generated training texts.

## Training trajectory 014
### Prefix length 3

Observed decisions, from oldest to newest:

- T000 | method
  Use a DeepSVDD hypersphere over learned text embeddings for anomaly detection, with squared distance to a center and support for one-class or soft-boundary objectives.

- T001 | experiment
  Evaluate continuous detector scores with ROC-AUC, including DeepSVDD distances, KNN confidence, and HRN outputs.

- T002 | method
  Estimate the DeepSVDD hypersphere center using only machine-generated training texts.

Actual immediate next decision:
Category: method
Decision: Ablate the sample population used by the standard DeepSVDD compactness loss between machine-only and all training samples, returning to machine-only before this objective was replaced.

## Training trajectory 015
### Prefix length 1

Observed decisions, from oldest to newest:

- T000 | experiment
  Use multi-quadrant Euler flows with both open and periodic boundaries as the primary autoencoder and emulator dataset.

Actual immediate next decision:
Category: experiment
Decision: Evaluate full held-out trajectories with repeated stochastic samples, physical-unit reconstruction and forecast errors, ensemble spread and skill, invariants, spectral and correlation errors, and rollout timing against both truth and decoded truth.

## Training trajectory 015
### Prefix length 2

Observed decisions, from oldest to newest:

- T000 | experiment
  Use multi-quadrant Euler flows with both open and periodic boundaries as the primary autoencoder and emulator dataset.

- T001 | experiment
  Evaluate full held-out trajectories with repeated stochastic samples, physical-unit reconstruction and forecast errors, ensemble spread and skill, invariants, spectral and correlation errors, and rollout timing against both truth and decoded truth.

Actual immediate next decision:
Category: experiment
Decision: Train the deterministic autoencoder with a variance-normalized per-field RMSE objective instead of plain MSE or MAE.

## Training trajectory 015
### Prefix length 3

Observed decisions, from oldest to newest:

- T000 | experiment
  Use multi-quadrant Euler flows with both open and periodic boundaries as the primary autoencoder and emulator dataset.

- T001 | experiment
  Evaluate full held-out trajectories with repeated stochastic samples, physical-unit reconstruction and forecast errors, ensemble spread and skill, invariants, spectral and correlation errors, and rollout timing against both truth and decoded truth.

- T002 | method
  Train the deterministic autoencoder with a variance-normalized per-field RMSE objective instead of plain MSE or MAE.

Actual immediate next decision:
Category: experiment
Decision: Preprocess Euler by applying log1p to positive thermodynamic fields and standardizing all five fields with fixed dataset statistics, replacing temporary clipping choices.

## Training trajectory 016
### Prefix length 1

Observed decisions, from oldest to newest:

- T000 | experiment
  Use scraped Braingle Math and Logic brainteasers as benchmark data, retaining each item’s question, answer, optional hint, popularity/fun rating, and difficulty rating.

Actual immediate next decision:
Category: experiment
Decision: Evaluate brainteaser solving across repeated examples and successive model settings.

## Training trajectory 016
### Prefix length 2

Observed decisions, from oldest to newest:

- T000 | experiment
  Use scraped Braingle Math and Logic brainteasers as benchmark data, retaining each item’s question, answer, optional hint, popularity/fun rating, and difficulty rating.

- T001 | experiment
  Evaluate brainteaser solving across repeated examples and successive model settings.

Actual immediate next decision:
Category: experiment
Decision: Run a prompt-strategy ablation for brainteaser solving on both Math and Logic: a basic full-reasoning prompt, an anti-brute-force/math prompt that permits brute force or code only when necessary, explicit versus implicit formal-logic prompting, and a combined condition that supplies the benchmark hint together with the anti-brute-force and rigorous-justification instructions.

## Training trajectory 016
### Prefix length 3

Observed decisions, from oldest to newest:

- T000 | experiment
  Use scraped Braingle Math and Logic brainteasers as benchmark data, retaining each item’s question, answer, optional hint, popularity/fun rating, and difficulty rating.

- T001 | experiment
  Evaluate brainteaser solving across repeated examples and successive model settings.

- T002 | ablation
  Run a prompt-strategy ablation for brainteaser solving on both Math and Logic: a basic full-reasoning prompt, an anti-brute-force/math prompt that permits brute force or code only when necessary, explicit versus implicit formal-logic prompting, and a combined condition that supplies the benchmark hint together with the anti-brute-force and rigorous-justification instructions.

Actual immediate next decision:
Category: experiment
Decision: Score direct-solving response correctness, excluding semantic-parsing outputs, with a binary model judge that receives the model response together with the Braingle reference solution.

## Training trajectory 017
### Prefix length 1

Observed decisions, from oldest to newest:

- T000 | experiment
  Compare instruction-triggered and explicit-token SST-2 backdoors as conventional attack baselines.

Actual immediate next decision:
Category: experiment
Decision: Measure infection on clean and fixed task-specific prompts using exact payload strings, keyword rules, sentiment judgments, and triggered versus untriggered accuracy.

## Training trajectory 017
### Prefix length 2

Observed decisions, from oldest to newest:

- T000 | experiment
  Compare instruction-triggered and explicit-token SST-2 backdoors as conventional attack baselines.

- T001 | experiment
  Measure infection on clean and fixed task-specific prompts using exact payload strings, keyword rules, sentiment judgments, and triggered versus untriggered accuracy.

Actual immediate next decision:
Category: ablation
Decision: Ablate full versus subset-limited infection and sweep poisoning rates for person-opinion objectives.

## Training trajectory 017
### Prefix length 3

Observed decisions, from oldest to newest:

- T000 | experiment
  Compare instruction-triggered and explicit-token SST-2 backdoors as conventional attack baselines.

- T001 | experiment
  Measure infection on clean and fixed task-specific prompts using exact payload strings, keyword rules, sentiment judgments, and triggered versus untriggered accuracy.

- T002 | ablation
  Ablate full versus subset-limited infection and sweep poisoning rates for person-opinion objectives.

Actual immediate next decision:
Category: experiment
Decision: Compare VIA with direct malicious query–response poisoning and an output-only direct-poisoning control at matched poison fractions.

## Training trajectory 018
### Prefix length 1

Observed decisions, from oldest to newest:

- T000 | ablation
  Ablate year conditioning in Stable-Diffusion LoRA fine-tuning and evaluate generated class-year images with CLIP-based KID against matching test folders.

Actual immediate next decision:
Category: experiment
Decision: Analyze generation quality by class dynamics and by the number of training images available for each class-year condition.

## Training trajectory 018
### Prefix length 2

Observed decisions, from oldest to newest:

- T000 | ablation
  Ablate year conditioning in Stable-Diffusion LoRA fine-tuning and evaluate generated class-year images with CLIP-based KID against matching test folders.

- T001 | experiment
  Analyze generation quality by class dynamics and by the number of training images available for each class-year condition.

Actual immediate next decision:
Category: method
Decision: Create the image pool with vehicle detection and deduplicate each source class using DINOv2 embedding similarity.

## Training trajectory 018
### Prefix length 3

Observed decisions, from oldest to newest:

- T000 | ablation
  Ablate year conditioning in Stable-Diffusion LoRA fine-tuning and evaluate generated class-year images with CLIP-based KID against matching test folders.

- T001 | experiment
  Analyze generation quality by class dynamics and by the number of training images available for each class-year condition.

- T002 | method
  Create the image pool with vehicle detection and deduplicate each source class using DINOv2 embedding similarity.

Actual immediate next decision:
Category: method
Decision: Annotate car brand and model with a two-stage pipeline in which Qwen proposes a label and GPT-4o verifies or corrects it before taxonomy normalization.

## Training trajectory 019
### Prefix length 1

Observed decisions, from oldest to newest:

- T000 | method
  Trace PyTorch computation graphs and replace supported operations with Taylor-mode primitives so composed neural networks can be differentiated as jets.

Actual immediate next decision:
Category: method
Decision: Generalize Taylor propagation to arbitrary order using integer partitions and Faà di Bruno multiplicities, including explicit high-order derivatives for nonlinear activations.

## Training trajectory 019
### Prefix length 2

Observed decisions, from oldest to newest:

- T000 | method
  Trace PyTorch computation graphs and replace supported operations with Taylor-mode primitives so composed neural networks can be differentiated as jets.

- T001 | method
  Generalize Taylor propagation to arbitrary order using integer partitions and Faà di Bruno multiplicities, including explicit high-order derivatives for nonlinear activations.

Actual immediate next decision:
Category: method
Decision: Represent jet coefficients as traceable replicas and commute replication through the graph to eliminate redundant work.

## Training trajectory 019
### Prefix length 3

Observed decisions, from oldest to newest:

- T000 | method
  Trace PyTorch computation graphs and replace supported operations with Taylor-mode primitives so composed neural networks can be differentiated as jets.

- T001 | method
  Generalize Taylor propagation to arbitrary order using integer partitions and Faà di Bruno multiplicities, including explicit high-order derivatives for nonlinear activations.

- T002 | method
  Represent jet coefficients as traceable replicas and commute replication through the graph to eliminate redundant work.

Actual immediate next decision:
Category: method
Decision: Estimate Laplacian and weighted-Laplacian traces from random directional derivatives, validate convergence against exact references, and benchmark how collapsed propagation scales with sample count.
