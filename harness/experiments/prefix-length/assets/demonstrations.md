# Reference research decision trajectories

These records are inputs for learning a next-decision prediction procedure. Decision descriptions state the scientific action without retrospective outcome clauses. Evidence excerpts, commit metadata, abstracts, and dates are omitted. Derive portable guidance from the records without copying domain-specific solutions.

## Demonstration idx=465

Project: LIMOPro: Reasoning Refinement for Efficient and Effective Test-time Scaling

Trajectory insight: The repository explores three ways to shorten distilled reasoning: whole-solution rewriting, sentence-level redundancy labels, and self-consistency selection. Each is eventually abandoned because it rewrites or removes steps without a stable account of their role. The retained direction instead separates progressive steps from verification and correction groups, then shifts importance measurement from two conceptual CIE formulas to an operational answer-perplexity ratio. This enables PIR to prune only low-importance functional groups while preserving the solution path. Random and category-only pruning become controls rather than the method, and the evaluation expands from an early LIMO-only protocol to matched original/refined comparisons across S1, LIMO, and LIMO-V2 on AIME, AMC, and GPQA Diamond. Chunking, direct-answer prompting, token-budget sweeps, and model-judge grading are tried but removed, leaving the final claim centered on targeted training-data refinement and accuracy-per-token gains rather than prompt-time suppression of reasoning.

Decisions:

- T000 D001 | method | superseded by D002
  Compress GAIR/LIMO mathematical reasoning by prompting DeepSeek-R1 to rewrite each complete solution, removing redundant reasoning while preserving essential logical steps and returning only the modified solution.

- T001 D002 | method | abandoned
  Replace complete-solution rewriting with period-level redundancy classification: give QwQ-32B-Preview each candidate step together with the question, correct answer, and preceding solution steps; label redundant as 2, nonredundant as 1, and uncertain as 0, and remove only steps labeled 2.

- T002 D003 | method | superseded by D005
  Select reasoning steps by self-consistency: sample answer-matching DeepSeek-R1 solutions, use Qwen2.5-72B-Instruct to find semantically recurring steps, keep steps appearing at least six times, and reconstruct them into a coherent solution.

- T003 D005 | method | abandoned
  Replace the Qwen reconstruction stage with a prompt that preserves every selected reasoning process, including double checks, using DeepSeek-R1-Distill-Qwen-32B or Claude-3.7-Sonnet.

- T004 D006 | method | superseded by D007
  Analyze each mathematical reasoning solution as coherent sentence groups under a five-pattern taxonomy—regular reasoning, backtracking verification, multiple verification, reflective deepening, and error correction—with typed relationships between reasoning steps and structured JSON output.

- T005 D007 | method | retained
  Analyze each solution through two sequential annotation stages: first partition every original sentence into coherent, typically three-to-four-sentence, nonoverlapping groups, then label every group as regular reasoning, backtracking verification, multiple verification, or error correction and link each functional group to the regular-reasoning group it checks or corrects, using DeepSeek-R1 or Claude-3.7-Sonnet as supported annotation backends.

- T006 D008 | method | superseded by D018
  Exclude overlong solutions from whole-prompt structural annotation.

- T007 D009 | ablation | abandoned
  Ablate random removal of contiguous functional-reasoning groups over multiple pruning ratios, with complete functional-group removal for traces of at least 16,000 tokens.

- T008 D010 | ablation | abandoned
  Ablate the functional categories by dropping each of backtracking verification, multiple verification, and error correction alone, retaining each alone, or removing all three.

- T009 D011 | experiment | superseded by D019
  Evaluate early LIMO pruning variants on AIME and AMC.

- T010 D016 | experiment | retained
  Test PIR on S1/Gemini reasoning data by structurally annotating and pruning it, fine-tuning S1-32B-P, and comparing it with S1-32B on AIME, AMC, and GPQA Diamond.

- T011 D012 | method | superseded by D013
  Evaluate each reasoning step with an initial Combined Information Effectiveness score defined as the normalized sum of step surprisal, -log2 p(step given its preceding steps), and the log ratio between correct-answer probability with the complete reasoning sequence and with that step removed; use low CIE values to identify steps for filtering.

- T012 D013 | method | superseded by D014
  Revise CIE so a step's information-content component is the mean normalized token-level surprisal conditioned on preceding steps and earlier tokens in that step, retain the normalized log correct-answer probability ratio as the answer-contribution component, and define sequence-level CIE as the mean of its step scores.

- T013 D014 | method | retained
  Replace the conceptual CIE formula with executable group importance based on final-answer perplexity with and without each annotated group. Score eligible functional groups with Qwen2.5-32B-Instruct and omit initial, answer-containing, overlong, or answerless cases.

- T014 D015 | method | retained
  Apply perplexity-guided importance pruning by combining adjacent functional groups linked to the same progressive step, averaging their D014 scores, and removing the lowest-ranked fraction while preserving progressive reasoning.

- T015 D017 | experiment | retained
  Test PIR on LIMO-V2/QwQ reasoning data by annotating and pruning ratio variants, fine-tuning LIMO-V2-P, and comparing it with LIMO-V2 on AIME, AMC, and GPQA Diamond.

- T016 D019 | experiment | retained
  Adopt a matched final evaluation across original-data controls and PIR-refined models from the Qwen, R1-Distill, QwQ, S1, LIMO, and LIMO-V2 families on AIME, AMC, and GPQA Diamond, reporting rule-graded accuracy, response tokens, and accuracy per token.

- T017 D020 | experiment | abandoned
  Ablate test-time generation budgets from 2,048 to 24,576 tokens for a selected PIR-pruned LIMO checkpoint and unpruned controls.

- T018 D018 | method | abandoned
  Replace whole-trace length exclusion with chunked two-stage annotation: group sentences within roughly 10,000-token chunks, merge and renumber the groups, and categorize the combined structure in a second Claude-3.7 call.

- T019 D021 | experiment | abandoned
  Add a direct-answer baseline that asks models to conclude without step-by-step reasoning under the same stochastic sampling and boxed-answer format.

- T020 D022 | experiment | superseded by D019
  Replace rule-based answer grading with a deterministic Qwen2.5-32B judge over the question, reference answer, and generated response.

## Demonstration idx=53

Project: Curvature Tuning: Provable Training-free Model Steering From a Single Parameter

Trajectory insight: The repository trajectory moves from defining and visualizing scalar-controlled ReLU replacements on toy classification tasks to fixed-weight evaluation across same-dataset classification, cross-dataset probing, robustness, segmentation, and multiple backbone families. The activation converges from SmoothReLU through BetaReLU to the retained BetaAgg mixture, while the transfer readout changes from a gradient-trained head to fitted logistic regression. Several substantive branches are later discontinued: normal/suboptimal/overfit training-regime comparisons, class-dependent Gaussian and Gaussian-mixture corruption, k-NN transfer, transferred-probe robustness, whole-network segmentation training, and the activation-family comparison. Architecture coverage also changes visibly, expanding across ResNet depths, briefly adding VGG-19, and then replacing VGG with Swin transformers. Later retained work increasingly centers on replace-then-probe, multi-label and regression transfers, and explicit aggregation rules for gains and preferred beta settings.

Decisions:

- T000 D001 | method | superseded by D026
  Introduce SmoothReLU as a sigmoid-gated smooth replacement for ReLU, controlled by a smoothness parameter, and provide a mapping that replaces ReLU modules with this activation.

- T001 D002 | experiment | retained
  Run a pre-training toy binary-classification ablation in which networks are trained with the replacement activation across a 0-to-1 parameter sweep.

- T002 D026 | method | superseded by D028
  Reparameterize the smooth ReLU replacement as BetaReLU, using beta in a sigmoid-gated activation so that beta controls the transition toward an unreplaced ReLU endpoint at beta=1.

- T003 D003 | experiment | retained
  Evaluate post-training activation replacement by training one ReLU toy classifier, independently copying it for each replacement setting, and comparing the resulting decision boundaries against the original ReLU boundary. The spiral-data variant uses 1,024 points with noise 0.7 and a 0.16 label-flip fraction, trains without weight decay using a StepLR decay factor of 0.1, and compares the fixed ReLU classifier with BetaAgg at beta values 0.9 and 0.5.

- T004 D004 | experiment | retained
  Evaluate same-dataset classification after fixed-weight ReLU-to-smooth-activation replacement across CIFAR-100, CIFAR-10, MNIST, and ImageNet models, comparing independently copied replacements against the ReLU endpoint using test or validation accuracy. CIFAR-100 and CIFAR-10 include normally trained, suboptimal, and 2,000-example subset-overfit ResNet-18 checkpoints; MNIST uses a three-channel-input ResNet-18 trained for 10 epochs under the normal condition; and ImageNet uses an off-the-shelf pretrained ResNet-18. The shared launcher sweeps beta from 0.5 to below 1 in steps of 0.01 under seeds 42, 43, and 44 with dataset-specific normalization and input adaptation.

- T005 D033 | experiment | superseded by D034
  Use a gradient-trained linear probe as the downstream transfer readout: freeze the pretrained backbone, replace its classifier, and optimize the new head for 50 epochs before evaluating post-hoc activation steering.

- T006 D005 | experiment | retained
  Evaluate frozen-feature multinomial-logistic-regression linear probes followed by fixed-weight ReLU-to-smooth-activation replacement across cross-dataset transfers. The original matrix uses source models from CIFAR-100, ImageNet, CIFAR-10, and MNIST and targets CIFAR-100, CIFAR-10, or MNIST, excluding same-dataset pairs. The protocol uses source-conditioned resizing/channel adaptation and target-dataset normalization, sweeps beta from 0.5 to below 1 in steps of 0.01, and runs under seeds 42, 43, and 44.

- T007 D030 | experiment | abandoned
  Compare post-hoc activation steering across base models trained under three regimes: normal learning-rate scheduling, a suboptimal constant-learning-rate schedule, and overfitting on a random 2,000-example subset.

- T008 D007 | experiment | abandoned
  Evaluate CIFAR-100-to-CIFAR-10 transfer with a 5-nearest-neighbor classifier by extracting features from ReLU and post-hoc BetaReLU versions of ResNet-18 checkpoints pretrained under normal, suboptimal, and overfit conditions, fitting and evaluating k-NN separately for each beta, and sweeping beta from 0.95 to below 1 in steps of 0.001 plus the ReLU endpoint.

- T009 D031 | experiment | superseded by D032
  Evaluate activation steering when the base CIFAR model is trained with label-dependent Gaussian noise, assigning each class a distinct noise distribution.

- T010 D032 | experiment | abandoned
  Replace the fixed class-specific Gaussian corruption with a class-conditioned mixture over shared Gaussian components, controlled by a dominance parameter alpha.

- T011 D009 | experiment | retained
  Evaluate same-dataset robustness after fixed-weight ReLU-to-smooth-activation replacement for CIFAR-10, CIFAR-100, and ImageNet under Linf, L2, and common-corruption threats. The shared pipeline compares the ReLU endpoint with independently copied replacements over beta values from 0.5 to below 1 in steps of 0.01, uses dataset-specific normalization and seeded attacks, and evaluates 1,000 examples per condition. ImageNet robustness inputs are conditionally resized to 256 and center-cropped to 224 before tensor conversion and normalization. Corruption inputs use distinct handling for PIL clean images and NumPy corruption arrays to preserve their dimension order.

- T012 D010 | experiment | abandoned
  Evaluate frozen-feature linear-probe transfer models for post-replacement robustness by comparing ReLU with independently copied BetaReLU replacements using RobustBench AutoAttack under Linf epsilon 8/255 and L2 epsilon 0.5.

- T013 D011 | experiment | retained
  Test post-hoc BetaReLU replacement on off-the-shelf RobustBench CIFAR-10 Linf models by holding their weights fixed, sweeping beta from 0.95 to below 1 in steps of 0.01 against the ReLU endpoint, and evaluating standard AutoAttack Linf robustness on all 10,000 CIFAR-10 test examples with epsilon 8/255.

- T014 D034 | experiment | retained
  Replace the gradient-trained transfer probe with multinomial logistic regression fitted directly on frozen backbone features, then copy its coefficients into a frozen linear head so no probe-training loop is required.

- T015 D014 | ablation | retained
  Compare the order of activation replacement and frozen-feature linear probing across the cross-dataset transfer matrix: either fit the logistic-regression probe on the ReLU source model and then sweep post-hoc replacements, or independently replace ReLU for each beta before extracting features and fitting that beta-specific probe. The protocol uses target-dataset normalization, beta values from 0.5 to below 1 in steps of 0.01, the ReLU endpoint, target test accuracy, and seeds 42, 43, and 44.

- T016 D015 | experiment | retained
  Evaluate BetaAgg steering for cross-task transfer from an ImageNet-pretrained ResNet-50 backbone to VOC2012 semantic segmentation with PSPNet-50: replace only backbone activations for each beta, freeze that backbone, train the pyramid-pooling, classifier, and auxiliary modules for 50 epochs, and evaluate the corresponding checkpoint using validation mIoU. The protocol uses seeds 42, 43, and 44 and training and validation batch sizes of 64.

- T017 D027 | method | retained
  Use one shared scalar beta per replaced activation module rather than materializing beta along selected feature axes, limiting the steering control to one value for each module.

- T018 D016 | experiment | abandoned
  Ablate segmentation training scope on VOC2012 with PSPNet-50 by comparing the frozen-backbone BetaAgg protocol against an end-to-end-trained ReLU network. In the whole-network arm, train the complete ReLU PSPNet with the backbone at the base learning rate and task modules at 10 times that rate, then hold that checkpoint fixed and evaluate post-hoc replacements.

- T019 D017 | experiment | retained
  Ablate which final model layers feed the cross-dataset linear probe by concatenating flattened outputs from the last top-k top-level model layers and comparing top-k values 1, 2, and 3 under seeds 42, 43, and 44. The implementation fits sklearn logistic regression when top-k is 1, but for top-k greater than 1 trains a linear head for 30 epochs with Adam at learning rate 0.001 and batch size 1,000 before freezing it.

- T020 D018 | experiment | retained
  Ablate the logistic-regression regularization strength in the cross-dataset linear-probe experiments by comparing C values 0.1 and 10 with the default C=1 under seeds 42, 43, and 44.

- T021 D028 | method | retained
  Extend the sigmoid-gated BetaSwish activation into BetaAgg by adding a beta-controlled softplus branch and combining the two branches with a mixture coefficient; make BetaAgg the default ReLU replacement.

- T022 D019 | ablation | abandoned
  Ablate the smooth replacement activation family on the fixed-weight spiral classifier by comparing the ReLU boundary against BetaSwish, beta-controlled softplus, and their BetaAgg mixture over beta values 0.7 and 0.5.

- T023 D029 | ablation | retained
  Adopt an equal mixture coefficient of 0.5 as the default BetaAgg composition across toy and downstream replacement paths, while exposing the coefficient so the pure softplus and sigmoid-gated endpoints can be evaluated separately.

- T024 D020 | experiment | retained
  Evaluate fixed-weight activation steering on a toy one-dimensional regression task by fitting a ReLU MLP to noisy samples from a scaled sinusoidal curve, independently replacing its ReLUs with BetaAgg, and plotting the ReLU and replacement predictions against the true curve. The configuration uses 30 samples, noise 0.3, width 64, depth 8, 20,000 training steps, seed 43, and compares beta values 0.9 and 0.5.

- T025 D021 | ablation | retained
  Ablate the BetaAgg mixture coefficient in the same- and cross-dataset post-replacement classification matrix by comparing the pure softplus-side setting coeff=0 and pure sigmoid-gated setting coeff=1 against the default mixture coeff=0.5.

- T026 D022 | experiment | retained
  Report post-replacement classification gains relative to the ReLU baseline using both per-condition normalized percentage improvement and absolute accuracy improvement, and summarize each with the unweighted mean across valid conditions. Apply the summaries separately to standard and robust accuracy, identifying robust conditions by dataset and attack, and exclude conditions whose baseline accuracy is zero from the per-condition maps and both overall means.

- T027 D035 | experiment | retained
  Make backbone depth an experimental factor by training and evaluating the steering method across ResNet-18, ResNet-34, ResNet-50, ResNet-101, and ResNet-152 instead of fixing ResNet-18.

- T028 D036 | experiment | superseded by D037
  Extend the backbone comparison beyond ResNets by adding VGG-19, including the input adaptation needed to train it on MNIST.

- T029 D023 | experiment | retained
  Evaluate ImageNet-to-CelebA transfer under the replace-then-probe protocol by independently replacing activations for each beta, fitting a frozen 40-output multi-label logistic-regression probe, and comparing against the ReLU endpoint under seeds 42, 43, and 44. Evaluate aggregate mean accuracy plus per-attribute accuracy, F1, balanced accuracy, and confusion counts, and analyze for each metric the attribute-specific best beta against the single beta that is best when averaged across attributes.

- T030 D024 | experiment | retained
  Evaluate ImageNet-to-dSprites transfer under the replace-then-probe protocol as orientation regression: independently replace activations for each beta, fit a frozen linear-regression head, and compare MSE against the ReLU endpoint under seeds 42, 43, and 44. Construct deterministic 70/30 dSprites train/test splits using random state 42 and sample 50,000 training and 10,000 test examples for the transfer evaluation.

- T031 D037 | experiment | retained
  Replace the VGG-19 comparison with Swin-T and then Swin-S transformer backbones, converting their GELU activations to ReLU before applying the same post-hoc activation steering procedure.

- T032 D025 | experiment | retained
  Report preferred activation settings only for dataset or robustness conditions where the replacement's accuracy exceeds the corresponding ReLU baseline. Compute each condition's beta mean and standard deviation across runs, then report the overall beta mean and standard deviation across the condition-level mean betas rather than flattening all runs.

## Demonstration idx=1354

Project: Fantastic Bugs and Where to Find Them in AI Benchmarks

Trajectory insight: The repository progresses from broad experimentation with psychometric, dependency, anomaly, IRT, reliability, and content-judge signals across many benchmarks toward a consolidated response-pattern pipeline built from tetrachoric correlation, mean pairwise Mokken Z, and item-total correlation, combined through Gaussianized ranks and voting. Along the way it adds synthetic evaluations and respondent-count, organization, recency, and size ablations, while judge protocols evolve from deterministic Gemini and GPT-4o variants to a retained grade-aware o1 GSM8K first pass. By the final revisions, IRT, isolation forest, AIR-Bench judging, multi-benchmark judge extensions, judge distillation, and several ensemble ablations are removed or abandoned, while the three-metric ensemble and simple variance and kappa baselines remain.

Decisions:

- T000 D002 | method | retained
  Score item validity from inter-item dependency patterns, comparing tetrachoric correlation, L1 distance correlation, adjusted mutual information, and Chatterjee xi with mean, median, or upper-quantile aggregation and explicit thresholds.

- T001 D007 | method | superseded by D017
  Use a deterministic Gemini judge given each question and official answer to classify incorrect, ambiguous, or biased items, and evaluate its binary predictions against platinum labels.

- T002 D005 | ablation | retained
  Ablate McDonald's omega latent-factor item scores across multiple factor structures, using general- and first-factor loadings as validity signals.

- T003 D001 | experiment | retained
  Evaluate invalid-item detectors on LLM-by-item binary response matrices, using platinum labels for GSM8K and MMLU High-School Mathematics and applying the pipeline across AIR-Bench, Thai Exam, MedQA, LegalBench, WikiFact, OpenBookQA, BoolQ, BBQ, medical MMLU subsets, and full HELM Lite MMLU.

- T004 D006 | method | retained
  Use classical reliability diagnostics as item-quality signals: corrected item-rest correlations and changes in Cronbach alpha or Guttman reliability when an item is deleted.

- T005 D004 | method | retained
  Use Mokken-style item scalability as an invalid-item signal, including item H and Z coefficients, monotonicity violations, and critical values; the Python pipeline ranks items by the mean pairwise Z coefficient.

- T006 D003 | ablation | abandoned
  Compare IRT-based invalid-item scores across 1PL, 2PL, and 3PL formulations and item-fit diagnostics; the Python 2PL route jointly estimated examinee ability, item difficulty, and item discrimination under the standard sigmoid(alpha × (theta - beta)) form with alternating parameter optimization.

- T007 D009 | method | retained
  Calibrate detector cutoffs from platinum-labeled items with one-split decision trees for selected scores, and use the learned GSM thresholds when inspecting datasets without platinum labels.

- T008 D014 | method | superseded by D015
  For AIR-Bench, use a deterministic Gemini judge that receives the prompt and its three-level risk category and flags category misalignment or ambiguity.

- T009 D008 | method | abandoned
  Treat each item's response vector as a point and use isolation-forest anomaly scores, especially adjusted depth and adjusted density, as invalid-item signals.

- T010 D010 | method | retained
  Normalize representative detector scores by within-metric Gaussian ranks and combine three final signals—tetrachoric correlation, item scalability, and item-total correlation—using a -0.5 rank threshold with majority, OR, and AND vote rules.

- T011 D012 | experiment | retained
  Use a controlled synthetic response benchmark in which known bad questions have ability-independent 25%-correct responses; the variant uses 300 test takers, 500 items, 20% bad items, and 10% random response flips.

- T012 D011 | experiment | retained
  Ablate the number of model respondents used to estimate item statistics by evaluating 10%-100% respondent fractions over ten independently shuffled seeds and reporting precision@50 for individual metrics and aggregate rankings on configured real or synthetic response data.

- T013 D013 | experiment | abandoned
  Construct a separate noisy 2PL simulation with discrimination parameters spanning -1 to 1, label negative-discrimination items as bad, and compare IRT discrimination against tetrachoric, scalability, reliability, and isolation scores.

- T014 D015 | method | superseded by D016
  Replace the first AIR-Bench judge with a GPT-4o judge that combines prompt, hierarchical risk category, and five example model responses and assigns ambiguity or grading-issue categories to invalid prompts.

- T015 D017 | method | superseded by D018
  Replace the question-and-answer-only Gemini judge with a GSM8K-specific GPT-4o first pass that receives each question, official answer key, and five sampled model responses, and classifies ambiguity, incorrect keys, or grading issues before expert review.

- T016 D016 | method | superseded by D019
  Use a GPT-4o AIR-Bench construct-validity judge based only on the prompt and three-level risk category, applying a permissive match to the most specific category and classifying invalid cases as ambiguous.

- T017 D018 | method | retained
  Use an o1-2024-12-17 GSM8K first pass, implemented through configurable provider batching, that samples 30 model responses per question, attaches each response's binary grade, and checks grading issues alongside ambiguity and incorrect answer keys.

- T018 D019 | method | abandoned
  Replace GPT-4o with o1-2024-12-17 for the AIR-Bench category-alignment judge while retaining prompt-only input with the three-level risk category and the permissive construct-validity criterion.

- T019 D020 | method | abandoned
  Extend the 30-response, grade-aware LLM validity judge beyond GSM8K using benchmark-specific prompts for MMLU High-School Mathematics, five-subject MMLU, OpenBookQA, MMLU Clinical Knowledge, MMLU Professional Medicine, MedQA, and ThaiExam.

- T020 D021 | method | abandoned
  Distill the o1 GSM validity judge into Gemma-3-27B using LoRA KTO, labeling generated judge outputs desirable when their binary predictions agree with o1, and evaluate the trained judge by exact prediction match.

- T021 D022 | method | retained
  Aggregate the detector metrics' Gaussianized within-metric ranks by their arithmetic mean (`gr_mean`) as an alternative continuous review ranking, using the three signals: tetrachoric correlation, item scalability, and item-total correlation.

- T022 D023 | experiment | retained
  Ablate respondent-source diversity by randomly selecting each possible number of model creator organizations across ten seeds and measuring detector precision@50.

- T023 D024 | experiment | retained
  Ablate respondent recency by successively restricting the response matrix to models released before each eligible release-date cutoff and measuring precision@50.

- T024 D025 | experiment | retained
  Ablate respondent model size by successively restricting the response matrix to models below each eligible parameter-count cutoff and measuring precision@50.

- T025 D026 | ablation | abandoned
  Ablate the Gaussian-rank ensemble's component set by averaging every combination of two or more detector metrics and comparing their invalid-item sensitivity on GSM8K.

- T026 D027 | ablation | abandoned
  Sweep each metric's Gaussian-rank decision threshold from -3 to 3.

- T027 D028 | experiment | retained
  Use prediction variance and Fleiss' kappa as simple baselines in sensitivity comparisons against the three primary invalid-item metrics, plotting these baselines for every configured dataset rather than only GSM8K.

## Demonstration idx=830

Project: Unifying Re-Identification, Attribute Inference, and Data Reconstruction Risks in Differential Privacy

Trajectory insight: The repository progresses from mechanism-agnostic calibration to operational advantage and FPR/FNR targets toward direct PLD-based DP-SGD calibration and evaluations on CIFAR-10, GPT-2, and UCI Adult. It abandons standalone Gaussian benchmark notebooks and successively replaces pointwise-minimum and grid/convex-hull trade-off constructions with direct piecewise-linear Neyman–Pearson evaluation, while replacing symmetric-error advantage reduction with direct epsilon-zero PLD advantage inversion.

Decisions:

- T000 D001 | method | retained
  Calibrate an Opacus-compatible DP mechanism directly to a target attack advantage by searching for the noise multiplier satisfying epsilon = 0 and delta = the advantage bound.

- T001 D002 | method | retained
  Calibrate noise to target attack FPR alpha and FNR beta by mapping candidate delta values to the required epsilon and selecting the delta/noise pair that minimizes noise; use bounded scalar minimization over delta as the sole optimization route, with bounded monotone inversion for the inner noise search.

- T002 D004 | experiment | retained
  Evaluate DP-SGD noise calibration for asymmetric attack errors by sweeping TPR from 0.05 to 0.5 and TNR over 0.9, 0.95, and 0.99, comparing fixed-delta calibration at delta 1e-5 under Connect-the-Dots PLD accounting with direct PLD FPR/FNR calibration.

- T003 D003 | experiment | retained
  Evaluate DP-SGD noise calibration over attack advantage values from 0.004 to 0.25 using five logarithmic points from 0.004 to 0.05 and five linear points from 0.05 to 0.25, comparing fixed-delta calibration at delta 1e-5 under Connect-the-Dots PLD accounting with direct PLD advantage calibration; use sample rate 0.001 and 10,000 steps.

- T004 D005 | experiment | abandoned
  Benchmark Gaussian-mechanism calibration across attack advantage values by comparing standard fixed-delta calibration, generic mechanism-agnostic advantage calibration, and the specialized exact Gaussian calibration.

- T005 D006 | experiment | abandoned
  Benchmark Gaussian-mechanism calibration for asymmetric FPR/FNR targets over TPR 0.1–0.5 and TNR 0.9, 0.95, and 0.99, comparing standard fixed-delta, generic mechanism-agnostic, and specialized exact Gaussian calibration; use exact noise 1/(Phi^-1(1-FPR) - Phi^-1(FNR)).

- T006 D007 | experiment | retained
  Evaluate recorded CIFAR-10 configurations with noise scales 4, 5, 6, 8, and 10 by relating test accuracy to attack TPR at FPR values 0.01, 0.05, and 0.1. Compose the normalization mechanism and every DP-SGD step across all epochs with PLDs, and compare exact symmetric attack-risk calibration against standard delta-1e-5 calibration under the same tight PLD accounting.

- T007 D008 | method | superseded by D014
  Measure the exact symmetric f-DP attack trade-off curve from a mechanism's privacy-loss distribution by computing FNR at a target FPR and the inverse FPR at a target FNR from the add/remove PLD PMFs, using 1e-4 discretization by default and the pointwise minimum of the two directions in evaluations; expose the calculation for DP-SGD and arbitrary supported PLDs.

- T008 D009 | method | retained
  Directly calibrate sampled-Gaussian DP-SGD noise to target attack FPR alpha and FNR beta by composing its privacy-loss distribution and inverting the bounded monotone noise-to-beta function, using a 1e-4 discretization grid by default and explicit beta and noise convergence tolerances.

- T009 D011 | experiment | retained
  Evaluate recorded GPT-2 configurations by relating test accuracy to attack TPR at FPR values 0.01, 0.05, and 0.1, comparing standard delta-1e-5 calibration under Connect-the-Dots PLD accounting with exact symmetric attack-risk calibration.

- T010 D010 | method | superseded by D016
  Directly calibrate sampled-Gaussian DP-SGD noise to a target attack advantage using the exact PLD trade-off, reducing advantage to symmetric error rates alpha = beta = (1 - advantage)/2 and invoking direct FPR/FNR calibration.

- T011 D013 | experiment | retained
  Evaluate Gaussian-noised UCI Adult education histograms by sweeping noise scales from 0.2 to 5 at FPR values 0.01, 0.05, and 0.1, using 100 repeated measurements per condition and comparing mean absolute histogram error versus attack TPR under standard delta-based and exact attack-risk calibration.

- T012 D012 | experiment | retained
  For the three GPT-2 configurations with test accuracy strictly between 0.55 and 0.705, compare the exact PLD-derived attack trade-off curve against the trade-off bound obtained from its delta-1e-5 epsilon/delta guarantee over 200 FPR values from 0 to 1.

- T013 D014 | method | superseded by D015
  Symmetrize add/remove PLD trade-off curves by evaluating the direct and inverse PLRV curves over a uniform alpha grid, taking their convex hull, and interpolating the resulting hull, using an alpha-grid step of 1e-4 by default.

- T014 D015 | method | retained
  Compute the symmetric f-DP trade-off curve directly from add/remove PLRV PMFs using piecewise-linear Neyman–Pearson evaluation, accounting for infinite support masses and explicitly symmetrizing nonsymmetric add/remove curves from their zero-loss breakpoint.

- T015 D016 | method | retained
  Directly calibrate sampled-Gaussian DP-SGD noise to a target attack advantage by composing its PLD, evaluating advantage as delta at epsilon zero, and inverting the bounded monotone noise-to-advantage function with explicit advantage and noise tolerances.

## Demonstration idx=1053

Project: Beyond Pairwise Connections: Extracting High-Order Functional Brain Network Structures under Global Constraints

Trajectory insight: The repository first established four-resolution network sharing, fixed-density straight-through adjacency learning, subject-contrastive classification, and individual-aware evaluation; it then briefly expanded datasets, split modes, visualization, and loss-balance ablations before abandoning or replacing those branches. Later revisions converged on adaptive-mass Gumbel-based adjacency hardening, restored the Cog State/SLIM/DynHCP domains and by-/across-individual splits, and added resolution-dependent adjacency normalization, fixed dataset subsampling, and five-run mean-and-standard-deviation reporting.

Decisions:

- T000 D002 | method | superseded by D008
  For trainable non-project networks, construct an undirected adjacency with a prescribed edge density by applying a symmetric Gumbel softmax, selecting hard top-k upper-triangular entries, symmetrizing them, and using a straight-through gradient.

- T001 D001 | method | retained
  Learn functional brain-network structures at four selectable resolutions by maintaining one network per sample, one per subject, one per class/group, or one shared across the entire project dataset.

- T002 D003 | method | retained
  Train the prediction model with task cross-entropy plus an optional subject-identity contrastive loss, weighted 0.5 relative to classification loss, that makes embeddings from homologous samples of the same person more alike, based on the stated assumption that an individual's brain state is stable.

- T003 D004 | experiment | retained
  Configure the evaluation across Cog State, SLIM, and DynHCP data, with DynHCP subtypes for Age, Gender, and Activity.

- T004 D005 | experiment | retained
  Include both by-individual and across-individual train/validation/test split protocols for the classification evaluation.

- T005 D006 | experiment | abandoned
  On DynHCP Gender, compare the four learned network resolutions visually by averaging male and female networks at sample and subject scales and averaging the other three learned scales for comparison with the project-scale network.

- T006 D007 | ablation | abandoned
  Ablate the coefficient beta on the contrastive loss relative to category cross-entropy to measure how loss balance affects prediction performance.

- T007 D009 | experiment | abandoned
  Expand the selectable evaluation domains to include BCI IV and TU graph datasets alongside Cog State, SLIM, and DynHCP, with subtype choices for 2a, 2b, ENZYMES, MUTAG, and PROTEINS_full.

- T008 D010 | experiment | superseded by D005
  Use random, by-individual, and leave-one-individual-out evaluation splits, with an explicit held-out individual index for leave-one-out evaluation.

- T009 D008 | method | superseded by D013
  Replace normalized fixed-density Gumbel-softmax adjacency selection with a symmetric Gumbel-sigmoid relaxation that derives a per-network k from summed soft edge probabilities and constructs a symmetrized straight-through hard adjacency.

- T010 D011 | experiment | superseded by D012
  Run each configured evaluation once rather than averaging accuracy and standard deviation over five repetitions.

- T011 D012 | experiment | retained
  Evaluate each configuration over five repetitions, report mean accuracy and standard deviation, and save that aggregate result.

- T012 D013 | method | retained
  Replace D008's adjacency relaxation with a gumbel_softkmax formulation that randomly initializes network scores, sigmoid-transforms selected scores, uses temperature 1, computes symmetric soft scores by applying sigmoid after averaging the Gumbel-perturbed matrix and its transpose, derives a per-network k from soft edge mass, and applies straight-through hardening.

- T013 D014 | method | retained
  Normalize the adjacency supplied to the graph encoder by the integer ratio of the number of samples to the number of learned networks at the selected resolution.

- T014 D015 | experiment | retained
  Randomly select 10 graph observations from each DynHCP dataset batch and 10 of the 100 indexed observations from each Cog State feature file when constructing the evaluation data.

## Demonstration idx=434

Project: The Third Pillar of Causal Analysis? A Measurement Perspective on Causal Representations

Trajectory insight: The ordered decisions move from a three-latent, two-view simulation and broad PCM/GCM diagnostics to a retained five-variable linear-Gaussian simulation with z0 as the sole shared latent across views. The active evaluation then centers on a PCM-based T-MEX mismatch score, alongside per-latent R² and proxy-adjusted causal-effect estimates, with representation quality varied through training-duration and private-latent-contamination ablations over 50 batches using view 1. Separate Multimodal3DIdent, PC causal-discovery, high-dimensional PCM/GCM, and B-MCC analyses were explored but later archived or removed.

Decisions:

- T000 D001 | experiment | superseded by D007
  Use a two-view numerical simulation in which view 0 observes latent indices [0,1] and view 1 observes [0,2], making z0 shared and z1 and z2 view-specific.

- T001 D002 | method | superseded by D006
  Evaluate learned shared representations with PCM and GCM conditional-independence diagnostics against both private latents jointly given the true shared latent z0, and include reverse tests that condition on each learned proxy instead of z0.

- T002 D003 | experiment | superseded by D008
  Generate dependent simulated latents from a three-variable linear-Gaussian structure encoded by B=[[0,0,0],[1,0,0],[1,1,0]], deriving the Gaussian covariance from the inverse of I-B.

- T003 D011 | experiment | retained
  Use per-latent squared Pearson correlation (R²) between a recovered shared representation and each of z0, z1, and z2 as a representation-quality baseline across the model conditions.

- T004 D004 | experiment | abandoned
  Evaluate learned Multimodal3DIdent text and image subset representations with COMETS conditional-independence tests against labeled factor blocks, using sampled validation data and linear, random-forest, or tuned-XGBoost regressions.

- T005 D005 | experiment | abandoned
  Test preservation of causal-discovery structure by running PC on true three-variable SEM samples and on a proxy representation that retains z0 while replacing z1 and z2 with their sum and difference.

- T006 D006 | method | retained
  Compute T-MEX per evaluation batch by applying PCM between a learned z0 proxy and each of z0, z1, and z2 while conditioning on the other four variables among z0, z1, z2, x, and y; at alpha 0.05, sum mismatch indicators relative to the expected pattern of dependence with z0 and conditional independence from z1 and z2, using nine PCM repetitions.

- T007 D007 | experiment | retained
  Replace the original measurement split with a five-latent, two-view simulation in which view 0 observes [z0,z1,z2] and view 1 observes [z0,x,y], leaving z0 as the sole shared latent.

- T008 D008 | experiment | retained
  Use a five-variable dependent linear-Gaussian simulation with coefficient matrix B=[[0,0,0,0,0],[1,0,0,0,1],[1,1,0,0,0],[1,0,0,0,0],[1,0,0,1,0]].

- T009 D009 | experiment | retained
  Ablate representation training duration by comparing Model A trained for 50,001 steps with Model B trained for 51 steps, evaluating 50 batches and.

- T010 D013 | experiment | abandoned
  Stress-test PCM and GCM in a high-dimensional synthetic setting over 30 replications with n=3000, a 10-dimensional conditioning variable, a 700-dimensional X, and random-forest regressions, evaluating both X versus Y given Z and X versus Z given Y configurations.

- T011 D014 | experiment | abandoned
  Use a Spearman block mean correlation coefficient (B-MCC) between target latent blocks and learned representation blocks as an additional representation-quality baseline across batches and representation conditions.

- T012 D012 | experiment | retained
  Evaluate downstream causal-effect estimation by fitting a linear regression of y on a recovered z0 proxy and x, taking the coefficient of x as the effect estimate for each batch and representation condition.

- T013 D010 | experiment | retained
  Add a synthetic proxy-contamination ablation, Model C, by taking Model A's recovered shared representations and adding 0.2*z1-0.1*z2 to both view-specific z0 proxies; evaluate the same 50 batches using view 1.

## Demonstration idx=305

Project: Differentially Private Quantiles with Smaller Error

Trajectory insight: The repository advances from a synthetic pilot and a fixed-window sequential sliced-quantile prototype to a retained binary-recursive mechanism. Its core ingredients become an automatically tuned k-ary continual-counting tree, clipped exponential-mechanism slices, and an endpoint-aware privacy admissibility condition derived from a simultaneous tree-noise bound. Synthetic distribution, slice-width, counting-backend, and domain-to-gap studies are explored and later removed, while the Kaplan comparison and repeated Adult age and working-hours evaluation remain as the retained empirical protocol.

Decisions:

- T000 D002 | experiment | superseded by D006
  Run an early pilot comparison on synthetic Gaussian or Gaussian-mixture data using equally spaced quantiles and normalized maximum and mean rank error, while varying the number of quantiles and privacy settings.

- T001 D003 | method | superseded by D009
  Estimate multiple quantiles by adding correlated continual-counting noise to target ranks, taking bounded slices around the noisy ranks, and applying exponential-mechanism quantile estimation while enforcing ordered outputs.

- T002 D004 | ablation | abandoned
  Ablate the exponential-mechanism slice radius or its logarithmic scale factor over broad ranges and multiple datasets or target quantiles, measuring repeated rank error and comparing it with a computed upper bound.

- T003 D005 | method | retained
  Use a k-ary continual-counting tree with two-sided geometric node noise, automatically select its branching factor by minimizing a worst-case variance bound, and characterize its simultaneous high-probability error with a Chernoff/union bound.

- T004 D001 | experiment | retained
  Use the Kaplan et al. Recursive differentially private approximate-quantiles algorithm as the principal comparison baseline.

- T005 D006 | experiment | abandoned
  Benchmark SliceQuantile against pure-DP and privacy-matched zCDP Kaplan baselines across Gaussian, uniform, Gaussian-mixture, Beta, and mixture-of-Beta data while sweeping the number of quantiles and reporting repeated rank error and runtime. Trial a Gaussian histogram baseline within this synthetic program.

- T006 D007 | ablation | abandoned
  Compare k-ary-tree and matrix-factorization continual-counting implementations inside the same sliced quantile mechanism, measuring maximum rank error and the frequency of random fallback outputs as the number of quantiles varies.

- T007 D008 | experiment | abandoned
  Evaluate sensitivity to the domain-to-gap ratio on synthetic Gaussian-mixture and constructed consecutive-integer mixture data by varying domain bounds, comparing SliceQuantile with pure-DP or zCDP Kaplan baselines under matched privacy accounting, testing assumed versus observed minimum gaps where applicable, and reporting rank error and runtime.

- T008 D009 | method | retained
  Use a binary recursive inference procedure for approximate sliced quantiles: perturb all target ranks with a k-ary continual-counting tree, form clipped slices around them, estimate the middle slice with the exponential mechanism, and recursively restrict the value bounds for left and right slices.

- T009 D011 | method | retained
  Expose an approximate-DP sliced quantile mechanism only when the k-ary continual-counting tree's simultaneous high-probability error bound is smaller than half the minimum spacing between consecutive target ranks, including the dataset endpoints, minus the slice radius.

- T010 D016 | experiment | retained
  Evaluate SliceQuantile on the Adult age and working-hours attributes against pure-DP and zCDP Kaplan baselines, replicating and de-tying records, sweeping 10–200 randomly selected target quantiles, running 200 repetitions, testing both bounded and unbounded adjacency for SliceQuantile while retaining the original baseline's swap=False inference, and reporting maximum rank error.

## Demonstration idx=472

Project: DreamPRM: Domain-Reweighted Process Reward Model for Multimodal Reasoning

Trajectory insight: The repository progresses from a small ScienceQA/MathVista setup with two-shot prompting, an unbounded reward head, stepwise best-of-four decoding, and a ReST baseline to a broader benchmark suite and a fixed five-step reasoning pipeline. The retained design uses Monte Carlo prefix targets, a capped multi-domain MMPR mixture, a sigmoid-bounded Qwen2-VL-2B reward model, MMMU-based meta data, and learned domain weights through bi-level optimization; inference correspondingly shifts to best-of-eight selection of complete InternVL-MPO trajectories using mean prefix log-odds, while ReST and the candidate-count ablation are removed.

Decisions:

- T000 D001 | experiment | retained
  Use a multimodal reasoning evaluation suite that began with ScienceQA and MathVista and expanded to repository runners/loaders for M3CoT, MMMU-Pro, MMStar, MM-Vet, MathVision, and WeMath, with dataset-specific prompt and answer handling.

- T001 D002 | method | superseded by D008
  Train the initial multimodal reward model as an unbounded scalar linear head over a vision-language model's final-token vocabulary logits, using mean-squared error against correctness or rollout-derived targets.

- T002 D003 | method | superseded by D009
  Elicit initial reasoning traces with a two-shot Phi-3.5 prompt that prepends worked examples and requests step-by-step answers.

- T003 D004 | method | retained
  Estimate each reasoning prefix's process target by repeatedly sampling continuations and setting its reward to the fraction judged correct.

- T004 D005 | method | superseded by D011
  Perform stepwise best-of-four decoding by sampling four candidate next-step continuations, scoring them with the PRM, appending the highest-scoring candidate, and repeating until an answer is complete.

- T005 D006 | experiment | abandoned
  Evaluate a three-round ReST-style refinement baseline that repeatedly generates a response and retains its highest-PRM-scoring reasoning prefix for the next round.

- T006 D007 | method | retained
  Construct an MMPR multi-domain PRM-training mixture by capping each listed source dataset at 1,000 prefixes, prioritizing examples with fractional Monte Carlo accuracy, then filling remaining capacity with paired and finally unpaired 0/1 examples.

- T007 D008 | method | retained
  Replace the raw reward with a sigmoid-bounded scalar probability produced from the final-token vocabulary logits, implemented for the final Qwen2-VL-2B PRM and trained with mean-squared error on Monte Carlo process targets.

- T008 D010 | method | retained
  Build the upper-level meta dataset from MMMU sampled responses by including contrasting correct/incorrect candidates and balanced always-correct and always-incorrect cases, then evaluate each example through five reasoning prefixes.

- T009 D012 | ablation | abandoned
  Run a candidate-count scaling ablation from one through four selected reasoning samples drawn from an eight-stream candidate pool.

- T010 D014 | method | retained
  Learn positive, mean-normalized weights for the MMPR source domains through bi-level optimization: the lower-level Qwen PRM minimizes domain-weighted MSE, while an upper-level meta objective updates the domain weights through differentiable inner optimization.

- T011 D009 | method | retained
  Generate reasoning traces with a fixed five-step structural prompt: restate the question, gather image evidence, identify background knowledge, reason with the evidence, and summarize before a formatted final answer.

- T012 D011 | method | retained
  Perform trajectory-level best-of-eight selection over pre-generated InternVL-MPO reasoning samples by scoring every available prefix with the PRM and choosing the complete response with the highest mean aggregated score; the final candidates are sampled at temperature 1.0.

- T013 D013 | method | retained
  Use Qwen2.5-7B-Instruct as a semantic correctness judge for free-form candidate answers and Monte Carlo continuation labels, with exact-match fallback when the judge returns incorrect.

- T014 D015 | experiment | retained
  Measure oracle best-of-N accuracy by marking a benchmark item correct whenever any candidate response among the configured output files is correct.

- T015 D016 | method | retained
  Aggregate five per-prefix PRM probabilities by converting them to log-odds and averaging before a sigmoid for the upper-level meta loss, while using the corresponding mean log-odds aggregation to rank complete BoN trajectories.

## Demonstration idx=894

Project: Exploiting Vocabulary Frequency Imbalance in Language Model Pre-training

Trajectory insight: The repository develops a four-vocabulary comparison (24K, 49K, 98K, and 196K) while repeatedly refining its controls: the PROPOSED method moves from forward-pass output-head normalization to a soft norm regularizer and then to retained post-optimizer unit-L2 renormalization of both input and output embeddings, while the canonical model shifts from a 24-layer configuration to a retained 12-layer untied configuration. The experiment suite then broadens through z-loss, model-scale, learning-rate, and embedding-tying ablations, but later narrows by retaining the 450M scale and learning-rate sweep while discontinuing the 1.3B and four-vocabulary tied-embedding suites.

Decisions:

- T000 D003 | ablation | retained
  Provide a z-loss ablation for pretraining by selecting LOSS_TYPE=ZLOSS and using fused linear cross entropy with an LSE-square scale of 1.0e-4.

- T001 D002 | experiment | superseded by D004
  For the separately selectable PROPOSED training path, constrain every output-embedding row to unit L2 norm by normalizing the language-model head weights before each forward pass.

- T002 D001 | experiment | retained
  Run vocabulary-size experiments at 24K, 49K, 98K, and 196K using the corresponding FineWeb-Edu tokenizer and tokenized dataset.

- T003 D010 | ablation | abandoned
  Run an embedding-tying ablation for the 12-layer Llama configuration by setting tie_word_embeddings to true and training tied variants at 24K, 49K, 98K, and 196K with the PROPOSED path and learning rate 6.0e-4.

- T004 D004 | experiment | superseded by D006
  Replace hard output-embedding normalization with a differentiable regularizer that penalizes the standard deviation of row norms and squared deviation of their mean from target norm 1.0, using default weights of 1e-2 for both terms.

- T005 D005 | experiment | superseded by D007
  Run the vocabulary-size experiments with a larger Llama configuration—24 layers, 24 attention heads, hidden size 1536, and intermediate size 4096—documented as eight times the non-embedding size of the 85M experiment configuration.

- T006 D006 | experiment | retained
  Replace the PROPOSED path's soft norm-statistics regularizer with hard post-optimizer unit-L2 renormalization: after every training step, independently normalize every row of both the input token-embedding matrix and the output LM-head matrix, and apply this method across the 24K, 49K, 98K, and 196K vocabulary runs.

- T007 D007 | experiment | retained
  Replace the canonical 24-layer large-model configuration with the smaller untied Llama configuration having 12 layers, 12 attention heads, hidden size 768, and intermediate size 2048 for the experiments.

- T008 D008 | experiment | retained
  Extend the four-vocabulary experiment to larger model scales by creating complete 450M and 1.3B Llama suites, using the PROPOSED embedding-renormalization path for 450M and PRETRAIN for 1.3B; revise their learning rates to 3.0e-4 and 2.0e-4 respectively.

- T009 D009 | ablation | retained
  Ablate the learning rate for the PROPOSED embedding-renormalization experiments over 7.5e-5, 1.5e-4, 1.2e-3, and 2.4e-3 at each of the 24K, 49K, 98K, and 196K vocabulary sizes, using per-device batch size 16 and two gradient-accumulation steps.

## Demonstration idx=958

Project: Global Minimizers of Sigmoid Contrastive Loss

Trajectory insight: The trajectory starts with a strong sign-based geometry hypothesis—positive matching similarities and negative non-matching similarities—but abandons that criterion in favor of the margin and an explicit relative-bias parameter. Synthetic experiments then separate the roles of bias, temperature, initialization, frozen modalities, adapters, and modality count; repeated runs later test that the bias effect is not an initialization artifact. In parallel, the real-data protocol is revised twice: an exploratory B/16-384 study becomes a full-validation base-patch16-224 test, and a broad 17-checkpoint SigLIP/SigLIP2 distance comparison becomes a focused eight-checkpoint analysis linking stored loss parameters to empirical margin and relative bias. The final xi experiment follows the geometry during training rather than only at convergence. Together, the evolution replaces an absolute sign claim with a parameter-aware account of how contrastive loss separates modalities.

Decisions:

- T000 D003 | experiment | retained
  Ablate fixed relative bias from -1 to 1 while training temperature in synthetic SigLIP, and measure loss and the gap between the worst matching and best non-matching similarities across both dense and five-point bias sweeps.

- T001 D006 | ablation | retained
  Compare fixed temperature 100 with a trainable temperature initialized at 10 across five fixed relative biases in two-modality synthetic SigLIP, measuring loss, learned scale, and matching/non-matching geometry.

- T002 D007 | experiment | retained
  Extend synthetic SigLIP from two to three and four modalities and then sweep 4, 6, 8, and 10 modalities, measuring loss and pooled matching/non-matching similarities across every modality pair.

- T003 D001 | experiment | superseded by D011
  Evaluate modality separation in pretrained SigLIP B/16-384 on ImageNet validation images and class-label text using matching/mismatching similarities and a perceptron probe.

- T004 D005 | ablation | retained
  Ablate frozen-modality SigLIP with and without a learned scalar adapter that embeds the two modalities into complementary added coordinates, comparing convergence and geometry before and after removing the adapter coordinate.

- T005 D002 | method | retained
  Parameterize SigLIP logits with an explicit relative bias multiplied by temperature, with independent controls over temperature and bias training, while retaining the conventional absolute-bias form as a comparison.

- T006 D004 | experiment | abandoned
  Test the geometric hypothesis that synthetic SigLIP optimization produces positive inner products for every matching pair and negative inner products for every non-matching pair.

- T007 D008 | experiment | retained
  Visualize the synthetic geometry directly with 20 classes in three dimensions, computing the matching/non-matching margin and plotting the resulting constellation after training with relative bias zero and trainable temperature.

- T008 D016 | ablation | retained
  Test whether the temperature conclusion generalizes when one modality is frozen and when four modalities are trained, comparing fixed temperatures 10 and 200 with trainable-temperature configurations under conventional and relative-bias parameterizations.

- T009 D010 | experiment | retained
  Ablate bias initialization and parameterization by matching five initial absolute biases under conventional and explicit relative-bias losses, then compare learned temperature, effective bias, loss, and final representation margin.

- T010 D017 | ablation | abandoned
  Explore a ten-point fixed-temperature sweep from 10 to 100 at relative bias 1.

- T011 D011 | experiment | retained
  Replace the initial pretrained study with a full ImageNet-validation evaluation of SigLIP base-patch16-224, comparing image-to-class-text similarities and probing whether image and text embeddings remain linearly separable by modality.

- T012 D012 | experiment | superseded by D013
  Compare representation geometry across 17 SigLIP and SigLIP2 checkpoints using paired distance, mean displacement, random-pair distance, and matching/mismatching similarity statistics; D013 soon replaces this protocol.

- T013 D013 | experiment | retained
  Replace the 17-checkpoint distance study with an eight-checkpoint SigLIP comparison that relates stored bias and inverse temperature to empirical matching margin and the midpoint-derived relative bias.

- T014 D015 | experiment | retained
  Test the robustness and training dynamics of the fixed-relative-bias result with 100 seeded runs at each of ten biases, tracking the mean and variability of margin and the empirical optimal relative bias throughout optimization.

- T015 D014 | experiment | retained
  Track a synthetic modality-displacement statistic xi throughout 20 SigLIP training runs, measuring matching-pair distance after subtracting the norm of their mean signed displacement.

## Demonstration idx=863

Project: STAR: A Benchmark for Astronomical Star Fields Super-Resolution

Trajectory insight: The repository trajectory moves from fixed on-the-fly PSF synthesis through offline patch degradation to full-field, WCS-aware, flux-corrected generation, ultimately retaining a sequential Gaussian–Poisson–Airy pipeline. In parallel, evaluation shifts from normalized aggregate TFE to source-matched mean absolute flux differences alongside masked PSNR and SSIM, while the benchmark expands across U-Net, SwinIR, Restormer, PromptIR, EDSR, and RCAN and applies a source-focused auxiliary reconstruction loss to multiple models.

Decisions:

- T000 D011 | method | retained
  Augment masked L1 reconstruction training with a source-focused auxiliary loss: detect and perform elliptical photometry on HR patches, form a flux-weighted Gaussian attention map from source position, shape, orientation, and absolute flux, and add attention-weighted absolute reconstruction error, generally with coefficient 0.01, to the image loss.

- T001 D001 | method | retained
  Construct the HST DRC dataset by selecting products with NCOMBINE equal to 4, spatially de-duplicating and partitioning observations using WCS/RA boundaries (train below RA 250 and evaluation above RA 255), checking train/test polygons at a 3-arcsec threshold, and extracting spatially aligned 256×256 HR and 128×128 LR half-stride pairs with more than 80% valid pixels after continuous zero regions are treated as missing.

- T002 D002 | method | superseded by D004
  Create each model input on the fly from a standardized target by convolving it with a fixed normalized PSF formed as the product of Airy and Gaussian atmospheric-turbulence components, with resolution reduction left inactive.

- T003 D003 | method | superseded by D012
  Evaluate astronomical reconstruction with a normalized flux-error measure computed after background subtraction and star detection, comparing detected-region or aperture flux between the prediction and ground truth and reporting it as TFE alongside conventional image metrics where enabled.

- T004 D004 | method | superseded by D008
  Generate paired 2× low-resolution inputs offline by randomly choosing a normalized isotropic-Gaussian or Airy PSF, sampling the 2× PSF parameter from [0.2, 2], applying order-3 downsampling, adding Gaussian noise with a sampled level from [0, 5]/255, and clipping values to be nonnegative.

- T005 D005 | experiment | retained
  Train and evaluate a single-channel 2× SwinIR baseline on 128×128 LR inputs using four depth-6 transformer stages, 90-dimensional embeddings, six heads per stage, window size 8, pixel-shuffle upsampling, batch size 16, and Adam at learning rate 0.0002 for 100 epochs with linear warm-up and cosine decay, using the revised paired dataset.

- T006 D012 | method | retained
  Evaluate the full validation set with PSNR and SSIM over valid masked pixels and with an unnormalized per-source flux-consistency error: subtract backgrounds, detect elliptical sources on the ground truth with SEP, measure prediction flux at the same source ellipses, and average the absolute source-flux differences.

- T007 D006 | experiment | retained
  Include a U-Net reconstruction model as a completed baseline comparison on the astronomical super-resolution benchmark.

- T008 D007 | experiment | retained
  Train and evaluate a single-channel 2× Restormer baseline with dimension 48, encoder/decoder block counts [4, 6, 6, 8], heads [1, 2, 4, 8], four refinement blocks, and Adam at learning rate 0.0002 for 100 epochs with linear warm-up and cosine decay.

- T009 D008 | method | superseded by D010
  Generate 2× LR observations at full-field scale by padding HR FITS fields, applying a normalized randomly selected Gaussian or Airy PSF, and performing WCS-aware exact reprojection before extracting corresponding HR/LR patches from scaled shared coordinates.

- T010 D009 | experiment | retained
  Train and evaluate a single-channel 2× PromptIR baseline with dimension 48, block counts [4, 6, 6, 8], heads [1, 2, 4, 8], four refinement blocks, enabled prompt decoder, bias-free convolutions, and Adam at learning rate 0.0002 for 100 epochs on the revised paired dataset; its supplied objective combines masked L1 with the source-focused auxiliary flux loss at coefficient 0.01.

- T011 D010 | method | superseded by D015
  Generate flux-corrected 2× paired fields by excluding HR values outside mean ±10 standard deviations, padding fields to a multiple of 256, convolving with a normalized Gaussian or Airy PSF selected from configurable ranges (Gaussian sigma 0.8–1.2 or Airy radius 1.5–1.9 by default), applying WCS-aware exact reprojection, and multiplying the LR image by the squared scale factor to preserve integrated flux.

- T012 D013 | experiment | retained
  Train and evaluate a single-channel 2× EDSR baseline with 32 residual blocks, 256 features, residual scale 0.1, and Adam at learning rate 0.0002 for 100 epochs on the Gaussian/Airy paired dataset.

- T013 D014 | experiment | retained
  Train and evaluate a single-channel 2× RCAN baseline with 64 features, 10 residual groups, 20 residual channel-attention blocks per group, reduction factor 16, and Adam at learning rate 0.0002 for 100 epochs on the Gaussian/Airy paired dataset.

- T014 D015 | method | retained
  Generate flux-corrected 2× paired fields with a sequential degradation pipeline: apply a normalized Gaussian PSF sampled from sigma 0.8–1.2 and normalized Poisson noise with scale 1000 to the padded HR field, perform the existing WCS-aware downsampling, and then apply an Airy PSF sampled from radius 1.9–2.2 in the LR domain.

## Demonstration idx=878

Project: Beyond the Surface: Enhancing LLM-as-a-Judge Alignment with Human via Internal Representations

Trajectory insight: The repository progresses from extracting per-layer score-token distributions and expected scores to broad point-wise benchmarking and prompting/aggregation ablations, while discontinuing score-range and pairwise-transfer experiments but retaining emotion and unanswerability evaluations. Its aggregation method is repeatedly refined: final-layer-biased learned weighting with cross-entropy is replaced by a learned cross-entropy/MSE mixture, then by uniformly initialized weighting over non-final stored layers only.

Decisions:

- T000 D001 | method | retained
  At the generated score-token position, project every transformer layer's hidden state through the language-model head, restrict to allowed numeric score tokens, softmax those logits, and compute both the most likely score and the probability-weighted expected score for each layer.

- T001 D004 | ablation | abandoned
  Ablate output-score granularity by generating rubric-free Flask prompts whose maximum allowed score is swept over 5, 9, 19, 29, 39, 49, 59, 69, 79, 89, and 99.

- T002 D002 | experiment | retained
  Evaluate point-wise judge scores against human ratings on Flask, HelpSteer, and BIGGen, with model-specific result/validation handling including Qwen alongside the previously supported model families.

- T003 D003 | ablation | retained
  Ablate evaluator prompting by comparing score-only prompts with prompts that require written feedback before the numeric score.

- T004 D005 | method | superseded by D010
  Aggregate score-token logits across layers using softmax-normalized layer weights initialized to favor the final layer and learned from same-model HelpSteer validation human-score labels with cross-entropy over shuffled mini-batches, then softmax the combined logits and take their numeric expectation.

- T005 D006 | ablation | retained
  Ablate score construction against human ratings with Pearson and Spearman correlations by comparing final-layer direct and expected scores, layerwise score averages, and tuned or uniform cross-layer aggregation.

- T006 D007 | experiment | abandoned
  Evaluate the point-wise scoring variants as pairwise preference judges by scoring chosen and rejected responses separately on HelpSteer preference and RewardBench data, treating ties as 0.5 and reporting accuracy or RewardBench section aggregation.

- T007 D008 | experiment | retained
  Evaluate the scoring variants on 1–9 character-emotion intensity prediction from dialogue using emotional_data processed from EQ-Bench, comparing direct, final-layer expected, tuned learned-weight cross-layer, and untuned uniform cross-layer scores with human emotion ratings using Pearson and Spearman correlations.

- T008 D009 | experiment | retained
  Evaluate direct, final-layer expected, tuned learned-weight cross-layer, and untuned uniform cross-layer scores as detectors of unanswerable questions on the answerability-labeled SelfAware dataset: classify scores at a threshold 75% through each score type's observed minimum-to-maximum range and report F1.

- T009 D010 | method | superseded by D011
  Aggregate all stored layers' score-token logits with softmax-normalized learned layer weights initialized to favor the final layer, and jointly train those weights and a sigmoid mixing coefficient on validation human scores using a learned mixture of cross-entropy over score classes and MSE on the probability-weighted expected score.

- T010 D011 | method | retained
  Aggregate only the non-final stored layers' score-token logits with uniformly initialized softmax-normalized learned layer weights, and jointly train those weights and a sigmoid mixing coefficient on validation human scores using a learned mixture of cross-entropy over score classes and MSE on the probability-weighted expected score.

## Demonstration idx=325

Project: CLEVER: A Curated Benchmark for Formally Verified Code Generation

Trajectory insight: The ordered decisions show a progression from assembling HumanEval-derived Lean tasks and experimenting with machine-value projections and `Option`-based failure semantics toward ideal Lean types, implementation-independent relational specifications, explicit preconditions, and stronger anti-vacuity constraints. The repository then separates specification, implementation, proof, and isomorphism evaluation into controlled task views, tightens task identity and `sorry`-free validation, and curates benchmark and few-shot membership by excluding unsupported or duplicative items.

Decisions:

- T000 D001 | method | retained
  Formulate verified-programming benchmark tasks in Lean by pairing candidate implementations with logical specifications and proof obligations, and bind the existential result witness to the candidate output with conjunction (`∃ result, impl ... = result ∧ spec result`) so that the returned result itself must satisfy the specification.

- T001 D002 | experiment | abandoned
  Evaluate each configured Lean task on an all-or-nothing point basis, awarding its assigned score only when the build log contains no `declaration uses 'sorry'` diagnostic for that task.

- T002 D003 | method | retained
  Use the indexed HumanEval problem corpus—retaining Python task descriptions and reference implementations alongside the earlier Rust material—as source material for tasks translated into Lean specifications.

- T003 D004 | method | superseded by D006
  Represent machine-level failure or out-of-domain behavior explicitly with `Option` outputs and preconditions, requiring `some` results to satisfy the specification and allowing `none` only when the corresponding precondition fails.

- T004 D005 | method | superseded by D006
  State numerical specifications over mathematical projections of machine values—mapping finite `Float` values to rationals and fixed-width integers to Lean integers—rather than performing the logical specification directly in machine arithmetic.

- T005 D006 | method | retained
  Specify HumanEval functional-correctness tasks directly over ideal, computable Lean types such as `Nat`, `Int`, `Rat`, `List`, and `String`, abstracting away language-specific machine representations, overflow, and underflow.

- T006 D007 | method | retained
  Prefer Lean specifications that state implementation-independent semantic or relational properties—including non-computable logical predicates—instead of directly reproducing a particular computable implementation, so the specification does not reveal solution logic except where no reasonable alternative is available.

- T007 D008 | method | retained
  Express intended input-domain constraints as antecedent preconditions in ideal-type correctness specifications, leaving behavior outside that domain unconstrained rather than making the constraint part of the returned functional proposition.

- T008 D009 | method | superseded by D025
  For source tasks with heterogeneous `Any`-typed signatures, first narrow the signature to supported Lean types when possible; if that is not possible, leave the task without a concrete Lean specification rather than inventing an unsupported encoding.

- T009 D010 | method | retained
  Use explicit acceptable-error bounds for numerical tasks that require approximate answers: accept polynomial roots with rational evaluation error at most `1 / 1000000`, and triangle areas whose squared result differs from Heron's expression by at most `1 / 10000`.

- T010 D011 | method | retained
  Evaluate functional correctness without imposing implementation-resource constraints that the formal task cannot currently enforce: do not expose implementation fuel or require termination within a fuel bound, and do not enforce source-level efficiency requirements.

- T011 D012 | method | retained
  Retain and write a formal task specification even when a provably terminating implementation is difficult or unavailable; a completed implementation and termination proof are not prerequisites for specifying such a task.

- T012 D013 | method | retained
  Include built, worked Lean sample problems containing task metadata, formal specifications, implementations, tests, invariants, specification-isomorphism material, and correctness-proof material as example artifacts.

- T013 D018 | method | retained
  Require task-level specifications to rule out vacuous solutions by constraining negative optional or Boolean outputs to a corresponding no-solution condition and by making essential output properties conjunctive rather than antecedents an implementation can falsify.

- T014 D016 | experiment | retained
  Evaluate Rust implementation generation followed by translation into Lean 4 with Aeneas, requiring the translated result to be a valid Lean program and to have a Lean proof that it satisfies the correctness specification.

- T015 D014 | method | retained
  Evaluate formal-specification generation from natural-language problem descriptions by requiring a Lean certificate that the generated specification is isomorphic to the human-written ground-truth specification.

- T016 D017 | method | retained
  Evaluate generation of Lean correctness proofs when the formal specification and Lean implementation are supplied, withholding the target correctness proof and correctness helper lemmas, and accepting a generated proof only when it compiles in Lean 4.

- T017 D015 | method | retained
  Evaluate direct Lean 4 implementation generation conditioned on a correctness specification, requiring a Lean proof that the generated implementation satisfies that specification.

- T018 D019 | method | retained
  Construct each task with only the shared Lean helper-definition blocks whose metadata names that benchmark problem or sample problem, rather than supplying the entire shared helper library.

- T019 D020 | method | retained
  Add a separate specification-isomorphism task component (`SPEC_ISOMORPHISM`, Task 4) whose task view withholds the existing isomorphism proof and all implementation, test, and correctness artifacts.

- T020 D021 | method | retained
  For task components whose generated view retains Lean test cases, convert the stored commented test-case lines into active Lean code before presenting and validating the task view.

- T021 D022 | method | retained
  Identify and retrieve benchmark tasks by their declared HumanEval problem ID, including when the loaded corpus has gaps, rather than treating a task's position in the loaded list as its identity.

- T022 D023 | method | retained
  For HumanEval problem 156, replace the inductive `roman_value_non_computable` oracle with executable Roman-syntax validation and decimal-conversion helpers, accepting outputs through a validity-and-decoding condition for inputs from 1 through 1000.

- T023 D024 | experiment | retained
  Validate proof-bearing Lean problem-view submissions on a file-wide, all-or-nothing no-`sorry` basis: any detected `sorry` makes both isomorphism and correctness unsuccessful, while a no-`sorry` view reports a stage successful only when its corresponding proof artifact is present.

- T024 D025 | method | retained
  Exclude HumanEval tasks whose heterogeneous `Any` signatures cannot be represented in Lean, rather than retaining placeholder benchmark files without concrete specifications or implementation signatures.

- T025 D026 | experiment | retained
  Exclude sample problem 3 from few-shot prompts because it duplicates HumanEval problem 55, while retaining the worked sample artifact outside the prompt set.

## Demonstration idx=1037

Project: Human Texts Are Outliers: Detecting LLM-generated Texts via Out-of-distribution Detection

Trajectory insight: The repository progresses from a machine-centered DeepSVDD detector and objective ablations to a broader suite of generator-family HRN and multiclass energy detectors. It then standardizes score-based evaluation across DeepFake, multilingual M4, and RAID, while later changes retain fixed machine-derived references, family-head aggregation, a clipped softplus DeepSVDD loss, released checkpoint-based configurations, and dataset-specific Gaussian conversion of scores to probabilities.

Decisions:

- T000 D001 | method | retained
  Use a DeepSVDD hypersphere over learned text embeddings for anomaly detection, with squared distance to a center and support for one-class or soft-boundary objectives.

- T001 D002 | experiment | retained
  Evaluate continuous detector scores with ROC-AUC, including DeepSVDD distances, KNN confidence, and HRN outputs.

- T002 D003 | method | retained
  Estimate the DeepSVDD hypersphere center using only machine-generated training texts.

- T003 D004 | method | superseded by D008
  Ablate the sample population used by the standard DeepSVDD compactness loss between machine-only and all training samples, returning to machine-only before this objective was replaced.

- T004 D005 | method | retained
  Ablate one-class and soft-boundary DeepSVDD objectives, updating the soft-boundary radius after five epochs and leaving the active configuration on the one-class objective.

- T005 D006 | ablation | retained
  Initialize OOD training from detector checkpoints: this choice was first used for DeepSVDD on DeepFake.

- T006 D007 | ablation | retained
  Ablate combining DeepSVDD with only the label-level contrastive loss versus the full multi-level contrastive objective, leaving the later active DeepSVDD configuration on the label-level-only branch.

- T007 D008 | method | superseded by D021
  Replace the standard DeepSVDD compactness loss with a raw supervised distance-difference objective equal to average machine distance from the center minus average human distance.

- T008 D009 | method | retained
  Use an HRN-style one-class scalar head trained on machine-text embeddings with a gradient penalty, producing sigmoid membership scores.

- T009 D010 | ablation | retained
  Use a 12th-power deviation from unit gradient norm for the HRN gradient penalty instead of the quadratic penalty.

- T010 D011 | method | retained
  Train a separate one-class HRN head for each configured nonhuman generator family across Deepfake, M4, and RAID, maintaining the family-specific heads together in a shared multi-head model.

- T011 D012 | method | retained
  Combine all configured generator-family HRN heads by averaging their sigmoid membership scores for overall machine-versus-human detection.

- T012 D013 | experiment | retained
  Evaluate each generator-family HRN head using one-versus-rest generator-family membership labels rather than the generic machine-versus-human label.

- T013 D014 | ablation | retained
  Freeze the complete pretrained text encoder while training the generator-family HRN heads.

- T014 D015 | method | retained
  Use a multiclass energy-based OOD detector whose head classifies machine-generator families, trains ID and OOD energies with a weighted squared-hinge margin penalty, and uses negative energy as the detection score.

- T015 D016 | experiment | abandoned
  Evaluate DeepSVDD on TuringBench using its training split, machine-only center estimation, and its test split for evaluation.

- T016 D017 | experiment | retained
  Evaluate the OOD detectors on the M4 multilingual benchmark using multilingual train plus development data for training and multilingual test data for evaluation, including DeepSVDD, Energy, and generator-family HRN variants.

- T017 D018 | experiment | retained
  For KNN, DeepSVDD, Energy, and HRN, augment continuous ROC-AUC evaluation with PR-AUC, TPR at 5% FPR, and FPR at 95% TPR; for binary summaries, choose a split-specific score threshold maximizing F1 and report accuracy, precision, recall, and F1.

- T018 D019 | experiment | retained
  Evaluate the repository's detectors on RAID using a derived Shengkun/Raid_split: split liamdugan/raid's training data 90/10, restrict the training side to attack-free examples and retain 80% of that side, use 20% of the held-out 10% as test data, and integrate the DeTeCtive/KNN, DeepSVDD, Energy, and generator-family HRN variants.

- T019 D020 | ablation | retained
  Ablate whether the machine-initialized DeepSVDD center remains trainable after initialization, ending with the center serialized as a non-trainable parameter.

- T020 D021 | method | retained
  Apply a softplus transformation to the average-machine-distance-minus-average-human-distance term used by the supervised DeepSVDD objective, clipping that difference to [-100, 100] before softplus.

- T021 D022 | ablation | retained
  Ablate the Energy detector's ID/OOD margin pair from (-25, -7) to (-27, -5), retaining the (-27, -5) setting.

- T022 D023 | ablation | retained
  Ablate RAID's HRN generator-family partition by splitting the former OpenAI-GPT group into GPT (gpt2 and gpt3) and ChatGPT (chatgpt and gpt4), increasing the nonhuman family groups from five to six and retaining the split grouping.

- T023 D024 | experiment | retained
  Evaluate DeepSVDD, generator-family HRN, and Energy detectors on the DeepFake cross_domains_cross_models benchmark, using its train/validation or test split arguments for training and evaluation.

- T024 D025 | method | retained
  Convert a detector output into an LLM-generation probability using dataset-specific Gaussian class-conditional score distributions whose parameters are estimated from the test set, assuming balanced class priors.

## Demonstration idx=806

Project: Lost in Latent Space: An Empirical Study of Latent Diffusion Models for Physics Emulation

Trajectory insight: LoLa develops a controlled comparison between compressed latent-space diffusion, pixel-space diffusion, and deterministic forecasting. It first fixes field-aware preprocessing and a patchified convolutional autoencoder, then varies bottleneck compression and capacity before standardizing the diffusion objective, context masks, optimizer, noise schedule, backbone, and sampler. Euler experiments expand to Rayleigh–Bénard convection and three-dimensional gravity cooling. Full-trajectory, ensemble, spectral, invariant, and timing metrics finally expose the tradeoff between compression, accuracy, uncertainty, and inference cost.

Decisions:

- T000 D058 | experiment | retained
  Use multi-quadrant Euler flows with both open and periodic boundaries as the primary autoencoder and emulator dataset.

- T001 D087 | experiment | retained
  Evaluate full held-out trajectories with repeated stochastic samples, physical-unit reconstruction and forecast errors, ensemble spread and skill, invariants, spectral and correlation errors, and rollout timing against both truth and decoded truth.

- T002 D109 | experiment | retained
  Train the deterministic autoencoder with a variance-normalized per-field RMSE objective instead of plain MSE or MAE.

- T003 D046 | experiment | retained
  Preprocess Euler by applying log1p to positive thermodynamic fields and standardizing all five fields with fixed dataset statistics, replacing temporary clipping choices.

- T004 D007 | experiment | retained
  Ablate latent bottleneck resolution and channel width across convolutional and vision-transformer autoencoders, retaining DCAE f32c64 as the default two-dimensional representation.

- T005 D008 | ablation | retained
  Compare small and large convolutional autoencoders at a fixed f32c64 bottleneck and retain the large-capacity variant.

- T006 D030 | experiment | retained
  Use an EDM-style fixed-variance preconditioned denoiser after abandoning a learned-variance Gaussian formulation.

- T007 D034 | experiment | retained
  Use a patchified convolutional autoencoder with learned channel projections, three residual blocks per stage, and no deepest-stage self-attention in the selected models.

- T008 D088 | experiment | retained
  Ablate noise-time sampling and noise schedules, retaining uniform time sampling and a bounded log-logit sigma schedule.

- T009 D047 | experiment | retained
  Apply system-appropriate spatial flips, permutations, and rolls during autoencoder training and validation.

- T010 D081 | experiment | retained
  Condition diffusion and deterministic surrogates on explicit contiguous temporal context masks, clamping observed states during diffusion inference.

- T011 D050 | experiment | retained
  Bound encoded states with the softclip2 latent-saturation transform.

- T012 D090 | experiment | retained
  Use a shared transformer backbone with unit spatiotemporal patches, global attention, coordinate features, adaptive modulation, and persistent input skips.

- T013 D063 | experiment | retained
  Add periodic Rayleigh–Bénard convection with standardized buoyancy, pressure, and velocity fields.

- T014 D064 | experiment | retained
  Use dataset-specific log transforms and standardization for Rayleigh–Bénard conditioning constants and gravity-cooling physical fields.

- T015 D068 | experiment | retained
  Ablate AdamW, Shampoo, SOAP, and PSGD, retaining PSGD for autoencoders and AdamW for diffusion and deterministic surrogates.

- T016 D077 | experiment | retained
  Train a pixel-space diffusion emulator on standardized physical trajectories as the direct counterpart to latent diffusion.

- T017 D085 | experiment | retained
  Train deterministic pixel- and latent-space transformer surrogates with trajectory-level mean-squared error.

- T018 D110 | experiment | retained
  Ablate training-window length and stride and retain five consecutive physical states at unit stride for all emulator families.

- T019 D101 | experiment | retained
  Sample conditioned diffusion forecasts with a 16-step Adams–Bashforth solver instead of the earlier LMS sampler.

- T020 D105 | experiment | retained
  Condition latent diffusion on noisy low-resolution physical observations using one-step MMPS likelihood guidance.

- T021 D107 | experiment | retained
  Add three-dimensional turbulence with gravity and cooling as an autoencoding and forecasting benchmark.

## Demonstration idx=259

Project: Creativity or Brute Force? Using Brainteasers as a Window into the Problem-Solving Abilities of Large Language Models

Trajectory insight: The repository trajectory moves from collecting complete Braingle Math and Logic corpora to curating difficult, annotated 250-item subsets and broadening evaluation across models and prompt conditions. It then expands beyond direct correctness into semantic parsing, hints, self-correction, solution sketches, step counts, brute-force labeling, and creative-versus-rudimentary step classification, while refining model-judge inputs, rubrics, configurability, and repeated-judgment reliability checks. Later decisions add Math and Logic reasoning taxonomies and structured interventions—human-step assistance and competition-style formal rewrites—for stratified comparisons with unassisted narrative solving.

Decisions:

- T000 D001 | method | retained
  Use scraped Braingle Math and Logic brainteasers as benchmark data, retaining each item’s question, answer, optional hint, popularity/fun rating, and difficulty rating.

- T001 D002 | experiment | retained
  Evaluate brainteaser solving across repeated examples and successive model settings.

- T002 D003 | experiment | retained
  Run a prompt-strategy ablation for brainteaser solving on both Math and Logic: a basic full-reasoning prompt, an anti-brute-force/math prompt that permits brute force or code only when necessary, explicit versus implicit formal-logic prompting, and a combined condition that supplies the benchmark hint together with the anti-brute-force and rigorous-justification instructions.

- T003 D004 | experiment | retained
  Score direct-solving response correctness, excluding semantic-parsing outputs, with a binary model judge that receives the model response together with the Braingle reference solution.

- T004 D005 | method | retained
  Create focused evaluation subsets consisting of 250 difficult Braingle Math items and 250 curated Logic items, retaining reference answers, completed hints, ratings, final-answer metadata, quality annotations, and structural fields such as problem type, answer flag, depth, width, state-space size, and clue count where available.

- T005 D006 | method | retained
  Add a semantic-parsing task that asks a model to convert each narrative problem into symbolic logical statements without solving the problem.

- T006 D007 | experiment | retained
  Compare ordinary brainteaser solving with hint-conditioned solving on both Math and Logic by appending each item’s completed benchmark hint only when the selected prompt requests use of a hint, rerunning the condition after the benchmark hints were filled in.

- T007 D008 | experiment | retained
  Characterize the complete and top-250-difficulty Braingle Math and Logic populations by measuring hint coverage, difficulty distributions, and reference-answer word and sentence lengths.

- T008 D010 | experiment | retained
  Aggregate binary correctness separately by solving prompt and report it for the full 250-item set and positional slices comprising the first 25, next 25, first 50, and remaining 200 items; additionally report correct-response token totals normalized by each slice size.

- T009 D009 | experiment | retained
  Use a configurable model judge that receives the original problem, generated response, and reference solution to assign binary brute-force or guess-and-check labels to generated solutions and; refine the rubric with a comprehensive-search definition and examples, consolidate prior human-label votes into one label per item, and compare generated versus reference strategy by model, prompt, benchmark difficulty, popularity, and.

- T010 D011 | experiment | retained
  Measure solution complexity by prompting model judges.

- T011 D012 | method | retained
  Add a solution-sketch task that supplies each problem together with its human reference solution and asks multiple models to summarize that solution into simple, logically complete sequential steps.

- T012 D013 | method | retained
  Evaluate self-correction and error-diagnosis by giving a model the problem, a candidate solution, and a supplied correct solution, then asking it to identify specific missed cases and logical errors; run both directions by treating either the human reference or the GPT-o3 basic-prompt solution as the candidate to be corrected, across Math and Logic and multiple models.

- T013 D014 | experiment | retained
  Evaluate generated solution sketches with a binary GPT-o3 adequacy judge that receives the problem, human solution, and model sketch and labels the sketch adequate only if it covers all original solution steps with sufficient detail and contains no errors; compare adequacy with direct-solving correctness by model and prompt, including the hint-conditioned runs.

- T014 D015 | experiment | retained
  Classify each key step in supplied human and model-generated solutions as creative or rudimentary using model judges including GPT-o3 and DSChat; define creative steps as problem-reducing insights and rudimentary steps as routine application, computation, trial-and-error, systematic exploration, code, guess-and-check, or otherwise human-infeasible computation, and report total, creative, and rudimentary step counts.

- T015 D016 | method | retained
  Categorize the Braingle Math benchmark by reasoning type: first try a ten-bucket, multi-label taxonomy generated by o3 and o4-mini, then finalize a single named category field with eight operative categories—Algebra, Arithmetic, Combinatorics, Number Theory, Geometry, Logic, Pattern, and Special Number—grouped for aggregate correctness analysis into Standard, Nonstandard, and Heuristic families, while retaining the automated category columns only as fields marked not to use.

- T016 D017 | experiment | retained
  Measure judge reliability by independently repeating each binary correctness judgment, generated-solution brute-force judgment, and human-reference brute-force judgment five times per item, storing all five labels and analyzing their vote-count distributions.

- T017 D019 | experiment | retained
  Add a step-breakdown-assisted solving condition in which a model receives the problem together with the extracted human-solution steps and is asked to produce a complete solution, enabling comparison with unassisted solving.

- T018 D018 | method | retained
  Categorize the curated Braingle Logic benchmark with named structural and reasoning labels—0D, 1D, 2D, Number, Clusters, Liars, Communication, Compound, Algorithm, Math, Pattern, Linguistic, and Tree—and use those labels both individually and in aggregate families—Simple/large, Complex/small, Math-like, and Heuristic—to stratify model correctness and strategy analysis.

- T019 D020 | experiment | retained
  Compare direct solving of selected Braingle Math narratives with solving formally rewritten, competition-style versions of the same problems: evaluate the top 30 correctly rewritten items with Qwen70, DeepSeek Reasoner, and GPT-o3, reporting correctness before and after rewriting and item-level correct-to-incorrect, unchanged, and incorrect-to-correct transitions.

## Demonstration idx=988

Project: Virus Infection Attack on LLMs: Your Poisoning Can Spread "VIA" Synthetic Data

Trajectory insight: VIA develops from inserting explicit malicious query–response payloads into benign SFT outputs to a self-propagating attack whose payload is a conditional behavior. Hijacking Point Selection evolves through several likelihood heuristics and settles on a continuation-frequency ratio for three-grams; semantic carrier ranking and contextual wrappers make insertions less conspicuous. Matched direct-poisoning, trigger, location, shell, rate, and n-gram ablations test attack design. Evaluation follows clean-query infection through five model generations, utility retention, BackdoorLLM tasks, detectability, and HPS stability across corpora.

Decisions:

- T000 D009 | experiment | retained
  Compare instruction-triggered and explicit-token SST-2 backdoors as conventional attack baselines.

- T001 D014 | experiment | retained
  Measure infection on clean and fixed task-specific prompts using exact payload strings, keyword rules, sentiment judgments, and triggered versus untriggered accuracy.

- T002 D027 | ablation | retained
  Ablate full versus subset-limited infection and sweep poisoning rates for person-opinion objectives.

- T003 D013 | experiment | retained
  Compare VIA with direct malicious query–response poisoning and an output-only direct-poisoning control at matched poison fractions.

- T004 D002 | experiment | retained
  Train poisoned and clean language models with matched LoRA SFT while repeatedly supervising EOS termination.

- T005 D017 | ablation | retained
  Compare VIA variants on mathematical-error propagation while measuring clean-task utility on ARC, GSM8K, HellaSwag, MMLU, TruthfulQA, and Winogrande.

- T006 D001 | method | retained
  Profile candidate SFT corpora by parsing their prompt, response, reasoning, and solution fields and measuring frequent n-grams.

- T007 D012 | method | retained
  Represent the VIA payload as an inserted conditional query–response behavior inside an otherwise benign SFT answer, converging from raw transcript variants to explicit task-conditioned statements.

- T008 D011 | ablation | retained
  Compare quotation wrappers with context-dependent semantic prefix and suffix shells generated from the neighboring response text.

- T009 D015 | method | retained
  Rank carrier records by semantic similarity between clean outputs and poison-topic keywords before applying HPS.

- T010 D033 | method | retained
  Select hijacking points by ranking candidate n-grams with their corpus frequency divided by the maximum frequency of an immediate continuation.

- T011 D019 | experiment | retained
  Evaluate VIA and direct poisoning across BackdoorLLM jailbreak, refusal, sentiment, and SST-2 tasks and ablate poison rate across attack families.

- T012 D036 | experiment | retained
  Compare poison-topic and benign query distributions through embedding clusters and direct topic-related query rates.

- T013 D007 | ablation | retained
  Ablate payload insertion at response start, response end, random interior boundaries, and HPS-selected chain phrases.

- T014 D038 | experiment | retained
  Train descendants on synthetic outputs from the preceding infected model through five generations for opinion, recommendation, and mathematical-error payloads.

- T015 D045 | experiment | retained
  Detect inserted payloads with sliding-window average token loss after abandoning response-judging and conditional-perplexity variants.

- T016 D039 | method | retained
  Use a MIXUP condition that randomly alternates HPS insertions and raw malicious examples within the poisoned subset.

- T017 D048 | experiment | retained
  Measure HPS ranking stability across corpus sizes and corpora using Spearman correlation and phrase-set Jaccard overlap.

- T018 D046 | ablation | retained
  Ablate HPS n-gram length from one through five tokens.

## Demonstration idx=1116

Project: CaMiT: A Time-Aware Car Model Dataset for Classification and Generation

Trajectory insight: CaMiT first builds a time-aware car dataset through detection, deduplication, multi-model weak labeling, human audit, and temporally stratified splits. It then studies two uses of time: conditioning image generation on model year and adapting visual classifiers as car classes and appearances change. In-domain self-supervised pretraining, CLIP/DINO baselines, LoRA adaptation, prototype and covariance classifiers, and replay are compared on a full train-year by test-year grid, with representation shift and class turnover explaining the observed temporal degradation.

Decisions:

- T000 D001 | ablation | retained
  Ablate year conditioning in Stable-Diffusion LoRA fine-tuning and evaluate generated class-year images with CLIP-based KID against matching test folders.

- T001 D002 | experiment | retained
  Analyze generation quality by class dynamics and by the number of training images available for each class-year condition.

- T002 D003 | method | retained
  Create the image pool with vehicle detection and deduplicate each source class using DINOv2 embedding similarity.

- T003 D004 | method | retained
  Annotate car brand and model with a two-stage pipeline in which Qwen proposes a label and GPT-4o verifies or corrects it before taxonomy normalization.

- T004 D005 | method | retained
  Train student classifiers from each weak label source and use four-way agreement among teachers and students as an annotation-confidence signal.

- T005 D006 | experiment | retained
  Human-audit a balanced sample of high-confidence annotations with three raters to estimate residual labeling error.

- T006 D007 | method | retained
  Construct a temporally stratified test set by clustering each class in representation space and sampling high-confidence examples across class-year clusters.

- T007 D008 | experiment | retained
  Pretrain car representations with MoCo and compare static, jointly updated, and year-by-year self-supervised schedules before frozen-backbone evaluation.

- T008 D009 | experiment | retained
  Evaluate DINOv2, CLIP, MoCo, and adapted variants on the complete train-year by test-year matrix, summarizing current, future, and past-year accuracy.

- T009 D010 | experiment | retained
  Adapt CLIP and MoCo classifiers over years with LoRA and an expanding cosine head, comparing isolated, sequential, and replay-based updates.

- T010 D011 | experiment | retained
  Compare nearest-mean, covariance-aware, random-projection, and linear classifiers for incremental learning over frozen yearly embeddings.

- T011 D012 | experiment | retained
  Measure within-class representation shift and dataset class turnover across years, including newly appearing and disappearing classes.

- T012 D013 | experiment | retained
  Compare static, incremental-pretraining, and incremental-classifier results with paired parametric and nonparametric significance tests.

## Demonstration idx=304

Project: Collapsing Taylor Mode Automatic Differentiation

Trajectory insight: The project generalized Taylor-mode automatic differentiation and then discovered that sums defining differential operators can be collapsed during propagation instead of materializing every derivative. The idea progressed from exact Laplacians to weighted and arbitrary mixed-partial operators, and from Monte Carlo Laplacians to exact and stochastic Bi-Laplacians. PyTorch benchmarks isolate replication and sum propagation, while JAX/FOLX and compilation studies test whether existing systems obtain the same savings.

Decisions:

- T000 D001 | method | retained
  Trace PyTorch computation graphs and replace supported operations with Taylor-mode primitives so composed neural networks can be differentiated as jets.

- T001 D002 | method | retained
  Generalize Taylor propagation to arbitrary order using integer partitions and Faà di Bruno multiplicities, including explicit high-order derivatives for nonlinear activations.

- T002 D003 | method | retained
  Represent jet coefficients as traceable replicas and commute replication through the graph to eliminate redundant work.

- T003 D004 | method | retained
  Estimate Laplacian and weighted-Laplacian traces from random directional derivatives, validate convergence against exact references, and benchmark how collapsed propagation scales with sample count.

- T004 D005 | method | retained
  Compute an exact Laplacian by summing second-order Taylor coefficients during forward propagation, reducing the carried derivative state from coordinate-wise terms to one collapsed sum.

- T005 D006 | method | retained
  Extend exact collapsing to weighted second-order operators by factoring positive-semidefinite coefficient matrices into directions and separating positive and negative spectra when necessary.

- T006 D007 | method | retained
  Generalize sum propagation to higher-order trace operators and arbitrary sums of mixed partials, retaining only lower-order derivatives still needed by the chain rule.

- T007 D008 | experiment | retained
  Benchmark exact Laplacian, weighted Laplacian, and Bi-Laplacian strategies on matched neural networks, comparing nested automatic differentiation, ordinary jets, and collapsed jets for correctness, runtime, and memory.

- T008 D009 | method | retained
  Estimate the Bi-Laplacian from Gaussian fourth directional derivatives with the required one-third factor, validate it against exact computation, and benchmark convergence and resource use.

- T009 D010 | method | retained
  Iterate through several exact Bi-Laplacian reconstructions before settling on applying the collapsed Laplacian transformation twice, with symmetry reducing the alternative directional-jet construction.

- T010 D011 | experiment | retained
  Compare against JAX jets and FOLX for exact operators and against JAX directional-derivative baselines for stochastic operators; remove the redundant JAX input-dimension sweep.

- T011 D012 | experiment | retained
  Ablate torch.compile across the PyTorch operator benchmarks, except for the unsupported nested-Hessian Bi-Laplacian baseline.
