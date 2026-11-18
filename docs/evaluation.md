# Evaluation protocol v1

The primary estimand is the mean, across held-out patients, of native-grid foreground macro-Dice. Each patient contributes four frames (2CH/4CH × ED/ES); each frame contributes three structures. Background is excluded from the headline score. Complete frame coverage is required, so patients and structures receive equal weight for overlap metrics.

## Split and selection

With `--official-dir`, `split` reads the publisher's corrected `subgroup_training.txt`, `subgroup_validation.txt` and `subgroup_testing.txt` files and requires 400/50/50 patients. Their source file hashes are recorded. The reported CAMUS experiment uses this membership, not guessed ID ranges or the paper's full cross-validation study.

Without that argument, `split` sorts patient IDs, shuffles with NumPy's seeded generator, and allocates `floor(.70N)` patients to train, the next `floor(.85N)-floor(.70N)` to validation, and the remainder to test. That custom mode is used by synthetic checks. Both modes store exact IDs and the manifest digest. All frames for a patient remain together. Pixel-identical images across partitions fail validation.

Training accesses only training and validation arrays. The checkpoint is selected by highest validation native-grid foreground macro-Dice; ties keep the earliest epoch. Evaluation verifies manifest and split hashes and rejects a checkpoint whose recorded fitting or validation IDs overlap test IDs. All raw arrays loaded during fitting and evaluation must match their prepared-file hashes. Checkpoints are loaded with `weights_only=True`.

No command prevents a researcher repeatedly inspecting the test set. A disciplined predeclared protocol is still required. Record the source archive identity, hyperparameters, hardware, dependencies, and every planned seed. Do not tune against test outputs or the failure gallery.

## Metrics

Dice is `2|P∩T|/(|P|+|T|)`; IoU is `|P∩T|/|P∪T|`. Values are on a 0–1 scale. HD95 is the maximum of the two directed 95th-percentile nearest-boundary distances. ASSD is the mean of the concatenated directed distances (weighted by boundary point counts). Boundaries are pixels removed by four-connected erosion with zero-valued exterior. Distances use `(spacing_y, spacing_x)` in millimetres on the **original image grid**. These conventions may differ from other implementations' pooled HD95 or contour definitions.

Empty-mask policy is explicit:

| Prediction / target | Dice / IoU | HD95 / ASSD | Count |
| --- | --- | --- | --- |
| Both empty | 1 / 1 | 0 / 0 | Both empty flags |
| Exactly one empty | 0 / 0 | Undefined (`null`) | Surface failure |
| Both nonempty | Computed | Computed in mm | No surface failure |

Surface summaries condition on defined frame/structure distances. Within a patient they average the remaining valid pairs; patients with no defined distances are excluded from that surface metric. Every summary includes excluded-pair counts, contributing patient counts, and surface-failure counts. This conditional distance estimate can look optimistic when predictions fail, so interpret it together with Dice and the failures. Undefined distances are never replaced with a perfect zero or silently removed from the failure count. Both-empty overlap scores are included, with empty-target counts exposed.

For each metric, average eligible frame/structure values within each patient, then average patient means. The 95% percentile interval resamples those patient means with replacement, using 2,000 seeded replicates by default. Frames and pixels are never treated as independent bootstrap observations. The report provides overall, per-structure, and view/phase/quality summaries using the same patient aggregation. Subgroups with one contributing patient have no CI. These are marginal descriptive intervals, with no multiplicity correction, hypothesis tests, or guarantees of adequate coverage in tiny groups. They capture test-patient sampling variation, not training or split uncertainty.

An all-background predictor is scored on the same held-out patients. It is a sanity reference, not a competitive segmentation baseline. Per-frame CSVs, per-patient JSON, and the worst four frame macro-Dice cases allow auditing. Do not interpret the synthetic run as a trained medical model.

## Before reporting a CAMUS result

1. Confirm acquisition terms, release identity, label semantics, image/mask registration and spacing on actual source files.
2. Predeclare the labelled development pool, patient split, primary metric, seeds and model selection budget.
3. Train to the predeclared budget and inspect only validation data during development.
4. Evaluate the frozen checkpoint on held-out patients, reporting Dice/IoU, distances, CIs, counts, quality subgroups and failures together.
5. Describe the single-site/device setting, manual reference limitations, and mismatch to the original challenge protocol. External validation and clinical review remain future work.
