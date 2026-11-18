# Synthetic software-check example

These outputs come from generated geometric fixtures, **not CAMUS patients or ultrasound acquisitions**. They demonstrate the runnable evaluation pipeline and deliberately expose imperfect predictions. They do not measure medical segmentation quality.

Reproduce from the repository root in a clean environment:

```bash
python -m camus_seg.cli demo --out runs/synthetic-demo
```

The fixture generator uses seed 7; split, model initialization and training use seed 42. There are 12 synthetic IDs with four frames each, divided 8/2/2. Training uses width 4, input size 64, batch size 8, 20 epochs and learning rate 0.005. Checkpoint selection uses validation Dice. The smoke report uses 200 patient-bootstrap replicates; the real evaluation command defaults to 2,000.

Files preserve the generated report, per-frame/per-patient metrics, selected checkpoint's provenance, exact split, and epoch history. Synthetic arrays and model weights are regenerated locally and are not distributed. See the report's `kind`, hash fields, subgroup sizes and surface-failure counts before interpreting any number. With only two test IDs, its confidence intervals have no useful research interpretation.

The four worst held-out fixture frames are shown below. Left: input geometry with noise. Middle: reference. Right: model prediction. The missing atrium predictions are counted as surface failures rather than assigned zero distance.

![Synthetic geometry failure cases; not clinical data](worst_cases.png)
