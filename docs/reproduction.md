# Reproduction details

Install the tested dependencies, obtain CAMUS under its source terms, prepare the four annotated frames per patient and preserve the publisher's corrected split as described in the root README. Retain the generated manifest and split: checkpoints bind to their exact bytes and patient identities. Reproducibility across different hardware, CUDA or library versions is not guaranteed.

The completed improved experiment used three checkpoints: a completed seed 42 from the earlier source, plus new seeds 43 and 44 with patience 10. The local project retains all of them. The public repository includes their histories and fingerprints but does not distribute weights or raw data. These instructions reconstruct the historical work only when those local checkpoints are unavailable; routine new work should use patience 10 and reuse completed valid checkpoints.

## Reconstruct only the inherited checkpoint

Use an isolated checkout of the source that trained seed 42. Do not rerun the original two-seed scheduler: its second trial was interrupted and superseded. From the current repository root, with the tested CUDA environment active:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
git worktree add --detach work/historical-v2 f41c270c43d95be68882cad6523f74afef503e10
PYTHONPATH="$PWD/work/historical-v2/src" python - <<'PY'
from pathlib import Path
from camus_seg.common import read_json, sha256, write_json
from camus_seg.experiment import experiment_data
from camus_seg.improve import prepared_records, seed_everything, train_seed

manifest_path, split_path = Path("data/prepared/manifest.json"), Path("data/split.json")
manifest, split = experiment_data(manifest_path, split_path)
plan = read_json("work/historical-v2/docs/improvement-plan.json")
out = Path("runs/reconstructed-v2")
out.mkdir(parents=True, exist_ok=False)
write_json(out / "run-plan.json", plan)
identity = dict(kind=manifest["kind"], source_tag=manifest["source_tag"],
    manifest_sha256=sha256(manifest_path), split_sha256=sha256(split_path),
    train_patients=split["train"], validation_patients=split["val"],
    plan_sha256=sha256(out / "run-plan.json"))
seed_everything(42)
training = prepared_records(manifest_path,
    [c for c in manifest["cases"] if c["patient"] in split["train"]], 256, True)
validation = prepared_records(manifest_path,
    [c for c in manifest["cases"] if c["patient"] in split["val"]], 256)
train_seed(training, validation, out / "seed42", plan["training"], identity, 42)
PY
```

This historical source uses the original fixed 120-epoch budget, preceding the user's stopping-policy change. It trains seed 42 only and performs no test evaluation. Its historical best epoch was 34. The archived plan records the original two-seed intention; only its first seed is reconstructed here. Return to the current package for the next command (`PYTHONPATH` above applies only to that one invocation).

## Run the two new trials and frozen comparison

```bash
python -m camus_seg.parallel_improve \
  --manifest data/prepared/manifest.json --split data/split.json \
  --plan docs/improvement-plan-v3.json \
  --baseline-patients results/camus-seed42/patient_metrics.json \
  --inherited runs/reconstructed-v2/seed42 --out runs/reconstructed-v3
```

For the retained local checkpoint, use `--inherited runs/improvement-v2/seed42` instead. Output directories must not already exist. The current source trains seeds 43 and 44 concurrently when memory permits, stops each after 10 consecutive non-improving validation epochs, scores all eight predeclared validation configurations, freezes the winner and evaluates it once. The completed run selected all three seeds without component cleanup. A reconstruction can yield different weights or selection; report its actual outcome instead of copying the historical result.

The stored `previous_attempts.json` is a historical cancellation record, not an output of a fresh reconstruction. To package the completed local run:

```bash
python scripts/report_improvement.py --run runs/improvement-v3 \
  --baseline results/camus-seed42 \
  --source-license results/camus-seed42/DATA_LICENSE.md \
  --out results/camus-v3-new-copy
```

The reporting helper requires the cancellation record for this historical follow-up bundle and checks the recorded stopping states, source/plan identities and checkpoint hashes. It does not retrain or reevaluate the test set.

## Original baseline

The original smaller model was trained at source revision `f40b0294d05890e0783c7c0c86e899d93e6e37df` for 50 epochs, selecting epoch 45. That source predates early stopping. For a fixed-budget historical reconstruction with the current legacy trainer, explicitly pass `--patience 0`; routine new training defaults to patience 10. The original numerical record and its checksums remain unchanged in `results/camus-seed42/`.

Test-set reuse is disclosed in every report: this follow-up compares against previously inspected test patients. A fresh external study would be required for independent confirmation.
