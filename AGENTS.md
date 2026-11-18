# Training and evaluation policy

- Stop a training trial after **10 consecutive validation epochs without improvement**. Keep its best validation checkpoint. Apply this to new training by default; disable only for an explicitly requested fixed-budget historical reproduction.
- Preserve completed checkpoints and reuse valid prior work instead of retraining it unnecessarily.
- Run independent trials concurrently when they fit the available GPU/RAM. Share prepared CPU tensors between workers and parallelize independent CPU metric calculations. Measure utilization and throughput; keep a VRAM/RAM reserve and avoid oversubscription that increases total runtime.
- Keep model/policy selection on patient-disjoint validation data. Freeze the chosen configuration before test scoring; disclose reuse of the previously inspected test set.
- Record early stopping, interrupted trials, source/data/checkpoint fingerprints, and actual epochs. Never describe an interrupted or early-stopped trial as completing its maximum epoch budget.
