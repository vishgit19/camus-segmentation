# Verification record

Completed on 18 November 2026 using WSL2, Python 3.9.18, PyTorch 2.7.1 with CUDA 12.6, and an NVIDIA RTX 4050 Laptop GPU. Exact versions are in [environment-wsl.json](environment-wsl.json). The working project, raw/prepared data, virtual environment and checkpoints are retained on D:.

## Software and runtime checks

- **32 pytest cases passed.** Coverage includes physical spacing/axis order, overlap and empty-mask policies, patient weighting and bootstrap repeatability, patient leakage and pixel duplicates, input/checkpoint fingerprints, NIfTI/MetaImage round trips, label validation, publisher membership, and cross-entropy value/gradient equivalence to PyTorch's reference. New checks cover letterbox/native-grid restoration, deep supervision, affine label alignment, component cleanup, validation selection and patient-paired comparison.
- The stopping tests cover exactly 10 consecutive non-improvements, counter reset on a strict gain, and real training that stops at epoch 11 when epoch 1 is best. Threaded preprocessing and validation scoring match their serial counterparts. Shared CPU tensors and spawned training workers are checked.
- A two-process AMP CUDA smoke run passed concurrent training, checkpoint reload and scoring with strict deterministic algorithms. Best checkpoints use atomic replacement. CUDA's NLLLoss2d kernel does not support strict determinism in this environment, so the equivalent one-hot log-softmax loss is checked against PyTorch's reference.
- The synthetic demo completed at epoch **19**, retaining epoch **9**, after 10 non-improving validation epochs. Its maximum budget was 20 epochs. Outputs in `examples/synthetic/` are software checks, not medical results.
- Standalone three-model CUDA inference passed on a training frame (`patient0001_2CH_ED`). The saved NIfTI preserves the input size, spacing, origin and direction exactly; labels are 0/1/2/3. Input/output hashes and all three checkpoint identities were verified. This check does not use test data for model selection.
- GitHub Actions runs all tests and the synthetic smoke command on Linux/Python 3.12. The 32-test runtime-policy commit passed CI; subsequent publication checks are visible in the repository's Actions tab.

## Data and original baseline

- The official download contains **5,506 files / 363,175,007 bytes**, selecting annotated ED/ES frames and metadata from all 500 patients. Every transferred file's SHA-256 was checked during the move to D:. The source download manifest SHA-256 is `22119474dbe78800fa62c23b030d3caff4b7998eb4e8478bd6a7da3b60032056`.
- All **2,000 real image/mask pairs** passed import checks. A training frame was visually checked for label semantics and alignment; its NIfTI header uses millimetres and 0.308-mm in-plane spacing. The publisher's explicit corrected **400/50/50** patient membership passed coverage, ID-overlap and exact-pixel-duplicate checks.
- The original fixed **50-epoch CUDA baseline** completed and selected epoch **45**. Test foreground macro-Dice was **0.8153 [0.7953, 0.8324]**, IoU **0.7052**, HD95 **20.22 mm**, and ASSD **4.48 mm** on 50 patients / 200 frames. Its failure gallery and all 12 checksummed artifacts are preserved unchanged in [camus-seed42](../results/camus-seed42/README.md).

## Completed improvement

The user requested stopping the first scheduler, patience 10, and parallel resource use. At cancellation, enhanced seed 42 had just completed 120 epochs; seed 43 had completed one epoch. The completed checkpoint was retained, and the partial seed 43 attempt was excluded. Version 2 performed no new test evaluation. Its cancellation record is included as `previous_attempts.json` in the follow-up result bundle.

| Trial | Actual epochs | Best epoch | Best validation Dice | Terminal state |
| --- | --- | --- | --- | --- |
| Inherited seed 42 | 120 | 34 | 0.911548 | Completed before the runtime-policy change |
| New seed 43 | 42 | 32 | 0.908066 | Exactly 10 consecutive non-improvements |
| New seed 44 | 48 | 38 | 0.910960 | Exactly 10 consecutive non-improvements |

The two new CUDA trials ran concurrently, sharing prepared CPU tensors. Preprocessing and validation surface metrics used up to 20 CPU threads. Observed GPU utilization reached 100%, using approximately 3 GB of its 6 GB VRAM. Early stopping saved 150 of the two trials' combined maximum 240 epochs. Concurrent trials share one physical GPU; no twofold throughput claim is made.

Validation scored all eight predeclared configurations: three individual checkpoints and their equal-probability ensemble, each raw or with largest-component cleanup. The frozen winner was **ensemble_raw**, validation Dice **0.916979**, without cleanup. Its configuration and checkpoint fingerprints were saved before test evaluation.

The winner was evaluated once on the same 50 test patients / 200 frames. Patient-mean Dice was **0.9141 [0.9071, 0.9202]**, IoU **0.8456**, HD95 **4.06 mm**, and ASSD **1.46 mm**. The paired Dice gain was **0.0988 [0.0851, 0.1155]**; the paired HD95 change was **−16.17 mm [−18.27, −14.17]**. Intervals use 2,000 patient-bootstrap replicates; paired comparisons retain patient identities. No undefined empty-mask distance occurred among 600 frame/structure pairs. This counter does not measure all anatomical errors.

All three learning trajectories and the four automatically selected lowest-Dice frames were visually inspected. The stronger masks remain imperfect, including displaced boundaries and atrial errors. No additional selection or retuning followed this test evaluation. [The complete follow-up bundle](../results/camus-v3/README.md) includes per-frame/per-patient metrics, subgroups, paired intervals, every validation candidate, training histories/statuses, plans, provenance, resource and cancellation records, licensed figures and checksums.

**This is a development follow-up on the previously inspected baseline test set, not a fresh independent confirmation.** The bundled comparison does not isolate individual changes or establish state-of-the-art, external or clinical performance. Intervals quantify patient sampling variation conditional on these trained models, not training-seed or development-selection uncertainty. Image-quality subgroups overlap by patient because quality is recorded separately for each view.

## Source identity

Mixed training-source versions are intentional because completed work was reused. Fingerprints cover the package's Python files in filename order. The original result bundle remains bound to its historical source, rather than being relabelled with the newer code hash.

| Use | Git revision | Package SHA-256 |
| --- | --- | --- |
| Original baseline | `f40b0294d05890e0783c7c0c86e899d93e6e37df` | `92321ad1837cf3b285fc1a0e8618f9775ddcbc5f087ccdae21ac9de2f7f00e43` |
| Inherited enhanced seed 42 | `f41c270c43d95be68882cad6523f74afef503e10` | `3a4e1b672166018aecfe84d94471f50bb642ef04cf6ed39d27cfab600020b0be` |
| New seeds 43/44 and frozen follow-up evaluation | `a01590b26e17a94eb3df8dae440c563aecab39a5` | `76118a7689fc494ccc29e513297106a3d8573456b39fdc560f5888c337bbd816` |

The current package retains the last fingerprint. Reporting scripts and documentation were completed afterward. Every member records its training-source, training-plan and checkpoint hashes. Baseline and follow-up artifact checksums are checked against the committed Git bytes. Raw data, prepared arrays and weights remain local and excluded from Git.
