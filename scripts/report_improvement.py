"""Publish auditable artifacts for the frozen validation-selected follow-up experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def estimate(summary, digits=4):
    if summary is None or summary["mean"] is None:
        return "undefined"
    text = f"{summary['mean']:.{digits}f}"
    if summary["ci95"] is not None:
        text += f" [{summary['ci95'][0]:.{digits}f}, {summary['ci95'][1]:.{digits}f}]"
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "baseline", "source-license", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    report, baseline = read(args.run / "test/report.json"), read(args.baseline / "report.json")
    config, plan = read(args.run / "selected_config.json"), read(args.run / "run-plan.json")
    candidates = read(args.run / "validation_selection.json")
    paired = read(args.run / "test/paired_comparison.json")
    if config != report["configuration"] or config["selection_partition"] != "val":
        raise ValueError("Test configuration differs from frozen validation selection")
    if not config["frozen_before_test"] or not config["test_seen_before"]:
        raise ValueError("Missing evaluation disclosure")
    if digest(args.run / "run-plan.json") != config["plan_sha256"]:
        raise ValueError("Frozen plan fingerprint mismatch")
    if report["split_sha256"] != baseline["split_sha256"] or report["test_patients"] != baseline["test_patients"]:
        raise ValueError("Comparison uses different patients or split")
    histories, provenances, statuses = {}, {}, {}
    for member in config["checkpoints"]:
        checkpoint = args.run / member["path"]
        if digest(checkpoint) != member["sha256"]:
            raise ValueError("Checkpoint fingerprint mismatch")
        seed = member["seed"]
        histories[seed] = read(checkpoint.parent / "history.json")
        provenances[seed] = read(checkpoint.parent / "provenance.json")
        statuses[seed] = read(checkpoint.parent / "training_status.json")
        if provenances[seed]["code_sha256"] != member["training_code_sha256"]:
            raise ValueError("Training source fingerprint differs from frozen member record")
        if statuses[seed]["epochs_completed"] != len(histories[seed]):
            raise ValueError("Training status/history mismatch")
        if statuses[seed]["best_epoch"] != member["epoch"]:
            raise ValueError("Selected checkpoint differs from best recorded epoch")
        if provenances[seed]["plan_sha256"] != member.get("training_plan_sha256", config["plan_sha256"]):
            raise ValueError("Member training plan fingerprint mismatch")
        reason = statuses[seed]["stop_reason"]
        if member.get("inherited"):
            if reason != "completed_before_user_policy_change":
                raise ValueError("Missing inherited-run disclosure")
        elif reason == "validation_patience":
            if statuses[seed]["patience"] != 10 or statuses[seed]["epochs_without_improvement"] != 10:
                raise ValueError("Early-stopping policy not satisfied")
        elif reason != "maximum_epochs" or len(histories[seed]) != plan["training"]["epochs"]:
            raise ValueError("Trial has not reached a valid terminal state")
    args.out.mkdir(parents=True, exist_ok=False)
    for path in (args.run / "test").iterdir():
        if path.is_file():
            shutil.copy2(path, args.out / path.name)
    for name in ("selected_config.json", "run-plan.json", "validation_selection.json", "resources.json", "previous_attempts.json"):
        shutil.copy2(args.run / name, args.out / name)
    for seed in histories:
        for name in ("history.json", "provenance.json", "training_status.json"):
            shutil.copy2(args.run / f"seed{seed}" / name, args.out / f"seed{seed}_{name}")
    shutil.copy2(args.source_license, args.out / "DATA_LICENSE.md")
    (args.out / "baseline_summary.json").write_text(json.dumps(dict(
        report_sha256=digest(args.baseline / "report.json"),
        model=baseline["model"], code_sha256=baseline["code_sha256"]), indent=2) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for seed, history in histories.items():
        epochs = [row["epoch"] for row in history]
        axes[0].plot(epochs, [r["train_loss"] for r in history], label=f"Seed {seed}")
        axes[1].plot(epochs, [r["val_native_macro_dice"] for r in history], label=f"Seed {seed}")
    axes[0].set(title="Augmented training loss", xlabel="Epoch", ylabel="Deep-supervision CE / Dice loss")
    axes[1].set(title="Native-grid validation Dice", xlabel="Epoch", ylabel="Foreground macro-Dice", ylim=(0, 1))
    for axis in axes:
        axis.grid(alpha=.2)
        axis.legend()
    fig.tight_layout()
    fig.savefig(args.out / "learning_curves.png", dpi=140)
    plt.close(fig)

    overall, original = report["model"]["overall"], baseline["model"]["overall"]
    seeds = [config["checkpoints"][index]["seed"] for index in config["members"]]
    lines = ["# Validation-selected CAMUS improvement", "",
             "**Evaluation status: this is a development follow-up on the same 50 test "
             "patients whose baseline results and failure gallery were already inspected. "
             "It is not a fresh independent confirmation.** All new checkpoint, seed, "
             "ensemble and cleanup selection used the 50 validation patients only.", "",
             f"The frozen choice is **{config['id']}**, using seed(s) {seeds}, "
             f"with component cleanup **{config['cleanup']}**. The original 400/50/50 "
             "patient membership is unchanged. The completed seed 42 checkpoint was reused; "
             "two new trials ran concurrently with a maximum of 120 epochs and patience 10. "
             "Actual epoch counts and stop reasons are recorded below. "
             "The selected configuration was evaluated once on all 200 test frames.", "",
             "| Patient-mean metric | Original | Selected model [95% CI] | Paired change [95% CI] |",
             "| --- | --- | --- | --- |"]
    for metric, label in (("dice", "Foreground macro-Dice"), ("iou", "Foreground macro-IoU"),
                           ("hd95_mm", "HD95 (mm)"), ("assd_mm", "ASSD (mm)")):
        lines.append(f"| {label} | {original[metric]['mean']:.4f} | {estimate(overall[metric])} | "
                     f"{estimate(paired[metric])} |")
    lines += ["", "Changes are new minus original: positive overlap and negative distances "
              "indicate improvement. Intervals use 2,000 patient-paired bootstrap replicates; "
              "they preserve the shared patient identities. They reflect patient sampling "
              "variation conditional on these trained models, not fresh-test independence "
              "or uncertainty over all training seeds.", "",
              f"Undefined empty-mask distance pairs: {overall['surface_failures']} / "
              f"{overall['n_frame_class_pairs']}. This is an empty-mask counter, not a count "
              "of all segmentation or anatomical errors. Conditional distance exclusions "
              "and sample counts are in report.json.", "",
              "## Structures", "",
              "| Structure | Original Dice | Selected Dice [95% CI] | Original HD95 | Selected HD95 |",
              "| --- | --- | --- | --- | --- |"]
    for name, summary in report["model"]["by_structure"].items():
        old = baseline["model"]["by_structure"][name]
        lines.append(f"| {name} | {old['dice']['mean']:.4f} | {estimate(summary['dice'])} | "
                     f"{estimate(old['hd95_mm'])} | {estimate(summary['hd95_mm'])} |")
    lines += ["", "## Complete validation selection", "",
              "| Candidate | Dice | HD95 (mm) | Empty-mask failures | Decision |",
              "| --- | --- | --- | --- | --- |"]
    for candidate in sorted(candidates, key=lambda c: -c["validation"]["dice"]):
        values = candidate["validation"]
        distance = f"{values['hd95_mm']:.4f}" if values['hd95_mm'] is not None else "undefined"
        lines.append(f"| {candidate['id']} | {values['dice']:.4f} | {distance} | "
                     f"{values['empty_mask_failures']} | {'Selected' if candidate['id'] == config['id'] else ''} |")
    lines += ["", "Each seed's checkpoint was selected by highest native-grid validation Dice. "
              f"Among the {len(candidates)} fixed inference candidates, configurations within 0.001 of "
              "the highest validation Dice were ranked by fewer empty-mask distance "
              "failures, lower mean HD95, fewer ensemble members, then lexical ID. "
              "No policy was chosen using the new test scores.", "",
              "## Training and model", "",
              "The model has four downsamplings (five levels), base width 16, instance "
              "normalization, LeakyReLU and transposed-convolution upsampling. Inputs use "
              "aspect-preserving 256-square letterboxing. Auxiliary decoder losses use "
              "nearest-downsampled labels at four scales, with normalized weights "
              "1, 0.5, 0.25 and 0.125. All distances are scored after native-grid restoration.", "",
              "The already completed seed 42 model was retained. New seeds 43 and 44 used "
              "up to 120 epochs, with immediate termination after 10 consecutive validation "
              "epochs without a strict Dice improvement. Batch size 16, AdamW at 0.001 "
              "with weight decay 0.0001, cosine scheduling to 0.00001, gradient norm "
              "clipping at 5, and mixed-precision CUDA were used. Paired affine augmentation "
              "uses +/-10-degree rotation, scale 0.9–1.1 and +/-8% translation; no mirroring. "
              "Losses are evaluated in float32. Strict deterministic algorithms remain enabled. "
              "The two new CUDA jobs share one prepared CPU tensor cache; data preparation "
              "and independent validation metric calculations use up to 20 CPU threads.", "",
              "| Seed | Actual epochs | Selected epoch | Best raw validation Dice | Parameters | Stop reason | Used |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for index, member in enumerate(config["checkpoints"]):
        seed, epoch = member["seed"], member["epoch"]
        score = histories[seed][epoch-1]["val_native_macro_dice"]
        lines.append(f"| {seed} | {len(histories[seed])} | {epoch} | {score:.4f} | "
                     f"{provenances[seed]['parameters']:,} | {statuses[seed]['stop_reason']} | {index in config['members']} |")
    lines += ["", "![Training and validation trajectories for all three seeds](learning_curves.png)", "",
              "## Image quality", "",
              "| Quality | Contributing patients | Selected Dice [95% CI] |",
              "| --- | --- | --- |"]
    for name, summary in report["model"]["subgroups"]["quality"].items():
        lines.append(f"| {name} | {summary['dice']['n_patients']} | {estimate(summary['dice'])} |")
    lines += ["", "Quality is per view, so a patient may contribute to multiple subgroups. "
              "These descriptive groups overlap and their counts need not sum to 50.", "",
              "## Remaining failure cases", "",
              "The four lowest foreground macro-Dice frames for the selected model are "
              "shown automatically. All qualities and all test patients remain included.", "",
              "![Worst held-out follow-up predictions](worst_cases.png)", "",
              "## Scope and attribution", "",
              "This comparison evaluates a bundled training/inference recipe; it does not "
              "isolate the causal contribution of each architectural or training change. "
              "There is no claim of state-of-the-art performance, external validation or "
              "clinical suitability. The original baseline and its immutable records are "
              "preserved in ../camus-seed42/. The version 2 scheduler was cancelled at the "
              "user's runtime-policy change: seed 42 had completed 120 epochs and seed 43 "
              "had run one epoch. That partial seed 43 attempt is excluded; its replacement "
              "and seed 44 are the new early-stopped trials. Mixed training-source versions "
              "are intentional for the reused checkpoint and are recorded per member. "
              "Checkpoint, source, split and plan fingerprints "
              "are retained, along with every validation candidate and all three training histories.", "",
              "Training choices were informed by [Kjelsrud et al., Lightweight image "
              "segmentation for echocardiography](https://arxiv.org/abs/2509.03631); "
              "this implementation and its evaluation are not a replication of that study.", "",
              "Code is MIT. CAMUS-derived failure figures retain CC BY-NC-SA 4.0 and "
              "the source's non-commercial scientific research terms in DATA_LICENSE.md. "
              "Frames are arranged with manual-reference and model-prediction panels. "
              "Source: [official CAMUS deposit](https://humanheart-project.creatis.insa-lyon.fr/database/#collection/6373703d73e9f0047faa1bc8). "
              "Leclerc et al., IEEE TMI 38(9), 2198–2210 (2019), "
              "[doi:10.1109/TMI.2019.2900516](https://doi.org/10.1109/TMI.2019.2900516). "
              "Raw data and weights remain local and are not redistributed here.", ""]
    (args.out / "README.md").write_text("\n".join(lines), encoding="utf-8")
    (args.out / "checksums.json").write_text(json.dumps(
        {p.name: digest(p) for p in sorted(args.out.iterdir()) if p.is_file()}, indent=2) + "\n")


if __name__ == "__main__":
    main()
