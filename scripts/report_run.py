"""Package metrics, learning curves, and failure figures from an evaluated CAMUS run."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def estimate(summary):
    if summary["mean"] is None:
        return "undefined"
    value = f"{summary['mean']:.4f}"
    interval = summary["ci95"]
    if interval is not None:
        value += f" [{interval[0]:.4f}, {interval[1]:.4f}]"
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("training", "evaluation", "split", "plan", "source-license", "out"):
        parser.add_argument("--"+name, required=True, type=Path)
    args = parser.parse_args()
    report = json.loads((args.evaluation / "report.json").read_text())
    history = json.loads((args.training / "history.json").read_text())
    provenance = json.loads((args.training / "provenance.json").read_text())
    split = json.loads(args.split.read_text())
    if report["kind"] != "camus":
        raise ValueError("This report template is for real CAMUS runs only")
    for key in ("manifest_sha256", "split_sha256"):
        if report[key] != provenance[key]:
            raise ValueError("Training and evaluation provenance differ")
    if hashlib.sha256(args.split.read_bytes()).hexdigest() != report["split_sha256"]:
        raise ValueError("The supplied split differs from the evaluated split")
    if hashlib.sha256((args.training / "best.pt").read_bytes()).hexdigest() != report["checkpoint_sha256"]:
        raise ValueError("Evaluation did not use this training checkpoint")
    args.out.mkdir(parents=True, exist_ok=False)
    for name in ("report.json", "frame_metrics.csv", "background_metrics.csv",
                 "patient_metrics.json", "worst_cases.png"):
        shutil.copy2(args.evaluation / name, args.out / name)
    for name in ("history.json", "provenance.json"):
        shutil.copy2(args.training / name, args.out / name)
    shutil.copy2(args.split, args.out / "split.json")
    shutil.copy2(args.plan, args.out / "run-plan.json")
    shutil.copy2(args.source_license, args.out / "DATA_LICENSE.md")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].plot(epochs, [row["train_loss"] for row in history], color="#176b87")
    axes[0].set(title="Training loss", xlabel="Epoch", ylabel="CE / soft-Dice loss")
    axes[1].plot(epochs, [row["val_native_macro_dice"] for row in history], color="#267b54")
    axes[1].axvline(report["selected_epoch"], color="#a24f24", linestyle="--", label="Selected epoch")
    axes[1].set(title="Validation Dice on the native grid", xlabel="Epoch", ylabel="Foreground macro-Dice", ylim=(0, 1))
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(args.out / "learning_curve.png", dpi=140)
    plt.close(fig)

    overall = report["model"]["overall"]
    lines = ["# Completed CAMUS baseline", "",
             f"One fixed {len(history)}-epoch {provenance['settings']['device']} run, "
             f"seed {provenance['settings']['seed']}; frozen membership: "
             f"{len(split['train'])} training / {len(split['val'])} validation / "
             f"{len(split['test'])} test patients. All four annotated frames per patient "
             "remain together. The selected checkpoint was evaluated once on "
             f"{len(report['test_patients'])} held-out test patients "
             f"({len(report['test_patients'])*4} frames).", "",
             f"**Primary patient-mean foreground macro-Dice: {estimate(overall['dice'])}.** "
             "Brackets show 95% percentile patient-bootstrap confidence intervals "
             f"({report['bootstrap']['repetitions']:,} replicates).", "",
             "| Scope | Dice | IoU | HD95 (mm) | ASSD (mm) | Empty-mask failures |",
             "| --- | --- | --- | --- | --- | --- |"]
    for scope, summary in [("All structures", overall), *report["model"]["by_structure"].items()]:
        lines.append(f"| {scope} | {estimate(summary['dice'])} | {estimate(summary['iou'])} | "
                     f"{estimate(summary['hd95_mm'])} | {estimate(summary['assd_mm'])} | "
                     f"{summary['surface_failures']} / {summary['n_frame_class_pairs']} |")
    lines += ["", "Distance summaries condition on defined surfaces; undefined pairs and "
              "contributing patient counts are recorded in report.json. Both-empty masks "
              "score Dice/IoU=1 and distance=0; exactly-one-empty masks score overlap=0, "
              "distance=null and an empty-mask surface failure. This counter does not count "
              "every segmentation error: fragmented masks or distant false-positive islands "
              "can have defined surfaces and large distances. Background is excluded from macro scores.", "",
              "## Image quality", "",
              "| Quality | Patients | Dice [95% CI] | Empty-mask failures |",
              "| --- | --- | --- | --- |"]
    for name, summary in report["model"]["subgroups"]["quality"].items():
        lines.append(f"| {name} | {summary['dice']['n_patients']} | {estimate(summary['dice'])} | "
                     f"{summary['surface_failures']} |")
    baseline = report["all_background"]["overall"]
    lines += ["", "Quality is recorded per view. A patient can contribute to more than one "
              "quality subgroup, so subgroup patient counts need not sum to the test-set size.", "",
              f"All-background reference Dice: {estimate(baseline['dice'])}. "
              "This checks metric behavior; it is not a competitive segmentation baseline.", "",
              f"The validation-selected checkpoint is epoch {report['selected_epoch']}. "
              "Settings and environment are in provenance.json; the predeclared budget and "
              "test policy are in run-plan.json. No augmentation, pretraining, postprocessing "
              "or test-driven tuning was used.", "",
              "![Training and validation learning curves](learning_curve.png)", "",
              "## Failure cases", "",
              "These are the four test frames with the lowest foreground macro-Dice, "
              "selected automatically. They include reference masks and model predictions.", "",
              "![Worst held-out CAMUS frames](worst_cases.png)", "",
              "## Interpretation and provenance", "",
              "This is a single-seed, single-site research baseline. Its confidence intervals "
              "capture held-out patient sampling variation, not optimization or split uncertainty. "
              "It includes poor-quality frames and uses the corrected public NIfTI release; "
              "it is not a direct reproduction of the original paper's complete evaluation. "
              "It provides no evidence of external or clinical validation.", "",
              "Code is MIT-licensed. The CAMUS-derived failure figure is an adaptation "
              "under CC BY-NC-SA 4.0, with the source's non-commercial scientific research "
              "terms retained in DATA_LICENSE.md. Frames are resized and arranged with "
              "reference/prediction panels; no raw dataset or model weights are distributed here.", "",
              "Source: [official CAMUS deposit](https://humanheart-project.creatis.insa-lyon.fr/database/#collection/6373703d73e9f0047faa1bc8). "
              "Leclerc et al., *Deep Learning for Segmentation Using an Open Large-Scale Dataset "
              "in 2D Echocardiography*, IEEE TMI 38(9), 2198–2210 (2019), "
              "[doi:10.1109/TMI.2019.2900516](https://doi.org/10.1109/TMI.2019.2900516).", ""]
    (args.out / "README.md").write_text("\n".join(lines), encoding="utf-8")
    checksums = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(args.out.iterdir()) if p.is_file()}
    (args.out / "checksums.json").write_text(json.dumps(checksums, indent=2) + "\n")


if __name__ == "__main__":
    main()
