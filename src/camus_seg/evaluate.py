import csv
from pathlib import Path

import numpy as np
import torch

from .common import LABELS, sha256, write_json
from .data import load_case
from .experiment import code_sha256, experiment_data
from .metrics import aggregate, binary_metrics
from .model import UNet, predict


def load_checkpoint(path, manifest_path, split_path, manifest, split, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    provenance = checkpoint["provenance"]
    if provenance["manifest_sha256"] != sha256(manifest_path):
        raise ValueError("Checkpoint was trained on a different manifest")
    if provenance["split_sha256"] != sha256(split_path):
        raise ValueError("Checkpoint was trained on a different split")
    if provenance["kind"] != manifest["kind"]:
        raise ValueError("Synthetic and CAMUS artifacts cannot be mixed")
    used = set(provenance["train_patients"]) | set(provenance["validation_patients"])
    if used & set(split["test"]):
        raise ValueError("Test patients were used in fitting or checkpoint selection")
    settings = provenance["settings"]
    model = UNet(settings["width"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model, settings["size"], checkpoint


def summarize(rows, repetitions, seed):
    return dict(overall=aggregate(rows, repetitions, seed),
                by_structure={name: aggregate([r for r in rows if r["structure"] == name],
                                              repetitions, seed) for name in LABELS.values()},
                subgroups={field: {value: aggregate([r for r in rows if r[field] == value],
                                                    repetitions, seed)
                                   for value in sorted({r[field] for r in rows})}
                           for field in ("view", "phase", "quality")})


def gallery(examples, out, kind):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    examples = sorted(examples, key=lambda x: (x[0], x[1]["patient"], x[1]["view"],
                                               x[1]["phase"]))[:4]
    fig, axes = plt.subplots(len(examples), 3, figsize=(9, 3*len(examples)), squeeze=False)
    palette = ListedColormap(["#101820", "#43c7db", "#f6b44b", "#e46fa4"])
    for axrow, (score, case, image, target, prediction) in zip(axes, examples):
        axrow[0].imshow(image, cmap="gray")
        axrow[1].imshow(target, cmap=palette, vmin=0, vmax=3, interpolation="nearest")
        axrow[2].imshow(prediction, cmap=palette, vmin=0, vmax=3, interpolation="nearest")
        for ax in axrow:
            ax.axis("off")
        axrow[0].set_title(f"{case['patient']} {case['view']} {case['phase']}", fontsize=9)
        axrow[1].set_title("Reference mask")
        axrow[2].set_title(f"Prediction | Dice {score:.3f}")
    title = "SYNTHETIC GEOMETRY | software check only" if kind == "synthetic" else "CAMUS test failures"
    fig.suptitle(title + "\nCyan: LV cavity | Amber: myocardium | Pink: left atrium")
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(Path(out) / "worst_cases.png", dpi=130)
    plt.close(fig)


def evaluate(manifest_path, split_path, checkpoint_path, out, *, device="cpu",
             repetitions=2000, seed=42, threads=2):
    if repetitions < 100 or threads < 1:
        raise ValueError("Use >=100 bootstrap repetitions and >=1 thread")
    torch.set_num_threads(threads)
    manifest, split = experiment_data(manifest_path, split_path)
    model, size, checkpoint = load_checkpoint(checkpoint_path, manifest_path, split_path,
                                              manifest, split, device)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    rows, baseline_rows, examples = [], [], []
    for case in manifest["cases"]:
        if case["patient"] not in split["test"]:
            continue
        image, target, spacing = load_case(manifest_path, case)
        prediction = predict(model, image, size, device)
        scores = []
        for label, name in LABELS.items():
            identity = {key: case[key] for key in ("patient", "view", "phase", "quality")}
            identity["structure"] = name
            metrics = binary_metrics(prediction == label, target == label, spacing)
            rows.append(dict(**identity, **metrics))
            baseline_rows.append(dict(**identity, **binary_metrics(np.zeros_like(target),
                                                                   target == label, spacing)))
            scores.append(metrics["dice"])
        examples.append((float(np.mean(scores)), case, image, target, prediction))
        # Retain only the worst four native-resolution cases, not the whole test set.
        examples.sort(key=lambda x: (x[0], x[1]["patient"], x[1]["view"], x[1]["phase"]))
        del examples[4:]
    report = dict(kind=manifest["kind"], source_tag=manifest["source_tag"],
                  protocol=split["protocol"], partition="test", test_patients=split["test"],
                  checkpoint_sha256=sha256(checkpoint_path),
                  manifest_sha256=sha256(manifest_path), split_sha256=sha256(split_path),
                  code_sha256=code_sha256(), selected_epoch=checkpoint["epoch"],
                  bootstrap=dict(unit="patient", repetitions=repetitions, seed=seed,
                                 method="percentile", confidence=.95),
                  model=summarize(rows, repetitions, seed),
                  all_background=summarize(baseline_rows, repetitions, seed))
    write_json(out / "report.json", report)
    for name, data in (("frame_metrics.csv", rows), ("background_metrics.csv", baseline_rows)):
        with (out / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    write_json(out / "patient_metrics.json", {
        p: aggregate([r for r in rows if r["patient"] == p], repetitions, seed)
        for p in split["test"]})
    gallery(examples, out, manifest["kind"])
    return report
