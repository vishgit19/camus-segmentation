"""Bounded, validation-selected improvement experiment. See docs/improvement-plan.json."""
import argparse
import csv
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .common import LABELS, read_json, sha256, write_json
from .data import load_case
from .enhanced import (EnhancedUNet, affine_augment, clean_components, deep_supervision_loss,
                       letterbox_input, letterbox_mask, native_probabilities)
from .evaluate import gallery, summarize
from .experiment import code_sha256, experiment_data
from .metrics import aggregate, binary_metrics, bootstrap_mean


def seed_everything(seed, threads=2):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepared_records(manifest_path, cases, size, training=False):
    records, inputs, masks = [], [], []
    for index, case in enumerate(cases, 1):
        image, mask, spacing = load_case(manifest_path, case)
        tensor = letterbox_input(image, size)
        if training:
            inputs.append(tensor)
            masks.append(letterbox_mask(mask, size))
        else:
            records.append(dict(case=case, tensor=tensor, target=mask.astype(np.uint8),
                                spacing=spacing))
        if index % 200 == 0:
            print(f"Cached {index}/{len(cases)} {'training' if training else 'validation'} frames", flush=True)
    return TensorDataset(torch.stack(inputs), torch.stack(masks)) if training else records


def native_dice(model, records, device, amp):
    scores = []
    for record in records:
        probabilities = native_probabilities(model, record["tensor"], record["target"].shape,
                                              device, amp)
        prediction = probabilities.argmax(0).cpu().numpy()
        for value in LABELS:
            p, t = prediction == value, record["target"] == value
            denominator = p.sum() + t.sum()
            scores.append(float(2*(p & t).sum()/denominator) if denominator else 1.)
    return float(np.mean(scores))


def train_seed(dataset, validation, out, settings, identity, seed):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    seed_everything(seed)
    device, amp = settings["device"], settings["amp"]
    if amp and device != "cuda":
        raise ValueError("AMP is supported only by the CUDA experiment")
    model = EnhancedUNet(settings["width"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"],
                                  weight_decay=settings["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, settings["epochs"], eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    loader = DataLoader(dataset, batch_size=settings["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(seed), num_workers=0,
                        pin_memory=device == "cuda")
    provenance = dict(**identity, settings=dict(**settings, seed=seed,
                      architecture="enhanced_unet_v2", preprocessing="letterbox"),
                      code_sha256=code_sha256(), parameters=sum(p.numel() for p in model.parameters()),
                      environment=dict(python=platform.python_version(), torch=str(torch.__version__),
                                       numpy=np.__version__, cuda=torch.version.cuda,
                                       gpu=torch.cuda.get_device_name() if device == "cuda" else None,
                                       deterministic_algorithms=True))
    write_json(out / "provenance.json", provenance)
    best, history = -1., []
    for epoch in range(1, settings["epochs"]+1):
        started, total = time.perf_counter(), 0.
        model.train()
        learning_rate = optimizer.param_groups[0]["lr"]
        for images, target in loader:
            images, target = images.to(device, non_blocking=True), target.to(device, non_blocking=True)
            images, target = affine_augment(images, target)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=amp):
                outputs = model(images)
                loss = deep_supervision_loss(outputs, target)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach()) * len(images)
        score = native_dice(model, validation, device, amp)
        if not np.isfinite(score):
            raise ValueError("Non-finite validation score")
        history.append(dict(epoch=epoch, train_loss=total/len(dataset), val_native_macro_dice=score,
                            lr=learning_rate, amp_scale=scaler.get_scale(),
                            seconds=time.perf_counter()-started))
        write_json(out / "history.json", history)
        if score > best:
            best = score
            torch.save(dict(state_dict=model.state_dict(), epoch=epoch, validation_dice=score,
                            provenance=provenance), out / "best.pt")
        scheduler.step()
        print(f"seed={seed} epoch={epoch}/{settings['epochs']} loss={total/len(dataset):.4f} "
              f"val_dice={score:.4f} seconds={history[-1]['seconds']:.1f}", flush=True)
    del model, optimizer, scaler
    if device == "cuda":
        torch.cuda.empty_cache()
    return out / "best.pt"


def load_member(path, identity, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    for key in ("manifest_sha256", "split_sha256", "train_patients", "validation_patients", "plan_sha256"):
        if checkpoint["provenance"][key] != identity[key]:
            raise ValueError("Checkpoint/data identity mismatch")
    settings = checkpoint["provenance"]["settings"]
    model = EnhancedUNet(settings["width"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model, checkpoint


def metric_rows(case, prediction, target, spacing):
    identity = {key: case[key] for key in ("patient", "view", "phase", "quality")}
    return [dict(**identity, structure=name, **binary_metrics(prediction == value,
                 target == value, spacing)) for value, name in LABELS.items()]


def point_summary(rows):
    patients = sorted({r["patient"] for r in rows})
    result = {}
    for key in ("dice", "hd95_mm"):
        means = []
        for patient in patients:
            values = [r[key] for r in rows if r["patient"] == patient and r[key] is not None]
            if values:
                means.append(float(np.mean(values)))
        result[key] = float(np.mean(means)) if means else None
    result["empty_mask_failures"] = sum(r["surface_failure"] for r in rows)
    return result


def select_candidate(candidates):
    maximum = max(c["validation"]["dice"] for c in candidates)
    eligible = [c for c in candidates if c["validation"]["dice"] >= maximum-.001]
    return min(eligible, key=lambda c: (c["validation"]["empty_mask_failures"],
               c["validation"]["hd95_mm"] if c["validation"]["hd95_mm"] is not None else float("inf"),
               len(c["members"]), c["id"]))


def validation_selection(models, validation, seeds, device, amp):
    candidates = []
    memberships = [[index] for index in range(len(models))] + [list(range(len(models)))]
    for members in memberships:
        name = "ensemble" if len(members) > 1 else f"seed{seeds[members[0]]}"
        for cleanup in (False, True):
            candidates.append(dict(id=name+("_lcc" if cleanup else "_raw"), members=members,
                                   cleanup=cleanup, rows=[]))
    for index, record in enumerate(validation, 1):
        probabilities = [native_probabilities(model, record["tensor"], record["target"].shape,
                                               device, amp) for model in models]
        predictions = {tuple(m): torch.stack([probabilities[i] for i in m]).mean(0).argmax(0).cpu().numpy()
                       for m in memberships}
        for candidate in candidates:
            prediction = predictions[tuple(candidate["members"])]
            if candidate["cleanup"]:
                prediction = clean_components(prediction)
            candidate["rows"].extend(metric_rows(record["case"], prediction, record["target"], record["spacing"]))
        if index % 50 == 0:
            print(f"Validation selection scored {index}/{len(validation)} frames", flush=True)
    for candidate in candidates:
        candidate["validation"] = point_summary(candidate.pop("rows"))
    return candidates


def paired_comparison(new_patient_metrics, baseline_path, repetitions=2000):
    baseline = read_json(baseline_path)
    if set(new_patient_metrics) != set(baseline):
        raise ValueError("Paired comparison requires identical test patient membership")
    comparison = {"direction": "new minus original; higher Dice/IoU, lower distance is better",
                  "test_status": "Development follow-up on a previously inspected test set"}
    for metric in ("dice", "iou", "hd95_mm", "assd_mm"):
        differences = []
        for patient in sorted(baseline):
            new, old = new_patient_metrics[patient][metric]["mean"], baseline[patient][metric]["mean"]
            if new is not None and old is not None:
                differences.append(new-old)
        comparison[metric] = bootstrap_mean(differences, repetitions, 42) if differences else None
    return comparison


def evaluate_frozen(manifest_path, split, manifest, models, config, out, settings, baseline):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    rows, background_rows, examples = [], [], []
    device, amp, size = settings["device"], settings["amp"], settings["size"]
    for case in manifest["cases"]:
        if case["patient"] not in split["test"]:
            continue
        image, target, spacing = load_case(manifest_path, case)
        tensor = letterbox_input(image, size)
        probabilities = [native_probabilities(models[i], tensor, target.shape, device, amp)
                         for i in config["members"]]
        prediction = torch.stack(probabilities).mean(0).argmax(0).cpu().numpy()
        if config["cleanup"]:
            prediction = clean_components(prediction)
        case_rows = metric_rows(case, prediction, target, spacing)
        rows.extend(case_rows)
        background_rows.extend(metric_rows(case, np.zeros_like(target), target, spacing))
        score = float(np.mean([r["dice"] for r in case_rows]))
        examples.append((score, case, image, target, prediction))
        examples.sort(key=lambda x: (x[0], x[1]["patient"], x[1]["view"], x[1]["phase"]))
        del examples[4:]
    report = dict(kind="camus", source_tag=manifest["source_tag"], protocol=split["protocol"],
                  partition="test", test_patients=split["test"],
                  manifest_sha256=config["manifest_sha256"], split_sha256=config["split_sha256"],
                  code_sha256=code_sha256(), configuration=config,
                  test_status="Follow-up on previously inspected test set; selection used validation only",
                  bootstrap=dict(unit="patient", repetitions=2000, seed=42, method="percentile", confidence=.95),
                  model=summarize(rows, 2000, 42), all_background=summarize(background_rows, 2000, 42))
    write_json(out / "report.json", report)
    patient_metrics = {p: aggregate([r for r in rows if r["patient"] == p], 2000, 42)
                       for p in split["test"]}
    write_json(out / "patient_metrics.json", patient_metrics)
    write_json(out / "paired_comparison.json", paired_comparison(patient_metrics, baseline))
    for name, data in (("frame_metrics.csv", rows), ("background_metrics.csv", background_rows)):
        with (out / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    gallery(examples, out, "camus")
    return report


def run(manifest_path, split_path, plan_path, baseline, out):
    plan, out = read_json(plan_path), Path(out)
    settings = plan["training"]
    if settings["epochs"] < 1 or settings["batch_size"] < 1 or settings["lr"] <= 0:
        raise ValueError("Invalid training budget")
    if len(settings["seeds"]) != len(set(settings["seeds"])) or not settings["seeds"]:
        raise ValueError("Seeds must be nonempty and unique")
    if settings["size"] % 16 or settings["size"] < 32:
        raise ValueError("Enhanced U-Net input must be >=32 and divisible by 16")
    if settings["device"] == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    manifest, split = experiment_data(manifest_path, split_path)
    if manifest["kind"] != "camus":
        raise ValueError("Improvement runner is restricted to real CAMUS data")
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "run-plan.json", plan)
    seed_everything(42)
    identity = dict(kind=manifest["kind"], source_tag=manifest["source_tag"],
                    manifest_sha256=sha256(manifest_path), split_sha256=sha256(split_path),
                    train_patients=split["train"], validation_patients=split["val"],
                    plan_sha256=sha256(out / "run-plan.json"))
    training = prepared_records(manifest_path, [c for c in manifest["cases"] if c["patient"] in split["train"]],
                                settings["size"], training=True)
    validation = prepared_records(manifest_path, [c for c in manifest["cases"] if c["patient"] in split["val"]],
                                  settings["size"])
    checkpoints = [train_seed(training, validation, out / f"seed{seed}", settings, identity, seed)
                   for seed in settings["seeds"]]
    loaded = [load_member(path, identity, settings["device"]) for path in checkpoints]
    models = [item[0] for item in loaded]
    candidates = validation_selection(models, validation, settings["seeds"], settings["device"], settings["amp"])
    write_json(out / "validation_selection.json", candidates)
    selected = select_candidate(candidates)
    config = dict(**selected, **identity, test_seen_before=True,
                  checkpoints=[dict(path=path.relative_to(out).as_posix(), sha256=sha256(path),
                                    seed=settings["seeds"][i], epoch=loaded[i][1]["epoch"],
                                    training_code_sha256=loaded[i][1]["provenance"]["code_sha256"])
                               for i, path in enumerate(checkpoints)],
                  selection_partition="val", frozen_before_test=True)
    write_json(out / "selected_config.json", config)
    print(f"FROZEN VALIDATION SELECTION: {config['id']} {config['validation']}", flush=True)
    report = evaluate_frozen(manifest_path, split, manifest, models, config, out / "test",
                              settings, baseline)
    print(f"COMPLETE: test Dice={report['model']['overall']['dice']}; "
          f"HD95={report['model']['overall']['hd95_mm']}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "split", "plan", "baseline-patients", "out"):
        parser.add_argument("--"+name, required=True)
    args = parser.parse_args()
    run(args.manifest, args.split, args.plan, args.baseline_patients, args.out)


if __name__ == "__main__":
    main()
