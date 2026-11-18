"""Resource-aware parallel trials with validation patience and reuse of completed work."""
import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import signal
import sys
import time

import torch
import torch.multiprocessing as mp

from .common import read_json, sha256, write_json
from .experiment import experiment_data
from .improve import (evaluate_frozen, load_member, prepared_records, seed_everything,
                      select_candidate, train_seed, validation_selection)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def worker(dataset, validation, root, settings, identity, seed):
    with (Path(root) / f"seed{seed}.log").open("w") as log:
        with redirect_stdout(Tee(sys.stdout, log)):
            train_seed(dataset, validation, Path(root) / f"seed{seed}", settings, identity, seed)


def fit_parallel(dataset, validation, out, settings, identity, seeds, workers=2):
    """Spawn independent CUDA contexts; CPU tensors are shared instead of duplicated."""
    for tensor in dataset.tensors:
        tensor.share_memory_()
    for record in validation:
        record["tensor"].share_memory_()
    context, processes = mp.get_context("spawn"), []
    waiting = list(seeds)
    previous_handler = signal.getsignal(signal.SIGTERM)

    def interrupted(signum, frame):
        raise KeyboardInterrupt("Parallel run terminated")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        while waiting or any(p.is_alive() for _, p in processes):
            active = sum(p.is_alive() for _, p in processes)
            while waiting and active < workers:
                seed = waiting.pop(0)
                process = context.Process(target=worker,
                    args=(dataset, validation, str(out), settings, identity, seed))
                process.start()
                processes.append((seed, process))
                active += 1
            write_json(Path(out) / "active_workers.json", dict(parent_pid=os.getpid(),
                workers=[dict(seed=seed, pid=p.pid, alive=p.is_alive(), exitcode=p.exitcode)
                         for seed, p in processes]))
            failures = [(seed, p.exitcode) for seed, p in processes if p.exitcode not in (None, 0)]
            if failures:
                raise RuntimeError(f"Training worker failed: {failures}")
            time.sleep(1)
        for _, process in processes:
            process.join()
        failures = [(seed, p.exitcode) for seed, p in processes if p.exitcode != 0]
        if failures:
            raise RuntimeError(f"Training worker failed: {failures}")
        write_json(Path(out) / "active_workers.json", dict(parent_pid=os.getpid(),
            training_complete=True, workers=[dict(seed=seed, pid=p.pid, alive=False, exitcode=p.exitcode)
                                            for seed, p in processes]))
    finally:
        for _, process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=15)
        signal.signal(signal.SIGTERM, previous_handler)
    return [Path(out) / f"seed{seed}" / "best.pt" for seed in seeds]


def run(manifest_path, split_path, plan_path, baseline, inherited, out):
    out, inherited = Path(out), Path(inherited)
    plan = read_json(plan_path)
    settings = plan["training"]
    if settings.get("patience") != 10:
        raise ValueError("This run must enforce the user-requested patience of10")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    manifest, split = experiment_data(manifest_path, split_path)
    if manifest["kind"] != "camus" or settings["size"] % 16:
        raise ValueError("Expected CAMUS data and an input divisible by16")
    if not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")
    free, total = torch.cuda.mem_get_info()
    requested = plan["resources"]["training_workers"]
    estimate = plan["resources"]["observed_single_trial_vram_mib"] * 1024**2
    workers = min(requested, max(1, int((free-512*1024**2)//estimate)))
    metric_workers = min(plan["resources"]["metric_workers"], len(os.sched_getaffinity(0)))
    prepare_workers = min(plan["resources"].get("prepare_workers", metric_workers), len(os.sched_getaffinity(0)))
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "run-plan.json", plan)
    write_json(out / "resources.json", dict(training_workers=workers, metric_workers=metric_workers,
               cpu_threads_available=len(os.sched_getaffinity(0)), gpu_free_bytes=free,
               gpu_total_bytes=total, shared_cpu_cache=True, prepare_workers=prepare_workers))
    identity = dict(kind=manifest["kind"], source_tag=manifest["source_tag"],
                    manifest_sha256=sha256(manifest_path), split_sha256=sha256(split_path),
                    train_patients=split["train"], validation_patients=split["val"],
                    plan_sha256=sha256(out / "run-plan.json"))
    inherited_target = out / "seed42"
    inherited_target.mkdir()
    for name in ("best.pt", "history.json", "provenance.json"):
        shutil.copy2(inherited / name, inherited_target / name)
    inherited_provenance = read_json(inherited_target / "provenance.json")
    inherited_identity = dict(identity, plan_sha256=inherited_provenance["plan_sha256"])
    model, inherited_checkpoint = load_member(inherited_target / "best.pt", inherited_identity, "cpu")
    del model
    write_json(inherited_target / "training_status.json", dict(inherited=True,
        epochs_completed=len(read_json(inherited_target / "history.json")),
        best_epoch=inherited_checkpoint["epoch"], best_validation_dice=inherited_checkpoint["validation_dice"],
        stop_reason="completed_before_user_policy_change", source=str(inherited)))
    seed_everything(42, 1)
    training = prepared_records(manifest_path, [c for c in manifest["cases"] if c["patient"] in split["train"]],
                                settings["size"], True, prepare_workers)
    validation = prepared_records(manifest_path, [c for c in manifest["cases"] if c["patient"] in split["val"]],
                                  settings["size"], workers=prepare_workers)
    print(f"Starting {workers} parallel training workers; {metric_workers} CPU metric threads", flush=True)
    paths = [inherited_target / "best.pt"] + fit_parallel(training, validation, out, settings,
                                                         identity, settings["seeds"], workers)
    seeds, loaded, members = [42] + settings["seeds"], [], []
    for index, path in enumerate(paths):
        provenance = read_json(path.parent / "provenance.json")
        model, checkpoint = load_member(path, dict(identity, plan_sha256=provenance["plan_sha256"]), "cuda")
        loaded.append(model)
        members.append(dict(path=path.relative_to(out).as_posix(), sha256=sha256(path), seed=seeds[index],
            epoch=checkpoint["epoch"], inherited=index == 0,
            training_plan_sha256=provenance["plan_sha256"], training_code_sha256=provenance["code_sha256"]))
    candidates = validation_selection(loaded, validation, seeds, "cuda", settings["amp"], metric_workers)
    write_json(out / "validation_selection.json", candidates)
    selected = select_candidate(candidates)
    config = dict(**selected, **identity, checkpoints=members, test_seen_before=True,
                  selection_partition="val", frozen_before_test=True,
                  frozen_utc=datetime.now(timezone.utc).isoformat())
    write_json(out / "selected_config.json", config)
    print(f"FROZEN: {config['id']} {config['validation']}", flush=True)
    report = evaluate_frozen(manifest_path, split, manifest, loaded, config, out / "test", settings, baseline)
    print(f"COMPLETE: Dice={report['model']['overall']['dice']} HD95={report['model']['overall']['hd95_mm']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "split", "plan", "baseline-patients", "inherited", "out"):
        parser.add_argument("--"+name, required=True)
    args = parser.parse_args()
    run(args.manifest, args.split, args.plan, args.baseline_patients, args.inherited, args.out)


if __name__ == "__main__":
    main()
