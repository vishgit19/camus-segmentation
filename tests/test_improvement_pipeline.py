import numpy as np
import pytest
import torch

from camus_seg.common import load_manifest, read_json, sha256, write_json
from camus_seg.data import make_split, synthetic
from camus_seg.enhanced import EnhancedUNet
from camus_seg.improve import (evaluate_frozen, load_member, prepared_records,
                               select_candidate, validation_selection)


def test_improvement_selection_and_export_on_geometric_fixtures(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    manifest_path = synthetic(tmp_path / "fixtures", patients=7, size=32)
    manifest = load_manifest(manifest_path)
    split_path = tmp_path / "split.json"
    split = make_split(manifest_path, split_path)
    validation = prepared_records(manifest_path,
        [c for c in manifest["cases"] if c["patient"] in split["val"]], 32)
    models = [EnhancedUNet(2).eval(), EnhancedUNet(2).eval()]
    candidates = validation_selection(models, validation, [42, 43], "cpu", False)
    assert candidates == validation_selection(models, validation, [42, 43], "cpu", False, metric_workers=3)
    assert len(candidates) == 6
    assert {c["id"] for c in candidates} == {
        "seed42_raw", "seed42_lcc", "seed43_raw", "seed43_lcc", "ensemble_raw", "ensemble_lcc"}
    selected = select_candidate(candidates)
    config = dict(**selected, manifest_sha256=sha256(manifest_path), split_sha256=sha256(split_path),
                  selection_partition="val", frozen_before_test=True)
    baseline = {p: {metric: dict(mean=.5) for metric in ("dice", "iou", "hd95_mm", "assd_mm")}
                for p in split["test"]}
    baseline_path = tmp_path / "baseline.json"
    write_json(baseline_path, baseline)
    report = evaluate_frozen(manifest_path, split, manifest, models, config, tmp_path / "export",
                             dict(size=32, device="cpu", amp=False), baseline_path)
    assert report["test_patients"] == split["test"]
    assert report["model"]["overall"]["dice"]["n_patients"] == 2
    assert np.isfinite(report["model"]["overall"]["dice"]["mean"])
    assert (tmp_path / "export/worst_cases.png").is_file()
    paired = read_json(tmp_path / "export/paired_comparison.json")
    assert paired["dice"]["n_patients"] == 2


def test_improvement_checkpoint_rejects_wrong_fitting_membership(tmp_path):
    identity = dict(manifest_sha256="manifest", split_sha256="split", train_patients=["train"],
                    validation_patients=["val"], plan_sha256="plan")
    provenance = dict(**identity, settings=dict(width=2))
    provenance["train_patients"] = ["test"]
    checkpoint = tmp_path / "invalid.pt"
    torch.save(dict(state_dict=EnhancedUNet(2).state_dict(), provenance=provenance), checkpoint)
    with pytest.raises(ValueError, match="identity"):
        load_member(checkpoint, identity, "cpu")


def test_parallel_preparation_preserves_tensors_and_order(tmp_path):
    torch.set_num_threads(1)
    manifest_path = synthetic(tmp_path / "fixtures", patients=7, size=32)
    cases = load_manifest(manifest_path)["cases"][:8]
    serial = prepared_records(manifest_path, cases, 32, True, workers=1)
    parallel = prepared_records(manifest_path, cases, 32, True, workers=4)
    for actual, expected in zip(parallel.tensors, serial.tensors):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
