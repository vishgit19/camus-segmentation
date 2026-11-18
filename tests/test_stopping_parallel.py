from pathlib import Path

import pytest
import torch
from torch.utils.data import TensorDataset

from camus_seg.common import read_json
from camus_seg.improve import train_seed
from camus_seg.parallel_improve import fit_parallel
from camus_seg.stopping import EarlyStopping


def test_patience_stops_exactly_on_tenth_consecutive_non_improvement():
    stopping = EarlyStopping(10)
    assert stopping.update(.8, 1) == (True, False)
    for epoch in range(2, 11):
        assert stopping.update(.8, epoch) == (False, False)
    assert stopping.update(.79, 11) == (False, True)
    assert stopping.best_epoch == 1
    assert stopping.update(.81, 12) == (True, False)
    assert stopping.bad_epochs == 0


def test_early_stopping_disabled_only_explicitly():
    stopping = EarlyStopping(0)
    for epoch in range(30):
        assert not stopping.update(.5, epoch)[1]
    with pytest.raises(ValueError):
        EarlyStopping(-1)


def fixtures():
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(42)
    images = torch.rand((4, 1, 32, 32), generator=generator)
    masks = torch.randint(4, (4, 32, 32), generator=generator)
    records = [dict(tensor=images[0], target=masks[0].numpy())]
    settings = dict(epochs=2, patience=10, min_delta=0., size=32, width=2, batch_size=2,
                    lr=.001, weight_decay=.0001, device="cpu", amp=False, threads=1)
    identity = dict(kind="synthetic", source_tag="unit_fixture", manifest_sha256="manifest",
                    split_sha256="split", train_patients=["train"], validation_patients=["val"],
                    plan_sha256="plan")
    return TensorDataset(images, masks), records, settings, identity


def test_real_training_loop_stops_and_keeps_best_checkpoint(tmp_path, monkeypatch):
    dataset, validation, settings, identity = fixtures()
    settings["epochs"] = 30
    monkeypatch.setattr("camus_seg.improve.native_dice", lambda *args: .5)
    checkpoint = train_seed(dataset, validation, tmp_path / "trial", settings, identity, 42)
    status = read_json(tmp_path / "trial/training_status.json")
    assert status["epochs_completed"] == 11
    assert status["epochs_without_improvement"] == 10
    assert status["stop_reason"] == "validation_patience"
    assert torch.load(checkpoint, weights_only=True)["epoch"] == 1
    assert not Path(str(checkpoint)+".tmp").exists()


def test_spawned_trials_share_cache_and_finish_independently(tmp_path):
    dataset, validation, settings, identity = fixtures()
    paths = fit_parallel(dataset, validation, tmp_path, settings, identity, [43, 44], 2)
    assert all(tensor.is_shared() for tensor in dataset.tensors)
    assert all(path.is_file() for path in paths)
    assert [read_json(path.parent / "provenance.json")["settings"]["seed"] for path in paths] == [43, 44]
    assert read_json(tmp_path / "active_workers.json")["training_complete"]
