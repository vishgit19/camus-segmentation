import copy

import numpy as np
import pytest
import SimpleITK as sitk

from camus_seg.common import load_manifest, read_json, validate_split, write_json
from camus_seg.data import load_case, make_split, official_split, prepare, synthetic, validate_arrays
from camus_seg.evaluate import evaluate
from camus_seg.experiment import train


def test_patient_split_and_leakage_guards(tmp_path):
    manifest_path = synthetic(tmp_path / "data", patients=10, size=32)
    manifest = load_manifest(manifest_path)
    split = make_split(manifest_path, tmp_path / "split.json")
    assert [len(split[p]) for p in ("train", "val", "test")] == [7, 1, 2]
    assert split == make_split(manifest_path, tmp_path / "split2.json")
    bad = copy.deepcopy(split)
    bad["test"][0] = bad["train"][0]
    with pytest.raises(ValueError, match="leakage"):
        validate_split(manifest, bad)
    bad_manifest = copy.deepcopy(manifest)
    train_case = next(c for c in manifest["cases"] if c["patient"] in split["train"])
    test_case = next(c for c in bad_manifest["cases"] if c["patient"] in split["test"])
    test_case["image_sha256"] = train_case["image_sha256"]
    with pytest.raises(ValueError, match="Identical"):
        validate_split(bad_manifest, split)
    case_path = manifest_path.parent / manifest["cases"][0]["file"]
    case_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        load_case(manifest_path, manifest["cases"][0])


@pytest.mark.parametrize("extension", [".nii.gz", ".mhd"])
def test_medical_io_preserves_grid_labels_and_quality(tmp_path, extension):
    raw = tmp_path / "raw" / "patient0001"
    raw.mkdir(parents=True)
    image = np.arange(30, dtype=np.float32).reshape(1, 5, 6)
    mask = np.arange(30, dtype=np.uint8).reshape(1, 5, 6) % 4
    for view in ("2CH", "4CH"):
        (raw / f"Info_{view}.cfg").write_text("ImageQuality: Poor\n")
        for phase in ("ED", "ES"):
            for suffix, array in (("", image), ("_gt", mask)):
                volume = sitk.GetImageFromArray(array)
                volume.SetSpacing((.25, .75, 1.0))
                sitk.WriteImage(volume, str(raw / f"patient0001_{view}_{phase}{suffix}{extension}"))
    path = prepare(tmp_path / "raw", tmp_path / "prepared", "synthetic_io_fixture")
    manifest = load_manifest(path)
    actual_image, actual_mask, spacing = load_case(path, manifest["cases"][0])
    np.testing.assert_allclose(actual_image, image[0])
    np.testing.assert_array_equal(actual_mask, mask[0])
    np.testing.assert_allclose(spacing, (.75, .25))
    assert manifest["cases"][0]["quality"] == "Poor"


def test_labels_cannot_be_silently_truncated():
    with pytest.raises(ValueError, match="integer labels"):
        validate_arrays(np.ones((2, 2)), np.full((2, 2), 1.5), (1, 1))


def test_publisher_membership_is_imported_and_validated(tmp_path):
    patients = [f"patient{i:04d}" for i in range(1, 501)]
    cases = [dict(patient=p, view=v, phase=t, image_sha256=f"{p}_{v}_{t}")
             for p in patients for v in ("2CH", "4CH") for t in ("ED", "ES")]
    manifest = tmp_path / "manifest.json"
    write_json(manifest, dict(kind="camus", source_tag="test", cases=cases))
    # Deliberately non-contiguous IDs: membership must come from these files.
    training, validation, testing = patients[100:], patients[:100:2], patients[1:100:2]
    for name, ids in (("training", training), ("validation", validation), ("testing", testing)):
        (tmp_path / f"subgroup_{name}.txt").write_text("\n".join(ids))
    split = official_split(manifest, tmp_path, tmp_path / "split.json")
    assert split["test"] == testing
    assert split["val"] == validation
    (tmp_path / "subgroup_testing.txt").write_text("\n".join(training[:50]))
    with pytest.raises(ValueError, match="leakage"):
        official_split(manifest, tmp_path, tmp_path / "rejected.json")


def test_training_evaluation_and_checkpoint_binding(tmp_path):
    manifest = synthetic(tmp_path / "data", patients=7, size=32)
    split = tmp_path / "split.json"
    make_split(manifest, split)
    checkpoint = train(manifest, split, tmp_path / "training", epochs=1, size=32,
                       width=2, threads=1, batch_size=4)
    report = evaluate(manifest, split, checkpoint, tmp_path / "evaluation", threads=1,
                      repetitions=100)
    assert report["kind"] == "synthetic"
    assert report["partition"] == "test"
    assert report["model"]["overall"]["dice"]["n_patients"] == 2
    assert report["all_background"]["overall"]["dice"]["mean"] == 0
    assert (tmp_path / "evaluation" / "worst_cases.png").is_file()
    assert read_json(tmp_path / "evaluation" / "report.json") == report
    modified = read_json(split)
    modified["train"][0], modified["test"][0] = modified["test"][0], modified["train"][0]
    changed_split = tmp_path / "changed.json"
    write_json(changed_split, modified)
    with pytest.raises(ValueError, match="different split"):
        evaluate(manifest, changed_split, checkpoint, tmp_path / "rejected", repetitions=100)
