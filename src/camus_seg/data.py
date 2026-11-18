import hashlib
import re
from pathlib import Path

import numpy as np

from .common import load_manifest, sha256, validate_split, write_json


def validate_arrays(image, mask, spacing):
    if image.ndim != 2 or image.shape != mask.shape or min(image.shape) < 2:
        raise ValueError("Image and mask must be matching 2D arrays")
    if not np.isfinite(image).all() or not np.isfinite(mask).all():
        raise ValueError("Non-finite image or mask")
    if not np.equal(mask, np.rint(mask)).all() or not set(np.unique(mask)) <= {0, 1, 2, 3}:
        raise ValueError("Expected integer labels: background=0, LV=1, myocardium=2, LA=3")
    if len(spacing) != 2 or not np.isfinite(spacing).all() or min(spacing) <= 0:
        raise ValueError("Expected positive finite (y, x) spacing in millimetres")


def load_case(manifest_path, case):
    path = Path(manifest_path).parent / case["file"]
    if sha256(path) != case["file_sha256"]:
        raise ValueError(f"Prepared data checksum changed: {case['file']}")
    with np.load(path, allow_pickle=False) as record:
        image, mask, spacing = record["image"], record["mask"], record["spacing_yx"]
    validate_arrays(image, mask, spacing)
    return image.astype(np.float32), mask.astype(np.int64), spacing


def save_case(out, patient, view, phase, image, mask, spacing, quality="Unknown", **source):
    validate_arrays(image, mask, spacing)
    path = Path(out) / f"{patient}_{view}_{phase}.npz"
    np.savez_compressed(path, image=image.astype(np.float32), mask=mask.astype(np.uint8),
                        spacing_yx=np.asarray(spacing, dtype=float))
    return dict(patient=patient, view=view, phase=phase, quality=quality, file=path.name,
                file_sha256=sha256(path),
                image_sha256=hashlib.sha256(np.ascontiguousarray(image, dtype=np.float32).tobytes()
                                            + str(image.shape).encode()).hexdigest(), **source)


def prepare(raw, out, source_tag):
    """Index one labelled release. Never infer official train/test IDs from number ranges."""
    import SimpleITK as sitk

    raw, out = Path(raw), Path(out)
    if not source_tag.strip():
        raise ValueError("Record the archive/release identity with --source-tag")
    out.mkdir(parents=True, exist_ok=False)
    pattern = re.compile(r"^(patient\d+)_(2CH|4CH)_(ED|ES)_gt(\.nii\.gz|\.mhd)$")
    cases, keys = [], set()
    for target in sorted(raw.rglob("*")):
        match = pattern.match(target.name)
        if not match:
            continue
        patient, view, phase, extension = match.groups()
        key = patient, view, phase
        if key in keys:
            raise ValueError(f"Duplicate {key}: do not merge releases or flatten old train/test IDs")
        keys.add(key)
        image_path = target.with_name(f"{patient}_{view}_{phase}{extension}")
        image_itk, mask_itk = sitk.ReadImage(str(image_path)), sitk.ReadImage(str(target))
        for getter in ("GetSize", "GetSpacing", "GetOrigin", "GetDirection"):
            if not np.allclose(getattr(image_itk, getter)(), getattr(mask_itk, getter)()):
                raise ValueError(f"Image/mask physical grids differ: {target}")
        image, mask = sitk.GetArrayFromImage(image_itk), sitk.GetArrayFromImage(mask_itk)
        if image.ndim == 3 and image.shape[0] == 1:
            image, mask = image[0], mask[0]
        spacing = image_itk.GetSpacing()[:2][::-1]
        quality = "Unknown"
        cfg = target.with_name(f"Info_{view}.cfg")
        if cfg.exists():
            found = re.search(r"ImageQuality\s*[:=]\s*(\w+)", cfg.read_text())
            if found:
                quality = found[1].capitalize()
        cases.append(save_case(out, patient, view, phase, image, mask, spacing, quality,
                               image_source=image_path.relative_to(raw).as_posix(),
                               mask_source=target.relative_to(raw).as_posix(),
                               image_source_sha256=sha256(image_path),
                               mask_source_sha256=sha256(target)))
    path = out / "manifest.json"
    write_json(path, dict(kind="camus", source_tag=source_tag,
                         label_map={"0": "background", "1": "lv_cavity", "2": "myocardium",
                                    "3": "left_atrium"}, cases=cases))
    load_manifest(path)
    return path


def make_split(manifest_path, out, seed=42):
    manifest = load_manifest(manifest_path)
    patients = sorted({c["patient"] for c in manifest["cases"]})
    if len(patients) < 7:
        raise ValueError("At least seven patients required for a 70/15/15 split")
    np.random.default_rng(seed).shuffle(patients)
    train_end, val_end = int(.70 * len(patients)), int(.85 * len(patients))
    split = dict(protocol="custom_patient_70_15_15_v1", seed=seed,
                 manifest_sha256=sha256(manifest_path), train=sorted(patients[:train_end]),
                 val=sorted(patients[train_end:val_end]), test=sorted(patients[val_end:]))
    validate_split(manifest, split)
    write_json(out, split)
    return split


def official_split(manifest_path, directory, out):
    """Use the publisher's explicit membership, including the corrected public split."""
    manifest = load_manifest(manifest_path)
    directory = Path(directory)
    files = {"train": "subgroup_training.txt", "val": "subgroup_validation.txt",
             "test": "subgroup_testing.txt"}
    split = dict(protocol="camus_public_publisher_400_50_50_v1",
                 manifest_sha256=sha256(manifest_path), source_files={})
    for partition, name in files.items():
        path = directory / name
        ids = path.read_text().split()
        if not all(re.fullmatch(r"patient\d{4}", patient) for patient in ids):
            raise ValueError(f"Invalid patient IDs in {path}")
        split[partition] = sorted(ids)
        split["source_files"][name] = sha256(path)
    if [len(split[p]) for p in ("train", "val", "test")] != [400, 50, 50]:
        raise ValueError("Expected the publisher's 400/50/50 membership")
    validate_split(manifest, split)
    write_json(out, split)
    return split


def synthetic(out, patients=12, size=64, seed=7):
    """Geometric fixtures only: not ultrasound images or patient observations."""
    if patients < 7 or size < 32:
        raise ValueError("Synthetic demo requires >=7 patients and size >=32")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    rng, cases = np.random.default_rng(seed), []
    yy, xx = np.mgrid[:size, :size] / size
    for index in range(patients):
        for view in ("2CH", "4CH"):
            for phase in ("ED", "ES"):
                cx, cy = .5 + rng.uniform(-.04, .04), .55 + rng.uniform(-.03, .03)
                radius = .18 if phase == "ED" else .13
                rr = ((xx-cx)/radius)**2 + ((yy-cy)/(radius*1.4))**2
                mask = np.zeros((size, size), dtype=np.uint8)
                mask[rr < 1.6] = 2
                mask[rr < 1] = 1
                mask[((xx-cx)/.15)**2 + ((yy-.18)/.12)**2 < 1] = 3
                image = np.clip(np.array([.08, .25, .75, .45])[mask]
                                + rng.normal(0, .08, mask.shape), 0, 1).astype(np.float32)
                cases.append(save_case(out, f"synthetic{index:04d}", view, phase,
                                       image, mask, (.3, .2), "Synthetic"))
    path = out / "manifest.json"
    write_json(path, dict(kind="synthetic", source_tag="geometric_fixture_v1", cases=cases))
    return path
