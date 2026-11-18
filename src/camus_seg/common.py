import hashlib
import json
from pathlib import Path

LABELS = {1: "lv_cavity", 2: "myocardium", 3: "left_atrium"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_manifest(path):
    manifest = read_json(path)
    cases = manifest["cases"]
    if not cases or manifest["kind"] not in ("camus", "synthetic"):
        raise ValueError("Expected a nonempty CAMUS or synthetic manifest")
    keys = [(c["patient"], c["view"], c["phase"]) for c in cases]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate patient/view/phase; check for colliding release IDs")
    expected = {(v, p) for v in ("2CH", "4CH") for p in ("ED", "ES")}
    for patient in {c["patient"] for c in cases}:
        if {(c["view"], c["phase"]) for c in cases if c["patient"] == patient} != expected:
            raise ValueError(f"Incomplete ED/ES and 2CH/4CH coverage: {patient}")
    return manifest


def validate_split(manifest, split):
    parts = [split[key] for key in ("train", "val", "test")]
    if any(not part or len(part) != len(set(part)) for part in parts):
        raise ValueError("Every split must be nonempty and contain unique patient IDs")
    if any(set(parts[i]) & set(parts[j]) for i in range(3) for j in range(i)):
        raise ValueError("Patient leakage between splits")
    if set().union(*map(set, parts)) != {c["patient"] for c in manifest["cases"]}:
        raise ValueError("Split must cover exactly the manifest's patients")
    partition = {patient: key for key in ("train", "val", "test") for patient in split[key]}
    seen = {}
    for case in manifest["cases"]:
        digest, current = case["image_sha256"], partition[case["patient"]]
        if digest in seen and seen[digest] != current:
            raise ValueError("Identical image pixels occur across partitions")
        seen[digest] = current
