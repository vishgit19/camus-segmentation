"""Download only annotated ED/ES frames from the official CAMUS Girder deposit."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.request import urlopen

BASE = "https://humanheart-project.creatis.insa-lyon.fr/database/api/v1/"
COLLECTION = "6373703d73e9f0047faa1bc8"
NIFTI = "63fde55f73e9f004868fb7ac"
SPLIT = "66e27d12961576b1bad4e4e1"


def request(route):
    for attempt in range(4):
        try:
            with urlopen(BASE + route, timeout=90) as response:
                return response.read()
        except OSError:
            if attempt == 3:
                raise
            time.sleep(attempt + 1)


def items(folder):
    return json.loads(request(f"item?folderId={folder}&limit=100"))


def download_item(item, directory, root):
    name = item["name"]
    if Path(name).name != name:
        raise ValueError("Unexpected source filename")
    path = directory / name
    if not path.exists() or path.stat().st_size != item["size"]:
        content = request(f"item/{item['_id']}/download")
        if len(content) != item["size"]:
            raise ValueError(f"Source size mismatch: {name}")
        path.write_bytes(content)
    return dict(file=path.relative_to(root).as_posix(), item_id=item["_id"],
                size=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.workers <= 12:
        raise ValueError("Use between 1 and 12 workers")
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    (root / "database_nifti").mkdir(exist_ok=True)
    (root / "database_split").mkdir(exist_ok=True)
    records = []
    for folder, directory in ((NIFTI, "database_nifti"), (SPLIT, "database_split")):
        for item in items(folder):
            records.append(download_item(item, root / directory, root))
    print((root / "database_nifti/LICENSE_TERMS.md").read_text(), flush=True)
    folders = json.loads(request(f"folder?parentType=folder&parentId={NIFTI}&limit=1000"))
    folders = [f for f in folders if re.fullmatch(r"patient\d{4}", f["name"])]
    if len(folders) != 500:
        raise ValueError(f"Expected 500 official patient folders, found {len(folders)}")

    def patient(folder):
        directory = root / "database_nifti" / folder["name"]
        directory.mkdir(exist_ok=True)
        chosen = [item for item in items(folder["_id"])
                  if re.fullmatch(r"patient\d{4}_(2CH|4CH)_(ED|ES)(_gt)?\.nii\.gz", item["name"])
                  or item["name"] in {"Info_2CH.cfg", "Info_4CH.cfg", "MANDATORY_CITATION.md"}]
        if len(chosen) != 11:
            raise ValueError(f"Unexpected source inventory for {folder['name']}: {len(chosen)}")
        return [download_item(item, directory, root) for item in chosen]

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(patient, folder) for folder in folders]
        for index, future in enumerate(as_completed(futures), 1):
            records.extend(future.result())
            if index % 10 == 0:
                print(f"Downloaded {index}/500 patients | {time.monotonic()-started:.0f}s", flush=True)
    manifest = dict(collection_id=COLLECTION, nifti_folder_id=NIFTI, split_folder_id=SPLIT,
                    retrieved_utc=datetime.now(timezone.utc).isoformat(),
                    selection="ED/ES 2CH/4CH paired frames plus metadata; no sequences",
                    files=sorted(records, key=lambda r: r["file"]))
    (root / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Finished: {len(records)} files, {sum(r['size'] for r in records)} bytes", flush=True)


if __name__ == "__main__":
    main()
