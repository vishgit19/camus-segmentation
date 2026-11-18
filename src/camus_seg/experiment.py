import hashlib
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .common import load_manifest, read_json, sha256, validate_split, write_json
from .data import load_case
from .model import UNet, input_tensor, predict_tensor, segmentation_loss


def code_sha256():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode() + path.read_bytes())
    return digest.hexdigest()


def experiment_data(manifest_path, split_path):
    manifest, split = load_manifest(manifest_path), read_json(split_path)
    if split["manifest_sha256"] != sha256(manifest_path):
        raise ValueError("Manifest differs from the frozen split")
    validate_split(manifest, split)
    return manifest, split


class Frames(Dataset):
    def __init__(self, manifest_path, cases, size):
        self.samples = []
        for case in cases:
            image, mask, _ = load_case(manifest_path, case)
            target = F.interpolate(torch.from_numpy(mask)[None, None].float(),
                                   (size, size), mode="nearest")[0, 0].long()
            self.samples.append((input_tensor(image, size), target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def train(manifest_path, split_path, out, *, epochs=50, size=256, width=16,
          batch_size=8, lr=.001, seed=42, device="cpu", threads=2):
    if min(epochs, batch_size, width, threads) < 1 or size < 16 or size % 4 or lr <= 0:
        raise ValueError("Invalid training settings")
    manifest, split = experiment_data(manifest_path, split_path)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
    model = UNet(width).to(device)
    training = [c for c in manifest["cases"] if c["patient"] in split["train"]]
    validation = [c for c in manifest["cases"] if c["patient"] in split["val"]]
    loader = DataLoader(Frames(manifest_path, training, size), batch_size=batch_size,
                        shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(seed))
    validation_data = []
    for case in validation:
        image, target, _ = load_case(manifest_path, case)
        validation_data.append((input_tensor(image, size), target.astype(np.uint8)))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    settings = dict(epochs=epochs, size=size, width=width, batch_size=batch_size,
                    lr=lr, seed=seed, device=device, threads=threads)
    provenance = dict(kind=manifest["kind"], source_tag=manifest["source_tag"],
                      manifest_sha256=sha256(manifest_path), split_sha256=sha256(split_path),
                      train_patients=split["train"], validation_patients=split["val"],
                      settings=settings, code_sha256=code_sha256(),
                      environment=dict(python=platform.python_version(),
                                       torch=str(torch.__version__), numpy=np.__version__,
                                       platform=platform.platform(),
                                       cuda=torch.version.cuda,
                                       gpu=torch.cuda.get_device_name() if device == "cuda" else None,
                                       deterministic_algorithms=True))
    write_json(out / "provenance.json", provenance)
    best, history = -1.0, []
    for epoch in range(1, epochs+1):
        started = time.perf_counter()
        model.train()
        total_loss = 0.0
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = segmentation_loss(model(images), masks)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(images)
        # Native-grid Dice; every validation patient has four frames and three structures.
        scores = []
        for tensor, target in validation_data:
            prediction = predict_tensor(model, tensor, target.shape, device)
            for label in (1, 2, 3):
                p, t = prediction == label, target == label
                denominator = p.sum() + t.sum()
                scores.append(float(2*(p & t).sum()/denominator) if denominator else 1.0)
        score = float(np.mean(scores))
        if not np.isfinite(score) or not np.isfinite(total_loss):
            raise ValueError("Training produced non-finite results")
        history.append(dict(epoch=epoch, train_loss=total_loss/len(training),
                            val_native_macro_dice=score, seconds=time.perf_counter()-started))
        write_json(out / "history.json", history)
        if score > best:
            best = score
            torch.save(dict(state_dict=model.state_dict(), epoch=epoch,
                            validation_dice=score, provenance=provenance), out / "best.pt")
        print(f"epoch={epoch} loss={total_loss/len(training):.4f} val_dice={score:.4f}", flush=True)
    return out / "best.pt"
