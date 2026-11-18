"""Run the frozen improved configuration on a 2D/singleton-z CAMUS NIfTI frame."""
import argparse
import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

from camus_seg.common import read_json, sha256, write_json
from camus_seg.enhanced import clean_components, letterbox_input, native_probabilities
from camus_seg.experiment import code_sha256
from camus_seg.improve import load_member


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "image", "out"):
        parser.add_argument("--"+name, required=True, type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Output already exists")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    config = read_json(args.run / "selected_config.json")
    plan = read_json(args.run / "run-plan.json")
    if not config["frozen_before_test"] or config["selection_partition"] != "val":
        raise ValueError("Expected a frozen validation-selected configuration")
    if sha256(args.run / "run-plan.json") != config["plan_sha256"]:
        raise ValueError("Plan changed")
    original = sitk.ReadImage(str(args.image))
    image = sitk.GetArrayFromImage(original)
    singleton = image.ndim == 3 and image.shape[0] == 1
    if singleton:
        image = image[0]
    if image.ndim != 2 or min(image.shape) < 2 or not np.isfinite(image).all():
        raise ValueError("Expected a finite 2D frame or singleton-z image")
    tensor = letterbox_input(image, plan["training"]["size"])
    probabilities, member_hashes = [], []
    for index in config["members"]:
        member = config["checkpoints"][index]
        checkpoint = (args.run / member["path"]).resolve()
        if not checkpoint.is_relative_to(args.run.resolve()) or sha256(checkpoint) != member["sha256"]:
            raise ValueError("Checkpoint path or fingerprint changed")
        identity = dict(config, plan_sha256=member.get("training_plan_sha256", config["plan_sha256"]))
        model, _ = load_member(checkpoint, identity, args.device)
        probabilities.append(native_probabilities(model, tensor, image.shape, args.device,
                                                   plan["training"]["amp"]))
        member_hashes.append(member["sha256"])
    prediction = torch.stack(probabilities).mean(0).argmax(0).cpu().numpy().astype(np.uint8)
    if config["cleanup"]:
        prediction = clean_components(prediction)
    output = sitk.GetImageFromArray(prediction[None] if singleton else prediction)
    output.CopyInformation(original)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(output, str(args.out))
    write_json(str(args.out)+".json", dict(configuration_id=config["id"],
               configuration_sha256=sha256(args.run / "selected_config.json"),
               checkpoint_sha256=member_hashes, input_sha256=sha256(args.image),
               output_sha256=sha256(args.out), package_code_sha256=code_sha256(),
               device=args.device, label_map={0:"background", 1:"lv_cavity", 2:"myocardium", 3:"left_atrium"}))
    print(f"Saved native-grid research mask: {args.out}")


if __name__ == "__main__":
    main()
