import argparse
from pathlib import Path

from .data import make_split, official_split, prepare, synthetic


def main():
    parser = argparse.ArgumentParser(description="CAMUS U-Net baseline and patient-level evaluation")
    sub = parser.add_subparsers(dest="command", required=True)
    index = sub.add_parser("prepare", help="Prepare one locally acquired labelled CAMUS release")
    index.add_argument("--raw", required=True)
    index.add_argument("--out", required=True)
    index.add_argument("--source-tag", required=True)
    split = sub.add_parser("split", help="Import publisher membership or freeze a patient split")
    split.add_argument("--manifest", required=True)
    split.add_argument("--out", required=True)
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--official-dir", help="Directory containing publisher subgroup_*.txt files")
    demo = sub.add_parser("demo", help="Run synthetic geometry through the complete pipeline")
    demo.add_argument("--out", required=True)
    for name in ("train", "evaluate"):
        command = sub.add_parser(name)
        command.add_argument("--manifest", required=True)
        command.add_argument("--split", required=True)
        command.add_argument("--out", required=True)
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
        command.add_argument("--threads", type=int, default=2)
        if name == "train":
            for arg, default in (("epochs", 50), ("size", 256), ("width", 16), ("batch-size", 8)):
                command.add_argument("--"+arg, type=int, default=default)
            command.add_argument("--lr", type=float, default=.001)
            command.add_argument("--patience", type=int, default=10,
                                 help="Stop after this many non-improving validation epochs; 0 disables")
            command.add_argument("--min-delta", type=float, default=0.)
        else:
            command.add_argument("--checkpoint", required=True)
            command.add_argument("--repetitions", type=int, default=2000)
    args = vars(parser.parse_args())
    name = args.pop("command")
    if name == "prepare":
        print(prepare(**args))
    elif name == "split":
        if args["official_dir"]:
            official_split(args["manifest"], args["official_dir"], args["out"])
        else:
            make_split(args["manifest"], args["out"], args["seed"])
    elif name == "demo":
        from .evaluate import evaluate
        from .experiment import train
        out = Path(args["out"])
        out.mkdir(parents=True, exist_ok=False)
        manifest = synthetic(out / "data")
        split_path = out / "split.json"
        make_split(manifest, split_path)
        checkpoint = train(manifest, split_path, out / "training", epochs=20,
                           size=64, width=4, batch_size=8, lr=.005)
        evaluate(manifest, split_path, checkpoint, out / "evaluation", repetitions=200)
        print(f"Synthetic software check complete: {out}. No CAMUS benchmark was run.")
    else:
        manifest, split_path, out = args.pop("manifest"), args.pop("split"), args.pop("out")
        if name == "train":
            from .experiment import train
            train(manifest, split_path, out, **args)
        else:
            from .evaluate import evaluate
            checkpoint = args.pop("checkpoint")
            evaluate(manifest, split_path, checkpoint, out, **args)


if __name__ == "__main__":
    main()
