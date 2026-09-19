"""Extract frozen backbone features into a feature cache.

    python scripts/extract_features.py --dataset cifar100 --root data/cifar100 \
        --checkpoint checkpoints/vit_b16.safetensors --output features/cifar100

The rows follow the sample order of ``orders/<dataset>.json`` (the order of the
caches used for the paper); Stanford Cars follows the dataset order.  ``--split``
restricts the work to one split, which is how the test features were kept out
of the way while the ridge coefficients were selected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import environment, write_json  # noqa: E402
from experiments.data import BACKBONES, DATASETS, extract_features, image_dataset, sha256_file  # noqa: E402


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def manifest_sha256(dataset: str, root: str, split: str) -> str:
    """Hash of the (relative path, label) list of an ImageFolder split."""
    source = image_dataset(dataset, root, split, with_transform=False)
    folder = Path(root) / split
    manifest = [[str(Path(path).relative_to(folder)).replace("\\", "/"), int(label)] for path, label in source.samples]
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--root", required=True, help="dataset root")
    parser.add_argument("--checkpoint", required=True, help="backbone weights")
    parser.add_argument("--output", required=True, help="feature cache directory")
    parser.add_argument("--split", choices=("train", "test", "both"), default="both")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    spec = DATASETS[args.dataset]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    metadata_path = output / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {
        "dataset": args.dataset, "backbone": spec["backbone"],
        "backbone_sha256": sha256_file(args.checkpoint), "backbone_source": BACKBONES[spec["backbone"]]["source"],
        "batch_size": args.batch_size,
    }
    if metadata["backbone_sha256"] != sha256_file(args.checkpoint):
        raise SystemExit("this cache was written with different backbone weights")
    splits = ("train", "test") if args.split == "both" else (args.split,)
    for split in splits:
        path = output / f"{split}.pt"
        if path.is_file():
            print(f"{path} exists; skipping")
            continue
        print(f"extracting {args.dataset} {split} ({spec[split]:,} images) on {args.device}", flush=True)
        packed = extract_features(args.dataset, args.root, split, args.checkpoint, device=args.device,
                                  batch_size=args.batch_size, num_workers=args.num_workers)
        torch.save(packed, path)
        metadata[f"{split}_features_sha256"] = tensor_sha256(packed["features"])
        metadata[f"{split}_labels_sha256"] = tensor_sha256(packed["labels"])
        if args.dataset != "cifar100":
            metadata[f"{split}_manifest_sha256"] = manifest_sha256(args.dataset, args.root, split)
        metadata["environment"] = environment(torch.device(args.device))
        write_json(metadata_path, metadata)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
