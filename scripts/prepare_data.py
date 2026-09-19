"""Fetch a backbone, prepare a dataset and extract its feature cache.

    python scripts/prepare_data.py cifar100
    python scripts/prepare_data.py cub200 --archive CUB_200_2011.tgz
    python scripts/prepare_data.py cars --root stanford-cars-download

CIFAR-100 is downloaded through torchvision.  CUB-200-2011 and Stanford Cars
are not redistributable: download them yourself and pass the official archive
or the extracted folder.  Checkpoints are fetched once into ``checkpoints/``
and verified against the SHA-256 recorded in ``experiments/data.py``.  A step
whose output already exists is reported and skipped, so the command can be run
again after an interruption.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiments.data import BACKBONES, DATASETS, sha256_file  # noqa: E402
from reproduce import resolve_device  # noqa: E402

# Where each backbone comes from.  The digests live in experiments/data.py.
CHECKPOINTS = {
    "vit_b16": {
        "path": "checkpoints/vit_b16.safetensors",
        "repository": "timm/vit_base_patch16_224.augreg2_in21k_ft_in1k",
        "filename": "model.safetensors",
    },
    "resnet50": {
        "path": "checkpoints/resnet50.pth",
        "url": "https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
    },
}


def verify(path: Path, backbone: str) -> None:
    spec = BACKBONES[backbone]
    size = path.stat().st_size
    if size != spec["size"]:
        raise SystemExit(f"{path} is {size} bytes, expected {spec['size']}")
    digest = sha256_file(path)
    if digest != spec["sha256"]:
        raise SystemExit(f"{path} has SHA-256 {digest}, expected {spec['sha256']}")
    print(f"verified {path}", flush=True)


def fetch_checkpoint(backbone: str) -> Path:
    source = CHECKPOINTS[backbone]
    path = ROOT / source["path"]
    if path.is_file():
        verify(path, backbone)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    if "repository" in source:
        from huggingface_hub import hf_hub_download

        print(f"downloading {source['repository']}/{source['filename']}", flush=True)
        shutil.copy(hf_hub_download(source["repository"], source["filename"]), path)
    else:
        print(f"downloading {source['url']}", flush=True)
        urllib.request.urlretrieve(source["url"], path)
    verify(path, backbone)
    return path


def run(argv: list[str]) -> None:
    print("$ " + subprocess.list2cmdline(argv), flush=True)
    code = subprocess.call(argv)
    if code:
        raise SystemExit(code)


def prepare_cifar100(data: Path) -> None:
    from torchvision.datasets import CIFAR100

    for train in (True, False):
        CIFAR100(root=str(data), train=train, download=True)


def prepare_cub200(data: Path, archive: Path | None) -> None:
    if (data / "train").is_dir() and (data / "test").is_dir():
        print(f"{data} already prepared", flush=True)
    else:
        if archive is None:
            raise SystemExit("CUB-200-2011 is not redistributed: pass --archive CUB_200_2011.tgz")
        run([sys.executable, str(ROOT / "scripts" / "prepare_cub.py"),
             "--archive", str(archive), "--output", str(data)])
    run([sys.executable, str(ROOT / "scripts" / "prepare_cub.py"), "--verify", str(data)])


def prepare_cars(data: Path, source: Path | None) -> None:
    if source is None:
        raise SystemExit("Stanford Cars is not redistributed: pass --root <downloaded folder>")
    run([sys.executable, str(ROOT / "scripts" / "prepare_cars.py"),
         "--root", str(source), "--output", str(data)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=sorted(DATASETS))
    parser.add_argument("--archive", type=Path, help="official CUB_200_2011.tgz")
    parser.add_argument("--root", type=Path, help="downloaded Stanford Cars folder")
    parser.add_argument("--data", type=Path, default=Path("data"), help="root of the prepared datasets")
    parser.add_argument("--features", type=Path, default=Path("features"), help="root of the feature caches")
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:N or cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    cache = args.features / args.dataset
    if all((cache / f"{split}.pt").is_file() for split in ("train", "test")):
        print(f"{cache} already holds both splits; nothing to do", flush=True)
        return

    data = args.data / args.dataset
    if args.dataset == "cifar100":
        prepare_cifar100(data)
    elif args.dataset == "cub200":
        prepare_cub200(data, args.archive)
    else:
        prepare_cars(data, args.root)

    checkpoint = fetch_checkpoint(DATASETS[args.dataset]["backbone"])
    run([sys.executable, str(ROOT / "scripts" / "extract_features.py"),
         "--dataset", args.dataset, "--root", str(data), "--checkpoint", str(checkpoint),
         "--output", str(cache), "--device", resolve_device(args.device),
         "--batch-size", str(args.batch_size), "--num-workers", str(args.num_workers)])
    print(f"feature cache ready: {cache}", flush=True)


if __name__ == "__main__":
    main()
