"""Datasets, frozen backbones and feature caches.

A feature cache is a directory with ``train.pt`` and ``test.pt``, each a dict
``{"features": FloatTensor[n, d], "labels": LongTensor[n]}``.  The rows of the
ViT-B/16 caches follow the sample order recorded in ``orders/<dataset>.json``,
which is the order of the caches used for the paper (see "Data and checkpoints"
in the README); the Stanford Cars caches follow the dataset order.  Samples are
embedded in batches of 128 within every group of that file, exactly as in the
original runs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "cifar100": {"num_classes": 100, "train": 50000, "test": 10000, "backbone": "vit_b16", "feature_dim": 768},
    "cub200": {"num_classes": 200, "train": 5994, "test": 5794, "backbone": "vit_b16", "feature_dim": 768},
    "cars": {"num_classes": 196, "train": 8144, "test": 8041, "backbone": "resnet50", "feature_dim": 2048},
}

BACKBONES = {
    "vit_b16": {
        "source": "https://huggingface.co/timm/vit_base_patch16_224.augreg2_in21k_ft_in1k (model.safetensors)",
        "timm_name": "vit_base_patch16_224",
        "size": 346284714,
        "sha256": "32aa17d6e17b43500f531d5f6dc9bc93e56ed8841b8a75682e1bb295d722405b",
    },
    "resnet50": {
        "source": "https://download.pytorch.org/models/resnet50-11ad3fa6.pth (torchvision IMAGENET1K_V2)",
        "size": 102540417,
        "sha256": "11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca",
    },
}

# Identity of the prepared CUB-200-2011 folders (see scripts/prepare_cub.py).
CUB_IDENTITY_SHA256 = "e374af9b576cb6b3503198ef3ea30fd0aa9d2e18c230ff8064e21d4f644af2ca"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def transform(dataset: str):
    from torchvision import transforms

    if dataset == "cars":
        return transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    # ViT-B/16: CIFAR-100 images are resized to 224 directly, CUB-200 images to 256 first.
    size = 224 if dataset == "cifar100" else 256
    return transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def image_dataset(dataset: str, root: str | Path, split: str, with_transform: bool = True):
    """The torchvision dataset whose indices the order files refer to."""
    from torchvision import datasets

    tf = transform(dataset) if with_transform else None
    root = Path(root)
    if dataset == "cifar100":
        return datasets.CIFAR100(root=str(root), train=split == "train", download=False, transform=tf)
    if dataset in ("cub200", "cars"):
        folder = root / split
        if not folder.is_dir():
            raise FileNotFoundError(f"expected an ImageFolder split at {folder}")
        return datasets.ImageFolder(str(folder), transform=tf)
    raise ValueError(f"unknown dataset {dataset}")


def targets(dataset) -> list[int]:
    return [int(value) for value in dataset.targets]


def load_order(dataset: str, split: str) -> list[list[int]]:
    """Groups of dataset indices; rows of the cache are their concatenation."""
    if dataset == "cars":
        return [list(range(DATASETS["cars"][split]))]
    payload = json.loads((ROOT / "orders" / f"{dataset}.json").read_text(encoding="utf-8"))
    return payload[split]


def load_backbone(name: str, checkpoint: str | Path, device: torch.device) -> torch.nn.Module:
    spec = BACKBONES[name]
    path = Path(checkpoint)
    if path.stat().st_size != spec["size"] or sha256_file(path) != spec["sha256"]:
        raise ValueError(f"checkpoint {path} does not match the expected {name} weights")
    if name == "vit_b16":
        import timm
        from safetensors.torch import load_file

        model = timm.create_model(spec["timm_name"], pretrained=False)
        model.load_state_dict(load_file(str(path)), strict=True)
        model.reset_classifier(0)
    else:
        from torchvision.models import resnet50

        model = resnet50(weights=None)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        model.fc = torch.nn.Identity()
    return model.eval().to(device)


@torch.no_grad()
def extract_features(dataset: str, root: str | Path, split: str, checkpoint: str | Path, *,
                     device: str = "cuda", batch_size: int = 128, num_workers: int = 2) -> dict:
    """Embed one split in the recorded sample order."""
    from torch.utils.data import DataLoader, Subset

    spec = DATASETS[dataset]
    device = torch.device(device)
    # The ViT caches were extracted with cuDNN disabled, the ResNet-50 caches with the defaults.
    vit = spec["backbone"] == "vit_b16"
    torch.backends.cudnn.deterministic = vit
    torch.backends.cudnn.enabled = not vit
    source = image_dataset(dataset, root, split)
    if len(source) != spec[split]:
        raise ValueError(f"{dataset}/{split} has {len(source)} images, expected {spec[split]}")
    model = load_backbone(spec["backbone"], checkpoint, device)
    features, labels = [], []
    for group in load_order(dataset, split):
        loader = DataLoader(Subset(source, group), batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=device.type == "cuda")
        for images, target in loader:
            features.append(model(images.to(device)).detach().cpu())
            labels.append(target.cpu().long())
    packed = {"features": torch.cat(features), "labels": torch.cat(labels)}
    if packed["features"].shape != (spec[split], spec["feature_dim"]) or not bool(torch.isfinite(packed["features"]).all()):
        raise RuntimeError("invalid extracted features")
    return packed


def load_features(cache_dir: str | Path, dataset: str, split: str) -> dict:
    payload = torch.load(Path(cache_dir) / f"{split}.pt", map_location="cpu", weights_only=True)
    spec = DATASETS[dataset]
    if set(payload) != {"features", "labels"} or payload["features"].shape != (spec[split], spec["feature_dim"]):
        raise ValueError(f"{cache_dir}/{split}.pt is not a {dataset} {split} feature cache")
    return payload
