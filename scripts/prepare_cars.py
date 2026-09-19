"""Locate and verify a Stanford Cars ImageFolder download.

Stanford Cars is not redistributed here.  Download the "Stanford Car Dataset
by classes folder" mirror (or the official devkit rearranged the same way)
into any directory; this script finds the ``train`` and ``test``
class-folder splits inside it, symlinks them to ``--output/train`` and
``--output/test``, and checks that the training split's file manifest
matches the one used for the paper.

    python scripts/prepare_cars.py --root <downloaded folder> --output data/cars
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

EXPECTED_TRAIN, EXPECTED_TEST = 8144, 8041
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_split(root: Path, split: str, expected: int) -> Path:
    candidates = [
        root / split, root / "car_data" / split, root / "car_data" / "car_data" / split,
        root / "stanford-cars-dataset" / split, root / "stanford-cars-dataset" / "car_data" / split,
        root / "stanford-cars-dataset" / "car_data" / "car_data" / split,
    ]
    matches = []
    for candidate in dict.fromkeys(path.resolve() for path in candidates if path.is_dir()):
        classes = [p for p in candidate.iterdir() if p.is_dir()]
        count = sum(1 for p in candidate.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        if len(classes) == 196 and count == expected:
            matches.append(candidate)
    if len(matches) != 1:
        raise SystemExit(f"expected exactly one {split} split with {expected} images under {root}; found {matches}")
    return matches[0]


def manifest_sha256(split_dir: Path) -> str:
    manifest = []
    for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        for image in sorted(p for p in class_dir.rglob("*") if p.is_file()):
            manifest.append([str(image.relative_to(split_dir)).replace("\\", "/"), class_dir.name])
    classes = sorted({name for _, name in manifest})
    index = {name: i for i, name in enumerate(classes)}
    manifest = [[path, index[name]] for path, name in manifest]
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest(), classes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, help="directory containing the downloaded Stanford Cars data")
    parser.add_argument("--output", required=True, help="directory to receive train/ and test/ symlinks")
    args = parser.parse_args()

    root, output = Path(args.root).resolve(), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    expected = json.loads((Path(__file__).resolve().parents[1] / "orders" / "cars_classes.json").read_text())

    train_dir = find_split(root, "train", EXPECTED_TRAIN)
    test_dir = find_split(root, "test", EXPECTED_TEST)
    digest, classes = manifest_sha256(train_dir)
    if classes != expected["classes"]:
        raise SystemExit("the class-name list does not match the one used for the paper")
    if digest != expected["train_manifest_sha256"]:
        print(f"WARNING: training manifest hash {digest} does not match {expected['train_manifest_sha256']}")
        print("This usually means a different image re-encoding of the same photos; features may differ slightly.")
    else:
        print("training manifest matches the one used for the paper")

    for split, source in (("train", train_dir), ("test", test_dir)):
        destination = output / split
        if destination.exists() or destination.is_symlink():
            print(f"{destination} already exists; leaving it in place")
            continue
        try:
            destination.symlink_to(source, target_is_directory=True)
        except OSError:
            import shutil
            shutil.copytree(source, destination)
        print(f"{destination} -> {source}")


if __name__ == "__main__":
    main()
