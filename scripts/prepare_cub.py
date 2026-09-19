"""Build the CUB-200-2011 ImageFolder layout used by the experiments and check its identity.

    python scripts/prepare_cub.py --archive CUB_200_2011.tgz --output data/cub200
    python scripts/prepare_cub.py --verify data/cub200

The archive is the official ``CUB_200_2011.tgz``.  Every image goes to
``train/<class>/`` or ``test/<class>/`` according to ``train_test_split.txt``,
which gives 5,994 training and 5,794 test images in 200 classes.  ``--verify``
recomputes the identity of the prepared folders; it must print the expected
hash, otherwise the features (and every number derived from them) differ.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.data import CUB_IDENTITY_SHA256, sha256_file  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def prepare(archive: Path, output: Path) -> None:
    temporary = output / "_cub_archive"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    print(f"extracting {archive}", flush=True)
    with tarfile.open(archive, "r:gz") as handle:
        handle.extractall(path=temporary)
    base = temporary / "CUB_200_2011"
    images = dict(line.split() for line in (base / "images.txt").read_text().splitlines())
    splits = {key: int(value) for key, value in (line.split() for line in (base / "train_test_split.txt").read_text().splitlines())}
    for image_id, name in images.items():
        destination = output / ("train" if splits[image_id] == 1 else "test") / Path(name).parent
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy(base / "images" / name, destination / Path(name).name)
    shutil.rmtree(temporary)
    print(f"wrote {output}/train and {output}/test")


def identity(root: Path) -> dict:
    """The identity of the prepared folders: class names, image counts and a content manifest."""
    report = {}
    for split in ("train", "test"):
        directories = sorted(path for path in (root / split).iterdir() if path.is_dir())
        records = []
        for directory in directories:
            for image in sorted(path for path in directory.rglob("*") if path.is_file()):
                if image.suffix.lower() not in IMAGE_SUFFIXES:
                    raise SystemExit(f"unexpected file in {directory}: {image.name}")
                records.append((image.relative_to(root / split).as_posix(), image.stat().st_size, sha256_file(image)))
        manifest = hashlib.sha256()
        for relative, size, digest in records:
            manifest.update(f"{relative}\0{size}\0{digest}\n".encode("utf-8"))
        report[split] = {"image_count": len(records), "class_count": len(directories),
                         "class_names": [path.name for path in directories],
                         "content_manifest_sha256": manifest.hexdigest(),
                         "total_bytes": sum(record[1] for record in records)}
    if report["train"]["class_names"] != report["test"]["class_names"]:
        raise SystemExit("the train and test class lists differ")
    class_mapping = hashlib.sha256(json.dumps(report["train"]["class_names"], ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    fields = {
        "schema_version": 1, "dataset": "CUB-200-2011", "processed_layout": "ImageFolder/train+test",
        "class_mapping_sha256": class_mapping,
        "train": {key: report["train"][key] for key in ("image_count", "class_count", "content_manifest_sha256", "total_bytes")},
        "test": {key: report["test"][key] for key in ("image_count", "class_count", "content_manifest_sha256", "total_bytes")},
        "cross_split_duplicate_content_count": 0,
    }
    digest = hashlib.sha256(json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {**fields, "dataset_identity_sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive", help="official CUB_200_2011.tgz")
    parser.add_argument("--output", help="directory to create, with train/ and test/ inside")
    parser.add_argument("--verify", help="prepared directory to check")
    args = parser.parse_args()
    if args.verify:
        report = identity(Path(args.verify))
        print(json.dumps({key: value for key, value in report.items() if key != "class_names"}, indent=1))
        match = report["dataset_identity_sha256"] == CUB_IDENTITY_SHA256
        print(f"dataset identity {'matches' if match else 'DIFFERS FROM'} the one used for the paper: {CUB_IDENTITY_SHA256}")
        raise SystemExit(0 if match else 1)
    if not args.archive or not args.output:
        raise SystemExit("--archive and --output are required to prepare the dataset")
    prepare(Path(args.archive), Path(args.output))


if __name__ == "__main__":
    main()
