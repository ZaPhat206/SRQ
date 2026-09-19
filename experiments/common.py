"""Shared helpers: configuration, result files, statistics, environment."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: dict) -> None:
    """Atomic JSON write (a partial file never replaces a complete one)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def environment(device: torch.device) -> dict:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
    }


def lock_precision() -> None:
    """Disable TF32 so that newer GPUs use the same FP32 arithmetic as the T4 of the paper."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def mean_std(values: list[float]) -> dict:
    """Mean and sample standard deviation."""
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="experiment configuration (configs/*.json)")
    parser.add_argument("--features", required=True, help="feature cache directory with train.pt (and test.pt)")
    parser.add_argument("--output", required=True, help="result directory")
    parser.add_argument("--device", default="cuda")
    return parser


def finite_or_none(value: float) -> float | None:
    return value if isinstance(value, (int, float)) and math.isfinite(value) else None
