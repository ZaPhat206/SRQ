"""Persistent-state accounting used by the paper tables.

The learner remains the source of truth: accounting sums its
``persistent_tensors()``.  Exact upper-triangle accounting changes only the
reported byte convention for the dense Gram tensor and leaves the solution
unchanged.
"""

from __future__ import annotations

import torch

from .ridge import RidgeLearner, tensor_bytes


def strict_lower_bytes(dimension: int, *, dtype: torch.dtype = torch.float32) -> int:
    """Bytes occupied by the strict lower triangle of a dense square tensor."""
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    return (dimension * (dimension - 1) // 2) * torch.empty((), dtype=dtype).element_size()


def learner_state_bytes(learner: RidgeLearner, *, exact_upper_triangle: bool = False) -> int:
    """Count the actual persistent learner tensors.

    ``exact_upper_triangle`` is valid only for :class:`ExactRidge`; it removes
    the dense Gram tensor's strict lower triangle from the reported accounting.
    """
    tensors = learner.persistent_tensors()
    total = sum(tensor_bytes(tensor) for tensor in tensors.values())
    if exact_upper_triangle:
        if "A" not in tensors or tensors["A"].ndim != 2 or tensors["A"].shape[0] != tensors["A"].shape[1]:
            raise ValueError("upper-triangle accounting requires a dense Exact Gram tensor named A")
        total -= strict_lower_bytes(tensors["A"].shape[0], dtype=tensors["A"].dtype)
    return total


def persistent_state_bytes(
    learner: RidgeLearner,
    *,
    projection: torch.Tensor | None = None,
    exact_upper_triangle: bool = False,
) -> int:
    """Count projection plus the learner's actual persistent state."""
    total = learner_state_bytes(learner, exact_upper_triangle=exact_upper_triangle)
    if projection is not None:
        total += tensor_bytes(projection)
    return total


def reported_exact_state_bytes(dense_state_bytes: int, dimension: int, *, dtype: torch.dtype = torch.float32) -> int:
    """Convert a dense Exact state total to lossless upper-triangle accounting."""
    return int(dense_state_bytes) - strict_lower_bytes(dimension, dtype=dtype)


__all__ = [
    "learner_state_bytes",
    "persistent_state_bytes",
    "reported_exact_state_bytes",
    "strict_lower_bytes",
]
