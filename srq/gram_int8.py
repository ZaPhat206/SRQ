"""INT8 Gram baselines: quantize the Gram matrix instead of its factor.

At task ``k`` the learner adds ``Phi_k^T Phi_k`` to its decoded matrix and
quantizes the sum again with the codec of SRQ-INT8 (FP32 diagonal, INT8
strict upper triangle).  The quantized system need not be positive definite,
so the ridge solve can fail.  Two repaired variants add a diagonal load:

* ``load="certified"``: Weyl's inequality with ``||N_k||_2 <= ||N_k||_inf`` for
  the symmetric quantization error ``N_k`` gives the lower bound
  ``l_k = l_{k-1} - ||N_k||_inf`` on the smallest eigenvalue of the stored
  matrix.  The load ``delta_k = max{0, mu_k - (lambda + l_k)}`` is the
  smallest one that raises the certified floor above the FP32 margin
  ``mu_k = 8 eps m max{||diag||_inf, lambda, 1}``.
* ``load="minimal"``: the smallest load in ``{0} U {mu_k 2^(j/4)}`` for which the
  FP32 Cholesky factorization succeeds, found by doubling and bisection
  (assuming that success is monotone in the load).  It has no certificate.

Neither load uses labels or accuracy.
"""

from __future__ import annotations

import math

import torch

from .codec import CompressedFactor
from .ridge import RidgeLearner, cholesky_solve


def quantization_error_metrics(reconstructed: torch.Tensor, reference: torch.Tensor, *, row_chunk_size: int = 256) -> tuple[float, float]:
    """``||E||_inf`` and ``||E||_F / max(||reference||_F, 1)`` for ``E = reconstructed - reference``."""
    device = reconstructed.device
    infinity_norm = torch.zeros((), device=device, dtype=torch.float64)
    squared_error = torch.zeros((), device=device, dtype=torch.float64)
    squared_reference = torch.zeros((), device=device, dtype=torch.float64)
    for start in range(0, len(reference), row_chunk_size):
        end = min(start + row_chunk_size, len(reference))
        difference = reconstructed[start:end] - reference[start:end]
        infinity_norm = torch.maximum(infinity_norm, difference.abs().sum(1, dtype=torch.float64).amax())
        squared_error.add_(difference.square().sum(dtype=torch.float64))
        squared_reference.add_(reference[start:end].square().sum(dtype=torch.float64))
    relative = torch.sqrt(squared_error / torch.clamp(squared_reference, min=1.0))
    return float(infinity_norm.item()), float(relative.item())


def _cholesky_succeeds(system: torch.Tensor) -> bool:
    _, info = torch.linalg.cholesky_ex((system + system.T) * 0.5)
    return int(info.max().item()) == 0


def minimal_cholesky_load(matrix: torch.Tensor, *, ridge_lambda: float, base_load: float,
                          steps_per_doubling: int = 4, maximum_load: float) -> dict:
    """Smallest grid load for which ``matrix + (lambda + load) I`` factors; ``matrix`` is restored."""
    if base_load <= 0 or maximum_load <= base_load or steps_per_doubling <= 0:
        raise ValueError("invalid load grid")
    original = matrix.diagonal().clone()
    attempts = 0

    def grid(index: int) -> float:
        return float(base_load) * 2.0 ** (index / steps_per_doubling)

    def succeeds(load: float) -> bool:
        nonlocal attempts
        attempts += 1
        matrix.diagonal().copy_(original + (ridge_lambda + load))
        return _cholesky_succeeds(matrix)

    try:
        if succeeds(0.0):
            return {"load": 0.0, "grid_index": None, "attempts": attempts}
        failed, index = -1, 0
        while not succeeds(grid(index)):
            failed = index
            index += steps_per_doubling
            if grid(index) > maximum_load:
                raise RuntimeError("minimal-load search exceeded its maximum load")
        succeeded = index
        while succeeded - failed > 1:
            middle = (failed + succeeded) // 2
            if succeeds(grid(middle)):
                succeeded = middle
            else:
                failed = middle
        return {"load": grid(succeeded), "grid_index": succeeded, "attempts": attempts}
    finally:
        matrix.diagonal().copy_(original)


class Int8GramRidge(RidgeLearner):
    """INT8 Gram learner with ``load`` in ``{"none", "certified", "minimal"}``."""

    def __init__(self, *, load: str = "none", block_size: int = 256, group_size: int = 64,
                 margin_multiplier: float = 8.0, error_chunk_size: int = 256,
                 steps_per_doubling: int = 4, maximum_load_to_ridge_ratio: float = 1.0e6, **kwargs) -> None:
        super().__init__(**kwargs)
        if load not in ("none", "certified", "minimal"):
            raise ValueError("load must be none, certified or minimal")
        self.load = load
        self.block_size = int(block_size)
        self.group_size = int(group_size)
        self.margin_multiplier = float(margin_multiplier)
        self.error_chunk_size = int(error_chunk_size)
        self.steps_per_doubling = int(steps_per_doubling)
        self.maximum_load_to_ridge_ratio = float(maximum_load_to_ridge_ratio)
        self.gram: CompressedFactor | None = None
        # Scalars kept between tasks by the repaired variants.
        self.lower_bound = torch.zeros((), device=self.device, dtype=torch.float64)
        self.diagonal_load = torch.zeros((), device=self.device, dtype=torch.float64)

    def _margin(self, reconstructed: torch.Tensor) -> float:
        scale = max(float(reconstructed.diagonal().abs().amax().item()), self.ridge_lambda, 1.0)
        return self.margin_multiplier * torch.finfo(self.dtype).eps * self.dimension * scale

    def update(self, codes: torch.Tensor, labels: torch.Tensor) -> None:
        values, labels = self._inputs(codes, labels)
        class_ids, cross, counts, targets = self._expanded(labels)
        updated = values.T @ values
        if self.load == "none":
            if self.gram is not None:
                updated = updated + self.gram.decode_symmetric(dtype=self.dtype)
        else:
            if self.gram is not None:
                updated.add_(self.gram.decode_symmetric(dtype=self.dtype))
            updated = (updated + updated.T) * 0.5
        gram, _ = CompressedFactor.encode(
            updated, block_size=self.block_size, group_size=self.group_size, mode="int8", in_place=False
        )
        reconstructed = gram.decode_symmetric(dtype=self.dtype)
        new_cross = cross + values.T @ targets
        new_counts = counts + targets.sum(0)
        diagnostics: dict = {}

        if self.load == "none":
            system = reconstructed + self.ridge_lambda * torch.eye(self.dimension, device=self.device, dtype=self.dtype)
            diagnostics["relative_storage_error"] = float(torch.linalg.vector_norm(reconstructed - updated).item()) / max(
                float(torch.linalg.vector_norm(updated).item()), 1.0
            )
            weights, residual = cholesky_solve(system, new_cross)
            load = 0.0
        elif self.load == "certified":
            error_bound, relative = quantization_error_metrics(reconstructed, updated, row_chunk_size=self.error_chunk_size)
            error_bound = math.nextafter(error_bound * (1.0 + 4.0 * torch.finfo(torch.float64).eps * self.dimension), math.inf)
            bound = math.nextafter(float(self.lower_bound.item()) - error_bound, -math.inf)
            margin = self._margin(reconstructed)
            load = max(0.0, margin - (self.ridge_lambda + bound))
            system = reconstructed
            system.diagonal().add_(self.ridge_lambda + load)
            weights, residual = cholesky_solve(system, new_cross)
            self.lower_bound.fill_(bound)
            diagnostics.update(
                relative_storage_error=relative, quantization_error_infinity_norm=error_bound,
                certified_lower_bound=bound, safety_margin=margin,
            )
        else:
            error_bound, relative = quantization_error_metrics(reconstructed, updated, row_chunk_size=self.error_chunk_size)
            del updated
            margin = self._margin(reconstructed)
            search = minimal_cholesky_load(
                reconstructed, ridge_lambda=self.ridge_lambda, base_load=margin,
                steps_per_doubling=self.steps_per_doubling,
                maximum_load=self.maximum_load_to_ridge_ratio * self.ridge_lambda,
            )
            load = float(search["load"])
            original = reconstructed.diagonal().clone()
            reconstructed.diagonal().copy_(original + (self.ridge_lambda + load))
            weights, residual = cholesky_solve(reconstructed, new_cross)
            diagnostics.update(
                relative_storage_error=relative, quantization_error_infinity_norm=error_bound,
                grid_base_load=margin, grid_index=search["grid_index"], cholesky_attempts=search["attempts"],
            )

        self.gram = gram
        self.diagonal_load.fill_(load)
        self.class_ids, self.cross, self.counts, self.weights = class_ids, new_cross, new_counts, weights
        self.total_rows += len(values)
        self.diagnostics = {
            "solver_relative_residual": residual, "diagonal_load": load,
            "effective_ridge_lambda": self.ridge_lambda + load, **diagnostics,
        }

    def persistent_tensors(self) -> dict[str, torch.Tensor]:
        tensors = self._common_tensors()
        if self.load == "certified":
            tensors.update(lower_bound=self.lower_bound, diagonal_load=self.diagonal_load)
        elif self.load == "minimal":
            tensors["diagonal_load"] = self.diagonal_load
        if self.gram is not None:
            tensors.update(self.gram.persistent_tensors("A"))
        return tensors
