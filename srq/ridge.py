"""Class-incremental ridge learners on fixed expanded features.

All learners keep the cross-statistic ``B`` (one column per class seen so far),
the per-class counts and the classifier ``W``, and differ in how they store
the system matrix ``A_t = lambda I + sum_k Phi_k^T Phi_k``:

* :class:`ExactRidge` stores the Gram matrix in FP32 (the uncompressed learner).
* :class:`SquareRootRidge` stores a compressed upper-triangular factor ``R`` with
  ``R^T R ~ A_t`` in INT8, FP16 or budget-limited mixed precision (SRQ).

No learner keeps samples, features or labels between tasks.
"""

from __future__ import annotations

import time

import torch

from .adaptive import AdaptiveFactor
from .codec import CompressedFactor
from .qr import blocked_qr_update


def tensor_bytes(tensor: torch.Tensor) -> int:
    """Bytes owned by a dense or sparse CSC tensor."""
    if tensor.layout == torch.strided:
        return tensor.numel() * tensor.element_size()
    if tensor.layout == torch.sparse_csc:
        return sum(part.numel() * part.element_size() for part in (tensor.values(), tensor.ccol_indices(), tensor.row_indices()))
    raise ValueError(f"unsupported tensor layout: {tensor.layout}")


def relative_residual(system: torch.Tensor, weights: torch.Tensor, cross: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm(system @ weights - cross)
    return float(numerator.item()) / max(float(torch.linalg.vector_norm(cross).item()), 1.0)


def relative_factor_residual(factor: torch.Tensor, weights: torch.Tensor, cross: torch.Tensor) -> float:
    """Residual of ``R^T R W = B`` without forming ``R^T R``."""
    numerator = torch.linalg.vector_norm(factor.T @ (factor @ weights) - cross)
    return float(numerator.item()) / max(float(torch.linalg.vector_norm(cross).item()), 1.0)


def cholesky_solve(system: torch.Tensor, cross: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Solve with the Cholesky factor of the symmetrized system."""
    symmetric = (system + system.T) * 0.5
    factor, info = torch.linalg.cholesky_ex(symmetric)
    if int(info.max().item()) != 0:
        raise RuntimeError("ridge system is not numerically positive definite")
    weights = torch.cholesky_solve(cross, factor)
    return weights, relative_residual(symmetric, weights, cross)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class RidgeLearner:
    """Cross-statistic, class counts and classifier shared by all learners."""

    def __init__(self, *, dimension: int, ridge_lambda: float, device: str | torch.device = "cpu",
                 dtype: torch.dtype = torch.float32) -> None:
        if dimension <= 0 or ridge_lambda <= 0:
            raise ValueError("dimension and ridge_lambda must be positive")
        self.dimension = int(dimension)
        self.ridge_lambda = float(ridge_lambda)
        self.device = torch.device(device)
        self.dtype = dtype
        self.cross = torch.zeros((self.dimension, 0), device=self.device, dtype=dtype)
        self.counts = torch.zeros(0, device=self.device, dtype=dtype)
        self.class_ids: list[int] = []
        self.weights: torch.Tensor | None = None
        self.total_rows = 0
        self.diagnostics: dict = {}

    def _inputs(self, codes: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values = codes.to(device=self.device, dtype=self.dtype)
        targets = labels.to(device=self.device, dtype=torch.long)
        if values.ndim != 2 or values.shape[1] != self.dimension:
            raise ValueError(f"codes must have shape (n, {self.dimension})")
        if targets.ndim != 1 or len(targets) != len(values) or not len(values):
            raise ValueError("labels must align with a non-empty code matrix")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("codes contain NaN or Inf")
        return values, targets

    def _expanded(self, labels: torch.Tensor) -> tuple[list[int], torch.Tensor, torch.Tensor, torch.Tensor]:
        """Class columns after this task, the expanded ``B`` and counts, and one-hot targets."""
        class_ids = sorted(set(self.class_ids) | set(map(int, labels.cpu().tolist())))
        new_column = {value: index for index, value in enumerate(class_ids)}
        cross = torch.zeros((self.dimension, len(class_ids)), device=self.device, dtype=self.dtype)
        counts = torch.zeros(len(class_ids), device=self.device, dtype=self.dtype)
        for old_index, value in enumerate(self.class_ids):
            cross[:, new_column[value]] = self.cross[:, old_index]
            counts[new_column[value]] = self.counts[old_index]
        columns = torch.tensor([new_column[int(value)] for value in labels.cpu().tolist()], device=self.device, dtype=torch.long)
        targets = torch.nn.functional.one_hot(columns, num_classes=len(class_ids)).to(self.dtype)
        return class_ids, cross, counts, targets

    def predict_logits(self, codes: torch.Tensor) -> torch.Tensor:
        if self.weights is None:
            raise RuntimeError("the learner has not been updated")
        return codes.to(device=self.device, dtype=self.weights.dtype) @ self.weights

    def predict(self, codes: torch.Tensor) -> torch.Tensor:
        columns = self.predict_logits(codes).argmax(1).cpu().tolist()
        return torch.tensor([self.class_ids[column] for column in columns])

    def _common_tensors(self) -> dict[str, torch.Tensor]:
        tensors = {"B": self.cross, "counts": self.counts}
        if self.weights is not None:
            tensors["W"] = self.weights
        return tensors

    def persistent_tensors(self) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def state_bytes(self) -> int:
        return sum(tensor_bytes(tensor) for tensor in self.persistent_tensors().values())

    def check_sample_free(self) -> None:
        """No stored tensor may have a dimension equal to the number of rows seen."""
        structural = {self.dimension, len(self.class_ids)}
        for name, tensor in self.persistent_tensors().items():
            if tensor.ndim >= 2 and self.total_rows not in structural and self.total_rows in tensor.shape:
                raise AssertionError(f"sample-level dimension in stored tensor {name}")


class ExactRidge(RidgeLearner):
    """Uncompressed learner: stores the Gram matrix ``sum_k Phi_k^T Phi_k`` in FP32."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.gram = torch.zeros((self.dimension, self.dimension), device=self.device, dtype=self.dtype)

    def update(self, codes: torch.Tensor, labels: torch.Tensor) -> None:
        values, labels = self._inputs(codes, labels)
        class_ids, cross, counts, targets = self._expanded(labels)
        self.gram.add_(values.T @ values)
        new_cross = cross + values.T @ targets
        new_counts = counts + targets.sum(0)
        system = self.gram.clone()
        system.diagonal().add_(self.ridge_lambda)
        symmetric = (system + system.T) * 0.5
        factor, info = torch.linalg.cholesky_ex(symmetric)
        if int(info.max().item()) != 0:
            raise RuntimeError("exact ridge system is not positive definite")
        weights = torch.cholesky_solve(new_cross, factor)
        residual = relative_residual(symmetric, weights, new_cross)
        self.class_ids, self.cross, self.counts, self.weights = class_ids, new_cross, new_counts, weights
        self.total_rows += len(values)
        self.diagnostics = {"solver_relative_residual": residual}

    def persistent_tensors(self) -> dict[str, torch.Tensor]:
        return {"A": self.gram, **self._common_tensors()}


class SquareRootRidge(RidgeLearner):
    """SRQ: a compressed square-root factor of the ridge system.

    ``storage`` is ``"int8"`` (SRQ-INT8), ``"fp16"`` (FP16 factor) or
    ``"adaptive"`` (SRQ-Adaptive with ``budget_fraction``).  The first factor
    is the Cholesky factor of ``Phi_1^T Phi_1 + lambda I``; later factors are
    computed from the stored factor by a blocked QR update
    (``factor_update="qr"``, Algorithm 1 of the paper).  ``factor_update=
    "cholesky"`` instead factors ``R_hat^T R_hat + Phi_t^T Phi_t`` directly; it
    gives the same factor up to rounding and is kept only to reproduce the
    ridge-coefficient selection of FLY-CL, which was run with it.
    """

    def __init__(
        self, *, storage: str = "int8", block_size: int = 256, group_size: int = 64,
        budget_fraction: float | None = None, factor_update: str = "qr", panel_size: int = 128,
        quantization_batch_blocks: int = 64, **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if storage not in ("int8", "fp16", "adaptive"):
            raise ValueError("storage must be int8, fp16 or adaptive")
        if (storage == "adaptive") != (budget_fraction is not None):
            raise ValueError("budget_fraction is required for, and only for, adaptive storage")
        if factor_update not in ("qr", "cholesky"):
            raise ValueError("factor_update must be qr or cholesky")
        if factor_update == "cholesky" and storage == "adaptive":
            raise ValueError("the Cholesky update is only provided for fixed storage")
        self.storage = storage
        self.block_size = int(block_size)
        self.group_size = int(group_size)
        self.budget_fraction = budget_fraction
        self.factor_update = factor_update
        self.panel_size = int(panel_size)
        self.quantization_batch_blocks = int(quantization_batch_blocks)
        self.factor: CompressedFactor | AdaptiveFactor | None = None
        self.compress_hook = None  # optional callable on the unquantized factor (benchmarks)

    def update(self, codes: torch.Tensor, labels: torch.Tensor, *, consume_codes: bool = False) -> None:
        """Update with one task.  ``consume_codes=True`` lets the QR overwrite ``codes``."""
        if self.factor_update == "cholesky":
            self._cholesky_update(codes, labels)
            return
        values, labels = self._inputs(codes, labels)
        class_ids, cross, counts, targets = self._expanded(labels)
        new_cross = cross + values.T @ targets
        new_counts = counts + targets.sum(0)
        if self.factor is None:
            system = values.T @ values
            system.diagonal().add_(self.ridge_lambda)
            symmetric = (system + system.T) * 0.5
            lower, info = torch.linalg.cholesky_ex(symmetric)
            if int(info.max().item()) != 0:
                raise RuntimeError("first SRQ factorization failed")
            del system, symmetric
            upper = lower.T
        else:
            previous = self.factor.decode(dtype=self.dtype)
            upper = blocked_qr_update(previous, values, panel_size=self.panel_size, preserve_rows=not consume_codes)
        if self.compress_hook is not None:
            self.compress_hook(upper)
        adaptive = None
        if self.storage == "adaptive":
            factor, factor_error, adaptive = AdaptiveFactor.encode(
                upper, block_size=self.block_size, group_size=self.group_size, budget_fraction=float(self.budget_fraction)
            )
        else:
            factor, factor_error = CompressedFactor.encode(
                upper, block_size=self.block_size, group_size=self.group_size, mode=self.storage,
                batch_blocks=self.quantization_batch_blocks, in_place=True,
            )
        if bool((upper.diagonal() <= 0).any()):
            raise RuntimeError("stored factor has a non-positive diagonal")
        intermediate = torch.linalg.solve_triangular(upper.T, new_cross, upper=False)
        weights = torch.linalg.solve_triangular(upper, intermediate, upper=True)
        residual = relative_factor_residual(upper, weights, new_cross)
        self.factor = factor
        self.class_ids, self.cross, self.counts, self.weights = class_ids, new_cross, new_counts, weights
        self.total_rows += len(values)
        self.diagnostics = {"solver_relative_residual": residual, "relative_factor_error": factor_error}
        if adaptive is not None:
            self.diagnostics.update(adaptive)

    def _cholesky_update(self, codes: torch.Tensor, labels: torch.Tensor) -> None:
        values, labels = self._inputs(codes, labels)
        class_ids, cross, counts, targets = self._expanded(labels)
        system = values.T @ values
        if self.factor is None:
            system.diagonal().add_(self.ridge_lambda)
        else:
            previous = self.factor.decode(dtype=self.dtype)
            system = system + previous.T @ previous
            del previous
        lower, info = torch.linalg.cholesky_ex((system + system.T) * 0.5)
        if int(info.max().item()) != 0:
            raise RuntimeError("SRQ Cholesky update failed")
        del system
        exact_upper = lower.T
        factor, _ = CompressedFactor.encode(
            exact_upper, block_size=self.block_size, group_size=self.group_size, mode=self.storage, in_place=False
        )
        upper = factor.decode(dtype=self.dtype)
        if bool((upper.diagonal() <= 0).any()):
            raise RuntimeError("stored factor has a non-positive diagonal")
        new_cross = cross + values.T @ targets
        new_counts = counts + targets.sum(0)
        intermediate = torch.linalg.solve_triangular(upper.T, new_cross, upper=False)
        weights = torch.linalg.solve_triangular(upper, intermediate, upper=True)
        residual = relative_factor_residual(upper, weights, new_cross)
        factor_error = float(torch.linalg.vector_norm(upper - exact_upper).item()) / max(
            float(torch.linalg.vector_norm(exact_upper).item()), 1.0
        )
        self.factor = factor
        self.class_ids, self.cross, self.counts, self.weights = class_ids, new_cross, new_counts, weights
        self.total_rows += len(values)
        self.diagnostics = {"solver_relative_residual": residual, "relative_factor_error": factor_error}

    def persistent_tensors(self) -> dict[str, torch.Tensor]:
        tensors = self._common_tensors()
        if self.factor is not None:
            tensors.update(self.factor.persistent_tensors("R"))
        return tensors

    def factor_bytes(self) -> int:
        return 0 if self.factor is None else sum(
            tensor_bytes(tensor) for tensor in self.factor.persistent_tensors("R").values()
        )


def timed_update(learner: RidgeLearner, codes: torch.Tensor, labels: torch.Tensor, **kwargs) -> float:
    """Update and return the wall-clock seconds, synchronizing CUDA before and after."""
    sync(learner.device)
    started = time.perf_counter()
    learner.update(codes, labels, **kwargs)
    sync(learner.device)
    return time.perf_counter() - started
