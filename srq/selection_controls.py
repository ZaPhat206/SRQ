"""Train-only selection controls for the beta=0.05 diagnostic.

The controls share SRQ's block layout and byte budget.  ``factor_error`` uses
the normal per-update factor-error selector; ``random_promotion`` fills the
same budget in a declared pseudorandom block order.  A ``static`` policy is
provided for auditing and reuses the first factor-error mask.
"""

from __future__ import annotations

import math
import random

import torch

from .adaptive import _int8_block, budget_bytes, greedy_select, score_blocks
from .codec import Block, _block_values, _write_block, block_layout
from .ridge import RidgeLearner, relative_factor_residual
from .qr import blocked_qr_update


def random_select(extra_costs: list[int], *, extra_budget: int, seed: int) -> tuple[list[bool], int]:
    order = list(range(len(extra_costs)))
    random.Random(int(seed)).shuffle(order)
    selected = [False] * len(extra_costs)
    used = 0
    for index in order:
        cost = int(extra_costs[index])
        if cost > 0 and used + cost <= extra_budget:
            selected[index] = True
            used += cost
    return selected, used


def _encode_with_mask(
    matrix: torch.Tensor, *, block_size: int, group_size: int, budget_fraction: float,
    selected: list[bool],
) -> tuple[object, float, dict]:
    """Encode a factor using a supplied FP16 mask and return a decoded matrix."""
    from .adaptive import AdaptiveFactor

    layout = block_layout(len(matrix), block_size)
    counts = [entries for _, _, entries in layout]
    base, full, extra_budget = budget_bytes(counts, group_size=group_size, budget_fraction=budget_fraction)
    costs = [2 * n - (n + 4 * math.ceil(n / group_size)) for n in counts]
    if len(selected) != len(layout) or any(flag and cost <= 0 for flag, cost in zip(selected, costs)):
        raise ValueError("invalid precision mask")
    used = sum(cost for flag, cost in zip(selected, costs) if flag)
    if used > extra_budget:
        raise ValueError("precision mask exceeds the byte budget")
    diagonal = matrix.diagonal().to(torch.float32).clone()
    blocks = []
    squared_error = torch.zeros((), device=matrix.device, dtype=torch.float64)
    squared_reference = diagonal.to(torch.float64).square().sum()
    for index, (row, col, _) in enumerate(layout):
        source = _block_values(matrix, row, col, block_size)
        squared_reference += source.to(torch.float64).square().sum()
        if selected[index]:
            values, scales = source.to(torch.float16), None
            decoded = values.to(source.dtype)
        else:
            values, scales, decoded = _int8_block(source, group_size)
        squared_error += (decoded.to(torch.float64) - source.to(torch.float64)).square().sum()
        blocks.append(Block(row, col, values, scales))
        _write_block(matrix, row, col, block_size, decoded)
    matrix.diagonal().copy_(diagonal.to(matrix.dtype))
    factor = AdaptiveFactor(
        dimension=len(matrix), block_size=block_size, group_size=group_size,
        budget_fraction=budget_fraction,
        diagonal=diagonal,
        precision_mask=torch.tensor(selected, device=matrix.device, dtype=torch.uint8),
        blocks=blocks,
    )
    error = float(torch.sqrt(squared_error / torch.clamp(squared_reference, min=1.0)).item())
    diagnostics = {
        "selected_fp16_blocks": sum(selected), "total_blocks": len(selected),
        "extra_budget_bytes": extra_budget, "used_extra_bytes": used,
        "factor_bytes": factor.state_bytes(), "int8_strict_upper_bytes": base,
        "fp16_strict_upper_bytes": full,
    }
    return factor, error, diagnostics


class SelectionControlRidge(RidgeLearner):
    """Square-root learner with a fixed, random, or factor-error mask policy."""

    def __init__(self, *, selection_policy: str, budget_fraction: float = 0.05,
                 allocation_seed: int = 2025, block_size: int = 256,
                 group_size: int = 64, panel_size: int = 128, **kwargs) -> None:
        super().__init__(**kwargs)
        if selection_policy not in {"factor_error", "random_promotion", "static"}:
            raise ValueError("unknown selection policy")
        self.selection_policy = selection_policy
        self.budget_fraction = float(budget_fraction)
        self.allocation_seed = int(allocation_seed)
        self.block_size = int(block_size)
        self.group_size = int(group_size)
        self.panel_size = int(panel_size)
        self.factor = None

    def _mask(self, upper: torch.Tensor) -> list[bool]:
        layout = block_layout(len(upper), self.block_size)
        counts = [entries for _, _, entries in layout]
        _, _, budget = budget_bytes(counts, group_size=self.group_size, budget_fraction=self.budget_fraction)
        costs = [2 * n - (n + 4 * math.ceil(n / self.group_size)) for n in counts]
        if self.selection_policy == "random_promotion":
            selected, _ = random_select(costs, extra_budget=budget, seed=self.allocation_seed + 1_000_003 * self.total_rows)
            return selected
        if self.selection_policy == "static" and self.factor is not None:
            return [bool(value) for value in self.factor.precision_mask.cpu().tolist()]
        scores = score_blocks(upper, block_size=self.block_size, group_size=self.group_size)
        selected, _ = greedy_select(scores["benefits"], scores["extra_costs"], budget)
        return selected

    def update(self, codes: torch.Tensor, labels: torch.Tensor) -> None:
        values, labels = self._inputs(codes, labels)
        class_ids, cross, counts, targets = self._expanded(labels)
        if self.factor is None:
            system = values.T @ values
            system.diagonal().add_(self.ridge_lambda)
            lower, info = torch.linalg.cholesky_ex((system + system.T) * 0.5)
            if int(info.max().item()) != 0:
                raise RuntimeError("first selection-control factorization failed")
            upper = lower.T
        else:
            upper = blocked_qr_update(self.factor.decode(dtype=self.dtype), values, panel_size=self.panel_size)
        selected = self._mask(upper)
        factor, factor_error, details = _encode_with_mask(
            upper, block_size=self.block_size, group_size=self.group_size,
            budget_fraction=self.budget_fraction, selected=selected,
        )
        intermediate = torch.linalg.solve_triangular(upper.T, cross + values.T @ targets, upper=False)
        weights = torch.linalg.solve_triangular(upper, intermediate, upper=True)
        new_cross = cross + values.T @ targets
        new_counts = counts + targets.sum(0)
        residual = relative_factor_residual(upper, weights, new_cross)
        self.factor = factor
        self.class_ids, self.cross, self.counts, self.weights = class_ids, new_cross, new_counts, weights
        self.total_rows += len(values)
        self.diagnostics = {
            "solver_relative_residual": residual, "relative_factor_error": factor_error,
            "selection_policy": self.selection_policy, **details,
        }

    def persistent_tensors(self) -> dict[str, torch.Tensor]:
        tensors = self._common_tensors()
        if self.factor is not None:
            tensors.update(self.factor.persistent_tensors("R"))
        return tensors


__all__ = ["SelectionControlRidge", "random_select"]
