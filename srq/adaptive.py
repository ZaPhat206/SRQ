"""Budget-limited mixed INT8/FP16 storage (SRQ-Adaptive).

Every upper-triangular block of the factor is encoded both ways.  Storing
block ``b`` in FP16 instead of INT8 reduces its squared Frobenius error by
``g_b = e8_b - e16_b`` and costs ``c_b = 2 n_b - (n_b + 4 ceil(n_b / 64))``
extra bytes.  With ``M8`` and ``M16`` the bytes of the all-INT8 and all-FP16
triangles, the extra bytes may not exceed ``floor(beta (M16 - M8))``.  Blocks
are promoted in decreasing order of ``g_b / c_b`` (ties by block index) while
their extra bytes still fit; a block that does not fit is skipped and later
blocks are still considered.  Blocks with ``g_b <= 0`` or ``c_b <= 0`` are never
promoted.  A one-byte precision flag per block is part of the state.
"""

from __future__ import annotations

import math
from collections import defaultdict

import torch

from .codec import Block, _block_values, _write_block, block_bounds, block_layout, check_finite


def _int8_block(values: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """INT8 codes, scales and decoded values of one block."""
    flat = values.reshape(-1)
    groups = math.ceil(len(flat) / group_size)
    padded = torch.zeros(groups * group_size, device=flat.device, dtype=flat.dtype)
    padded[: len(flat)] = flat
    rows = padded.reshape(groups, group_size)
    maxima = rows.abs().amax(1)
    scales = torch.where(maxima > 0, maxima / 127.0, torch.ones_like(maxima)).to(torch.float32)
    quantized = torch.round(rows / scales.to(rows.dtype)[:, None]).clamp(-127, 127).to(torch.int8)
    encoded = quantized.reshape(-1)[: len(flat)]
    decoded = encoded.to(flat.dtype) * scales.to(flat.dtype).repeat_interleave(group_size)[: len(flat)]
    return encoded, scales, decoded


def score_blocks(matrix: torch.Tensor, *, block_size: int, group_size: int) -> dict:
    """Encode every block both ways and return errors, benefits and costs."""
    layout = block_layout(len(matrix), block_size)
    int8_blocks, int8_errors, fp16_errors, reference_sums = [], [], [], []
    extra_costs, value_counts = [], []
    for row, col, _ in layout:
        source = _block_values(matrix, row, col, block_size)
        encoded, scales, decoded_int8 = _int8_block(source, group_size)
        encoded_fp16 = source.to(torch.float16)
        if not bool(torch.isfinite(encoded_fp16).all()):
            raise ValueError("FP16 candidate overflowed")
        decoded_fp16 = encoded_fp16.to(source.dtype)
        int8_blocks.append((encoded, scales))
        int8_errors.append((decoded_int8 - source).square().sum())
        fp16_errors.append((decoded_fp16 - source).square().sum())
        reference_sums.append(source.square().sum())
        entries = int(source.numel())
        extra_costs.append(2 * entries - (entries + 4 * math.ceil(entries / group_size)))
        value_counts.append(entries)
    benefits = (torch.stack(int8_errors) - torch.stack(fp16_errors)).detach().to(torch.float64).cpu().tolist()
    return {
        "layout": layout, "int8_blocks": int8_blocks, "int8_errors": int8_errors,
        "fp16_errors": fp16_errors, "reference_sums": reference_sums,
        "benefits": benefits, "extra_costs": extra_costs, "value_counts": value_counts,
    }


def budget_bytes(value_counts: list[int], *, group_size: int, budget_fraction: float) -> tuple[int, int, int]:
    """Bytes of the all-INT8 and all-FP16 triangles and the extra-byte budget."""
    if not 0.0 <= float(budget_fraction) <= 1.0:
        raise ValueError("budget_fraction must lie in [0, 1]")
    base = sum(entries + 4 * math.ceil(entries / group_size) for entries in value_counts)
    full = 2 * sum(value_counts)
    return base, full, math.floor(float(budget_fraction) * (full - base))


def greedy_select(benefits: list[float], extra_costs: list[int], extra_budget: int) -> tuple[list[bool], int]:
    """Promote blocks by benefit per extra byte; skip blocks that do not fit."""
    order = sorted(range(len(benefits)), key=lambda index: (-benefits[index] / max(extra_costs[index], 1), index))
    selected = [False] * len(benefits)
    used = 0
    for index in order:
        cost = extra_costs[index]
        if benefits[index] <= 0 or cost <= 0:
            continue
        if used + cost <= extra_budget:
            selected[index] = True
            used += cost
    return selected, used


class AdaptiveFactor:
    """FP32 diagonal, per-block precision flags and INT8 or FP16 blocks."""

    def __init__(
        self, *, dimension: int, block_size: int, group_size: int, budget_fraction: float,
        diagonal: torch.Tensor, precision_mask: torch.Tensor, blocks: list[Block],
    ) -> None:
        if precision_mask.dtype != torch.uint8 or precision_mask.shape != (len(blocks),):
            raise ValueError("precision mask must hold one uint8 flag per block")
        self.dimension = int(dimension)
        self.block_size = int(block_size)
        self.group_size = int(group_size)
        self.budget_fraction = float(budget_fraction)
        self.mode = "adaptive"
        self.diagonal = diagonal
        self.precision_mask = precision_mask
        self.blocks = blocks

    @property
    def device(self) -> torch.device:
        return self.diagonal.device

    @classmethod
    def encode(
        cls, matrix: torch.Tensor, *, block_size: int, group_size: int, budget_fraction: float,
    ) -> tuple["AdaptiveFactor", float, dict]:
        """Select the precision of every block and overwrite ``matrix`` with the decoded factor."""
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not len(matrix):
            raise ValueError("matrix must be square and non-empty")
        if matrix.dtype not in (torch.float32, torch.float64):
            raise ValueError("matrix must use FP32 or FP64")
        check_finite(matrix, block_size)
        dimension = len(matrix)
        diagonal = matrix.diagonal().to(torch.float32).clone()
        scores = score_blocks(matrix, block_size=block_size, group_size=group_size)
        layout, value_counts = scores["layout"], scores["value_counts"]
        base_bytes, full_bytes, extra_budget = budget_bytes(
            value_counts, group_size=group_size, budget_fraction=budget_fraction
        )
        selected, used_extra = greedy_select(scores["benefits"], scores["extra_costs"], extra_budget)

        blocks = []
        for index, (row, col, _) in enumerate(layout):
            source = _block_values(matrix, row, col, block_size)
            if selected[index]:
                values, scales = source.to(torch.float16), None
                decoded = values.to(source.dtype)
            else:
                values, scales = scores["int8_blocks"][index]
                decoded = values.to(source.dtype) * scales.to(source.dtype).repeat_interleave(group_size)[: len(values)]
            blocks.append(Block(row, col, values, scales))
            _write_block(matrix, row, col, block_size, decoded)

        int8_errors, fp16_errors = scores["int8_errors"], scores["fp16_errors"]
        original_diagonal = matrix.diagonal().to(torch.float64)
        diagonal_error = (diagonal.to(torch.float64) - original_diagonal).square().sum()
        diagonal_reference = original_diagonal.square().sum()
        mask = torch.tensor(selected, device=matrix.device, dtype=torch.uint8)
        chosen = torch.where(mask.to(torch.bool), torch.stack(fp16_errors), torch.stack(int8_errors))
        squared_error = diagonal_error + chosen.to(torch.float64).sum()
        squared_reference = diagonal_reference + torch.stack(scores["reference_sums"]).to(torch.float64).sum()
        matrix.diagonal().copy_(diagonal.to(matrix.dtype))
        denominator = torch.clamp(squared_reference, min=1.0)
        error = float(torch.sqrt(squared_error / denominator).item())
        all_int8 = float(torch.sqrt((diagonal_error + torch.stack(int8_errors).to(torch.float64).sum()) / denominator).item())
        all_fp16 = float(torch.sqrt((diagonal_error + torch.stack(fp16_errors).to(torch.float64).sum()) / denominator).item())

        factor = cls(
            dimension=dimension, block_size=block_size, group_size=group_size,
            budget_fraction=budget_fraction, diagonal=diagonal, precision_mask=mask, blocks=blocks,
        )
        diagnostics = {
            "selected_fp16_blocks": sum(selected),
            "total_blocks": len(selected),
            "selected_fp16_values": sum(count for count, flag in zip(value_counts, selected) if flag),
            "total_strict_upper_values": sum(value_counts),
            "int8_strict_upper_bytes": base_bytes,
            "fp16_strict_upper_bytes": full_bytes,
            "extra_budget_bytes": extra_budget,
            "used_extra_bytes": used_extra,
            "factor_bytes": factor.state_bytes(),
            "factor_budget_ceiling_bytes": 4 * dimension + len(selected) + base_bytes + extra_budget,
            "all_int8_relative_factor_error": all_int8,
            "all_fp16_relative_factor_error": all_fp16,
        }
        return factor, error, diagnostics

    def decode(self, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        matrix = torch.zeros((self.dimension, self.dimension), device=self.device, dtype=dtype)
        matrix.diagonal().copy_(self.diagonal.to(dtype))
        for block in self.blocks:
            rs, re = block_bounds(block.row, self.block_size, self.dimension)
            cs, ce = block_bounds(block.col, self.block_size, self.dimension)
            if block.scales is None:
                decoded = block.values.to(dtype)
            else:
                decoded = block.values.to(dtype) * block.scales.to(dtype).repeat_interleave(self.group_size)[: len(block.values)]
            if block.row == block.col:
                indices = torch.triu_indices(re - rs, ce - cs, offset=1, device=self.device)
                local = matrix[rs:re, cs:ce]
                local[indices[0], indices[1]] = decoded
            else:
                matrix[rs:re, cs:ce] = decoded.reshape(re - rs, ce - cs)
        return matrix

    def persistent_tensors(self, prefix: str) -> dict[str, torch.Tensor]:
        tensors = {f"{prefix}.diagonal": self.diagonal, f"{prefix}.precision_mask": self.precision_mask}
        for index, block in enumerate(self.blocks):
            tensors[f"{prefix}.block_{index}.values"] = block.values
            if block.scales is not None:
                tensors[f"{prefix}.block_{index}.scales"] = block.scales
        return tensors

    def state_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.persistent_tensors("factor").values())


def batched_benefits(
    matrix: torch.Tensor, *, block_size: int, group_size: int, batch_blocks: int,
) -> tuple[list[float], list[int], list[int]]:
    """The benefits of :func:`score_blocks`, computed for batches of equally shaped blocks.

    Per-entry arithmetic is that of :func:`score_blocks`; only the order of the
    per-block summation differs, so benefits can differ in their last bits.
    Used only to measure the scoring time and to check that the selected
    blocks agree.
    """
    dimension, device = len(matrix), matrix.device
    layout = block_layout(dimension, block_size)
    shapes: dict[tuple[bool, int, int], list[int]] = defaultdict(list)
    extra_costs, value_counts = [], []
    for index, (row, col, entries) in enumerate(layout):
        rs, re = block_bounds(row, block_size, dimension)
        cs, ce = block_bounds(col, block_size, dimension)
        shapes[(row == col, re - rs, ce - cs)].append(index)
        extra_costs.append(2 * entries - (entries + 4 * math.ceil(entries / group_size)))
        value_counts.append(entries)

    benefits = [0.0] * len(layout)
    for (diagonal_block, rows, columns), members in shapes.items():
        if diagonal_block:
            local_rows, local_columns = torch.triu_indices(rows, columns, offset=1, device=device)
        else:
            grid_rows, grid_columns = torch.meshgrid(
                torch.arange(rows, device=device), torch.arange(columns, device=device), indexing="ij"
            )
            local_rows, local_columns = grid_rows.reshape(-1), grid_columns.reshape(-1)
        entries = int(local_rows.numel())
        if entries == 0:
            continue
        groups = math.ceil(entries / group_size)
        for start in range(0, len(members), batch_blocks):
            chunk = members[start : start + batch_blocks]
            row_starts = torch.tensor([layout[index][0] * block_size for index in chunk], device=device)
            column_starts = torch.tensor([layout[index][1] * block_size for index in chunk], device=device)
            source = matrix[row_starts[:, None] + local_rows[None, :], column_starts[:, None] + local_columns[None, :]]
            padded = torch.zeros((len(chunk), groups * group_size), device=device, dtype=source.dtype)
            padded[:, :entries] = source
            grouped = padded.reshape(len(chunk), groups, group_size)
            maxima = grouped.abs().amax(2)
            scales = torch.where(maxima > 0, maxima / 127.0, torch.ones_like(maxima)).to(torch.float32)
            quantized = torch.round(grouped / scales.to(grouped.dtype)[:, :, None]).clamp(-127, 127).to(torch.int8)
            decoded = quantized.reshape(len(chunk), -1)[:, :entries].to(source.dtype) * scales.to(
                source.dtype
            ).repeat_interleave(group_size, dim=1)[:, :entries]
            int8_error = (decoded - source).square().sum(1)
            fp16_error = (source.to(torch.float16).to(source.dtype) - source).square().sum(1)
            for index, value in zip(chunk, (int8_error - fp16_error).to(torch.float64).cpu().tolist()):
                benefits[index] = value
    return benefits, extra_costs, value_counts
