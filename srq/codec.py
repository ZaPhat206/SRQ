"""Block-wise storage of the strict upper triangle of a square matrix.

The strict upper triangle is partitioned into blocks of ``block_size x
block_size`` index pairs (a diagonal block keeps only its strict upper part).
Each block is stored either in FP16 or with symmetric group-wise INT8
quantization: its entries, in row-major order, are split into groups of
``group_size`` consecutive entries, and every group stores one FP32 scale
``max|r| / 127`` (1 for an all-zero group) and INT8 values
``clip(round(r / scale), -127, 127)``.  Groups restart in every block.  The
diagonal is always kept in FP32.

Quantization acts on each group independently, so encoding blocks of equal
length together in batches does not change a single stored byte.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import torch

MODES = ("int8", "fp16")


@dataclass(frozen=True)
class Block:
    """Encoded values of one upper-triangular block."""

    row: int
    col: int
    values: torch.Tensor
    scales: torch.Tensor | None


def block_bounds(index: int, block_size: int, dimension: int) -> tuple[int, int]:
    start = index * block_size
    return start, min(start + block_size, dimension)


def block_layout(dimension: int, block_size: int) -> list[tuple[int, int, int]]:
    """``(row, col, entries)`` of every upper-triangular block in storage order."""
    count = math.ceil(dimension / block_size)
    layout = []
    for row in range(count):
        rows = block_bounds(row, block_size, dimension)
        rows = rows[1] - rows[0]
        for col in range(row, count):
            columns = block_bounds(col, block_size, dimension)
            columns = columns[1] - columns[0]
            entries = rows * columns if row != col else rows * (rows - 1) // 2
            layout.append((row, col, entries))
    return layout


def factor_bytes(dimension: int, *, block_size: int, group_size: int, mode: str) -> int:
    """Bytes of an encoded factor: FP32 diagonal plus the encoded blocks."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    values = scales = 0
    for _, _, entries in block_layout(dimension, block_size):
        values += entries
        scales += math.ceil(entries / group_size)
    if mode == "fp16":
        return 4 * dimension + 2 * values
    return 4 * dimension + values + 4 * scales


def quantize_int8_rows(values: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Group-wise INT8 codes and FP32 scales of every row of ``values``."""
    if values.ndim != 2:
        raise ValueError("values must be a matrix")
    rows, columns = values.shape
    if columns == 0:
        return (
            torch.empty((rows, 0), device=values.device, dtype=torch.int8),
            torch.empty((rows, 0), device=values.device, dtype=torch.float32),
        )
    groups = math.ceil(columns / group_size)
    padded = torch.zeros((rows, groups * group_size), device=values.device, dtype=values.dtype)
    padded[:, :columns] = values
    grouped = padded.reshape(rows, groups, group_size)
    maxima = grouped.abs().amax(2)
    scales = torch.where(maxima > 0, maxima / 127.0, torch.ones_like(maxima)).to(torch.float32)
    quantized = torch.round(grouped / scales.to(grouped.dtype)[:, :, None]).clamp(-127, 127).to(torch.int8)
    return quantized.reshape(rows, -1)[:, :columns], scales


def dequantize_int8(values: torch.Tensor, scales: torch.Tensor, group_size: int, dtype: torch.dtype) -> torch.Tensor:
    """Decode the INT8 values of one block."""
    return values.to(dtype) * scales.to(dtype).repeat_interleave(group_size)[: len(values)]


def _block_values(matrix: torch.Tensor, row: int, col: int, block_size: int) -> torch.Tensor:
    dimension = len(matrix)
    rs, re = block_bounds(row, block_size, dimension)
    cs, ce = block_bounds(col, block_size, dimension)
    local = matrix[rs:re, cs:ce]
    if row == col:
        indices = torch.triu_indices(len(local), len(local), offset=1, device=matrix.device)
        return local[indices[0], indices[1]]
    # A block cut from a large row-major matrix is not contiguous.
    return local.contiguous().view(-1)


def _write_block(matrix: torch.Tensor, row: int, col: int, block_size: int, decoded: torch.Tensor) -> None:
    dimension = len(matrix)
    rs, re = block_bounds(row, block_size, dimension)
    cs, ce = block_bounds(col, block_size, dimension)
    local = matrix[rs:re, cs:ce]
    if row == col:
        indices = torch.triu_indices(len(local), len(local), offset=1, device=matrix.device)
        local[indices[0], indices[1]] = decoded
    else:
        local.copy_(decoded.reshape(re - rs, ce - cs))


def check_finite(matrix: torch.Tensor, block_size: int) -> None:
    """Row-chunked finiteness check that avoids an m-by-m boolean temporary."""
    finite = torch.ones((), device=matrix.device, dtype=torch.bool)
    rows = max(block_size, min(len(matrix), 1024))
    for start in range(0, len(matrix), rows):
        finite.logical_and_(torch.isfinite(matrix[start : start + rows]).all())
    if not bool(finite):
        raise ValueError("matrix contains NaN or Inf")


class CompressedFactor:
    """FP32 diagonal plus the strict upper triangle in INT8 or FP16 blocks."""

    def __init__(
        self, *, dimension: int, block_size: int, group_size: int, mode: str,
        diagonal: torch.Tensor, blocks: list[Block],
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if diagonal.shape != (dimension,) or diagonal.dtype != torch.float32:
            raise ValueError("diagonal must be an FP32 vector of length dimension")
        if len(blocks) != len(block_layout(dimension, block_size)):
            raise ValueError("incomplete block list")
        self.dimension = int(dimension)
        self.block_size = int(block_size)
        self.group_size = int(group_size)
        self.mode = mode
        self.diagonal = diagonal
        self.blocks = blocks

    @property
    def device(self) -> torch.device:
        return self.diagonal.device

    @classmethod
    def encode(
        cls, matrix: torch.Tensor, *, block_size: int, group_size: int, mode: str,
        batch_blocks: int = 64, in_place: bool = True,
    ) -> tuple["CompressedFactor", float | None]:
        """Encode the upper triangle of ``matrix``.

        With ``in_place=True`` the entries of ``matrix`` are overwritten by their
        decoded values, so ``matrix`` becomes the stored factor without a second
        dense allocation, and the relative Frobenius error of the encoding,
        ``||decoded - matrix|| / max(||matrix||, 1)`` over the upper triangle, is
        returned.  With ``in_place=False`` ``matrix`` is left unchanged and the
        error is ``None``.
        """
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not len(matrix):
            raise ValueError("matrix must be square and non-empty")
        if matrix.dtype not in (torch.float32, torch.float64):
            raise ValueError("matrix must use FP32 or FP64")
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if min(block_size, group_size, batch_blocks) <= 0:
            raise ValueError("block, group and batch sizes must be positive")
        check_finite(matrix, block_size)

        dimension = len(matrix)
        diagonal = matrix.diagonal().to(torch.float32).clone()
        layout = block_layout(dimension, block_size)
        by_length: dict[int, list[int]] = defaultdict(list)
        for index, (_, _, entries) in enumerate(layout):
            by_length[entries].append(index)
        encoded: list[tuple[torch.Tensor, torch.Tensor | None] | None] = [None] * len(layout)

        if in_place:
            # The diagnostic sums are accumulated in FP64 in a fixed order.
            original_diagonal = matrix.diagonal().to(torch.float64)
            squared_reference = original_diagonal.square().sum()
            squared_error = (diagonal.to(torch.float64) - original_diagonal).square().sum()
            matrix.diagonal().copy_(diagonal.to(matrix.dtype))

        for indices in by_length.values():
            for start in range(0, len(indices), batch_blocks):
                chunk = indices[start : start + batch_blocks]
                stacked = torch.stack([_block_values(matrix, *layout[index][:2], block_size) for index in chunk])
                if mode == "int8":
                    values, scales = quantize_int8_rows(stacked, group_size)
                    for position, index in enumerate(chunk):
                        encoded[index] = (values[position], scales[position])
                    if in_place:
                        decoded = values.to(stacked.dtype) * scales.to(stacked.dtype).repeat_interleave(
                            group_size, dim=1
                        )[:, : stacked.shape[1]]
                else:
                    values = stacked.to(torch.float16)
                    if not bool(torch.isfinite(values).all()):
                        raise ValueError("FP16 encoding overflowed")
                    for position, index in enumerate(chunk):
                        encoded[index] = (values[position], None)
                    if in_place:
                        decoded = values.to(stacked.dtype)
                if in_place:
                    difference = decoded - stacked
                    squared_error.add_(difference.square().sum(dtype=torch.float64))
                    squared_reference.add_(stacked.square().sum(dtype=torch.float64))
                    for position, index in enumerate(chunk):
                        _write_block(matrix, layout[index][0], layout[index][1], block_size, decoded[position])
                    del decoded, difference
                del stacked

        blocks = [Block(row, col, *encoded[index]) for index, (row, col, _) in enumerate(layout)]
        factor = cls(
            dimension=dimension, block_size=block_size, group_size=group_size,
            mode=mode, diagonal=diagonal, blocks=blocks,
        )
        if not in_place:
            return factor, None
        error = float(torch.sqrt(squared_error / torch.clamp(squared_reference, min=1.0)).item())
        return factor, error

    def decode(self, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Dense upper-triangular matrix with the stored values."""
        matrix = torch.zeros((self.dimension, self.dimension), device=self.device, dtype=dtype)
        matrix.diagonal().copy_(self.diagonal.to(dtype))
        for block in self.blocks:
            rs, re = block_bounds(block.row, self.block_size, self.dimension)
            cs, ce = block_bounds(block.col, self.block_size, self.dimension)
            if block.scales is not None:
                decoded = dequantize_int8(block.values, block.scales, self.group_size, dtype)
            else:
                decoded = block.values.to(dtype)
            if block.row == block.col:
                size = re - rs
                indices = torch.triu_indices(size, size, offset=1, device=self.device)
                local = matrix[rs:re, cs:ce]
                local[indices[0], indices[1]] = decoded
            else:
                matrix[rs:re, cs:ce] = decoded.reshape(re - rs, ce - cs)
        return matrix

    def decode_symmetric(self, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Dense symmetric matrix whose upper triangle holds the stored values."""
        upper = self.decode(dtype=dtype)
        return upper + upper.T - torch.diag_embed(upper.diagonal())

    def persistent_tensors(self, prefix: str) -> dict[str, torch.Tensor]:
        tensors = {f"{prefix}.diagonal": self.diagonal}
        for index, block in enumerate(self.blocks):
            tensors[f"{prefix}.block_{index}.values"] = block.values
            if block.scales is not None:
                tensors[f"{prefix}.block_{index}.scales"] = block.scales
        return tensors
