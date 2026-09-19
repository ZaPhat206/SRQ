"""Self-contained tests for the storage codec (no dependency on any other repo)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import srq


def _upper(dimension: int, seed: int) -> torch.Tensor:
    matrix = torch.triu(torch.randn(dimension, dimension, generator=torch.Generator().manual_seed(seed)))
    matrix.diagonal().abs_().add_(1.0)
    return matrix


@pytest.mark.parametrize("mode", ["int8", "fp16"])
@pytest.mark.parametrize("dimension,block,group", [(64, 32, 8), (300, 256, 64), (600, 256, 64)])
def test_codec_roundtrip_and_bytes(mode, dimension, block, group):
    original = _upper(dimension, 1)
    factor, error = srq.CompressedFactor.encode(original.clone(), block_size=block, group_size=group, mode=mode, in_place=False)
    decoded = factor.decode()
    assert torch.equal(decoded.tril(-1), torch.zeros_like(decoded.tril(-1)))
    assert torch.equal(decoded.diagonal(), original.diagonal())
    assert error is None  # in_place=False never computes the diagnostic
    tolerance = 2e-2 if mode == "fp16" else 0.05
    relative = float(torch.linalg.vector_norm(decoded - original) / torch.linalg.vector_norm(original))
    assert relative < tolerance
    actual_bytes = sum(srq.tensor_bytes(t) for t in factor.persistent_tensors("R").values())
    assert actual_bytes == srq.factor_bytes(dimension, block_size=block, group_size=group, mode=mode)


def test_int8_smaller_than_fp16_smaller_than_fp32():
    dimension = 512
    original = _upper(dimension, 2)
    int8_factor, _ = srq.CompressedFactor.encode(original.clone(), block_size=256, group_size=64, mode="int8", in_place=False)
    fp16_factor, _ = srq.CompressedFactor.encode(original.clone(), block_size=256, group_size=64, mode="fp16", in_place=False)
    int8_bytes = sum(srq.tensor_bytes(t) for t in int8_factor.persistent_tensors("R").values())
    fp16_bytes = sum(srq.tensor_bytes(t) for t in fp16_factor.persistent_tensors("R").values())
    fp32_bytes = 4 * dimension * dimension
    assert int8_bytes < fp16_bytes < fp32_bytes


def test_symmetric_decode():
    dimension = 128
    original = _upper(dimension, 3)
    factor, _ = srq.CompressedFactor.encode(original.clone(), block_size=64, group_size=16, mode="int8", in_place=False)
    symmetric = factor.decode_symmetric()
    assert torch.equal(symmetric, symmetric.T)
    assert torch.equal(symmetric.triu(), factor.decode())


def test_in_place_overwrites_matrix_with_decoded_values():
    dimension = 128
    original = _upper(dimension, 4)
    working = original.clone()
    factor, error = srq.CompressedFactor.encode(working, block_size=64, group_size=16, mode="int8", in_place=True)
    assert torch.equal(working.triu(), factor.decode())
    assert error is not None and error >= 0


def test_nan_or_inf_rejected():
    matrix = _upper(64, 5)
    matrix[3, 3] = float("nan")
    with pytest.raises(ValueError):
        srq.CompressedFactor.encode(matrix, block_size=32, group_size=8, mode="int8", in_place=False)
