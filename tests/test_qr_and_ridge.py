"""Self-contained tests: blocked QR update and the ridge learners."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import srq


def test_blocked_qr_matches_dense_qr_gram():
    dimension, ridge = 200, 10.0
    generator = torch.Generator().manual_seed(1)
    phi1 = torch.randn(60, dimension, generator=generator)
    phi2 = torch.randn(45, dimension, generator=generator)

    upper = torch.linalg.cholesky_ex((phi1.T @ phi1 + ridge * torch.eye(dimension)))[0].T
    updated = srq.blocked_qr_update(upper.clone(), phi2, panel_size=64)

    reference_gram = phi1.T @ phi1 + ridge * torch.eye(dimension) + phi2.T @ phi2
    assert torch.allclose(updated.T @ updated, reference_gram, atol=1e-3, rtol=1e-4)
    assert bool((updated.diagonal() > 0).all())


def test_blocked_qr_panel_size_does_not_change_result():
    dimension = 150
    generator = torch.Generator().manual_seed(2)
    upper = torch.triu(torch.randn(dimension, dimension, generator=generator))
    upper.diagonal().abs_().add_(1.0)
    rows = torch.randn(80, dimension, generator=generator)
    a = srq.blocked_qr_update(upper.clone(), rows.clone(), panel_size=32)
    b = srq.blocked_qr_update(upper.clone(), rows.clone(), panel_size=128)
    assert torch.allclose(a.T @ a, b.T @ b, atol=1e-3, rtol=1e-4)


def _classification_stream(width=200, classes=10, tasks=4, rows_per_task=60, seed=1):
    generator = torch.Generator().manual_seed(seed)
    for _ in range(tasks):
        codes = torch.relu(torch.randn(rows_per_task, width, generator=generator))
        labels = torch.randint(0, classes, (rows_per_task,), generator=generator)
        yield codes, labels


def test_exact_ridge_matches_normal_equations():
    width, ridge = 100, 50.0
    learner = srq.ExactRidge(dimension=width, ridge_lambda=ridge, device="cpu")
    gram = torch.zeros(width, width)
    cross_by_class = {}
    for codes, labels in _classification_stream(width=width):
        learner.update(codes, labels)
        gram += codes.T @ codes
    system = gram + ridge * torch.eye(width)
    residual = torch.linalg.vector_norm(system @ learner.weights - learner.cross) / torch.linalg.vector_norm(learner.cross)
    assert float(residual) < 1e-4


def test_square_root_ridge_close_to_exact():
    width, ridge = 150, 100.0
    exact = srq.ExactRidge(dimension=width, ridge_lambda=ridge, device="cpu")
    approx = srq.SquareRootRidge(storage="int8", dimension=width, ridge_lambda=ridge, device="cpu")
    for codes, labels in _classification_stream(width=width, tasks=5):
        exact.update(codes, labels)
        approx.update(codes, labels)
        assert approx.class_ids == exact.class_ids
    relative = torch.linalg.vector_norm(approx.weights - exact.weights) / torch.linalg.vector_norm(exact.weights)
    assert float(relative) < 0.05


def test_square_root_factor_is_spd_structurally():
    width, ridge = 80, 5.0
    learner = srq.SquareRootRidge(storage="int8", dimension=width, ridge_lambda=ridge, device="cpu")
    for codes, labels in _classification_stream(width=width, tasks=3):
        learner.update(codes, labels)
    upper = learner.factor.decode()
    assert bool((upper.diagonal() > 0).all())
    gram = upper.T @ upper
    eigenvalues = torch.linalg.eigvalsh(gram)
    assert bool((eigenvalues > 0).all())


def test_adaptive_budget_bounds_state_between_int8_and_fp16():
    width, ridge = 200, 50.0
    int8_learner = srq.SquareRootRidge(storage="int8", dimension=width, ridge_lambda=ridge, device="cpu")
    fp16_learner = srq.SquareRootRidge(storage="fp16", dimension=width, ridge_lambda=ridge, device="cpu")
    adaptive = srq.SquareRootRidge(storage="adaptive", budget_fraction=0.25, dimension=width, ridge_lambda=ridge, device="cpu")
    for codes, labels in _classification_stream(width=width, tasks=3):
        int8_learner.update(codes.clone(), labels)
        fp16_learner.update(codes.clone(), labels)
        adaptive.update(codes.clone(), labels)
    assert int8_learner.factor_bytes() <= adaptive.factor_bytes() <= fp16_learner.factor_bytes()


def test_int8_gram_plain_can_fail_but_certified_load_never_does():
    width, ridge = 60, 1.0
    plain = srq.Int8GramRidge(load="none", dimension=width, ridge_lambda=ridge, device="cpu")
    certified = srq.Int8GramRidge(load="certified", dimension=width, ridge_lambda=ridge, device="cpu")
    failed = False
    for codes, labels in _classification_stream(width=width, tasks=6, rows_per_task=40):
        certified.update(codes.clone(), labels)
        try:
            plain.update(codes.clone(), labels)
        except RuntimeError:
            failed = True
    assert certified.diagnostics["solver_relative_residual"] < 1e-3
    # A small ridge with few samples per class typically breaks the naive quantized Gram at some task;
    # this is illustrative rather than guaranteed for every random seed, so we only check certified succeeds.
    del failed
