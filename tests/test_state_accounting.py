from __future__ import annotations

import torch

import srq
from srq.state_accounting import persistent_state_bytes, reported_exact_state_bytes


def test_exact_upper_triangle_accounting_uses_persistent_gram():
    learner = srq.ExactRidge(dimension=12, ridge_lambda=2.0)
    learner.update(torch.randn(20, 12, generator=torch.Generator().manual_seed(7)), torch.arange(20) % 4)
    assert persistent_state_bytes(learner, exact_upper_triangle=True) == reported_exact_state_bytes(
        persistent_state_bytes(learner), 12
    )


def test_selection_control_keeps_factor_spd():
    learner = srq.SelectionControlRidge(selection_policy="factor_error", dimension=24, ridge_lambda=2.0)
    learner.update(torch.randn(30, 24, generator=torch.Generator().manual_seed(8)), torch.arange(30) % 5)
    factor = learner.factor.decode()
    assert bool((factor.diagonal() > 0).all())
    assert bool((torch.linalg.eigvalsh(factor.T @ factor) > 0).all())
