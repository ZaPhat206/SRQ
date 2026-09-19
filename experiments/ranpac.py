"""RanPAC pipeline: random ReLU features, ridge calibration and sample-weighted evaluation.

RanPAC reports, after task ``t``, the accuracy on all evaluation samples of the
classes seen so far, and we keep that definition for every RanPAC experiment.
"""

from __future__ import annotations

import math
import time

import torch

import srq
from experiments import protocol


class Stream:
    """A class-incremental validation stream on the CIFAR-100 training set."""

    def __init__(self, train: dict, *, num_classes: int, num_tasks: int, class_order_seed: int, split_seed: int,
                 validation_fraction: float = 0.2) -> None:
        self.order = protocol.class_order(class_order_seed, num_classes)
        tasks = protocol.task_split(train["labels"], self.order, num_tasks)
        self.training, self.validation = protocol.train_validation_split(train["labels"], tasks, split_seed, validation_fraction)


class Encoder:
    """``x -> ReLU(x W)`` with the first ``width`` columns of a Gaussian matrix."""

    def __init__(self, feature_dim: int, width: int, seed: int, device: torch.device, *, generated_width: int | None = None) -> None:
        self.projection = srq.ranpac_projection(feature_dim, width, seed, generated_width=generated_width).to(device)
        self.device = device

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        return torch.relu(features.to(device=self.device, dtype=torch.float32) @ self.projection)

    def bytes(self) -> int:
        return srq.tensor_bytes(self.projection)


def calibrate_ridge(encoder, features: torch.Tensor, labels: torch.Tensor, fit: torch.Tensor, validation: torch.Tensor,
                    candidates: list[float], num_classes: int, batch_size: int = 256) -> dict:
    """Ridge coefficient with the smallest validation MSE on one-hot targets (dual eigendecomposition)."""
    fit_codes = srq.encode_in_batches(encoder, features, fit, batch_size)
    validation_codes = srq.encode_in_batches(encoder, features, validation, batch_size)
    fit_targets = torch.nn.functional.one_hot(labels[fit].to(device=fit_codes.device, dtype=torch.long), num_classes=num_classes).to(fit_codes.dtype)
    validation_targets = torch.nn.functional.one_hot(
        labels[validation].to(device=validation_codes.device, dtype=torch.long), num_classes=num_classes
    ).to(validation_codes.dtype)
    kernel = fit_codes @ fit_codes.T
    kernel = (kernel + kernel.T) * 0.5
    eigenvalues, eigenvectors = torch.linalg.eigh(kernel)
    eigenvalues.clamp_min_(0)
    coordinates = eigenvectors.T @ fit_targets
    validation_kernel = validation_codes @ fit_codes.T
    scores = []
    for ridge in candidates:
        dual = eigenvectors @ (coordinates / (eigenvalues[:, None] + ridge))
        mse = float(torch.mean((validation_kernel @ dual - validation_targets) ** 2).item())
        if not math.isfinite(mse):
            raise RuntimeError(f"non-finite calibration score at lambda={ridge}")
        scores.append({"ridge_lambda": float(ridge), "validation_mse": mse})
    selected = min(scores, key=lambda item: (item["validation_mse"], item["ridge_lambda"]))
    return {"selected_ridge_lambda": selected["ridge_lambda"], "scores": scores}


def seen_accuracy(encoder, learners: dict, features: torch.Tensor, labels: torch.Tensor, indices: torch.Tensor,
                  batch_size: int = 256) -> dict[str, float]:
    """Accuracy (%) of every learner on the rows ``indices``."""
    correct = {name: 0 for name in learners}
    for start in range(0, len(indices), batch_size):
        batch = indices[start : start + batch_size]
        codes = encoder(features[batch])
        truth = labels[batch].cpu()
        for name, learner in learners.items():
            correct[name] += int((learner.predict(codes) == truth).sum().item())
    return {name: 100.0 * value / len(indices) for name, value in correct.items()}


def run_stream(encoder, learners: dict, features: torch.Tensor, labels: torch.Tensor, training: list[torch.Tensor],
               evaluation: list[torch.Tensor], *, extra_bytes: int, encode_batch_size: int = 256,
               evaluation_batch_size: int = 256, on_task=None) -> dict:
    """Train all learners on the same codes, task by task, and evaluate on the seen evaluation rows."""
    device = next(iter(learners.values())).device
    records = []
    update_seconds = {name: 0.0 for name in learners}
    for task, rows in enumerate(training):
        codes = srq.encode_in_batches(encoder, features, rows, encode_batch_size)
        task_seconds = {}
        for name, learner in learners.items():
            task_seconds[name] = srq.timed_update(learner, codes, labels[rows])
            update_seconds[name] += task_seconds[name]
        del codes
        accuracy = seen_accuracy(encoder, learners, features, labels, torch.cat(evaluation[: task + 1]), evaluation_batch_size)
        record = {"task": task + 1, "accuracy": accuracy, "update_seconds": task_seconds, "state_bytes": {},
                  "solver_relative_residual": {}, "diagnostics": {}}
        for name, learner in learners.items():
            record["state_bytes"][name] = extra_bytes + learner.state_bytes()
            record["solver_relative_residual"][name] = learner.diagnostics["solver_relative_residual"]
            record["diagnostics"][name] = {k: v for k, v in learner.diagnostics.items() if k != "solver_relative_residual"}
        if on_task is not None:
            on_task(task, record)
        records.append(record)
        print(f"task {task + 1}/{len(training)} " + " ".join(f"{n}={a:.3f}" for n, a in accuracy.items()), flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return summarize_records(records, update_seconds)


def summarize_records(records: list[dict], update_seconds: dict[str, float]) -> dict:
    names = list(records[0]["accuracy"])
    return {
        "records": records,
        "aia": {name: sum(r["accuracy"][name] for r in records) / len(records) for name in names},
        "final_accuracy": {name: records[-1]["accuracy"][name] for name in names},
        "final_state_bytes": {name: records[-1]["state_bytes"][name] for name in names},
        "update_seconds": update_seconds,
    }


def square_root(storage: str, width: int, ridge: float, device: torch.device, budget_fraction: float | None = None) -> srq.SquareRootRidge:
    return srq.SquareRootRidge(storage=storage, budget_fraction=budget_fraction, dimension=width, ridge_lambda=ridge, device=device)


def learner(method: str, width: int, ridge: float, device: torch.device, budget_fraction: float = 0.25) -> srq.RidgeLearner:
    if method == "exact":
        return srq.ExactRidge(dimension=width, ridge_lambda=ridge, device=device)
    if method in ("srq_int8", "srq_fp16"):
        return square_root(method[4:], width, ridge, device)
    if method == "srq_adaptive":
        return square_root("adaptive", width, ridge, device, budget_fraction)
    raise ValueError(f"unknown method {method}")


def timed(function, device: torch.device):
    srq.sync(device)
    started = time.perf_counter()
    value = function()
    srq.sync(device)
    return value, time.perf_counter() - started
