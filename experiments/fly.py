"""FLY-CL pipeline: WTA code caches, learners and the FLY-CL evaluation.

FLY-CL reports, after task ``t``, the mean over the tasks seen so far of the
accuracy on each task's evaluation samples (task-balanced accuracy), and we
keep that definition for every FLY-CL experiment.
"""

from __future__ import annotations

import math
import statistics

import torch

import srq


class CodeCache:
    """Active WTA indices and values of every row of a feature matrix (kept on the CPU)."""

    def __init__(self, features: torch.Tensor, *, width: int, synaptic_degree: int, coding_level: float,
                 seed: int, device: torch.device, batch_size: int = 256) -> None:
        self.width = int(width)
        self.projection = srq.fly_projection(features.shape[1], width, synaptic_degree, seed, device)
        self.indices, self.values = srq.fly_code_cache(self.projection, features, coding_level, batch_size=batch_size)
        self.device = device

    def dense(self, rows: torch.Tensor) -> torch.Tensor:
        return srq.dense_codes(self.indices[rows], self.values[rows], self.width, self.device)

    def projection_bytes(self) -> int:
        return srq.tensor_bytes(self.projection)


def make_learner(method: str, *, width: int, ridge_lambda: float, device: torch.device, config: dict) -> srq.RidgeLearner:
    common = {"dimension": width, "ridge_lambda": ridge_lambda, "device": device}
    storage = {"block_size": config.get("block_size", 256), "group_size": config.get("group_size", 64)}
    if method == "exact":
        return srq.ExactRidge(**common)
    if method in ("srq_int8", "srq_fp16"):
        return srq.SquareRootRidge(storage=method[4:], panel_size=config.get("panel_size", 128), **storage, **common)
    if method == "srq_adaptive":
        return srq.SquareRootRidge(storage="adaptive", budget_fraction=config.get("budget_fraction", 0.25),
                                   panel_size=config.get("panel_size", 128), **storage, **common)
    if method == "srq_int8_cholesky":
        return srq.SquareRootRidge(storage="int8", factor_update="cholesky", **storage, **common)
    if method in ("int8_gram", "int8_gram_certified", "int8_gram_minimal"):
        load = {"int8_gram": "none", "int8_gram_certified": "certified", "int8_gram_minimal": "minimal"}[method]
        return srq.Int8GramRidge(load=load, **storage, **common)
    raise ValueError(f"unknown method {method}")


def task_predictions(learner: srq.RidgeLearner, parts: list[torch.Tensor], task: int, cache: CodeCache,
                     labels: torch.Tensor, batch_size: int = 256) -> tuple[list[float], list[torch.Tensor], list[torch.Tensor]]:
    """Per-task accuracies (%), predictions and FP64 logits on the evaluation parts of tasks ``0..task``."""
    accuracies, predictions, logits = [], [], []
    weights = learner.weights
    for previous in range(task + 1):
        indices = parts[previous]
        task_predictions_, task_logits = [], []
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            values = cache.dense(selected).to(weights.dtype) @ weights
            columns = values.argmax(1).detach().cpu().tolist()
            task_predictions_.append(torch.tensor([learner.class_ids[column] for column in columns]))
            task_logits.append(values.detach().cpu().to(torch.float64))
        predicted = torch.cat(task_predictions_)
        accuracies.append(100.0 * float((predicted == labels[indices].cpu()).float().mean()))
        predictions.append(predicted)
        logits.append(torch.cat(task_logits))
    return accuracies, predictions, logits


def task_accuracy(learner: srq.RidgeLearner, indices: torch.Tensor, cache: CodeCache, labels: torch.Tensor,
                  batch_size: int = 256) -> float:
    """Accuracy (%) on one task's evaluation rows, counted over batches."""
    correct = 0
    for start in range(0, len(indices), batch_size):
        selected = indices[start : start + batch_size]
        columns = (cache.dense(selected).to(learner.weights.dtype) @ learner.weights).argmax(1).detach().cpu().tolist()
        predictions = torch.tensor([learner.class_ids[column] for column in columns])
        correct += int((predictions == labels[selected].cpu()).sum().item())
    return 100.0 * correct / max(len(indices), 1)


def stage_accuracy(learner: srq.RidgeLearner, parts: list[torch.Tensor], task: int, cache: CodeCache,
                   labels: torch.Tensor, batch_size: int = 256) -> float:
    """Task-balanced accuracy after ``task``: mean of the per-task accuracies of tasks ``0..task``."""
    values = [task_accuracy(learner, parts[previous], cache, labels, batch_size) for previous in range(task + 1)]
    return sum(values) / len(values)


def paired_stage_metrics(exact: srq.RidgeLearner, other: srq.RidgeLearner, parts: list[torch.Tensor], task: int,
                         cache: CodeCache, labels: torch.Tensor, batch_size: int = 256) -> dict:
    """Task-balanced accuracies of two learners after ``task`` (used by the ridge selection)."""
    exact_accuracy, other_accuracy = [], []
    for previous in range(task + 1):
        indices = parts[previous]
        exact_correct = other_correct = rows = 0
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            codes = cache.dense(selected).to(exact.weights.dtype)
            truth = labels[selected].cpu()
            for learner, counter in ((exact, "exact"), (other, "other")):
                columns = (codes.to(learner.weights.dtype) @ learner.weights).argmax(1).detach().cpu().tolist()
                correct = int((torch.tensor([learner.class_ids[c] for c in columns]) == truth).sum())
                if counter == "exact":
                    exact_correct += correct
                else:
                    other_correct += correct
            rows += len(selected)
        exact_accuracy.append(100.0 * exact_correct / max(rows, 1))
        other_accuracy.append(100.0 * other_correct / max(rows, 1))
    return {"exact": sum(exact_accuracy) / len(exact_accuracy), "other": sum(other_accuracy) / len(other_accuracy)}


def accuracy_summary(matrix: list[list[float]]) -> dict:
    """Stage accuracies, AIA and final accuracy of an accuracy matrix (row t: tasks 0..t)."""
    stages = [statistics.fmean(row) for row in matrix]
    return {"stage_accuracy": stages, "aia": statistics.fmean(stages), "final_accuracy": statistics.fmean(matrix[-1])}


def prediction_agreement(reference: list[torch.Tensor], other: list[torch.Tensor]) -> float:
    return statistics.fmean([float((a == b).float().mean()) for a, b in zip(reference, other)])


def relative_logit_error(reference: list[torch.Tensor], other: list[torch.Tensor]) -> float:
    numerator = sum(float(((b - a) ** 2).sum()) for a, b in zip(reference, other))
    denominator = max(sum(float((a ** 2).sum()) for a in reference), 1.0)
    return math.sqrt(numerator / denominator)
