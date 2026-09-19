"""Class orders, task splits and validation splits.

Every function is deterministic given its seeds.  Index tensors refer to rows
of a feature cache, whose sample order is fixed by the files in ``orders/``.
"""

from __future__ import annotations

import random

import torch


def class_order(seed: int, num_classes: int) -> list[int]:
    """Random permutation of the class ids."""
    return random.Random(seed).sample(list(range(num_classes)), num_classes)


def task_split(labels: torch.Tensor, order: list[int], num_tasks: int) -> list[torch.Tensor]:
    """Row indices of every task when the classes of ``order`` are split into equal tasks."""
    per_task = len(order) // num_tasks
    return [torch.isin(labels, torch.tensor(order[i * per_task : (i + 1) * per_task])).nonzero().flatten() for i in range(num_tasks)]


def increment_split(labels: torch.Tensor, order: list[int], increments: list[int]) -> list[torch.Tensor]:
    """Row indices of every task for a schedule of class increments (e.g. 16 + 9 x 20)."""
    if sum(increments) != len(order):
        raise ValueError("increments must cover the class order")
    parts, offset = [], 0
    for increment in increments:
        classes = torch.tensor(order[offset : offset + increment], dtype=labels.dtype)
        parts.append(torch.nonzero(torch.isin(labels.cpu(), classes), as_tuple=False).squeeze(1))
        offset += increment
    return parts


def train_validation_split(labels: torch.Tensor, task_indices: list[torch.Tensor], seed: int,
                           fraction: float) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Class-stratified split of every task into training and validation rows.

    One generator seeded with ``seed`` visits the tasks in order and the classes
    of a task in increasing id; ``max(1, int(n * fraction))`` rows of a class
    with ``n`` rows go to validation.
    """
    if not 0 < fraction < 1:
        raise ValueError("fraction must lie in (0, 1)")
    generator = torch.Generator().manual_seed(seed)
    training, validation = [], []
    for indices in task_indices:
        task_labels = labels[indices]
        train_parts, validation_parts = [], []
        for value in torch.unique(task_labels):
            class_indices = indices[task_labels == value]
            permuted = class_indices[torch.randperm(len(class_indices), generator=generator)]
            count = max(1, int(len(permuted) * fraction))
            validation_parts.append(permuted[:count])
            train_parts.append(permuted[count:])
        training.append(torch.cat(train_parts))
        validation.append(torch.cat(validation_parts))
    return training, validation


def nested_split(labels: torch.Tensor, order: list[int], num_tasks: int, *, split_seed: int,
                 outer_fraction: float = 0.2, inner_fraction: float = 0.2) -> dict[str, list[torch.Tensor]]:
    """Class-stratified nested split used to select the ridge coefficient of FLY-CL.

    Each class is permuted with its own generator (seed ``1000 split_seed +
    class``); ``round(n outer_fraction)`` rows form the outer validation set,
    and ``round(n' inner_fraction)`` of the remaining ``n'`` rows form the inner
    validation set.  Returns per-task index lists for ``inner_fit``,
    ``inner_validation``, ``outer_fit`` and ``outer_validation``.
    """
    per_class = {}
    for value in sorted(map(int, torch.unique(labels).tolist())):
        indices = torch.nonzero(labels == value).flatten()
        generator = torch.Generator().manual_seed(split_seed * 1000 + value)
        indices = indices[torch.randperm(len(indices), generator=generator)]
        outer = max(1, int(round(len(indices) * outer_fraction)))
        development, outer_validation = indices[outer:], indices[:outer]
        inner = max(1, int(round(len(development) * inner_fraction)))
        per_class[value] = (development[inner:], development[:inner], development, outer_validation)
    per_task = len(order) // num_tasks
    names = ("inner_fit", "inner_validation", "outer_fit", "outer_validation")
    parts: dict[str, list[torch.Tensor]] = {name: [] for name in names}
    for task in range(num_tasks):
        classes = order[task * per_task : (task + 1) * per_task]
        for position, name in enumerate(names):
            parts[name].append(torch.cat([per_class[value][position] for value in classes]))
    return parts


def calibration_split(labels: torch.Tensor, training_parts: list[torch.Tensor], *, seed: int,
                      fit_per_class: int = 20, validation_per_class: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-class fitting and validation rows for the RanPAC ridge calibration (generator seed ``seed + 404``)."""
    allowed = torch.cat(training_parts).cpu()
    allowed_labels = labels[allowed].cpu()
    generator = torch.Generator().manual_seed(seed + 404)
    required = fit_per_class + validation_per_class
    fit, validation = [], []
    for value in sorted(map(int, torch.unique(allowed_labels).tolist())):
        class_indices = allowed[allowed_labels == value]
        if len(class_indices) < required:
            raise ValueError(f"class {value} has fewer than {required} rows")
        selected = class_indices[torch.randperm(len(class_indices), generator=generator)[:required]]
        fit.append(selected[:fit_per_class])
        validation.append(selected[fit_per_class:])
    return torch.cat(fit), torch.cat(validation)


def first_task_split(first_task: torch.Tensor, *, seed: int, fit_fraction: float = 0.8) -> tuple[torch.Tensor, torch.Tensor]:
    """Random fitting/validation split of the first task (ridge selection on Stanford Cars)."""
    values = first_task.cpu().tolist()
    random.Random(seed).shuffle(values)
    boundary = int(len(values) * fit_fraction)
    if not 0 < boundary < len(values):
        raise ValueError("empty ridge-selection split")
    return torch.tensor(values[:boundary]), torch.tensor(values[boundary:])
