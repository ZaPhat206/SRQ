"""Ridge-coefficient selection for FLY-CL by nested validation on the training set.

20% of every class is held out, the rest is split again into 80% fitting and
20% inner validation, and every candidate is scored by the inner validation
AIA averaged over three development replicates (class order and projection).
For the full-width models the score of a replicate is the mean of the AIA of
Exact and SRQ-INT8; for the reduced-width Exact model it is its own AIA.  A
candidate is discarded when a solve fails or its relative solver residual
exceeds the tolerance.  The largest mean score wins, ties going to the larger
coefficient.  SRQ-INT8 is computed here with the Cholesky form of the factor
update, as in the original selection runs (the factor is the same up to rounding).

    python -m experiments.fly_selection --config configs/lambda_fly_cifar100.json \
        --features features/cifar100 --output results/lambda_fly_cifar100
"""

from __future__ import annotations

import statistics
from pathlib import Path

import torch

from experiments import common, fly, protocol
from experiments.data import DATASETS, load_features


def exact_residual(learner, ridge: float) -> float:
    """Relative residual of the unsymmetrized system ``(G + lambda I) W = B``."""
    system = learner.gram + ridge * torch.eye(learner.dimension, device=learner.device, dtype=learner.dtype)
    residual = torch.linalg.vector_norm(system @ learner.weights - learner.cross) / max(
        float(torch.linalg.vector_norm(learner.cross)), 1.0
    )
    return float(residual)


def run_unit(config: dict, ridge: float, train: dict, cache: fly.CodeCache, fit: list[torch.Tensor],
             validation: list[torch.Tensor], device: torch.device) -> dict:
    width = config["width"]
    exact = fly.make_learner("exact", width=width, ridge_lambda=ridge, device=device, config=config)
    other = None
    if config["family"] == "exact_and_srq":
        other = fly.make_learner("srq_int8_cholesky", width=width, ridge_lambda=ridge, device=device, config=config)
    stages_exact, stages_other, residuals_exact, residuals_other = [], [], [], []
    batch = config.get("evaluation_batch_size", 256)
    try:
        for task, rows in enumerate(fit):
            codes = cache.dense(rows)
            labels = train["labels"][rows]
            exact.update(codes, labels)
            residuals_exact.append(exact_residual(exact, ridge))
            if other is not None:
                other.update(codes, labels)
                residuals_other.append(other.diagnostics["solver_relative_residual"])
                metrics = fly.paired_stage_metrics(exact, other, validation, task, cache, train["labels"], batch)
                stages_exact.append(metrics["exact"])
                stages_other.append(metrics["other"])
            else:
                stages_exact.append(fly.stage_accuracy(exact, validation, task, cache, train["labels"], batch))
            del codes
    except (RuntimeError, torch.linalg.LinAlgError) as error:
        return {"status": "failed", "error": str(error)}
    unit = {"status": "complete", "exact_aia": sum(stages_exact) / len(stages_exact),
            "exact_maximum_residual": max(residuals_exact)}
    if other is not None:
        unit.update(srq_aia=sum(stages_other) / len(stages_other), srq_maximum_residual=max(residuals_other))
    return unit


def main() -> None:
    args = common.base_parser(__doc__).parse_args()
    config = common.read_json(args.config)
    device = torch.device(args.device)
    common.lock_precision()
    train = load_features(args.features, config["dataset"], "train")
    num_classes = DATASETS[config["dataset"]]["num_classes"]
    output = Path(args.output)
    grid = list(map(float, config["ridge_grid"]))

    def unit_path(ridge: float, index: int) -> Path:
        return output / "units" / f"lambda_{ridge:g}_replicate_{index}.json"

    for index, replicate in enumerate(config["development_replicates"]):
        if all(unit_path(ridge, index).is_file() for ridge in grid):
            continue
        order = protocol.class_order(replicate["class_order_seed"], num_classes)
        parts = protocol.nested_split(train["labels"], order, config["num_tasks"], split_seed=config["split_seed"],
                                      outer_fraction=config["outer_fraction"], inner_fraction=config["inner_fraction"])
        cache = fly.CodeCache(train["features"], width=config["width"], synaptic_degree=config["synaptic_degree"],
                              coding_level=config["coding_level"], seed=replicate["projection_seed"], device=device)
        for ridge in grid:
            if unit_path(ridge, index).is_file():
                continue
            unit = {"ridge_lambda": ridge, "replicate": replicate,
                    **run_unit(config, ridge, train, cache, parts["inner_fit"], parts["inner_validation"], device)}
            common.write_json(unit_path(ridge, index), unit)
            print(f"lambda={ridge:g} replicate={index} {unit['status']}", flush=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
        del cache

    candidates = []
    tolerance = config["maximum_solver_relative_residual"]
    for ridge in grid:
        units = [common.read_json(unit_path(ridge, index)) for index in range(len(config["development_replicates"]))]
        valid = all(
            unit["status"] == "complete"
            and unit["exact_maximum_residual"] <= tolerance
            and unit.get("srq_maximum_residual", 0.0) <= tolerance
            for unit in units
        )
        score = None
        if valid:
            per_replicate = [(u["exact_aia"] + u["srq_aia"]) / 2 if "srq_aia" in u else u["exact_aia"] for u in units]
            score = statistics.fmean(per_replicate)
        candidates.append({"ridge_lambda": ridge, "valid": valid, "score": score, "units": units})
    best = max((c for c in candidates if c["valid"]), key=lambda c: (c["score"], c["ridge_lambda"]))
    common.write_json(output / "selection.json", {
        "experiment": config["experiment"], "selected_ridge_lambda": best["ridge_lambda"],
        "candidates": [{k: v for k, v in c.items() if k != "units"} for c in candidates],
    })
    print(f"selected lambda = {best['ridge_lambda']:g}", flush=True)


if __name__ == "__main__":
    main()
