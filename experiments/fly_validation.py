"""Quantizing the Gram matrix versus its factor (Table 2).

FLY-CL at width 10,000 on a validation split of the CIFAR-100 training set:
Exact, INT8 Gram (plain, certified load, minimal load), FP16 factor and
SRQ-INT8.  Accuracy after task ``t`` is the FLY-CL task-balanced accuracy on
the validation samples of tasks ``1..t``.

    python -m experiments.fly_validation --config configs/table2_mechanism.json \
        --features features/cifar100 --output results/table2_mechanism
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

import srq
from experiments import common, fly, protocol
from experiments.data import DATASETS, load_features


def validation_stream(config: dict, train: dict) -> tuple[list[int], list[torch.Tensor], list[torch.Tensor]]:
    order = protocol.class_order(config["seed"], DATASETS[config["dataset"]]["num_classes"])
    tasks = protocol.task_split(train["labels"], order, config["num_tasks"])
    training, validation = protocol.train_validation_split(train["labels"], tasks, config["seed"], config["validation_fraction"])
    return order, training, validation


def run_method(method: str, config: dict, train: dict, cache: fly.CodeCache, training: list[torch.Tensor],
               validation: list[torch.Tensor], device: torch.device) -> dict:
    learner = fly.make_learner(method, width=config["width"], ridge_lambda=config["ridge_lambda"], device=device, config=config)
    stages, tasks = [], []
    started = time.perf_counter()
    for task, rows in enumerate(training):
        codes = cache.dense(rows)
        try:
            seconds = srq.timed_update(learner, codes, train["labels"][rows])
        except RuntimeError as error:
            if not method.startswith("int8_gram"):
                raise
            return {"method": method, "status": "failed", "failed_task": task + 1, "error": str(error),
                    "stage_accuracy": stages, "tasks": tasks}
        del codes
        stages.append(fly.stage_accuracy(learner, validation, task, cache, train["labels"], config.get("evaluation_batch_size", 256)))
        tasks.append({
            "task": task + 1, "update_seconds": seconds,
            "state_bytes": cache.projection_bytes() + learner.state_bytes(),
            **learner.diagnostics,
        })
        print(f"{method} task {task + 1}/{len(training)} accuracy={stages[-1]:.4f}", flush=True)
    return {
        "method": method, "status": "complete",
        "aia": sum(stages) / len(stages), "final_accuracy": stages[-1], "stage_accuracy": stages,
        "final_state_bytes": tasks[-1]["state_bytes"],
        "total_update_seconds": sum(item["update_seconds"] for item in tasks),
        "seconds": time.perf_counter() - started, "tasks": tasks,
    }


def main() -> None:
    args = common.base_parser(__doc__).parse_args()
    config = common.read_json(args.config)
    device = torch.device(args.device)
    common.lock_precision()
    train = load_features(args.features, config["dataset"], "train")
    order, training, validation = validation_stream(config, train)
    cache = fly.CodeCache(train["features"], width=config["width"], synaptic_degree=config["synaptic_degree"],
                          coding_level=config["coding_level"], seed=config["projection_seed"], device=device)
    output = Path(args.output)
    results = {}
    for method in config["methods"]:
        path = output / f"{method}.json"
        if path.is_file():
            results[method] = common.read_json(path)
            continue
        results[method] = {**run_method(method, config, train, cache, training, validation, device),
                           "class_order": order, "environment": common.environment(device)}
        common.write_json(path, results[method])
        if device.type == "cuda":
            torch.cuda.empty_cache()
    summary = {"experiment": config["experiment"], "methods": {}}
    for method, result in results.items():
        entry = {"status": result["status"]}
        if result["status"] == "complete":
            entry.update(aia=result["aia"], final_accuracy=result["final_accuracy"], final_state_bytes=result["final_state_bytes"])
            loads = [task.get("diagonal_load", 0.0) for task in result["tasks"]]
            if method in ("int8_gram_certified", "int8_gram_minimal"):
                ridge = config["ridge_lambda"]
                entry.update(load_over_lambda=[load / ridge for load in loads],
                             effective_ridge_over_lambda=[(ridge + load) / ridge for load in loads])
        else:
            entry["failed_task"] = result["failed_task"]
        summary["methods"][method] = entry
    common.write_json(output / "summary.json", summary)


if __name__ == "__main__":
    main()
