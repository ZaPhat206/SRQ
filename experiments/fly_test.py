"""FLY-CL on the official test sets (Table 1 and the FLY-CL block of Table 3).

For every replicate (class order and projection seed), all methods of the
configuration are trained on the full training set, task by task, on the same
WTA codes, and are evaluated on the test samples of all tasks seen so far.

    python -m experiments.fly_test --config configs/table1_fly_cifar100.json \
        --features features/cifar100 --output results/table1_fly_cifar100
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

import srq
from experiments import common, fly, protocol
from experiments.data import DATASETS, load_features


def run_replicate(config: dict, replicate: dict, train: dict, test: dict, device: torch.device) -> dict:
    dataset = config["dataset"]
    num_classes = DATASETS[dataset]["num_classes"]
    order = protocol.class_order(replicate["class_order_seed"], num_classes)
    labels = torch.cat((train["labels"], test["labels"]))
    offset = len(train["labels"])
    training_parts = protocol.task_split(train["labels"], order, config["num_tasks"])
    test_parts = [part + offset for part in protocol.task_split(test["labels"], order, config["num_tasks"])]

    width = config["width"]
    cache = fly.CodeCache(
        torch.cat((train["features"], test["features"])), width=width, synaptic_degree=config["synaptic_degree"],
        coding_level=config["coding_level"], seed=replicate["projection_seed"], device=device,
        batch_size=config.get("encode_batch_size", 256),
    )
    methods = config["methods"]
    learners = {name: fly.make_learner(name, width=width, ridge_lambda=config["ridge_lambda"], device=device, config=config)
                for name in methods}
    record = {name: {"accuracy_matrix": [], "state_bytes": [], "update_seconds": [], "inference_seconds": [],
                     "solver_relative_residual": [], "diagnostics": []} for name in methods}
    comparisons = {name: {"prediction_agreement": [], "relative_logit_error": []} for name in methods if name != "exact"}
    batch = config.get("evaluation_batch_size", 256)

    for task, rows in enumerate(training_parts):
        codes = cache.dense(rows)
        task_labels = labels[rows]
        for name, learner in learners.items():
            record[name]["update_seconds"].append(srq.timed_update(learner, codes, task_labels))
            learner.check_sample_free()
        del codes
        class_ids = learners[methods[0]].class_ids
        if any(learner.class_ids != class_ids for learner in learners.values()):
            raise AssertionError("class columns differ between methods")
        predictions, logits = {}, {}
        for name, learner in learners.items():
            srq.sync(device)
            started = time.perf_counter()
            row, predictions[name], logits[name] = fly.task_predictions(learner, test_parts, task, cache, labels, batch)
            srq.sync(device)
            item = record[name]
            item["inference_seconds"].append(time.perf_counter() - started)
            item["accuracy_matrix"].append(row)
            item["state_bytes"].append(cache.projection_bytes() + learner.state_bytes())
            item["solver_relative_residual"].append(learner.diagnostics["solver_relative_residual"])
            item["diagnostics"].append({k: v for k, v in learner.diagnostics.items() if k != "solver_relative_residual"})
        if "exact" in learners:
            for name in comparisons:
                comparisons[name]["prediction_agreement"].append(fly.prediction_agreement(predictions["exact"], predictions[name]))
                comparisons[name]["relative_logit_error"].append(fly.relative_logit_error(logits["exact"], logits[name]))
        print(f"replicate {replicate['class_order_seed']} task {task + 1}/{len(training_parts)} "
              + " ".join(f"{name}={sum(record[name]['accuracy_matrix'][-1]) / (task + 1):.3f}" for name in methods), flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result = {"replicate": replicate, "class_order": order, "methods": {}}
    for name in methods:
        item = record[name]
        result["methods"][name] = {
            **fly.accuracy_summary(item["accuracy_matrix"]),
            **item,
            "final_state_bytes": item["state_bytes"][-1],
            "total_update_seconds": sum(item["update_seconds"]),
            **comparisons.get(name, {}),
        }
    return result


def summarize(config: dict, units: list[dict]) -> dict:
    summary = {"experiment": config["experiment"], "dataset": config["dataset"], "width": config["width"], "methods": {}}
    for name in config["methods"]:
        results = [unit["methods"][name] for unit in units]
        summary["methods"][name] = {
            metric: common.mean_std([float(result[metric]) for result in results])
            for metric in ("aia", "final_accuracy", "final_state_bytes", "total_update_seconds")
        }
        if name != "exact" and "exact" in config["methods"]:
            exact = [unit["methods"]["exact"] for unit in units]
            summary["methods"][name]["paired_aia_minus_exact"] = common.mean_std(
                [result["aia"] - reference["aia"] for result, reference in zip(results, exact)]
            )
    return summary


def main() -> None:
    args = common.base_parser(__doc__).parse_args()
    config = common.read_json(args.config)
    device = torch.device(args.device)
    common.lock_precision()
    train = load_features(args.features, config["dataset"], "train")
    test = load_features(args.features, config["dataset"], "test")
    output = Path(args.output)
    units = []
    for index, replicate in enumerate(config["replicates"]):
        path = output / f"replicate_{index}.json"
        if path.is_file():
            units.append(common.read_json(path))
            print(f"restored {path}", flush=True)
            continue
        unit = run_replicate(config, replicate, train, test, device)
        unit["environment"] = common.environment(device)
        common.write_json(path, unit)
        units.append(unit)
    common.write_json(output / "summary.json", summarize(config, units))


if __name__ == "__main__":
    main()
