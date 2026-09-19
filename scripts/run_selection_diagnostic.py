"""Run the train-only beta=0.05 selection diagnostic."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import srq
from experiments import common, protocol, ranpac
from experiments.data import DATASETS, load_features


def run_unit(config, stream, method, train, device, allocation_seed=None):
    spec = DATASETS[config["dataset"]]
    order = protocol.class_order(stream["class_order_seed"], spec["num_classes"])
    tasks = protocol.task_split(train["labels"], order, config["num_tasks"])
    training, validation = protocol.train_validation_split(train["labels"], tasks, stream["split_seed"], config["validation_fraction"])
    encoder = ranpac.Encoder(spec["feature_dim"], config["width"], stream["projection_seed"], device)
    if method == "exact":
        learner = srq.ExactRidge(dimension=config["width"], ridge_lambda=config["ridge_lambda"], device=device)
    elif method == "fixed_int8":
        learner = srq.SquareRootRidge(storage="int8", dimension=config["width"], ridge_lambda=config["ridge_lambda"], device=device)
    else:
        policy = "factor_error" if method == "factor_error" else "random_promotion"
        learner = srq.SelectionControlRidge(
            selection_policy=policy, allocation_seed=allocation_seed or 2025,
            budget_fraction=config["budget_fraction"], block_size=config["block_size"],
            group_size=config["group_size"], panel_size=config["panel_size"],
            dimension=config["width"], ridge_lambda=config["ridge_lambda"], device=device,
        )
    records = []
    for task, rows in enumerate(training):
        codes = srq.encode_in_batches(encoder, train["features"], rows, 256)
        srq.timed_update(learner, codes, train["labels"][rows])
        del codes
        accuracy = ranpac.seen_accuracy(encoder, {"learner": learner}, train["features"], train["labels"], torch.cat(validation[: task + 1]), 256)["learner"]
        print(f"stream {stream['stream_id']} {method} task {task + 1}/{len(training)}: {accuracy:.3f}", flush=True)
        records.append({"task": task + 1, "validation_accuracy_percent": accuracy,
                        "total_persistent_bytes": encoder.bytes() + learner.state_bytes(),
                        "solver_relative_residual": learner.diagnostics["solver_relative_residual"]})
    return {"schema_version": 1, "status": "complete", "uses_test_set": False,
            "unit_id": f"{stream['stream_id']}__{method}", "stream_id": stream["stream_id"],
            "method": method, "budget": config["budget_fraction"] if method not in ("exact", "fixed_int8") else None,
            "allocation_seed": allocation_seed, "stream": stream, "records": records,
            "validation_aia_percent": sum(r["validation_accuracy_percent"] for r in records) / len(records),
            "final_validation_accuracy_percent": records[-1]["validation_accuracy_percent"],
            "final_total_persistent_bytes": records[-1]["total_persistent_bytes"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    common.lock_precision()
    train = load_features(args.features, config["dataset"], "train")
    args.output.mkdir(parents=True, exist_ok=True)
    for stream in config["streams"]:
        methods = [("exact", None), ("fixed_int8", None), ("factor_error", None)]
        methods.extend(("random_promotion", seed) for seed in config["random_allocation_seeds"])
        for method, seed in methods:
            suffix = f"_{seed}" if seed is not None else ""
            path = args.output / f"{method}_{stream['stream_id']}{suffix}.json"
            if path.exists():
                print(f"restored {path}", flush=True)
                continue
            unit = run_unit(config, stream, method, train, device, seed)
            path.write_text(json.dumps(unit, indent=2) + "\n", encoding="utf-8")
            print(f"  wrote {path.name}: validation AIA {unit['validation_aia_percent']:.3f}", flush=True)


if __name__ == "__main__":
    main()
