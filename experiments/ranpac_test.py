"""RanPAC on the official test sets (RanPAC blocks of Table 3).

CIFAR-100 (ViT-B/16, 10 tasks of 10 classes) at widths 10,000 and 20,000 with
the ridge coefficient calibrated on the validation split, and Stanford Cars
(ResNet-50, 16 classes then nine tasks of 20) at width 10,000 with the ridge
coefficient selected on an 80/20 split of the first task's training data.

    python -m experiments.ranpac_test --config configs/table3_ranpac_cifar100.json \
        --features features/cifar100 --output results/table3_ranpac_cifar100
"""

from __future__ import annotations

import math
from pathlib import Path

import torch

import srq
from experiments import common, protocol, ranpac
from experiments.data import DATASETS, load_features


def select_ridge_first_task(fit_codes: torch.Tensor, fit_labels: torch.Tensor, validation_codes: torch.Tensor,
                            validation_labels: torch.Tensor, candidates: list[float]) -> tuple[float, list[dict]]:
    """Smallest validation MSE of the dual ridge solution in FP64 (ties to the smaller coefficient)."""
    classes = sorted(set(map(int, fit_labels.cpu().tolist())))
    columns = {value: index for index, value in enumerate(classes)}
    device = fit_codes.device
    fit_targets = torch.nn.functional.one_hot(torch.tensor([columns[int(v)] for v in fit_labels.cpu().tolist()]),
                                              num_classes=len(classes)).to(device=device, dtype=torch.float64)
    validation_targets = torch.nn.functional.one_hot(torch.tensor([columns[int(v)] for v in validation_labels.cpu().tolist()]),
                                                     num_classes=len(classes)).to(device=device, dtype=torch.float64)
    fit, validation = fit_codes.to(torch.float64), validation_codes.to(torch.float64)
    gram, cross_kernel = fit @ fit.T, validation @ fit.T
    identity = torch.eye(len(fit), device=device, dtype=torch.float64)
    losses = []
    for ridge in candidates:
        alpha = torch.linalg.solve(gram + float(ridge) * identity, fit_targets)
        loss = float(torch.mean((cross_kernel @ alpha - validation_targets) ** 2).item())
        if not math.isfinite(loss):
            raise RuntimeError("non-finite ridge-selection loss")
        losses.append({"ridge_lambda": float(ridge), "validation_mse": loss})
    selected = min(losses, key=lambda item: (item["validation_mse"], item["ridge_lambda"]))
    return selected["ridge_lambda"], losses


def run_unit(config: dict, replicate: dict, width: int, method: str, train: dict, test: dict, device: torch.device) -> dict:
    spec = DATASETS[config["dataset"]]
    order = protocol.class_order(replicate["class_order_seed"], spec["num_classes"])
    batch = config["encode_batch_size"]
    if config.get("class_increments"):
        training = protocol.increment_split(train["labels"], order, config["class_increments"])
        test_parts = protocol.increment_split(test["labels"], order, config["class_increments"])
        features, labels = train["features"], train["labels"]
        evaluation_features, evaluation_labels = test["features"], test["labels"]
        encoder = ranpac.Encoder(spec["feature_dim"], width, replicate["projection_seed"], device)
        fit, validation = protocol.first_task_split(training[0], seed=replicate["ridge_split_seed"],
                                                    fit_fraction=config["ridge_fit_fraction"])
        ridge, losses = select_ridge_first_task(
            srq.encode_in_batches(encoder, features, fit, batch), labels[fit],
            srq.encode_in_batches(encoder, features, validation, batch), labels[validation], config["ridge_grid"],
        )
    else:
        training = protocol.task_split(train["labels"], order, config["num_tasks"])
        offset = len(train["labels"])
        test_parts = [part + offset for part in protocol.task_split(test["labels"], order, config["num_tasks"])]
        features = evaluation_features = torch.cat((train["features"], test["features"]))
        labels = evaluation_labels = torch.cat((train["labels"], test["labels"]))
        encoder = ranpac.Encoder(spec["feature_dim"], width, replicate["projection_seed"], device, generated_width=config["generated_width"])
        ridge, losses = config["ridge_lambda"][str(width)], None
    learner = ranpac.learner(method, width, ridge, device, config.get("budget_fraction", 0.25))
    records, update_seconds = [], 0.0
    for task, rows in enumerate(training):
        codes = srq.encode_in_batches(encoder, features, rows, batch)
        update_seconds += srq.timed_update(learner, codes, labels[rows])
        del codes
        accuracy = ranpac.seen_accuracy(encoder, {method: learner}, evaluation_features, evaluation_labels,
                                        torch.cat(test_parts[: task + 1]), config["evaluation_batch_size"])[method]
        records.append({"task": task + 1, "accuracy": accuracy, "state_bytes": encoder.bytes() + learner.state_bytes(),
                        "solver_relative_residual": learner.diagnostics["solver_relative_residual"],
                        **{k: v for k, v in learner.diagnostics.items() if k.startswith(("selected_", "total_blocks", "factor_"))}})
        print(f"seed {replicate['class_order_seed']} width {width} {method} task {task + 1}: {accuracy:.3f}", flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    accuracies = [record["accuracy"] for record in records]
    return {
        "replicate": replicate, "width": width, "method": method, "ridge_lambda": ridge, "ridge_selection": losses,
        "class_order": order, "records": records,
        "aia": sum(accuracies) / len(accuracies), "final_accuracy": accuracies[-1],
        "final_state_bytes": records[-1]["state_bytes"], "total_update_seconds": update_seconds,
    }


def main() -> None:
    args = common.base_parser(__doc__).parse_args()
    config = common.read_json(args.config)
    device = torch.device(args.device)
    common.lock_precision()
    train = load_features(args.features, config["dataset"], "train")
    test = load_features(args.features, config["dataset"], "test")
    output = Path(args.output)
    units = []
    for replicate in config["replicates"]:
        for width in config["widths"]:
            for method in config["methods"]:
                path = output / "units" / f"seed_{replicate['class_order_seed']}_width_{width}_{method}.json"
                if not path.is_file():
                    unit = run_unit(config, replicate, width, method, train, test, device)
                    unit["environment"] = common.environment(device)
                    common.write_json(path, unit)
                units.append(common.read_json(path))
    summary = {"experiment": config["experiment"], "blocks": {}}
    for width in config["widths"]:
        block = {}
        exact = {u["replicate"]["class_order_seed"]: u for u in units if u["width"] == width and u["method"] == "exact"}
        for method in config["methods"]:
            selected = sorted((u for u in units if u["width"] == width and u["method"] == method),
                              key=lambda u: u["replicate"]["class_order_seed"])
            block[method] = {metric: common.mean_std([float(u[metric]) for u in selected])
                             for metric in ("aia", "final_accuracy", "final_state_bytes", "total_update_seconds")}
            block[method]["paired_aia_minus_exact"] = common.mean_std(
                [u["aia"] - exact[u["replicate"]["class_order_seed"]]["aia"] for u in selected]
            )
        summary["blocks"][str(width)] = block
    common.write_json(output / "summary.json", summary)


if __name__ == "__main__":
    main()
