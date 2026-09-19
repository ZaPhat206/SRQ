"""Recompute manuscript statistics from the published raw result units.

This reporter performs no training.  Exact rows are converted from the raw
dense state to the paper's lossless upper-triangle convention by removing only
the strict lower triangle of the FP32 Gram tensor.

Sign convention: dAIA = AIA_method - AIA_Exact, matching the manuscript.
A negative value is below Exact and a positive value is above it.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

MB = 1_000_000.0


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def mean(values):
    return statistics.fmean(values)


def std(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def fmt(values, digits=3):
    return f"{mean(values):.{digits}f} +/- {std(values):.{digits}f}"


def exact_upper_state(dense_bytes: int, width: int, element_bytes: int = 4) -> int:
    return int(dense_bytes) - element_bytes * width * (width - 1) // 2


def table1(root: Path) -> None:
    print("Table 1")
    for dataset, label in (("fly_cifar100", "CIFAR-100"), ("fly_cub200", "CUB-200")):
        payload = read(root / "table1" / f"{dataset}_units.json")
        units = payload["units"]
        width = 10_000
        rows = {}
        for method in ("exact", "srq_int8"):
            items = [unit["methods"][method] for unit in units]
            states = [
                exact_upper_state(item["final_state_bytes"], width) if method == "exact" else item["final_state_bytes"]
                for item in items
            ]
            rows[method] = (items, states)
            print(f"  {label:9s} {method:8s} AIA {fmt([i['aia'] for i in items])} "
                  f"state {mean(states) / MB:.1f} MB")
        reduction = 100.0 * (1.0 - mean(rows["srq_int8"][1]) / mean(rows["exact"][1]))
        print(f"    reduction {reduction:.1f}%")


def table2(root: Path) -> None:
    print("Table 2")
    payload = read(root / "table2" / "summary.json")
    width = 10_000
    labels = {
        "exact": "Exact", "int8_gram": "INT8 Gram", "int8_gram_certified": "INT8 Gram + certified load",
        "int8_gram_minimal": "INT8 Gram + minimal load", "srq_fp16": "FP16 factor", "srq_int8": "SRQ-INT8",
    }
    for method, label in labels.items():
        item = payload["methods"][method]
        if item["status"] != "complete":
            print(f"  {label}: ridge solve fails at task {item['failed_task']}")
            continue
        state = item["final_state_bytes"]
        if method == "exact":
            state = exact_upper_state(state, width)
        print(f"  {label}: AIA {item['aia']:.3f}, state {state / MB:.1f} MB")


def table3(root: Path) -> None:
    print("Table 3")
    sources = [
        ("FLY-CL CIFAR-100 width 20,000", "fly_cifar100_20k_units.json", lambda u: True),
        ("RanPAC CIFAR-100 width 10,000", "ranpac_cifar100_units.json", lambda u: u["width"] == 10_000),
        ("RanPAC CIFAR-100 width 20,000", "ranpac_cifar100_units.json", lambda u: u["width"] == 20_000),
        ("RanPAC Stanford Cars width 10,000", "ranpac_cars_units.json", lambda u: True),
    ]
    for title, filename, predicate in sources:
        payload = read(root / "table3" / filename)
        raw_units = [u for u in payload["units"] if predicate(u)]
        if filename.startswith("fly_"):
            units = [{"method": method, **u["methods"][method], "width": payload["width"]}
                     for u in raw_units for method in ("exact", "srq_int8", "srq_adaptive")]
        else:
            units = raw_units
        by_method = {method: [u for u in units if u["method"] == method] for method in ("exact", "srq_int8", "srq_adaptive")}
        exact = by_method["exact"]
        exact_states = [exact_upper_state(u["final_state_bytes"], units[0]["width"]) for u in exact]
        print(f"  {title}")
        for method in ("exact", "srq_int8", "srq_adaptive"):
            items = by_method[method]
            states = [exact_upper_state(u["final_state_bytes"], units[0]["width"]) if method == "exact" else u["final_state_bytes"] for u in items]
            reduction = 0.0 if method == "exact" else 100.0 * (1.0 - mean(states) / mean(exact_states))
            paired = "" if method == "exact" else f", paired dAIA {fmt([u['aia'] - e['aia'] for u, e in zip(items, exact)], digits=4)}"
            print(f"    {method:12s} AIA {fmt([u['aia'] for u in items])}, state {mean(states) / MB:.1f} MB, "
                  f"reduction {reduction:.1f}%{paired}")


def selection(root: Path) -> None:
    print("Selection diagnostic beta=0.05")
    directory = root / "selection_diagnostic"
    exact = {u["stream_id"]: u for u in (read(path) for path in sorted(directory.glob("exact_*.json")))}
    fixed = [read(path) for path in sorted(directory.glob("fixed_int8_*.json"))]
    adaptive = [read(path) for path in sorted(directory.glob("factor_error_*.json"))]
    random_units = [read(path) for path in sorted(directory.glob("random_promotion_*.json"))]
    fixed_delta = [u["validation_aia_percent"] - exact[u["stream_id"]]["validation_aia_percent"] for u in fixed]
    adaptive_delta = [u["validation_aia_percent"] - exact[u["stream_id"]]["validation_aia_percent"] for u in adaptive]
    random_delta = [u["validation_aia_percent"] - exact[u["stream_id"]]["validation_aia_percent"] for u in random_units]
    print(f"  fixed INT8 dAIA: {fmt(fixed_delta)}")
    print(f"  random promotion dAIA: {fmt(random_delta)}")
    print(f"  factor-error criterion dAIA: {fmt(adaptive_delta)}")


def budget_sensitivity(root: Path) -> None:
    print("Budget sensitivity")
    path = root / "budget_sensitivity" / "units.json"
    if not path.exists():
        print("  unavailable: budget_sensitivity/units.json not found")
        return
    payload = read(path)
    units = payload["units"]
    reference_mb = float(payload.get("exact_upper_triangle_reference_mb", 877.5))
    for beta in sorted({float(unit["beta"]) for unit in units}):
        selected = [unit for unit in units if float(unit["beta"]) == beta]
        deltas = [unit["validation_aia_percent"] - unit["exact_validation_aia_percent"] for unit in selected]
        states = [unit["final_total_persistent_bytes"] / MB for unit in selected]
        reduction = 100.0 * (1.0 - mean(states) / reference_mb)
        print(f"  beta {beta:.2f}: dAIA {fmt(deltas)}, state {mean(states):.1f} MB, reduction {reduction:.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results/paper"))
    args = parser.parse_args()
    root = args.results
    table1(root)
    table2(root)
    table3(root)
    selection(root)
    budget_sensitivity(root)


if __name__ == "__main__":
    main()
