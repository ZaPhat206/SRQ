"""Run one paper artifact per command.

    python scripts/reproduce.py tables     # recompute every table from the published units
    python scripts/reproduce.py figure2    # redraw Figure 2 from the published units
    python scripts/reproduce.py table1     # retrain Table 1 on CIFAR-100 and CUB-200
    python scripts/reproduce.py all        # every training target whose features are present

The training targets are thin wrappers over the ``scripts/run_*.py`` entry
points.  Every command is printed before it runs, so any single run can also be
copied and executed on its own.  A target whose feature cache is missing is
reported and skipped.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

MB = 1_000_000.0

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Run:
    """One training command: a runner, its configuration and its feature cache."""

    script: str
    config: str
    features: str
    output: str
    extra: tuple[str, ...] = ()
    splits: tuple[str, ...] = ("train", "test")


TARGETS: dict[str, tuple[Run, ...]] = {
    "table1": (
        Run("run_table1.py", "table1_fly_cifar100.json", "cifar100", "table1_fly_cifar100"),
        Run("run_table1.py", "table1_fly_cub200.json", "cub200", "table1_fly_cub200"),
    ),
    "table2": (
        Run("run_table2.py", "table2_mechanism.json", "cifar100", "table2_mechanism"),
    ),
    "table3": (
        Run("run_table3.py", "table3_fly_cifar100_20k.json", "cifar100", "table3_fly_cifar100_20k",
            ("--family", "fly")),
        Run("run_table3.py", "table3_ranpac_cifar100.json", "cifar100", "table3_ranpac_cifar100",
            ("--family", "ranpac")),
        Run("run_table3.py", "table3_ranpac_cars.json", "cars", "table3_ranpac_cars",
            ("--family", "ranpac")),
    ),
    "table4": (
        Run("run_selection_diagnostic.py", "selection_diagnostic_beta_0.05.json", "cifar100",
            "selection_diagnostic", splits=("train",)),
    ),
}

ORDER = ("table1", "table2", "table3", "table4")


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def command(run: Run, features: Path, results: Path, device: str) -> list[str]:
    return [
        sys.executable, str(ROOT / "scripts" / run.script), *run.extra,
        "--config", str(ROOT / "configs" / "paper" / run.config),
        "--features", str(features / run.features),
        "--output", str(results / run.output),
        "--device", device,
    ]


def missing_splits(run: Run, features: Path) -> list[str]:
    return [split for split in run.splits if not (features / run.features / f"{split}.pt").is_file()]


def execute(argv: list[str], dry_run: bool) -> int:
    print("$ " + subprocess.list2cmdline(argv), flush=True)
    return 0 if dry_run else subprocess.call(argv)


def report(output: Path) -> None:
    """Print the summary a finished run just wrote, in the reporter's layout."""
    path = output / "summary.json"
    if not path.is_file():
        units = sorted(output.rglob("*.json"))
        print(f"{output}: {len(units)} unit files, no summary" if units else f"{output}: no output", flush=True)
        return
    summary = json.loads(path.read_text(encoding="utf-8-sig"))
    blocks = summary.get("blocks") or {"": summary.get("methods", {})}
    print(f"\n{path}")
    for width, methods in blocks.items():
        if width:
            print(f"  width {int(width):,}")
        indent = "    " if width else "  "
        for name, values in methods.items():
            line = (f"{indent}{name:13s} AIA {values['aia']['mean']:.3f} +/- {values['aia']['std']:.3f}"
                    f"  state {values['final_state_bytes']['mean'] / MB:.1f} MB")
            paired = values.get("paired_aia_minus_exact")
            if paired:
                line += f"  dAIA {paired['mean']:+.4f} +/- {paired['std']:.4f}"
            print(line)
    if any("exact" in methods for methods in blocks.values()):
        print("  state is the stored total; the paper converts Exact to its upper triangle", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", choices=("tables", "figure2", *ORDER, "all"))
    parser.add_argument("--features", type=Path, default=Path("features"), help="root of the feature caches")
    parser.add_argument("--results", type=Path, default=Path("results"), help="root of the run outputs")
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:N or cpu")
    parser.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    args = parser.parse_args()

    if args.target == "tables":
        raise SystemExit(execute(
            [sys.executable, str(ROOT / "scripts" / "make_tables.py"),
             "--results", str(ROOT / "results" / "paper")], args.dry_run))
    if args.target == "figure2":
        raise SystemExit(execute(
            [sys.executable, str(ROOT / "scripts" / "make_budget_sensitivity_figure.py"),
             "--results", str(ROOT / "results" / "paper" / "budget_sensitivity" / "units.json"),
             "--output", str(ROOT / "generated" / "fig_budget_sensitivity.pdf")], args.dry_run))

    device = resolve_device(args.device)
    selected = ORDER if args.target == "all" else (args.target,)
    skipped = []
    for name in selected:
        for run in TARGETS[name]:
            absent = missing_splits(run, args.features)
            if absent:
                paths = ", ".join(f"{args.features / run.features / split}.pt" for split in absent)
                print(f"{'would skip' if args.dry_run else 'skipping'} {run.config}: missing {paths}", flush=True)
                skipped.append(run.config)
                if not args.dry_run:
                    continue
            code = execute(command(run, args.features, args.results, device), args.dry_run)
            if code:
                raise SystemExit(code)
            if not args.dry_run:
                report(args.results / run.output)
    if not args.dry_run and len(skipped) == sum(len(TARGETS[name]) for name in selected):
        raise SystemExit("nothing to run: prepare a feature cache with scripts/prepare_data.py")


if __name__ == "__main__":
    main()
