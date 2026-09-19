"""Aggregate the train-only budget diagnostic and draw the manuscript figure.

The input contains one raw validation result per stream and budget.  Means,
sample standard deviations, and state reductions are computed here so the
figure cannot depend on rounded manuscript values.

Sign convention: DeltaAIA = AIA_method - AIA_Exact, matching the manuscript.
A negative value is below Exact and a positive value is above it.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

MB = 1_000_000.0
DEFAULT_REFERENCE_MB = 877.5
OPERATING_BETA = 0.25

# Figure style, matching the manuscript.
WIDTH_IN, HEIGHT_IN = 3.2, 1.85           # printed size, about 81 x 47 mm
TEXT = "#262626"
AXIS = "#595959"
MAIN = "#3D5A73"                           # muted slate blue
MAIN_LIGHT = "#8FA3B5"                     # error bars
ZERO = "#A8B0B8"
ACCENT = "#B5562B"                         # restrained warm accent


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate(payload: dict, *, reference_mb: float | None = None) -> list[dict]:
    """Return one aggregate row for each budget in the public raw artifact."""
    units = payload["units"]
    reference_mb = float(payload.get("exact_upper_triangle_reference_mb", reference_mb or DEFAULT_REFERENCE_MB))
    budgets = sorted({float(unit["beta"]) for unit in units})
    rows = []
    for beta in budgets:
        selected = [unit for unit in units if float(unit["beta"]) == beta]
        deltas = [unit["validation_aia_percent"] - unit["exact_validation_aia_percent"] for unit in selected]
        states = [unit["final_total_persistent_bytes"] / MB for unit in selected]
        rows.append({
            "beta": beta,
            "delta_aia_mean": statistics.fmean(deltas),
            "delta_aia_std": statistics.stdev(deltas),
            "state_mb_mean": statistics.fmean(states),
            "state_mb_std": statistics.stdev(states),
            "reduction_percent": 100.0 * (1.0 - statistics.fmean(states) / reference_mb),
            "n": len(selected),
        })
    return rows


def format_stat(mean: float, std: float) -> str:
    return f"{mean:+.3f} +/- {std:.3f}"


def tick_label(beta: float) -> str:
    return "0" if beta == 0.0 else f"{beta:.2f}"


def render_figure(rows: list[dict], output: Path) -> None:
    """Draw the manuscript figure: DeltaAIA against evenly spaced budgets."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.6,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": TEXT,
        "xtick.color": AXIS,
        "ytick.color": AXIS,
        "xtick.labelcolor": TEXT,
        "ytick.labelcolor": TEXT,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.8,
        "ytick.major.size": 2.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.labelpad": 3.0,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })

    delta = [row["delta_aia_mean"] for row in rows]
    std = [row["delta_aia_std"] for row in rows]
    ticks = [tick_label(row["beta"]) for row in rows]
    x = list(range(len(rows)))
    op = next(i for i, row in enumerate(rows) if row["beta"] == OPERATING_BETA)

    fig, ax = plt.subplots(figsize=(WIDTH_IN, HEIGHT_IN))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("white")

    ax.axhline(0.0, color=ZERO, lw=0.7, ls=(0, (4, 2.5)), zorder=1)
    ax.errorbar(x, delta, yerr=std, fmt="none", ecolor=MAIN_LIGHT,
                elinewidth=0.8, capsize=2.2, capthick=0.8, zorder=2)
    ax.plot(x, delta, color=MAIN, lw=1.2, solid_capstyle="round", zorder=3)
    rest = [i for i in x if i != op]
    ax.plot([x[i] for i in rest], [delta[i] for i in rest], ls="none", marker="o",
            ms=4.2, mfc=MAIN, mec="white", mew=0.8, zorder=4)
    ax.plot(x[op], delta[op], ls="none", marker="D", ms=5.0, mfc=ACCENT,
            mec="white", mew=0.8, zorder=5)
    ax.set_xlim(-0.35, len(x) - 0.65)
    ax.set_ylim(-0.35, 0.05)
    ax.set_xticks(x, ticks)
    # the operating point is marked by its tick label, not by a floating note
    ax.get_xticklabels()[op].set_color(ACCENT)
    ax.set_yticks([-0.3, -0.2, -0.1, 0.0])
    ax.set_xlabel(r"Budget $\beta$")
    ax.set_ylabel(r"Validation $\Delta$AIA (pp)")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_bounds(-0.3, 0.0)
    ax.spines["bottom"].set_bounds(0, len(x) - 1)

    fig.tight_layout(pad=0.15)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results/paper/budget_sensitivity/units.json"))
    parser.add_argument("--output", type=Path, help="write the figure as a vector PDF")
    args = parser.parse_args()
    rows = aggregate(read(args.results))
    print("beta | DeltaAIA (pp) | State (MB) | Reduction | n")
    for row in rows:
        print(
            f"{row['beta']:.2f} | {format_stat(row['delta_aia_mean'], row['delta_aia_std'])} | "
            f"{row['state_mb_mean']:.1f} +/- {row['state_mb_std']:.1f} | "
            f"{row['reduction_percent']:.1f}% | {row['n']}"
        )
    if args.output:
        render_figure(rows, args.output)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
