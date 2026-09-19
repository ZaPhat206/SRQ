from pathlib import Path

from scripts.make_budget_sensitivity_figure import aggregate, format_stat, read


ROOT = Path(__file__).resolve().parents[1]


def test_budget_aggregation_uses_raw_streams_and_common_formatter():
    rows = aggregate(read(ROOT / "results/paper/budget_sensitivity/units.json"))
    assert [row["beta"] for row in rows] == [0.0, 0.05, 0.1, 0.25, 0.5, 1.0]
    # dAIA = method - Exact, so the zero-budget point sits below Exact.
    assert format_stat(rows[0]["delta_aia_mean"], rows[0]["delta_aia_std"]) == "-0.287 +/- 0.047"
    assert format_stat(rows[1]["delta_aia_mean"], rows[1]["delta_aia_std"]) == "-0.004 +/- 0.016"
    assert format_stat(rows[2]["delta_aia_mean"], rows[2]["delta_aia_std"]) == "+0.001 +/- 0.006"
    assert [round(row["reduction_percent"], 1) for row in rows] == [67.0, 65.9, 64.8, 61.6, 56.3, 45.6]
    assert all(row["n"] == 3 for row in rows)
