"""Run a Table 3 configuration.

Use the FLY-CL configuration with ``experiments.fly_test`` and either RanPAC
configuration with ``experiments.ranpac_test``.
"""
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser()
parser.add_argument("--family", choices=("fly", "ranpac"), required=True)
args, remainder = parser.parse_known_args()
sys.argv = [sys.argv[0], *remainder]
runpy.run_module("experiments.fly_test" if args.family == "fly" else "experiments.ranpac_test", run_name="__main__")
