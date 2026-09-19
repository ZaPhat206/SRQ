"""Run the controlled Table 2 validation configuration."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
runpy.run_module("experiments.fly_validation", run_name="__main__")
