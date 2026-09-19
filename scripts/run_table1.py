"""Run one Table 1 FLY-CL configuration."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
runpy.run_module("experiments.fly_test", run_name="__main__")
