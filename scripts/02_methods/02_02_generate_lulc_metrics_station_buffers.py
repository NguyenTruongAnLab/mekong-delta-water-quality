#!/usr/bin/env python3
"""02_02_generate_lulc_metrics_station_buffers.py

Generate station-buffer LULC composition/configuration metrics.

Primary output cache used by attribution scripts:
- analysis/cache/lulc_metrics_corrected_16km.csv (and other radii if configured)

Note: This can be I/O heavy because it reads raster products.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    """Return repository root directory."""
    return Path(__file__).resolve().parents[2]


def run_module(module: str) -> None:
    """Run `python -m <module>` from the repository root."""
    subprocess.run([sys.executable, "-m", module], cwd=str(_repo_root()), check=True)


def main() -> None:
    run_module("analysis.generators.generate_lulc_metrics_jaxa_aligned")


if __name__ == "__main__":
    main()
