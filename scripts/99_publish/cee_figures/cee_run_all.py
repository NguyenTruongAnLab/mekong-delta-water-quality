#!/usr/bin/env python3
"""
Master runner for all 7 CEE Main Figures + 12 CEE Supplementary Figures.
Outputs 300 DPI PNG + Type-42 vector PDF to `manuscript/figures/cee_composites/`.
"""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

SCRIPTS = [
    # 7 Main Figures
    "cee_fig1_regime_shift.py",
    "cee_fig2_reactor.py",
    "cee_fig3_estuarine_mixing.py",
    "cee_fig4_decoupling.py",
    "cee_fig5_tidal_decomposition.py",
    "cee_fig6_attribution.py",
    "cee_fig7_schematic.py",
    # 12 Supplementary Figures
    "cee_sfig1_extended_trends.py",
    "cee_sfig2_aou_thermodynamics.py",
    "cee_sfig3_reactor_all_parameters.py",
    "cee_sfig4_monthly_hysteresis.py",
    "cee_sfig5_wrtds_flow_normalization.py",
    "cee_sfig6_lulc_trajectory_groups.py",
    "cee_sfig7_landscape_trajectories.py",
    "cee_sfig8_shap_cod_hydroclimate.py",
    "cee_sfig9_buffer_scale.py",
    "cee_sfig10_network_cross_validation.py",
    "cee_sfig11_rainfall_crosscheck.py",
    "cee_sfig12_fertilizer_gradient.py",
]


def main() -> int:
    only_sfigs = "--sfigs-only" in sys.argv
    targets = [s for s in SCRIPTS if (not only_sfigs or s.startswith("cee_sfig"))]
    print(f"Running {len(targets)} CEE figure scripts...")
    for s in targets:
        p = HERE / s
        res = subprocess.run([sys.executable, str(p)], check=False)
        if res.returncode != 0:
            print(f"FAILED: {s} (exit {res.returncode})", file=sys.stderr)
            return res.returncode
    print("All CEE figure scripts completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
