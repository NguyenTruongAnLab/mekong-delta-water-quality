#!/usr/bin/env python3
"""
Nature Water: Generate All Supplementary Tables
===============================================
Single entry point to regenerate every supplementary table (S1-S5).
The scripts use the shared _nw_common_tables.py for consistent data loading.

Usage:
    python scripts/99_publish/nw_tables/nw_run_all.py
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

# Add parent directory to path to ensure modules can be found if run directly
sys.path.insert(0, str(Path(__file__).resolve().parent))

def main() -> None:
    t0 = time.time()

    print("=" * 70)
    print("NATURE WATER: Complete Supplementary Table Pipeline")
    print("=" * 70)

    # ── Table S1 ──
    print("\n▸ Supplementary Table S1: Multivariate Screening")
    print("-" * 50)
    from nw_table_s1_multivariate_screening import main as run_s1
    run_s1()

    # ── Table S2 ──
    print("\n▸ Supplementary Table S2: SHAP Importance")
    print("-" * 50)
    from nw_table_s2_shap_importance import main as run_s2
    run_s2()

    # ── Table S3 ──
    print("\n▸ Supplementary Table S3: Model Performance")
    print("-" * 50)
    from nw_table_s3_model_performance import main as run_s3
    run_s3()

    # ── Table S4 ──
    print("\n▸ Supplementary Table S4: COD Robustness")
    print("-" * 50)
    from nw_table_s4_cod_robustness import main as run_s4
    run_s4()

    # ── Table S5 ──
    print("\n▸ Supplementary Table S5: COD Fixed Effects")
    print("-" * 50)
    from nw_table_s5_cod_fixed_effects import main as run_s5
    run_s5()

    elapsed = time.time() - t0
    print(f"\n✓ Complete table pipeline finished in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
