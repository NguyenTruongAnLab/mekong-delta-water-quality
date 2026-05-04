#!/usr/bin/env python3
"""
Nature Water — Generate ALL Figures + Metrics + Audit
======================================================

Single entry point to regenerate every figure (main + extended data)
and then audit all manuscript claims against computed metrics.

Usage:
    python scripts/99_publish/nw_figures/nw_run_all.py

Output directory structure:
    manuscript/figures/nw_composites/
        Figure_1_Regime_Shift.{pdf,png}
        Figure_2_Biogeochemical_Reactor.{pdf,png}
        Figure_3_Hydrodynamic_Decoupling.{pdf,png}
        Figure_4_Landscape_Trajectories.{pdf,png}
        Figure_5_Driver_Attribution.{pdf,png}
        Figure_6_Attribution_Illusion.{pdf,png}
        ED_Figure_1_Extended_Trends.{pdf,png}
        ED_Figure_1_Network_CrossValidation.{pdf,png}  (auxiliary)
        ED_Figure_2_Buffer_Scale.{pdf,png}              (SI Fig 3)
        ED_Figure_2_Mixing_Diagnostics.{pdf,png}
        ED_Figure_3_HFI_TR_over_Q.{pdf,png}
        ED_Figure_4_Tidal_Decomposition.{pdf,png}
        ED_Figure_5_LULC_Trajectory_Groups.{pdf,png}
        ED_Figure_6_SHAP_COD_HydroClimate.{pdf,png}
        ED_Figure_7_Monthly_Hysteresis_Decoupling.{pdf,png}
        ED_Figure_8_Reactor_All_Parameters.{pdf,png}
        Figure_S7_Fertilizer_Gradient.{pdf,png}         (SI Fig 7)

    scripts/99_publish/nw_figures/metrics/
        Figure_1_Regime_Shift_metrics.csv
        Figure_2_Biogeochemical_Reactor_metrics.csv
        ...
        audit_results.csv
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> None:
    t0 = time.time()

    # Apply style once
    from _nw_common import apply_style, ZENODO_DOI
    apply_style()

    print("=" * 70)
    print("NATURE WATER — Complete Figure Pipeline")
    print("=" * 70)

    skipped: list[str] = []
    errors: list[str] = []

    def _run(label: str, fn, *args, **kwargs):
        """Run a figure function, catching restricted-data errors gracefully."""
        try:
            fn(*args, **kwargs)
        except SystemExit as e:
            if e.code == 1:
                print(f"  ⚠ SKIPPED (restricted data needed — see above)")
                skipped.append(label)
            else:
                raise
        except FileNotFoundError as e:
            print(f"  ⚠ SKIPPED: {e}")
            skipped.append(label)
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            errors.append(label)

    # ── Main Text Figures ──
    print("\n▸ MAIN TEXT FIGURES")
    print("-" * 40)

    from nw_fig1_regime_shift import figure_1
    print("  Figure 1: Spatiotemporal Regime Shift")
    _run("Fig 1", figure_1)

    from nw_fig2_reactor import figure_2
    print("  Figure 2: Biogeochemical Reactor")
    _run("Fig 2", figure_2)

    from nw_fig3_decoupling import figure_3
    print("  Figure 3: Hydrodynamic Decoupling")
    _run("Fig 3", figure_3)

    from nw_fig4_landscape import figure_4
    print("  Figure 4: Landscape Trajectories")
    _run("Fig 4", figure_4)

    from nw_fig5_attribution import figure_5
    print("  Figure 5: Driver Attribution")
    _run("Fig 5", figure_5)

    from nw_fig6_illusion import figure_6
    print("  Figure 6: Attribution Illusion")
    _run("Fig 6", figure_6)

    # this figure manually prepare by illutrator, so skip for now
    # print("  Figure 7: Conceptual Schematic")

    # ── Extended Data Figures ──
    print("\n▸ EXTENDED DATA FIGURES")
    print("-" * 40)

    from nw_ed_figs_1_3_validation import ed_figure_1, ed_figure_2, ed_figure_3
    print("  ED Fig 1: Cross-Network Validation")
    _run("ED Fig 1", ed_figure_1)
    print("  ED Fig 2: Buffer Scale Sensitivity")
    _run("ED Fig 2", ed_figure_2)
    print("  ED Fig 3: Extended Multi-Parameter Trends  [→ article ed1]")
    _run("ED Fig 3", ed_figure_3)

    from nw_ed_figs_2_4_hydrodynamics import ed_figure_4, ed_figure_5, ed_figure_6
    print("  ED Fig 4: Conservative Mixing Diagnostics  [→ article ed2]")
    _run("ED Fig 4", ed_figure_4)
    print("  ED Fig 5: Hydrodynamic Flushing Index  [→ article ed3]")
    _run("ED Fig 5", ed_figure_5)
    print("  ED Fig 6: Tidal Range Decomposition  [→ article ed4]")
    _run("ED Fig 6", ed_figure_6)

    from nw_ed_figs_5_6_lulc_attribution import ed_figure_7, ed_figure_8
    print("  ED Fig 7: LULC Trajectory Clustering  [→ article ed5]")
    _run("ED Fig 7", ed_figure_7)
    print("  ED Fig 8: SHAP COD HydroClimate Comparison  [→ article ed6]")
    _run("ED Fig 8", ed_figure_8)

    from nw_ed_fig7_hysteresis_decoupling import ed_figure_9
    print("  ED Fig 9: Monthly Hysteresis Decoupling  [→ article ed7]")
    _run("ED Fig 9", ed_figure_9)

    from nw_ed_fig8_reactor_all_parameters import ed_figure_10
    print("  ED Fig 10: Reactor Diagnostics (All Parameters)  [→ article ed8]")
    _run("ED Fig 10", ed_figure_10)

    # ── Supplementary Figures ──
    print("\n▸ SUPPLEMENTARY FIGURES")
    print("-" * 40)

    from nw_sfig7_fertilizer_gradient import main as sfig7
    print("  Supp Fig 7: Fertilizer Gradient")
    _run("Supp Fig 7", sfig7)

    from nw_sfig8_wrtds_flow_normalization import main as sfig8
    print("  Supp Fig 8: WRTDS Flow Normalization")
    _run("Supp Fig 8", sfig8)

    # ── Manuscript Audit ──
    print("\n▸ MANUSCRIPT METRICS AUDIT")
    print("-" * 40)

    from nw_audit_manuscript import audit_all
    print("  Audit: verifying manuscript claims against computed metrics")
    _run("Audit", audit_all)

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"PIPELINE SUMMARY")
    print(f"{'=' * 70}")
    if skipped:
        print(f"  ⚠ Skipped ({len(skipped)}): {', '.join(skipped)}")
        print(f"    → These need restricted data. Request access via Zenodo: {ZENODO_DOI}")
    if errors:
        print(f"  ✗ Errors ({len(errors)}): {', '.join(errors)}")
    if not skipped and not errors:
        print(f"  ✓ All figures generated successfully")
    print(f"  ⏱  Total time: {elapsed:.1f}s")
    print()
