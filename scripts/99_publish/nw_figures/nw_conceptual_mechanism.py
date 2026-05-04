#!/usr/bin/env python3
"""Conceptual Figure: Conveyor Belt → Mixing Bowl transition.

Panel A (Pre-2004): "The Conveyor Belt" - Fast flow, pollution flushed out
Panel B (Post-2004): "The Mixing Bowl" - Slow flow, pollution retained & processed
Panel C: Attribution Illusion schematic

Outputs:
- manuscript/figures/nw_composites/Figure_Conceptual_Mechanism.{png,pdf}
"""
from __future__ import annotations
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from _nw_common import save_fig, FIG_OUT

plt.rcParams["font.family"] = "Arial"


def _draw_delta_cross_section(ax, phase: str):
    """Draw a stylized delta cross-section showing flow dynamics."""
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-2.5, 5.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # Ground / riverbed
    ground_x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    ground_y = np.array([0.5, 0.3, 0.1, 0, -0.1, -0.2, -0.3, -0.5, -0.7, -0.9, -1.2])
    ax.fill_between(ground_x, ground_y, -2.5, color="#d2b48c", alpha=0.3, zorder=1)
    ax.plot(ground_x, ground_y, color="#8B7355", lw=1.5, zorder=2)

    # Water surface
    if phase == "pre":
        water_level = 2.5
        water_color = "#4393c3"
        water_alpha = 0.3
        flow_speed_label = r"High Q ($\sim$18,000 $\mathrm{m^3\,s^{-1}}$)"
    else:
        water_level = 1.8
        water_color = "#d6604d"
        water_alpha = 0.25
        flow_speed_label = r"Low Q ($\sim$12,000 $\mathrm{m^3\,s^{-1}}$)"

    ax.fill_between(ground_x, ground_y, water_level,
                    color=water_color, alpha=water_alpha, zorder=1)

    # Dam symbol upstream
    if phase == "post":
        ax.add_patch(FancyBboxPatch((0.3, ground_y[0]), 0.4, water_level - ground_y[0] + 0.5,
                                     boxstyle="round,pad=0.05",
                                     facecolor="#555555", edgecolor="black",
                                     lw=1.5, zorder=5))
        ax.text(0.5, water_level + 1.0, "Upstream\ndams", fontsize=6,
                ha="center", va="bottom", fontweight="bold", color="#555")

    # Flow arrows
    if phase == "pre":
        for y_pos in [1.0, 1.5, 2.0]:
            ax.annotate("", xy=(9.5, y_pos), xytext=(1.0, y_pos),
                        arrowprops=dict(arrowstyle="-|>", color="#2166ac",
                                        lw=2.5, mutation_scale=20),
                        zorder=4)
        ax.text(5, 2.7, flow_speed_label, fontsize=8, ha="center",
                color="#2166ac", fontweight="bold")
    else:
        # Slower, meandering flow with tidal intrusion arrows
        for y_pos in [0.5, 1.0]:
            ax.annotate("", xy=(7.0, y_pos), xytext=(1.5, y_pos),
                        arrowprops=dict(arrowstyle="-|>", color="#b2182b",
                                        lw=1.5, mutation_scale=15, alpha=0.7),
                        zorder=4)
        # Tidal intrusion arrows (incoming from sea)
        for y_pos in [0.3, 0.8]:
            ax.annotate("", xy=(6.0, y_pos + 0.3), xytext=(9.5, y_pos + 0.3),
                        arrowprops=dict(arrowstyle="-|>", color="#762a83",
                                        lw=1.5, mutation_scale=12, alpha=0.6,
                                        linestyle="dashed"),
                        zorder=4)
        ax.text(8.5, 2.2, "Tidal\nintrusion", fontsize=6, ha="center",
                color="#762a83", fontstyle="italic")
        ax.text(5, 2.2, flow_speed_label, fontsize=8, ha="center",
                color="#b2182b", fontweight="bold")

    # Pollution sources (top)
    src_positions = [2.5, 5.0, 7.5]
    src_labels = ["Agriculture", "Urban", "Aquaculture"]
    src_colors = ["#4daf4a", "#e41a1c", "#377eb8"]
    for x, lab, clr in zip(src_positions, src_labels, src_colors):
        ax.annotate("", xy=(x, water_level + 0.1), xytext=(x, water_level + 1.2),
                    arrowprops=dict(arrowstyle="-|>", color=clr, lw=1.5,
                                    mutation_scale=12),
                    zorder=4)
        ax.text(x, water_level + 1.4, lab, fontsize=6, ha="center",
                color=clr, fontweight="bold")

    # Pollution fate visualization
    if phase == "pre":
        # Pollution arrows exiting at the sea end
        for i, (x, clr) in enumerate(zip(src_positions, src_colors)):
            y_start = (water_level + ground_y[int(x)]) / 2
            ax.plot([x, 10], [y_start, y_start - 0.3],
                    color=clr, lw=1, alpha=0.5, ls="--", zorder=3)
        # "Flushed out" label
        ax.text(10.2, 0.5, r"Pollution" + "\n" + r"flushed $\rightarrow$" + "\nto sea",
                fontsize=7, ha="left", va="center",
                color="#2166ac", fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="#2166ac",
                         alpha=0.8, boxstyle="round,pad=0.3"))
        # Residence time
        ax.text(5, -1.5, r"Short residence time ($\tau$)",
                fontsize=8, ha="center", color="#2166ac", fontweight="bold")
    else:
        # Pollution spiraling/retained in the delta
        theta = np.linspace(0, 4 * np.pi, 100)
        for x_c, clr in zip([3.5, 6.0], ["#d6604d", "#8B0000"]):
            r = 0.3 + 0.15 * theta / (4 * np.pi)
            spiral_x = x_c + r * np.cos(theta)
            spiral_y = 0.5 + r * np.sin(theta) * 0.6
            ax.plot(spiral_x, spiral_y, color=clr, lw=1, alpha=0.6, zorder=3)
        # "Retained" label
        ax.text(10.2, 0.0,
                "Pollution\nretained\n& processed",
                fontsize=7, ha="left", va="center",
                color="#b2182b", fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="#b2182b",
                         alpha=0.8, boxstyle="round,pad=0.3"))
        # Residence time
        ax.text(5, -1.5, r"Long residence time ($\tau$)",
                fontsize=8, ha="center", color="#b2182b", fontweight="bold")

    # Sea label on right
    ax.text(10.0, water_level + 0.3, "Sea", fontsize=8,
            ha="center", fontweight="bold", color="#4393c3")

    # Labels
    ax.text(0, water_level + 0.3, "Upstream", fontsize=8,
            ha="center", fontweight="bold", color="#555")


def _draw_cq_schematic(ax):
    """Draw C-Q relationship change schematic."""
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 1.1)

    # Phase 1: strong C-Q coupling (dilution)
    q1 = np.linspace(0.1, 1.0, 50)
    c1_dilution = 0.9 * q1 ** (-0.3)  # chemodynamic
    ax.plot(q1, c1_dilution / c1_dilution.max(), color="#2166ac", lw=2.5,
            label=r"Phase 1: Chemodynamic ($b < 0$)")

    # Phase 2: chemostatic (flat)
    c2_chemo = np.ones_like(q1) * 0.65
    ax.plot(q1, c2_chemo, color="#b2182b", lw=2.5, ls="--",
            label=r"Phase 2: Chemostatic ($b \approx 0$)")

    ax.set_xlabel(r"Discharge Q ($\mathrm{m^3\,s^{-1}}$)", fontsize=9)
    ax.set_ylabel(r"Concentration C (mg $\mathrm{L^{-1}}$)", fontsize=9)
    ax.legend(fontsize=7, loc="upper right", frameon=True, framealpha=0.9)
    ax.set_title("C-Q Decoupling", fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    fig = plt.figure(figsize=(180 / 25.4, 200 / 25.4), dpi=300)

    # Top row: conceptual diagrams
    ax_pre = fig.add_axes([0.02, 0.55, 0.47, 0.40])
    ax_post = fig.add_axes([0.52, 0.55, 0.47, 0.40])

    _draw_delta_cross_section(ax_pre, "pre")
    ax_pre.set_title(r"$\mathbf{a}$  Pre-2004: The Conveyor Belt",
                     fontsize=10, fontweight="bold", loc="left", pad=8)

    _draw_delta_cross_section(ax_post, "post")
    ax_post.set_title(r"$\mathbf{b}$  Post-2004: The Mixing Bowl",
                      fontsize=10, fontweight="bold", loc="left", pad=8)

    # Bottom row: C-Q schematic + key metrics
    ax_cq = fig.add_axes([0.08, 0.08, 0.38, 0.38])
    _draw_cq_schematic(ax_cq)
    ax_cq.set_title(r"$\mathbf{c}$  C-Q Decoupling Evidence",
                     fontsize=10, fontweight="bold", loc="left", pad=8)

    # Key metrics summary panel
    ax_metrics = fig.add_axes([0.55, 0.08, 0.42, 0.38])
    ax_metrics.axis("off")
    ax_metrics.set_title(r"$\mathbf{d}$  Key Evidence Summary",
                         fontsize=10, fontweight="bold", loc="left", pad=8)

    metrics_text = (
        r"$\bullet$ COD Chemostatic Index: 0.84 $\rightarrow$ 0.49" + "\n"
        r"   (transport-limited $\rightarrow$ source-limited)" + "\n\n"
        r"$\bullet$ TN non-conservative fraction:" + "\n"
        r"   +16.3% (net in-delta generation, $p=0.028$)" + "\n\n"
        r"$\bullet$ COD self-purification capacity:" + "\n"
        r"   $-$10.9% $\rightarrow$ $-$1.4% ($p=0.010$)" + "\n\n"
        r"$\bullet$ TR/Q flushing index: +3.2%/yr" + "\n"
        r"   (tidal dominance increasing)" + "\n\n"
        r"$\bullet$ LULC attribution (panel FE):" + "\n"
        r"   $\Delta$R$^2$ = 0.005 (temporal confounding)" + "\n"
        r"   $\Rightarrow$ 98\% of apparent LULC effect" + "\n"
        r"   is co-trending, not causal"
    )
    ax_metrics.text(0.05, 0.95, metrics_text, transform=ax_metrics.transAxes,
                    fontsize=8, va="top", ha="left", family="Arial",
                    linespacing=1.4,
                    bbox=dict(facecolor="#f7f7f7", edgecolor="#cccccc",
                             alpha=0.9, boxstyle="round,pad=0.5"))

    save_fig(fig, "Figure_Conceptual_Mechanism")
    plt.close(fig)


if __name__ == "__main__":
    main()
