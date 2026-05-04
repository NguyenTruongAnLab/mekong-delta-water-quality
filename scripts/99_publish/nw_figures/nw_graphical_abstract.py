#!/usr/bin/env python3
"""Graphical Abstract for Nature Water submission.

Visual summary of the paper in a single landscape figure:
- Left: Study area + regime shift timeline
- Center: "Conveyor Belt → Mixing Bowl" conceptual transition
- Right: Attribution illusion (naïve vs causal)

Outputs:
- manuscript/figures/nw_composites/Graphical_Abstract.{png,pdf}
"""
from __future__ import annotations
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np
from _nw_common import save_fig, FIG_OUT

plt.rcParams["font.family"] = "Arial"

BLUE = "#2166ac"
RED = "#b2182b"
GREY = "#636363"


def _panel_regime_shift(ax):
    """Stylized regime shift timeline."""
    np.random.seed(42)
    years = np.arange(1985, 2023)

    # COD trend with break
    base_pre = 4.0 + 0.02 * (years[:20] - 1985)
    base_post = 5.2 + 0.06 * (years[20:] - 2004)
    cod = np.concatenate([base_pre, base_post]) + np.random.normal(0, 0.3, len(years))

    ax.plot(years, cod, "-o", color=RED, ms=2, lw=1.2, alpha=0.85)
    ax.axvline(2004.5, color=GREY, lw=1.5, ls="--", alpha=0.7)
    ax.fill_betweenx([2, 8], 1985, 2004.5, color=BLUE, alpha=0.06)
    ax.fill_betweenx([2, 8], 2004.5, 2022, color=RED, alpha=0.06)

    ax.text(1994, 7.2, "Phase 1", fontsize=8, ha="center", color=BLUE,
            fontweight="bold")
    ax.text(2013, 7.2, "Phase 2", fontsize=8, ha="center", color=RED,
            fontweight="bold")
    ax.text(2004.5, 7.8, "Regime\nshift", fontsize=7, ha="center",
            color=GREY, fontweight="bold")

    ax.set_ylabel(r"COD (mg $\mathrm{L^{-1}}$)", fontsize=8)
    ax.set_xlabel("Year", fontsize=8)
    ax.set_ylim(2, 8.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7)


def _panel_conceptual(ax):
    """Conveyor Belt → Mixing Bowl schematic."""
    ax.set_xlim(-0.5, 11)
    ax.set_ylim(-1, 7)
    ax.axis("off")

    # ── Pre-2004 (top) ──
    # River channel
    ax.fill_between([0, 10], [4.2, 4.2], [5.8, 5.8],
                    color=BLUE, alpha=0.15)
    ax.plot([0, 10], [5.8, 5.8], color=BLUE, lw=1)
    ax.plot([0, 10], [4.2, 4.2], color=BLUE, lw=1)

    # Flow arrows (strong)
    for y in [4.7, 5.0, 5.3]:
        ax.annotate("", xy=(9.8, y), xytext=(0.5, y),
                    arrowprops=dict(arrowstyle="-|>", color=BLUE,
                                    lw=2, mutation_scale=15))

    # Pollution dots flowing through
    for x in [2, 4, 6, 8]:
        ax.plot(x, 5.0, "o", color="#d6604d", ms=5, alpha=0.5, zorder=5)

    ax.text(5, 6.3, r"Pre-2004: Conveyor Belt", fontsize=9,
            ha="center", fontweight="bold", color=BLUE)
    ax.text(10.3, 5.0, r"$\rightarrow$ Sea", fontsize=7,
            ha="left", va="center", color=BLUE)
    ax.text(5, 3.7, r"High Q $\rightarrow$ short $\tau$ $\rightarrow$ pollution exported",
            fontsize=7, ha="center", color=BLUE, fontstyle="italic")

    # ── Post-2004 (bottom) ──
    ax.fill_between([0, 10], [0.2, 0.2], [1.8, 1.8],
                    color=RED, alpha=0.12)
    ax.plot([0, 10], [1.8, 1.8], color=RED, lw=1)
    ax.plot([0, 10], [0.2, 0.2], color=RED, lw=1)

    # Weak flow
    ax.annotate("", xy=(7.0, 1.0), xytext=(0.5, 1.0),
                arrowprops=dict(arrowstyle="-|>", color=RED,
                                lw=1.5, mutation_scale=12, alpha=0.6))
    # Tidal intrusion
    ax.annotate("", xy=(5.0, 1.0), xytext=(9.5, 1.0),
                arrowprops=dict(arrowstyle="-|>", color="#762a83",
                                lw=1.2, mutation_scale=10, alpha=0.5,
                                linestyle="dashed"))

    # Spiraling pollution
    theta = np.linspace(0, 3 * np.pi, 60)
    for cx in [3, 6]:
        r = 0.15 + 0.08 * theta / (3 * np.pi)
        ax.plot(cx + r * np.cos(theta), 1.0 + r * np.sin(theta) * 0.5,
                color="#d6604d", lw=1, alpha=0.7)

    ax.text(5, 2.3, r"Post-2004: Mixing Bowl", fontsize=9,
            ha="center", fontweight="bold", color=RED)
    ax.text(5, -0.3, r"Low Q $\rightarrow$ long $\tau$ $\rightarrow$ pollution retained",
            fontsize=7, ha="center", color=RED, fontstyle="italic")

    # Transition arrow
    ax.annotate("", xy=(5, 2.0), xytext=(5, 3.5),
                arrowprops=dict(arrowstyle="-|>", color=GREY,
                                lw=2.5, mutation_scale=20))
    ax.text(5.2, 2.8, "2004", fontsize=8, fontweight="bold", color=GREY)


def _panel_illusion(ax):
    """Attribution illusion: naïve vs panel FE."""
    labels = [r"Na$\ddot{\mathrm{i}}$ve OLS", "Station+Year FE", "FE + LULC"]
    r2 = [0.301, 0.649, 0.654]
    colors_bar = ["#fc8d62", "#66c2a5", "#8da0cb"]

    x = np.arange(len(labels))
    bars = ax.bar(x, r2, color=colors_bar, edgecolor="white", lw=0.5, width=0.6)

    # Annotation
    for i, (v, b) in enumerate(zip(r2, bars)):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=7, fontweight="bold")

    # Delta R2 bracket
    ax.annotate("", xy=(2, 0.654), xytext=(1, 0.649),
                arrowprops=dict(arrowstyle="<->", color=RED, lw=1.2))
    ax.text(1.5, 0.67, r"$\Delta$R$^2$ = 0.005", fontsize=7,
            ha="center", color=RED, fontweight="bold")
    ax.text(1.5, 0.73, "LULC effect\nis illusory!", fontsize=6,
            ha="center", color=RED, fontstyle="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel(r"R$^2$", fontsize=8)
    ax.set_ylim(0, 0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7)


def main():
    fig = plt.figure(figsize=(180 / 25.4, 90 / 25.4), dpi=300)

    # 3 panels in a row
    ax1 = fig.add_axes([0.06, 0.15, 0.26, 0.75])
    ax2 = fig.add_axes([0.36, 0.05, 0.33, 0.90])
    ax3 = fig.add_axes([0.74, 0.15, 0.24, 0.75])

    _panel_regime_shift(ax1)
    ax1.set_title(r"$\mathbf{a}$  Regime Shift (2004)",
                  fontsize=9, fontweight="bold", loc="left")

    _panel_conceptual(ax2)
    ax2.set_title(r"$\mathbf{b}$  Mechanism",
                  fontsize=9, fontweight="bold", loc="left")

    _panel_illusion(ax3)
    ax3.set_title(r"$\mathbf{c}$  Attribution Illusion",
                  fontsize=9, fontweight="bold", loc="left")

    save_fig(fig, "Graphical_Abstract")
    plt.close(fig)


if __name__ == "__main__":
    main()
