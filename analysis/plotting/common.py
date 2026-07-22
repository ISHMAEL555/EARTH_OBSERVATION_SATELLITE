"""
analysis/plotting/common.py

Common plotting utilities.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from .style import (
    FIGURE_SIZE,
    DPI,
    GRID_ALPHA,
    GRID_STYLE,
    TICK_FONT_SIZE,
    TITLE_FONT_SIZE,
    LABEL_FONT_SIZE,
    LEGEND_FONT_SIZE,
    LINE_WIDTH,
    COLORS,
)


# ==========================================================
# Figure Utilities
# ==========================================================

def create_figure():
    """Create a standard figure."""

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    return fig, ax


def configure_axes(ax):
    """Apply common axis formatting."""

    ax.grid(
        True,
        linestyle=GRID_STYLE,
        alpha=GRID_ALPHA,
    )

    ax.set_axisbelow(True)

    ax.tick_params(
        axis="both",
        labelsize=TICK_FONT_SIZE,
    )


# ==========================================================
# 3D Utilities
# ==========================================================

def set_equal_axis_3d(ax):
    """
    Force equal scaling on all three axes.
    """

    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])

    centers = limits.mean(axis=1)

    radius = 0.5 * np.max(limits[:, 1] - limits[:, 0])

    ax.set_xlim3d(
        centers[0] - radius,
        centers[0] + radius,
    )

    ax.set_ylim3d(
        centers[1] - radius,
        centers[1] + radius,
    )

    ax.set_zlim3d(
        centers[2] - radius,
        centers[2] + radius,
    )


# ==========================================================
# Plot Helpers
# ==========================================================

def plot_horizontal_limit(
    ax,
    value,
    label,
    color="red",
):
    """Plot a horizontal requirement line."""

    ax.axhline(
        value,
        linestyle="--",
        linewidth=1.4,
        color=color,
        alpha=0.8,
        label=label,
    )


def plot_vector_norm(
    vector,
):
    """
    Compute vector magnitude.
    """

    vector = np.asarray(vector)

    return np.linalg.norm(
        vector,
        axis=1,
    )


# ==========================================================
# File Utilities
# ==========================================================

def save_figure(
    fig,
    filename,
    output_directory,
):
    """
    Save figure to disk.
    """

    output_directory = Path(output_directory)

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = output_directory / filename

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"✓ Saved {output_path}")


# ==========================================================
# Validation
# ==========================================================

def validate_telemetry(
    telemetry,
    keys,
):
    """
    Ensure telemetry contains required channels.
    """

    for key in keys:

        if key not in telemetry:

            raise KeyError(
                f"Telemetry channel '{key}' not found."
            )


# ==========================================================
# Generic Time-Series Plot
# ==========================================================

def plot_timeseries(
    *,
    time,
    data,
    labels,
    title,
    ylabel,
    filename,
    output_directory,
):
    """
    Generic multi-channel plotting utility.
    """

    fig, ax = create_figure()

    data = np.asarray(data)

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    for i in range(data.shape[1]):

        ax.plot(
            time,
            data[:, i],
            linewidth=LINE_WIDTH,
            color=COLORS[i % len(COLORS)],
            label=labels[i],
        )

    ax.set_title(
        title,
        fontsize=TITLE_FONT_SIZE,
    )

    ax.set_xlabel(
        "Time [s]",
        fontsize=LABEL_FONT_SIZE,
    )

    ax.set_ylabel(
        ylabel,
        fontsize=LABEL_FONT_SIZE,
    )

    configure_axes(ax)

    ax.legend(
        fontsize=LEGEND_FONT_SIZE,
    )

    save_figure(
        fig,
        filename,
        output_directory,
    )