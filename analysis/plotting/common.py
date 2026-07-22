"""
analysis/plotting/common.py

Common plotting utilities.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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


def create_figure():
    """Create a standard figure."""

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    return fig, ax


def configure_axes(ax):

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


def save_figure(
    fig,
    filename,
    output_directory,
):

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


def validate_telemetry(
    telemetry: dict,
    keys: list[str],
):

    for key in keys:

        if key not in telemetry:

            raise KeyError(
                f"Telemetry channel '{key}' not found."
            )


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