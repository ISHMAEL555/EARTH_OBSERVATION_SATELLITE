"""
analysis/plotting/wheel_momentum.py

Reaction wheel momentum plotting.
"""

from __future__ import annotations

import numpy as np

from .common import (
    create_figure,
    configure_axes,
    save_figure,
)

from .style import (
    LINE_WIDTH,
    TITLE_FONT_SIZE,
    LABEL_FONT_SIZE,
    LEGEND_FONT_SIZE,
    COLORS,
)


def plot_wheel_momentum(
    telemetry: dict,
    output_directory="analysis/figures",
):
    """
    Plot reaction wheel momentum.
    """

    time = np.asarray(
        telemetry["time"]
    )

    wheel_momentum = np.asarray(
        telemetry["wheel_momentum"]
    )

    fig, ax = create_figure()

    number_of_wheels = wheel_momentum.shape[1]

    for i in range(number_of_wheels):

        ax.plot(
            time,
            wheel_momentum[:, i],
            linewidth=LINE_WIDTH,
            color=COLORS[i % len(COLORS)],
            label=f"Wheel {i+1}",
        )

    ax.set_title(
        "Reaction Wheel Momentum",
        fontsize=TITLE_FONT_SIZE,
    )

    ax.set_xlabel(
        "Time [s]",
        fontsize=LABEL_FONT_SIZE,
    )

    ax.set_ylabel(
        "Momentum [N·m·s]",
        fontsize=LABEL_FONT_SIZE,
    )

    configure_axes(ax)

    ax.legend(
        fontsize=LEGEND_FONT_SIZE,
    )

    save_figure(
        fig,
        "wheel_momentum.png",
        output_directory,
    )