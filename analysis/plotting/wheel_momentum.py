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
    plot_horizontal_limit,
)

from .style import (
    LINE_WIDTH,
    TITLE_FONT_SIZE,
    LABEL_FONT_SIZE,
    LEGEND_FONT_SIZE,
    COLORS,
    SATURATION_COLOR,
)


def plot_wheel_momentum(
    telemetry: dict,
    output_directory: str = "analysis/figures",
    saturation_limit: float | None = None,
):
    """
    Plot reaction wheel momentum history.

    Parameters
    ----------
    telemetry : dict
        Simulation telemetry.

    saturation_limit : float, optional
        Maximum allowable wheel momentum [N·m·s].
    """

    required = [
        "time",
        "wheel_momentum",
    ]

    for key in required:

        if key not in telemetry:

            raise KeyError(
                f"Missing telemetry channel '{key}'"
            )

    time = np.asarray(
        telemetry["time"]
    )

    wheel_momentum = np.asarray(
        telemetry["wheel_momentum"]
    )

    fig, ax = create_figure()

    number_of_wheels = wheel_momentum.shape[1]

    for i in range(number_of_wheels):

        peak = np.max(
            np.abs(
                wheel_momentum[:, i]
            )
        )

        ax.plot(
            time,
            wheel_momentum[:, i],
            linewidth=LINE_WIDTH,
            color=COLORS[i % len(COLORS)],
            label=f"RW{i+1}  (Peak={peak:.3f})",
        )

    # --------------------------------------------------
    # Saturation Limits
    # --------------------------------------------------

    if saturation_limit is not None:

        plot_horizontal_limit(
            ax,
            saturation_limit,
            "Upper Limit",
            color=SATURATION_COLOR,
        )

        plot_horizontal_limit(
            ax,
            -saturation_limit,
            "Lower Limit",
            color=SATURATION_COLOR,
        )

    ax.set_title(
        "Reaction Wheel Momentum",
        fontsize=TITLE_FONT_SIZE,
        fontweight="bold",
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
        ncol=2,
    )

    save_figure(
        fig,
        "wheel_momentum.png",
        output_directory,
    )