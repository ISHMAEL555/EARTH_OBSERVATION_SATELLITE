"""
analysis/plotting/attitude.py

Quaternion attitude tracking plots.
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
    REFERENCE_LINE_WIDTH,
    REFERENCE_LINE_STYLE,
    REFERENCE_ALPHA,
    TITLE_FONT_SIZE,
    LABEL_FONT_SIZE,
    LEGEND_FONT_SIZE,
    COLORS,
)


def plot_attitude(
    telemetry: dict,
    output_directory: str = "analysis/figures",
):
    """
    Plot actual and reference quaternion histories.
    """

    required = [
        "time",
        "quaternion",
        "reference_quaternion",
    ]

    for key in required:
        if key not in telemetry:
            raise KeyError(f"Missing telemetry channel '{key}'")

    time = np.asarray(telemetry["time"])

    quaternion = np.asarray(
        telemetry["quaternion"]
    )

    reference = np.asarray(
        telemetry["reference_quaternion"]
    )

    labels = [
        r"$q_0$",
        r"$q_1$",
        r"$q_2$",
        r"$q_3$",
    ]

    fig, ax = create_figure()

    for i in range(4):

        # Actual quaternion

        ax.plot(
            time,
            quaternion[:, i],
            color=COLORS[i],
            linewidth=LINE_WIDTH,
            label=f"{labels[i]}",
        )

        # Reference quaternion

        ax.plot(
            time,
            reference[:, i],
            linestyle=REFERENCE_LINE_STYLE,
            linewidth=REFERENCE_LINE_WIDTH,
            alpha=REFERENCE_ALPHA,
            color=COLORS[i],
        )

    ax.set_title(
        "Quaternion Attitude Tracking",
        fontsize=TITLE_FONT_SIZE,
        fontweight="bold",
    )

    ax.set_xlabel(
        "Time [s]",
        fontsize=LABEL_FONT_SIZE,
    )

    ax.set_ylabel(
        "Quaternion",
        fontsize=LABEL_FONT_SIZE,
    )

    ax.set_ylim(
        -1.05,
        1.05,
    )

    configure_axes(ax)

    ax.legend(
        ncol=4,
        fontsize=LEGEND_FONT_SIZE,
        loc="upper center",
    )

    save_figure(
        fig,
        "attitude.png",
        output_directory,
    )