"""
analysis/plotting/attitude.py

Quaternion attitude tracking plots.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

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


def plot_attitude(
    telemetry: dict,
    output_directory: str = "analysis/figures",
) -> None:
    """
    Plot quaternion tracking.

    Parameters
    ----------
    telemetry : dict
        Simulation telemetry dictionary.

    output_directory : str
        Directory where figures are saved.
    """

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

        ax.plot(
            time,
            quaternion[:, i],
            linewidth=LINE_WIDTH,
            color=COLORS[i],
            label=f"{labels[i]} Actual",
        )

        ax.plot(
            time,
            reference[:, i],
            "--",
            linewidth=1.2,
            color=COLORS[i],
            alpha=0.65,
            label=f"{labels[i]} Reference",
        )

    ax.set_title(
        "Quaternion Attitude Tracking",
        fontsize=TITLE_FONT_SIZE,
    )

    ax.set_xlabel(
        "Time [s]",
        fontsize=LABEL_FONT_SIZE,
    )

    ax.set_ylabel(
        "Quaternion",
        fontsize=LABEL_FONT_SIZE,
    )

    configure_axes(ax)

    ax.legend(
        fontsize=LEGEND_FONT_SIZE,
        ncol=2,
    )

    save_figure(
        fig,
        "attitude.png",
        output_directory,
    )