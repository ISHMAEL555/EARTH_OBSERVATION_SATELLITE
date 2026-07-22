"""
analysis/plotting/control_effort.py

Control torque magnitude.
"""

from __future__ import annotations

import numpy as np

from .common import (
    create_figure,
    configure_axes,
    save_figure,
    plot_vector_norm,
)

from .style import (
    LINE_WIDTH,
    TITLE_FONT_SIZE,
    LABEL_FONT_SIZE,
)


def plot_control_effort(
    telemetry: dict,
    output_directory: str = "analysis/figures",
):
    """
    Plot control torque magnitude.

    Computes

        ||T|| = sqrt(Tx² + Ty² + Tz²)
    """

    required = [
        "time",
        "control_torque",
    ]

    for key in required:

        if key not in telemetry:

            raise KeyError(
                f"Missing telemetry channel '{key}'"
            )

    time = np.asarray(
        telemetry["time"]
    )

    control_torque = np.asarray(
        telemetry["control_torque"]
    )

    torque_norm = plot_vector_norm(
        control_torque
    )

    fig, ax = create_figure()

    ax.plot(
        time,
        torque_norm,
        linewidth=LINE_WIDTH,
        label=r"$||T||$",
    )

    ax.set_title(
        "Control Torque Magnitude",
        fontsize=TITLE_FONT_SIZE,
        fontweight="bold",
    )

    ax.set_xlabel(
        "Time [s]",
        fontsize=LABEL_FONT_SIZE,
    )

    ax.set_ylabel(
        r"$||T||$ [N·m]",
        fontsize=LABEL_FONT_SIZE,
    )

    configure_axes(ax)

    ax.legend()

    save_figure(
        fig,
        "control_effort.png",
        output_directory,
    )