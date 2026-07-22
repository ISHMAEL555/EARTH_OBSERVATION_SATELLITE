"""
analysis/plotting/quaternion_norm.py

Quaternion normalization verification.
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
)


def plot_quaternion_norm(
    telemetry: dict,
    output_directory: str = "analysis/figures",
):
    """
    Plot quaternion norm.

    A valid attitude quaternion should remain equal to one
    throughout the simulation.
    """

    required = [
        "time",
        "quaternion",
    ]

    for key in required:

        if key not in telemetry:

            raise KeyError(
                f"Missing telemetry channel '{key}'"
            )

    time = np.asarray(
        telemetry["time"]
    )

    quaternion = np.asarray(
        telemetry["quaternion"]
    )

    quaternion_norm = np.linalg.norm(
        quaternion,
        axis=1,
    )

    fig, ax = create_figure()

    ax.plot(
        time,
        quaternion_norm,
        linewidth=LINE_WIDTH,
        label=r"$||q||$",
    )

    ax.axhline(
        1.0,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="Ideal",
    )

    ax.set_title(
        "Quaternion Normalization",
        fontsize=TITLE_FONT_SIZE,
        fontweight="bold",
    )

    ax.set_xlabel(
        "Time [s]",
        fontsize=LABEL_FONT_SIZE,
    )

    ax.set_ylabel(
        r"$||q||$",
        fontsize=LABEL_FONT_SIZE,
    )

    # Zoom around unity
    margin = max(
        1e-6,
        np.max(np.abs(quaternion_norm - 1.0)) * 1.2,
    )

    ax.set_ylim(
        1.0 - margin,
        1.0 + margin,
    )

    configure_axes(ax)

    ax.legend()

    save_figure(
        fig,
        "quaternion_norm.png",
        output_directory,
    )