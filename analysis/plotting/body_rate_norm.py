"""
analysis/plotting/body_rate_norm.py

Body angular rate magnitude.
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


def plot_body_rate_norm(
    telemetry: dict,
    output_directory: str = "analysis/figures",
):
    """
    Plot body angular velocity magnitude.

    Computes

        ||ω|| = sqrt(wx² + wy² + wz²)
    """

    required = [
        "time",
        "body_rates",
    ]

    for key in required:

        if key not in telemetry:

            raise KeyError(
                f"Missing telemetry channel '{key}'"
            )

    time = np.asarray(
        telemetry["time"]
    )

    body_rates = np.asarray(
        telemetry["body_rates"]
    )

    body_rate_norm = plot_vector_norm(
        body_rates
    )

    fig, ax = create_figure()

    ax.plot(
        time,
        body_rate_norm,
        linewidth=LINE_WIDTH,
        label=r"$||\omega||$",
    )

    ax.set_title(
        "Body Angular Velocity Magnitude",
        fontsize=TITLE_FONT_SIZE,
        fontweight="bold",
    )

    ax.set_xlabel(
        "Time [s]",
        fontsize=LABEL_FONT_SIZE,
    )

    ax.set_ylabel(
        r"$||\omega||$ [rad/s]",
        fontsize=LABEL_FONT_SIZE,
    )

    configure_axes(ax)

    ax.legend()

    save_figure(
        fig,
        "body_rate_norm.png",
        output_directory,
    )