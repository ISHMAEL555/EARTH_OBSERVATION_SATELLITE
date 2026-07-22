"""
analysis/plotting/attitude_error.py

Attitude pointing error.
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


def plot_attitude_error(
    telemetry: dict,
    output_directory: str = "analysis/figures",
):
    """
    Plot spacecraft attitude error in degrees.

    The error angle is computed from the quaternion error

        q_error = q_ref ⊗ q_actual^-1

    using

        theta = 2 arccos(q0)
    """

    required = [
        "time",
        "quaternion",
        "reference_quaternion",
    ]

    for key in required:

        if key not in telemetry:

            raise KeyError(
                f"Missing telemetry channel '{key}'"
            )

    time = np.asarray(
        telemetry["time"]
    )

    q = np.asarray(
        telemetry["quaternion"]
    )

    q_ref = np.asarray(
        telemetry["reference_quaternion"]
    )

    # -----------------------------------------
    # Quaternion inverse
    # -----------------------------------------

    q_inv = q.copy()

    q_inv[:, 1:] *= -1.0

    # -----------------------------------------
    # Quaternion multiplication
    # -----------------------------------------

    q_error = np.empty_like(q)

    for i in range(len(time)):

        w1, x1, y1, z1 = q_ref[i]

        w2, x2, y2, z2 = q_inv[i]

        q_error[i] = [

            w1*w2 - x1*x2 - y1*y2 - z1*z2,

            w1*x2 + x1*w2 + y1*z2 - z1*y2,

            w1*y2 - x1*z2 + y1*w2 + z1*x2,

            w1*z2 + x1*y2 - y1*x2 + z1*w2,

        ]

    # -----------------------------------------
    # Numerical safety
    # -----------------------------------------

    scalar = np.clip(
        np.abs(q_error[:, 0]),
        -1.0,
        1.0,
    )

    attitude_error = np.degrees(
        2.0 * np.arccos(scalar)
    )

    fig, ax = create_figure()

    ax.plot(
        time,
        attitude_error,
        linewidth=LINE_WIDTH,
    )

    ax.set_title(
        "Attitude Pointing Error",
        fontsize=TITLE_FONT_SIZE,
        fontweight="bold",
    )

    ax.set_xlabel(
        "Time [s]",
        fontsize=LABEL_FONT_SIZE,
    )

    ax.set_ylabel(
        "Pointing Error [deg]",
        fontsize=LABEL_FONT_SIZE,
    )

    configure_axes(ax)

    save_figure(
        fig,
        "attitude_error.png",
        output_directory,
    )