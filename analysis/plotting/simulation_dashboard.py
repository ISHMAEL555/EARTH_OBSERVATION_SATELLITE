"""
analysis/plotting/simulation_dashboard.py

Mission summary dashboard for the Earth Observation Satellite.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .common import (
    plot_vector_norm,
    configure_axes,
    save_figure,
)

from .style import (
    FIGURE_SIZE,
    TITLE_FONT_SIZE,
    LINE_WIDTH,
)


def plot_simulation_dashboard(
    telemetry: dict,
    output_directory="analysis/figures",
):
    """
    Generate an engineering dashboard summarizing
    the spacecraft ADCS performance.
    """

    time = np.asarray(
        telemetry["time"]
    )

    q = np.asarray(
        telemetry["quaternion"]
    )

    q_ref = np.asarray(
        telemetry["reference_quaternion"]
    )

    body_rates = np.asarray(
        telemetry["body_rates"]
    )

    control = np.asarray(
        telemetry["control_torque"]
    )

    wheel = np.asarray(
        telemetry["wheel_momentum"]
    )

    # -----------------------------------------
    # Quaternion error
    # -----------------------------------------

    q_inv = q.copy()

    q_inv[:,1:] *= -1

    q_error = np.empty_like(q)

    for i in range(len(time)):

        w1,x1,y1,z1 = q_ref[i]

        w2,x2,y2,z2 = q_inv[i]

        q_error[i] = [

            w1*w2 - x1*x2 - y1*y2 - z1*z2,

            w1*x2 + x1*w2 + y1*z2 - z1*y2,

            w1*y2 - x1*z2 + y1*w2 + z1*x2,

            w1*z2 + x1*y2 - y1*x2 + z1*w2,

        ]

    attitude_error = np.degrees(
        2*np.arccos(
            np.clip(
                np.abs(q_error[:,0]),
                -1,
                1,
            )
        )
    )

    body_norm = plot_vector_norm(
        body_rates
    )

    control_norm = plot_vector_norm(
        control
    )

    # -----------------------------------------
    # Dashboard
    # -----------------------------------------

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13,8),
    )

    fig.suptitle(
        "Earth Observation Satellite\nADCS Performance Dashboard",
        fontsize=TITLE_FONT_SIZE,
        fontweight="bold",
    )

    # -----------------------------------------

    ax = axes[0,0]

    ax.plot(
        time,
        attitude_error,
        linewidth=LINE_WIDTH,
    )

    ax.set_title("Attitude Error")

    ax.set_ylabel("[deg]")

    configure_axes(ax)

    # -----------------------------------------

    ax = axes[0,1]

    ax.plot(
        time,
        body_norm,
        linewidth=LINE_WIDTH,
    )

    ax.set_title("Body Rate Magnitude")

    ax.set_ylabel("[rad/s]")

    configure_axes(ax)

    # -----------------------------------------

    ax = axes[1,0]

    ax.plot(
        time,
        control_norm,
        linewidth=LINE_WIDTH,
    )

    ax.set_title("Control Effort")

    ax.set_ylabel("[Nm]")

    ax.set_xlabel("Time [s]")

    configure_axes(ax)

    # -----------------------------------------

    ax = axes[1,1]

    for i in range(
        wheel.shape[1]
    ):

        ax.plot(
            time,
            wheel[:,i],
            linewidth=LINE_WIDTH,
            label=f"RW{i+1}",
        )

    ax.set_title(
        "Wheel Momentum"
    )

    ax.set_xlabel(
        "Time [s]"
    )

    ax.set_ylabel(
        "[Nms]"
    )

    configure_axes(ax)

    ax.legend()

    save_figure(
        fig,
        "simulation_dashboard.png",
        output_directory,
    )