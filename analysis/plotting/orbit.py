"""
analysis/plotting/orbit.py

Professional 3D orbit visualization.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .common import (
    save_figure,
    set_equal_axis_3d,
)

from .style import (
    FIGURE_SIZE,
    TITLE_FONT_SIZE,
    EARTH_RADIUS,
    EARTH_COLOR,
    ORBIT_COLOR,
    START_COLOR,
    END_COLOR,
)


def plot_orbit(
    telemetry: dict,
    output_directory="analysis/figures",
):
    """
    Plot spacecraft trajectory in the Earth-Centered Inertial (ECI) frame.
    """

    position = np.asarray(
        telemetry["position_eci"]
    )

    fig = plt.figure(figsize=FIGURE_SIZE)

    ax = fig.add_subplot(
        111,
        projection="3d",
    )

    # ======================================================
    # Earth
    # ======================================================

    u = np.linspace(0, 2 * np.pi, 80)

    v = np.linspace(0, np.pi, 40)

    x = EARTH_RADIUS * np.outer(
        np.cos(u),
        np.sin(v),
    )

    y = EARTH_RADIUS * np.outer(
        np.sin(u),
        np.sin(v),
    )

    z = EARTH_RADIUS * np.outer(
        np.ones_like(u),
        np.cos(v),
    )

    ax.plot_surface(
        x,
        y,
        z,
        color=EARTH_COLOR,
        alpha=0.65,
        linewidth=0,
        shade=True,
    )

    # ======================================================
    # Orbit
    # ======================================================

    ax.plot(
        position[:, 0],
        position[:, 1],
        position[:, 2],
        color=ORBIT_COLOR,
        linewidth=2.5,
        label="Orbit",
    )

    # ======================================================
    # Start / End
    # ======================================================

    ax.scatter(
        *position[0],
        color=START_COLOR,
        s=60,
        label="Start",
    )

    ax.scatter(
        *position[-1],
        color=END_COLOR,
        s=60,
        label="End",
    )

    # ======================================================
    # Labels
    # ======================================================

    ax.set_xlabel("X [m]")

    ax.set_ylabel("Y [m]")

    ax.set_zlabel("Z [m]")

    ax.set_title(
        "Earth Observation Satellite Orbit",
        fontsize=TITLE_FONT_SIZE,
    )

    # ======================================================
    # Equal Scaling
    # ======================================================

    set_equal_axis_3d(ax)

    # ======================================================
    # Camera
    # ======================================================

    ax.view_init(
        elev=25,
        azim=40,
    )

    ax.legend()

    save_figure(
        fig,
        "orbit.png",
        output_directory,
    )