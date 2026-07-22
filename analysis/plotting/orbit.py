"""
analysis/plotting/orbit.py

3D orbit visualization.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .style import (
    FIGURE_SIZE,
    TITLE_FONT_SIZE,
)


def plot_orbit(
    telemetry: dict,
    output_directory="analysis/figures",
):
    """
    Plot spacecraft trajectory in ECI.
    """

    from pathlib import Path

    position = np.asarray(
        telemetry["position_eci"]
    )

    fig = plt.figure(
        figsize=FIGURE_SIZE
    )

    ax = fig.add_subplot(
        111,
        projection="3d",
    )

    ax.plot(
        position[:, 0],
        position[:, 1],
        position[:, 2],
        linewidth=2,
        label="Orbit",
    )

    ax.scatter(
        0,
        0,
        0,
        s=200,
        label="Earth",
    )

    ax.set_xlabel("X [m]")

    ax.set_ylabel("Y [m]")

    ax.set_zlabel("Z [m]")

    ax.set_title(
        "Orbit Trajectory",
        fontsize=TITLE_FONT_SIZE,
    )

    ax.legend()

    output_directory = Path(
        output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.tight_layout()

    fig.savefig(
        output_directory / "orbit.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)