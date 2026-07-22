"""
analysis/plotting/body_rates.py
"""

from __future__ import annotations

import numpy as np

from .common import (
    validate_telemetry,
    plot_timeseries,
)


def plot_body_rates(
    telemetry,
    output_directory="analysis/figures",
):

    validate_telemetry(
        telemetry,
        [
            "time",
            "body_rates",
        ],
    )

    plot_timeseries(

        time=np.asarray(
            telemetry["time"]
        ),

        data=np.asarray(
            telemetry["body_rates"]
        ),

        labels=[
            r"$\omega_x$",
            r"$\omega_y$",
            r"$\omega_z$",
        ],

        title="Body Angular Velocity",

        ylabel="Angular Velocity [rad/s]",

        filename="body_rates.png",

        output_directory=output_directory,

    )