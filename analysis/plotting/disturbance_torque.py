from __future__ import annotations

import numpy as np

from .common import (
    validate_telemetry,
    plot_timeseries,
)


def plot_disturbance_torque(
    telemetry,
    output_directory="analysis/figures",
):

    validate_telemetry(
        telemetry,
        [
            "time",
            "disturbance_torque",
        ],
    )

    plot_timeseries(

        time=np.asarray(
            telemetry["time"]
        ),

        data=np.asarray(
            telemetry["disturbance_torque"]
        ),

        labels=[
            r"$T_x$",
            r"$T_y$",
            r"$T_z$",
        ],

        title="Disturbance Torque",

        ylabel="Torque [Nm]",

        filename="disturbance_torque.png",

        output_directory=output_directory,

    )