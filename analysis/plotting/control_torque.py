from __future__ import annotations

import numpy as np

from .common import (
    validate_telemetry,
    plot_timeseries,
)


def plot_control_torque(
    telemetry,
    output_directory="analysis/figures",
):

    validate_telemetry(
        telemetry,
        [
            "time",
            "control_torque",
        ],
    )

    plot_timeseries(

        time=np.asarray(
            telemetry["time"]
        ),

        data=np.asarray(
            telemetry["control_torque"]
        ),

        labels=[
            r"$T_x$",
            r"$T_y$",
            r"$T_z$",
        ],

        title="Control Torque",

        ylabel="Torque [Nm]",

        filename="control_torque.png",

        output_directory=output_directory,

    )