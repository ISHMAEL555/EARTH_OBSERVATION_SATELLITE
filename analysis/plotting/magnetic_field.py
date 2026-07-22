from __future__ import annotations

import numpy as np

from .common import (
    validate_telemetry,
    plot_timeseries,
)


def plot_magnetic_field(
    telemetry,
    output_directory="analysis/figures",
):

    validate_telemetry(
        telemetry,
        [
            "time",
            "magnetic_field_body",
        ],
    )

    plot_timeseries(

        time=np.asarray(
            telemetry["time"]
        ),

        data=np.asarray(
            telemetry["magnetic_field_body"]
        ),

        labels=[
            r"$B_x$",
            r"$B_y$",
            r"$B_z$",
        ],

        title="Magnetic Field",

        ylabel="Magnetic Field [T]",

        filename="magnetic_field.png",

        output_directory=output_directory,

    )