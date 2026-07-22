"""
analysis/metrics.py

Simulation performance metrics.
"""

from __future__ import annotations

import numpy as np


def compute_metrics(
    telemetry: dict,
):
    """
    Compute summary metrics.
    """

    omega = np.asarray(
        telemetry["body_rates"]
    )

    control = np.asarray(
        telemetry["control_torque"]
    )

    wheel = np.asarray(
        telemetry["wheel_momentum"]
    )

    metrics = {

        "simulation_time":
            float(
                telemetry["time"][-1]
            ),

        "samples":
            len(
                telemetry["time"]
            ),

        "maximum_body_rate":
            float(
                np.max(
                    np.linalg.norm(
                        omega,
                        axis=1,
                    )
                )
            ),

        "maximum_control_torque":
            float(
                np.max(
                    np.linalg.norm(
                        control,
                        axis=1,
                    )
                )
            ),

        "maximum_wheel_momentum":
            float(
                np.max(
                    np.abs(
                        wheel
                    )
                )
            ),

    }

    return metrics