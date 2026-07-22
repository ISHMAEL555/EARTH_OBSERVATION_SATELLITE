"""
scenarios/nadir_pointing/telemetry.py

Telemetry manager for the Nadir Pointing mission.

Responsible only for logging simulation data.
"""

from __future__ import annotations

import numpy as np


# ==========================================================
# Initialization
# ==========================================================

def initialize_telemetry(scenario):
    """
    Initialize telemetry storage.
    """

    scenario.telemetry = {

        "time": [],

        "position_eci": [],
        "velocity_eci": [],

        "quaternion": [],
        "body_rates": [],

        "reference_quaternion": [],
        "reference_body_rates": [],

        "control_torque": [],
        "disturbance_torque": [],
        "total_torque": [],

        "wheel_momentum": [],

        "magnetic_field_body": [],

    }


# ==========================================================
# Logging
# ==========================================================

def log(scenario):
    """
    Record one simulation sample.
    """

    telemetry = scenario.telemetry

    spacecraft = scenario.sim.spacecraft

    telemetry["time"].append(
        scenario.simulator.time
    )

    telemetry["position_eci"].append(
        scenario.position_eci.copy()
    )

    telemetry["velocity_eci"].append(
        scenario.velocity_eci.copy()
    )

    telemetry["quaternion"].append(
        spacecraft.q.copy()
    )

    telemetry["body_rates"].append(
        spacecraft.omega.copy()
    )

    telemetry["reference_quaternion"].append(
        scenario.q_ref.copy()
    )

    telemetry["reference_body_rates"].append(
        scenario.omega_ref.copy()
    )

    telemetry["control_torque"].append(
        scenario.control_torque.copy()
    )

    telemetry["disturbance_torque"].append(
        scenario.disturbance_torque.copy()
    )

    telemetry["total_torque"].append(
        scenario.total_torque.copy()
    )

    telemetry["wheel_momentum"].append(
        scenario.wheel_momentum.copy()
    )

    telemetry["magnetic_field_body"].append(
        scenario.magnetic_field_body.copy()
    )


# ==========================================================
# Finalization
# ==========================================================

def finalize(scenario):
    """
    Convert telemetry lists into NumPy arrays.
    """

    telemetry = scenario.telemetry

    for key in telemetry:

        telemetry[key] = np.asarray(

            telemetry[key]

        )

    return telemetry