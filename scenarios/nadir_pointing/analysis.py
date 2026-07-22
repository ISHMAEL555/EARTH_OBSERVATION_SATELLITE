"""
scenarios/nadir_pointing/analysis.py

Mission performance analysis.

This module computes quantitative performance metrics from
simulation telemetry.

No plotting is performed here.
"""

from __future__ import annotations

import numpy as np


class MissionAnalysis:
    """
    Analyze mission telemetry.
    """

    def __init__(self, telemetry):

        self.telemetry = telemetry

        self.results = {}

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def run(self):
        """
        Execute all analyses.
        """

        self.compute_pointing_error()

        self.compute_body_rate_statistics()

        self.compute_control_statistics()

        self.compute_reaction_wheel_statistics()

        return self.results

    # --------------------------------------------------
    # Pointing Performance
    # --------------------------------------------------

    def compute_pointing_error(self):

        q = self.telemetry["quaternion"]

        q_ref = self.telemetry["reference_quaternion"]

        error_deg = []

        for qi, qr in zip(q, q_ref):

            dq = quaternion_error(qi, qr)

            angle = 2.0 * np.arccos(

                np.clip(np.abs(dq[0]), -1.0, 1.0)

            )

            error_deg.append(np.degrees(angle))

        error_deg = np.asarray(error_deg)

        self.results["pointing"] = {

            "mean": np.mean(error_deg),

            "max": np.max(error_deg),

            "rms": np.sqrt(np.mean(error_deg ** 2))

        }

    # --------------------------------------------------
    # Body Rates
    # --------------------------------------------------

    def compute_body_rate_statistics(self):

        omega = self.telemetry["body_rates"]

        magnitude = np.linalg.norm(

            omega,

            axis=1

        )

        self.results["body_rates"] = {

            "max": np.max(magnitude),

            "mean": np.mean(magnitude)

        }

    # --------------------------------------------------
    # Control Torque
    # --------------------------------------------------

    def compute_control_statistics(self):

        torque = self.telemetry["control_torque"]

        magnitude = np.linalg.norm(

            torque,

            axis=1

        )

        self.results["control"] = {

            "peak": np.max(magnitude),

            "average": np.mean(magnitude)

        }

    # --------------------------------------------------
    # Reaction Wheels
    # --------------------------------------------------

    def compute_reaction_wheel_statistics(self):

        H = self.telemetry["wheel_momentum"]

        H_mag = np.linalg.norm(

            H,

            axis=1

        )

        self.results["reaction_wheels"] = {

            "maximum_momentum": np.max(H_mag),

            "average_momentum": np.mean(H_mag)

        }


# ==========================================================
# Utilities
# ==========================================================

def quaternion_error(q, q_ref):
    """
    Quaternion error.

    q_error = q_ref * q^{-1}
    """

    q_inv = q.copy()

    q_inv[1:] *= -1.0

    return quaternion_multiply(

        q_ref,

        q_inv

    )


def quaternion_multiply(q1, q2):

    w1, x1, y1, z1 = q1

    w2, x2, y2, z2 = q2

    return np.array([

        w1*w2 - x1*x2 - y1*y2 - z1*z2,

        w1*x2 + x1*w2 + y1*z2 - z1*y2,

        w1*y2 - x1*z2 + y1*w2 + z1*x2,

        w1*z2 + x1*y2 - y1*x2 + z1*w2

    ])