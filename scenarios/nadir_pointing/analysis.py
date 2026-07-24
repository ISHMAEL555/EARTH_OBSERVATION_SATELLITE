"""
scenarios/nadir_pointing/analysis.py

Mission performance analysis.

This module computes quantitative performance metrics from
simulation telemetry.

No plotting is performed here.
"""

from __future__ import annotations

import numpy as np

from models.dynamics.quaternion import (
    inverse,
    multiply,
)


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
        """
        Compute spacecraft pointing error with respect to
        the reference attitude.
        """

        q = self.telemetry["quaternion"]
        q_ref = self.telemetry["reference_quaternion"]

        error_deg = np.zeros(len(q))

        for i, (qi, qr) in enumerate(zip(q, q_ref)):

            dq = multiply(qr, inverse(qi))

            angle = 2.0 * np.arccos(
                np.clip(np.abs(dq[0]), -1.0, 1.0)
            )

            error_deg[i] = np.degrees(angle)

        self.results["pointing"] = {

            "mean": np.mean(error_deg),

            "max": np.max(error_deg),

            "rms": np.sqrt(np.mean(error_deg ** 2)),

        }

    # --------------------------------------------------
    # Body Rates
    # --------------------------------------------------

    def compute_body_rate_statistics(self):

        omega = self.telemetry["body_rates"]

        magnitude = np.linalg.norm(
            omega,
            axis=1,
        )

        self.results["body_rates"] = {

            "max": np.max(magnitude),

            "mean": np.mean(magnitude),

        }

    # --------------------------------------------------
    # Control Torque
    # --------------------------------------------------

    def compute_control_statistics(self):

        torque = self.telemetry["control_torque"]

        magnitude = np.linalg.norm(
            torque,
            axis=1,
        )

        self.results["control"] = {

            "peak": np.max(magnitude),

            "average": np.mean(magnitude),

        }

    # --------------------------------------------------
    # Reaction Wheels
    # --------------------------------------------------

    def compute_reaction_wheel_statistics(self):

        H = self.telemetry["wheel_momentum"]

        H_mag = np.linalg.norm(
            H,
            axis=1,
        )

        self.results["reaction_wheels"] = {

            "maximum_momentum": np.max(H_mag),

            "average_momentum": np.mean(H_mag),

        }