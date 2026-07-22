"""
analysis/metrics.py

Engineering performance metrics for the
Earth Observation Satellite ADCS simulation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ==========================================================
# Data Class
# ==========================================================

@dataclass
class SimulationMetrics:
    """
    Summary of ADCS performance.
    """

    maximum_attitude_error: float
    rms_attitude_error: float

    settling_time: float

    maximum_body_rate: float
    rms_body_rate: float

    maximum_control_torque: float
    rms_control_torque: float

    maximum_wheel_momentum: float

    quaternion_norm_error: float


# ==========================================================
# Generic Utilities
# ==========================================================

def vector_norm(vector: np.ndarray) -> np.ndarray:
    """
    Compute vector magnitude.
    """

    return np.linalg.norm(
        np.asarray(vector),
        axis=1,
    )


def peak(signal: np.ndarray) -> float:
    """
    Peak absolute value.
    """

    return float(
        np.max(
            np.abs(signal)
        )
    )


def rms(signal: np.ndarray) -> float:
    """
    Root Mean Square.
    """

    signal = np.asarray(signal)

    return float(
        np.sqrt(
            np.mean(signal ** 2)
        )
    )


# ==========================================================
# Quaternion Utilities
# ==========================================================

def quaternion_inverse(
    quaternion: np.ndarray,
) -> np.ndarray:
    """
    Quaternion inverse.
    """

    quaternion = np.asarray(quaternion)

    q_inv = quaternion.copy()

    q_inv[:, 1:] *= -1.0

    return q_inv


def quaternion_multiply(
    q1: np.ndarray,
    q2: np.ndarray,
) -> np.ndarray:
    """
    Quaternion multiplication.
    """

    result = np.empty_like(q1)

    for i in range(len(q1)):

        w1, x1, y1, z1 = q1[i]

        w2, x2, y2, z2 = q2[i]

        result[i] = [

            w1*w2 - x1*x2 - y1*y2 - z1*z2,

            w1*x2 + x1*w2 + y1*z2 - z1*y2,

            w1*y2 - x1*z2 + y1*w2 + z1*x2,

            w1*z2 + x1*y2 - y1*x2 + z1*w2,

        ]

    return result


def quaternion_norm(
    quaternion: np.ndarray,
) -> np.ndarray:
    """
    Quaternion norm.
    """

    return np.linalg.norm(
        quaternion,
        axis=1,
    )


def attitude_error(
    quaternion: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """
    Pointing error [deg].
    """

    q_error = quaternion_multiply(

        reference,

        quaternion_inverse(
            quaternion
        ),
    )

    scalar = np.clip(

        np.abs(
            q_error[:, 0]
        ),

        -1.0,

        1.0,

    )

    return np.degrees(

        2.0 * np.arccos(scalar)

    )


# ==========================================================
# Dynamic Metrics
# ==========================================================

def settling_time(
    time: np.ndarray,
    signal: np.ndarray,
    tolerance: float,
) -> float:
    """
    Compute settling time.
    """

    signal = np.abs(signal)

    for i in range(len(signal)):

        if np.all(signal[i:] <= tolerance):

            return float(time[i])

    return float("nan")


# ==========================================================
# Master Metric Function
# ==========================================================

def compute_metrics(
    telemetry: dict,
    attitude_tolerance_deg: float = 0.05,
) -> SimulationMetrics:
    """
    Compute all ADCS performance metrics.
    """

    attitude = attitude_error(

        telemetry["quaternion"],

        telemetry["reference_quaternion"],

    )

    body_rate = vector_norm(

        telemetry["body_rates"]

    )

    control = vector_norm(

        telemetry["control_torque"]

    )

    wheel = np.asarray(

        telemetry["wheel_momentum"]

    )

    q_norm = quaternion_norm(

        telemetry["quaternion"]

    )

    return SimulationMetrics(

        maximum_attitude_error=peak(attitude),

        rms_attitude_error=rms(attitude),

        settling_time=settling_time(

            telemetry["time"],

            attitude,

            attitude_tolerance_deg,

        ),

        maximum_body_rate=peak(body_rate),

        rms_body_rate=rms(body_rate),

        maximum_control_torque=peak(control),

        rms_control_torque=rms(control),

        maximum_wheel_momentum=peak(wheel),

        quaternion_norm_error=peak(
            q_norm - 1.0
        ),

    )