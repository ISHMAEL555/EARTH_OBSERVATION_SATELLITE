"""
attitude_kinematics.py
======================

Quaternion attitude kinematics.

References
----------
- Schaub & Junkins
- Markley & Crassidis
"""

import numpy as np

from .quaternion import (
    enforce_unique,
)


# =============================================================================
# Validation
# =============================================================================

def _validate_quaternion(q):
    """
    Validate quaternion.
    """

    q = np.asarray(
        q,
        dtype=float,
    )

    if q.shape != (4,):
        raise ValueError(
            "Quaternion must have shape (4,)."
        )

    return q


def _validate_angular_velocity(omega):
    """
    Validate angular velocity.
    """

    omega = np.asarray(
        omega,
        dtype=float,
    )

    if omega.shape != (3,):
        raise ValueError(
            "Angular velocity must have shape (3,)."
        )

    return omega


def _validate_time_step(dt):
    """
    Validate integration time step.
    """

    dt = float(dt)

    if dt <= 0.0:
        raise ValueError(
            "Time step must be positive."
        )

    return dt


# =============================================================================
# Omega Matrix
# =============================================================================

def omega_matrix(
    omega,
):
    """
    Construct the quaternion Omega matrix.
    """

    omega = _validate_angular_velocity(
        omega
    )

    wx, wy, wz = omega

    return np.array(

        [

            [0.0, -wx, -wy, -wz],

            [wx, 0.0, wz, -wy],

            [wy, -wz, 0.0, wx],

            [wz, wy, -wx, 0.0],

        ],

        dtype=float,

    )


# =============================================================================
# Quaternion Derivative
# =============================================================================

def quaternion_derivative(
    q,
    omega,
):
    """
    Compute quaternion derivative.
    """

    q = _validate_quaternion(q)

    Omega = omega_matrix(
        omega
    )

    return 0.5 * (
        Omega @ q
    )


# =============================================================================
# Euler Integration
# =============================================================================

def propagate_euler(
    q,
    omega,
    dt,
):
    """
    First-order Euler propagation.
    """

    q = _validate_quaternion(q)
    omega = _validate_angular_velocity(omega)
    dt = _validate_time_step(dt)

    q_dot = quaternion_derivative(
        q,
        omega,
    )

    q_new = q + q_dot * dt

    return enforce_unique(
        q_new
    )


# =============================================================================
# Runge-Kutta 4 Integration
# =============================================================================

def propagate_rk4(
    q,
    omega,
    dt,
):
    """
    Fourth-order Runge-Kutta propagation.
    """

    q = _validate_quaternion(q)
    omega = _validate_angular_velocity(omega)
    dt = _validate_time_step(dt)

    k1 = quaternion_derivative(
        q,
        omega,
    )

    k2 = quaternion_derivative(
        q + 0.5 * dt * k1,
        omega,
    )

    k3 = quaternion_derivative(
        q + 0.5 * dt * k2,
        omega,
    )

    k4 = quaternion_derivative(
        q + dt * k3,
        omega,
    )

    q_new = q + (dt / 6.0) * (

        k1

        + 2.0 * k2

        + 2.0 * k3

        + k4

    )

    return enforce_unique(
        q_new
    )