"""
rigid_body.py
=============

Rigid-body rotational dynamics utilities.

This module implements Euler's rotational equations for a rigid spacecraft.

References
----------
- Schaub & Junkins, Analytical Mechanics of Space Systems
- Markley & Crassidis, Fundamentals of Spacecraft Attitude Determination and Control
"""

import numpy as np


# =============================================================================
# Constants
# =============================================================================

EPS = 1.0e-12


# =============================================================================
# Validation
# =============================================================================

def _validate_inertia_matrix(J):
    """
    Validate spacecraft inertia matrix.
    """

    J = np.asarray(J, dtype=float)

    if J.shape != (3, 3):
        raise ValueError(
            "Inertia matrix must have shape (3,3)."
        )

    if not np.allclose(J, J.T, atol=EPS):
        raise ValueError(
            "Inertia matrix must be symmetric."
        )

    if abs(np.linalg.det(J)) < EPS:
        raise ValueError(
            "Inertia matrix is singular."
        )

    return J


def _validate_vector(v, name):
    """
    Validate 3-element vector.
    """

    v = np.asarray(v, dtype=float)

    if v.shape != (3,):
        raise ValueError(
            f"{name} must have shape (3,)."
        )

    return v


# =============================================================================
# Angular Momentum
# =============================================================================

def angular_momentum(J, omega):
    """
    Compute spacecraft angular momentum.

    Parameters
    ----------
    J : ndarray (3,3)
        Inertia matrix.

    omega : ndarray (3,)
        Body angular velocity [rad/s].

    Returns
    -------
    ndarray (3,)
        Angular momentum [N·m·s].
    """

    J = _validate_inertia_matrix(J)
    omega = _validate_vector(
        omega,
        "Angular velocity",
    )

    return J @ omega


# =============================================================================
# Gyroscopic Torque
# =============================================================================

def gyroscopic_torque(J, omega):
    """
    Compute gyroscopic coupling torque.

        omega × (J omega)
    """

    H = angular_momentum(
        J,
        omega,
    )

    return np.cross(
        omega,
        H,
    )


# =============================================================================
# Angular Acceleration
# =============================================================================

def angular_acceleration(
    J,
    omega,
    total_torque,
):
    """
    Compute body angular acceleration.

    Euler rotational equation

        J ω_dot =
            T - ω × (Jω)
    """

    J = _validate_inertia_matrix(J)

    omega = _validate_vector(
        omega,
        "Angular velocity",
    )

    total_torque = _validate_vector(
        total_torque,
        "Total torque",
    )

    gyro = gyroscopic_torque(
        J,
        omega,
    )

    omega_dot = np.linalg.solve(
        J,
        total_torque - gyro,
    )

    return omega_dot


# =============================================================================
# Rotational Kinetic Energy
# =============================================================================

def rotational_kinetic_energy(
    J,
    omega,
):
    """
    Compute rotational kinetic energy.

        T = 1/2 ωᵀJω
    """

    H = angular_momentum(
        J,
        omega,
    )

    return 0.5 * omega @ H


# =============================================================================
# Euler Rotational Equations
# =============================================================================

def euler_equations(
    J,
    omega,
    total_torque,
):
    """
    Wrapper for Euler's rotational equations.

    Returns
    -------
    omega_dot : ndarray (3,)
    """

    return angular_acceleration(
        J,
        omega,
        total_torque,
    )