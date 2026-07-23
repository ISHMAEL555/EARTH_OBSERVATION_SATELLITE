import numpy as np
import pytest

from models.dynamics.attitude_kinematics import (
    omega_matrix,
    quaternion_derivative,
    propagate_euler,
    propagate_rk4,
)


# =============================================================================
# TC-001
# Function : omega_matrix()
# Requirement:
# Omega matrix shall have dimensions (4,4).
# =============================================================================

def test_omega_matrix_dimensions():

    omega = np.array([1.0, 2.0, 3.0])

    Omega = omega_matrix(omega)

    assert Omega.shape == (4, 4)


# =============================================================================
# TC-002
# Function : quaternion_derivative()
# Requirement:
# Zero angular velocity shall produce zero quaternion derivative.
# =============================================================================

def test_zero_angular_velocity():

    q = np.array([1.0, 0.0, 0.0, 0.0])

    omega = np.zeros(3)

    q_dot = quaternion_derivative(
        q,
        omega,
    )

    assert np.allclose(
        q_dot,
        np.zeros(4),
        atol=1e-12,
    )


# =============================================================================
# TC-003
# Function : propagate_euler()
# Requirement:
# Euler propagation shall preserve quaternion normalization.
# =============================================================================

def test_euler_normalization():

    q = np.array([1.0, 0.0, 0.0, 0.0])

    omega = np.deg2rad(
        [0.5, 0.2, -0.3]
    )

    q_new = propagate_euler(
        q,
        omega,
        0.1,
    )

    assert np.isclose(
        np.linalg.norm(q_new),
        1.0,
        atol=1e-12,
    )


# =============================================================================
# TC-004
# Function : propagate_rk4()
# Requirement:
# RK4 propagation shall preserve quaternion normalization.
# =============================================================================

def test_rk4_normalization():

    q = np.array([1.0, 0.0, 0.0, 0.0])

    omega = np.deg2rad(
        [0.5, 0.2, -0.3]
    )

    q_new = propagate_rk4(
        q,
        omega,
        0.1,
    )

    assert np.isclose(
        np.linalg.norm(q_new),
        1.0,
        atol=1e-12,
    )


# =============================================================================
# TC-005
# Function : propagate_rk4()
# Requirement:
# Zero time step shall raise ValueError.
# =============================================================================

def test_zero_timestep():

    q = np.array([1.0, 0.0, 0.0, 0.0])

    omega = np.deg2rad(
        [1.0, 2.0, 3.0]
    )

    with pytest.raises(ValueError):

        propagate_rk4(
            q,
            omega,
            0.0,
        )


# =============================================================================
# TC-006
# Function : propagate_rk4()
# Requirement:
# Negative time step shall raise ValueError.
# =============================================================================

def test_negative_timestep():

    q = np.array([1.0, 0.0, 0.0, 0.0])

    omega = np.deg2rad(
        [1.0, 2.0, 3.0]
    )

    with pytest.raises(ValueError):

        propagate_rk4(
            q,
            omega,
            -1.0,
        )


# =============================================================================
# TC-007
# Function : propagate_rk4()
# Requirement:
# Stationary spacecraft shall maintain its attitude.
# =============================================================================

def test_stationary_spacecraft():

    q = np.array([1.0, 0.0, 0.0, 0.0])

    omega = np.zeros(3)

    q_new = propagate_rk4(
        q,
        omega,
        1.0,
    )

    assert np.allclose(
        q_new,
        q,
        atol=1e-12,
    )


# =============================================================================
# TC-008
# Function : propagate_euler(), propagate_rk4()
# Requirement:
# Euler and RK4 shall agree for sufficiently small time steps.
# =============================================================================

def test_euler_vs_rk4_small_dt():

    q = np.array([1.0, 0.0, 0.0, 0.0])

    omega = np.deg2rad(
        [0.5, 0.2, -0.3]
    )

    dt = 1.0e-4

    q_euler = propagate_euler(
        q,
        omega,
        dt,
    )

    q_rk4 = propagate_rk4(
        q,
        omega,
        dt,
    )

    assert np.allclose(
        q_euler,
        q_rk4,
        atol=1e-8,
    )


# =============================================================================
# TC-009
# Function : omega_matrix()
# Requirement:
# Invalid angular velocity size shall raise ValueError.
# =============================================================================

def test_invalid_omega_size():

    with pytest.raises(ValueError):

        omega_matrix(
            np.array([1.0, 2.0])
        )


# =============================================================================
# TC-010
# Function : quaternion_derivative()
# Requirement:
# Invalid quaternion size shall raise ValueError.
# =============================================================================

def test_invalid_quaternion_size():

    omega = np.zeros(3)

    with pytest.raises(ValueError):

        quaternion_derivative(
            np.array([1.0, 0.0, 0.0]),
            omega,
        )


# =============================================================================
# TC-011
# Function : propagate_euler()
# Requirement:
# Invalid quaternion size shall raise ValueError.
# =============================================================================

def test_propagate_euler_invalid_quaternion():

    omega = np.zeros(3)

    with pytest.raises(ValueError):

        propagate_euler(
            np.array([1.0, 0.0, 0.0]),
            omega,
            0.1,
        )


# =============================================================================
# TC-012
# Function : propagate_rk4()
# Requirement:
# Invalid angular velocity size shall raise ValueError.
# =============================================================================

def test_propagate_rk4_invalid_omega():

    q = np.array([1.0, 0.0, 0.0, 0.0])

    with pytest.raises(ValueError):

        propagate_rk4(
            q,
            np.array([1.0, 2.0]),
            0.1,
        )