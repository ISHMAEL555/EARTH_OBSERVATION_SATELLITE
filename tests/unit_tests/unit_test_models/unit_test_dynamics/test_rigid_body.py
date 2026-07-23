import numpy as np
import pytest

from models.dynamics.rigid_body import (
    angular_momentum,
    gyroscopic_torque,
    angular_acceleration,
    rotational_kinetic_energy,
    euler_equations,
)


# =============================================================================
# Test Data
# =============================================================================

J = np.diag([10.0, 12.0, 15.0])

omega = np.array([
    0.10,
    -0.20,
    0.30,
])

torque = np.array([
    0.50,
    -0.30,
    0.20,
])


# =============================================================================
# TC-001
# Function : angular_momentum()
# Requirement:
# Angular momentum shall equal Jω.
# =============================================================================

def test_angular_momentum():

    H = angular_momentum(J, omega)

    expected = J @ omega

    assert np.allclose(H, expected)


# =============================================================================
# TC-002
# Function : gyroscopic_torque()
# Requirement:
# Gyroscopic torque shall equal ω × (Jω).
# =============================================================================

def test_gyroscopic_torque():

    gyro = gyroscopic_torque(J, omega)

    expected = np.cross(
        omega,
        J @ omega,
    )

    assert np.allclose(gyro, expected)


# =============================================================================
# TC-003
# Function : angular_acceleration()
# Requirement:
# Angular acceleration shall satisfy Euler's equation.
# =============================================================================

def test_angular_acceleration():

    omega_dot = angular_acceleration(
        J,
        omega,
        torque,
    )

    expected = np.linalg.solve(
        J,
        torque - np.cross(
            omega,
            J @ omega,
        ),
    )

    assert np.allclose(
        omega_dot,
        expected,
    )


# =============================================================================
# TC-004
# Function : euler_equations()
# Requirement:
# Wrapper shall return same result as angular_acceleration().
# =============================================================================

def test_euler_equations():

    omega_dot1 = angular_acceleration(
        J,
        omega,
        torque,
    )

    omega_dot2 = euler_equations(
        J,
        omega,
        torque,
    )

    assert np.allclose(
        omega_dot1,
        omega_dot2,
    )


# =============================================================================
# TC-005
# Function : rotational_kinetic_energy()
# Requirement:
# Rotational kinetic energy shall equal 0.5ωᵀJω.
# =============================================================================

def test_rotational_kinetic_energy():

    energy = rotational_kinetic_energy(
        J,
        omega,
    )

    expected = 0.5 * omega @ (J @ omega)

    assert np.isclose(
        energy,
        expected,
    )


# =============================================================================
# TC-006
# Requirement:
# Zero angular velocity shall produce zero momentum.
# =============================================================================

def test_zero_angular_momentum():

    H = angular_momentum(
        J,
        np.zeros(3),
    )

    assert np.allclose(
        H,
        np.zeros(3),
    )


# =============================================================================
# TC-007
# Requirement:
# Zero angular velocity shall produce zero gyroscopic torque.
# =============================================================================

def test_zero_gyroscopic_torque():

    gyro = gyroscopic_torque(
        J,
        np.zeros(3),
    )

    assert np.allclose(
        gyro,
        np.zeros(3),
    )


# =============================================================================
# TC-008
# Requirement:
# Zero torque and zero angular velocity shall produce zero acceleration.
# =============================================================================

def test_zero_acceleration():

    omega_dot = angular_acceleration(
        J,
        np.zeros(3),
        np.zeros(3),
    )

    assert np.allclose(
        omega_dot,
        np.zeros(3),
    )


# =============================================================================
# TC-009
# Requirement:
# Invalid inertia matrix size shall raise ValueError.
# =============================================================================

def test_invalid_inertia_size():

    with pytest.raises(ValueError):

        angular_momentum(
            np.eye(2),
            omega,
        )


# =============================================================================
# TC-010
# Requirement:
# Non-symmetric inertia matrix shall raise ValueError.
# =============================================================================

def test_non_symmetric_inertia():

    J_bad = np.array([
        [10, 1, 0],
        [0, 12, 0],
        [0, 0, 15],
    ])

    with pytest.raises(ValueError):

        angular_momentum(
            J_bad,
            omega,
        )


# =============================================================================
# TC-011
# Requirement:
# Singular inertia matrix shall raise ValueError.
# =============================================================================

def test_singular_inertia():

    J_bad = np.zeros((3, 3))

    with pytest.raises(ValueError):

        angular_momentum(
            J_bad,
            omega,
        )


# =============================================================================
# TC-012
# Requirement:
# Invalid angular velocity size shall raise ValueError.
# =============================================================================

def test_invalid_omega():

    with pytest.raises(ValueError):

        angular_momentum(
            J,
            np.array([1.0, 2.0]),
        )


# =============================================================================
# TC-013
# Requirement:
# Invalid torque size shall raise ValueError.
# =============================================================================

def test_invalid_torque():

    with pytest.raises(ValueError):

        angular_acceleration(
            J,
            omega,
            np.array([1.0]),
        )


# =============================================================================
# TC-014
# Requirement:
# Rotational kinetic energy shall always be non-negative.
# =============================================================================

def test_positive_kinetic_energy():

    energy = rotational_kinetic_energy(
        J,
        omega,
    )

    assert energy >= 0.0