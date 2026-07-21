"""
Unit tests for spacecraft.py
"""

import numpy as np
import pytest

from models.spacecraft import Spacecraft


# ==========================================================
# Fixtures
# ==========================================================

@pytest.fixture
def spacecraft():

    inertia = np.diag([10.0, 12.0, 15.0])

    return Spacecraft(
        inertia=inertia,
        mass=20.0,
    )


# ==========================================================
# Initialization
# ==========================================================

def test_default_initialization(spacecraft):

    q, omega = spacecraft.get_state()

    assert np.allclose(
        q,
        np.array([1.0, 0.0, 0.0, 0.0])
    )

    assert np.allclose(
        omega,
        np.zeros(3)
    )


def test_custom_initialization():

    q0 = np.array(
        [0.70710678, 0.70710678, 0.0, 0.0]
    )

    omega0 = np.array(
        [0.1, 0.2, 0.3]
    )

    spacecraft = Spacecraft(
        inertia=np.diag([5.0, 6.0, 7.0]),
        mass=15.0,
        q0=q0,
        omega0=omega0,
    )

    q, omega = spacecraft.get_state()

    assert np.allclose(
        omega,
        omega0
    )

    assert np.isclose(
        np.linalg.norm(q),
        1.0
    )


def test_invalid_inertia():

    with pytest.raises(ValueError):

        Spacecraft(
            inertia=np.eye(2),
            mass=10.0,
        )


# ==========================================================
# Constructor Validation
# ==========================================================

def test_invalid_mass():

    with pytest.raises(ValueError):

        Spacecraft(
            inertia=np.eye(3),
            mass=0.0,
        )


def test_invalid_quaternion_shape():

    with pytest.raises(ValueError):

        Spacecraft(
            inertia=np.eye(3),
            mass=10.0,
            q0=np.ones(3),
        )


def test_zero_quaternion():

    with pytest.raises(ValueError):

        Spacecraft(
            inertia=np.eye(3),
            mass=10.0,
            q0=np.zeros(4),
        )


def test_invalid_omega_shape():

    with pytest.raises(ValueError):

        Spacecraft(
            inertia=np.eye(3),
            mass=10.0,
            omega0=np.ones(2),
        )


# ==========================================================
# Quaternion
# ==========================================================

def test_quaternion_normalization(spacecraft):

    spacecraft.set_state(
        q=np.array([2.0, 0.0, 0.0, 0.0]),
        omega=np.zeros(3),
    )

    q, _ = spacecraft.get_state()

    assert np.isclose(
        np.linalg.norm(q),
        1.0
    )


def test_zero_angular_velocity_quaternion_derivative(spacecraft):

    q_dot = spacecraft.quaternion_derivative()

    assert np.allclose(
        q_dot,
        np.zeros(4)
    )


# ==========================================================
# Angular Dynamics
# ==========================================================

def test_zero_torque_zero_acceleration(spacecraft):

    alpha = spacecraft.angular_acceleration(
        np.zeros(3)
    )

    assert np.allclose(
        alpha,
        np.zeros(3)
    )


def test_applied_torque_generates_acceleration(spacecraft):

    torque = np.array(
        [1.0, 0.0, 0.0]
    )

    alpha = spacecraft.angular_acceleration(
        torque
    )

    assert alpha[0] > 0.0

    assert np.isclose(alpha[1], 0.0)
    assert np.isclose(alpha[2], 0.0)


# ==========================================================
# Propagation
# ==========================================================

def test_propagation_updates_state(spacecraft):

    _, omega_before = spacecraft.get_state()

    spacecraft.propagate(
        total_torque=np.array([0.5, 0.0, 0.0]),
        dt=0.1,
    )

    q_after, omega_after = spacecraft.get_state()

    assert not np.allclose(
        omega_before,
        omega_after,
    )

    assert np.isclose(
        np.linalg.norm(q_after),
        1.0,
    )


def test_invalid_timestep(spacecraft):

    with pytest.raises(ValueError):

        spacecraft.propagate(
            total_torque=np.zeros(3),
            dt=0.0,
        )


# ==========================================================
# Utility Functions
# ==========================================================

def test_set_state(spacecraft):

    q = np.array(
        [1.0, 0.0, 0.0, 0.0]
    )

    omega = np.array(
        [1.0, 2.0, 3.0]
    )

    spacecraft.set_state(
        q,
        omega,
    )

    q_new, omega_new = spacecraft.get_state()

    assert np.allclose(
        omega_new,
        omega,
    )

    assert np.isclose(
        np.linalg.norm(q_new),
        1.0,
    )


def test_reset(spacecraft):

    spacecraft.set_state(
        q=np.array(
            [0.5, 0.5, 0.5, 0.5]
        ),
        omega=np.array(
            [1.0, 2.0, 3.0]
        ),
    )

    spacecraft.reset()

    q, omega = spacecraft.get_state()

    assert np.allclose(
        q,
        np.array(
            [1.0, 0.0, 0.0, 0.0]
        )
    )

    assert np.allclose(
        omega,
        np.zeros(3)
    )

    # ==========================================================
# Physics Verification
# ==========================================================

def test_zero_state_remains_zero(spacecraft):
    """
    If the spacecraft starts at rest with zero applied torque,
    it should remain at rest.
    """

    q_before, omega_before = spacecraft.get_state()

    spacecraft.propagate(
        total_torque=np.zeros(3),
        dt=0.1,
    )

    q_after, omega_after = spacecraft.get_state()

    assert np.allclose(q_before, q_after)
    assert np.allclose(omega_before, omega_after)


def test_zero_external_torque_produces_valid_state(spacecraft):
    """
    With non-zero angular velocity and zero external torque,
    the spacecraft should still produce a valid numerical state.
    """

    spacecraft.set_state(
        q=np.array([1.0, 0.0, 0.0, 0.0]),
        omega=np.array([0.1, 0.2, 0.3]),
    )

    spacecraft.propagate(
        total_torque=np.zeros(3),
        dt=0.1,
    )

    q, omega = spacecraft.get_state()

    assert np.all(np.isfinite(q))
    assert np.all(np.isfinite(omega))
    assert np.isclose(np.linalg.norm(q), 1.0)


def test_zero_torque_propagation_is_stable(spacecraft):
    """
    Long-duration torque-free propagation should remain numerically stable.
    """

    spacecraft.set_state(
        q=np.array([1.0, 0.0, 0.0, 0.0]),
        omega=np.array([0.1, 0.2, 0.3]),
    )

    for _ in range(100):

        spacecraft.propagate(
            total_torque=np.zeros(3),
            dt=0.01,
        )

    q, omega = spacecraft.get_state()

    assert np.all(np.isfinite(q))
    assert np.all(np.isfinite(omega))
    assert np.isclose(np.linalg.norm(q), 1.0)


def test_long_term_quaternion_normalization(spacecraft):
    """
    Quaternion should remain normalized after many propagation steps.
    """

    for _ in range(1000):

        spacecraft.propagate(
            total_torque=np.zeros(3),
            dt=0.01,
        )

    q, _ = spacecraft.get_state()

    assert np.isclose(
        np.linalg.norm(q),
        1.0,
        atol=1e-10,
    )


def test_state_remains_finite_after_long_propagation(spacecraft):
    """
    State should remain finite under long-duration propagation.
    """

    for _ in range(1000):

        spacecraft.propagate(
            total_torque=np.array([0.01, 0.02, 0.01]),
            dt=0.01,
        )

    q, omega = spacecraft.get_state()

    assert np.all(np.isfinite(q))
    assert np.all(np.isfinite(omega))

    # ==========================================================
# Validation
# ==========================================================

def test_invalid_set_state_quaternion(spacecraft):
    """
    Quaternion must have four elements.
    """

    with pytest.raises(ValueError):

        spacecraft.set_state(
            q=np.ones(3),
            omega=np.zeros(3),
        )


def test_invalid_set_state_omega(spacecraft):
    """
    Angular velocity must have three elements.
    """

    with pytest.raises(ValueError):

        spacecraft.set_state(
            q=np.array([1.0, 0.0, 0.0, 0.0]),
            omega=np.ones(2),
        )


def test_invalid_set_state_zero_quaternion(spacecraft):
    """
    Zero quaternion is not physically valid.
    """

    with pytest.raises(ValueError):

        spacecraft.set_state(
            q=np.zeros(4),
            omega=np.zeros(3),
        )


def test_invalid_torque_shape(spacecraft):
    """
    Applied torque vector must have three elements.
    """

    with pytest.raises(ValueError):

        spacecraft.angular_acceleration(
            np.ones(2)
        )


# ==========================================================
# Properties
# ==========================================================

def test_inertia_property(spacecraft):

    assert np.allclose(
        spacecraft.inertia,
        np.diag([10.0, 12.0, 15.0]),
    )


# ==========================================================
# Additional Robustness Tests
# ==========================================================

def test_get_state_returns_copy(spacecraft):
    """
    get_state() should return copies instead of references.
    """

    q, omega = spacecraft.get_state()

    q[0] = 999.0
    omega[0] = 999.0

    q_internal, omega_internal = spacecraft.get_state()

    assert q_internal[0] != 999.0
    assert omega_internal[0] != 999.0


def test_multiple_reset_calls(spacecraft):
    """
    reset() should always restore the default state.
    """

    spacecraft.set_state(
        np.array([0.5, 0.5, 0.5, 0.5]),
        np.array([1.0, 2.0, 3.0]),
    )

    spacecraft.reset()
    spacecraft.reset()

    q, omega = spacecraft.get_state()

    assert np.allclose(
        q,
        np.array([1.0, 0.0, 0.0, 0.0]),
    )

    assert np.allclose(
        omega,
        np.zeros(3),
    )


def test_quaternion_remains_normalized_after_multiple_updates(spacecraft):
    """
    Quaternion should remain normalized after repeated updates.
    """

    for _ in range(500):

        spacecraft.propagate(
            total_torque=np.array([0.05, -0.02, 0.03]),
            dt=0.02,
        )

    q, _ = spacecraft.get_state()

    assert np.isclose(
        np.linalg.norm(q),
        1.0,
        atol=1e-10,
    )


def test_zero_torque_zero_initial_rate(spacecraft):
    """
    Zero torque with zero initial angular velocity should
    produce zero angular acceleration.
    """

    alpha = spacecraft.angular_acceleration(
        np.zeros(3)
    )

    assert np.allclose(
        alpha,
        np.zeros(3),
    )