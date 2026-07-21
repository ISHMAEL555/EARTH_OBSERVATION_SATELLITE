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

    assert np.isclose(
        alpha[1],
        0.0
    )

    assert np.isclose(
        alpha[2],
        0.0
    )


# ==========================================================
# Propagation
# ==========================================================

def test_propagation_updates_state(spacecraft):

    q_before, omega_before = spacecraft.get_state()

    spacecraft.propagate(
        total_torque=np.array([0.5, 0.0, 0.0]),
        dt=0.1,
    )

    q_after, omega_after = spacecraft.get_state()

    assert not np.allclose(
        omega_before,
        omega_after
    )

    assert np.isclose(
        np.linalg.norm(q_after),
        1.0
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
        omega
    )

    assert np.isclose(
        np.linalg.norm(q_new),
        1.0
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