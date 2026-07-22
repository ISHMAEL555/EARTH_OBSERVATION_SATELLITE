"""
tests/unit_tests/unit_test_models/unit_test_actuators/test_rw_pyramid.py

Unit Tests
----------

Subsystem:
    Four-Wheel Pyramidal Reaction Wheel Assembly

Author:
    Kowluri Ishmael

Description
-----------
Unit tests for the Reaction Wheel Pyramid actuator model.
"""

import numpy as np
import pytest

from models.actuators.rw_pyramid import RWPyramid


# ==========================================================
# TC-RW-001
# ==========================================================

def test_initialization():
    """Verify correct initialization."""

    rw = RWPyramid()

    assert np.allclose(rw.commanded_body_torque, np.zeros(3))
    assert np.allclose(rw.commanded_wheel_torque, np.zeros(4))
    assert np.allclose(rw.actual_wheel_torque, np.zeros(4))
    assert np.allclose(rw.wheel_momentum, np.zeros(4))
    assert np.allclose(rw.body_torque, np.zeros(3))


# ==========================================================
# TC-RW-002
# ==========================================================

def test_torque_allocation():
    """Verify body torque allocation."""

    rw = RWPyramid()

    command = np.array([0.10, 0.00, 0.00])

    rw.update(command, dt=0.1)

    assert rw.commanded_wheel_torque.shape == (4,)
    assert np.all(np.isfinite(rw.commanded_wheel_torque))


# ==========================================================
# TC-RW-003
# ==========================================================

def test_positive_torque_saturation():
    """Verify positive wheel torque saturation."""

    rw = RWPyramid()

    command = np.array([100.0, 100.0, 100.0])

    rw.update(command, dt=0.1)

    assert np.all(
        rw.actual_wheel_torque <= rw.max_torque
    )


# ==========================================================
# TC-RW-004
# ==========================================================

def test_negative_torque_saturation():
    """Verify negative wheel torque saturation."""

    rw = RWPyramid()

    command = np.array([-100.0, -100.0, -100.0])

    rw.update(command, dt=0.1)

    assert np.all(
        rw.actual_wheel_torque >= -rw.max_torque
    )


# ==========================================================
# TC-RW-005
# ==========================================================

def test_momentum_integration():
    """Verify wheel momentum integration."""

    rw = RWPyramid()

    command = np.array([0.20, 0.00, 0.00])

    rw.update(command, dt=0.5)

    assert np.any(
        np.abs(rw.wheel_momentum) > 0.0
    )


# ==========================================================
# TC-RW-006
# ==========================================================

def test_momentum_saturation():
    """Verify wheel momentum saturation."""

    rw = RWPyramid()

    command = np.array([100.0, 100.0, 100.0])

    for _ in range(500):

        rw.update(command, dt=0.1)

    assert np.all(
        np.abs(rw.wheel_momentum)
        <= rw.max_momentum
    )


# ==========================================================
# TC-RW-007
# ==========================================================

def test_total_momentum():
    """Verify total body momentum calculation."""

    rw = RWPyramid()

    command = np.array([0.10, 0.05, 0.00])

    rw.update(command, dt=1.0)

    H = rw.get_total_momentum()

    assert H.shape == (3,)
    assert np.all(np.isfinite(H))


# ==========================================================
# TC-RW-008
# ==========================================================

def test_total_momentum_capacity():
    """Verify total momentum capacity."""

    rw = RWPyramid()

    expected = rw.num_wheels * rw.max_momentum

    assert np.isclose(
        rw.get_total_momentum_capacity(),
        expected
    )


# ==========================================================
# TC-RW-009
# ==========================================================

def test_zero_torque_command():
    """Verify zero command produces zero outputs."""

    rw = RWPyramid()

    rw.update(np.zeros(3), dt=0.1)

    assert np.allclose(
        rw.actual_wheel_torque,
        np.zeros(4)
    )

    assert np.allclose(
        rw.wheel_momentum,
        np.zeros(4)
    )

    assert np.allclose(
        rw.body_torque,
        np.zeros(3)
    )


# ==========================================================
# TC-RW-010
# ==========================================================

def test_invalid_body_torque_vector():
    """Verify invalid input vector."""

    rw = RWPyramid()

    with pytest.raises(ValueError):

        rw.update(
            np.array([1.0, 2.0]),
            dt=0.1,
        )


# ==========================================================
# TC-RW-011
# ==========================================================

def test_nan_input():
    """Verify NaN input rejection."""

    rw = RWPyramid()

    with pytest.raises(ValueError):

        rw.update(
            np.array([np.nan, 0.0, 0.0]),
            dt=0.1,
        )


# ==========================================================
# TC-RW-012
# ==========================================================

def test_reset():
    """Verify reset functionality."""

    rw = RWPyramid()

    rw.update(
        np.array([0.2, 0.1, 0.0]),
        dt=1.0,
    )

    rw.reset()

    assert np.allclose(
        rw.commanded_body_torque,
        np.zeros(3)
    )

    assert np.allclose(
        rw.commanded_wheel_torque,
        np.zeros(4)
    )

    assert np.allclose(
        rw.actual_wheel_torque,
        np.zeros(4)
    )

    assert np.allclose(
        rw.wheel_momentum,
        np.zeros(4)
    )

    assert np.allclose(
        rw.body_torque,
        np.zeros(3)
    )