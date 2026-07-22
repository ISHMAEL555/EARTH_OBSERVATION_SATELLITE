"""
tests/unit_tests/unit_test_models/unit_test_actuators/test_rw_pyramid.py

Unit Tests

Subsystem
---------
Four-Wheel Pyramidal Reaction Wheel Assembly
"""

import numpy as np
import pytest

from models.actuators.rw_pyramid import RWPyramid


# ==========================================================
# Test Configuration
# ==========================================================

MAX_TORQUE = 0.05          # N·m
MAX_MOMENTUM = 1.0         # N·m·s

WHEEL_AXES = np.array(
    [
        [1.0, -1.0, -1.0, 1.0],
        [1.0,  1.0, -1.0,-1.0],
        [1.0,  1.0,  1.0, 1.0],
    ],
    dtype=float,
)

WHEEL_AXES /= np.linalg.norm(
    WHEEL_AXES,
    axis=0,
)


# ==========================================================
# Fixtures
# ==========================================================

@pytest.fixture
def rw():
    return RWPyramid(
        wheel_axes=WHEEL_AXES,
        max_torque=MAX_TORQUE,
        max_momentum=MAX_MOMENTUM,
    )


# ==========================================================
# TC-RW-001
# ==========================================================

def test_constructor(rw):
    """Verify constructor."""

    assert rw.num_wheels == 4

    assert rw.wheel_axes.shape == (3, 4)

    assert rw.max_torque == pytest.approx(
        MAX_TORQUE
    )

    assert rw.max_momentum == pytest.approx(
        MAX_MOMENTUM
    )

    assert np.allclose(
        rw.actual_wheel_torque,
        np.zeros(4),
    )

    assert np.allclose(
        rw.wheel_momentum,
        np.zeros(4),
    )

    assert np.allclose(
        rw.body_torque,
        np.zeros(3),
    )


# ==========================================================
# TC-RW-002
# ==========================================================

def test_invalid_max_torque():

    with pytest.raises(ValueError):

        RWPyramid(
            wheel_axes=WHEEL_AXES,
            max_torque=0.0,
            max_momentum=1.0,
        )


# ==========================================================
# TC-RW-003
# ==========================================================

def test_invalid_max_momentum():

    with pytest.raises(ValueError):

        RWPyramid(
            wheel_axes=WHEEL_AXES,
            max_torque=0.05,
            max_momentum=0.0,
        )


# ==========================================================
# TC-RW-004
# ==========================================================

def test_invalid_wheel_axes():

    with pytest.raises(ValueError):

        RWPyramid(
            wheel_axes=np.eye(4),
            max_torque=0.05,
            max_momentum=1.0,
        )


# ==========================================================
# TC-RW-005
# ==========================================================

def test_update_returns_vectors(rw):

    wheel_torque, momentum, body_torque = rw.update(
        np.array([0.1, 0.0, 0.0]),
        dt=0.1,
    )

    assert wheel_torque.shape == (4,)
    assert momentum.shape == (4,)
    assert body_torque.shape == (3,)


# ==========================================================
# TC-RW-006
# ==========================================================

def test_positive_torque_saturation(rw):

    wheel_torque, _, _ = rw.update(
        np.array([100.0, 100.0, 100.0]),
        dt=0.1,
    )

    assert np.all(
        wheel_torque <= MAX_TORQUE
    )


# ==========================================================
# TC-RW-007
# ==========================================================

def test_negative_torque_saturation(rw):

    wheel_torque, _, _ = rw.update(
        np.array([-100.0, -100.0, -100.0]),
        dt=0.1,
    )

    assert np.all(
        wheel_torque >= -MAX_TORQUE
    )


# ==========================================================
# TC-RW-008
# ==========================================================

def test_momentum_integration(rw):

    _, momentum, _ = rw.update(
        np.array([0.2, 0.0, 0.0]),
        dt=0.5,
    )

    assert np.any(
        np.abs(momentum) > 0.0
    )


# ==========================================================
# TC-RW-009
# ==========================================================

def test_momentum_saturation(rw):

    for _ in range(500):

        rw.update(
            np.array([100.0, 100.0, 100.0]),
            dt=0.1,
        )

    assert np.all(
        np.abs(rw.wheel_momentum)
        <= MAX_MOMENTUM
    )


# ==========================================================
# TC-RW-010
# ==========================================================

def test_total_momentum(rw):

    rw.update(
        np.array([0.10, 0.05, 0.00]),
        dt=1.0,
    )

    H = rw.get_total_momentum()

    assert H.shape == (3,)

    assert np.all(
        np.isfinite(H)
    )


# ==========================================================
# TC-RW-011
# ==========================================================

def test_total_momentum_capacity(rw):

    expected = (
        rw.num_wheels
        * rw.max_momentum
    )

    assert rw.get_total_momentum_capacity() == pytest.approx(
        expected
    )


# ==========================================================
# TC-RW-012
# ==========================================================

def test_zero_command(rw):

    wheel_torque, momentum, body_torque = rw.update(
        np.zeros(3),
        dt=0.1,
    )

    assert np.allclose(
        wheel_torque,
        np.zeros(4),
    )

    assert np.allclose(
        momentum,
        np.zeros(4),
    )

    assert np.allclose(
        body_torque,
        np.zeros(3),
    )


# ==========================================================
# TC-RW-013
# ==========================================================

def test_invalid_body_torque_vector(rw):

    with pytest.raises(ValueError):

        rw.update(
            np.array([1.0, 2.0]),
            dt=0.1,
        )


# ==========================================================
# TC-RW-014
# ==========================================================

def test_invalid_dt(rw):

    with pytest.raises(ValueError):

        rw.update(
            np.zeros(3),
            dt=0.0,
        )


# ==========================================================
# TC-RW-015
# ==========================================================

def test_nan_input(rw):

    with pytest.raises(ValueError):

        rw.update(
            np.array([np.nan, 0.0, 0.0]),
            dt=0.1,
        )


# ==========================================================
# TC-RW-016
# ==========================================================

def test_reset(rw):

    rw.update(
        np.array([0.2, 0.1, 0.0]),
        dt=1.0,
    )

    rw.reset()

    assert np.allclose(
        rw.actual_wheel_torque,
        np.zeros(4),
    )

    assert np.allclose(
        rw.wheel_momentum,
        np.zeros(4),
    )

    assert np.allclose(
        rw.body_torque,
        np.zeros(3),
    )