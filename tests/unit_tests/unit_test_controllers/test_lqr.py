"""
Unit Tests
----------

Subsystem
---------
LQR Controller

Author
------
Kowluri Ishmael
"""

import numpy as np
import pytest

from controllers.lqr import LQR


# ==========================================================
# TC-LQR-001
# ==========================================================

def test_initialization():

    controller = LQR()

    assert controller.A.shape == (6, 6)
    assert controller.B.shape == (6, 3)

    assert controller.K.shape == (3, 6)

    assert np.allclose(controller.state, np.zeros(6))
    assert np.allclose(controller.commanded_torque, np.zeros(3))


# ==========================================================
# TC-LQR-002
# ==========================================================

def test_gain_matrix():

    controller = LQR()

    assert controller.K.shape == (3, 6)
    assert np.all(np.isfinite(controller.K))


# ==========================================================
# TC-LQR-003
# ==========================================================

def test_zero_error():

    controller = LQR()

    q = np.array([1.0, 0.0, 0.0, 0.0])

    torque = controller.update(
        q,
        q,
        np.zeros(3)
    )

    assert np.allclose(torque, np.zeros(3), atol=1e-8)


# ==========================================================
# TC-LQR-004
# ==========================================================

def test_attitude_error():

    controller = LQR()

    angle = np.deg2rad(10.0)

    q_current = np.array([1,0,0,0])

    q_desired = np.array([
        np.cos(angle/2),
        np.sin(angle/2),
        0,
        0
    ])

    torque = controller.update(
        q_current,
        q_desired,
        np.zeros(3)
    )

    assert np.linalg.norm(torque) > 0


# ==========================================================
# TC-LQR-005
# ==========================================================

def test_rate_damping():

    controller = LQR()

    q = np.array([1,0,0,0])

    omega = np.array([
        0.1,
        -0.2,
        0.3
    ])

    torque = controller.update(
        q,
        q,
        omega
    )

    assert np.linalg.norm(torque) > 0


# ==========================================================
# TC-LQR-006
# ==========================================================

def test_combined_error():

    controller = LQR()

    angle = np.deg2rad(15)

    q_current = np.array([1,0,0,0])

    q_desired = np.array([
        np.cos(angle/2),
        0,
        np.sin(angle/2),
        0
    ])

    omega = np.array([
        0.1,
        0.2,
        0.3
    ])

    torque = controller.update(
        q_current,
        q_desired,
        omega
    )

    assert np.linalg.norm(torque) > 0


# ==========================================================
# TC-LQR-007
# ==========================================================

def test_shortest_rotation():

    controller = LQR()

    torque = controller.update(
        np.array([1,0,0,0]),
        np.array([-1,0,0,0]),
        np.zeros(3)
    )

    assert np.all(np.isfinite(torque))


# ==========================================================
# TC-LQR-008
# ==========================================================

def test_invalid_quaternion_dimension():

    controller = LQR()

    with pytest.raises(ValueError):

        controller.update(
            np.array([1,0,0]),
            np.array([1,0,0,0]),
            np.zeros(3)
        )


# ==========================================================
# TC-LQR-009
# ==========================================================

def test_invalid_rate_dimension():

    controller = LQR()

    with pytest.raises(ValueError):

        controller.update(
            np.array([1,0,0,0]),
            np.array([1,0,0,0]),
            np.array([0,0])
        )


# ==========================================================
# TC-LQR-010
# ==========================================================

def test_nan_quaternion():

    controller = LQR()

    with pytest.raises(ValueError):

        controller.update(
            np.array([np.nan,0,0,0]),
            np.array([1,0,0,0]),
            np.zeros(3)
        )


# ==========================================================
# TC-LQR-011
# ==========================================================

def test_zero_quaternion():

    controller = LQR()

    with pytest.raises(ValueError):

        controller.update(
            np.zeros(4),
            np.array([1,0,0,0]),
            np.zeros(3)
        )


# ==========================================================
# TC-LQR-012
# ==========================================================

def test_reset():

    controller = LQR()

    angle = np.deg2rad(20)

    controller.update(

        np.array([1,0,0,0]),

        np.array([
            np.cos(angle/2),
            np.sin(angle/2),
            0,
            0
        ]),

        np.array([
            0.1,
            0.2,
            0.3
        ])
    )

    controller.reset()

    assert np.allclose(controller.state, np.zeros(6))
    assert np.allclose(controller.commanded_torque, np.zeros(3))
    assert np.allclose(controller.attitude_error, np.zeros(3))
    assert np.allclose(controller.rate_error, np.zeros(3))


# ==========================================================
# TC-LQR-013
# ==========================================================

def test_output_dimension():

    controller = LQR()

    torque = controller.update(
        np.array([1,0,0,0]),
        np.array([1,0,0,0]),
        np.zeros(3)
    )

    assert torque.shape == (3,)