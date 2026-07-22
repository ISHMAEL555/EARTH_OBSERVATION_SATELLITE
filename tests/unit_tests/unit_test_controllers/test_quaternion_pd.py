"""
Unit Tests
----------

Subsystem
---------
Quaternion PD Controller

Author
------
Kowluri Ishmael
"""

import numpy as np
import pytest

from controllers.quaternion_pd import QuaternionPD


# ==========================================================
# TC-PD-001
# Controller Initialization
# ==========================================================

def test_initialization():

    controller = QuaternionPD()

    assert np.allclose(controller.commanded_torque, np.zeros(3))
    assert np.allclose(controller.attitude_error, np.zeros(3))
    assert np.allclose(controller.rate_error, np.zeros(3))


# ==========================================================
# TC-PD-002
# Zero attitude error
# ==========================================================

def test_zero_error():

    controller = QuaternionPD()

    q = np.array([1.0, 0.0, 0.0, 0.0])
    omega = np.zeros(3)

    torque = controller.update(q, q, omega)

    assert np.allclose(torque, np.zeros(3))


# ==========================================================
# TC-PD-003
# Pure attitude error
# ==========================================================

def test_attitude_error_generates_torque():

    controller = QuaternionPD()

    angle = np.deg2rad(10.0)

    q_current = np.array([1.0, 0.0, 0.0, 0.0])

    q_desired = np.array([
        np.cos(angle / 2),
        np.sin(angle / 2),
        0.0,
        0.0
    ])

    torque = controller.update(
        q_current,
        q_desired,
        np.zeros(3)
    )

    assert np.linalg.norm(torque) > 0.0


# ==========================================================
# TC-PD-004
# Pure rate damping
# ==========================================================

def test_rate_damping():

    controller = QuaternionPD()

    q = np.array([1.0, 0.0, 0.0, 0.0])

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

    assert np.linalg.norm(torque) > 0.0


# ==========================================================
# TC-PD-005
# Combined attitude and rate error
# ==========================================================

def test_combined_error():

    controller = QuaternionPD()

    angle = np.deg2rad(20.0)

    q_current = np.array([1.0, 0.0, 0.0, 0.0])

    q_desired = np.array([
        np.cos(angle / 2),
        0.0,
        np.sin(angle / 2),
        0.0
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

    assert np.linalg.norm(torque) > 0.0


# ==========================================================
# TC-PD-006
# Shortest quaternion rotation
# ==========================================================

def test_shortest_rotation():

    controller = QuaternionPD()

    q_current = np.array([1.0, 0.0, 0.0, 0.0])

    q_desired = np.array([
        -1.0,
        0.0,
        0.0,
        0.0
    ])

    torque = controller.update(
        q_current,
        q_desired,
        np.zeros(3)
    )

    assert np.all(np.isfinite(torque))


# ==========================================================
# TC-PD-007
# Invalid quaternion dimension
# ==========================================================

def test_invalid_quaternion_dimension():

    controller = QuaternionPD()

    with pytest.raises(ValueError):

        controller.update(
            np.array([1.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.zeros(3)
        )


# ==========================================================
# TC-PD-008
# Invalid body-rate dimension
# ==========================================================

def test_invalid_rate_dimension():

    controller = QuaternionPD()

    with pytest.raises(ValueError):

        controller.update(
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.0])
        )


# ==========================================================
# TC-PD-009
# NaN quaternion
# ==========================================================

def test_nan_quaternion():

    controller = QuaternionPD()

    with pytest.raises(ValueError):

        controller.update(
            np.array([np.nan, 0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.zeros(3)
        )


# ==========================================================
# TC-PD-010
# Zero quaternion
# ==========================================================

def test_zero_quaternion():

    controller = QuaternionPD()

    with pytest.raises(ValueError):

        controller.update(
            np.zeros(4),
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.zeros(3)
        )


# ==========================================================
# TC-PD-011
# Reset controller
# ==========================================================

def test_reset():

    controller = QuaternionPD()

    angle = np.deg2rad(20.0)

    controller.update(

        np.array([1.0, 0.0, 0.0, 0.0]),

        np.array([
            np.cos(angle / 2),
            np.sin(angle / 2),
            0.0,
            0.0
        ]),

        np.array([
            0.1,
            0.2,
            0.3
        ])
    )

    controller.reset()

    assert np.allclose(controller.commanded_torque, np.zeros(3))
    assert np.allclose(controller.attitude_error, np.zeros(3))
    assert np.allclose(controller.rate_error, np.zeros(3))


# ==========================================================
# TC-PD-012
# Output dimension
# ==========================================================

def test_output_dimension():

    controller = QuaternionPD()

    torque = controller.update(

        np.array([1.0, 0.0, 0.0, 0.0]),

        np.array([1.0, 0.0, 0.0, 0.0]),

        np.zeros(3)
    )

    assert torque.shape == (3,)