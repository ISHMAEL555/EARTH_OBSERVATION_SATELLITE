"""
tests/unit_tests/unit_test_controllers/test_quaternion_pd.py

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
# Test Configuration
# ==========================================================

KP = np.diag([5.0, 5.0, 5.0])
KD = np.diag([2.0, 2.0, 2.0])


# ==========================================================
# Fixtures
# ==========================================================

@pytest.fixture
def controller():

    return QuaternionPD(

        proportional_gain=KP,

        derivative_gain=KD,

    )


# ==========================================================
# TC-PD-001
# ==========================================================

def test_constructor(controller):

    assert np.allclose(controller.Kp, KP)

    assert np.allclose(controller.Kd, KD)


# ==========================================================
# TC-PD-002
# ==========================================================

def test_invalid_proportional_gain():

    with pytest.raises(ValueError):

        QuaternionPD(

            proportional_gain=np.eye(2),

            derivative_gain=KD,

        )


# ==========================================================
# TC-PD-003
# ==========================================================

def test_invalid_derivative_gain():

    with pytest.raises(ValueError):

        QuaternionPD(

            proportional_gain=KP,

            derivative_gain=np.eye(2),

        )


# ==========================================================
# TC-PD-004
# ==========================================================

def test_zero_error(controller):

    q = np.array([1.0, 0.0, 0.0, 0.0])

    torque = controller.compute(

        q,

        q,

        np.zeros(3),

    )

    assert np.allclose(

        torque,

        np.zeros(3),

    )


# ==========================================================
# TC-PD-005
# ==========================================================

def test_attitude_error_generates_torque(controller):

    angle = np.deg2rad(10.0)

    q_current = np.array([

        1.0,

        0.0,

        0.0,

        0.0,

    ])

    q_desired = np.array([

        np.cos(angle / 2),

        np.sin(angle / 2),

        0.0,

        0.0,

    ])

    torque = controller.compute(

        q_current,

        q_desired,

        np.zeros(3),

    )

    assert np.linalg.norm(torque) > 0.0


# ==========================================================
# TC-PD-006
# ==========================================================

def test_rate_damping(controller):

    q = np.array([1.0, 0.0, 0.0, 0.0])

    omega = np.array([

        0.1,

        -0.2,

        0.3,

    ])

    torque = controller.compute(

        q,

        q,

        omega,

    )

    assert np.linalg.norm(torque) > 0.0


# ==========================================================
# TC-PD-007
# ==========================================================

def test_combined_error(controller):

    angle = np.deg2rad(20.0)

    q_current = np.array([

        1.0,

        0.0,

        0.0,

        0.0,

    ])

    q_desired = np.array([

        np.cos(angle / 2),

        0.0,

        np.sin(angle / 2),

        0.0,

    ])

    omega = np.array([

        0.1,

        0.2,

        0.3,

    ])

    torque = controller.compute(

        q_current,

        q_desired,

        omega,

    )

    assert np.linalg.norm(torque) > 0.0


# ==========================================================
# TC-PD-008
# ==========================================================

def test_shortest_rotation(controller):

    torque = controller.compute(

        np.array([1.0, 0.0, 0.0, 0.0]),

        np.array([-1.0, 0.0, 0.0, 0.0]),

        np.zeros(3),

    )

    assert np.all(np.isfinite(torque))


# ==========================================================
# TC-PD-009
# ==========================================================

def test_invalid_quaternion_dimension(controller):

    with pytest.raises(ValueError):

        controller.compute(

            np.array([1.0, 0.0, 0.0]),

            np.array([1.0, 0.0, 0.0, 0.0]),

            np.zeros(3),

        )


# ==========================================================
# TC-PD-010
# ==========================================================

def test_invalid_rate_dimension(controller):

    with pytest.raises(ValueError):

        controller.compute(

            np.array([1.0, 0.0, 0.0, 0.0]),

            np.array([1.0, 0.0, 0.0, 0.0]),

            np.array([0.0, 0.0]),

        )


# ==========================================================
# TC-PD-011
# ==========================================================

def test_nan_quaternion(controller):

    with pytest.raises(ValueError):

        controller.compute(

            np.array([np.nan, 0.0, 0.0, 0.0]),

            np.array([1.0, 0.0, 0.0, 0.0]),

            np.zeros(3),

        )


# ==========================================================
# TC-PD-012
# ==========================================================

def test_zero_quaternion(controller):

    with pytest.raises(ValueError):

        controller.compute(

            np.zeros(4),

            np.array([1.0, 0.0, 0.0, 0.0]),

            np.zeros(3),

        )


# ==========================================================
# TC-PD-013
# ==========================================================

def test_output_dimension(controller):

    torque = controller.compute(

        np.array([1.0, 0.0, 0.0, 0.0]),

        np.array([1.0, 0.0, 0.0, 0.0]),

        np.zeros(3),

    )

    assert torque.shape == (3,)


# ==========================================================
# TC-PD-014
# ==========================================================

def test_output_is_finite(controller):

    torque = controller.compute(

        np.array([1.0, 0.0, 0.0, 0.0]),

        np.array([1.0, 0.0, 0.0, 0.0]),

        np.zeros(3),

    )

    assert np.all(np.isfinite(torque))