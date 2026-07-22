"""
tests/unit_tests/unit_test_controllers/test_lqr.py

Unit Tests
----------

Subsystem
---------
Continuous-Time LQR Controller

Author
------
Kowluri Ishmael
"""

import numpy as np
import pytest

from controllers.lqr import LQR


# ==========================================================
# Test Configuration
# ==========================================================

INERTIA = np.diag([12.0, 10.0, 8.0])

Q = np.diag([
    10.0,
    10.0,
    10.0,
    1.0,
    1.0,
    1.0,
])

R = np.diag([
    0.1,
    0.1,
    0.1,
])


# ==========================================================
# Fixtures
# ==========================================================

@pytest.fixture
def controller():

    return LQR(

        inertia_matrix=INERTIA,

        state_weight=Q,

        control_weight=R,

    )


# ==========================================================
# TC-LQR-001
# Constructor
# ==========================================================

def test_constructor(controller):

    assert controller.A.shape == (6, 6)

    assert controller.B.shape == (6, 3)

    assert controller.K.shape == (3, 6)


# ==========================================================
# TC-LQR-002
# Gain Matrix
# ==========================================================

def test_gain_matrix(controller):

    assert np.all(np.isfinite(controller.K))


# ==========================================================
# TC-LQR-003
# Invalid Inertia
# ==========================================================

def test_invalid_inertia():

    with pytest.raises(ValueError):

        LQR(

            inertia_matrix=np.eye(2),

            state_weight=Q,

            control_weight=R,

        )


# ==========================================================
# TC-LQR-004
# Invalid Q
# ==========================================================

def test_invalid_Q():

    with pytest.raises(ValueError):

        LQR(

            inertia_matrix=INERTIA,

            state_weight=np.eye(5),

            control_weight=R,

        )


# ==========================================================
# TC-LQR-005
# Invalid R
# ==========================================================

def test_invalid_R():

    with pytest.raises(ValueError):

        LQR(

            inertia_matrix=INERTIA,

            state_weight=Q,

            control_weight=np.eye(2),

        )


# ==========================================================
# TC-LQR-006
# Zero Error
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

        atol=1e-8,

    )


# ==========================================================
# TC-LQR-007
# Attitude Error
# ==========================================================

def test_attitude_error(controller):

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
# TC-LQR-008
# Rate Error
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
# TC-LQR-009
# Combined Error
# ==========================================================

def test_combined_error(controller):

    angle = np.deg2rad(15.0)

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
# TC-LQR-010
# Shortest Rotation
# ==========================================================

def test_shortest_rotation(controller):

    torque = controller.compute(

        np.array([1.0, 0.0, 0.0, 0.0]),

        np.array([-1.0, 0.0, 0.0, 0.0]),

        np.zeros(3),

    )

    assert np.all(np.isfinite(torque))


# ==========================================================
# TC-LQR-011
# Invalid Quaternion Dimension
# ==========================================================

def test_invalid_quaternion_dimension(controller):

    with pytest.raises(ValueError):

        controller.compute(

            np.array([1.0, 0.0, 0.0]),

            np.array([1.0, 0.0, 0.0, 0.0]),

            np.zeros(3),

        )


# ==========================================================
# TC-LQR-012
# Invalid Body Rate Dimension
# ==========================================================

def test_invalid_rate_dimension(controller):

    with pytest.raises(ValueError):

        controller.compute(

            np.array([1.0, 0.0, 0.0, 0.0]),

            np.array([1.0, 0.0, 0.0, 0.0]),

            np.array([0.0, 0.0]),

        )


# ==========================================================
# TC-LQR-013
# NaN Quaternion
# ==========================================================

def test_nan_quaternion(controller):

    with pytest.raises(ValueError):

        controller.compute(

            np.array([np.nan, 0.0, 0.0, 0.0]),

            np.array([1.0, 0.0, 0.0, 0.0]),

            np.zeros(3),

        )


# ==========================================================
# TC-LQR-014
# Zero Quaternion
# ==========================================================

def test_zero_quaternion(controller):

    with pytest.raises(ValueError):

        controller.compute(

            np.zeros(4),

            np.array([1.0, 0.0, 0.0, 0.0]),

            np.zeros(3),

        )


# ==========================================================
# TC-LQR-015
# Output Dimension
# ==========================================================

def test_output_dimension(controller):

    torque = controller.compute(

        np.array([1.0, 0.0, 0.0, 0.0]),

        np.array([1.0, 0.0, 0.0, 0.0]),

        np.zeros(3),

    )

    assert torque.shape == (3,)


# ==========================================================
# TC-LQR-016
# Finite Output
# ==========================================================

def test_output_is_finite(controller):

    torque = controller.compute(

        np.array([1.0, 0.0, 0.0, 0.0]),

        np.array([1.0, 0.0, 0.0, 0.0]),

        np.zeros(3),

    )

    assert np.all(np.isfinite(torque))