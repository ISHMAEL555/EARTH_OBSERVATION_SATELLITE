"""
Integration Test

Module
------
LQR

Objective
---------
Verify that the LQR controller integrates correctly
with spacecraft attitude states and produces valid
control torque commands.

This test validates subsystem interfaces only.
"""

import numpy as np

from controllers.lqr import LQR


def create_controller():
    """
    Construct a nominal LQR controller.
    """

    return LQR(
        inertia_matrix=np.diag([10.0, 12.0, 8.0]),
        state_weight=np.diag(
            [100, 100, 100, 10, 10, 10]
        ),
        control_weight=np.diag(
            [1, 1, 1]
        ),
    )


def test_lqr_initialization():
    """
    Verify controller initializes correctly.
    """

    controller = create_controller()

    assert controller.A.shape == (6, 6)
    assert controller.B.shape == (6, 3)
    assert controller.K.shape == (3, 6)

    assert np.all(np.isfinite(controller.K))


def test_zero_error_zero_rates():
    """
    Zero attitude error should produce
    zero commanded torque.
    """

    controller = create_controller()

    torque = controller.compute(
        current_quaternion=np.array(
            [1.0, 0.0, 0.0, 0.0]
        ),
        desired_quaternion=np.array(
            [1.0, 0.0, 0.0, 0.0]
        ),
        body_rates=np.zeros(3),
    )

    assert torque.shape == (3,)
    assert np.allclose(
        torque,
        np.zeros(3),
        atol=1e-12,
    )


def test_lqr_output():
    """
    Verify controller generates a valid
    torque command.
    """

    controller = create_controller()

    torque = controller.compute(
        current_quaternion=np.array(
            [0.999, 0.03, 0.01, 0.00]
        ),
        desired_quaternion=np.array(
            [1.0, 0.0, 0.0, 0.0]
        ),
        body_rates=np.array(
            [0.01, -0.02, 0.005]
        ),
    )

    assert torque.shape == (3,)
    assert np.all(np.isfinite(torque))


def test_multiple_controller_updates():
    """
    Verify repeated controller evaluations.
    """

    controller = create_controller()

    desired = np.array(
        [1.0, 0.0, 0.0, 0.0]
    )

    for angle in np.linspace(
        0.0,
        np.deg2rad(10.0),
        20,
    ):

        current = np.array(
            [
                np.cos(angle / 2.0),
                np.sin(angle / 2.0),
                0.0,
                0.0,
            ]
        )

        torque = controller.compute(
            current_quaternion=current,
            desired_quaternion=desired,
            body_rates=np.array(
                [0.01, 0.0, -0.005]
            ),
        )

        assert torque.shape == (3,)
        assert np.all(np.isfinite(torque))


def test_closed_loop_interface():
    """
    Verify compatibility with spacecraft state
    variables.
    """

    controller = create_controller()

    spacecraft_quaternion = np.array(
        [0.998, 0.05, 0.02, 0.01]
    )

    desired_quaternion = np.array(
        [1.0, 0.0, 0.0, 0.0]
    )

    spacecraft_rates = np.array(
        [0.02, -0.01, 0.005]
    )

    torque = controller.compute(
        current_quaternion=spacecraft_quaternion,
        desired_quaternion=desired_quaternion,
        body_rates=spacecraft_rates,
    )

    assert torque.shape == (3,)
    assert np.all(np.isfinite(torque))