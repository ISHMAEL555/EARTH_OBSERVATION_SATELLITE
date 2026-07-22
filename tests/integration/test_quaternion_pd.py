"""
Integration Test

Module
------
QuaternionPD

Objective
---------
Verify that the QuaternionPD controller integrates correctly
with spacecraft attitude states and produces valid control
torque commands.

This test validates subsystem interfaces only.
"""

import numpy as np

from controllers.quaternion_pd import QuaternionPD


def create_controller():
    """
    Construct a nominal QuaternionPD controller.
    """

    return QuaternionPD(
        proportional_gain=5.0 * np.eye(3),
        derivative_gain=2.0 * np.eye(3),
    )


def test_quaternion_pd_initialization():
    """
    Verify the QuaternionPD controller initializes correctly.
    """

    controller = create_controller()

    assert controller.Kp.shape == (3, 3)
    assert controller.Kd.shape == (3, 3)


def test_quaternion_pd_zero_error():
    """
    Verify zero attitude error and zero body rates produce
    zero commanded torque.
    """

    controller = create_controller()

    torque = controller.compute(
        current_quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        desired_quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        body_rates=np.zeros(3),
    )

    assert torque.shape == (3,)
    assert np.allclose(torque, np.zeros(3), atol=1e-12)
    assert np.all(np.isfinite(torque))


def test_quaternion_pd_output():
    """
    Verify the controller produces a valid body torque command.
    """

    controller = create_controller()

    torque = controller.compute(
        current_quaternion=np.array(
            [0.999, 0.030, 0.010, 0.000]
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


def test_quaternion_pd_multiple_updates():
    """
    Verify repeated controller evaluations produce valid
    torque commands.
    """

    controller = create_controller()

    desired_quaternion = np.array(
        [1.0, 0.0, 0.0, 0.0]
    )

    for angle in np.linspace(0.0, np.deg2rad(10.0), 20):

        current_quaternion = np.array(
            [
                np.cos(angle / 2.0),
                np.sin(angle / 2.0),
                0.0,
                0.0,
            ]
        )

        body_rates = np.array(
            [
                0.01,
                0.0,
                -0.005,
            ]
        )

        torque = controller.compute(
            current_quaternion=current_quaternion,
            desired_quaternion=desired_quaternion,
            body_rates=body_rates,
        )

        assert torque.shape == (3,)
        assert np.all(np.isfinite(torque))


def test_quaternion_pd_closed_loop_interface():
    """
    Verify the controller accepts spacecraft attitude states
    and produces actuator-ready body torque commands.
    """

    controller = create_controller()

    spacecraft_quaternion = np.array(
        [0.998, 0.050, 0.020, 0.010]
    )

    desired_quaternion = np.array(
        [1.0, 0.0, 0.0, 0.0]
    )

    spacecraft_body_rates = np.array(
        [0.02, -0.01, 0.005]
    )

    commanded_torque = controller.compute(
        current_quaternion=spacecraft_quaternion,
        desired_quaternion=desired_quaternion,
        body_rates=spacecraft_body_rates,
    )

    assert commanded_torque.shape == (3,)
    assert np.all(np.isfinite(commanded_torque))