"""
Integration Test

Module
------
RWPyramid

Objective
---------
Verify that the Reaction Wheel Pyramid integrates correctly
with spacecraft control torque commands.

This test validates subsystem interfaces only.
"""

import numpy as np

from models.actuators.rw_pyramid import RWPyramid


def create_reaction_wheels():
    """
    Construct a nominal four-wheel pyramid.
    """

    beta = np.deg2rad(54.7356)

    wheel_axes = np.array(
        [
            [
                np.cos(beta),
                -np.cos(beta),
                -np.cos(beta),
                np.cos(beta),
            ],
            [
                np.cos(beta),
                np.cos(beta),
                -np.cos(beta),
                -np.cos(beta),
            ],
            [
                np.sin(beta),
                np.sin(beta),
                np.sin(beta),
                np.sin(beta),
            ],
        ]
    )

    wheel_axes /= np.linalg.norm(
        wheel_axes,
        axis=0,
        keepdims=True,
    )

    return RWPyramid(
        wheel_axes=wheel_axes,
        max_torque=0.2,
        max_momentum=10.0,
    )


def test_reaction_wheels_initialization():
    """
    Verify reaction wheel assembly initializes correctly.
    """

    rw = create_reaction_wheels()

    assert rw.num_wheels == 4

    assert rw.wheel_axes.shape == (3, 4)

    assert rw.actual_wheel_torque.shape == (4,)

    assert rw.wheel_momentum.shape == (4,)

    assert rw.body_torque.shape == (3,)


def test_zero_command():
    """
    Verify zero commanded torque produces zero wheel outputs.
    """

    rw = create_reaction_wheels()

    wheel_torque, momentum, body_torque = rw.update(
        commanded_body_torque=np.zeros(3),
        dt=0.1,
    )

    assert np.allclose(
        wheel_torque,
        0.0,
    )

    assert np.allclose(
        momentum,
        0.0,
    )

    assert np.allclose(
        body_torque,
        0.0,
    )


def test_reaction_wheel_output():
    """
    Verify actuator generates valid outputs.
    """

    rw = create_reaction_wheels()

    wheel_torque, momentum, body_torque = rw.update(
        commanded_body_torque=np.array(
            [0.05, -0.03, 0.02]
        ),
        dt=0.1,
    )

    assert wheel_torque.shape == (4,)

    assert momentum.shape == (4,)

    assert body_torque.shape == (3,)

    assert np.all(np.isfinite(wheel_torque))

    assert np.all(np.isfinite(momentum))

    assert np.all(np.isfinite(body_torque))


def test_multiple_updates():
    """
    Verify repeated actuator updates.
    """

    rw = create_reaction_wheels()

    for _ in range(50):

        wheel_torque, momentum, body_torque = rw.update(
            commanded_body_torque=np.array(
                [0.02, -0.01, 0.01]
            ),
            dt=0.1,
        )

        assert np.all(np.isfinite(wheel_torque))

        assert np.all(np.isfinite(momentum))

        assert np.all(np.isfinite(body_torque))


def test_controller_interface():
    """
    Verify compatibility with controller torque commands.
    """

    rw = create_reaction_wheels()

    controller_torque = np.array(
        [0.04, -0.02, 0.01]
    )

    wheel_torque, momentum, body_torque = rw.update(
        commanded_body_torque=controller_torque,
        dt=0.1,
    )

    assert wheel_torque.shape == (4,)

    assert momentum.shape == (4,)

    assert body_torque.shape == (3,)


def test_total_momentum_interface():
    """
    Verify total momentum interface.
    """

    rw = create_reaction_wheels()

    rw.update(
        commanded_body_torque=np.array(
            [0.05, 0.01, -0.02]
        ),
        dt=0.1,
    )

    total_momentum = rw.get_total_momentum()

    assert total_momentum.shape == (3,)

    assert np.all(np.isfinite(total_momentum))


def test_reset():
    """
    Verify actuator reset.
    """

    rw = create_reaction_wheels()

    rw.update(
        commanded_body_torque=np.array(
            [0.05, 0.01, -0.02]
        ),
        dt=0.1,
    )

    rw.reset()

    assert np.allclose(
        rw.actual_wheel_torque,
        0.0,
    )

    assert np.allclose(
        rw.wheel_momentum,
        0.0,
    )

    assert np.allclose(
        rw.body_torque,
        0.0,
    )