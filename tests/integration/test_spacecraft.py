"""
Integration Test

Module
------
Spacecraft

Objective
---------
Verify that the Spacecraft dynamics model integrates correctly
with the simulation framework.

This test validates interface compatibility only.
Dynamic correctness is verified by the unit tests.
"""

import numpy as np

from models.spacecraft import Spacecraft


def test_spacecraft_initialization():
    """
    Verify spacecraft initialization.
    """

    spacecraft = Spacecraft(
        inertia=np.diag([20.0, 25.0, 30.0]),
        mass=100.0,
    )

    q, omega = spacecraft.get_state()

    assert spacecraft.mass == 100.0

    assert spacecraft.inertia.shape == (3, 3)

    assert q.shape == (4,)
    assert omega.shape == (3,)

    assert np.all(np.isfinite(q))
    assert np.all(np.isfinite(omega))


def test_spacecraft_propagation_interface():
    """
    Verify spacecraft propagation interface.
    """

    spacecraft = Spacecraft(
        inertia=np.diag([20.0, 25.0, 30.0]),
        mass=100.0,
    )

    spacecraft.propagate(
        total_torque=np.zeros(3),
        dt=0.1,
    )

    q, omega = spacecraft.get_state()

    assert q.shape == (4,)
    assert omega.shape == (3,)

    assert np.isclose(np.linalg.norm(q), 1.0)

    assert np.all(np.isfinite(q))
    assert np.all(np.isfinite(omega))


def test_spacecraft_multiple_propagation():
    """
    Verify repeated propagation calls.
    """

    spacecraft = Spacecraft(
        inertia=np.diag([20.0, 25.0, 30.0]),
        mass=100.0,
    )

    for _ in range(100):

        spacecraft.propagate(
            total_torque=np.array([0.001, -0.002, 0.0005]),
            dt=0.1,
        )

    q, omega = spacecraft.get_state()

    assert q.shape == (4,)
    assert omega.shape == (3,)

    assert np.isclose(
        np.linalg.norm(q),
        1.0,
        atol=1e-10,
    )

    assert np.all(np.isfinite(q))
    assert np.all(np.isfinite(omega))


def test_spacecraft_reset_interface():
    """
    Verify spacecraft reset.
    """

    spacecraft = Spacecraft(
        inertia=np.diag([20.0, 25.0, 30.0]),
        mass=100.0,
    )

    spacecraft.propagate(
        total_torque=np.array([0.01, 0.01, 0.01]),
        dt=1.0,
    )

    spacecraft.reset()

    q, omega = spacecraft.get_state()

    np.testing.assert_allclose(
        q,
        np.array([1.0, 0.0, 0.0, 0.0]),
    )

    np.testing.assert_allclose(
        omega,
        np.zeros(3),
    )