"""
Integration Test

Module
------
Orbit

Objective
---------
Verify that the Orbit model integrates correctly with the
simulation framework.

This test validates the interface only.
Orbital mechanics accuracy is covered by the unit tests.
"""

import numpy as np

from models.environment.orbit import Orbit


def test_orbit_initialization():
    """
    Verify that the Orbit model initializes correctly.
    """

    orbit = Orbit(
        mu=3.986004418e14,
        semi_major_axis=6878e3,
    )

    assert orbit.mu == 3.986004418e14
    assert orbit.semi_major_axis == 6878e3
    assert orbit.eccentricity == 0.0

    assert orbit.mean_motion > 0.0
    assert orbit.period > 0.0


def test_orbit_propagation_interface():
    """
    Verify that the orbit propagator provides
    position and velocity with the expected interface.
    """

    orbit = Orbit(
        mu=3.986004418e14,
        semi_major_axis=6878e3,
    )

    position_eci, velocity_eci = orbit.propagate(
        time=0.0,
    )

    assert isinstance(position_eci, np.ndarray)
    assert isinstance(velocity_eci, np.ndarray)

    assert position_eci.shape == (3,)
    assert velocity_eci.shape == (3,)

    assert np.all(np.isfinite(position_eci))
    assert np.all(np.isfinite(velocity_eci))


def test_orbit_multiple_time_calls():
    """
    Verify that the orbit model can be called
    repeatedly during a simulation.
    """

    orbit = Orbit(
        mu=3.986004418e14,
        semi_major_axis=6878e3,
    )

    for t in np.arange(0.0, 600.0, 10.0):

        position_eci, velocity_eci = orbit.propagate(t)

        assert position_eci.shape == (3,)
        assert velocity_eci.shape == (3,)

        assert np.all(np.isfinite(position_eci))
        assert np.all(np.isfinite(velocity_eci))