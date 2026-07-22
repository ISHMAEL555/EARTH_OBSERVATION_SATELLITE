"""
Unit tests for models/environment/orbit.py
"""

import numpy as np
import pytest

from models.environment.orbit import Orbit

from tests.test_config.tolerances import (
    RTOL_DEFAULT,
    ATOL_DOT_PRODUCT,
)


# ==========================================================
# Test Constants
# ==========================================================

MU = 3.986004418e14                     # [m^3/s^2]
EARTH_RADIUS = 6378137.0                # [m]
ALTITUDE = 600e3                        # [m]

SEMI_MAJOR_AXIS = EARTH_RADIUS + ALTITUDE

ECCENTRICITY = 0.0
INCLINATION = np.deg2rad(98.0)
RAAN = 0.0
ARGUMENT_OF_PERIGEE = 0.0
TRUE_ANOMALY = 0.0


# ==========================================================
# Fixtures
# ==========================================================

@pytest.fixture
def orbit():
    """Create a nominal circular orbit."""

    return Orbit(
        mu=MU,
        semi_major_axis=SEMI_MAJOR_AXIS,
        eccentricity=ECCENTRICITY,
        inclination=INCLINATION,
        raan=RAAN,
        argument_of_perigee=ARGUMENT_OF_PERIGEE,
        true_anomaly=TRUE_ANOMALY,
    )


# ==========================================================
# Initialization
# ==========================================================

def test_constructor(orbit):
    """Verify constructor stores orbital parameters."""

    assert orbit.mu == MU
    assert orbit.semi_major_axis == SEMI_MAJOR_AXIS
    assert orbit.eccentricity == ECCENTRICITY
    assert orbit.inclination == INCLINATION
    assert orbit.raan == RAAN
    assert orbit.argument_of_perigee == ARGUMENT_OF_PERIGEE
    assert orbit.initial_true_anomaly == TRUE_ANOMALY


def test_invalid_gravitational_parameter():

    with pytest.raises(ValueError):

        Orbit(
            mu=-1.0,
            semi_major_axis=SEMI_MAJOR_AXIS,
        )


def test_invalid_semi_major_axis():

    with pytest.raises(ValueError):

        Orbit(
            mu=MU,
            semi_major_axis=-100.0,
        )


def test_invalid_eccentricity():

    with pytest.raises(ValueError):

        Orbit(
            mu=MU,
            semi_major_axis=SEMI_MAJOR_AXIS,
            eccentricity=1.2,
        )


# ==========================================================
# Propagation
# ==========================================================

def test_returns_numpy_arrays(orbit):

    position, velocity = orbit.propagate(0.0)

    assert isinstance(position, np.ndarray)
    assert isinstance(velocity, np.ndarray)


def test_returns_correct_shapes(orbit):

    position, velocity = orbit.propagate(100.0)

    assert position.shape == (3,)
    assert velocity.shape == (3,)


def test_radius_constant(orbit):
    """Circular orbit radius must remain constant."""

    for time in np.linspace(0.0, orbit.period, 20):

        position, _ = orbit.propagate(time)

        radius = np.linalg.norm(position)

        assert radius == pytest.approx(
            SEMI_MAJOR_AXIS,
            rel=RTOL_DEFAULT,
        )


def test_velocity_perpendicular_to_position(orbit):
    """Position and velocity must remain orthogonal."""

    position, velocity = orbit.propagate(1000.0)

    dot = np.dot(position, velocity)

    assert dot == pytest.approx(
        0.0,
        abs=ATOL_DOT_PRODUCT,
    )


def test_orbital_speed_constant(orbit):
    """Orbital speed must remain constant."""

    _, velocity0 = orbit.propagate(0.0)

    _, velocity1 = orbit.propagate(2500.0)

    assert np.linalg.norm(
        velocity0
    ) == pytest.approx(
        np.linalg.norm(velocity1),
        rel=RTOL_DEFAULT,
    )


# ==========================================================
# Orbital Mechanics
# ==========================================================

def test_specific_mechanical_energy_conserved(orbit):
    """Specific mechanical energy must remain constant."""

    r1, v1 = orbit.propagate(0.0)

    r2, v2 = orbit.propagate(1500.0)

    energy1 = (
        np.linalg.norm(v1) ** 2 / 2.0
        - orbit.mu / np.linalg.norm(r1)
    )

    energy2 = (
        np.linalg.norm(v2) ** 2 / 2.0
        - orbit.mu / np.linalg.norm(r2)
    )

    assert energy1 == pytest.approx(
        energy2,
        rel=RTOL_DEFAULT,
    )


def test_angular_momentum_conserved(orbit):
    """Angular momentum magnitude must remain constant."""

    r1, v1 = orbit.propagate(0.0)

    r2, v2 = orbit.propagate(2000.0)

    h1 = np.cross(r1, v1)

    h2 = np.cross(r2, v2)

    assert np.linalg.norm(
        h1
    ) == pytest.approx(
        np.linalg.norm(h2),
        rel=RTOL_DEFAULT,
    )


def test_repeatability_after_one_period(orbit):
    """State must repeat after one orbital period."""

    r0, v0 = orbit.propagate(0.0)

    r1, v1 = orbit.propagate(
        orbit.period
    )

    assert np.allclose(
        r0,
        r1,
        rtol=RTOL_DEFAULT,
    )

    assert np.allclose(
        v0,
        v1,
        rtol=RTOL_DEFAULT,
    )


# ==========================================================
# Input Validation
# ==========================================================

def test_negative_time(orbit):

    with pytest.raises(ValueError):

        orbit.propagate(-1.0)


def test_invalid_time_type(orbit):

    with pytest.raises(TypeError):

        orbit.propagate("100")