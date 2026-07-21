"""
Unit tests for models/environment/orbit.py
"""

import numpy as np
import pytest

from models.environment.orbit import Orbit

from tests.test_config.orbit import TEST_TIMES
from tests.test_config.tolerances import (
    ATOL_DOT_PRODUCT,
    RTOL_DEFAULT,
)


# ==========================================================
# Fixtures
# ==========================================================

@pytest.fixture
def orbit():
    """Create a default Orbit instance."""
    return Orbit()


# ==========================================================
# Initialization
# ==========================================================

def test_default_initialization(orbit):
    """Verify default orbit initialization."""

    assert orbit.time == 0.0
    assert orbit.eccentricity == 0.0

    assert orbit.position.shape == (3,)
    assert orbit.velocity.shape == (3,)

    assert orbit.radius == pytest.approx(
        orbit.semi_major_axis,
        rel=RTOL_DEFAULT,
    )


def test_initial_position(orbit):
    """Verify the initial position vector."""

    expected = np.array(
        [
            orbit.semi_major_axis,
            0.0,
            0.0,
        ]
    )

    assert np.allclose(orbit.position, expected)


def test_initial_velocity(orbit):
    """Verify the initial velocity vector."""

    expected = np.array(
        [
            0.0,
            orbit.semi_major_axis * orbit.mean_motion,
            0.0,
        ]
    )

    assert np.allclose(orbit.velocity, expected)


# ==========================================================
# Orbit Propagation
# ==========================================================

def test_update_changes_time(orbit):
    """Verify simulation time updates correctly."""

    orbit.update(100.0)

    assert orbit.time == 100.0


def test_radius_remains_constant(orbit):
    """Verify circular orbit radius remains constant."""

    for t in TEST_TIMES:

        orbit.update(t)

        assert orbit.radius == pytest.approx(
            orbit.semi_major_axis,
            rel=RTOL_DEFAULT,
        )


def test_position_velocity_shapes(orbit):
    """Verify state vector dimensions."""

    orbit.update(500.0)

    assert orbit.position.shape == (3,)
    assert orbit.velocity.shape == (3,)


def test_velocity_is_perpendicular_to_position(orbit):
    """Verify position and velocity remain orthogonal."""

    orbit.update(750.0)

    dot = np.dot(
        orbit.position,
        orbit.velocity,
    )

    assert dot == pytest.approx(
        0.0,
        abs=ATOL_DOT_PRODUCT,
    )


def test_orbital_speed_constant(orbit):
    """Verify orbital speed remains constant."""

    speed_initial = np.linalg.norm(orbit.velocity)

    orbit.update(1000.0)

    speed_final = np.linalg.norm(orbit.velocity)

    assert speed_initial == pytest.approx(
        speed_final,
        rel=RTOL_DEFAULT,
    )


# ==========================================================
# Reset
# ==========================================================

def test_reset(orbit):
    """Verify reset restores the initial state."""

    orbit.update(500.0)

    orbit.reset()

    expected_position = np.array(
        [
            orbit.semi_major_axis,
            0.0,
            0.0,
        ]
    )

    expected_velocity = np.array(
        [
            0.0,
            orbit.semi_major_axis * orbit.mean_motion,
            0.0,
        ]
    )

    assert orbit.time == 0.0

    assert np.allclose(
        orbit.position,
        expected_position,
    )

    assert np.allclose(
        orbit.velocity,
        expected_velocity,
    )