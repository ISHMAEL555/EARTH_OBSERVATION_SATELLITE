"""
Unit tests for models/environment/magnetic_field.py
"""

import numpy as np
import pytest

from models.environment.magnetic_field import MagneticField

from tests.test_config.orbit import MIN_ALTITUDE
from tests.test_config.tolerances import RTOL_DEFAULT


# ==========================================================
# Test Constants
# ==========================================================

EARTH_MAGNETIC_DIPOLE_MOMENT = np.array(
    [
        0.0,
        0.0,
        7.94e22,
    ]
)

EARTH_RADIUS = 6378137.0


# ==========================================================
# Fixtures
# ==========================================================

@pytest.fixture
def magnetic_field():
    """Create a default magnetic field model."""

    return MagneticField(
        EARTH_MAGNETIC_DIPOLE_MOMENT
    )


@pytest.fixture
def spacecraft_position():
    """Nominal spacecraft position in ECI."""

    return np.array(
        [
            EARTH_RADIUS + MIN_ALTITUDE,
            0.0,
            0.0,
        ]
    )


# ==========================================================
# Initialization
# ==========================================================

def test_default_initialization(magnetic_field):
    """Verify constructor stores dipole moment."""

    assert np.allclose(
        magnetic_field.magnetic_dipole_moment,
        EARTH_MAGNETIC_DIPOLE_MOMENT,
    )


def test_invalid_dipole_shape():
    """Dipole moment must be a 3-vector."""

    with pytest.raises(ValueError):

        MagneticField(
            np.zeros(2)
        )


# ==========================================================
# Compute
# ==========================================================

def test_compute_returns_vector(
    magnetic_field,
    spacecraft_position,
):
    """Returned magnetic field must be a 3-vector."""

    magnetic_field_eci = magnetic_field.compute(
        spacecraft_position
    )

    assert magnetic_field_eci.shape == (3,)


def test_compute_returns_numpy_array(
    magnetic_field,
    spacecraft_position,
):
    """Returned object must be a NumPy array."""

    magnetic_field_eci = magnetic_field.compute(
        spacecraft_position
    )

    assert isinstance(
        magnetic_field_eci,
        np.ndarray,
    )


def test_field_is_finite(
    magnetic_field,
    spacecraft_position,
):
    """Magnetic field must contain finite values."""

    magnetic_field_eci = magnetic_field.compute(
        spacecraft_position
    )

    assert np.all(
        np.isfinite(
            magnetic_field_eci
        )
    )


def test_repeatability(
    magnetic_field,
    spacecraft_position,
):
    """Repeated evaluations should produce identical results."""

    B1 = magnetic_field.compute(
        spacecraft_position
    )

    B2 = magnetic_field.compute(
        spacecraft_position
    )

    assert np.allclose(
        B1,
        B2,
        rtol=RTOL_DEFAULT,
    )


# ==========================================================
# Input Validation
# ==========================================================

def test_invalid_position_shape(
    magnetic_field,
):
    """Position vector must have shape (3,)."""

    with pytest.raises(ValueError):

        magnetic_field.compute(
            np.zeros(2)
        )


def test_zero_position_vector(
    magnetic_field,
):
    """Zero position vector is invalid."""

    with pytest.raises(ValueError):

        magnetic_field.compute(
            np.zeros(3)
        )


# ==========================================================
# Physical Consistency
# ==========================================================

def test_field_decreases_with_altitude(
    magnetic_field,
):
    """Magnetic field magnitude decreases with altitude."""

    low_position = np.array(
        [
            EARTH_RADIUS + 300e3,
            0.0,
            0.0,
        ]
    )

    high_position = np.array(
        [
            EARTH_RADIUS + 1000e3,
            0.0,
            0.0,
        ]
    )

    B_low = magnetic_field.compute(
        low_position
    )

    B_high = magnetic_field.compute(
        high_position
    )

    assert (
        np.linalg.norm(B_low)
        >
        np.linalg.norm(B_high)
    )


def test_inverse_cube_relationship(
    magnetic_field,
):
    """
    Verify magnetic field approximately follows
    the inverse-cube law.
    """

    r1 = EARTH_RADIUS + 400e3
    r2 = EARTH_RADIUS + 800e3

    B1 = magnetic_field.compute(
        np.array([r1, 0.0, 0.0])
    )

    B2 = magnetic_field.compute(
        np.array([r2, 0.0, 0.0])
    )

    measured_ratio = (
        np.linalg.norm(B1)
        /
        np.linalg.norm(B2)
    )

    expected_ratio = (
        r2 / r1
    ) ** 3

    assert np.isclose(
        measured_ratio,
        expected_ratio,
        rtol=5e-2,
    )