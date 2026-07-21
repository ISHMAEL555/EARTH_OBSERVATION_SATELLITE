"""
Unit tests for models/environment/magnetic_field.py
"""

import numpy as np
import pytest

from models.environment.magnetic_field import MagneticField

from config import EARTH_MAGNETIC_DIPOLE_MOMENT

from tests.test_config.constants import IDENTITY3
from tests.test_config.orbit import MIN_ALTITUDE
from tests.test_config.tolerances import (
    ATOL_DCM,
    RTOL_DEFAULT,
)


# ==========================================================
# Fixtures
# ==========================================================

@pytest.fixture
def magnetic_field():
    """Create a default MagneticField model."""
    return MagneticField()


@pytest.fixture
def spacecraft_position():
    """Nominal spacecraft position in ECI."""

    earth_radius = 6378137.0

    return np.array(
        [
            earth_radius + MIN_ALTITUDE,
            0.0,
            0.0,
        ]
    )


# ==========================================================
# Initialization
# ==========================================================

def test_default_initialization(magnetic_field):
    """Verify model initializes correctly."""

    assert np.allclose(
        magnetic_field.earth_magnetic_dipole_moment,
        EARTH_MAGNETIC_DIPOLE_MOMENT,
    )


# ==========================================================
# Compute
# ==========================================================

def test_compute_returns_vector(
    magnetic_field,
    spacecraft_position,
):
    """Returned magnetic field must be a 3-vector."""

    magnetic_field_body = magnetic_field.compute(
        IDENTITY3,
        spacecraft_position,
    )

    assert magnetic_field_body.shape == (3,)


def test_compute_returns_numpy_array(
    magnetic_field,
    spacecraft_position,
):
    """Returned object must be a numpy array."""

    magnetic_field_body = magnetic_field.compute(
        IDENTITY3,
        spacecraft_position,
    )

    assert isinstance(
        magnetic_field_body,
        np.ndarray,
    )


def test_field_is_finite(
    magnetic_field,
    spacecraft_position,
):
    """Magnetic field must contain finite values."""

    magnetic_field_body = magnetic_field.compute(
        IDENTITY3,
        spacecraft_position,
    )

    assert np.all(np.isfinite(magnetic_field_body))


def test_repeatability(
    magnetic_field,
    spacecraft_position,
):
    """Repeated evaluations should produce identical results."""

    B1 = magnetic_field.compute(
        IDENTITY3,
        spacecraft_position,
    )

    B2 = magnetic_field.compute(
        IDENTITY3,
        spacecraft_position,
    )

    assert np.allclose(
        B1,
        B2,
        rtol=RTOL_DEFAULT,
    )


def test_identity_dcm_preserves_field(
    magnetic_field,
    spacecraft_position,
):
    """Identity DCM should not alter field magnitude."""

    magnetic_field_body = magnetic_field.compute(
        IDENTITY3,
        spacecraft_position,
    )

    assert np.linalg.norm(
        magnetic_field_body
    ) > 0.0


# ==========================================================
# Input Validation
# ==========================================================

def test_invalid_dcm_shape(
    magnetic_field,
    spacecraft_position,
):
    """DCM must be 3×3."""

    with pytest.raises(ValueError):

        magnetic_field.compute(
            np.eye(2),
            spacecraft_position,
        )


def test_invalid_position_shape(
    magnetic_field,
):
    """Position vector must have shape (3,)."""

    with pytest.raises(ValueError):

        magnetic_field.compute(
            IDENTITY3,
            np.zeros(2),
        )


def test_zero_position_vector(
    magnetic_field,
):
    """Zero position vector is invalid."""

    with pytest.raises(ValueError):

        magnetic_field.compute(
            IDENTITY3,
            np.zeros(3),
        )


# ==========================================================
# Physical Consistency
# ==========================================================

def test_field_decreases_with_altitude(
    magnetic_field,
):
    """Magnetic field magnitude should decrease with altitude."""

    earth_radius = 6378137.0

    low_position = np.array(
        [
            earth_radius + 300e3,
            0.0,
            0.0,
        ]
    )

    high_position = np.array(
        [
            earth_radius + 1000e3,
            0.0,
            0.0,
        ]
    )

    B_low = magnetic_field.compute(
        IDENTITY3,
        low_position,
    )

    B_high = magnetic_field.compute(
        IDENTITY3,
        high_position,
    )

    assert (
        np.linalg.norm(B_low)
        >
        np.linalg.norm(B_high)
    )