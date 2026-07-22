"""
Unit tests for models/disturbances/gravity_gradient.py
"""

import numpy as np
import pytest

from models.disturbances.gravity_gradient import GravityGradient

from tests.test_config.constants import (
    IDENTITY3,
    ZERO_VECTOR3,
)

from tests.test_config.tolerances import (
    ATOL_TORQUE,
    RTOL_DEFAULT,
)


# ==========================================================
# Test Constants
# ==========================================================

MU = 3.986004418e14              # [m^3/s^2]

EARTH_RADIUS = 6378137.0         # [m]

ORBIT_RADIUS = EARTH_RADIUS + 600e3

INERTIA_MATRIX = np.diag(
    [
        10.0,
        12.0,
        15.0,
    ]
)


# ==========================================================
# Fixtures
# ==========================================================

@pytest.fixture
def gravity_gradient():
    """Create a nominal gravity-gradient model."""

    return GravityGradient(
        gravitational_parameter=MU,
    )


@pytest.fixture
def orbit_radius():

    return ORBIT_RADIUS


@pytest.fixture
def radial_vector():

    return np.array(
        [
            1.0,
            0.0,
            0.0,
        ]
    )


# ==========================================================
# Initialization
# ==========================================================

def test_constructor(gravity_gradient):
    """Verify constructor stores parameters."""

    assert (
        gravity_gradient.gravitational_parameter
        == pytest.approx(
            MU,
            rel=RTOL_DEFAULT,
        )
    )


def test_invalid_gravitational_parameter():

    with pytest.raises(ValueError):

        GravityGradient(
            gravitational_parameter=-1.0,
        )


# ==========================================================
# Compute
# ==========================================================

def test_compute_returns_vector(
    gravity_gradient,
    orbit_radius,
    radial_vector,
):
    """Compute should return a 3-vector."""

    torque = gravity_gradient.compute(
        IDENTITY3,
        INERTIA_MATRIX,
        radial_vector,
        orbit_radius,
    )

    assert torque.shape == (3,)


def test_compute_returns_finite_values(
    gravity_gradient,
    orbit_radius,
    radial_vector,
):
    """Returned torque should contain finite values."""

    torque = gravity_gradient.compute(
        IDENTITY3,
        INERTIA_MATRIX,
        radial_vector,
        orbit_radius,
    )

    assert np.all(
        np.isfinite(torque)
    )


def test_zero_torque_for_principal_axis(
    gravity_gradient,
    orbit_radius,
):
    """Principal-axis alignment should produce zero torque."""

    inertia = np.diag(
        [
            10.0,
            12.0,
            15.0,
        ]
    )

    radial = np.array(
        [
            1.0,
            0.0,
            0.0,
        ]
    )

    torque = gravity_gradient.compute(
        IDENTITY3,
        inertia,
        radial,
        orbit_radius,
    )

    assert np.allclose(
        torque,
        ZERO_VECTOR3,
        atol=ATOL_TORQUE,
    )


def test_nonzero_torque_off_principal_axis(
    gravity_gradient,
    orbit_radius,
):
    """Off-axis radial direction should generate torque."""

    inertia = np.diag(
        [
            10.0,
            12.0,
            15.0,
        ]
    )

    radial = np.array(
        [
            1.0,
            1.0,
            0.0,
        ]
    )

    torque = gravity_gradient.compute(
        IDENTITY3,
        inertia,
        radial,
        orbit_radius,
    )

    assert np.linalg.norm(
        torque
    ) > 0.0


def test_radial_vector_normalization(
    gravity_gradient,
    orbit_radius,
):
    """Scaling the radial vector should not change the result."""

    torque1 = gravity_gradient.compute(
        IDENTITY3,
        INERTIA_MATRIX,
        np.array(
            [
                1.0,
                0.0,
                0.0,
            ]
        ),
        orbit_radius,
    )

    torque2 = gravity_gradient.compute(
        IDENTITY3,
        INERTIA_MATRIX,
        np.array(
            [
                100.0,
                0.0,
                0.0,
            ]
        ),
        orbit_radius,
    )

    assert np.allclose(
        torque1,
        torque2,
        atol=ATOL_TORQUE,
    )


def test_identity_dcm_preserves_result(
    gravity_gradient,
    orbit_radius,
):
    """Identity DCM should preserve the radial direction."""

    radial = np.array(
        [
            1.0,
            1.0,
            0.0,
        ]
    )

    torque1 = gravity_gradient.compute(
        IDENTITY3,
        INERTIA_MATRIX,
        radial,
        orbit_radius,
    )

    torque2 = gravity_gradient.compute(
        np.eye(3),
        INERTIA_MATRIX,
        radial,
        orbit_radius,
    )

    assert np.allclose(
        torque1,
        torque2,
        atol=ATOL_TORQUE,
    )


# ==========================================================
# Input Validation
# ==========================================================

def test_invalid_dcm(
    gravity_gradient,
    orbit_radius,
    radial_vector,
):

    with pytest.raises(ValueError):

        gravity_gradient.compute(
            np.eye(2),
            INERTIA_MATRIX,
            radial_vector,
            orbit_radius,
        )


def test_invalid_inertia_matrix(
    gravity_gradient,
    orbit_radius,
    radial_vector,
):

    with pytest.raises(ValueError):

        gravity_gradient.compute(
            IDENTITY3,
            np.eye(2),
            radial_vector,
            orbit_radius,
        )


def test_invalid_radial_vector(
    gravity_gradient,
    orbit_radius,
):

    with pytest.raises(ValueError):

        gravity_gradient.compute(
            IDENTITY3,
            INERTIA_MATRIX,
            np.ones(2),
            orbit_radius,
        )


def test_zero_radial_vector(
    gravity_gradient,
    orbit_radius,
):

    with pytest.raises(ValueError):

        gravity_gradient.compute(
            IDENTITY3,
            INERTIA_MATRIX,
            np.zeros(3),
            orbit_radius,
        )


def test_zero_orbit_radius(
    gravity_gradient,
    radial_vector,
):

    with pytest.raises(ValueError):

        gravity_gradient.compute(
            IDENTITY3,
            INERTIA_MATRIX,
            radial_vector,
            0.0,
        )


def test_negative_orbit_radius(
    gravity_gradient,
    radial_vector,
):

    with pytest.raises(ValueError):

        gravity_gradient.compute(
            IDENTITY3,
            INERTIA_MATRIX,
            radial_vector,
            -7000e3,
        )