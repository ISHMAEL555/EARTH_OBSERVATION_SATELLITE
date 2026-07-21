"""
Unit tests for models/disturbances/gravity_gradient.py
"""

import numpy as np
import pytest

from config import J, MU, R_EARTH

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
# Fixtures
# ==========================================================

@pytest.fixture
def gravity_gradient():
    """
    Create default GravityGradient model.
    """
    return GravityGradient()


@pytest.fixture
def orbit_radius():
    """
    Nominal orbit radius (600 km altitude).
    """
    return R_EARTH + 600e3


@pytest.fixture
def radial_vector():
    """
    Unit radial vector in ECI.
    """
    return np.array([
        1.0,
        0.0,
        0.0,
    ])


# ==========================================================
# Initialization
# ==========================================================

def test_default_initialization(gravity_gradient):
    """
    Verify constructor.
    """

    assert gravity_gradient.mu == pytest.approx(
        MU,
        rel=RTOL_DEFAULT,
    )


# ==========================================================
# Compute
# ==========================================================

def test_compute_returns_vector(
    gravity_gradient,
    orbit_radius,
    radial_vector,
):
    """
    Compute should return a 3-vector.
    """

    torque = gravity_gradient.compute(
        IDENTITY3,
        J,
        radial_vector,
        orbit_radius,
    )

    assert torque.shape == (3,)


def test_compute_returns_finite_values(
    gravity_gradient,
    orbit_radius,
    radial_vector,
):
    """
    Output should contain finite values.
    """

    torque = gravity_gradient.compute(
        IDENTITY3,
        J,
        radial_vector,
        orbit_radius,
    )

    assert np.all(np.isfinite(torque))


def test_zero_torque_for_principal_axis(
    gravity_gradient,
    orbit_radius,
):
    """
    Principal-axis alignment should produce zero torque.
    """

    inertia = np.diag([
        10.0,
        12.0,
        15.0,
    ])

    radial = np.array([
        1.0,
        0.0,
        0.0,
    ])

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
    """
    Off-axis radial vector should generate torque.
    """

    inertia = np.diag([
        10.0,
        12.0,
        15.0,
    ])

    radial = np.array([
        1.0,
        1.0,
        0.0,
    ])

    torque = gravity_gradient.compute(
        IDENTITY3,
        inertia,
        radial,
        orbit_radius,
    )

    assert np.linalg.norm(torque) > 0.0


def test_radial_vector_normalization(
    gravity_gradient,
    orbit_radius,
):
    """
    Radial vector magnitude should not affect the result.
    """

    torque1 = gravity_gradient.compute(
        IDENTITY3,
        J,
        np.array([
            1.0,
            0.0,
            0.0,
        ]),
        orbit_radius,
    )

    torque2 = gravity_gradient.compute(
        IDENTITY3,
        J,
        np.array([
            100.0,
            0.0,
            0.0,
        ]),
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
    """
    Identity DCM should preserve the radial direction.
    """

    inertia = np.diag([
        10.0,
        20.0,
        30.0,
    ])

    radial = np.array([
        1.0,
        1.0,
        0.0,
    ])

    torque1 = gravity_gradient.compute(
        IDENTITY3,
        inertia,
        radial,
        orbit_radius,
    )

    torque2 = gravity_gradient.compute(
        np.eye(3),
        inertia,
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
    """
    Invalid DCM should raise ValueError.
    """

    with pytest.raises(ValueError):

        gravity_gradient.compute(
            np.eye(2),
            J,
            radial_vector,
            orbit_radius,
        )


def test_invalid_inertia_matrix(
    gravity_gradient,
    orbit_radius,
    radial_vector,
):
    """
    Invalid inertia matrix should raise ValueError.
    """

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
    """
    Invalid radial vector should raise ValueError.
    """

    with pytest.raises(ValueError):

        gravity_gradient.compute(
            IDENTITY3,
            J,
            np.ones(2),
            orbit_radius,
        )


def test_zero_orbit_radius(
    gravity_gradient,
    radial_vector,
):
    """
    Zero orbit radius should raise ValueError.
    """

    with pytest.raises(ValueError):

        gravity_gradient.compute(
            IDENTITY3,
            J,
            radial_vector,
            0.0,
        )


def test_negative_orbit_radius(
    gravity_gradient,
    radial_vector,
):
    """
    Negative orbit radius should raise ValueError.
    """

    with pytest.raises(ValueError):

        gravity_gradient.compute(
            IDENTITY3,
            J,
            radial_vector,
            -7000e3,
        )