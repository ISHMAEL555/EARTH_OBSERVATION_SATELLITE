"""
Unit tests for models/disturbances/atmospheric_drag.py
"""

import numpy as np
import pytest

from config import R_EARTH

from models.disturbances.atmospheric_drag import AtmosphericDrag

from tests.test_config.constants import (
    IDENTITY3,
    ZERO_VECTOR3,
)

from tests.test_config.tolerances import (
    ATOL_FORCE,
    ATOL_TORQUE,
    RTOL_DEFAULT,
)


# ==========================================================
# Fixtures
# ==========================================================

@pytest.fixture
def atmospheric_drag():
    """
    Create a default AtmosphericDrag model.
    """

    return AtmosphericDrag(

        earth_radius=R_EARTH,

        atmosphere_surface_density=1.225,

        atmosphere_scale_height=8500.0,

        drag_coefficient=2.2,

        reference_area=0.4,

        center_of_pressure=np.array(
            [
                0.10,
                0.0,
                0.0,
            ]
        ),
    )


@pytest.fixture
def spacecraft_position():
    """
    Nominal spacecraft position.
    """

    return np.array(
        [
            R_EARTH + 600e3,
            0.0,
            0.0,
        ]
    )


@pytest.fixture
def spacecraft_velocity():
    """
    Nominal spacecraft velocity.
    """

    return np.array(
        [
            0.0,
            7550.0,
            0.0,
        ]
    )


# ==========================================================
# Initialization
# ==========================================================

def test_default_initialization(atmospheric_drag):
    """
    Verify constructor stores parameters correctly.
    """

    assert atmospheric_drag.earth_radius == R_EARTH

    assert atmospheric_drag.drag_coefficient == pytest.approx(
        2.2
    )

    assert atmospheric_drag.reference_area == pytest.approx(
        0.4
    )

    assert atmospheric_drag.center_of_pressure.shape == (3,)


# ==========================================================
# Atmospheric Density
# ==========================================================

def test_density_positive(atmospheric_drag):
    """
    Density at sea level should be positive.
    """

    density = atmospheric_drag._compute_density(0.0)

    assert density > 0.0


def test_density_decreases_with_altitude(
    atmospheric_drag,
):
    """
    Density should decrease with altitude.
    """

    rho0 = atmospheric_drag._compute_density(0.0)

    rho600 = atmospheric_drag._compute_density(
        600e3
    )

    assert rho600 < rho0


def test_negative_altitude_handled(
    atmospheric_drag,
):
    """
    Negative altitude should be clipped to zero.
    """

    rho_negative = atmospheric_drag._compute_density(
        -100.0
    )

    rho_zero = atmospheric_drag._compute_density(
        0.0
    )

    assert rho_negative == pytest.approx(
        rho_zero,
        rel=RTOL_DEFAULT,
    )


# ==========================================================
# Drag Force
# ==========================================================

def test_zero_velocity_returns_zero_force(
    atmospheric_drag,
):
    """
    Zero velocity should produce zero drag force.
    """

    force = atmospheric_drag._compute_drag_force(

        density=1.225,

        spacecraft_velocity_eci=np.zeros(3),
    )

    assert np.allclose(
        force,
        ZERO_VECTOR3,
        atol=ATOL_FORCE,
    )


def test_drag_force_opposes_velocity(
    atmospheric_drag,
):
    """
    Drag force should oppose velocity.
    """

    velocity = np.array(
        [
            7500.0,
            0.0,
            0.0,
        ]
    )

    force = atmospheric_drag._compute_drag_force(
        density=1e-12,
        spacecraft_velocity_eci=velocity,
    )

    assert force[0] < 0.0


def test_drag_force_is_finite(
    atmospheric_drag,
):
    """
    Drag force should contain finite values.
    """

    force = atmospheric_drag._compute_drag_force(

        density=1e-12,

        spacecraft_velocity_eci=np.array(
            [
                7500.0,
                0.0,
                0.0,
            ]
        ),
    )

    assert np.all(np.isfinite(force))


def test_drag_force_increases_with_velocity(
    atmospheric_drag,
):
    """
    Drag force magnitude should increase with speed.
    """

    density = 1e-12

    force1 = atmospheric_drag._compute_drag_force(

        density,

        np.array(
            [
                7000.0,
                0.0,
                0.0,
            ]
        ),
    )

    force2 = atmospheric_drag._compute_drag_force(

        density,

        np.array(
            [
                8000.0,
                0.0,
                0.0,
            ]
        ),
    )

    assert np.linalg.norm(force2) > np.linalg.norm(force1)


# ==========================================================
# Atmospheric Drag Torque
# ==========================================================

def test_compute_returns_vector(
    atmospheric_drag,
    spacecraft_position,
    spacecraft_velocity,
):
    """
    Compute should return a 3-vector.
    """

    torque = atmospheric_drag.compute(

        IDENTITY3,

        spacecraft_position,

        spacecraft_velocity,
    )

    assert torque.shape == (3,)


def test_compute_returns_finite_values(
    atmospheric_drag,
    spacecraft_position,
    spacecraft_velocity,
):
    """
    Computed torque should contain finite values.
    """

    torque = atmospheric_drag.compute(

        IDENTITY3,

        spacecraft_position,

        spacecraft_velocity,
    )

    assert np.all(np.isfinite(torque))


def test_zero_center_of_pressure_returns_zero_torque(
    spacecraft_position,
    spacecraft_velocity,
):
    """
    Zero center of pressure should produce zero torque.
    """

    model = AtmosphericDrag(

        earth_radius=R_EARTH,

        atmosphere_surface_density=1.225,

        atmosphere_scale_height=8500.0,

        drag_coefficient=2.2,

        reference_area=0.4,

        center_of_pressure=np.zeros(3),
    )

    torque = model.compute(

        IDENTITY3,

        spacecraft_position,

        spacecraft_velocity,
    )

    assert np.allclose(

        torque,

        ZERO_VECTOR3,

        atol=ATOL_TORQUE,
    )


# ==========================================================
# Input Validation
# ==========================================================

def test_invalid_dcm_shape(
    atmospheric_drag,
    spacecraft_position,
    spacecraft_velocity,
):
    """
    Invalid DCM should raise ValueError.
    """

    with pytest.raises(ValueError):

        atmospheric_drag.compute(

            np.eye(2),

            spacecraft_position,

            spacecraft_velocity,
        )


def test_invalid_position_shape(
    atmospheric_drag,
    spacecraft_velocity,
):
    """
    Invalid position vector should raise ValueError.
    """

    with pytest.raises(ValueError):

        atmospheric_drag.compute(

            IDENTITY3,

            np.zeros(2),

            spacecraft_velocity,
        )


def test_invalid_velocity_shape(
    atmospheric_drag,
    spacecraft_position,
):
    """
    Invalid velocity vector should raise ValueError.
    """

    with pytest.raises(ValueError):

        atmospheric_drag.compute(

            IDENTITY3,

            spacecraft_position,

            np.zeros(2),
        )