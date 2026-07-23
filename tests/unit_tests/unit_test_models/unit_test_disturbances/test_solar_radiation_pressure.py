"""
Unit tests for models/disturbances/solar_radiation_pressure.py
"""

import numpy as np
import pytest

from models.disturbances.solar_radiation_pressure import (
    SolarRadiationPressure,
)

from models.disturbances.disturbance_state import (
    DisturbanceState,
)

from tests.test_config.tolerances import (
    ATOL_FORCE,
    RTOL_DEFAULT,
)


# ==========================================================
# Fixtures
# ==========================================================

@pytest.fixture
def srp():
    """Create a nominal SRP model."""

    return SolarRadiationPressure(
        solar_radiation_pressure=4.56e-6,
        reflectivity_coefficient=1.5,
        reference_area=0.4,
    )


@pytest.fixture
def state():
    """Nominal disturbance state."""

    return DisturbanceState(

        time=0.0,

        position_eci=np.array(
            [
                7000e3,
                0.0,
                0.0,
            ]
        ),

        velocity_eci=np.array(
            [
                0.0,
                7500.0,
                0.0,
            ]
        ),

        orbit_radius=7000e3,

        radial_unit_vector_eci=np.array(
            [
                1.0,
                0.0,
                0.0,
            ]
        ),

        body_to_eci_dcm=np.eye(3),

        inertia_matrix=np.diag(
            [
                120.0,
                100.0,
                80.0,
            ]
        ),

        magnetic_field_eci=np.array(
            [
                2.0e-5,
                0.0,
                0.0,
            ]
        ),

        magnetic_field_body=np.array(
            [
                2.0e-5,
                0.0,
                0.0,
            ]
        ),

        atmospheric_density=1.0e-12,

        solar_vector_eci=np.array(
            [
                1.0,
                0.0,
                0.0,
            ]
        ),
    )


# ==========================================================
# Initialization
# ==========================================================

def test_constructor(srp):
    """Verify constructor stores parameters."""

    assert srp.solar_radiation_pressure == pytest.approx(
        4.56e-6,
        rel=RTOL_DEFAULT,
    )

    assert srp.reflectivity_coefficient == pytest.approx(
        1.5,
        rel=RTOL_DEFAULT,
    )

    assert srp.reference_area == pytest.approx(
        0.4,
        rel=RTOL_DEFAULT,
    )


# ==========================================================
# Constructor Validation
# ==========================================================

def test_invalid_pressure():

    with pytest.raises(ValueError):

        SolarRadiationPressure(
            solar_radiation_pressure=0.0,
            reflectivity_coefficient=1.5,
            reference_area=0.4,
        )


def test_invalid_reflectivity():

    with pytest.raises(ValueError):

        SolarRadiationPressure(
            solar_radiation_pressure=4.56e-6,
            reflectivity_coefficient=0.0,
            reference_area=0.4,
        )


def test_invalid_reference_area():

    with pytest.raises(ValueError):

        SolarRadiationPressure(
            solar_radiation_pressure=4.56e-6,
            reflectivity_coefficient=1.5,
            reference_area=0.0,
        )


# ==========================================================
# Compute Validation
# ==========================================================

def test_invalid_solar_vector_shape(srp, state):

    state.solar_vector_eci = np.zeros(2)

    with pytest.raises(ValueError):
        srp.compute(state)


def test_zero_solar_vector_returns_zero_force(srp, state):

    state.solar_vector_eci = np.zeros(3)

    force = srp.compute(state)

    assert np.allclose(
        force,
        np.zeros(3),
        atol=ATOL_FORCE,
    )

# ==========================================================
# Solar Force
# ==========================================================

def test_compute_returns_vector(srp, state):
    """Compute should return a 3-vector."""

    force = srp.compute(state)

    assert force.shape == (3,)


def test_compute_returns_finite_values(srp, state):
    """Computed force should contain finite values."""

    force = srp.compute(state)

    assert np.all(
        np.isfinite(force)
    )


def test_force_opposes_sun_direction(srp, state):
    """Force should oppose the Sun direction."""

    force = srp.compute(state)

    assert force[0] < 0.0


def test_force_magnitude(srp, state):
    """Verify SRP force magnitude."""

    force = srp.compute(state)

    expected = (
        4.56e-6
        * 1.5
        * 0.4
    )

    assert np.linalg.norm(
        force
    ) == pytest.approx(
        expected,
        rel=RTOL_DEFAULT,
    )


def test_solar_vector_normalized(srp, state):
    """Solar vector should be normalized internally."""

    state.solar_vector_eci = np.array(
        [
            10.0,
            0.0,
            0.0,
        ]
    )

    force = srp.compute(state)

    expected = np.array(
        [
            -4.56e-6 * 1.5 * 0.4,
            0.0,
            0.0,
        ]
    )

    assert np.allclose(
        force,
        expected,
        atol=ATOL_FORCE,
    )


def test_force_opposes_arbitrary_solar_vector(srp, state):
    """Force should oppose any valid solar direction."""

    state.solar_vector_eci = np.array(
        [
            0.0,
            1.0,
            0.0,
        ]
    )

    force = srp.compute(state)

    expected = np.array(
        [
            0.0,
            -4.56e-6 * 1.5 * 0.4,
            0.0,
        ]
    )

    assert np.allclose(
        force,
        expected,
        atol=ATOL_FORCE,
    )


def test_input_state_not_modified(srp, state):
    """The disturbance state should not be modified."""

    original = state.solar_vector_eci.copy()

    srp.compute(state)

    assert np.array_equal(
        state.solar_vector_eci,
        original,
    )