"""
Unit tests for models/disturbances/solar_radiation_pressure.py
"""

import numpy as np
import pytest

from models.disturbances.solar_radiation_pressure import (
    SolarRadiationPressure,
)

from tests.test_config.constants import (
    ZERO_VECTOR3,
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

        sun_direction_eci=np.array(
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

    assert srp.sun_direction_eci.shape == (3,)


# ==========================================================
# Constructor Validation
# ==========================================================

def test_invalid_pressure():

    with pytest.raises(ValueError):

        SolarRadiationPressure(

            solar_radiation_pressure=0.0,

            reflectivity_coefficient=1.5,

            reference_area=0.4,

            sun_direction_eci=np.array(
                [
                    1.0,
                    0.0,
                    0.0,
                ]
            ),
        )


def test_invalid_reflectivity():

    with pytest.raises(ValueError):

        SolarRadiationPressure(

            solar_radiation_pressure=4.56e-6,

            reflectivity_coefficient=0.0,

            reference_area=0.4,

            sun_direction_eci=np.array(
                [
                    1.0,
                    0.0,
                    0.0,
                ]
            ),
        )


def test_invalid_reference_area():

    with pytest.raises(ValueError):

        SolarRadiationPressure(

            solar_radiation_pressure=4.56e-6,

            reflectivity_coefficient=1.5,

            reference_area=0.0,

            sun_direction_eci=np.array(
                [
                    1.0,
                    0.0,
                    0.0,
                ]
            ),
        )


def test_invalid_sun_direction_shape():

    with pytest.raises(ValueError):

        SolarRadiationPressure(

            solar_radiation_pressure=4.56e-6,

            reflectivity_coefficient=1.5,

            reference_area=0.4,

            sun_direction_eci=np.zeros(2),
        )


def test_zero_sun_direction():

    with pytest.raises(ValueError):

        SolarRadiationPressure(

            solar_radiation_pressure=4.56e-6,

            reflectivity_coefficient=1.5,

            reference_area=0.4,

            sun_direction_eci=np.zeros(3),
        )


# ==========================================================
# Solar Force
# ==========================================================

def test_compute_returns_vector(srp):
    """Compute should return a 3-vector."""

    force = srp.compute()

    assert force.shape == (3,)


def test_compute_returns_finite_values(srp):
    """Computed force should contain finite values."""

    force = srp.compute()

    assert np.all(
        np.isfinite(force)
    )


def test_force_opposes_sun_direction(srp):
    """Force should oppose the Sun direction."""

    force = srp.compute()

    assert force[0] < 0.0


def test_force_magnitude():

    model = SolarRadiationPressure(

        solar_radiation_pressure=4.56e-6,

        reflectivity_coefficient=1.5,

        reference_area=0.4,

        sun_direction_eci=np.array(
            [
                1.0,
                0.0,
                0.0,
            ]
        ),
    )

    force = model.compute()

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


def test_sun_direction_normalized():
    """Sun direction should be normalized internally."""

    model = SolarRadiationPressure(

        solar_radiation_pressure=4.56e-6,

        reflectivity_coefficient=1.5,

        reference_area=0.4,

        sun_direction_eci=np.array(
            [
                10.0,
                0.0,
                0.0,
            ]
        ),
    )

    assert np.allclose(

        model.sun_direction_eci,

        np.array(
            [
                1.0,
                0.0,
                0.0,
            ]
        ),

        atol=ATOL_FORCE,
    )


def test_zero_reference_direction():
    """Force should align with the negative Sun direction."""

    model = SolarRadiationPressure(

        solar_radiation_pressure=4.56e-6,

        reflectivity_coefficient=1.5,

        reference_area=0.4,

        sun_direction_eci=np.array(
            [
                0.0,
                1.0,
                0.0,
            ]
        ),
    )

    force = model.compute()

    assert np.allclose(

        force,

        np.array(
            [
                0.0,
                -4.56e-6 * 1.5 * 0.4,
                0.0,
            ]
        ),

        atol=ATOL_FORCE,
    )