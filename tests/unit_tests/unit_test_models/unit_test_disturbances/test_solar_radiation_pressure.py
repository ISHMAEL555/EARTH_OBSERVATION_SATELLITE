"""
Unit tests for models/disturbances/solar_radiation_pressure.py
"""

import numpy as np
import pytest

from models.disturbances.solar_radiation_pressure import (
    SolarRadiationPressure,
)

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
def srp():
    """
    Default Solar Radiation Pressure model.
    """

    return SolarRadiationPressure(

        solar_radiation_pressure=4.56e-6,

        reflectivity_coefficient=1.5,

        reference_area=0.4,

        center_of_pressure=np.array(
            [
                0.10,
                0.0,
                0.0,
            ]
        ),

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

def test_default_initialization(srp):
    """
    Verify constructor stores parameters.
    """

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

    assert srp.center_of_pressure.shape == (3,)

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
            center_of_pressure=np.zeros(3),
            sun_direction_eci=np.array([1.0,0.0,0.0]),
        )


def test_invalid_reflectivity():

    with pytest.raises(ValueError):

        SolarRadiationPressure(
            solar_radiation_pressure=4.56e-6,
            reflectivity_coefficient=0.0,
            reference_area=0.4,
            center_of_pressure=np.zeros(3),
            sun_direction_eci=np.array([1.0,0.0,0.0]),
        )


def test_invalid_reference_area():

    with pytest.raises(ValueError):

        SolarRadiationPressure(
            solar_radiation_pressure=4.56e-6,
            reflectivity_coefficient=1.5,
            reference_area=0.0,
            center_of_pressure=np.zeros(3),
            sun_direction_eci=np.array([1.0,0.0,0.0]),
        )


def test_invalid_center_of_pressure():

    with pytest.raises(ValueError):

        SolarRadiationPressure(
            solar_radiation_pressure=4.56e-6,
            reflectivity_coefficient=1.5,
            reference_area=0.4,
            center_of_pressure=np.zeros(2),
            sun_direction_eci=np.array([1.0,0.0,0.0]),
        )


def test_invalid_sun_direction_shape():

    with pytest.raises(ValueError):

        SolarRadiationPressure(
            solar_radiation_pressure=4.56e-6,
            reflectivity_coefficient=1.5,
            reference_area=0.4,
            center_of_pressure=np.zeros(3),
            sun_direction_eci=np.zeros(2),
        )


def test_zero_sun_direction():

    with pytest.raises(ValueError):

        SolarRadiationPressure(
            solar_radiation_pressure=4.56e-6,
            reflectivity_coefficient=1.5,
            reference_area=0.4,
            center_of_pressure=np.zeros(3),
            sun_direction_eci=np.zeros(3),
        )


# ==========================================================
# Solar Force
# ==========================================================

def test_force_returns_vector(srp):

    force = srp._compute_solar_force()

    assert force.shape == (3,)


def test_force_is_finite(srp):

    force = srp._compute_solar_force()

    assert np.all(np.isfinite(force))


def test_force_opposes_sun_direction(srp):

    force = srp._compute_solar_force()

    assert force[0] < 0.0


def test_sun_direction_normalized():

    model = SolarRadiationPressure(

        solar_radiation_pressure=4.56e-6,

        reflectivity_coefficient=1.5,

        reference_area=0.4,

        center_of_pressure=np.zeros(3),

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
        np.array([1.0,0.0,0.0]),
        atol=ATOL_FORCE,
    )


# ==========================================================
# Compute
# ==========================================================

def test_compute_returns_vector(srp):

    torque = srp.compute(
        IDENTITY3,
    )

    assert torque.shape == (3,)


def test_compute_returns_finite_values(srp):

    torque = srp.compute(
        IDENTITY3,
    )

    assert np.all(np.isfinite(torque))


def test_zero_center_of_pressure_returns_zero_torque():

    model = SolarRadiationPressure(

        solar_radiation_pressure=4.56e-6,

        reflectivity_coefficient=1.5,

        reference_area=0.4,

        center_of_pressure=np.zeros(3),

        sun_direction_eci=np.array(
            [
                1.0,
                0.0,
                0.0,
            ]
        ),
    )

    torque = model.compute(
        IDENTITY3,
    )

    assert np.allclose(
        torque,
        ZERO_VECTOR3,
        atol=ATOL_TORQUE,
    )


# ==========================================================
# Input Validation
# ==========================================================

def test_invalid_dcm(srp):

    with pytest.raises(ValueError):

        srp.compute(
            np.eye(2),
        )