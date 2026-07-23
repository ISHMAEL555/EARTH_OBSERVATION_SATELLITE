"""
Unit tests for models/disturbances/disturbance_state.py
"""

import numpy as np
import pytest

from models.disturbances.disturbance_state import (
    DisturbanceState,
)


# ==========================================================
# Fixtures
# ==========================================================

@pytest.fixture
def disturbance_state():
    """Create a nominal DisturbanceState."""

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
                1.0e-5,
                -3.0e-5,
            ]
        ),

        magnetic_field_body=np.array(
            [
                2.0e-5,
                1.0e-5,
                -3.0e-5,
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
# Construction
# ==========================================================

def test_valid_construction(
    disturbance_state,
):
    """Verify a valid state can be created."""

    assert isinstance(
        disturbance_state,
        DisturbanceState,
    )


# ==========================================================
# Scalar Conversion
# ==========================================================

def test_scalar_conversion(
    disturbance_state,
):
    """Scalars should be stored as floats."""

    assert isinstance(
        disturbance_state.time,
        float,
    )

    assert isinstance(
        disturbance_state.orbit_radius,
        float,
    )

    assert isinstance(
        disturbance_state.atmospheric_density,
        float,
    )


# ==========================================================
# Array Conversion
# ==========================================================

def test_array_conversion():
    """Array-like inputs should become NumPy arrays."""

    state = DisturbanceState(

        time=0,

        position_eci=[1, 2, 3],

        velocity_eci=[4, 5, 6],

        orbit_radius=7000e3,

        radial_unit_vector_eci=[1, 0, 0],

        body_to_eci_dcm=np.eye(3),

        inertia_matrix=np.eye(3),

        magnetic_field_eci=[0, 0, 0],

        magnetic_field_body=[0, 0, 0],

        atmospheric_density=0,

        solar_vector_eci=[1, 0, 0],
    )

    assert isinstance(
        state.position_eci,
        np.ndarray,
    )

    assert isinstance(
        state.velocity_eci,
        np.ndarray,
    )

    assert isinstance(
        state.radial_unit_vector_eci,
        np.ndarray,
    )


# ==========================================================
# Vector Validation
# ==========================================================

def test_invalid_position_shape():

    with pytest.raises(ValueError):

        DisturbanceState(

            time=0,

            position_eci=np.zeros(2),

            velocity_eci=np.zeros(3),

            orbit_radius=1,

            radial_unit_vector_eci=np.zeros(3),

            body_to_eci_dcm=np.eye(3),

            inertia_matrix=np.eye(3),

            magnetic_field_eci=np.zeros(3),

            magnetic_field_body=np.zeros(3),

            atmospheric_density=0,

            solar_vector_eci=np.zeros(3),
        )


def test_invalid_velocity_shape():

    with pytest.raises(ValueError):

        DisturbanceState(

            time=0,

            position_eci=np.zeros(3),

            velocity_eci=np.zeros(2),

            orbit_radius=1,

            radial_unit_vector_eci=np.zeros(3),

            body_to_eci_dcm=np.eye(3),

            inertia_matrix=np.eye(3),

            magnetic_field_eci=np.zeros(3),

            magnetic_field_body=np.zeros(3),

            atmospheric_density=0,

            solar_vector_eci=np.zeros(3),
        )


def test_invalid_radial_vector_shape():

    with pytest.raises(ValueError):

        DisturbanceState(

            time=0,

            position_eci=np.zeros(3),

            velocity_eci=np.zeros(3),

            orbit_radius=1,

            radial_unit_vector_eci=np.zeros(2),

            body_to_eci_dcm=np.eye(3),

            inertia_matrix=np.eye(3),

            magnetic_field_eci=np.zeros(3),

            magnetic_field_body=np.zeros(3),

            atmospheric_density=0,

            solar_vector_eci=np.zeros(3),
        )


def test_invalid_magnetic_field_eci_shape():

    with pytest.raises(ValueError):

        DisturbanceState(

            time=0,

            position_eci=np.zeros(3),

            velocity_eci=np.zeros(3),

            orbit_radius=1,

            radial_unit_vector_eci=np.zeros(3),

            body_to_eci_dcm=np.eye(3),

            inertia_matrix=np.eye(3),

            magnetic_field_eci=np.zeros(2),

            magnetic_field_body=np.zeros(3),

            atmospheric_density=0,

            solar_vector_eci=np.zeros(3),
        )


def test_invalid_magnetic_field_body_shape():

    with pytest.raises(ValueError):

        DisturbanceState(

            time=0,

            position_eci=np.zeros(3),

            velocity_eci=np.zeros(3),

            orbit_radius=1,

            radial_unit_vector_eci=np.zeros(3),

            body_to_eci_dcm=np.eye(3),

            inertia_matrix=np.eye(3),

            magnetic_field_eci=np.zeros(3),

            magnetic_field_body=np.zeros(2),

            atmospheric_density=0,

            solar_vector_eci=np.zeros(3),
        )


def test_invalid_solar_vector_shape():

    with pytest.raises(ValueError):

        DisturbanceState(

            time=0,

            position_eci=np.zeros(3),

            velocity_eci=np.zeros(3),

            orbit_radius=1,

            radial_unit_vector_eci=np.zeros(3),

            body_to_eci_dcm=np.eye(3),

            inertia_matrix=np.eye(3),

            magnetic_field_eci=np.zeros(3),

            magnetic_field_body=np.zeros(3),

            atmospheric_density=0,

            solar_vector_eci=np.zeros(2),
        )


# ==========================================================
# Matrix Validation
# ==========================================================

def test_invalid_body_to_eci_dcm_shape():

    with pytest.raises(ValueError):

        DisturbanceState(

            time=0,

            position_eci=np.zeros(3),

            velocity_eci=np.zeros(3),

            orbit_radius=1,

            radial_unit_vector_eci=np.zeros(3),

            body_to_eci_dcm=np.eye(2),

            inertia_matrix=np.eye(3),

            magnetic_field_eci=np.zeros(3),

            magnetic_field_body=np.zeros(3),

            atmospheric_density=0,

            solar_vector_eci=np.zeros(3),
        )


def test_invalid_inertia_matrix_shape():

    with pytest.raises(ValueError):

        DisturbanceState(

            time=0,

            position_eci=np.zeros(3),

            velocity_eci=np.zeros(3),

            orbit_radius=1,

            radial_unit_vector_eci=np.zeros(3),

            body_to_eci_dcm=np.eye(3),

            inertia_matrix=np.eye(2),

            magnetic_field_eci=np.zeros(3),

            magnetic_field_body=np.zeros(3),

            atmospheric_density=0,

            solar_vector_eci=np.zeros(3),
        )