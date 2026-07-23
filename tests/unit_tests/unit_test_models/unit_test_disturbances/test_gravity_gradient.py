"""
Unit tests for models/disturbances/gravity_gradient.py
"""

import numpy as np
import pytest

from models.disturbances.gravity_gradient import (
    GravityGradient,
)

from models.disturbances.disturbance_state import (
    DisturbanceState,
)

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
def state():
    """Nominal disturbance state."""

    return DisturbanceState(

        time=0.0,

        position_eci=np.array(
            [
                ORBIT_RADIUS,
                0.0,
                0.0,
            ]
        ),

        velocity_eci=np.array(
            [
                0.0,
                7550.0,
                0.0,
            ]
        ),

        orbit_radius=ORBIT_RADIUS,

        radial_unit_vector_eci=np.array(
            [
                1.0,
                0.0,
                0.0,
            ]
        ),

        body_to_eci_dcm=IDENTITY3,

        inertia_matrix=INERTIA_MATRIX,

        magnetic_field_eci=np.zeros(3),

        magnetic_field_body=np.zeros(3),

        atmospheric_density=0.0,

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

def test_constructor(gravity_gradient):

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
    state,
):

    torque = gravity_gradient.compute(state)

    assert torque.shape == (3,)


def test_compute_returns_finite_values(
    gravity_gradient,
    state,
):

    torque = gravity_gradient.compute(state)

    assert np.all(
        np.isfinite(torque)
    )


def test_zero_torque_for_principal_axis(
    gravity_gradient,
    state,
):

    state.radial_unit_vector_eci = np.array(
        [
            1.0,
            0.0,
            0.0,
        ]
    )

    state.inertia_matrix = np.diag(
        [
            10.0,
            12.0,
            15.0,
        ]
    )

    torque = gravity_gradient.compute(state)

    assert np.allclose(
        torque,
        ZERO_VECTOR3,
        atol=ATOL_TORQUE,
    )


def test_nonzero_torque_off_principal_axis(
    gravity_gradient,
    state,
):

    state.radial_unit_vector_eci = np.array(
        [
            1.0,
            1.0,
            0.0,
        ]
    )

    torque = gravity_gradient.compute(state)

    assert np.linalg.norm(
        torque
    ) > 0.0


def test_radial_vector_normalization(
    gravity_gradient,
    state,
):

    torque1 = gravity_gradient.compute(state)

    state.radial_unit_vector_eci = np.array(
        [
            100.0,
            0.0,
            0.0,
        ]
    )

    torque2 = gravity_gradient.compute(state)

    assert np.allclose(
        torque1,
        torque2,
        atol=ATOL_TORQUE,
    )


def test_identity_dcm_preserves_result(
    gravity_gradient,
    state,
):

    state.radial_unit_vector_eci = np.array(
        [
            1.0,
            1.0,
            0.0,
        ]
    )

    torque1 = gravity_gradient.compute(state)

    state.body_to_eci_dcm = np.eye(3)

    torque2 = gravity_gradient.compute(state)

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
    state,
):

    state.body_to_eci_dcm = np.eye(2)

    with pytest.raises(ValueError):
        gravity_gradient.compute(state)


def test_invalid_inertia_matrix(
    gravity_gradient,
    state,
):

    state.inertia_matrix = np.eye(2)

    with pytest.raises(ValueError):
        gravity_gradient.compute(state)


def test_invalid_radial_vector(
    gravity_gradient,
    state,
):

    state.radial_unit_vector_eci = np.ones(2)

    with pytest.raises(ValueError):
        gravity_gradient.compute(state)


def test_zero_radial_vector(
    gravity_gradient,
    state,
):

    state.radial_unit_vector_eci = np.zeros(3)

    with pytest.raises(ValueError):
        gravity_gradient.compute(state)


def test_zero_orbit_radius(
    gravity_gradient,
    state,
):

    state.orbit_radius = 0.0

    with pytest.raises(ValueError):
        gravity_gradient.compute(state)


def test_negative_orbit_radius(
    gravity_gradient,
    state,
):

    state.orbit_radius = -7000e3

    with pytest.raises(ValueError):
        gravity_gradient.compute(state)


def test_input_state_not_modified(
    gravity_gradient,
    state,
):
    """GravityGradient should not modify DisturbanceState."""

    original = state.radial_unit_vector_eci.copy()

    gravity_gradient.compute(state)

    assert np.array_equal(
        state.radial_unit_vector_eci,
        original,
    )