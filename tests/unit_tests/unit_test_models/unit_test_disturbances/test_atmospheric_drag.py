"""
Unit tests for models/disturbances/atmospheric_drag.py
"""

import numpy as np
import pytest

from models.disturbances.atmospheric_drag import (
    AtmosphericDrag,
)

from models.disturbances.disturbance_state import (
    DisturbanceState,
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
def atmospheric_drag():
    """Create a nominal atmospheric drag model."""

    return AtmosphericDrag(
        drag_coefficient=2.2,
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
                7550.0,
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

        magnetic_field_eci=np.zeros(3),

        magnetic_field_body=np.zeros(3),

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

def test_constructor(atmospheric_drag):

    assert atmospheric_drag.drag_coefficient == pytest.approx(
        2.2
    )

    assert atmospheric_drag.reference_area == pytest.approx(
        0.4
    )


def test_invalid_drag_coefficient():

    with pytest.raises(ValueError):

        AtmosphericDrag(
            drag_coefficient=-1.0,
            reference_area=0.4,
        )


def test_invalid_reference_area():

    with pytest.raises(ValueError):

        AtmosphericDrag(
            drag_coefficient=2.2,
            reference_area=-1.0,
        )
# ==========================================================
# Drag Force
# ==========================================================

def test_compute_returns_vector(
    atmospheric_drag,
    state,
):

    force = atmospheric_drag.compute(
        state
    )

    assert force.shape == (3,)


def test_compute_returns_finite_values(
    atmospheric_drag,
    state,
):

    force = atmospheric_drag.compute(
        state
    )

    assert np.all(
        np.isfinite(force)
    )


def test_zero_velocity_returns_zero_force(
    atmospheric_drag,
    state,
):

    state.velocity_eci = np.zeros(3)

    force = atmospheric_drag.compute(
        state
    )

    assert np.allclose(
        force,
        ZERO_VECTOR3,
        atol=ATOL_FORCE,
    )


def test_drag_force_opposes_velocity(
    atmospheric_drag,
    state,
):

    state.velocity_eci = np.array(
        [
            7500.0,
            0.0,
            0.0,
        ]
    )

    force = atmospheric_drag.compute(
        state
    )

    assert force[0] < 0.0


def test_drag_force_increases_with_velocity(
    atmospheric_drag,
    state,
):

    state.velocity_eci = np.array(
        [
            7000.0,
            0.0,
            0.0,
        ]
    )

    force1 = atmospheric_drag.compute(
        state
    )

    state.velocity_eci = np.array(
        [
            8000.0,
            0.0,
            0.0,
        ]
    )

    force2 = atmospheric_drag.compute(
        state
    )

    assert np.linalg.norm(
        force2
    ) > np.linalg.norm(
        force1
    )


def test_drag_force_scales_with_density(
    atmospheric_drag,
    state,
):

    state.atmospheric_density = 1.0e-13

    force1 = atmospheric_drag.compute(
        state
    )

    state.atmospheric_density = 2.0e-13

    force2 = atmospheric_drag.compute(
        state
    )

    assert np.linalg.norm(
        force2
    ) == pytest.approx(
        2.0 * np.linalg.norm(force1),
        rel=RTOL_DEFAULT,
    )


# ==========================================================
# Input Validation
# ==========================================================

def test_negative_density(
    atmospheric_drag,
    state,
):

    state.atmospheric_density = -1.0

    with pytest.raises(ValueError):
        atmospheric_drag.compute(state)


def test_invalid_velocity_shape(
    atmospheric_drag,
    state,
):

    state.velocity_eci = np.zeros(2)

    with pytest.raises(ValueError):
        atmospheric_drag.compute(state)


def test_invalid_velocity_type(
    atmospheric_drag,
    state,
):

    state.velocity_eci = np.eye(3)

    with pytest.raises(ValueError):
        atmospheric_drag.compute(state)


def test_input_state_not_modified(
    atmospheric_drag,
    state,
):
    """AtmosphericDrag should not modify DisturbanceState."""

    original = state.velocity_eci.copy()

    atmospheric_drag.compute(state)

    assert np.array_equal(
        state.velocity_eci,
        original,
    )