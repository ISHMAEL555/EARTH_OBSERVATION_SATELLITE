"""
Unit tests for models/disturbances/atmospheric_drag.py
"""

import numpy as np
import pytest

from models.disturbances.atmospheric_drag import AtmosphericDrag

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
def density():
    """Representative atmospheric density."""

    return 1.0e-12


@pytest.fixture
def spacecraft_velocity():
    """Representative spacecraft velocity."""

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

def test_constructor(atmospheric_drag):
    """Verify constructor stores parameters."""

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
    density,
    spacecraft_velocity,
):
    """Compute should return a 3-vector."""

    force = atmospheric_drag.compute(
        density,
        spacecraft_velocity,
    )

    assert force.shape == (3,)


def test_compute_returns_finite_values(
    atmospheric_drag,
    density,
    spacecraft_velocity,
):
    """Computed drag force should contain finite values."""

    force = atmospheric_drag.compute(
        density,
        spacecraft_velocity,
    )

    assert np.all(np.isfinite(force))


def test_zero_velocity_returns_zero_force(
    atmospheric_drag,
    density,
):
    """Zero velocity should produce zero drag."""

    force = atmospheric_drag.compute(
        density,
        np.zeros(3),
    )

    assert np.allclose(
        force,
        ZERO_VECTOR3,
        atol=ATOL_FORCE,
    )


def test_drag_force_opposes_velocity(
    atmospheric_drag,
    density,
):
    """Drag must oppose the velocity vector."""

    velocity = np.array(
        [
            7500.0,
            0.0,
            0.0,
        ]
    )

    force = atmospheric_drag.compute(
        density,
        velocity,
    )

    assert force[0] < 0.0


def test_drag_force_increases_with_velocity(
    atmospheric_drag,
    density,
):
    """Drag magnitude should increase with speed."""

    force1 = atmospheric_drag.compute(
        density,
        np.array(
            [
                7000.0,
                0.0,
                0.0,
            ]
        ),
    )

    force2 = atmospheric_drag.compute(
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


def test_drag_force_scales_with_density(
    atmospheric_drag,
    spacecraft_velocity,
):
    """Drag magnitude should increase with density."""

    force1 = atmospheric_drag.compute(
        1.0e-13,
        spacecraft_velocity,
    )

    force2 = atmospheric_drag.compute(
        2.0e-13,
        spacecraft_velocity,
    )

    assert np.linalg.norm(force2) == pytest.approx(
        2.0 * np.linalg.norm(force1),
        rel=RTOL_DEFAULT,
    )


# ==========================================================
# Input Validation
# ==========================================================

def test_negative_density(
    atmospheric_drag,
    spacecraft_velocity,
):
    """Negative density should raise ValueError."""

    with pytest.raises(ValueError):

        atmospheric_drag.compute(
            -1.0,
            spacecraft_velocity,
        )


def test_invalid_velocity_shape(
    atmospheric_drag,
    density,
):
    """Velocity must have shape (3,)."""

    with pytest.raises(ValueError):

        atmospheric_drag.compute(
            density,
            np.zeros(2),
        )


def test_invalid_velocity_type(
    atmospheric_drag,
    density,
):
    """Velocity must be array-like with shape (3,)."""

    with pytest.raises(ValueError):

        atmospheric_drag.compute(
            density,
            np.eye(3),
        )