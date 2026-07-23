"""
Unit tests for models/disturbances/disturbance_manager.py
"""

import numpy as np
import pytest

from models.disturbances.disturbance_manager import (
    DisturbanceManager,
)

from models.disturbances.disturbance_state import (
    DisturbanceState,
)


# ==========================================================
# Dummy Disturbance Models
# ==========================================================

class DummyDisturbance:
    """
    Dummy disturbance model returning a constant torque.
    """

    def __init__(
        self,
        torque,
    ):

        self.torque = np.asarray(
            torque,
            dtype=float,
        )

    def compute(
        self,
        state,
    ):

        return self.torque


class DummyCounterDisturbance:
    """
    Counts the number of compute() calls.
    """

    def __init__(self):

        self.calls = 0

    def compute(
        self,
        state,
    ):

        self.calls += 1

        return np.array(
            [
                1.0,
                2.0,
                3.0,
            ]
        )


class InvalidDisturbance:
    """
    Does not implement compute().
    """

    pass


class NoneDisturbance:
    """
    Returns None.
    """

    def compute(
        self,
        state,
    ):

        return None


class InvalidShapeDisturbance:
    """
    Returns an invalid torque vector.
    """

    def compute(
        self,
        state,
    ):

        return np.zeros(
            (
                3,
                1,
            )
        )


# ==========================================================
# Fixtures
# ==========================================================

@pytest.fixture
def manager():

    return DisturbanceManager()


@pytest.fixture
def state():

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

def test_default_initialization(
    manager,
):
    """Manager should start empty."""

    assert len(manager) == 0

    assert manager.models == ()


# ==========================================================
# Registration
# ==========================================================

def test_add_model(
    manager,
):
    """Verify disturbance registration."""

    manager.add(

        DummyDisturbance(
            [
                1.0,
                0.0,
                0.0,
            ]
        )

    )

    assert len(manager) == 1


def test_add_invalid_model(
    manager,
):
    """Missing compute() should raise TypeError."""

    with pytest.raises(TypeError):

        manager.add(
            InvalidDisturbance()
        )


# ==========================================================
# Compute
# ==========================================================

def test_compute_empty_manager(
    manager,
    state,
):
    """Empty manager should return zero torque."""

    torque = manager.compute(
        state
    )

    assert np.allclose(
        torque,
        np.zeros(3),
    )


def test_compute_single_model(
    manager,
    state,
):
    """Single disturbance."""

    manager.add(

        DummyDisturbance(
            [
                1.0,
                2.0,
                3.0,
            ]
        )

    )

    torque = manager.compute(
        state
    )

    assert np.allclose(

        torque,

        np.array(
            [
                1.0,
                2.0,
                3.0,
            ]
        ),

    )


def test_compute_multiple_models(
    manager,
    state,
):
    """Multiple disturbances should sum."""

    manager.add(

        DummyDisturbance(
            [
                1.0,
                2.0,
                3.0,
            ]
        )

    )

    manager.add(

        DummyDisturbance(
            [
                4.0,
                5.0,
                6.0,
            ]
        )

    )

    torque = manager.compute(
        state
    )

    assert np.allclose(

        torque,

        np.array(
            [
                5.0,
                7.0,
                9.0,
            ]
        ),

    )


def test_compute_calls_each_model_once(
    manager,
    state,
):
    """Each disturbance model should be evaluated once."""

    model1 = DummyCounterDisturbance()
    model2 = DummyCounterDisturbance()

    manager.add(model1)
    manager.add(model2)

    manager.compute(state)

    assert model1.calls == 1
    assert model2.calls == 1


def test_compute_returns_numpy_array(
    manager,
    state,
):
    """Output should be a NumPy array."""

    manager.add(

        DummyDisturbance(
            [
                1.0,
                2.0,
                3.0,
            ]
        )

    )

    torque = manager.compute(
        state
    )

    assert isinstance(
        torque,
        np.ndarray,
    )


def test_compute_returns_finite_values(
    manager,
    state,
):
    """Output should contain finite values."""

    manager.add(

        DummyDisturbance(
            [
                1.0,
                2.0,
                3.0,
            ]
        )

    )

    torque = manager.compute(
        state
    )

    assert np.all(
        np.isfinite(
            torque
        )
    )


# ==========================================================
# Validation
# ==========================================================

def test_none_return_raises(
    manager,
    state,
):
    """Returning None should raise ValueError."""

    manager.add(
        NoneDisturbance()
    )

    with pytest.raises(ValueError):

        manager.compute(state)


def test_invalid_shape_raises(
    manager,
    state,
):
    """Returned torque must have shape (3,)."""

    manager.add(
        InvalidShapeDisturbance()
    )

    with pytest.raises(ValueError):

        manager.compute(state)


# ==========================================================
# Utilities
# ==========================================================

def test_clear(
    manager,
):
    """Verify clear()."""

    manager.add(

        DummyDisturbance(
            [
                1.0,
                0.0,
                0.0,
            ]
        )

    )

    manager.clear()

    assert len(manager) == 0


def test_models_property(
    manager,
):
    """models property should be read-only."""

    model = DummyDisturbance(
        [
            1.0,
            0.0,
            0.0,
        ]
    )

    manager.add(model)

    assert manager.models == (model,)


def test_iteration(
    manager,
):
    """Verify iteration over registered models."""

    model = DummyDisturbance(
        [
            1.0,
            0.0,
            0.0,
        ]
    )

    manager.add(model)

    assert list(manager) == [model]


def test_repr(
    manager,
):
    """Verify string representation."""

    assert repr(manager) == "DisturbanceManager(models=0)"