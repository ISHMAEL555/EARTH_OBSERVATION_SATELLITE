"""
Unit tests for models/disturbances/disturbances.py
"""

import numpy as np
import pytest

from models.disturbances.disturbances import Disturbances

from tests.test_config.constants import (
    ZERO_VECTOR3,
)

from tests.test_config.tolerances import (
    ATOL_TORQUE,
)


# ==========================================================
# Dummy Disturbance Models
# ==========================================================

class DummyDisturbance:
    """
    Dummy disturbance model returning a constant torque.
    """

    def __init__(self, torque):

        self.torque = np.asarray(
            torque,
            dtype=float,
        )

    def compute(self, **kwargs):

        return self.torque


class DummyCounterDisturbance:
    """
    Dummy disturbance model counting compute() calls.
    """

    def __init__(self):

        self.calls = 0

    def compute(self, **kwargs):

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


# ==========================================================
# Fixtures
# ==========================================================

@pytest.fixture
def disturbances():

    return Disturbances()


# ==========================================================
# Initialization
# ==========================================================

def test_default_initialization(disturbances):
    """
    Manager should start empty.
    """

    assert len(disturbances._disturbance_models) == 0


# ==========================================================
# Add
# ==========================================================

def test_add_disturbance(disturbances):
    """
    Verify disturbance registration.
    """

    model = DummyDisturbance(
        [
            1.0,
            0.0,
            0.0,
        ]
    )

    disturbances.add(model)

    assert len(disturbances._disturbance_models) == 1


def test_add_invalid_disturbance(disturbances):
    """
    Missing compute() should raise TypeError.
    """

    with pytest.raises(TypeError):

        disturbances.add(
            InvalidDisturbance()
        )


# ==========================================================
# Remove
# ==========================================================

def test_remove_disturbance(disturbances):
    """
    Verify removal.
    """

    model = DummyDisturbance(
        [
            1.0,
            0.0,
            0.0,
        ]
    )

    disturbances.add(model)

    disturbances.remove(model)

    assert len(disturbances._disturbance_models) == 0


def test_remove_nonexistent_disturbance(disturbances):
    """
    Removing an unregistered disturbance should raise ValueError.
    """

    model = DummyDisturbance(
        [
            1.0,
            0.0,
            0.0,
        ]
    )

    with pytest.raises(ValueError):

        disturbances.remove(model)


# ==========================================================
# Clear
# ==========================================================

def test_clear_disturbances(disturbances):
    """
    Verify clear().
    """

    disturbances.add(
        DummyDisturbance(
            [
                1.0,
                0.0,
                0.0,
            ]
        )
    )

    disturbances.add(
        DummyDisturbance(
            [
                0.0,
                1.0,
                0.0,
            ]
        )
    )

    disturbances.clear()

    assert len(disturbances._disturbance_models) == 0


# ==========================================================
# Compute
# ==========================================================

def test_compute_empty_manager(disturbances):
    """
    Empty manager should return zero torque.
    """

    torque = disturbances.compute()

    assert np.allclose(
        torque,
        ZERO_VECTOR3,
        atol=ATOL_TORQUE,
    )


def test_compute_single_disturbance(disturbances):
    """
    Single disturbance should be returned unchanged.
    """

    disturbances.add(

        DummyDisturbance(
            [
                1.0,
                2.0,
                3.0,
            ]
        )
    )

    torque = disturbances.compute()

    assert np.allclose(

        torque,

        np.array(
            [
                1.0,
                2.0,
                3.0,
            ]
        ),

        atol=ATOL_TORQUE,
    )


def test_compute_multiple_disturbances(disturbances):
    """
    Total torque should equal the sum of all disturbances.
    """

    disturbances.add(
        DummyDisturbance(
            [
                1.0,
                2.0,
                3.0,
            ]
        )
    )

    disturbances.add(
        DummyDisturbance(
            [
                4.0,
                5.0,
                6.0,
            ]
        )
    )

    torque = disturbances.compute()

    expected = np.array(
        [
            5.0,
            7.0,
            9.0,
        ]
    )

    assert np.allclose(
        torque,
        expected,
        atol=ATOL_TORQUE,
    )


def test_compute_calls_each_model_once(disturbances):
    """
    Each registered disturbance should be evaluated exactly once.
    """

    model1 = DummyCounterDisturbance()
    model2 = DummyCounterDisturbance()

    disturbances.add(model1)
    disturbances.add(model2)

    disturbances.compute()

    assert model1.calls == 1
    assert model2.calls == 1


def test_compute_returns_vector(disturbances):
    """
    Returned disturbance should be a 3-vector.
    """

    disturbances.add(
        DummyDisturbance(
            [
                1.0,
                2.0,
                3.0,
            ]
        )
    )

    torque = disturbances.compute()

    assert torque.shape == (3,)


def test_compute_returns_finite_values(disturbances):
    """
    Returned disturbance should contain finite values.
    """

    disturbances.add(
        DummyDisturbance(
            [
                1.0,
                2.0,
                3.0,
            ]
        )
    )

    torque = disturbances.compute()

    assert np.all(np.isfinite(torque))