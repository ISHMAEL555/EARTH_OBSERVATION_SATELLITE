"""
Unit tests for models/disturbances/disturbances.py
"""

import numpy as np
import pytest

from models.disturbances.disturbances import Disturbances


# ==========================================================
# Dummy Disturbance Models
# ==========================================================

class DummyDisturbance:
    """
    Dummy disturbance model returning a constant vector.
    """

    def __init__(self, output):

        self.output = np.asarray(
            output,
            dtype=float,
        )

    def compute(self, **kwargs):

        return self.output


class DummyCounterDisturbance:
    """
    Counts the number of compute() calls.
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
    """Manager should start empty."""

    assert len(
        disturbances._disturbance_models
    ) == 0


# ==========================================================
# Registration
# ==========================================================

def test_add_disturbance(disturbances):
    """Verify disturbance registration."""

    model = DummyDisturbance(
        [
            1.0,
            0.0,
            0.0,
        ]
    )

    disturbances.add(model)

    assert len(
        disturbances._disturbance_models
    ) == 1


def test_add_invalid_disturbance(disturbances):
    """Missing compute() should raise TypeError."""

    with pytest.raises(TypeError):

        disturbances.add(
            InvalidDisturbance()
        )


# ==========================================================
# Remove
# ==========================================================

def test_remove_disturbance(disturbances):
    """Verify removal."""

    model = DummyDisturbance(
        [
            1.0,
            0.0,
            0.0,
        ]
    )

    disturbances.add(model)

    disturbances.remove(model)

    assert len(
        disturbances._disturbance_models
    ) == 0


def test_remove_nonexistent_disturbance(disturbances):
    """Removing an unknown model should raise ValueError."""

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
    """Verify clear()."""

    disturbances.add(
        DummyDisturbance(
            [1.0, 0.0, 0.0]
        )
    )

    disturbances.add(
        DummyDisturbance(
            [0.0, 1.0, 0.0]
        )
    )

    disturbances.clear()

    assert len(
        disturbances._disturbance_models
    ) == 0


# ==========================================================
# Compute
# ==========================================================

def test_compute_empty_manager(disturbances):
    """Empty manager should return an empty list."""

    outputs = disturbances.compute()

    assert outputs == []


def test_compute_single_disturbance(disturbances):
    """Single disturbance should return one output."""

    disturbances.add(

        DummyDisturbance(
            [
                1.0,
                2.0,
                3.0,
            ]
        )

    )

    outputs = disturbances.compute()

    assert len(outputs) == 1

    assert np.allclose(

        outputs[0],

        np.array(
            [
                1.0,
                2.0,
                3.0,
            ]
        ),

    )


def test_compute_multiple_disturbances(disturbances):
    """Multiple disturbances should return outputs in registration order."""

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

    outputs = disturbances.compute()

    assert len(outputs) == 2

    assert np.allclose(
        outputs[0],
        np.array(
            [
                1.0,
                2.0,
                3.0,
            ]
        ),
    )

    assert np.allclose(
        outputs[1],
        np.array(
            [
                4.0,
                5.0,
                6.0,
            ]
        ),
    )


def test_compute_calls_each_model_once(disturbances):
    """Each disturbance model should be evaluated exactly once."""

    model1 = DummyCounterDisturbance()
    model2 = DummyCounterDisturbance()

    disturbances.add(model1)
    disturbances.add(model2)

    disturbances.compute()

    assert model1.calls == 1
    assert model2.calls == 1


def test_compute_returns_numpy_arrays(disturbances):
    """Outputs should be NumPy arrays."""

    disturbances.add(
        DummyDisturbance(
            [
                1.0,
                2.0,
                3.0,
            ]
        )
    )

    outputs = disturbances.compute()

    assert isinstance(
        outputs[0],
        np.ndarray,
    )


def test_compute_returns_finite_values(disturbances):
    """Outputs should contain finite values."""

    disturbances.add(
        DummyDisturbance(
            [
                1.0,
                2.0,
                3.0,
            ]
        )
    )

    outputs = disturbances.compute()

    assert np.all(
        np.isfinite(outputs[0])
    )