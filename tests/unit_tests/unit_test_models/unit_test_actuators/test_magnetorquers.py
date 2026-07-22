"""
Unit tests for models/actuators/magnetorquers.py
"""

import numpy as np
import pytest

from models.actuators.magnetorquers import Magnetorquers


MAX_DIPOLE = 5.0


# ==========================================================
# Fixtures
# ==========================================================

@pytest.fixture
def magnetorquers():
    """Create a nominal magnetorquer model."""

    return Magnetorquers(
        max_dipole=MAX_DIPOLE,
    )


# ==========================================================
# Initialization
# ==========================================================

def test_constructor(magnetorquers):
    """Verify constructor stores parameters."""

    assert magnetorquers.max_dipole == pytest.approx(
        MAX_DIPOLE
    )


def test_invalid_max_dipole():
    """Maximum dipole must be positive."""

    with pytest.raises(ValueError):

        Magnetorquers(
            max_dipole=0.0,
        )


# ==========================================================
# Compute
# ==========================================================

def test_compute_returns_vectors(magnetorquers):
    """Compute should return two 3-vectors."""

    actual_dipole, torque = magnetorquers.compute(
        np.array([1.0, 2.0, 3.0]),
        np.array([2e-5, -1e-5, 3e-5]),
    )

    assert actual_dipole.shape == (3,)
    assert torque.shape == (3,)


def test_compute_returns_finite_values(magnetorquers):
    """Outputs should contain finite values."""

    actual_dipole, torque = magnetorquers.compute(
        np.array([1.0, 2.0, 3.0]),
        np.array([2e-5, -1e-5, 3e-5]),
    )

    assert np.all(np.isfinite(actual_dipole))
    assert np.all(np.isfinite(torque))


# ==========================================================
# Saturation
# ==========================================================

def test_positive_saturation(magnetorquers):
    """Positive dipole saturation."""

    actual_dipole, _ = magnetorquers.compute(
        np.array([100.0, 100.0, 100.0]),
        np.array([2e-5, 1e-5, 3e-5]),
    )

    expected = np.full(3, MAX_DIPOLE)

    assert np.allclose(
        actual_dipole,
        expected,
    )


def test_negative_saturation(magnetorquers):
    """Negative dipole saturation."""

    actual_dipole, _ = magnetorquers.compute(
        np.array([-100.0, -100.0, -100.0]),
        np.array([2e-5, 1e-5, 3e-5]),
    )

    expected = np.full(3, -MAX_DIPOLE)

    assert np.allclose(
        actual_dipole,
        expected,
    )


def test_no_saturation(magnetorquers):
    """Commands within limits should pass unchanged."""

    command = np.array([1.0, -2.0, 3.0])

    actual_dipole, _ = magnetorquers.compute(
        command,
        np.array([2e-5, 1e-5, 3e-5]),
    )

    assert np.allclose(
        actual_dipole,
        command,
    )


# ==========================================================
# Physics
# ==========================================================

def test_zero_command(magnetorquers):
    """Zero dipole should produce zero torque."""

    _, torque = magnetorquers.compute(
        np.zeros(3),
        np.array([2e-5, -1e-5, 3e-5]),
    )

    assert np.allclose(
        torque,
        np.zeros(3),
    )


def test_zero_magnetic_field(magnetorquers):
    """Zero magnetic field should produce zero torque."""

    _, torque = magnetorquers.compute(
        np.array([1.0, 2.0, 3.0]),
        np.zeros(3),
    )

    assert np.allclose(
        torque,
        np.zeros(3),
    )


def test_parallel_vectors(magnetorquers):
    """Parallel vectors should produce zero torque."""

    _, torque = magnetorquers.compute(
        np.array([1.0, 0.0, 0.0]),
        np.array([2.0, 0.0, 0.0]),
    )

    assert np.allclose(
        torque,
        np.zeros(3),
    )


def test_cross_product(magnetorquers):
    """Verify cross product."""

    _, torque = magnetorquers.compute(
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    )

    expected = np.array(
        [
            0.0,
            0.0,
            1.0,
        ]
    )

    assert np.allclose(
        torque,
        expected,
    )


def test_right_hand_rule(magnetorquers):
    """Verify right-hand rule."""

    _, torque = magnetorquers.compute(
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    )

    expected = np.array(
        [
            1.0,
            0.0,
            0.0,
        ]
    )

    assert np.allclose(
        torque,
        expected,
    )


# ==========================================================
# Input Validation
# ==========================================================

def test_invalid_command_vector(magnetorquers):
    """Invalid commanded dipole."""

    with pytest.raises(ValueError):

        magnetorquers.compute(
            np.array([1.0, 2.0]),
            np.zeros(3),
        )


def test_invalid_magnetic_field(magnetorquers):
    """Invalid magnetic field."""

    with pytest.raises(ValueError):

        magnetorquers.compute(
            np.zeros(3),
            np.array([1.0, 2.0]),
        )


def test_nan_command(magnetorquers):
    """NaN commanded dipole."""

    with pytest.raises(ValueError):

        magnetorquers.compute(
            np.array([np.nan, 0.0, 0.0]),
            np.zeros(3),
        )


def test_nan_magnetic_field(magnetorquers):
    """NaN magnetic field."""

    with pytest.raises(ValueError):

        magnetorquers.compute(
            np.zeros(3),
            np.array([0.0, np.nan, 0.0]),
        )