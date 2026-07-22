"""
test_magnetorquers.py

Unit tests for the Three-Axis Magnetorquer Assembly.
"""

import numpy as np
import pytest

from models.actuators.magnetorquers import Magnetorquers


# ==========================================================
# TC-MTQ-001
# ==========================================================

def test_initialization():
    """Verify actuator initializes to zero state."""

    mtq = Magnetorquers()

    assert np.allclose(mtq.commanded_dipole, np.zeros(3))
    assert np.allclose(mtq.actual_dipole, np.zeros(3))
    assert np.allclose(mtq.body_torque, np.zeros(3))


# ==========================================================
# TC-MTQ-002
# ==========================================================

def test_command_storage():
    """Verify commanded dipole is stored correctly."""

    mtq = Magnetorquers()

    command = np.array([1.0, 2.0, 3.0])

    magnetic_field = np.array([
        2.0e-5,
        -1.5e-5,
        3.5e-5,
    ])

    mtq.update(command, magnetic_field, dt=0.1)

    assert np.allclose(mtq.commanded_dipole, command)


# ==========================================================
# TC-MTQ-003
# ==========================================================

def test_positive_saturation():
    """Verify positive saturation."""

    mtq = Magnetorquers()

    command = np.array([100.0, 100.0, 100.0])

    magnetic_field = np.array([
        2e-5,
        1e-5,
        3e-5,
    ])

    mtq.update(command, magnetic_field, dt=0.1)

    expected = np.full(3, mtq.max_dipole)

    assert np.allclose(mtq.actual_dipole, expected)


# ==========================================================
# TC-MTQ-004
# ==========================================================

def test_negative_saturation():
    """Verify negative saturation."""

    mtq = Magnetorquers()

    command = np.array([-100.0, -100.0, -100.0])

    magnetic_field = np.array([
        2e-5,
        1e-5,
        3e-5,
    ])

    mtq.update(command, magnetic_field, dt=0.1)

    expected = np.full(3, -mtq.max_dipole)

    assert np.allclose(mtq.actual_dipole, expected)


# ==========================================================
# TC-MTQ-005
# ==========================================================

def test_zero_magnetic_field():
    """Verify zero magnetic field produces zero torque."""

    mtq = Magnetorquers()

    command = np.array([5.0, 2.0, -1.0])

    magnetic_field = np.zeros(3)

    mtq.update(command, magnetic_field, dt=0.1)

    assert np.allclose(mtq.body_torque, np.zeros(3))


# ==========================================================
# TC-MTQ-006
# ==========================================================

def test_zero_command():
    """Verify zero dipole command produces zero torque."""

    mtq = Magnetorquers()

    command = np.zeros(3)

    magnetic_field = np.array([
        2e-5,
        -1e-5,
        3e-5,
    ])

    mtq.update(command, magnetic_field, dt=0.1)

    assert np.allclose(mtq.body_torque, np.zeros(3))


# ==========================================================
# TC-MTQ-007
# ==========================================================

def test_parallel_vectors():
    """Verify parallel vectors produce zero torque."""

    mtq = Magnetorquers()

    command = np.array([1.0, 0.0, 0.0])

    magnetic_field = np.array([2.0, 0.0, 0.0])

    mtq.update(command, magnetic_field, dt=0.1)

    assert np.allclose(mtq.body_torque, np.zeros(3))


# ==========================================================
# TC-MTQ-008
# ==========================================================

def test_cross_product():
    """Verify orthogonal vectors."""

    mtq = Magnetorquers()

    command = np.array([1.0, 0.0, 0.0])

    magnetic_field = np.array([0.0, 1.0, 0.0])

    mtq.update(command, magnetic_field, dt=0.1)

    expected = np.array([0.0, 0.0, 1.0])

    assert np.allclose(mtq.body_torque, expected)


# ==========================================================
# TC-MTQ-009
# ==========================================================

def test_right_hand_rule():
    """Verify right-hand rule."""

    mtq = Magnetorquers()

    command = np.array([0.0, 1.0, 0.0])

    magnetic_field = np.array([0.0, 0.0, 1.0])

    mtq.update(command, magnetic_field, dt=0.1)

    expected = np.array([1.0, 0.0, 0.0])

    assert np.allclose(mtq.body_torque, expected)


# ==========================================================
# TC-MTQ-010
# ==========================================================

def test_invalid_command_vector():
    """Verify invalid command vector."""

    mtq = Magnetorquers()

    magnetic_field = np.zeros(3)

    with pytest.raises(ValueError):

        mtq.update(
            np.array([1.0, 2.0]),
            magnetic_field,
            dt=0.1,
        )


# ==========================================================
# TC-MTQ-011
# ==========================================================

def test_invalid_magnetic_field():
    """Verify invalid magnetic field."""

    mtq = Magnetorquers()

    with pytest.raises(ValueError):

        mtq.update(
            np.zeros(3),
            np.array([1.0, 2.0]),
            dt=0.1,
        )


# ==========================================================
# TC-MTQ-012
# ==========================================================

def test_nan_input():
    """Verify NaN input."""

    mtq = Magnetorquers()

    command = np.array([np.nan, 0.0, 0.0])

    magnetic_field = np.zeros(3)

    with pytest.raises(ValueError):

        mtq.update(
            command,
            magnetic_field,
            dt=0.1,
        )


# ==========================================================
# TC-MTQ-013
# ==========================================================

def test_reset():
    """Verify reset restores zero state."""

    mtq = Magnetorquers()

    command = np.array([1.0, 2.0, 3.0])

    magnetic_field = np.array([
        2e-5,
        -1e-5,
        3e-5,
    ])

    mtq.update(command, magnetic_field, dt=0.1)

    mtq.reset()

    assert np.allclose(mtq.commanded_dipole, np.zeros(3))
    assert np.allclose(mtq.actual_dipole, np.zeros(3))
    assert np.allclose(mtq.body_torque, np.zeros(3))