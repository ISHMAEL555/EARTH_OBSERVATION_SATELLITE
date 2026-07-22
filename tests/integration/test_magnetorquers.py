"""
Integration Test

Module
------
Magnetorquers

Objective
---------
Verify that the Magnetorquers model integrates correctly
with commanded dipole moments and the spacecraft magnetic
field to produce valid control torque commands.

This test validates subsystem interfaces only.
"""

import numpy as np

from models.actuators.magnetorquers import Magnetorquers


def create_magnetorquers():
    """
    Construct a nominal three-axis magnetorquer assembly.
    """

    return Magnetorquers(
        max_dipole=0.25,
    )


def test_magnetorquers_initialization():
    """
    Verify magnetorquers initialize correctly.
    """

    mtq = create_magnetorquers()

    assert mtq.max_dipole.shape == (3,)
    assert np.all(mtq.max_dipole > 0.0)


def test_zero_dipole():
    """
    Verify zero commanded dipole produces zero control torque.
    """

    mtq = create_magnetorquers()

    actual_dipole, body_torque = mtq.compute(
        commanded_dipole=np.zeros(3),
        magnetic_field_body=np.array(
            [2.0e-5, -1.5e-5, 3.0e-5]
        ),
    )

    assert actual_dipole.shape == (3,)
    assert body_torque.shape == (3,)

    assert np.allclose(actual_dipole, 0.0)
    assert np.allclose(body_torque, 0.0)


def test_magnetorquer_output():
    """
    Verify actuator generates valid magnetic torque.
    """

    mtq = create_magnetorquers()

    actual_dipole, body_torque = mtq.compute(
        commanded_dipole=np.array(
            [0.10, -0.05, 0.08]
        ),
        magnetic_field_body=np.array(
            [2.5e-5, -1.0e-5, 3.5e-5]
        ),
    )

    assert actual_dipole.shape == (3,)
    assert body_torque.shape == (3,)

    assert np.all(np.isfinite(actual_dipole))
    assert np.all(np.isfinite(body_torque))


def test_multiple_evaluations():
    """
    Verify repeated actuator evaluations.
    """

    mtq = create_magnetorquers()

    magnetic_field = np.array(
        [2.5e-5, -1.0e-5, 3.5e-5]
    )

    for _ in range(50):

        actual_dipole, body_torque = mtq.compute(
            commanded_dipole=np.array(
                [0.05, -0.02, 0.01]
            ),
            magnetic_field_body=magnetic_field,
        )

        assert np.all(np.isfinite(actual_dipole))
        assert np.all(np.isfinite(body_torque))


def test_controller_interface():
    """
    Verify compatibility with controller-generated
    magnetic dipole commands.
    """

    mtq = create_magnetorquers()

    controller_dipole = np.array(
        [0.08, -0.04, 0.02]
    )

    magnetic_field = np.array(
        [2.0e-5, -1.5e-5, 3.0e-5]
    )

    actual_dipole, body_torque = mtq.compute(
        commanded_dipole=controller_dipole,
        magnetic_field_body=magnetic_field,
    )

    assert actual_dipole.shape == (3,)
    assert body_torque.shape == (3,)

    assert np.all(np.isfinite(actual_dipole))
    assert np.all(np.isfinite(body_torque))


def test_environment_interface():
    """
    Verify compatibility with magnetic field values
    produced by the environment model.
    """

    mtq = create_magnetorquers()

    body_field = np.array(
        [1.8e-5, 2.4e-5, -3.2e-5]
    )

    actual_dipole, body_torque = mtq.compute(
        commanded_dipole=np.array(
            [0.05, 0.04, -0.02]
        ),
        magnetic_field_body=body_field,
    )

    assert actual_dipole.shape == (3,)
    assert body_torque.shape == (3,)

    assert np.all(np.isfinite(actual_dipole))
    assert np.all(np.isfinite(body_torque))