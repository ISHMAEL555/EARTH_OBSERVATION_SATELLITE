"""
Integration Test

Module
------
MagneticField

Objective
---------
Verify that the MagneticField model integrates correctly
with the Orbit model and provides a valid magnetic field.

This test validates interface compatibility only.
Magnetic field model accuracy is verified by unit tests.
"""

import numpy as np

from models.environment.orbit import Orbit
from models.environment.magnetic_field import MagneticField


def test_magnetic_field_initialization():
    """
    Verify magnetic field model initialization.
    """

    magnetic_field = MagneticField(
        magnetic_dipole_moment=np.array(
            [0.0, 0.0, 7.94e22]
        )
    )

    assert magnetic_field.magnetic_dipole_moment.shape == (3,)
    assert np.all(
        np.isfinite(
            magnetic_field.magnetic_dipole_moment
        )
    )


def test_magnetic_field_orbit_integration():
    """
    Verify the magnetic field model accepts the
    Orbit model output.
    """

    orbit = Orbit(
        mu=3.986004418e14,
        semi_major_axis=6878e3,
    )

    magnetic_field = MagneticField(
        magnetic_dipole_moment=np.array(
            [0.0, 0.0, 7.94e22]
        )
    )

    position_eci, _ = orbit.propagate(
        time=0.0,
    )

    B_eci = magnetic_field.compute(
        spacecraft_position_eci=position_eci,
    )

    assert isinstance(B_eci, np.ndarray)
    assert B_eci.shape == (3,)
    assert np.all(np.isfinite(B_eci))


def test_magnetic_field_multiple_calls():
    """
    Verify repeated magnetic field evaluations
    over a mission timeline.
    """

    orbit = Orbit(
        mu=3.986004418e14,
        semi_major_axis=6878e3,
    )

    magnetic_field = MagneticField(
        magnetic_dipole_moment=np.array(
            [0.0, 0.0, 7.94e22]
        )
    )

    for t in np.arange(0.0, 600.0, 10.0):

        position_eci, _ = orbit.propagate(
            time=t,
        )

        B_eci = magnetic_field.compute(
            spacecraft_position_eci=position_eci,
        )

        assert B_eci.shape == (3,)
        assert np.all(np.isfinite(B_eci))


def test_environment_chain():
    """
    Verify the Orbit -> MagneticField
    integration chain.
    """

    orbit = Orbit(
        mu=3.986004418e14,
        semi_major_axis=6878e3,
    )

    magnetic_field = MagneticField(
        magnetic_dipole_moment=np.array(
            [0.0, 0.0, 7.94e22]
        )
    )

    for time in [0.0, 60.0, 120.0, 300.0]:

        position_eci, velocity_eci = orbit.propagate(
            time=time,
        )

        B_eci = magnetic_field.compute(position_eci)

        assert position_eci.shape == (3,)
        assert velocity_eci.shape == (3,)
        assert B_eci.shape == (3,)

        assert np.all(np.isfinite(position_eci))
        assert np.all(np.isfinite(velocity_eci))
        assert np.all(np.isfinite(B_eci))