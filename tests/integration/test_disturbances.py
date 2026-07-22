"""
Integration Test

Module
------
Disturbances

Objective
---------
Verify that the Disturbances manager integrates correctly
with the environment and disturbance models.

This test validates interface compatibility only.
"""

import numpy as np

from models.disturbances.disturbances import Disturbances
from models.environment.orbit import Orbit
from models.environment.magnetic_field import MagneticField


class DummyDisturbance:
    """
    Simple disturbance model used for integration testing.
    """

    def compute(self, **kwargs):

        position = kwargs["spacecraft_position_eci"]

        return 1e-6 * position


def test_disturbance_manager_initialization():
    """
    Verify disturbance manager initializes correctly.
    """

    disturbances = Disturbances()

    assert len(disturbances._disturbance_models) == 0


def test_disturbance_registration():
    """
    Verify disturbance models can be registered.
    """

    disturbances = Disturbances()

    model = DummyDisturbance()

    disturbances.add(model)

    assert len(disturbances._disturbance_models) == 1


def test_disturbance_evaluation():
    """
    Verify registered disturbance models are evaluated.
    """

    disturbances = Disturbances()

    disturbances.add(DummyDisturbance())

    outputs = disturbances.compute(
        spacecraft_position_eci=np.array(
            [7000e3, 0.0, 0.0]
        )
    )

    assert isinstance(outputs, list)

    assert len(outputs) == 1

    assert outputs[0].shape == (3,)

    assert np.all(np.isfinite(outputs[0]))


def test_disturbance_chain():
    """
    Verify Orbit -> Environment -> Disturbances chain.
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

    disturbances = Disturbances()

    disturbances.add(DummyDisturbance())

    for t in [0.0, 60.0, 120.0, 300.0]:

        position_eci, velocity_eci = orbit.propagate(t)

        B_eci = magnetic_field.compute(position_eci)

        outputs = disturbances.compute(
            spacecraft_position_eci=position_eci,
            magnetic_field_eci=B_eci,
            spacecraft_velocity_eci=velocity_eci,
        )

        assert len(outputs) == 1

        assert outputs[0].shape == (3,)

        assert np.all(np.isfinite(outputs[0]))


def test_multiple_disturbance_models():
    """
    Verify multiple disturbance models are evaluated.
    """

    disturbances = Disturbances()

    disturbances.add(DummyDisturbance())
    disturbances.add(DummyDisturbance())
    disturbances.add(DummyDisturbance())

    outputs = disturbances.compute(
        spacecraft_position_eci=np.array(
            [7000e3, 0.0, 0.0]
        )
    )

    assert len(outputs) == 3

    for disturbance in outputs:

        assert disturbance.shape == (3,)

        assert np.all(np.isfinite(disturbance))