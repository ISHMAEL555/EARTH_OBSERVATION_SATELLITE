"""
Integration Test

Module
------
Mission Guidance Wrapper

Objective
---------
Verify that the mission guidance wrapper correctly
interfaces with the guidance subsystem and updates
the scenario reference states.

This test validates subsystem interfaces only.
"""

import numpy as np

from scenarios.nadir_pointing.guidance import (
    update_guidance,
    reference_quaternion,
    reference_body_rates,
)


class DummyGuidance:
    """
    Dummy guidance model.
    """

    def compute(
        self,
        position_eci,
        velocity_eci,
    ):

        return (
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.zeros(3),
        )


class DummyScenario:

    def __init__(self):

        self.guidance = DummyGuidance()

        self.position_eci = np.array(
            [
                7000e3,
                0.0,
                0.0,
            ]
        )

        self.velocity_eci = np.array(
            [
                0.0,
                7500.0,
                0.0,
            ]
        )

        self.q_ref = None

        self.omega_ref = None


def test_guidance_update():
    """
    Verify guidance updates the reference states.
    """

    scenario = DummyScenario()

    update_guidance(scenario)

    assert scenario.q_ref.shape == (4,)
    assert scenario.omega_ref.shape == (3,)

    assert np.all(np.isfinite(scenario.q_ref))
    assert np.all(np.isfinite(scenario.omega_ref))


def test_reference_quaternion():
    """
    Verify reference quaternion interface.
    """

    scenario = DummyScenario()

    update_guidance(scenario)

    q_ref = reference_quaternion(
        scenario
    )

    assert q_ref.shape == (4,)
    assert np.all(np.isfinite(q_ref))


def test_reference_body_rates():
    """
    Verify reference body-rate interface.
    """

    scenario = DummyScenario()

    update_guidance(scenario)

    omega_ref = reference_body_rates(
        scenario
    )

    assert omega_ref.shape == (3,)
    assert np.all(np.isfinite(omega_ref))


def test_multiple_guidance_updates():
    """
    Verify repeated guidance evaluations.
    """

    scenario = DummyScenario()

    for _ in range(20):

        update_guidance(scenario)

        assert scenario.q_ref.shape == (4,)
        assert scenario.omega_ref.shape == (3,)

        assert np.all(np.isfinite(scenario.q_ref))
        assert np.all(np.isfinite(scenario.omega_ref))