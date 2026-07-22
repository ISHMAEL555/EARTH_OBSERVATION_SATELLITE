"""
scenarios/nadir_pointing/initialization.py

Mission initialization utilities.

Responsible for preparing the simulation before the first
integration step.

This module performs NO dynamics or control computations.
"""

from __future__ import annotations

import numpy as np


def initialize(scenario) -> None:
    """
    Initialize the complete mission.

    Parameters
    ----------
    scenario : NadirPointingScenario
    """

    initialize_simulator(scenario)

    initialize_spacecraft(scenario)

    initialize_actuators(scenario)

    initialize_environment(scenario)

    initialize_guidance(scenario)

    initialize_logger(scenario)


# ==========================================================
# Simulator
# ==========================================================


def initialize_simulator(scenario) -> None:
    """
    Reset simulator time.
    """

    scenario.simulator.reset()


# ==========================================================
# Spacecraft
# ==========================================================


def initialize_spacecraft(scenario) -> None:
    """
    Reset spacecraft state.
    """

    scenario.sim.spacecraft.reset()

    scenario.position_eci = np.zeros(3)

    scenario.velocity_eci = np.zeros(3)

    scenario.q_ref = np.array(
        [1.0, 0.0, 0.0, 0.0]
    )

    scenario.omega_ref = np.zeros(3)


# ==========================================================
# Actuators
# ==========================================================


def initialize_actuators(scenario) -> None:
    """
    Reset actuator models.
    """

    if hasattr(scenario.sim, "reaction_wheels"):

        scenario.sim.reaction_wheels.reset()

    if hasattr(scenario.sim, "magnetorquers"):

        pass


# ==========================================================
# Environment
# ==========================================================


def initialize_environment(scenario) -> None:
    """
    Initialize environment variables.
    """

    scenario.magnetic_field_eci = np.zeros(3)

    scenario.magnetic_field_body = np.zeros(3)

    scenario.disturbance_torque = np.zeros(3)


# ==========================================================
# Guidance
# ==========================================================


def initialize_guidance(scenario) -> None:
    """
    Initialize guidance variables.
    """

    scenario.control_torque = np.zeros(3)

    scenario.total_torque = np.zeros(3)


# ==========================================================
# Telemetry
# ==========================================================


def initialize_logger(scenario) -> None:
    """
    Reset telemetry.
    """

    scenario.logger.reset()