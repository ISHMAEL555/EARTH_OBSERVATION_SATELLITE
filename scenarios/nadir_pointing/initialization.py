"""
scenarios/nadir_pointing/initialization.py

Mission initialization utilities.

Responsible for preparing the simulation before the first
integration step.

This module performs NO dynamics or control computations.
"""

from __future__ import annotations

import numpy as np

from .telemetry import initialize_telemetry


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

    initialize_telemetry_storage(scenario)


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

        # Initialize wheel momentum from the model if available
        if hasattr(scenario.sim.reaction_wheels, "wheel_momentum"):

            scenario.wheel_momentum = (
                scenario.sim.reaction_wheels.wheel_momentum.copy()
            )

        else:

            # Default for a 4-wheel pyramid
            scenario.wheel_momentum = np.zeros(4)

    else:

        scenario.wheel_momentum = np.zeros(4)

    scenario.actual_rw_torque = np.zeros(3)

    scenario.rw_body_torque = np.zeros(3)

    scenario.actual_dipole = np.zeros(3)

    scenario.magnetorquer_torque = np.zeros(3)


# ==========================================================
# Environment
# ==========================================================

def initialize_environment(scenario) -> None:
    """
    Initialize environment variables.
    """

    scenario.magnetic_field_eci = np.zeros(3)

    scenario.magnetic_field_body = np.zeros(3)

    scenario.atmospheric_density = 0.0

    scenario.sun_vector_eci = np.zeros(3)

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

def initialize_telemetry_storage(scenario) -> None:
    """
    Initialize telemetry storage.
    """

    initialize_telemetry(scenario)