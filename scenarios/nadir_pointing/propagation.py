"""
scenarios/nadir_pointing/propagation.py

Mission propagation routines.

Responsible for propagating the orbital and attitude dynamics.

This module contains NO controller logic.
"""

from __future__ import annotations


# ==========================================================
# Public Interface
# ==========================================================

def propagate(scenario) -> None:
    """
    Execute one propagation step.
    """

    propagate_orbit(scenario)
    propagate_spacecraft(scenario)


# ==========================================================
# Orbit
# ==========================================================

def propagate_orbit(scenario) -> None:
    """
    Propagate the spacecraft orbit.
    """

    (
        scenario.position_eci,
        scenario.velocity_eci,
    ) = scenario.sim.orbit.propagate(
        scenario.simulator.time
    )


# ==========================================================
# Spacecraft
# ==========================================================

def propagate_spacecraft(scenario) -> None:
    """
    Propagate spacecraft attitude dynamics.
    """

    scenario.sim.spacecraft.propagate(
        total_torque=scenario.total_torque,
        dt=scenario.sim.time_step,
    )


# ==========================================================
# Convenience Functions
# ==========================================================

def orbital_state(scenario):
    """
    Return the current orbital state.
    """

    return (
        scenario.position_eci,
        scenario.velocity_eci,
    )


def attitude_state(scenario):
    """
    Return the current spacecraft attitude state.
    """

    spacecraft = scenario.sim.spacecraft

    return (
        spacecraft.quaternion,
        spacecraft.angular_velocity,
    )


def body_to_eci_dcm(scenario):
    """
    Return the body-to-ECI Direction Cosine Matrix.
    """

    return scenario.sim.spacecraft.body_to_eci_dcm


def eci_to_body_dcm(scenario):
    """
    Return the ECI-to-body Direction Cosine Matrix.
    """

    return scenario.sim.spacecraft.eci_to_body_dcm