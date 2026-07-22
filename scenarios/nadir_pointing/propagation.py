"""
scenarios/nadir_pointing/propagation.py

Mission propagation routines.

Responsible for propagating the orbital and attitude dynamics.

This module contains NO controller logic.
"""

from __future__ import annotations

import numpy as np


# ==========================================================
# Public Interface
# ==========================================================

def propagate(scenario) -> None:
    """
    Execute one propagation step.

    Parameters
    ----------
    scenario : NadirPointingScenario
    """

    propagate_orbit(scenario)

    propagate_spacecraft(scenario)


# ==========================================================
# Orbit
# ==========================================================

def propagate_orbit(scenario) -> None:
    """
    Propagate spacecraft orbit.
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
    Return current orbital state.
    """

    return (

        scenario.position_eci,

        scenario.velocity_eci,

    )


def attitude_state(scenario):
    """
    Return spacecraft attitude.
    """

    return (

        scenario.sim.spacecraft.q,

        scenario.sim.spacecraft.omega,

    )


def body_to_eci_dcm(scenario):
    """
    Return body-to-ECI DCM.

    Temporary implementation.

    Replace later with

        spacecraft.body_to_eci_dcm
    """

    q0, q1, q2, q3 = scenario.sim.spacecraft.q

    return np.array(

        [

            [

                1 - 2*(q2*q2 + q3*q3),

                2*(q1*q2 - q0*q3),

                2*(q1*q3 + q0*q2),

            ],

            [

                2*(q1*q2 + q0*q3),

                1 - 2*(q1*q1 + q3*q3),

                2*(q2*q3 - q0*q1),

            ],

            [

                2*(q1*q3 - q0*q2),

                2*(q2*q3 + q0*q1),

                1 - 2*(q1*q1 + q2*q2),

            ],

        ]

    )


def eci_to_body_dcm(scenario):
    """
    Return ECI-to-body DCM.
    """

    return body_to_eci_dcm(scenario).T