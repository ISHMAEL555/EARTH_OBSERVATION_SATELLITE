"""
scenarios/nadir_pointing/control.py

Attitude Control Pipeline

Responsibilities
----------------
1. Attitude Controller
2. Reaction Wheel Actuation
3. Magnetorquer Actuation
4. Total Applied Body Torque
"""

from __future__ import annotations

import numpy as np


# ==========================================================
# Public Interface
# ==========================================================

def update_control(scenario):
    """
    Execute the complete ADCS control pipeline.
    """

    update_controller(scenario)
    update_reaction_wheels(scenario)
    update_magnetorquers(scenario)
    compute_total_torque(scenario)


# ==========================================================
# Controller
# ==========================================================

def update_controller(scenario):
    """
    Compute the commanded spacecraft body torque.
    """

    spacecraft = scenario.sim.spacecraft

    scenario.control_torque = (

        scenario.sim.controller.compute(

            current_quaternion=spacecraft.quaternion,

            desired_quaternion=scenario.q_ref,

            body_rates=spacecraft.angular_velocity,

            desired_body_rates=scenario.omega_ref,

        )

    )


# ==========================================================
# Reaction Wheels
# ==========================================================

def update_reaction_wheels(scenario):
    """
    Execute the reaction wheel actuator model.
    """

    (
        scenario.actual_rw_torque,
        scenario.wheel_momentum,
        scenario.rw_body_torque,
    ) = (

        scenario.sim.reaction_wheels.update(

            commanded_body_torque=scenario.control_torque,

            dt=scenario.sim.time_step,

        )

    )


# ==========================================================
# Magnetorquers
# ==========================================================

def update_magnetorquers(scenario):
    """
    Execute the magnetorquer actuator model.

    Currently, the commanded magnetic dipole is zero.
    Momentum dumping logic will be added later.
    """

    commanded_dipole = np.zeros(3)

    (
        scenario.actual_dipole,
        scenario.magnetorquer_torque,
    ) = (

        scenario.sim.magnetorquers.compute(

            commanded_dipole,

            scenario.magnetic_field_body,

        )

    )


# ==========================================================
# Total Applied Torque
# ==========================================================

def compute_total_torque(scenario):
    """
    Compute the total torque acting on the spacecraft body.
    """

    scenario.total_torque = (

        scenario.rw_body_torque
        + scenario.magnetorquer_torque
        + scenario.disturbance_torque

    )