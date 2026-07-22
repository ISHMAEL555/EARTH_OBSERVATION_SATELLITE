"""
scenarios/nadir_pointing/control.py

Attitude Control Pipeline

Responsibilities
----------------
1. Controller
2. Reaction Wheels
3. Magnetorquers
4. Total Applied Torque
"""

from __future__ import annotations

import numpy as np

from models.disturbances.disturbance_state import DisturbanceState


# ==========================================================
# Public Interface
# ==========================================================

def update_control(scenario):
    """
    Execute the complete ADCS control pipeline.
    """

    update_environment(scenario)

    update_disturbances(scenario)

    update_controller(scenario)

    update_reaction_wheels(scenario)

    update_magnetorquers(scenario)

    compute_total_torque(scenario)


# ==========================================================
# Environment
# ==========================================================

def update_environment(scenario):
    """
    Evaluate environmental models.
    """

    scenario.magnetic_field_eci = (

        scenario.sim.magnetic_field.compute(

            scenario.position_eci

        )

    )

    C_bi = scenario.sim.spacecraft.body_to_eci_dcm()

    scenario.magnetic_field_body = (

        C_bi.T

        @ scenario.magnetic_field_eci

    )


# ==========================================================
# Disturbances
# ==========================================================

def update_disturbances(scenario):
    """
    Compute disturbance torque.
    """

    state = DisturbanceState(

        time=scenario.simulator.time,

        position_eci=scenario.position_eci,

        velocity_eci=scenario.velocity_eci,

        body_to_eci_dcm=

            scenario.sim.spacecraft.body_to_eci_dcm(),

        inertia_matrix=

            scenario.sim.spacecraft.inertia,

        magnetic_field_eci=

            scenario.magnetic_field_eci,

        magnetic_field_body=

            scenario.magnetic_field_body,

        atmospheric_density=0.0,

        solar_vector_eci=np.zeros(3),

    )

    scenario.disturbance_torque = (

        scenario.sim.disturbances.compute(

            state

        )

    )


# ==========================================================
# Controller
# ==========================================================

def update_controller(scenario):
    """
    Compute desired body torque.
    """

    scenario.control_torque = (

        scenario.sim.controller.compute(

            current_quaternion=

                scenario.sim.spacecraft.q,

            desired_quaternion=

                scenario.q_ref,

            body_rates=

                scenario.sim.spacecraft.omega,

            desired_body_rates=

                scenario.omega_ref,

        )

    )


# ==========================================================
# Reaction Wheels
# ==========================================================

def update_reaction_wheels(scenario):
    """
    Execute reaction wheel model.
    """

    (

        scenario.actual_rw_torque,

        scenario.wheel_momentum,

        scenario.rw_body_torque,

    ) = (

        scenario.sim.reaction_wheels.update(

            commanded_body_torque=

                scenario.control_torque,

            dt=scenario.sim.time_step,

        )

    )


# ==========================================================
# Magnetorquers
# ==========================================================

def update_magnetorquers(scenario):
    """
    Execute magnetorquer model.

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
# Total Torque
# ==========================================================

def compute_total_torque(scenario):
    """
    Sum all body torques.
    """

    scenario.total_torque = (

        scenario.rw_body_torque

        +

        scenario.magnetorquer_torque

        +

        scenario.disturbance_torque

    )