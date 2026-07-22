"""
scenarios/nadir_pointing/environment.py

Mission environment update routines.

Responsible for

- Magnetic Field
- Atmosphere
- Sun
- Disturbances

No control logic belongs here.
"""

from __future__ import annotations

import numpy as np

from models.disturbances.disturbance_state import DisturbanceState


# ==========================================================
# Public Interface
# ==========================================================

def update_environment(scenario):
    """
    Update all environment models.
    """

    update_magnetic_field(scenario)

    update_atmosphere(scenario)

    update_sun(scenario)

    update_disturbances(scenario)


# ==========================================================
# Magnetic Field
# ==========================================================

def update_magnetic_field(scenario):

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
# Atmosphere
# ==========================================================

def update_atmosphere(scenario):
    """
    Placeholder atmosphere model.
    """

    scenario.atmospheric_density = 0.0


# ==========================================================
# Sun
# ==========================================================

def update_sun(scenario):
    """
    Placeholder sun model.
    """

    scenario.sun_vector_eci = np.zeros(3)


# ==========================================================
# Disturbances
# ==========================================================

def update_disturbances(scenario):
    """
    Update all disturbance models.
    """

    position_eci = scenario.position_eci

    orbit_radius = np.linalg.norm(position_eci)

    if orbit_radius < 1e-12:
        raise RuntimeError(
            "Invalid orbit radius."
        )

    radial_unit_vector_eci = (
        position_eci
        / orbit_radius
    )

    disturbance_state = DisturbanceState(

        time=scenario.simulator.time,

        # Orbit
        position_eci=position_eci,
        velocity_eci=scenario.velocity_eci,

        orbit_radius=orbit_radius,
        radial_unit_vector_eci=radial_unit_vector_eci,

        # Spacecraft
        body_to_eci_dcm=
            scenario.sim.spacecraft.body_to_eci_dcm(),

        inertia_matrix=
            scenario.sim.spacecraft.inertia,

        # Environment
        magnetic_field_eci=
            scenario.magnetic_field_eci,

        magnetic_field_body=
            scenario.magnetic_field_body,

        atmospheric_density=
            scenario.atmospheric_density,

        solar_vector_eci=
            scenario.sun_vector_eci,

    )

    scenario.disturbance_torque = (
        scenario.sim.disturbances.compute(
            disturbance_state
        )
    )