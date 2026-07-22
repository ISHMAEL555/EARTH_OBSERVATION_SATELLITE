"""
Integration Test

Module:
    Simulation

Objective
---------
Verify that the Simulation container correctly stores all
subsystem models and mission parameters.

This test does NOT validate any algorithms.
It only verifies the integration interfaces.
"""

import numpy as np

from simulator.simulation import Simulation

from models.environment.orbit import Orbit
from models.environment.magnetic_field import MagneticField

from models.spacecraft import Spacecraft

from models.disturbances.disturbances import Disturbances

from models.actuators.rw_pyramid import RWPyramid
from models.actuators.magnetorquers import Magnetorquers

from controllers.quaternion_pd import QuaternionPD


def test_simulation_initialization():
    """
    Verify the Simulation object is correctly assembled.
    """

    # ==========================================================
    # Environment
    # ==========================================================

    orbit = Orbit(
        mu=3.986004418e14,
        semi_major_axis=6878e3,
    )

    magnetic_field = MagneticField(
        magnetic_dipole_moment=np.array(
            [0.0, 0.0, 7.94e22]
        )
    )

    # ==========================================================
    # Spacecraft
    # ==========================================================

    spacecraft = Spacecraft(
        inertia=np.diag([20.0, 25.0, 30.0]),
        mass=100.0,
    )

    # ==========================================================
    # Disturbances
    # ==========================================================

    disturbances = Disturbances()

    # ==========================================================
    # Actuators
    # ==========================================================

    wheel_axes = np.array([
        [1.0, 0.0, 0.0, 0.577],
        [0.0, 1.0, 0.0, 0.577],
        [0.0, 0.0, 1.0, 0.577],
    ])

    reaction_wheels = RWPyramid(
        wheel_axes=wheel_axes,
        max_torque=0.05,
        max_momentum=5.0,
    )

    magnetorquers = Magnetorquers(
        max_dipole=np.array([10.0, 10.0, 10.0]),
    )

    # ==========================================================
    # Controller
    # ==========================================================

    controller = QuaternionPD(
        proportional_gain=np.eye(3),
        derivative_gain=np.eye(3),
    )

    # ==========================================================
    # Simulation
    # ==========================================================

    simulation = Simulation(
        simulation_time=600.0,
        time_step=0.1,
        orbit=orbit,
        magnetic_field=magnetic_field,
        spacecraft=spacecraft,
        disturbances=disturbances,
        reaction_wheels=reaction_wheels,
        magnetorquers=magnetorquers,
        controller=controller,
    )

    # ==========================================================
    # Verification
    # ==========================================================

    assert simulation.simulation_time == 600.0
    assert simulation.time_step == 0.1

    assert simulation.orbit is orbit
    assert simulation.magnetic_field is magnetic_field

    assert simulation.spacecraft is spacecraft

    assert simulation.disturbances is disturbances

    assert simulation.reaction_wheels is reaction_wheels
    assert simulation.magnetorquers is magnetorquers

    assert simulation.controller is controller