"""
simulator/simulation.py

Simulation container.

This module defines the complete spacecraft simulation state used by
all mission scenarios.

The Simulation object does not perform any computations itself.
It simply groups together every model required by a scenario.
"""

from dataclasses import dataclass
from typing import Any

from models.environment.orbit import Orbit
from models.environment.magnetic_field import MagneticField

from models.spacecraft import Spacecraft

from models.disturbances.disturbances import Disturbances

from models.actuators.rw_pyramid import RWPyramid
from models.actuators.magnetorquers import Magnetorquers


@dataclass(slots=True)
class Simulation:
    """
    Complete simulation container.

    Every scenario receives one Simulation object.
    """

    # ==========================================================
    # Mission
    # ==========================================================

    simulation_time: float
    time_step: float

    # ==========================================================
    # Environment
    # ==========================================================

    orbit: Orbit

    magnetic_field: MagneticField

    # ==========================================================
    # Vehicle
    # ==========================================================

    spacecraft: Spacecraft

    # ==========================================================
    # Disturbances
    # ==========================================================

    disturbances: Disturbances

    # ==========================================================
    # Actuators
    # ==========================================================

    reaction_wheels: RWPyramid

    magnetorquers: Magnetorquers

    # ==========================================================
    # Controller
    # ==========================================================

    controller: Any