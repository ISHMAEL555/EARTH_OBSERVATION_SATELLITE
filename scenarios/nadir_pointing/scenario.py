"""
scenarios/nadir_pointing/scenario.py

Nadir Pointing Mission Scenario

This module orchestrates the execution of a nadir-pointing
Earth observation mission.

Mission Flow
------------
Initialization
    ↓
Propagation
    ↓
Environment
    ↓
Guidance
    ↓
Control
    ↓
Telemetry
"""

from __future__ import annotations

from .initialization import initialize
from .propagation import propagate
from .environment import update_environment
from .guidance import update_guidance
from .control import update_control
from .telemetry import log, finalize


class NadirPointingScenario:
    """
    Nadir Pointing Mission Scenario.
    """

    def __init__(
        self,
        simulation,
        simulator,
        guidance,
    ):
        """
        Parameters
        ----------
        simulation : Simulation
            Complete simulation object.

        simulator : Simulator
            Simulation time manager.

        guidance : GuidanceLaw
            Mission guidance law.
        """

        self.sim = simulation

        self.simulator = simulator

        self.guidance = guidance

        # ==================================================
        # Mission Variables
        # ==================================================

        self.position_eci = None

        self.velocity_eci = None

        self.q_ref = None

        self.omega_ref = None

        self.magnetic_field_eci = None

        self.magnetic_field_body = None

        self.atmospheric_density = 0.0

        self.sun_vector_eci = None

        self.disturbance_torque = None

        self.control_torque = None

        self.total_torque = None

        self.actual_rw_torque = None

        self.rw_body_torque = None

        self.wheel_momentum = None

        self.actual_dipole = None

        self.magnetorquer_torque = None

        # ==================================================
        # Telemetry
        # ==================================================

        self.telemetry = {}

    # ======================================================
    # Mission Lifecycle
    # ======================================================

    def initialize(self):
        """
        Initialize mission.
        """

        initialize(self)

    def update(self):
        """
        Execute one simulation step.
        """

        # ------------------------------------------
        # Dynamics
        # ------------------------------------------

        propagate(self)

        # ------------------------------------------
        # Sensor Models
        # ------------------------------------------

        self.update_sensors()

        # ------------------------------------------
        # State Estimation
        # ------------------------------------------

        self.update_estimator()

        # ------------------------------------------
        # Environment
        # ------------------------------------------

        update_environment(self)

        # ------------------------------------------
        # Guidance
        # ------------------------------------------

        update_guidance(self)

        # ------------------------------------------
        # Controller + Actuators
        # ------------------------------------------

        update_control(self)

        # ------------------------------------------
        # Telemetry
        # ------------------------------------------

        log(self)

        # ------------------------------------------
        # Advance Simulation Clock
        # ------------------------------------------

        self.simulator.step()

    def run(self):
        """
        Run complete mission.
        """

        self.initialize()

        while not self.simulator.finished:

            self.update()

        return self.finalize()

    def finalize(self):
        """
        Finalize simulation.
        """

        return finalize(self)

    # ======================================================
    # Placeholder Interfaces
    # ======================================================

    def update_sensors(self):
        """
        Sensor models.

        Will later include

        - Gyroscope
        - Magnetometer
        - Earth Sensor
        - Sun Sensor
        - Star Tracker
        """

        pass

    def update_estimator(self):
        """
        Navigation filter.

        Will later include

        - EKF
        - MEKF
        - UKF
        """

        pass