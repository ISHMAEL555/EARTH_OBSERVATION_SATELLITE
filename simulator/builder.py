"""
simulator/builder.py

Constructs the complete spacecraft simulation.

The builder receives configuration dictionaries and creates all
simulation objects.
"""

import numpy as np

from simulator.simulation import Simulation

from models.environment.orbit import Orbit
from models.environment.magnetic_field import MagneticField

from models.spacecraft import Spacecraft

from models.disturbances.disturbances import Disturbances
from models.disturbances.gravity_gradient import GravityGradient
from models.disturbances.atmospheric_drag import AtmosphericDrag
from models.disturbances.solar_radiation_pressure import (
    SolarRadiationPressure,
)

from models.actuators.rw_pyramid import RWPyramid
from models.actuators.magnetorquers import Magnetorquers

from controllers.quaternion_pd import QuaternionPD
from controllers.lqr import LQR


class Builder:

    """
    Builds a complete Simulation object.
    """

    def __init__(

        self,

        mission: dict,

        controller: dict,

        initial_conditions: dict,

    ):

        self.mission = mission

        self.controller_cfg = controller

        self.initial = initial_conditions

    # ==========================================================
    # Public
    # ==========================================================

    def build(self) -> Simulation:

        orbit = self.build_orbit()

        magnetic_field = self.build_magnetic_field()

        spacecraft = self.build_spacecraft()

        disturbances = self.build_disturbances()

        reaction_wheels = self.build_reaction_wheels()

        magnetorquers = self.build_magnetorquers()

        controller = self.build_controller(spacecraft)

        return Simulation(

            simulation_time=self.mission["simulation_time"],

            time_step=self.mission["time_step"],

            orbit=orbit,

            magnetic_field=magnetic_field,

            spacecraft=spacecraft,

            disturbances=disturbances,

            reaction_wheels=reaction_wheels,

            magnetorquers=magnetorquers,

            controller=controller,

        )

    # ==========================================================
    # Environment
    # ==========================================================

    def build_orbit(self):

        cfg = self.mission["orbit"]

        return Orbit(

            mu=cfg["mu"],

            semi_major_axis=cfg["semi_major_axis"],

            eccentricity=cfg["eccentricity"],

            inclination=cfg["inclination"],

            raan=cfg["raan"],

            argument_of_perigee=cfg["argument_of_perigee"],

            true_anomaly=cfg["true_anomaly"],

        )

    def build_magnetic_field(self):

        return MagneticField(

            magnetic_dipole_moment=np.asarray(

                self.mission["earth"]["magnetic_dipole"]

            )

        )

    # ==========================================================
    # Spacecraft
    # ==========================================================

    def build_spacecraft(self):

        cfg = self.mission["spacecraft"]

        return Spacecraft(

            inertia=np.asarray(cfg["inertia"]),

            mass=cfg["mass"],

            q0=np.asarray(self.initial["quaternion"]),

            omega0=np.asarray(self.initial["angular_velocity"]),

        )

        # ==========================================================
    # Disturbances
    # ==========================================================

    def build_disturbances(self):

        disturbances = Disturbances()

        cfg = self.mission["disturbances"]

        # ------------------------------------------------------
        # Gravity Gradient
        # ------------------------------------------------------

        disturbances.add(

            GravityGradient(

                gravitational_parameter=cfg["gravity_gradient"]["mu"]

            )

        )

        # ------------------------------------------------------
        # Atmospheric Drag
        # ------------------------------------------------------

        disturbances.add(

            AtmosphericDrag(

                drag_coefficient=cfg["drag"]["cd"],

                reference_area=cfg["drag"]["area"],

            )

        )

        # ------------------------------------------------------
        # Solar Radiation Pressure
        # ------------------------------------------------------

        disturbances.add(

            SolarRadiationPressure(

                solar_radiation_pressure=cfg["srp"]["pressure"],

                reflectivity_coefficient=cfg["srp"]["cr"],

                reference_area=cfg["srp"]["area"],

            )

        )

        return disturbances
    
    # ==========================================================
    # Actuators
    # ==========================================================

    def build_reaction_wheels(self):

        cfg = self.mission["reaction_wheels"]

        return RWPyramid(

            wheel_axes=np.asarray(cfg["axes"]),

            max_torque=cfg["max_torque"],

            max_momentum=cfg["max_momentum"],

        )

    def build_magnetorquers(self):

        cfg = self.mission["magnetorquers"]

        return Magnetorquers(

            max_dipole=cfg["max_dipole"]

        )

    # ==========================================================
    # Controller
    # ==========================================================

    def build_controller(self, spacecraft):

        controller_type = self.controller_cfg["type"].lower()

        if controller_type == "pd":

            return QuaternionPD(

                proportional_gain=np.asarray(

                    self.controller_cfg["kp"]

                ),

                derivative_gain=np.asarray(

                    self.controller_cfg["kd"]

                ),

            )

        if controller_type == "lqr":

            return LQR(

                inertia_matrix=spacecraft.inertia,

                state_weight=np.asarray(

                    self.controller_cfg["Q"]

                ),

                control_weight=np.asarray(

                    self.controller_cfg["R"]

                ),

            )

        raise ValueError(

            f"Unsupported controller '{controller_type}'."

        )