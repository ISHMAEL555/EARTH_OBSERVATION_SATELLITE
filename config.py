"""
config.py

Global configuration file for the Earth Observation Satellite
Attitude Determination and Control System (ADCS) Simulation.

This module serves as the single source of truth for all mission,
spacecraft, environment, actuator, controller, and simulation
parameters used throughout the project.

Notes
-----
- This file stores only configuration parameters.
- No subsystem physics should be implemented here.
- Derived quantities should be computed inside their respective models.
"""

import numpy as np

# ==========================================================
# Mission
# ==========================================================

MISSION = {

    "name": "Advanced Earth Observation Satellite",

    "mission_duration": 5.0,      # years

}

# ==========================================================
# Spacecraft
# ==========================================================

SPACECRAFT = {

    "mass": None,                 # kg

    "inertia": np.array([
        [9.833,    -0.06692,  -0.05295],
        [-0.06692, 14.11,     -0.002176],
        [-0.05295, -0.002176, 16.01],
    ]),

}

# ==========================================================
# Orbit
# ==========================================================

ORBIT = {

    # Classical Orbital Elements

    "altitude": 600e3,                    # m

    "inclination": np.deg2rad(98.0),      # rad

    "eccentricity": 0.0,

    "raan": 0.0,                          # rad

    "argument_of_perigee": 0.0,           # rad

    "true_anomaly": 0.0,                  # rad

    # Earth Constants

    "earth_radius": 6378137.0,            # m

    "gravitational_parameter": 3.986004418e14,

}

# ==========================================================
# Environment
# ==========================================================

ENVIRONMENT = {

    "use_igrf": True,

    "include_aerodynamics": True,

    "include_solar_radiation_pressure": True,

    "earth_magnetic_dipole": np.array([
        0.0,
        0.0,
        7.94e22,
    ]),

}

# ==========================================================
# Reaction Wheel Geometry
# ==========================================================

RW_SKEW_ANGLE = np.deg2rad(45.0)

c = np.cos(RW_SKEW_ANGLE)
s = np.sin(RW_SKEW_ANGLE)

# ==========================================================
# Actuators
# ==========================================================

ACTUATORS = {

    # ------------------------------------------------------
    # Four-Wheel Pyramidal Reaction Wheel Assembly
    # ------------------------------------------------------

    "reaction_wheels": {

        "configuration": "pyramid",

        "num_wheels": 4,

        "skew_angle": RW_SKEW_ANGLE,

        "wheel_axes": np.array([
            [ c,  0.0,  s],
            [0.0,  c,  s],
            [-c, 0.0,  s],
            [0.0, -c,  s]
        ]).T,

        "max_torque": 0.475,      # N·m per wheel

        "max_momentum": 1.40,     # N·m·s per wheel

    },

    # ------------------------------------------------------
    # Three-Axis Magnetorquer Assembly
    # ------------------------------------------------------

    "magnetorquers": {

        "num_rods": 3,

        "axes": np.eye(3),

        "max_dipole": 18.28,      # A·m²

    },

}

# ==========================================================
# Controllers
# ==========================================================

CONTROLLERS = {

    "pd": {

        "Kp": np.diag([
            12.0,
            15.0,
            18.0,
        ]),

        "Kd": np.diag([
            40.0,
            48.0,
            55.0,
        ]),

        "Ki": np.diag([
            0.5,
            0.5,
            0.5,
        ]),

    },

    "momentum_dumping": {

        "gain": 1.5e4,

        "threshold": 0.7,

    },


    "lqr": {

        "Q": np.diag([
            100.0,
            100.0,
            100.0,
            10.0,
            10.0,
            10.0
        ]),

        "R": np.diag([
            1.0,
            1.0,
            1.0
        ])
    }

}

# ==========================================================
# Simulation
# ==========================================================

SIMULATION = {

    "time_step": 0.05,

    "slew_duration": 400.0,

    "nadir_orbits": 5,

    "nadir_settling_time": 100.0,

}

# ==========================================================
# Robustness Analysis
# ==========================================================

ROBUSTNESS = {

    "inertia_uncertainty": 0.10,

    "misalignment_deg": 5.0,

}

# ==========================================================
# Output
# ==========================================================

OUTPUT = {

    "save_plots": True,

    "show_plots": False,

}

# ==========================================================
# Randomization
# ==========================================================

RANDOM = {

    "seed": 42,

}