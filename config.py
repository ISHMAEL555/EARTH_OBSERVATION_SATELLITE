"""
config.py

Global configuration file for the Earth Observation Satellite ADCS Simulation.

This file serves as the single source of truth for all spacecraft,
orbit, actuator, controller, environment, and simulation parameters.
"""

import numpy as np

# ==========================================================
# Spacecraft Inertia Properties
# ==========================================================

J = np.array([
    [9.833,    -0.06692,  -0.05295],
    [-0.06692, 14.11,     -0.002176],
    [-0.05295, -0.002176, 16.01],
])  # kg·m²

# ==========================================================
# Orbital Parameters
# ==========================================================

ALTITUDE_KM = 600.0                     # km

MU = 3.986004418e14                     # m³/s²
R_EARTH = 6378137.0                     # m

a = R_EARTH + ALTITUDE_KM * 1000.0      # Semi-major axis [m]
n = np.sqrt(MU / a**3)                  # Mean motion [rad/s]

ORBIT_PERIOD = 2.0 * np.pi / n          # seconds

# ==========================================================
# Environment Models
# ==========================================================

USE_IGRF = True
INCLUDE_AERO = True
INCLUDE_SRP = True

# ----------------------------------------------------------
# Earth's Magnetic Dipole Model
# ----------------------------------------------------------
# Centered dipole approximation.
# Dipole aligned with +Z axis in ECI.
# Units: A·m²

EARTH_MAGNETIC_DIPOLE_MOMENT = np.array([
    0.0,
    0.0,
    7.94e22,
])

# ==========================================================
# Reaction Wheels
# ==========================================================

H_RW_MAX = 1.4                  # N·m·s
TAU_RW_MAX = 0.475              # N·m

# ==========================================================
# Variable Speed CMGs
# ==========================================================

VSCMG_TAU_MAX = np.array([
    0.031,
    0.031,
    0.041,
])  # N·m

VSCMG_H_MAX = np.array([
    0.049,
    0.049,
    0.098,
])  # N·m·s

# ==========================================================
# Magnetorquers
# ==========================================================

M_MAX = 18.28                   # A·m²

# ==========================================================
# Singularity Avoidance
# ==========================================================

D_THRESHOLD = 0.20

SR_DAMPING = 0.02

NULL_MOTION_GAIN = 0.5

SINGULARITY_MEASURE = "cond"

# ==========================================================
# Controller Gains
# ==========================================================

KP = np.diag([
    12.0,
    15.0,
    18.0,
])

KD = np.diag([
    40.0,
    48.0,
    55.0,
])

KI = np.diag([
    0.5,
    0.5,
    0.5,
])

# ==========================================================
# Momentum Dumping
# ==========================================================

K_DUMP = 1.5e4

H_DUMP_THRESHOLD = 0.7 * H_RW_MAX * 4

# ==========================================================
# Simulation
# ==========================================================

DT = 0.05                        # seconds

SIM_TIME_SLEW = 400.0            # seconds

SIM_TIME_NADIR = (
    5.0 * ORBIT_PERIOD + 100.0
)

# ==========================================================
# Robustness Analysis
# ==========================================================

INERTIA_UNCERTAINTY = 0.10        # ±10%

MISALIGNMENT_DEG = 5.0            # degrees

# ==========================================================
# Random Seed
# ==========================================================

SEED = 42

np.random.seed(SEED)

# ==========================================================
# Output
# ==========================================================

SAVE_PLOTS = True

SHOW_PLOTS = False