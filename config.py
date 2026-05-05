"""
ADCS Hybrid RW + VSCMG Configuration
Single Source of Truth - Strictly following assignment document
"""

import numpy as np

# ========================= SPACECRAFT INERTIA =========================
J = np.array([
    [9.833,    -0.06692,  -0.05295],
    [-0.06692, 14.11,     -0.002176],
    [-0.05295, -0.002176, 16.01]
])  # kg·m² (exact from document)

# ========================= ORBIT PARAMETERS =========================
ALTITUDE_KM = 600.0
MU = 3.986004418e14          # m³/s²
R_EARTH = 6378137.0          # m
a = R_EARTH + ALTITUDE_KM * 1000.0
n = np.sqrt(MU / a**3)

ORBIT_PERIOD = 2 * np.pi / n
print(f"Orbit period: {ORBIT_PERIOD/60:.2f} minutes")

# ========================= ACTUATORS (Exact from Document) =========================
# Reaction Wheels - Pyramidal cluster
H_RW_MAX = 1.4          # Nms per wheel
TAU_RW_MAX = 0.475      # Nm per wheel (475 mNm)

# VSCMG - Effective capability in body frame
VSCMG_TAU_MAX = np.array([0.031, 0.031, 0.041])   # Nm   [x, y, z]
VSCMG_H_MAX   = np.array([0.049, 0.049, 0.098])   # Nms  [x, y, z]

# Magnetorquers
M_MAX = 18.28           # Am² per axis

# ========================= SINGULARITY AVOIDANCE =========================
D_THRESHOLD = 0.2           # Singular measure threshold (tunable)
SR_DAMPING = 0.02           # Singularity Robust damping factor λ
NULL_MOTION_GAIN = 0.5      # Null motion gain
SINGULARITY_MEASURE = "cond"  # "det" or "cond"

# ========================= CONTROLLER GAINS (Starting values - will tune) =========================
KP = np.diag([12.0, 15.0, 18.0])
KD = np.diag([40.0, 48.0, 55.0])
KI = np.diag([0.5, 0.5, 0.5])

# Magnetic Dumping
K_DUMP = 1.5e4
H_DUMP_THRESHOLD = 0.7 * 1.4 * 4   # ~70% of total RW capacity as trigger

# ========================= SIMULATION =========================
DT = 0.05                    # seconds
SIM_TIME_SLEW = 400.0
SIM_TIME_NADIR = 5 * ORBIT_PERIOD + 100.0   # 5 full orbits + margin

# Robustness
INERTIA_UNCERTAINTY = 0.10   # ±10%
MISALIGNMENT_DEG = 5.0       # ±5°

# Reproducibility
SEED = 42
np.random.seed(SEED)

# Environment
USE_IGRF = True
INCLUDE_AERO = True
INCLUDE_SRP = True

# Output control
SAVE_PLOTS = True
SHOW_PLOTS = False