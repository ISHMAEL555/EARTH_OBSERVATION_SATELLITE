"""
verification/verify_controller.py

Quaternion PD Controller Verification

This script verifies the engineering behaviour of the
Quaternion PD controller.

Objectives
----------
1. Verify quaternion error computation
2. Verify rotation axis and angle
3. Verify proportional contribution
4. Verify derivative contribution
5. Verify total commanded torque
6. Verify whether the commanded torque tends to reduce
   the attitude error.

Author
------
Ishmael
"""

from __future__ import annotations

import numpy as np

from controllers.quaternion_pd import QuaternionPD

from models.dynamics.quaternion import (
    multiply,
    inverse,
    enforce_unique,
)


# ==========================================================
# Helper Functions
# ==========================================================

def print_header(title: str):

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_vector(name: str, value):

    print(f"\n{name}")
    print(value)


# ==========================================================
# Controller Configuration
# ==========================================================

Kp = np.diag([0.08, 0.08, 0.08])

Kd = np.diag([0.20, 0.20, 0.20])

controller = QuaternionPD(
    proportional_gain=Kp,
    derivative_gain=Kd,
)


# ==========================================================
# Mission Case
# ==========================================================

current_quaternion = np.array(
    [1.0, 0.0, 0.0, 0.0]
)

desired_quaternion = np.array(
    [
        0.5,
        -0.5,
        0.5,
        0.5,
    ]
)

body_rates = np.zeros(3)

desired_body_rates = np.zeros(3)


# ==========================================================
# Quaternion Error
# ==========================================================

q_error = multiply(
    desired_quaternion,
    inverse(current_quaternion),
)

q_error = enforce_unique(q_error)

rotation_angle = np.degrees(
    2.0 * np.arccos(
        np.clip(q_error[0], -1.0, 1.0)
    )
)

if np.linalg.norm(q_error[1:]) > 1e-12:

    rotation_axis = (
        q_error[1:]
        / np.linalg.norm(q_error[1:])
    )

else:

    rotation_axis = np.zeros(3)


# ==========================================================
# Controller Output
# ==========================================================

attitude_error = q_error[1:]

rate_error = (
    body_rates
    - desired_body_rates
)

p_torque = -Kp @ attitude_error

d_torque = -Kd @ rate_error

total_torque = controller.compute(
    current_quaternion=current_quaternion,
    desired_quaternion=desired_quaternion,
    body_rates=body_rates,
    desired_body_rates=desired_body_rates,
)


# ==========================================================
# Verification Report
# ==========================================================

print_header(
    "QUATERNION PD CONTROLLER VERIFICATION"
)

print("\nMISSION CASE")
print("-----------------------------")
print("Identity  →  LVLH Nadir")

print_vector(
    "Current Quaternion",
    current_quaternion,
)

print_vector(
    "Desired Quaternion",
    desired_quaternion,
)

print_vector(
    "Quaternion Error",
    q_error,
)

print(
    "\nRotation Angle (deg)"
)
print(rotation_angle)

print_vector(
    "Rotation Axis",
    rotation_axis,
)

print_header(
    "PROPORTIONAL TERM"
)

print_vector(
    "Attitude Error",
    attitude_error,
)

print_vector(
    "P Torque",
    p_torque,
)

print_header(
    "DERIVATIVE TERM"
)

print_vector(
    "Rate Error",
    rate_error,
)

print_vector(
    "D Torque",
    d_torque,
)

print_header(
    "TOTAL CONTROLLER OUTPUT"
)

print_vector(
    "Commanded Torque",
    total_torque,
)

print_header(
    "ENGINEERING INTERPRETATION"
)

print(
    """
Questions to verify:

✓ Is the quaternion error physically correct?

✓ Is the rotation axis correct?

✓ Does the proportional torque oppose the error?

✓ Does the derivative torque oppose the rate error?

✓ If this torque is applied to the spacecraft,
  will the attitude error decrease?

If any answer is NO, investigate the controller,
actuator sign convention, or quaternion convention.
"""
)