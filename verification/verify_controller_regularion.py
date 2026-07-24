"""
verification/verify_controller_regulation.py

Closed-loop verification of the Quaternion PD controller.

Objective
---------
Verify that the spacecraft converges to a fixed reference attitude.

This verification intentionally excludes

- Orbit
- Guidance
- Disturbances
- Actuators

Only:

Controller -> Spacecraft Dynamics
"""

import numpy as np

from controllers.quaternion_pd import QuaternionPD

from models.spacecraft import Spacecraft

from models.dynamics.quaternion import (
    multiply,
    inverse,
    enforce_unique,
)

from scenarios.nadir_pointing.config import (
    MISSION,
    CONTROLLER,
)


# ==========================================================
# Helper Functions
# ==========================================================

def quaternion_from_axis_angle(axis, angle_deg):
    """
    Create quaternion from axis-angle.
    """

    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)

    theta = np.radians(angle_deg)

    return np.array([
        np.cos(theta / 2.0),
        *(axis * np.sin(theta / 2.0)),
    ])


def attitude_error_deg(q_ref, q):

    q_error = multiply(
        q_ref,
        inverse(q),
    )

    q_error = enforce_unique(q_error)

    angle = np.degrees(
        2.0 * np.arccos(
            np.clip(
                q_error[0],
                -1.0,
                1.0,
            )
        )
    )

    return angle


# ==========================================================
# Simulation Parameters
# ==========================================================

dt = 0.1

simulation_time = 300.0

steps = int(simulation_time / dt)


# ==========================================================
# Initial Condition
# ==========================================================

# Change this value to:
# 10, 30, 60, 90, 120

initial_error = 10.0

q0 = quaternion_from_axis_angle(
    axis=[1, 0, 0],
    angle_deg=initial_error,
)

omega0 = np.zeros(3)

q_ref = np.array([
    1.0,
    0.0,
    0.0,
    0.0,
])

omega_ref = np.zeros(3)


# ==========================================================
# Controller
# ==========================================================

controller = QuaternionPD(
    proportional_gain=np.array(
        CONTROLLER["kp"],
        dtype=float,
    ),
    derivative_gain=np.array(
        CONTROLLER["kd"],
        dtype=float,
    ),
)


# ==========================================================
# Spacecraft
# ==========================================================

spacecraft = Spacecraft(
    inertia=np.array(
        MISSION["spacecraft"]["inertia"],
        dtype=float,
    ),
    mass=MISSION["spacecraft"]["mass"],
    q0=q0,
    omega0=omega0,
)


# ==========================================================
# Data Storage
# ==========================================================

time_history = []

error_history = []

rate_history = []

torque_history = []


# ==========================================================
# Simulation
# ==========================================================

print("=" * 70)
print("QUATERNION PD REGULATION VERIFICATION")
print("=" * 70)

print(f"Initial Attitude Error : {initial_error:.1f} deg")
print(f"Simulation Time        : {simulation_time:.1f} s")
print(f"Time Step              : {dt:.2f} s")

print("\n--------------------------------------------------------------")
print(f"{'Step':>6} {'Error(deg)':>14} {'|ω|':>12} {'|τ|':>12}")
print("--------------------------------------------------------------")

for step in range(steps):

    q = spacecraft.quaternion

    omega = spacecraft.angular_velocity

    torque = controller.compute(
        current_quaternion=q,
        desired_quaternion=q_ref,
        body_rates=omega,
        desired_body_rates=omega_ref,
    )

    spacecraft.propagate(
        total_torque=torque,
        dt=dt,
    )

    error = attitude_error_deg(
        q_ref,
        spacecraft.quaternion,
    )

    rate = np.linalg.norm(
        spacecraft.angular_velocity,
    )

    torque_mag = np.linalg.norm(
        torque,
    )

    time_history.append(step * dt)
    error_history.append(error)
    rate_history.append(rate)
    torque_history.append(torque_mag)

    if step % 100 == 0:

        print(
            f"{step:6d}"
            f"{error:14.3f}"
            f"{rate:12.5f}"
            f"{torque_mag:12.5f}"
        )


# ==========================================================
# Results
# ==========================================================

print("\n" + "=" * 70)

print("RESULTS")

print("=" * 70)

print(f"Initial Error : {error_history[0]:.3f} deg")

print(f"Final Error   : {error_history[-1]:.3f} deg")

print(f"Maximum Rate  : {max(rate_history):.6f} rad/s")

print(f"Maximum Torque: {max(torque_history):.6f} Nm")

print(f"Final Quaternion : {spacecraft.quaternion}")

print(f"Final Body Rate  : {spacecraft.angular_velocity}")

print(f"Rotational Energy: {spacecraft.rotational_kinetic_energy:.6f} J")

print("=" * 70)

if error_history[-1] < 1.0:

    print("PASS : Spacecraft converged.")

else:

    print("FAIL : Spacecraft did not converge.")

print("=" * 70)