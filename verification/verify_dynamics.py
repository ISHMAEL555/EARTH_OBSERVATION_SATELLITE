"""
verification/verify_dynamics.py

Verify one-step spacecraft dynamics.

Objective
---------
Verify that the spacecraft responds correctly to the
controller torque.

Pipeline

Orbit
    ↓
Guidance
    ↓
Controller
    ↓
Spacecraft Dynamics
    ↓
Attitude Error Evolution
"""

import numpy as np

from scenarios.nadir_pointing.config import (
    MISSION,
    CONTROLLER,
    INITIAL_CONDITIONS,
)

from simulator.builder import Builder

from guidance.nadir_pointing import (
    NadirPointingGuidance,
)

from models.dynamics.quaternion import (
    multiply,
    inverse,
    enforce_unique,
)


# ==========================================================
# Helper
# ==========================================================

def attitude_error_angle(q_ref, q):

    q_error = multiply(
        q_ref,
        inverse(q),
    )

    q_error = enforce_unique(q_error)

    return np.degrees(
        2.0
        * np.arccos(
            np.clip(
                q_error[0],
                -1.0,
                1.0,
            )
        )
    )


# ==========================================================
# Build Simulation
# ==========================================================

print("=" * 70)
print("SPACECRAFT DYNAMICS VERIFICATION")
print("=" * 70)

builder = Builder(
    mission=MISSION,
    controller=CONTROLLER,
    initial_conditions=INITIAL_CONDITIONS,
)

sim = builder.build()

spacecraft = sim.spacecraft
controller = sim.controller

guidance = NadirPointingGuidance()

# ==========================================================
# Orbit
# ==========================================================

r, v = sim.orbit.propagate(0.0)

# ==========================================================
# Guidance
# ==========================================================

q_ref, omega_ref = guidance.compute(
    r,
    v,
)

print("\n" + "=" * 70)
print("GUIDANCE")
print("=" * 70)

print("\nPosition ECI [m]")
print(r)

print("\nVelocity ECI [m/s]")
print(v)

print("\nReference Quaternion")
print(q_ref)

print("\nReference Angular Velocity [rad/s]")
print(omega_ref)

print("\nReference Angular Velocity Magnitude")
print(np.linalg.norm(omega_ref))

# ==========================================================
# Initial Spacecraft State
# ==========================================================

q = spacecraft.quaternion
omega = spacecraft.angular_velocity

print("\n" + "=" * 70)
print("INITIAL SPACECRAFT STATE")
print("=" * 70)

print("\nCurrent Quaternion")
print(q)

print("\nCurrent Angular Velocity [rad/s]")
print(omega)

print("\nDesired Angular Velocity [rad/s]")
print(omega_ref)

print("\nRate Error")
print(
    omega - omega_ref
)

initial_error = attitude_error_angle(
    q_ref,
    q,
)

print("\nInitial Attitude Error [deg]")
print(initial_error)

# ==========================================================
# Controller
# ==========================================================

torque = controller.compute(
    current_quaternion=q,
    desired_quaternion=q_ref,
    body_rates=omega,
    desired_body_rates=omega_ref,
)

print("\n" + "=" * 70)
print("CONTROLLER")
print("=" * 70)

print("\nCommanded Torque [Nm]")
print(torque)

# ==========================================================
# Propagation
# ==========================================================

print("\nQuaternion Before")
print(spacecraft.quaternion)

print("\nAngular Velocity Before")
print(spacecraft.angular_velocity)

spacecraft.propagate(
    total_torque=torque,
    dt=sim.time_step,
)

print("\nQuaternion After")
print(spacecraft.quaternion)

print("\nAngular Velocity After")
print(spacecraft.angular_velocity)

# ==========================================================
# Error Evolution
# ==========================================================

q_new = spacecraft.quaternion

new_error = attitude_error_angle(
    q_ref,
    q_new,
)

print("\n" + "=" * 70)
print("ATTITUDE ERROR EVOLUTION")
print("=" * 70)

print("\nInitial Error [deg]")
print(initial_error)

print("\nNew Error [deg]")
print(new_error)

print("\nError Change [deg]")
print(new_error - initial_error)

# ==========================================================
# Spacecraft Properties
# ==========================================================

print("\n" + "=" * 70)
print("SPACECRAFT STATE")
print("=" * 70)

print("\nAngular Momentum")
print(spacecraft.angular_momentum)

print("\nRotational Kinetic Energy")
print(spacecraft.rotational_kinetic_energy)

# ==========================================================
# Result
# ==========================================================

print("\n" + "=" * 70)

if new_error < initial_error:

    print("PASS")
    print("✓ Attitude error decreased.")
    print("✓ Spacecraft responded correctly.")

else:

    print("FAIL")
    print("✗ Attitude error increased.")
    print("✗ Spacecraft is rotating away from the reference.")
    print("Investigate:")
    print("  • Quaternion convention")
    print("  • Reference angular velocity frame")
    print("  • Spacecraft dynamics")
    print("  • Control torque sign")

print("=" * 70)