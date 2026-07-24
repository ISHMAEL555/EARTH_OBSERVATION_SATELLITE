"""
verification/monitor.py

Closed-loop verification monitor.

Inspects the major simulation states during execution.

Author
------
Ishmael
"""

from __future__ import annotations

import numpy as np

from models.dynamics.quaternion import (
    quaternion_to_dcm,
    multiply,
    inverse,
)


class VerificationMonitor:

    def __init__(
        self,
        interval: int = 1,
        first_n_steps: int = 20,
    ):

        self.interval = interval
        self.first_n_steps = first_n_steps

    # ==========================================================
    # Public Interface
    # ==========================================================

    def inspect(self, scenario):

        step = scenario.simulator.current_step

        if step % self.interval != 0:
            return

        if step >= self.first_n_steps:
            return

        print("\n")
        print("=" * 80)
        print(f"SIMULATION STEP : {step + 1}")
        print(f"TIME            : {scenario.simulator.time:.3f} s")
        print("=" * 80)

        self._orbit(scenario)
        self._guidance(scenario)
        self._spacecraft(scenario)
        self._error(scenario)
        self._controller(scenario)
        self._actuators(scenario)
        self._disturbances(scenario)
        self._validation(scenario)

    # ==========================================================
    # Orbit
    # ==========================================================

    def _orbit(self, s):

        print("\n[ORBIT]")

        if s.position_eci is None:
            print("Orbit not initialized.")
            return

        print("Position ECI")
        print(s.position_eci)

        print("Velocity ECI")
        print(s.velocity_eci)

        print(
            "Radius (km):",
            np.linalg.norm(s.position_eci) / 1000.0,
        )

        print(
            "Speed (m/s):",
            np.linalg.norm(s.velocity_eci),
        )

    # ==========================================================
    # Guidance
    # ==========================================================

    def _guidance(self, s):

        print("\n[GUIDANCE]")

        if s.q_ref is None:
            print("Guidance not available.")
            return

        print("Reference Quaternion")
        print(s.q_ref)

        print("Reference Body Rates")
        print(s.omega_ref)

        print("Reference DCM")
        print(quaternion_to_dcm(s.q_ref))

    # ==========================================================
    # Spacecraft
    # ==========================================================

    def _spacecraft(self, s):

        sc = s.sim.spacecraft

        print("\n[SPACECRAFT]")

        print("Current Quaternion")
        print(sc.quaternion)

        print("Current Body Rates (rad/s)")
        print(sc.angular_velocity)

        print("Current DCM")
        print(sc.body_to_eci_dcm)

        print("Angular Momentum (N·m·s)")
        print(sc.angular_momentum)

        print("Rotational Kinetic Energy (J)")
        print(sc.rotational_kinetic_energy)

    # ==========================================================
    # Error
    # ==========================================================

    def _error(self, s):

        if s.q_ref is None:
            return

        sc = s.sim.spacecraft

        print("\n[ATTITUDE ERROR]")

        q_error = multiply(
            s.q_ref,
            inverse(sc.quaternion),
        )

        print("Quaternion Error")
        print(q_error)

        angle = np.degrees(
            2.0
            * np.arccos(
                np.clip(
                    abs(q_error[0]),
                    -1.0,
                    1.0,
                )
            )
        )

        print(f"Attitude Error (deg): {angle:.6f}")

    # ==========================================================
    # Controller
    # ==========================================================

    def _controller(self, s):

        print("\n[CONTROLLER]")

        if s.control_torque is None:
            print("Controller not executed.")
            return

        print("Commanded Torque")
        print(s.control_torque)

    # ==========================================================
    # Actuators
    # ==========================================================

    def _actuators(self, s):

        print("\n[ACTUATORS]")

        print("Reaction Wheel Torque")
        print(s.actual_rw_torque)

        print("Reaction Wheel Body Torque")
        print(s.rw_body_torque)

        print("Wheel Momentum")
        print(s.wheel_momentum)

        print("Magnetorquer Dipole")
        print(s.actual_dipole)

        print("Magnetorquer Torque")
        print(s.magnetorquer_torque)

    # ==========================================================
    # Disturbances
    # ==========================================================

    def _disturbances(self, s):

        print("\n[DISTURBANCES]")

        print("Disturbance Torque")
        print(s.disturbance_torque)

        print("Total Torque")
        print(s.total_torque)

    # ==========================================================
    # Validation
    # ==========================================================

    def _validation(self, s):

        sc = s.sim.spacecraft

        print("\n[VALIDATION]")

        q_norm = np.linalg.norm(sc.quaternion)

        dcm = sc.body_to_eci_dcm

        ortho = np.linalg.norm(
            dcm @ dcm.T - np.eye(3)
        )

        det = np.linalg.det(dcm)

        print(f"Quaternion Norm     : {q_norm:.12f}")
        print(f"Orthogonality Error : {ortho:.3e}")
        print(f"det(DCM)            : {det:.12f}")

        print("=" * 80)