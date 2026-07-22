"""
scenarios/nadir_pointing/logging.py

Simulation telemetry logger.

Stores all simulation data generated during the mission.
"""

from __future__ import annotations

import numpy as np


class SimulationLogger:
    """
    Stores simulation telemetry.

    Each call to log() records one simulation time step.
    """

    def __init__(self):

        self.reset()

    # ==========================================================
    # Reset
    # ==========================================================

    def reset(self) -> None:

        self.time = []

        self.position = []
        self.velocity = []

        self.quaternion = []
        self.reference_quaternion = []

        self.angular_velocity = []
        self.reference_angular_velocity = []

        self.control_torque = []

        self.reaction_wheel_torque = []
        self.reaction_wheel_momentum = []

        self.magnetorquer_dipole = []
        self.magnetorquer_torque = []

        self.disturbance_torque = []

    # ==========================================================
    # Logging
    # ==========================================================

    def log(
        self,
        *,
        time: float,
        position: np.ndarray,
        velocity: np.ndarray,
        quaternion: np.ndarray,
        reference_quaternion: np.ndarray,
        angular_velocity: np.ndarray,
        reference_angular_velocity: np.ndarray,
        control_torque: np.ndarray,
        reaction_wheel_torque: np.ndarray,
        reaction_wheel_momentum: np.ndarray,
        magnetorquer_dipole: np.ndarray,
        magnetorquer_torque: np.ndarray,
        disturbance_torque: np.ndarray,
    ) -> None:

        self.time.append(float(time))

        self.position.append(position.copy())
        self.velocity.append(velocity.copy())

        self.quaternion.append(quaternion.copy())
        self.reference_quaternion.append(
            reference_quaternion.copy()
        )

        self.angular_velocity.append(
            angular_velocity.copy()
        )

        self.reference_angular_velocity.append(
            reference_angular_velocity.copy()
        )

        self.control_torque.append(
            control_torque.copy()
        )

        self.reaction_wheel_torque.append(
            reaction_wheel_torque.copy()
        )

        self.reaction_wheel_momentum.append(
            reaction_wheel_momentum.copy()
        )

        self.magnetorquer_dipole.append(
            magnetorquer_dipole.copy()
        )

        self.magnetorquer_torque.append(
            magnetorquer_torque.copy()
        )

        self.disturbance_torque.append(
            disturbance_torque.copy()
        )

    # ==========================================================
    # Export
    # ==========================================================

    def as_dict(self) -> dict:

        return {

            "time": np.asarray(self.time),

            "position": np.asarray(self.position),

            "velocity": np.asarray(self.velocity),

            "quaternion": np.asarray(self.quaternion),

            "reference_quaternion": np.asarray(
                self.reference_quaternion
            ),

            "angular_velocity": np.asarray(
                self.angular_velocity
            ),

            "reference_angular_velocity": np.asarray(
                self.reference_angular_velocity
            ),

            "control_torque": np.asarray(
                self.control_torque
            ),

            "reaction_wheel_torque": np.asarray(
                self.reaction_wheel_torque
            ),

            "reaction_wheel_momentum": np.asarray(
                self.reaction_wheel_momentum
            ),

            "magnetorquer_dipole": np.asarray(
                self.magnetorquer_dipole
            ),

            "magnetorquer_torque": np.asarray(
                self.magnetorquer_torque
            ),

            "disturbance_torque": np.asarray(
                self.disturbance_torque
            ),
        }