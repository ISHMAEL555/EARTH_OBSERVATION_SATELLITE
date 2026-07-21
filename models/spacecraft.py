"""
spacecraft.py
==============

Rigid-body spacecraft attitude dynamics model.

This module propagates the spacecraft rotational motion using
Euler's rotational equations and quaternion kinematics.

State
-----
q     : Unit quaternion [q0, q1, q2, q3] (scalar-first)
omega : Body angular velocity [rad/s]

Inputs
------
total_torque : Net external torque applied to the spacecraft [N·m]

Outputs
-------
Updated spacecraft attitude and angular velocity.

Integration
-----------
Forward Euler
"""

import numpy as np


class Spacecraft:
    """
    3-DOF rigid-body spacecraft attitude dynamics.

    Assumptions
    -----------
    - Rigid spacecraft
    - Constant inertia matrix
    - Quaternion attitude representation
    - Translational dynamics neglected
    - External torques supplied by the simulator
    """

    def __init__(
        self,
        inertia: np.ndarray,
        mass: float,
        q0: np.ndarray | None = None,
        omega0: np.ndarray | None = None,
    ) -> None:

        # --------------------------------------------------
        # Mass
        # --------------------------------------------------

        if mass <= 0:
            raise ValueError("Mass must be positive.")

        self.mass = float(mass)

        # --------------------------------------------------
        # Inertia
        # --------------------------------------------------

        self.J = np.asarray(inertia, dtype=float)

        if self.J.shape != (3, 3):
            raise ValueError("Inertia matrix must have shape (3,3).")

        self.J_inv = np.linalg.inv(self.J)

        # --------------------------------------------------
        # Quaternion
        # --------------------------------------------------

        if q0 is None:

            self.q = np.array(
                [1.0, 0.0, 0.0, 0.0],
                dtype=float,
            )

        else:

            q0 = np.asarray(q0, dtype=float)

            if q0.shape != (4,):
                raise ValueError(
                    "Quaternion must have shape (4,)."
                )

            norm = np.linalg.norm(q0)

            if norm < 1e-12:
                raise ValueError(
                    "Quaternion norm cannot be zero."
                )

            self.q = q0 / norm

        # --------------------------------------------------
        # Angular Velocity
        # --------------------------------------------------

        if omega0 is None:

            self.omega = np.zeros(
                3,
                dtype=float,
            )

        else:

            omega0 = np.asarray(
                omega0,
                dtype=float,
            )

            if omega0.shape != (3,):
                raise ValueError(
                    "Angular velocity must have shape (3,)."
                )

            self.omega = omega0

    # ======================================================
    # Properties
    # ======================================================

    @property
    def inertia(self) -> np.ndarray:
        """
        Return spacecraft inertia matrix.
        """

        return self.J

    # ======================================================
    # Quaternion Kinematics
    # ======================================================

    def quaternion_derivative(self) -> np.ndarray:
        """
        Compute quaternion derivative.
        """

        wx, wy, wz = self.omega

        Omega = np.array([
            [0.0, -wx, -wy, -wz],
            [wx,  0.0,  wz, -wy],
            [wy, -wz,  0.0,  wx],
            [wz,  wy, -wx,  0.0],
        ])

        return 0.5 * Omega @ self.q

    # ======================================================
    # Rotational Dynamics
    # ======================================================

    def angular_acceleration(
        self,
        total_torque: np.ndarray,
    ) -> np.ndarray:
        """
        Compute angular acceleration.
        """

        total_torque = np.asarray(
            total_torque,
            dtype=float,
        )

        if total_torque.shape != (3,):
            raise ValueError(
                "Torque must have shape (3,)."
            )

        gyroscopic = np.cross(
            self.omega,
            self.J @ self.omega,
        )

        return self.J_inv @ (
            total_torque - gyroscopic
        )

    # ======================================================
    # Propagation
    # ======================================================

    def propagate(
        self,
        total_torque: np.ndarray,
        dt: float,
    ) -> None:
        """
        Propagate spacecraft dynamics using
        Forward Euler integration.
        """

        if dt <= 0:
            raise ValueError(
                "Time step must be positive."
            )

        omega_dot = self.angular_acceleration(
            total_torque
        )

        self.omega += omega_dot * dt

        q_dot = self.quaternion_derivative()

        self.q += q_dot * dt

        norm = np.linalg.norm(self.q)

        if norm < 1e-12:
            raise RuntimeError(
                "Quaternion norm collapsed."
            )

        self.q /= norm

    # ======================================================
    # Utilities
    # ======================================================

    def get_state(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return spacecraft state.
        """

        return (
            self.q.copy(),
            self.omega.copy(),
        )

    def set_state(
        self,
        q: np.ndarray,
        omega: np.ndarray,
    ) -> None:
        """
        Set spacecraft state.
        """

        q = np.asarray(
            q,
            dtype=float,
        )

        omega = np.asarray(
            omega,
            dtype=float,
        )

        if q.shape != (4,):
            raise ValueError(
                "Quaternion must have shape (4,)."
            )

        if omega.shape != (3,):
            raise ValueError(
                "Angular velocity must have shape (3,)."
            )

        norm = np.linalg.norm(q)

        if norm < 1e-12:
            raise ValueError(
                "Quaternion norm cannot be zero."
            )

        self.q = q / norm
        self.omega = omega

    def reset(self) -> None:
        """
        Reset spacecraft state.
        """

        self.q = np.array(
            [1.0, 0.0, 0.0, 0.0],
            dtype=float,
        )

        self.omega = np.zeros(
            3,
            dtype=float,
        )