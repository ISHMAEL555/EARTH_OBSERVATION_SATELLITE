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
"""

import numpy as np


class Spacecraft:
    """
    3-DOF rigid-body spacecraft attitude dynamics.

    Notes
    -----
    Assumptions:
    - Rigid spacecraft
    - Constant inertia matrix
    - Quaternion attitude representation
    - Translational dynamics are neglected
    - All external torques are supplied by the simulator
    """

    def __init__(
        self,
        inertia: np.ndarray,
        mass: float,
        q0: np.ndarray = None,
        omega0: np.ndarray = None,
    ):
        """
        Initialize spacecraft model.

        Parameters
        ----------
        inertia : ndarray (3x3)
            Spacecraft inertia matrix [kg·m²]

        mass : float
            Spacecraft mass [kg]

        q0 : ndarray (4,), optional
            Initial quaternion (scalar-first)

        omega0 : ndarray (3,), optional
            Initial angular velocity [rad/s]
        """

        self.mass = mass

        self.J = np.asarray(inertia, dtype=float)

        if self.J.shape != (3, 3):
            raise ValueError("Inertia matrix must be 3x3.")

        self.J_inv = np.linalg.inv(self.J)

        self.q = (
            np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
            if q0 is None
            else np.asarray(q0, dtype=float)
        )

        self.q /= np.linalg.norm(self.q)

        self.omega = (
            np.zeros(3, dtype=float)
            if omega0 is None
            else np.asarray(omega0, dtype=float)
        )

    # ==========================================================
    # Quaternion Kinematics
    # ==========================================================

    def quaternion_derivative(self) -> np.ndarray:
        """
        Compute quaternion derivative.

        Returns
        -------
        ndarray (4,)
            Quaternion derivative.
        """

        wx, wy, wz = self.omega

        Omega = np.array([
            [0.0, -wx, -wy, -wz],
            [wx,   0.0,  wz, -wy],
            [wy,  -wz,  0.0,  wx],
            [wz,   wy, -wx,  0.0]
        ])

        return 0.5 * Omega @ self.q

    # ==========================================================
    # Rotational Dynamics
    # ==========================================================

    def angular_acceleration(self, total_torque: np.ndarray) -> np.ndarray:
        """
        Compute spacecraft angular acceleration.

        Parameters
        ----------
        total_torque : ndarray (3,)
            Total applied external torque [N·m]

        Returns
        -------
        ndarray (3,)
            Angular acceleration [rad/s²]
        """

        total_torque = np.asarray(total_torque, dtype=float)

        gyroscopic = np.cross(
            self.omega,
            self.J @ self.omega
        )

        return self.J_inv @ (total_torque - gyroscopic)

    # ==========================================================
    # State Propagation
    # ==========================================================

    def propagate(
        self,
        total_torque: np.ndarray,
        dt: float,
    ) -> None:
        """
        Propagate spacecraft dynamics using Forward Euler integration.

        Parameters
        ----------
        total_torque : ndarray (3,)
            Net applied torque [N·m]

        dt : float
            Simulation time step [s]
        """

        if dt <= 0.0:
            raise ValueError("Time step must be positive.")

        # Angular dynamics
        omega_dot = self.angular_acceleration(total_torque)
        self.omega += omega_dot * dt

        # Quaternion kinematics
        q_dot = self.quaternion_derivative()
        self.q += q_dot * dt

        # Normalize quaternion
        self.q /= np.linalg.norm(self.q)

    # ==========================================================
    # Utility Functions
    # ==========================================================

    def get_state(self):
        """
        Return current spacecraft state.

        Returns
        -------
        tuple
            (quaternion, angular_velocity)
        """

        return self.q.copy(), self.omega.copy()

    def set_state(
        self,
        q: np.ndarray,
        omega: np.ndarray,
    ) -> None:
        """
        Set spacecraft state.

        Parameters
        ----------
        q : ndarray (4,)
            Quaternion.

        omega : ndarray (3,)
            Angular velocity [rad/s]
        """

        self.q = np.asarray(q, dtype=float)
        self.q /= np.linalg.norm(self.q)

        self.omega = np.asarray(omega, dtype=float)

    def reset(self) -> None:
        """
        Reset spacecraft to default initial conditions.
        """

        self.q = np.array(
            [1.0, 0.0, 0.0, 0.0],
            dtype=float
        )

        self.omega = np.zeros(3, dtype=float)