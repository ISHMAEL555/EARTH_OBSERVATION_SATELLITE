"""
spacecraft.py
==============

Rigid-body spacecraft attitude dynamics model.

This module represents the spacecraft state and propagates the
attitude using the reusable dynamics library.

State
-----
q     : Unit quaternion [q0, q1, q2, q3] (scalar-first)
omega : Body angular velocity [rad/s]

Assumptions
-----------
- Rigid spacecraft
- Constant inertia matrix
- Quaternion attitude representation
- Translational dynamics neglected
- External torques supplied by the simulator
"""

import copy
import numpy as np

from models.dynamics.quaternion import (
    normalize,
    quaternion_to_dcm,
)

from models.dynamics.attitude_kinematics import (
    propagate_rk4,
)

from models.dynamics.rigid_body import (
    angular_acceleration,
    angular_momentum,
    rotational_kinetic_energy,
)


class Spacecraft:
    """
    Three-degree-of-freedom rigid-body spacecraft.
    """

    def __init__(
        self,
        inertia: np.ndarray,
        mass: float,
        q0: np.ndarray | None = None,
        omega0: np.ndarray | None = None,
    ) -> None:

        # ==========================================================
        # Mass
        # ==========================================================

        if mass <= 0.0:
            raise ValueError(
                "Mass must be positive."
            )

        self.mass = float(mass)

        # ==========================================================
        # Inertia
        # ==========================================================

        self.J = np.asarray(
            inertia,
            dtype=float,
        )

        if self.J.shape != (3, 3):
            raise ValueError(
                "Inertia matrix must have shape (3,3)."
            )

        # ==========================================================
        # Initial Quaternion
        # ==========================================================

        if q0 is None:

            self.q = np.array(
                [1.0, 0.0, 0.0, 0.0],
                dtype=float,
            )

        else:

            self.q = normalize(q0)

        # ==========================================================
        # Initial Angular Velocity
        # ==========================================================

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

            self.omega = omega0.copy()

    # ==========================================================
    # Dynamics
    # ==========================================================

    def propagate(
        self,
        total_torque: np.ndarray,
        dt: float,
    ) -> None:
        """
        Propagate spacecraft state.

        Parameters
        ----------
        total_torque : ndarray (3,)
            Net applied body torque [N·m].

        dt : float
            Time step [s].
        """

        if dt <= 0.0:
            raise ValueError(
                "Time step must be positive."
            )

        omega_dot = angular_acceleration(
            self.J,
            self.omega,
            total_torque,
        )

        self.omega += omega_dot * dt

        self.q = propagate_rk4(
            self.q,
            self.omega,
            dt,
        )

    # ==========================================================
    # State Management
    # ==========================================================

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

        omega = np.asarray(
            omega,
            dtype=float,
        )

        if omega.shape != (3,):
            raise ValueError(
                "Angular velocity must have shape (3,)."
            )

        self.q = normalize(q)
        self.omega = omega.copy()

    def reset(
        self,
    ) -> None:
        """
        Reset spacecraft to default state.
        """

        self.q = np.array(
            [1.0, 0.0, 0.0, 0.0],
            dtype=float,
        )

        self.omega = np.zeros(
            3,
            dtype=float,
        )

    def copy(
        self,
    ):
        """
        Return deep copy of spacecraft.
        """

        return copy.deepcopy(self)

    # ==========================================================
    # Properties
    # ==========================================================

    @property
    def inertia(
        self,
    ) -> np.ndarray:
        """
        Spacecraft inertia matrix.
        """

        return self.J.copy()

    @property
    def quaternion(
        self,
    ) -> np.ndarray:
        """
        Current attitude quaternion.
        """

        return self.q.copy()

    @property
    def angular_velocity(
        self,
    ) -> np.ndarray:
        """
        Current body angular velocity.
        """

        return self.omega.copy()

    @property
    def body_to_eci_dcm(
        self,
    ) -> np.ndarray:
        """
        Body-to-ECI Direction Cosine Matrix.
        """

        return quaternion_to_dcm(
            self.q,
        )

    @property
    def eci_to_body_dcm(
        self,
    ) -> np.ndarray:
        """
        ECI-to-body Direction Cosine Matrix.
        """

        return self.body_to_eci_dcm.T

    @property
    def angular_momentum(
        self,
    ) -> np.ndarray:
        """
        Spacecraft angular momentum.
        """

        return angular_momentum(
            self.J,
            self.omega,
        )

    @property
    def rotational_kinetic_energy(
        self,
    ) -> float:
        """
        Spacecraft rotational kinetic energy.
        """

        return rotational_kinetic_energy(
            self.J,
            self.omega,
        )