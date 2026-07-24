"""
models/actuators/rw_pyramid.py

Four-Wheel Pyramidal Reaction Wheel Assembly

Current Model
-------------
- Ideal actuator
- Static torque allocation
- Torque saturation
- Momentum saturation
- Perfect wheel dynamics

Future Models
-------------
- First-order actuator dynamics
- Friction
- Bearing losses
- Motor electrical model
- Wheel imbalance
- Failure modes

References
----------
- Markley & Crassidis
- Wie
- ECSS-E-ST-60-30C
"""

import numpy as np


class RWPyramid:
    """
    Four-wheel pyramidal reaction wheel assembly.

    Parameters
    ----------
    wheel_axes : ndarray (3,N)
        Wheel spin-axis unit vectors expressed in the body frame.

    max_torque : float
        Maximum wheel torque [N·m].

    max_momentum : float
        Maximum wheel momentum [N·m·s].
    """

    def __init__(
        self,
        wheel_axes: np.ndarray,
        max_torque: float,
        max_momentum: float,
    ):

        wheel_axes = np.asarray(
            wheel_axes,
            dtype=float,
        )

        if wheel_axes.ndim != 2:
            raise ValueError(
                "wheel_axes must be a 2-D array."
            )

        if wheel_axes.shape[0] != 3:
            raise ValueError(
                "wheel_axes must have shape (3,N)."
            )

        if max_torque <= 0.0:
            raise ValueError(
                "max_torque must be positive."
            )

        if max_momentum <= 0.0:
            raise ValueError(
                "max_momentum must be positive."
            )

        self.wheel_axes = wheel_axes.copy()

        self.num_wheels = self.wheel_axes.shape[1]

        self.max_torque = float(max_torque)

        self.max_momentum = float(max_momentum)

        # Allocation matrix

        self.allocation_matrix = self.wheel_axes

        self.allocation_matrix_pinv = np.linalg.pinv(
            self.allocation_matrix
        )

        # =====================================================
        # Dynamic States
        # =====================================================

        self.actual_wheel_torque = np.zeros(
            self.num_wheels
        )

        self.wheel_momentum = np.zeros(
            self.num_wheels
        )

        self.body_torque = np.zeros(3)

    # =========================================================
    # Public Interface
    # =========================================================

    def update(
        self,
        commanded_body_torque: np.ndarray,
        dt: float,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """
        Advance the reaction wheel assembly.

        Parameters
        ----------
        commanded_body_torque : ndarray (3,)
            Desired spacecraft body torque [N·m].

        dt : float
            Simulation time step [s].

        Returns
        -------
        actual_wheel_torque : ndarray (N,)
            Saturated wheel torques [N·m].

        wheel_momentum : ndarray (N,)
            Stored wheel momentum [N·m·s].

        body_torque : ndarray (3,)
            Generated spacecraft body torque [N·m].
        """

        commanded_body_torque = self._validate_vector(
            commanded_body_torque,
            "commanded_body_torque",
            3,
        )

        if dt <= 0.0:
            raise ValueError(
                "dt must be positive."
            )

        commanded_wheel_torque = -(

            self.allocation_matrix_pinv

            @ commanded_body_torque

        )

        self.actual_wheel_torque = np.clip(

            commanded_wheel_torque,

            -self.max_torque,

            self.max_torque,

        )

        self.wheel_momentum += (

            self.actual_wheel_torque * dt

        )

        self.wheel_momentum = np.clip(

            self.wheel_momentum,

            -self.max_momentum,

            self.max_momentum,

        )

        self.body_torque = -(

            self.allocation_matrix

            @ self.actual_wheel_torque

        )

        return (

            self.actual_wheel_torque.copy(),

            self.wheel_momentum.copy(),

            self.body_torque.copy(),

        )

    # =========================================================
    # Utility Functions
    # =========================================================

    def get_total_momentum(self) -> np.ndarray:
        """
        Total spacecraft angular momentum stored
        in the reaction wheel cluster.
        """

        return (
            self.wheel_axes
            @ self.wheel_momentum
        )

    def get_total_momentum_capacity(self) -> float:
        """
        Total scalar momentum capacity.
        """

        return (
            self.num_wheels
            * self.max_momentum
        )

    def reset(self) -> None:
        """
        Reset dynamic actuator states.
        """

        self.actual_wheel_torque.fill(0.0)

        self.wheel_momentum.fill(0.0)

        self.body_torque.fill(0.0)

    # =========================================================
    # Validation
    # =========================================================

    @staticmethod
    def _validate_vector(
        vector: np.ndarray,
        name: str,
        size: int,
    ) -> np.ndarray:

        vector = np.asarray(
            vector,
            dtype=float,
        )

        if vector.shape != (size,):
            raise ValueError(
                f"{name} must have shape ({size},)."
            )

        if not np.all(
            np.isfinite(vector)
        ):
            raise ValueError(
                f"{name} contains invalid values."
            )

        return vector