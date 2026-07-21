"""
models/disturbances/disturbances.py

Disturbance Model Manager

Registers and evaluates all environmental disturbance models.
"""

from typing import List
import numpy as np


class Disturbances:
    """
    Manages spacecraft disturbance models.

    Any disturbance model implementing a `compute()` method can be
    registered with this class.
    """

    def __init__(self):
        """
        Initialize an empty disturbance manager.
        """

        self._disturbance_models: List[object] = []

    # ==========================================================
    # Register Disturbance
    # ==========================================================

    def add(
        self,
        disturbance_model,
    ):
        """
        Register a disturbance model.

        Parameters
        ----------
        disturbance_model : object
            Disturbance model implementing a compute() method.
        """

        if not hasattr(disturbance_model, "compute"):
            raise TypeError(
                "Disturbance model must implement a compute() method."
            )

        self._disturbance_models.append(
            disturbance_model
        )

    # ==========================================================
    # Remove Disturbance
    # ==========================================================

    def remove(
        self,
        disturbance_model,
    ):
        """
        Remove a disturbance model.
        """

        self._disturbance_models.remove(
            disturbance_model
        )

    # ==========================================================
    # Clear Disturbances
    # ==========================================================

    def clear(self):
        """
        Remove all registered disturbance models.
        """

        self._disturbance_models.clear()

    # ==========================================================
    # Total Disturbance Torque
    # ==========================================================

    def compute(
        self,
        **kwargs,
    ) -> np.ndarray:
        """
        Compute the total disturbance torque.

        Parameters
        ----------
        **kwargs
            Keyword arguments forwarded to each disturbance model.

        Returns
        -------
        total_disturbance_torque : ndarray (3,)
            Total disturbance torque expressed in the
            spacecraft body frame [N·m].
        """

        total_disturbance_torque = np.zeros(3)

        for disturbance_model in self._disturbance_models:

            total_disturbance_torque += disturbance_model.compute(
                **kwargs
            )

        return total_disturbance_torque