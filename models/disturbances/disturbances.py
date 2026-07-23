"""
models/disturbances/disturbances.py

Disturbance manager.
"""

from __future__ import annotations

import numpy as np

from models.disturbances.disturbance_state import DisturbanceState


class Disturbances:
    """
    Collection of disturbance models.
    """

    def __init__(self):

        self._disturbance_models = []

    # ------------------------------------------------------
    # Registration
    # ------------------------------------------------------

    def add(
        self,
        disturbance,
    ):
        """
        Register a disturbance model.
        """

        if not hasattr(
            disturbance,
            "compute",
        ):
            raise TypeError(
                "Disturbance model must implement "
                "'compute(state)'."
            )

        self._disturbance_models.append(
            disturbance
        )

    # ------------------------------------------------------
    # Compute Total Disturbance Torque
    # ------------------------------------------------------

    def compute(
        self,
        state: DisturbanceState,
    ) -> np.ndarray:
        """
        Compute the total disturbance torque.

        Parameters
        ----------
        state : DisturbanceState
            Current spacecraft/environment state.

        Returns
        -------
        ndarray, shape (3,)
            Total disturbance torque expressed in the body frame.
        """

        total_torque = np.zeros(
            3,
            dtype=float,
        )

        for disturbance in self._disturbance_models:

            contribution = disturbance.compute(
                state
            )

            if contribution is None:
                continue

            contribution = np.asarray(
                contribution,
                dtype=float,
            )

            if contribution.shape != (3,):
                raise ValueError(
                    f"{type(disturbance).__name__} "
                    "must return a (3,) torque vector."
                )

            total_torque += contribution

        return total_torque