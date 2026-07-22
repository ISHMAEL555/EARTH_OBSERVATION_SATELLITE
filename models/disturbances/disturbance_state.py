"""
Common disturbance input state.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class DisturbanceState:

    time: float

    position_eci: np.ndarray

    velocity_eci: np.ndarray

    body_to_eci_dcm: np.ndarray

    inertia_matrix: np.ndarray

    magnetic_field_eci: np.ndarray

    magnetic_field_body: np.ndarray

    atmospheric_density: float

    solar_vector_eci: np.ndarray