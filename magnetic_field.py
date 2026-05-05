"""
models/environment/magnetic_field.py

IGRF Magnetic Field Model (Recommended)
"""

import numpy as np
# from pyIGRF import igrf_value   # Uncomment when running

class MagneticField:
    def __init__(self, use_igrf: bool = True):
        self.use_igrf = use_igrf

    def get_B_body(self, pos_eci: np.ndarray, q: np.ndarray, t: float) -> np.ndarray:
        """
        Return magnetic field in body frame [Tesla]
        """
        if self.use_igrf:
            # Placeholder - replace with actual pyIGRF call in real run
            # lat, lon, alt = ... convert from pos_eci
            B_eci = np.array([2e-5, 1e-5, -3e-5]) * np.sin(t/100)  # simulated variation
        else:
            # Tilted dipole approximation
            B_eci = np.array([0.0, 0.0, -3e-5])

        # Transform to body frame
        dcm = quat_to_dcm(q)
        B_body = dcm @ B_eci
        return B_body