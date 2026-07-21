"""
models/environment/gravity_gradient.py

Gravity Gradient Torque Model (Mandatory)
"""

import numpy as np


def gravity_gradient_torque(q: np.ndarray, J: np.ndarray, orbit_radius: float) -> np.ndarray:
    """
    Compute gravity gradient torque in body frame.
    
    T_gg = 3*mu / R^3 * (r_hat × J r_hat)
    """
    mu = 3.986004418e14
    R = orbit_radius

    # r_hat in body frame = DCM^T * [1,0,0] (nadir pointing assumption)
    dcm = quat_to_dcm(q)                    # Need import from utils
    r_hat_body = dcm.T @ np.array([1.0, 0.0, 0.0])

    # Compute torque
    J_r = J @ r_hat_body
    T_gg = (3 * mu / R**3) * np.cross(r_hat_body, J_r)

    return T_gg