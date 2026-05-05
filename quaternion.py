"""
utils/quaternion.py

Quaternion mathematical utilities.
Scalar-first convention: q = [w, x, y, z]
Verified correct - May 2026
"""

import numpy as np


def quat_mult(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions: q1 * q2"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.array([w, x, y, z])


def quat_conj(q: np.ndarray) -> np.ndarray:
    """Quaternion conjugate"""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_norm(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion to unit length"""
    q_norm = np.linalg.norm(q)
    if q_norm > 1e-12:
        return q / q_norm
    else:
        return np.array([1.0, 0.0, 0.0, 0.0])


def quat_to_dcm(q: np.ndarray) -> np.ndarray:
    """Quaternion to Direction Cosine Matrix (Body → Inertial)"""
    q = quat_norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*y**2 - 2*z**2,  2*x*y - 2*z*w,  2*x*z + 2*y*w],
        [2*x*y + 2*z*w,  1 - 2*x**2 - 2*z**2,  2*y*z - 2*x*w],
        [2*x*z - 2*y*w,  2*y*z + 2*x*w,  1 - 2*x**2 - 2*y**2]
    ])


def quat_error(q_target: np.ndarray, q_current: np.ndarray) -> np.ndarray:
    """Compute error quaternion: q_err = q_target * q_current^-1"""
    q_err = quat_mult(q_target, quat_conj(q_current))
    return quat_norm(q_err)


def mrp_to_quat(sigma: np.ndarray) -> np.ndarray:
    """Modified Rodrigues Parameters to Quaternion"""
    sigma2 = np.dot(sigma, sigma)
    denom = 1.0 + sigma2
    q = np.zeros(4)
    q[0] = (1.0 - sigma2) / denom
    q[1:] = 2.0 * sigma / denom
    return q


def quat_to_mrp(q: np.ndarray) -> np.ndarray:
    """Quaternion to Modified Rodrigues Parameters"""
    q = quat_norm(q)
    if q[0] < 0.0:
        q = -q
    return q[1:] / (1.0 + q[0])