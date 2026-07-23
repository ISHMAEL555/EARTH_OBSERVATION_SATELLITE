"""
quaternion.py
=============

Quaternion mathematics utilities.

Convention
----------
- Scalar-first quaternion:
    q = [q0, q1, q2, q3]

- Hamilton quaternion multiplication

- Unit quaternions represent attitude.

References
----------
- Schaub & Junkins
- Markley & Crassidis
"""

import numpy as np


# =============================================================================
# Constants
# =============================================================================

EPS = 1.0e-12


# =============================================================================
# Validation
# =============================================================================

def _validate_quaternion(
    q,
):
    """
    Validate quaternion.

    Parameters
    ----------
    q : ndarray (4,)

    Returns
    -------
    ndarray (4,)
    """

    q = np.asarray(
        q,
        dtype=float,
    )

    if q.shape != (4,):
        raise ValueError(
            "Quaternion must have shape (4,)."
        )

    return q


# =============================================================================
# Quaternion Utilities
# =============================================================================

def normalize(
    q,
):
    """
    Normalize quaternion.
    """

    q = _validate_quaternion(q)

    norm = np.linalg.norm(q)

    if norm < EPS:
        raise ValueError(
            "Quaternion norm cannot be zero."
        )

    return q / norm


def enforce_unique(
    q,
):
    """
    Normalize quaternion and enforce a unique representation.

    Ensures

        q0 >= 0

    so that q and -q are not both used.
    """

    q = normalize(q)

    if q[0] < 0.0:
        q *= -1.0

    return q


def is_unit(
    q,
    atol=1e-10,
):
    """
    Check if quaternion is unit length.
    """

    q = _validate_quaternion(q)

    return np.isclose(
        np.linalg.norm(q),
        1.0,
        atol=atol,
    )


def equivalent(
    q1,
    q2,
    atol=1e-8,
):
    """
    Check quaternion equivalence.

    Two quaternions represent the same attitude if

        q1 ≈ q2

    or

        q1 ≈ -q2
    """

    q1 = normalize(q1)
    q2 = normalize(q2)

    return (
        np.allclose(
            q1,
            q2,
            atol=atol,
        )
        or
        np.allclose(
            q1,
            -q2,
            atol=atol,
        )
    )


# =============================================================================
# Quaternion Conjugate
# =============================================================================

def conjugate(
    q,
):
    """
    Quaternion conjugate.
    """

    q = _validate_quaternion(q)

    return np.array(
        [
            q[0],
            -q[1],
            -q[2],
            -q[3],
        ]
    )


def inverse(
    q,
):
    """
    Quaternion inverse.

    For unit quaternions

        q⁻¹ = q*
    """

    q = normalize(q)

    return conjugate(q)


# =============================================================================
# Hamilton Product
# =============================================================================

def multiply(
    q1,
    q2,
):
    """
    Hamilton quaternion multiplication.
    """

    q1 = _validate_quaternion(q1)
    q2 = _validate_quaternion(q2)

    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array(

        [

            w1*w2 - x1*x2 - y1*y2 - z1*z2,

            w1*x2 + x1*w2 + y1*z2 - z1*y2,

            w1*y2 - x1*z2 + y1*w2 + z1*x2,

            w1*z2 + x1*y2 - y1*x2 + z1*w2,

        ],

        dtype=float,

    )


# =============================================================================
# Quaternion → Direction Cosine Matrix
# =============================================================================

def quaternion_to_dcm(
    q,
):
    """
    Convert quaternion to Direction Cosine Matrix.
    """

    q = normalize(q)

    q0, q1, q2, q3 = q

    return np.array(

        [

            [
                q0**2 + q1**2 - q2**2 - q3**2,
                2.0*(q1*q2 + q0*q3),
                2.0*(q1*q3 - q0*q2),
            ],

            [
                2.0*(q1*q2 - q0*q3),
                q0**2 - q1**2 + q2**2 - q3**2,
                2.0*(q2*q3 + q0*q1),
            ],

            [
                2.0*(q1*q3 + q0*q2),
                2.0*(q2*q3 - q0*q1),
                q0**2 - q1**2 - q2**2 + q3**2,
            ],

        ],

        dtype=float,

    )


# =============================================================================
# Direction Cosine Matrix → Quaternion
# =============================================================================

def dcm_to_quaternion(
    C,
):
    """
    Convert Direction Cosine Matrix to quaternion.
    """

    C = np.asarray(
        C,
        dtype=float,
    )

    if C.shape != (3, 3):
        raise ValueError(
            "Direction Cosine Matrix must have shape (3,3)."
        )

    tr = np.trace(C)

    if tr > 0.0:

        s = 2.0 * np.sqrt(tr + 1.0)

        q0 = 0.25 * s
        q1 = (C[1, 2] - C[2, 1]) / s
        q2 = (C[2, 0] - C[0, 2]) / s
        q3 = (C[0, 1] - C[1, 0]) / s

    elif C[0, 0] > C[1, 1] and C[0, 0] > C[2, 2]:

        s = 2.0 * np.sqrt(
            1.0 + C[0, 0] - C[1, 1] - C[2, 2]
        )

        q0 = (C[1, 2] - C[2, 1]) / s
        q1 = 0.25 * s
        q2 = (C[0, 1] + C[1, 0]) / s
        q3 = (C[0, 2] + C[2, 0]) / s

    elif C[1, 1] > C[2, 2]:

        s = 2.0 * np.sqrt(
            1.0 + C[1, 1] - C[0, 0] - C[2, 2]
        )

        q0 = (C[2, 0] - C[0, 2]) / s
        q1 = (C[0, 1] + C[1, 0]) / s
        q2 = 0.25 * s
        q3 = (C[1, 2] + C[2, 1]) / s

    else:

        s = 2.0 * np.sqrt(
            1.0 + C[2, 2] - C[0, 0] - C[1, 1]
        )

        q0 = (C[0, 1] - C[1, 0]) / s
        q1 = (C[0, 2] + C[2, 0]) / s
        q2 = (C[1, 2] + C[2, 1]) / s
        q3 = 0.25 * s

    return enforce_unique(
        np.array(
            [
                q0,
                q1,
                q2,
                q3,
            ]
        )
    )