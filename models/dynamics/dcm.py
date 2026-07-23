"""
dcm.py
======

Direction Cosine Matrix (DCM) utilities.

Convention
----------
- Passive rotations (Schaub & Junkins)
- Right-handed coordinate system
- Rotation matrices are orthogonal with determinant +1.
"""

import numpy as np


# =============================================================================
# Constants
# =============================================================================

EPS = 1.0e-12


# =============================================================================
# Valid Euler Sequences
# =============================================================================

SYMMETRIC_SEQUENCES = {
    (1, 2, 1), (1, 3, 1),
    (2, 1, 2), (2, 3, 2),
    (3, 1, 3), (3, 2, 3),
}

ASYMMETRIC_SEQUENCES = {
    (1, 2, 3), (1, 3, 2),
    (2, 1, 3), (2, 3, 1),
    (3, 1, 2), (3, 2, 1),
}

VALID_SEQUENCES = (
    SYMMETRIC_SEQUENCES |
    ASYMMETRIC_SEQUENCES
)


# =============================================================================
# Validation
# =============================================================================

def _validate_dcm(C):
    """
    Validate Direction Cosine Matrix.
    """

    C = np.asarray(
        C,
        dtype=float,
    )

    if C.shape != (3, 3):
        raise ValueError(
            "Direction Cosine Matrix must have shape (3,3)."
        )

    return C


# =============================================================================
# Elementary Rotations
# =============================================================================

def C1(angle):

    c = np.cos(angle)
    s = np.sin(angle)

    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, s],
            [0.0, -s, c],
        ],
        dtype=float,
    )


def C2(angle):

    c = np.cos(angle)
    s = np.sin(angle)

    return np.array(
        [
            [c, 0.0, -s],
            [0.0, 1.0, 0.0],
            [s, 0.0, c],
        ],
        dtype=float,
    )


def C3(angle):

    c = np.cos(angle)
    s = np.sin(angle)

    return np.array(
        [
            [c, s, 0.0],
            [-s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


ROTATION_MAP = {
    1: C1,
    2: C2,
    3: C3,
}


# =============================================================================
# Euler Angles -> DCM
# =============================================================================

def euler_to_dcm(
    sequence,
    euler_angles,
):
    """
    Convert Euler angles (deg) to Direction Cosine Matrix.
    """

    sequence = tuple(sequence)

    if sequence not in VALID_SEQUENCES:
        raise ValueError(
            f"Invalid Euler sequence {sequence}."
        )

    angles = np.asarray(
        euler_angles,
        dtype=float,
    )

    if angles.shape != (3,):
        raise ValueError(
            "Exactly three Euler angles are required."
        )

    angles = np.deg2rad(
        angles
    )

    BN = np.eye(3)

    for axis, angle in zip(
        sequence,
        angles,
    ):

        BN = (
            ROTATION_MAP[axis](angle)
            @ BN
        )

    return BN


# =============================================================================
# DCM Utilities
# =============================================================================

def transpose(C):
    """
    Transpose of a Direction Cosine Matrix.
    """

    C = _validate_dcm(C)

    return C.T


def inverse(C):
    """
    Inverse of a Direction Cosine Matrix.
    """

    C = _validate_dcm(C)

    return C.T


def determinant(C):
    """
    Determinant of a Direction Cosine Matrix.
    """

    C = _validate_dcm(C)

    return np.linalg.det(C)


def orthogonality_error(C):
    """
    Frobenius norm of

        C Cᵀ - I
    """

    C = _validate_dcm(C)

    return np.linalg.norm(
        C @ C.T - np.eye(3),
        ord="fro",
    )


def is_rotation_matrix(
    C,
    atol=1e-10,
):
    """
    Check if matrix is a valid rotation matrix.
    """

    C = _validate_dcm(C)

    orthogonal = np.allclose(
        C @ C.T,
        np.eye(3),
        atol=atol,
    )

    det = np.isclose(
        np.linalg.det(C),
        1.0,
        atol=atol,
    )

    return orthogonal and det