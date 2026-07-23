import numpy as np
import pytest

from models.dynamics.dcm import (
    C1,
    C2,
    C3,
    euler_to_dcm,
    transpose,
    inverse,
    determinant,
    orthogonality_error,
    is_rotation_matrix,
)


# =============================================================================
# TC-001
# Function : C1()
# Requirement:
# Rotation matrix shall have dimensions (3,3).
# =============================================================================

def test_C1_dimensions():

    C = C1(np.deg2rad(30.0))

    assert C.shape == (3, 3)


# =============================================================================
# TC-002
# Function : C2()
# Requirement:
# Rotation matrix shall have dimensions (3,3).
# =============================================================================

def test_C2_dimensions():

    C = C2(np.deg2rad(45.0))

    assert C.shape == (3, 3)


# =============================================================================
# TC-003
# Function : C3()
# Requirement:
# Rotation matrix shall have dimensions (3,3).
# =============================================================================

def test_C3_dimensions():

    C = C3(np.deg2rad(60.0))

    assert C.shape == (3, 3)


# =============================================================================
# TC-004
# Function : euler_to_dcm()
# Requirement:
# Identity Euler angles shall generate the identity DCM.
# =============================================================================

def test_identity_euler_angles():

    C = euler_to_dcm(
        (3, 2, 1),
        [0.0, 0.0, 0.0],
    )

    assert np.allclose(
        C,
        np.eye(3),
        atol=1e-12,
    )


# =============================================================================
# TC-005
# Function : euler_to_dcm()
# Requirement:
# Generated DCM shall be orthogonal.
# =============================================================================

def test_dcm_orthogonality():

    C = euler_to_dcm(
        (3, 2, 1),
        [20.0, -15.0, 45.0],
    )

    assert np.allclose(
        C @ C.T,
        np.eye(3),
        atol=1e-12,
    )


# =============================================================================
# TC-006
# Function : determinant()
# Requirement:
# Rotation matrix determinant shall equal +1.
# =============================================================================

def test_determinant():

    C = euler_to_dcm(
        (3, 2, 1),
        [10.0, 20.0, 30.0],
    )

    assert np.isclose(
        determinant(C),
        1.0,
        atol=1e-12,
    )


# =============================================================================
# TC-007
# Function : transpose()
# Requirement:
# DCM transpose shall equal matrix transpose.
# =============================================================================

def test_transpose():

    C = euler_to_dcm(
        (3, 2, 1),
        [15.0, 5.0, -10.0],
    )

    assert np.allclose(
        transpose(C),
        C.T,
        atol=1e-12,
    )


# =============================================================================
# TC-008
# Function : inverse()
# Requirement:
# Inverse of a DCM shall equal its transpose.
# =============================================================================

def test_inverse_equals_transpose():

    C = euler_to_dcm(
        (3, 2, 1),
        [10.0, 25.0, -5.0],
    )

    assert np.allclose(
        inverse(C),
        C.T,
        atol=1e-12,
    )


# =============================================================================
# TC-009
# Function : orthogonality_error()
# Requirement:
# Orthogonality error shall be near zero.
# =============================================================================

def test_orthogonality_error():

    C = euler_to_dcm(
        (3, 2, 1),
        [40.0, 30.0, 20.0],
    )

    assert np.isclose(
        orthogonality_error(C),
        0.0,
        atol=1e-12,
    )


# =============================================================================
# TC-010
# Function : is_rotation_matrix()
# Requirement:
# Valid DCM shall return True.
# =============================================================================

def test_valid_rotation_matrix():

    C = euler_to_dcm(
        (3, 2, 1),
        [30.0, 20.0, 10.0],
    )

    assert is_rotation_matrix(C)


# =============================================================================
# TC-011
# Function : is_rotation_matrix()
# Requirement:
# Invalid matrix shall return False.
# =============================================================================

def test_invalid_rotation_matrix():

    C = np.eye(3)
    C[0, 0] = 2.0

    assert not is_rotation_matrix(C)


# =============================================================================
# TC-012
# Function : euler_to_dcm()
# Requirement:
# Invalid Euler sequence shall raise ValueError.
# =============================================================================

def test_invalid_sequence():

    with pytest.raises(ValueError):

        euler_to_dcm(
            (1, 1, 1),
            [10.0, 20.0, 30.0],
        )


# =============================================================================
# TC-013
# Function : euler_to_dcm()
# Requirement:
# Invalid Euler angle size shall raise ValueError.
# =============================================================================

def test_invalid_angle_size():

    with pytest.raises(ValueError):

        euler_to_dcm(
            (3, 2, 1),
            [10.0, 20.0],
        )


# =============================================================================
# TC-014
# Function : transpose()
# Requirement:
# Invalid DCM dimensions shall raise ValueError.
# =============================================================================

def test_invalid_transpose():

    with pytest.raises(ValueError):

        transpose(
            np.eye(2),
        )


# =============================================================================
# TC-015
# Function : inverse()
# Requirement:
# Invalid DCM dimensions shall raise ValueError.
# =============================================================================

def test_invalid_inverse():

    with pytest.raises(ValueError):

        inverse(
            np.eye(4),
        )


# =============================================================================
# TC-016
# Function : determinant()
# Requirement:
# Invalid DCM dimensions shall raise ValueError.
# =============================================================================

def test_invalid_determinant():

    with pytest.raises(ValueError):

        determinant(
            np.eye(2),
        )


# =============================================================================
# TC-017
# Function : orthogonality_error()
# Requirement:
# Invalid DCM dimensions shall raise ValueError.
# =============================================================================

def test_invalid_orthogonality_error():

    with pytest.raises(ValueError):

        orthogonality_error(
            np.eye(2),
        )


# =============================================================================
# TC-018
# Function : is_rotation_matrix()
# Requirement:
# Invalid DCM dimensions shall raise ValueError.
# =============================================================================

def test_invalid_rotation_matrix_size():

    with pytest.raises(ValueError):

        is_rotation_matrix(
            np.eye(4),
        )