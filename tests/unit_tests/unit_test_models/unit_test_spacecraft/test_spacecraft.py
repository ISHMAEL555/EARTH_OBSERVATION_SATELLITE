"""
Unit tests for spacecraft.py
"""

import numpy as np
import pytest

from models.spacecraft import Spacecraft


# ==========================================================
# Fixtures
# ==========================================================

@pytest.fixture
def spacecraft():

    return Spacecraft(
        inertia=np.diag([10.0, 12.0, 15.0]),
        mass=20.0,
    )


# ==========================================================
# Initialization
# ==========================================================

def test_default_initialization(spacecraft):

    q, omega = spacecraft.get_state()

    assert np.allclose(
        q,
        np.array([1.0, 0.0, 0.0, 0.0]),
    )

    assert np.allclose(
        omega,
        np.zeros(3),
    )


def test_custom_initialization():

    spacecraft = Spacecraft(
        inertia=np.diag([5.0, 6.0, 7.0]),
        mass=15.0,
        q0=np.array(
            [0.70710678, 0.70710678, 0.0, 0.0]
        ),
        omega0=np.array(
            [0.1, 0.2, 0.3]
        ),
    )

    q, omega = spacecraft.get_state()

    assert np.isclose(
        np.linalg.norm(q),
        1.0,
    )

    assert np.allclose(
        omega,
        np.array([0.1, 0.2, 0.3]),
    )


def test_invalid_mass():

    with pytest.raises(ValueError):

        Spacecraft(
            inertia=np.eye(3),
            mass=0.0,
        )


def test_invalid_inertia():

    with pytest.raises(ValueError):

        Spacecraft(
            inertia=np.eye(2),
            mass=10.0,
        )


def test_invalid_quaternion_shape():

    with pytest.raises(ValueError):

        Spacecraft(
            inertia=np.eye(3),
            mass=10.0,
            q0=np.ones(3),
        )


def test_zero_quaternion():

    with pytest.raises(ValueError):

        Spacecraft(
            inertia=np.eye(3),
            mass=10.0,
            q0=np.zeros(4),
        )


def test_invalid_omega_shape():

    with pytest.raises(ValueError):

        Spacecraft(
            inertia=np.eye(3),
            mass=10.0,
            omega0=np.ones(2),
        )
# ==========================================================
# Propagation
# ==========================================================

def test_propagation_updates_state(spacecraft):
    """
    Applying a non-zero torque should update the spacecraft state.
    """

    q_before, omega_before = spacecraft.get_state()

    spacecraft.propagate(
        total_torque=np.array([0.5, 0.0, 0.0]),
        dt=0.1,
    )

    q_after, omega_after = spacecraft.get_state()

    assert not np.allclose(
        omega_before,
        omega_after,
    )

    assert not np.allclose(
        q_before,
        q_after,
    )

    assert np.isclose(
        np.linalg.norm(q_after),
        1.0,
        atol=1e-12,
    )


def test_invalid_timestep(spacecraft):
    """
    Time step must be positive.
    """

    with pytest.raises(ValueError):

        spacecraft.propagate(
            total_torque=np.zeros(3),
            dt=0.0,
        )


def test_negative_timestep(spacecraft):
    """
    Negative integration step should raise an exception.
    """

    with pytest.raises(ValueError):

        spacecraft.propagate(
            total_torque=np.zeros(3),
            dt=-0.01,
        )


def test_zero_state_remains_zero(spacecraft):
    """
    Zero torque with zero angular velocity should leave the
    spacecraft unchanged.
    """

    q_before, omega_before = spacecraft.get_state()

    spacecraft.propagate(
        total_torque=np.zeros(3),
        dt=0.1,
    )

    q_after, omega_after = spacecraft.get_state()

    assert np.allclose(
        q_before,
        q_after,
    )

    assert np.allclose(
        omega_before,
        omega_after,
    )


def test_zero_torque_with_initial_rotation(spacecraft):
    """
    Torque-free propagation should preserve a valid state.
    """

    spacecraft.set_state(
        q=np.array([1.0, 0.0, 0.0, 0.0]),
        omega=np.array([0.1, 0.2, 0.3]),
    )

    spacecraft.propagate(
        total_torque=np.zeros(3),
        dt=0.1,
    )

    q, omega = spacecraft.get_state()

    assert np.all(
        np.isfinite(q)
    )

    assert np.all(
        np.isfinite(omega)
    )

    assert np.isclose(
        np.linalg.norm(q),
        1.0,
        atol=1e-12,
    )


def test_long_duration_zero_torque(spacecraft):
    """
    Long-duration torque-free propagation should remain stable.
    """

    spacecraft.set_state(
        q=np.array([1.0, 0.0, 0.0, 0.0]),
        omega=np.array([0.1, 0.2, 0.3]),
    )

    for _ in range(1000):

        spacecraft.propagate(
            total_torque=np.zeros(3),
            dt=0.01,
        )

    q, omega = spacecraft.get_state()

    assert np.all(
        np.isfinite(q)
    )

    assert np.all(
        np.isfinite(omega)
    )

    assert np.isclose(
        np.linalg.norm(q),
        1.0,
        atol=1e-10,
    )


def test_long_duration_constant_torque(spacecraft):
    """
    Constant applied torque should not produce numerical instability.
    """

    for _ in range(1000):

        spacecraft.propagate(
            total_torque=np.array(
                [0.01, 0.02, 0.03]
            ),
            dt=0.01,
        )

    q, omega = spacecraft.get_state()

    assert np.all(
        np.isfinite(q)
    )

    assert np.all(
        np.isfinite(omega)
    )

    assert np.isclose(
        np.linalg.norm(q),
        1.0,
        atol=1e-10,
    )


def test_quaternion_remains_normalized(spacecraft):
    """
    Quaternion should remain normalized after repeated propagation.
    """

    for _ in range(500):

        spacecraft.propagate(
            total_torque=np.array(
                [0.05, -0.02, 0.01]
            ),
            dt=0.02,
        )

    q, _ = spacecraft.get_state()

    assert np.isclose(
        np.linalg.norm(q),
        1.0,
        atol=1e-10,
    )


def test_state_remains_finite(spacecraft):
    """
    Spacecraft state should remain finite during long simulation.
    """

    for _ in range(2000):

        spacecraft.propagate(
            total_torque=np.array(
                [0.03, -0.01, 0.02]
            ),
            dt=0.005,
        )

    q, omega = spacecraft.get_state()

    assert np.all(
        np.isfinite(q)
    )

    assert np.all(
        np.isfinite(omega)
    )

# ==========================================================
# State Management
# ==========================================================

def test_get_state(spacecraft):
    """
    get_state() should return the current spacecraft state.
    """

    q, omega = spacecraft.get_state()

    assert np.allclose(
        q,
        np.array([1.0, 0.0, 0.0, 0.0]),
    )

    assert np.allclose(
        omega,
        np.zeros(3),
    )


def test_set_state(spacecraft):
    """
    set_state() should update the spacecraft state.
    """

    q = np.array(
        [0.5, 0.5, 0.5, 0.5]
    )

    omega = np.array(
        [1.0, 2.0, 3.0]
    )

    spacecraft.set_state(
        q,
        omega,
    )

    q_new, omega_new = spacecraft.get_state()

    assert np.isclose(
        np.linalg.norm(q_new),
        1.0,
    )

    assert np.allclose(
        omega_new,
        omega,
    )


def test_reset(spacecraft):
    """
    reset() should restore the default spacecraft state.
    """

    spacecraft.set_state(
        q=np.array(
            [0.5, 0.5, 0.5, 0.5]
        ),
        omega=np.array(
            [1.0, 2.0, 3.0]
        ),
    )

    spacecraft.reset()

    q, omega = spacecraft.get_state()

    assert np.allclose(
        q,
        np.array([1.0, 0.0, 0.0, 0.0]),
    )

    assert np.allclose(
        omega,
        np.zeros(3),
    )


def test_multiple_reset_calls(spacecraft):
    """
    Calling reset() multiple times should always
    restore the default state.
    """

    spacecraft.set_state(
        np.array([0.5, 0.5, 0.5, 0.5]),
        np.array([1.0, 2.0, 3.0]),
    )

    spacecraft.reset()
    spacecraft.reset()

    q, omega = spacecraft.get_state()

    assert np.allclose(
        q,
        np.array([1.0, 0.0, 0.0, 0.0]),
    )

    assert np.allclose(
        omega,
        np.zeros(3),
    )


# ==========================================================
# Copy
# ==========================================================

def test_copy(spacecraft):
    """
    copy() should return a new spacecraft object with
    identical state.
    """

    spacecraft_copy = spacecraft.copy()

    assert spacecraft_copy is not spacecraft

    q1, omega1 = spacecraft.get_state()
    q2, omega2 = spacecraft_copy.get_state()

    assert np.allclose(
        q1,
        q2,
    )

    assert np.allclose(
        omega1,
        omega2,
    )


def test_copy_is_independent(spacecraft):
    """
    Deep copy should be independent from the original object.
    """

    spacecraft_copy = spacecraft.copy()

    spacecraft_copy.set_state(
        np.array([0.0, 1.0, 0.0, 0.0]),
        np.array([1.0, 2.0, 3.0]),
    )

    q1, omega1 = spacecraft.get_state()
    q2, omega2 = spacecraft_copy.get_state()

    assert not np.allclose(
        q1,
        q2,
    )

    assert not np.allclose(
        omega1,
        omega2,
    )


# ==========================================================
# Validation
# ==========================================================

def test_invalid_set_state_quaternion(spacecraft):
    """
    Quaternion must have four elements.
    """

    with pytest.raises(ValueError):

        spacecraft.set_state(
            np.ones(3),
            np.zeros(3),
        )


def test_invalid_set_state_zero_quaternion(spacecraft):
    """
    Zero quaternion is not physically valid.
    """

    with pytest.raises(ValueError):

        spacecraft.set_state(
            np.zeros(4),
            np.zeros(3),
        )


def test_invalid_set_state_omega(spacecraft):
    """
    Angular velocity must have three elements.
    """

    with pytest.raises(ValueError):

        spacecraft.set_state(
            np.array(
                [1.0, 0.0, 0.0, 0.0]
            ),
            np.ones(2),
        )


# ==========================================================
# Robustness
# ==========================================================

def test_get_state_returns_copy(spacecraft):
    """
    get_state() should return copies instead of references.
    """

    q, omega = spacecraft.get_state()

    q[0] = 999.0
    omega[0] = 999.0

    q_internal, omega_internal = spacecraft.get_state()

    assert q_internal[0] != 999.0
    assert omega_internal[0] != 999.0


def test_set_state_copies_arrays(spacecraft):
    """
    set_state() should copy the supplied arrays.
    """

    q = np.array(
        [1.0, 0.0, 0.0, 0.0]
    )

    omega = np.array(
        [1.0, 2.0, 3.0]
    )

    spacecraft.set_state(
        q,
        omega,
    )

    q[:] = 0.0
    omega[:] = 0.0

    q_internal, omega_internal = spacecraft.get_state()

    assert not np.allclose(
        q_internal,
        np.zeros(4),
    )

    assert not np.allclose(
        omega_internal,
        np.zeros(3),
    )

# ==========================================================
# Properties
# ==========================================================

def test_inertia_property(spacecraft):
    """
    inertia property should return the spacecraft inertia matrix.
    """

    assert np.allclose(
        spacecraft.inertia,
        np.diag([10.0, 12.0, 15.0]),
    )


def test_inertia_property_returns_copy(spacecraft):
    """
    inertia property should return a copy instead of a reference.
    """

    J = spacecraft.inertia

    J[0, 0] = 999.0

    assert spacecraft.inertia[0, 0] != 999.0


def test_quaternion_property(spacecraft):
    """
    quaternion property should return the current attitude.
    """

    q = spacecraft.quaternion

    assert np.allclose(
        q,
        np.array([1.0, 0.0, 0.0, 0.0]),
    )


def test_quaternion_property_returns_copy(spacecraft):
    """
    quaternion property should return a copy.
    """

    q = spacecraft.quaternion

    q[0] = 999.0

    assert spacecraft.quaternion[0] != 999.0


def test_angular_velocity_property(spacecraft):
    """
    angular_velocity property should return the current body rate.
    """

    spacecraft.set_state(
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([1.0, 2.0, 3.0]),
    )

    assert np.allclose(
        spacecraft.angular_velocity,
        np.array([1.0, 2.0, 3.0]),
    )


def test_angular_velocity_property_returns_copy(spacecraft):
    """
    angular_velocity property should return a copy.
    """

    omega = spacecraft.angular_velocity

    omega[0] = 999.0

    assert spacecraft.angular_velocity[0] != 999.0


# ==========================================================
# Direction Cosine Matrix
# ==========================================================

def test_body_to_eci_dcm_identity(spacecraft):
    """
    Identity quaternion should produce an identity DCM.
    """

    C = spacecraft.body_to_eci_dcm

    assert np.allclose(
        C,
        np.eye(3),
    )


def test_body_to_eci_dcm_is_rotation_matrix(spacecraft):
    """
    DCM should remain orthogonal.
    """

    spacecraft.set_state(
        np.array([0.5, 0.5, 0.5, 0.5]),
        np.zeros(3),
    )

    C = spacecraft.body_to_eci_dcm

    assert np.allclose(
        C @ C.T,
        np.eye(3),
        atol=1e-12,
    )

    assert np.isclose(
        np.linalg.det(C),
        1.0,
        atol=1e-12,
    )


def test_eci_to_body_is_inverse(spacecraft):
    """
    ECI-to-body DCM should be the inverse of body-to-ECI.
    """

    spacecraft.set_state(
        np.array([0.5, 0.5, 0.5, 0.5]),
        np.zeros(3),
    )

    C_BE = spacecraft.body_to_eci_dcm
    C_EB = spacecraft.eci_to_body_dcm

    assert np.allclose(
        C_BE @ C_EB,
        np.eye(3),
        atol=1e-12,
    )

    assert np.allclose(
        C_EB @ C_BE,
        np.eye(3),
        atol=1e-12,
    )


# ==========================================================
# Physics Properties
# ==========================================================

def test_angular_momentum_property(spacecraft):
    """
    Angular momentum should equal Jω.
    """

    omega = np.array([1.0, 2.0, 3.0])

    spacecraft.set_state(
        np.array([1.0, 0.0, 0.0, 0.0]),
        omega,
    )

    H = spacecraft.angular_momentum

    assert np.allclose(
        H,
        spacecraft.inertia @ omega,
    )


def test_zero_angular_velocity_has_zero_momentum(spacecraft):
    """
    Zero angular velocity should produce zero angular momentum.
    """

    assert np.allclose(
        spacecraft.angular_momentum,
        np.zeros(3),
    )


def test_rotational_kinetic_energy_positive(spacecraft):
    """
    Rotational kinetic energy should be positive.
    """

    spacecraft.set_state(
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([1.0, 2.0, 3.0]),
    )

    assert spacecraft.rotational_kinetic_energy > 0.0


def test_zero_angular_velocity_has_zero_energy(spacecraft):
    """
    Zero angular velocity should produce zero kinetic energy.
    """

    assert np.isclose(
        spacecraft.rotational_kinetic_energy,
        0.0,
    )


# ==========================================================
# Consistency
# ==========================================================

def test_quaternion_norm_after_reset(spacecraft):
    """
    Quaternion should remain normalized after reset.
    """

    spacecraft.reset()

    assert np.isclose(
        np.linalg.norm(spacecraft.quaternion),
        1.0,
    )


def test_multiple_property_access(spacecraft):
    """
    Repeated property access should not modify the spacecraft state.
    """

    q_before, omega_before = spacecraft.get_state()

    for _ in range(100):

        _ = spacecraft.quaternion
        _ = spacecraft.angular_velocity
        _ = spacecraft.inertia
        _ = spacecraft.body_to_eci_dcm
        _ = spacecraft.eci_to_body_dcm
        _ = spacecraft.angular_momentum
        _ = spacecraft.rotational_kinetic_energy

    q_after, omega_after = spacecraft.get_state()

    assert np.allclose(
        q_before,
        q_after,
    )

    assert np.allclose(
        omega_before,
        omega_after,
    )