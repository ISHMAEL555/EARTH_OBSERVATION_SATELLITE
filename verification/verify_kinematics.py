import numpy as np

from models.dynamics.attitude_kinematics import (
    quaternion_derivative,
    propagate_rk4,
)

from models.dynamics.quaternion import (
    quaternion_to_dcm,
)

q = np.array(
    [1.,0.,0.,0.]
)

omega = np.array(
    [0.,0.,1.]
)

print("="*60)
print("KINEMATICS VERIFICATION")
print("="*60)

qdot = quaternion_derivative(
    q,
    omega,
)

print("\nQuaternion Derivative")
print(qdot)

q_new = propagate_rk4(
    q,
    omega,
    0.01,
)

print("\nQuaternion After")
print(q_new)

print("\nDCM")
print(
    quaternion_to_dcm(
        q_new
    )
)