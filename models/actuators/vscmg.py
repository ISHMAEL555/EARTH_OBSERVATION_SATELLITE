"""
models/actuators/vscmg.py

Variable-Speed Control Moment Gyro (VSCMG) Cluster
Implements hybrid steering law with singularity avoidance.
"""

import numpy as np


class VSCMG:
    """
    VSCMG Cluster Model with Mixture Steering Law (MP + Singularity Robust + Null Motion)
    """

    def __init__(self):
        # Effective capability from assignment document (used for limits)
        self.tau_max = np.array([0.031, 0.031, 0.041])   # Nm   [x, y, z]
        self.h_max   = np.array([0.049, 0.049, 0.098])   # Nms  [x, y, z]

        # Internal state - simplified 4-VSCMG pyramid (common configuration)
        self.gimbal_angles = np.zeros(4)      # radians
        self.wheel_speeds = np.zeros(4)       # rad/s (variable speed)

        # Singularity parameters
        self.d_threshold = 0.2
        self.sr_damping = 0.02
        self.null_gain = 0.5

    def singularity_measure(self, A: np.ndarray) -> float:
        """Compute singularity measure D"""
        ATA = A @ A.T
        if np.linalg.det(ATA) < 1e-12:
            return 0.0
        if np.linalg.cond(ATA) > 1e8:
            return 0.0
        return 1.0 / np.linalg.cond(ATA)          # Higher = better (away from singularity)

    def steering_matrix(self) -> np.ndarray:
        """
        Compute VSCMG Jacobian (steering matrix) A.
        Simplified model - returns effective 3x4 matrix.
        """
        # For simulation purposes, we use a time-varying but bounded steering matrix
        # In full implementation this would depend on gimbal angles
        s = np.sin(self.gimbal_angles)
        c = np.cos(self.gimbal_angles)

        # Simplified pyramid-like steering matrix (4 columns)
        A = np.array([
            [c[0], -s[1], -c[2],  s[3]],
            [s[0],  c[1], -s[2], -c[3]],
            [0.8,   0.8,   1.0,   1.0]   # z has higher authority
        ]) * 0.035  # scaled to roughly match given torque capability

        return A

    def allocate_torque(self, tau_cmd: np.ndarray) -> tuple:
        """
        Mixture Steering Law:
            - Far from singularity → Moore-Penrose + Null Motion
            - Near singularity   → Singularity Robust Inverse

        Returns
        -------
        tau_vscmg : np.ndarray (3,)   Actual torque delivered
        """
        A = self.steering_matrix()
        D = self.singularity_measure(A)

        # Mixture logic
        if D > self.d_threshold:
            # === Loop 1: Moore-Penrose Pseudo-Inverse ===
            A_pinv = np.linalg.pinv(A)
            gimbal_rates_cmd = A_pinv @ tau_cmd
        else:
            # === Loop 2: Singularity Robust Inverse ===
            ATA = A @ A.T
            ATA_reg = ATA + self.sr_damping * np.eye(3)
            A_pinv = A.T @ np.linalg.inv(ATA_reg)
            gimbal_rates_cmd = A_pinv @ tau_cmd

        # Apply null motion (gradient-based escape)
        if D < 0.5:
            null_motion = self.null_gain * (D - 0.3) * np.ones(4)
            gimbal_rates_cmd += null_motion

        # Compute delivered torque (with saturation)
        tau_vscmg = A @ gimbal_rates_cmd
        tau_vscmg = np.clip(tau_vscmg, -self.tau_max, self.tau_max)

        # Update internal states (simplified integration)
        self.gimbal_angles += gimbal_rates_cmd * 0.05   # dt assumed
        self.gimbal_angles = np.mod(self.gimbal_angles, 2*np.pi)

        return tau_vscmg

    def get_total_momentum(self) -> np.ndarray:
        """Return current VSCMG momentum contribution (simplified)"""
        # In full model this would be h = sum (wheel_momentum * direction)
        h = np.array([0.0, 0.0, 0.0])
        h[2] = np.sum(self.wheel_speeds) * 0.025   # higher z capability
        h[:2] = np.random.normal(0, 0.01, 2)       # small variation
        return np.clip(h, -self.h_max, self.h_max)

    def reset(self):
        """Reset VSCMG state"""
        self.gimbal_angles = np.zeros(4)
        self.wheel_speeds = np.zeros(4)