"""
controllers/torque_allocator.py

Hybrid torque allocation between RW Pyramid and VSCMG
"""

import numpy as np
from models.actuators.rw_pyramid import RWPyramid
from models.actuators.vscmg import VSCMG
from config import D_THRESHOLD


class TorqueAllocator:
    """
    Splits commanded torque between RW and VSCMG clusters
    """

    def __init__(self):
        self.rw = RWPyramid()
        self.vscmg = VSCMG()

    def allocate(self, tau_cmd: np.ndarray, dt: float):
        """
        Hybrid allocation with priority to VSCMG for high torque / agility
        """
        # Simple blending: VSCMG for primary torque, RW for storage & fine control
        tau_vscmg = self.vscmg.allocate_torque(tau_cmd)

        # Remaining torque to RW
        tau_remaining = tau_cmd - tau_vscmg
        tau_rw = self.rw.allocate_torque(tau_remaining)

        # Update momentum states
        self.rw.update_momentum(tau_rw, dt)

        # Total applied torque (actual)
        tau_total = tau_vscmg + tau_rw

        return tau_total, tau_rw, tau_vscmg

    def get_total_momentum(self) -> np.ndarray:
        """Combined RW + VSCMG momentum"""
        return self.rw.get_total_momentum() + self.vscmg.get_total_momentum()

    def reset(self):
        self.rw.reset()
        self.vscmg.reset()