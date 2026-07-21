"""
simulation/simulator.py

Main Closed-Loop ADCS Simulator
Integrates everything: spacecraft, actuators, controllers, environment.
"""

import numpy as np
from config import DT, J
from models.spacecraft import Spacecraft
from controllers.attitude_controller import AttitudeController
from controllers.torque_allocator import TorqueAllocator
from controllers.magnetic_dumping import MagneticDumper
from models.environment.orbit import Orbit
from models.environment.disturbances import Disturbances
from models.environment.magnetic_field import MagneticField
from simulation.integrator import rk4_step
from utils.quaternion import quat_to_dcm


class ADCSSimulator:
    """
    Integrated ADCS Simulation Engine
    """

    def __init__(self):
        self.sc = Spacecraft(J)
        self.controller = AttitudeController()
        self.allocator = TorqueAllocator()
        self.dumper = MagneticDumper()
        
        self.orbit = Orbit()
        self.disturbances = Disturbances(J)
        self.mag_field = MagneticField(use_igrf=True)
        
        self.t = 0.0
        self.history = []   # For post-processing

    def derivative(self, state, t):
        """Derivative function for integrator (if needed)"""
        # Currently using internal stepwise integration
        pass

    def run_scenario(self, q_target_func, duration: float, enable_dumping: bool = False):
        """
        Run a complete simulation scenario.
        
        q_target_func : function(t) -> desired quaternion
        """
        self.reset()
        results = {
            'time': [], 'q': [], 'omega': [], 'tau_cmd': [], 'h_total': [],
            'pointing_error': [], 'B_body': [], 'm_cmd': []
        }

        steps = int(duration / DT)

        for i in range(steps):
            self.t = i * DT

            # Get current state
            q_current = self.sc.q.copy()
            omega = self.sc.omega.copy()
            h_total = self.allocator.get_total_momentum()

            # Desired attitude
            q_target = q_target_func(self.t)

            # Attitude control torque
            tau_ctrl = self.controller.compute_torque(q_target, q_current, omega, DT)

            # Magnetic dumping (if enabled)
            tau_dump = np.zeros(3)
            if enable_dumping:
                pos_eci, _ = self.orbit.get_position_velocity(self.t)
                B_body = self.mag_field.get_B_body(pos_eci, q_current, self.t)
                tau_dump = self.dumper.get_torque(h_total, B_body)

            # Total torque command
            tau_cmd = tau_ctrl + tau_dump

            # Allocate to actuators
            tau_applied, _, _ = self.allocator.allocate(tau_cmd, DT)

            # Disturbances
            pos_eci, _ = self.orbit.get_position_velocity(self.t)
            tau_dist = self.disturbances.compute_total_disturbance(
                q_current, omega, pos_eci, self.t
            )

            # Total torque on spacecraft
            tau_total = tau_applied + tau_dist

            # Propagate spacecraft
            self.sc.dynamics(tau_total, DT)

            # Logging
            error_deg = np.rad2deg(np.linalg.norm(self.sc.get_attitude_error(q_target))) * 3.0  # 3σ approx

            results['time'].append(self.t)
            results['q'].append(self.sc.q.copy())
            results['omega'].append(omega.copy())
            results['tau_cmd'].append(tau_cmd.copy())
            results['h_total'].append(h_total.copy())
            results['pointing_error'].append(error_deg)

        return results

    def reset(self):
        """Reset all modules"""
        self.sc.reset()
        self.controller.reset()
        self.allocator.reset()
        self.dumper.reset()
        self.orbit.reset()
        self.t = 0.0