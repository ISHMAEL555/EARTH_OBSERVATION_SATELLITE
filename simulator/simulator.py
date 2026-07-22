"""
simulator/simulator.py

Generic simulation engine.

Responsible for advancing simulation time.

No mission-specific logic belongs here.
"""

import numpy as np


class Simulator:

    def __init__(self, simulation):

        self.sim = simulation

        self.time = 0.0

    @property
    def finished(self):

        return self.time >= self.sim.simulation_time

    def step(self):

        self.time += self.sim.time_step

    def reset(self):

        self.time = 0.0