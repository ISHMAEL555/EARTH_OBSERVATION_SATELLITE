"""
simulator/simulator.py

Generic simulation engine.

Responsible for advancing simulation time.

No mission-specific logic belongs here.
"""


class Simulator:
    """
    Generic simulation clock.

    Responsible only for:
    - advancing simulation time
    - counting simulation steps
    - determining when the simulation ends
    """

    def __init__(self, simulation):

        self.sim = simulation

        self.time = 0.0

        self.current_step = 0

    @property
    def finished(self):
        """
        Return True when the simulation reaches its final time.
        """

        return self.time >= self.sim.simulation_time

    def step(self):
        """
        Advance the simulation by one timestep.
        """

        self.time += self.sim.time_step

        self.current_step += 1

    def reset(self):
        """
        Reset simulator state.
        """

        self.time = 0.0

        self.current_step = 0