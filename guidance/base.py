"""
guidance/base.py

Base class for all spacecraft guidance laws.
"""

from abc import ABC, abstractmethod


class Guidance(ABC):

    @abstractmethod
    def compute(
        self,
        position_eci,
        velocity_eci,
    ):
        """
        Compute reference attitude and angular velocity.
        """
        pass