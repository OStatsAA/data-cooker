"""Variable abstract class module"""

from abc import ABC, abstractmethod
import numpy as np


class Variable(ABC):
    """Variable base abstract class"""

    @property
    @abstractmethod
    def label(self) -> str:
        """Returns variable label

        Returns:
            str: variable label
        """

    @property
    @abstractmethod
    def missing_values_fraction(self) -> float:
        """Returns fraction of missing values

        Returns:
            float: fraction of missing variables
        """

    @abstractmethod
    def simulate_values(self, size: int) -> np.ndarray:
        """Runs simulation to generate values for the variable

        Args:
            size (int): number of entries (array length)

        Returns:
            np.ndarray: values in ndarray format
        """
