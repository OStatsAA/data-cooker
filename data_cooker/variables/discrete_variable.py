"""
Discrete variable module
"""

import numpy as np
from scipy.stats import rv_discrete, poisson

from data_cooker.utils import is_valid_fraction
from .variable import Variable
from .error_messages import invalid_missing_values_fraction_msg


class DiscreteVariable(Variable):
    """
    A discrete variable defined by a distribution
    """
    __values = []

    def __init__(self,
                 label: str,
                 distribution: rv_discrete = poisson(1),
                 missing_values_fraction: float = 0) -> None:

        self.__label: str = label
        self.__distribtuion: rv_discrete = distribution

        if not is_valid_fraction(missing_values_fraction):
            msg: str = invalid_missing_values_fraction_msg(
                missing_values_fraction, label)
            raise ValueError(msg)

        self.__missing_values_fraction: float = missing_values_fraction

    @property
    def label(self) -> str:
        return self.__label

    @property
    def missing_values_fraction(self) -> float:
        return self.__missing_values_fraction

    def simulate_values(self, size: int) -> np.ndarray:
        self.__values = self.__distribtuion.rvs(size=size)
        return self.__values
