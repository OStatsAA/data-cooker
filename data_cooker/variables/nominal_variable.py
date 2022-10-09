"""
Discrete variable module
"""

import numpy as np

from data_cooker.utils import is_valid_fraction
from .variable import Variable
from .error_messages import invalid_missing_values_fraction_msg

CATEGORIES_PREFIX: str = "Cat"


class NominalVariable(Variable):
    """
    A nominal variable defined by a distribution
    """

    def __init__(self,
                 label: str,
                 categories_count: int,
                 missing_values_fraction: float = 0) -> None:

        self.__label: str = label
        self.__values = []
        self.__categories = self.__init_categories_options(categories_count)

        if not is_valid_fraction(missing_values_fraction):
            msg: str = invalid_missing_values_fraction_msg(
                missing_values_fraction, label)
            raise ValueError(msg)

        self.__missing_values_fraction: float = missing_values_fraction

    def __init_categories_options(self, categories_count) -> list:
        categories = np.arange(
            1, categories_count+1).astype(str)
        return [CATEGORIES_PREFIX +
                category for category in categories]

    @property
    def label(self) -> str:
        return self.__label

    @property
    def missing_values_fraction(self) -> float:
        return self.__missing_values_fraction

    @property
    def categories(self) -> list:
        """Get categories

        Returns:
            np.ndarray: categories
        """
        return self.__categories

    def simulate_values(self, size: int) -> np.ndarray[str]:
        self.__values = np.random.choice(self.__categories, size)
        return self.__values
