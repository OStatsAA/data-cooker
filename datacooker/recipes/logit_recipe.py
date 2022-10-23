# pylint: disable=no-member
# pylint: disable=invalid-name
"""
LogitRecipe module
"""

import numpy as np
from .recipe import Recipe


class LogitRecipe(Recipe):
    """
    Recipe for cooking a dataset that may be modeled by a logistic regression
    """

    def _Recipe__apply_model(self, error_values):
        z_values = self._Recipe__model(self._Recipe__data, error_values)
        prob = 1 / (1 + np.exp(-z_values))
        return np.random.binomial(1, prob)
