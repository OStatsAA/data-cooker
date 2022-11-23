# pylint: disable=no-member
# pylint: disable=invalid-name
"""
PoissonRecipe module
"""

import numpy as np
from .recipe import Recipe


class PoissonRecipe(Recipe):
    """
    Recipe for cooking a dataset that may be modeled by a GLM using Poisson family
    """

    def _Recipe__apply_model(self, error_values):
        z_values = self._Recipe__model(self._Recipe__data, error_values)
        lambda_param = np.exp(z_values)
        return np.random.poisson(lambda_param)
