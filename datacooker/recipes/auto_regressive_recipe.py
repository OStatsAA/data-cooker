"""
AutoRegressionRecipe module
"""


from typing import Callable
import numpy as np
from datacooker.recipes import Recipe


class AutoRegressiveRecipe(Recipe):
    """
    Recipe for cooking a dataset generated by an auto regression AR(p) process.
    """

    def __init__(self,
                 model_function: Callable,
                 coefficients: list[float],
                 result_label: str = 'result') -> None:
        super().__init__(model_function, result_label)
        self.__coefficients: np.ndarray = np.array(coefficients)

    def _apply_model(self, error_values) -> np.ndarray:
        results = self._model(self._data, error_values)
        results_count = len(results)
        lags_count = len(self.__coefficients)
        flipped_coefs = np.flip(self.__coefficients)

        for i in range(lags_count, results_count):
            results[i] += np.sum(results[i - lags_count: i] * flipped_coefs)

        return results