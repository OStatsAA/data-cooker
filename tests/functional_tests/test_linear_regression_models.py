"""Functional tests to validate generated data"""

import math
from scipy.stats import norm
from statsmodels.regression.linear_model import OLS, RegressionResults
from datacooker.recipe import Recipe
from datacooker.variables import ContinousVariable


def test_linear_regression_one_var() -> None:
    """
    Adding one continous variable with no intercept and no error
    should result in a dataset that fitted to a linear regression
    model, should have coeficient = 1 for 'x' and pvalue = 0.
    """
    recipe = Recipe(lambda variables, error: 0 + variables["x"])
    recipe.add_variable(ContinousVariable("x"))
    data = recipe.cook()
    model = OLS(data['result'], data['x'])
    results: RegressionResults = model.fit()
    x_coef: float = results.params['x']
    x_coef_pvalue: float = results.pvalues['x']

    assert math.isclose(x_coef, 1)
    assert math.isclose(x_coef_pvalue, 0)


def test_linear_regression_one_var_with_error_component() -> None:
    """
    Adding one continous variable with error component
    should result in a dataset that fitted to a linear regression
    model, should have a coeficient with standard error greater than 0.0001
    """
    recipe = Recipe(lambda variables, error: 0 + variables["x"] + error)
    recipe.add_variable(ContinousVariable("x"))
    recipe.add_error(lambda variables, size: norm().rvs(size=size))
    data = recipe.cook()
    model = OLS(data['result'], data['x'])
    results: RegressionResults = model.fit()
    coef_std_error = math.sqrt(results.cov_params()['x'])

    assert coef_std_error > 0.0001
