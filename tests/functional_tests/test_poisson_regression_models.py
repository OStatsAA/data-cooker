"""
Functional tests to validate generated data follow a poisson regression model
"""

from statsmodels.api import GLM, families
from statsmodels.genmod.generalized_linear_model import GLMResults
from datacooker.recipes import PoissonRecipe
from datacooker.variables import ContinousVariable


def test_poisson_regression_one_var() -> None:
    """
    PoissonRecipe should be able to generate a dataset that may be fitted to a \
    poisson regression. The model function is the linear combination of \
    variables.
    """
    variable_true_coefficient = 10
    recipe = PoissonRecipe(lambda variables, error: 0 + variable_true_coefficient * variables["x"])

    recipe.add_variable(ContinousVariable("x"))
    data = recipe.cook()

    poisson_family = families.Poisson()
    results: GLMResults = GLM(data['result'], data['x'], poisson_family).fit()

    coef_int = results.conf_int().values[0]
    assert coef_int[0] < variable_true_coefficient < coef_int[1]
    assert results.pvalues.values[0] < .05
