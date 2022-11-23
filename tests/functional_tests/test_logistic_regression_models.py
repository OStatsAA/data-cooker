"""
Functional tests to validate generated data follow a logistic regression model
"""

from statsmodels.api import GLM, families
from statsmodels.genmod.generalized_linear_model import GLMResults
from datacooker.recipes import LogitRecipe
from datacooker.variables import ContinousVariable


def test_logistic_regression_one_var() -> None:
    """
    LogitRecipe should be able to generate a dataset that may be fitted to a \
    logistic regression. The model function is the linear com bination of \
    variables. The LogitRecipe applies Logit fn to create resulting dichotomic values.
    """
    variable_true_coefficient = 10
    recipe = LogitRecipe(lambda variables, error: 0 +
                         variable_true_coefficient * variables["x"])

    recipe.add_variable(ContinousVariable("x"))
    data = recipe.cook()

    binomial_family = families.Binomial()
    results: GLMResults = GLM(data['result'], data['x'], binomial_family).fit()

    coef_int = results.conf_int().values[0]
    assert coef_int[0] < variable_true_coefficient < coef_int[1]
    assert results.pvalues.values[0] < .05
