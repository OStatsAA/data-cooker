"""Functional tests to validate generated data"""

from scipy.stats import norm
from statsmodels.tsa.arima.model import ARIMA
from datacooker.recipes import AutoRegressiveRecipe
from datacooker.variables import ContinousVariable

def test_simple_ar_1_model() -> None:
    """
    Testing AR(1) ignoring exogenous variable x.
    """
    ar_l1 = 0.75
    recipe = AutoRegressiveRecipe(lambda variables, error: 0 + error, [ar_l1])
    recipe.add_variable(ContinousVariable("x"))
    recipe.add_error(lambda variables, size: norm().rvs(size=size))
    data = recipe.cook(size=500)

    model = ARIMA(data['result'], order=(1, 0, 0)).fit()
    ar_l1_min, ar_l1_max = model.conf_int().loc['ar.L1']

    assert ar_l1_min <= ar_l1 <= ar_l1_max
