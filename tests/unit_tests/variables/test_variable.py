"""ContinousVariable tests module"""

import pytest
from datacooker.variables.continous_variable import ContinousVariable


def test_variable_should_not_accept_invalid_missing_values_fraction() -> None:
    """
    Tests if Variable raises an error for invalid missing values fraction
    """
    with pytest.raises(ValueError):
        assert ContinousVariable('x', missing_values_fraction=1.1)
