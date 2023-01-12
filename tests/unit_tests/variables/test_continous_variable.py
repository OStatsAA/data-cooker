"""ContinousVariable tests module"""

import numpy as np
from datacooker.variables.continous_variable import ContinousVariable


def test_should_generate_values_from_linspace_like_functions() -> None:
    """
    Tests if ContinousVariable generates values from a linspace-like function
    """
    start = 1
    end = 10
    size = 50
    variable = ContinousVariable('x', (np.linspace, start, end))
    data = variable.simulate_values(size)

    assert len(data) == size
    assert data.min() == start
    assert data.max() == end
