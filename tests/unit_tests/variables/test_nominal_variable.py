"""NominalVariable tests module"""

import numpy as np
from data_cooker.variables.nominal_variable import NominalVariable


def test_should_generate_c_categories_n_times() -> None:
    """
    Tests if NominalVariable generates the correct number of categories
    and values size
    """
    categories_count = 5
    size = 50
    variable = NominalVariable('x', categories_count)
    data = variable.simulate_values(size)

    assert len(np.unique(data)) == categories_count
    assert len(data) == size
