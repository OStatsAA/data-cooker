"""
Variables error messages testing module
"""

from scipy.stats import norm, poisson

from datacooker.variables.error_messages import invalid_distribution_type_msg
from datacooker.variables.error_messages import invalid_missing_values_fraction_msg


def test_if_missing_values_fraction_msg_contains_fraction_and_label_info() -> None:
    """
    Tests if error message contains variable label,
    the fraction argument and fraction type
    """
    label = "Test label"
    fraction = 1
    message = invalid_missing_values_fraction_msg(fraction, label)
    assert label in message
    assert str(fraction) in message
    assert str(type(fraction)) in message


def test_if_invalid_distribution_type_msg_contains_expected_and_assigned_info() -> None:
    """
    Tests if error message contains variable label,
    assigned and expected distribution types (scipy's rv_continous x rv_discrete)
    """
    label = "Test label"
    assigned = poisson(1)
    expected = norm(1)
    message = invalid_distribution_type_msg(assigned, expected, label)
    assert label in message
    assert str(type(assigned)) in message
    assert str(type(expected)) in message
