"""
is_valid_fraction testing module
"""

from math import inf
from random import random
import pytest

from data_cooker.utils.is_valid_fraction import is_valid_fraction


def test_passes_when_fraction_is_zero() -> None:
    """Zero is a valid fraction"""
    fraction = 0
    assert is_valid_fraction(fraction)


def test_fails_when_fraction_is_one() -> None:
    """One should not be considered a valid fraction"""
    fraction = 1
    assert not is_valid_fraction(fraction)


def test_passes_when_fraction_is_in_valid_range() -> None:
    """Should pass if number is in range [0,1)"""
    fraction = random()
    assert is_valid_fraction(fraction)


@pytest.mark.parametrize('fraction', [-inf, -1, 1, 1.1, inf])
def test_fails_when_fraction_is_out_of_range(fraction) -> None:
    """Should fail if number is out of range [0,1)"""
    assert not is_valid_fraction(fraction)


@pytest.mark.parametrize('fraction', ["0", "0.5", "1", True, object, [0]])
def test_fails_when_fraction_is_not_a_number(fraction) -> None:
    """Should fail if number is not a number"""
    assert not is_valid_fraction(fraction)
