"""Recipe module tests"""

import pytest
from data_cooker.recipe import Recipe
from data_cooker.variables import ContinousVariable, DiscreteVariable, NominalVariable


def test_recipe_accepts_one_independent_vars_only() -> None:
    """Recipe is valid even only a single indepedent variable in included"""
    size = 10
    variable = ContinousVariable("x")
    recipe = Recipe(lambda data, error: data['x'])
    recipe.add_variable(variable)
    data = recipe.cook(size)

    assert len(data) == size


def test_recipe_adds_many_variables_at_once() -> None:
    """Tests if variables added to recipe have columns in output dataframe"""
    var_a = ContinousVariable("a")
    var_b = ContinousVariable("b")
    recipe = Recipe(lambda vars, error: vars['a'] - vars['b'])
    recipe.add_variables([var_a, var_b])
    data = recipe.cook(10)
    expected_columns = ["a", "b", "result"]

    assert data.columns.to_list() == expected_columns


def test_throws_error_if_no_indepedent_variable_is_added() -> None:
    """Should throw an error if no indepedent variable is included"""
    recipe = Recipe(lambda data, error: data['x'])
    with pytest.raises(ValueError):
        assert recipe.cook()


def test_recipes_accepts_corr_variable() -> None:
    """Should have method to set a single correlated variable"""
    recipe = Recipe(lambda data, error: data['x'])
    recipe.add_variable(ContinousVariable("x"))
    recipe.add_corr_variable("corr_x", lambda data: data["x"] * .5)
    data = recipe.cook(10)
    expected_columns = ["x", "corr_x", "result"]

    assert data.columns.to_list() == expected_columns


def test_recipes_accepts_many_corr_variables() -> None:
    """Should have method to set many correlated variable"""
    recipe = Recipe(lambda data, error: data['x'])
    recipe.add_variable(ContinousVariable("x"))

    def corr_x1_fn(data):
        return data["x"] * .5

    def corr_x2_fn(data):
        return data["x"] * .25
    recipe.add_corr_variables(["corr_x1", "corr_x2"], [corr_x1_fn, corr_x2_fn])
    data = recipe.cook(10)
    expected_columns = ["x", "corr_x1", "corr_x2", "result"]

    assert data.columns.to_list() == expected_columns


def test_throw_error_if_corr_variables_labels_and_functions_lenths_are_diff() -> None:
    """
    Should accept many correlated variables only if length of labels match the length of functions
    """
    recipe = Recipe(lambda data, error: data['x'])
    recipe.add_variable(ContinousVariable("x"))

    def corr_x1_fn(data):
        return data["x"] * .5

    with pytest.raises(ValueError):
        assert recipe.add_corr_variables(["corr_x1", "corr_x2"], [corr_x1_fn])


def test_recipes_accepts_error_variable() -> None:
    """Should have method to set a single error variable"""
    recipe = Recipe(lambda data, error: data['x'])
    recipe.add_variable(DiscreteVariable("x"))
    recipe.add_error(lambda data, size: data["x"] * .5)
    data = recipe.cook(10)
    expected_columns = ["x", "result"]

    assert data.columns.to_list() == expected_columns


def test_recipe_should_apply_missing_entries_fraction() -> None:
    """
    Should apply each variable missing data fraction.
    Result should not be affected
    """
    size: int = 10
    fraction: float = .25
    recipe = Recipe(lambda data, error: 0 + data['x'])
    recipe.add_variable(ContinousVariable(
        "x", missing_values_fraction=fraction))
    data = recipe.cook(size)

    assert data['x'].count() == (size - int(fraction * size))
    assert data['result'].count() == size


def test_recipe_data_should_have_nominal_var_as_dummies() -> None:
    """
    Recipe should expand NominalVariable to dummy columns
    """
    recipe = Recipe(lambda data, error: 0 + data['x.Cat1'] + data['x.Cat2'])
    recipe.add_variable(NominalVariable('x', 2))
    data = recipe.cook(100)
    expected_columns = ["x.Cat1", "x.Cat2", "result"]

    assert data.columns.tolist() == expected_columns


def test_recipe_nominal_var_missing_data_whole_row() -> None:
    """
    Recipe should apply missing data to a NominalVariable to all columns in dummy set
    """
    recipe = Recipe(lambda data, error: 0 + data['x.Cat1'] + data['x.Cat2'])
    recipe.add_variable(NominalVariable('x', 2, missing_values_fraction=.5))
    data = recipe.cook(100)
    expected_values_count = [50, 50, 100]

    assert data.count().to_list() == expected_values_count
