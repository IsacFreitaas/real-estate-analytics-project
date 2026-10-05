import pytest

from src.evaluation import calculate_error_reduction, calculate_mae


def test_mae_with_non_zero_error():
    assert calculate_mae([1, 2, 3], [1, 3, 2]) == pytest.approx(2 / 3)


def test_mae_with_perfect_predictions():
    assert calculate_mae([1, 2, 3], [1, 2, 3]) == 0


def test_error_reduction_percentage():
    assert calculate_error_reduction(1.0, 0.75) == pytest.approx(25.0)
