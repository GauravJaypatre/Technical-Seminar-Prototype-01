import pytest
from src.stats_utils import sample_variance

def test_empty_list_raises():
    with pytest.raises(ValueError):
        sample_variance([])

def test_single_element_raises():
    # FAILS ON UNPATCHED CODE
    with pytest.raises(ValueError, match="at least 2"):
        sample_variance([10.0])

def test_unbiased_variance_calculation():
    # FAILS ON UNPATCHED CODE (returns 4.0 instead of 32/7 ≈ 4.5714)
    data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    assert pytest.approx(sample_variance(data), rel=1e-4) == 32.0 / 7.0
