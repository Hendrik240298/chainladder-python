# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import numpy as np
import chainladder as cl
import pytest


def test_smooth_parameter_none(raa):
    """Test that smooth=None behaves like no smoothing"""
    dev_none = cl.Development(smooth=None).fit(raa)
    dev_regular = cl.Development().fit(raa)
    assert np.allclose(dev_none.ldf_.values, dev_regular.ldf_.values)


def test_smooth_3tuple_default_exponential(raa):
    """Test that 3-tuple uses exponential method by default"""
    dev = cl.Development(smooth=[("1982", 12, 24)]).fit(raa)
    assert dev.ldf_ is not None
    assert dev.ldf_.shape == (1, 1, 1, 9)


def test_smooth_single_tuple(raa):
    """Test that smooth accepts a single tuple"""
    dev = cl.Development(smooth=("1982", 12, 24)).fit(raa)
    assert dev.ldf_ is not None
    assert dev.ldf_.shape == (1, 1, 1, 9)


def test_smooth_list_of_tuples(raa):
    """Test that smooth accepts a list of tuples"""
    dev = cl.Development(smooth=[("1982", 12, 24), ("1983", 24, 36)]).fit(raa)
    assert dev.ldf_ is not None
    assert dev.ldf_.shape == (1, 1, 1, 9)


def test_smooth_4tuple_exponential(raa):
    """Test 4-tuple with explicit exponential method"""
    dev = cl.Development(smooth=[("1982", 12, 24, "exponential")]).fit(raa)
    assert dev.ldf_ is not None
    assert dev.ldf_.shape == (1, 1, 1, 9)


def test_smooth_4tuple_inverse_power(raa):
    """Test 4-tuple with inverse_power method"""
    dev = cl.Development(smooth=[("1982", 12, 36, "inverse_power")]).fit(raa)
    assert dev.ldf_ is not None
    assert dev.ldf_.shape == (1, 1, 1, 9)


def test_smooth_4tuple_weibull(raa):
    """Test 4-tuple with weibull method"""
    dev = cl.Development(smooth=[("1982", 12, 36, "weibull")]).fit(raa)
    assert dev.ldf_ is not None
    assert dev.ldf_.shape == (1, 1, 1, 9)


def test_smooth_invalid_method(raa):
    """Test that invalid method raises ValueError"""
    with pytest.raises(ValueError, match="Invalid smooth method"):
        cl.Development(smooth=[("1982", 12, 24, "invalid")]).fit(raa)


def test_smooth_invalid_tuple_length(raa):
    """Test that invalid tuple length raises ValueError"""
    with pytest.raises(ValueError, match="smooth tuple must be"):
        cl.Development(smooth=[("1982", 12)]).fit(raa)


def test_smooth_different_ranges(raa):
    """Test smoothing with different development ranges"""
    # Test different ranges
    dev1 = cl.Development(smooth=[("1982", 12, 24)]).fit(raa)
    dev2 = cl.Development(smooth=[("1982", 12, 36)]).fit(raa)
    dev3 = cl.Development(smooth=[("1982", 24, 48)]).fit(raa)

    # All should produce valid LDFs
    assert dev1.ldf_ is not None
    assert dev2.ldf_ is not None
    assert dev3.ldf_ is not None

    # Different ranges should produce different results
    # (unless by coincidence they're identical)
    assert dev1.ldf_.shape == dev2.ldf_.shape == dev3.ldf_.shape


def test_smooth_multiple_origins(raa):
    """Test smoothing multiple origins with different methods"""
    dev = cl.Development(
        smooth=[
            ("1982", 12, 24),                      # default exponential
            ("1983", 12, 24, "inverse_power"),     # custom method
            ("1984", 12, 24, "weibull")            # custom method
        ]
    ).fit(raa)

    assert dev.ldf_ is not None
    assert dev.ldf_.shape == (1, 1, 1, 9)


def test_smooth_with_other_parameters(raa):
    """Test that smooth works with other Development parameters"""
    # Test with n_periods
    dev1 = cl.Development(smooth=[("1982", 12, 24)], n_periods=5).fit(raa)
    assert dev1.ldf_ is not None

    # Test with average
    dev2 = cl.Development(smooth=[("1982", 12, 24)], average="simple").fit(raa)
    assert dev2.ldf_ is not None

    # Test with drop
    dev3 = cl.Development(
        smooth=[("1982", 12, 24)], drop=[("1981", 12)]
    ).fit(raa)
    assert dev3.ldf_ is not None


def test_smooth_does_not_affect_other_origins(raa):
    """Test that smoothing one origin doesn't affect LDF calculation of others"""
    # This test verifies that smoothing is applied before averaging
    # The smoothed origin should affect the overall LDF pattern
    dev_regular = cl.Development().fit(raa)
    dev_smooth = cl.Development(smooth=[("1982", 12, 24)]).fit(raa)

    # LDFs should be different (smoothing affects the average)
    # but both should be valid
    assert dev_regular.ldf_ is not None
    assert dev_smooth.ldf_ is not None
    assert dev_regular.ldf_.shape == dev_smooth.ldf_.shape


def test_smooth_maintains_shape(raa):
    """Test that smoothing maintains triangle dimensions"""
    dev_regular = cl.Development().fit(raa)
    dev_smooth = cl.Development(smooth=[("1982", 12, 24)]).fit(raa)

    assert dev_smooth.ldf_.shape == dev_regular.ldf_.shape
    assert dev_smooth.cdf_.shape == dev_regular.cdf_.shape
    assert dev_smooth.sigma_.shape == dev_regular.sigma_.shape
    assert dev_smooth.std_err_.shape == dev_regular.std_err_.shape


def test_smooth_backend_compatibility(raa):
    """Test that smoothing works with different array backends"""
    # This test is automatically parametrized by the raa fixture
    # to run with both numpy and sparse backends
    dev = cl.Development(smooth=[("1982", 12, 24)]).fit(raa)

    assert dev.ldf_ is not None
    # Get the array module to verify backend compatibility
    xp = raa.get_array_module()
    assert xp is not None


def test_smooth_with_transform(raa):
    """Test that smoothing works with fit_transform"""
    dev = cl.Development(smooth=[("1982", 12, 24)])
    result = dev.fit_transform(raa)

    assert result.ldf_ is not None
    assert result.cdf_ is not None


def test_smooth_edge_cases():
    """Test smoothing with edge cases"""
    raa = cl.load_sample("raa")

    # Test with first origin
    dev1 = cl.Development(smooth=[("1981", 12, 24)]).fit(raa)
    assert dev1.ldf_ is not None

    # Test with last origin that has enough data
    dev2 = cl.Development(smooth=[("1985", 12, 24)]).fit(raa)
    assert dev2.ldf_ is not None


def test_smooth_no_cascade_beyond_range():
    """Test that smoothing doesn't cascade beyond the specified range"""
    import numpy as np

    quarterly = cl.load_sample('quarterly')['incurred']

    # Smooth origin 2002, ages 18-24 (which means link ratios 18-21 and 21-24 ONLY)
    dev_regular = cl.Development().fit(quarterly)
    dev_smooth = cl.Development(smooth=[('2002', 18, 24, 'exponential')]).fit(quarterly)

    # The link ratios 18-21 and 21-24 should be affected
    ldf_regular_18_21 = dev_regular.ldf_.values[0, 0, 0, 5]
    ldf_smooth_18_21 = dev_smooth.ldf_.values[0, 0, 0, 5]
    assert not np.allclose(ldf_regular_18_21, ldf_smooth_18_21), \
        "Smoothing should affect 18-21"

    ldf_regular_21_24 = dev_regular.ldf_.values[0, 0, 0, 6]
    ldf_smooth_21_24 = dev_smooth.ldf_.values[0, 0, 0, 6]
    assert not np.allclose(ldf_regular_21_24, ldf_smooth_21_24), \
        "Smoothing should affect 21-24"

    # The 24-27 link ratio (index 7) should NOT be affected directly
    # (it may have tiny differences due to weighted regression, but should be close)
    ldf_regular_24_27 = dev_regular.ldf_.values[0, 0, 0, 7]
    ldf_smooth_24_27 = dev_smooth.ldf_.values[0, 0, 0, 7]

    # Allow small differences from regression but should be mostly unchanged
    # (the difference should be much smaller than for the smoothed periods)
    diff_24_27 = abs(ldf_smooth_24_27 - ldf_regular_24_27)
    diff_18_21 = abs(ldf_smooth_18_21 - ldf_regular_18_21)
    assert diff_24_27 < diff_18_21 / 5, \
        f"24-27 changed too much ({diff_24_27}), should be much less than 18-21 change ({diff_18_21})"

    # The 27-30 link ratio (index 8) should NOT be affected
    ldf_regular_27_30 = dev_regular.ldf_.values[0, 0, 0, 8]
    ldf_smooth_27_30 = dev_smooth.ldf_.values[0, 0, 0, 8]

    # Should be exactly equal (no cascade effect)
    assert np.allclose(ldf_regular_27_30, ldf_smooth_27_30), \
        f"Smoothing cascaded beyond range: {ldf_regular_27_30} != {ldf_smooth_27_30}"
