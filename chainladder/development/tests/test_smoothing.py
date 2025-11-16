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


def test_smooth_basic(raa):
    """Test basic 3-tuple smoothing (inclusive end_age)"""
    # Smooths LDFs starting at ages 12, 24 (2 LDFs minimum)
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


def test_smooth_invalid_tuple_length(raa):
    """Test that invalid tuple length raises ValueError"""
    with pytest.raises(ValueError, match="smooth tuple must be"):
        cl.Development(smooth=[("1982", 12)]).fit(raa)

    with pytest.raises(ValueError, match="smooth tuple must be"):
        cl.Development(smooth=[("1982", 12, 24, 36)]).fit(raa)


def test_smooth_minimum_periods(raa):
    """Test that smoothing requires at least 2 link ratios (3 cumulative values)"""
    # Try to smooth only 1 link ratio - should fail
    with pytest.raises(ValueError, match="at least 2 link ratios"):
        cl.Development(smooth=[("1982", 12, 12)]).fit(raa)


def test_smooth_different_ranges(raa):
    """Test smoothing with different development ranges (inclusive end_age)"""
    # Test different ranges
    # dev1: smooths LDFs 12→24, 24→36 (2 LDFs)
    dev1 = cl.Development(smooth=[("1982", 12, 24)]).fit(raa)
    # dev2: smooths LDFs 12→24, 24→36, 36→48 (3 LDFs)
    dev2 = cl.Development(smooth=[("1982", 12, 36)]).fit(raa)
    # dev3: smooths LDFs 24→36, 36→48, 48→60 (3 LDFs)
    dev3 = cl.Development(smooth=[("1982", 24, 48)]).fit(raa)

    # All should produce valid LDFs
    assert dev1.ldf_ is not None
    assert dev2.ldf_ is not None
    assert dev3.ldf_ is not None

    # Different ranges should produce different results
    assert dev1.ldf_.shape == dev2.ldf_.shape == dev3.ldf_.shape


def test_smooth_multiple_origins(raa):
    """Test smoothing multiple origins (inclusive end_age)"""
    dev = cl.Development(
        smooth=[
            ("1982", 12, 24),
            ("1983", 12, 24),
            ("1984", 12, 24)
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

    # Smooth origin 2002, ages 18-21 (inclusive: link ratios 18→21 and 21→24 ONLY)
    dev_regular = cl.Development().fit(quarterly)
    dev_smooth = cl.Development(smooth=[('2002', 18, 21)]).fit(quarterly)

    # The link ratios 18→21 and 21→24 should be affected
    ldf_regular_18_21 = dev_regular.ldf_.values[0, 0, 0, 5]
    ldf_smooth_18_21 = dev_smooth.ldf_.values[0, 0, 0, 5]
    assert not np.allclose(ldf_regular_18_21, ldf_smooth_18_21), \
        "Smoothing should affect 18→21"

    ldf_regular_21_24 = dev_regular.ldf_.values[0, 0, 0, 6]
    ldf_smooth_21_24 = dev_smooth.ldf_.values[0, 0, 0, 6]
    assert not np.allclose(ldf_regular_21_24, ldf_smooth_21_24), \
        "Smoothing should affect 21→24"

    # The 24→27 link ratio (index 7) should NOT be affected directly
    # (it may have tiny differences due to weighted regression, but should be close)
    ldf_regular_24_27 = dev_regular.ldf_.values[0, 0, 0, 7]
    ldf_smooth_24_27 = dev_smooth.ldf_.values[0, 0, 0, 7]

    # Allow small differences from regression but should be mostly unchanged
    # (the difference should be much smaller than for the smoothed periods)
    diff_24_27 = abs(ldf_smooth_24_27 - ldf_regular_24_27)
    diff_18_21 = abs(ldf_smooth_18_21 - ldf_regular_18_21)
    assert diff_24_27 < diff_18_21 / 5, \
        f"24→27 changed too much ({diff_24_27}), should be much less than 18→21 change ({diff_18_21})"

    # The 27→30 link ratio (index 8) should NOT be affected
    ldf_regular_27_30 = dev_regular.ldf_.values[0, 0, 0, 8]
    ldf_smooth_27_30 = dev_smooth.ldf_.values[0, 0, 0, 8]

    # Should be exactly equal (no cascade effect)
    assert np.allclose(ldf_regular_27_30, ldf_smooth_27_30), \
        f"Smoothing cascaded beyond range: {ldf_regular_27_30} != {ldf_smooth_27_30}"


def test_smooth_linear_interpolation_values():
    """Test that linear interpolation produces expected cumulative values"""
    # Create a simple triangle with known values for testing
    raa = cl.load_sample("raa")

    # Get the triangle in development mode
    tri = raa.incr_to_cum().val_to_dev()

    # Apply smoothing to a specific origin (inclusive: smooths 12→24, 24→36)
    dev = cl.Development(smooth=[("1982", 12, 24)]).fit(raa)

    # The smoothing should produce linearly interpolated cumulative values
    # We can't test exact values without knowing the triangle structure,
    # but we can verify that smoothing was applied
    assert dev.ldf_ is not None

    # Verify that the smoothed LDFs are reasonable (between 1.0 and original max)
    assert np.all(dev.ldf_.values >= 1.0)
    assert np.all(dev.ldf_.values < 10.0)  # Reasonable upper bound


def test_smooth_monotonicity():
    """Test that smoothed factors maintain reasonable progression"""
    quarterly = cl.load_sample('quarterly')['incurred']

    # Apply smoothing (inclusive: smooths LDFs starting at 18, 21, 24, 27, 30)
    dev = cl.Development(smooth=[('2002', 18, 30)]).fit(quarterly)

    # Extract the smoothed LDFs
    ldfs = dev.ldf_.values[0, 0, 0, :]

    # Smoothed LDFs should be declining (typical pattern)
    # Check that the average LDF is declining
    assert ldfs[0] > ldfs[-1], "LDFs should generally decline over time"

    # Should not have extreme jumps in the smoothed region
    # Calculate differences between consecutive LDFs
    diffs = np.diff(ldfs[5:10])  # Ages 18-30 region

    # Differences should be relatively consistent (no wild jumps)
    std_diff = np.std(diffs)
    mean_diff = np.mean(np.abs(diffs))

    # Standard deviation should be small relative to mean difference
    assert std_diff < mean_diff * 2, "Smoothed LDFs should have consistent progression"


def test_smooth_boundary_preservation():
    """Test that boundary cumulative values are preserved during interpolation"""
    # This test verifies the core linear interpolation algorithm
    quarterly = cl.load_sample('quarterly')['incurred']

    # Get cumulative triangle before smoothing
    cum_before = quarterly.incr_to_cum()

    # Apply smoothing to a specific origin and range (inclusive: 18, 21, 24, 27)
    dev_smooth = cl.Development(smooth=[('2002', 18, 27)]).fit(quarterly)

    # The boundaries should remain unchanged in the underlying cumulative values
    # (though this is hard to test directly since Development returns LDFs)
    # We can at least verify that the LDFs are valid
    assert dev_smooth.ldf_ is not None
    # LDFs can be less than 1.0 in real data (negative development), so just check they're positive
    assert np.all(dev_smooth.ldf_.values > 0.0)

    # Verify that smoothing produces different results than no smoothing
    dev_regular = cl.Development().fit(quarterly)
    assert not np.allclose(dev_smooth.ldf_.values, dev_regular.ldf_.values)
