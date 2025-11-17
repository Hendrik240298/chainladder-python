"""
Test for monotonicity and stability issues in Weibull tail extrapolation
"""
import numpy as np
import chainladder as cl
import matplotlib.pyplot as plt

def test_weibull_monotonicity():
    """Test if Weibull tail factors are monotonically decreasing"""
    print("=" * 80)
    print("Testing Weibull Tail Factor Monotonicity")
    print("=" * 80)

    # Load sample data
    raa = cl.load_sample('raa')

    # Fit Weibull tail with extended extrapolation
    tail_weibull = cl.TailCurve(
        curve='weibull',
        fit_period=(12, None),
        extrap_periods=200
    ).fit(raa)

    print(f"\nSlope (b): {tail_weibull.slope_.values[0, 0]:.6f}")
    print(f"Intercept (log(a)): {tail_weibull.intercept_.values[0, 0]:.6f}")
    print(f"Scale parameter (a): {np.exp(tail_weibull.intercept_.values[0, 0]):.6f}")

    # Get LDF values
    ldf_values = tail_weibull.ldf_.values[0, 0, 0, :]
    ages = np.arange(1, len(ldf_values) + 1)

    print(f"\nTotal extrapolation periods: {len(ldf_values)}")
    print(f"\nFirst 20 LDF values:")
    for i in range(min(20, len(ldf_values))):
        print(f"  Period {i+1:3d}: LDF = {ldf_values[i]:.6f}")

    print(f"\nLast 20 LDF values:")
    for i in range(max(0, len(ldf_values) - 20), len(ldf_values)):
        print(f"  Period {i+1:3d}: LDF = {ldf_values[i]:.6f}")

    # Check for monotonicity
    print("\n" + "-" * 80)
    print("Checking for monotonicity violations...")
    print("-" * 80)

    diffs = np.diff(ldf_values)
    increases = diffs > 0
    if np.any(increases):
        print(f"\nWARNING: Found {np.sum(increases)} periods where LDF increases!")
        increase_indices = np.where(increases)[0]
        print(f"\nFirst 10 violations:")
        for idx in increase_indices[:10]:
            print(f"  Period {idx+1} -> {idx+2}: {ldf_values[idx]:.8f} -> {ldf_values[idx+1]:.8f} (diff: {diffs[idx]:.8f})")
    else:
        print("\nGOOD: LDFs are monotonically decreasing!")

    # Check for invalid values
    print("\n" + "-" * 80)
    print("Checking for invalid LDF values...")
    print("-" * 80)

    if np.any(ldf_values < 1.0):
        invalid = ldf_values < 1.0
        print(f"\nERROR: Found {np.sum(invalid)} LDF values < 1.0!")
        invalid_indices = np.where(invalid)[0]
        for idx in invalid_indices[:10]:
            print(f"  Period {idx+1}: LDF = {ldf_values[idx]:.8f}")
    else:
        print("\nGOOD: All LDF values >= 1.0")

    if np.any(ldf_values == 0):
        zero = ldf_values == 0
        print(f"\nERROR: Found {np.sum(zero)} LDF values = 0!")
    else:
        print("GOOD: No LDF values = 0")

    if np.any(np.isnan(ldf_values)):
        print("\nERROR: Found NaN values!")
    else:
        print("GOOD: No NaN values")

    # Calculate the underlying Weibull parameters
    print("\n" + "=" * 80)
    print("Analyzing Weibull Formula Behavior")
    print("=" * 80)

    a = np.exp(tail_weibull.intercept_.values[0, 0])
    b = tail_weibull.slope_.values[0, 0]

    print(f"\nWeibull parameters:")
    print(f"  a (scale) = {a:.6f}")
    print(f"  b (shape) = {b:.6f}")

    # Manually calculate LDFs to understand the formula
    t = np.arange(1, 201)
    exponent = a * t**b
    exp_neg_exponent = np.exp(-exponent)
    denominator = 1 - exp_neg_exponent
    manual_ldf = 1 / denominator - 1

    print(f"\nManual calculation at selected ages:")
    test_ages = [1, 5, 10, 20, 50, 100, 150, 200]
    for age in test_ages:
        idx = age - 1
        if idx < len(t):
            print(f"  Age {age:3d}: exp(-a*t^b) = {exp_neg_exponent[idx]:.6e}, "
                  f"denom = {denominator[idx]:.6e}, LDF = {manual_ldf[idx]:.6f}")

    # Check when denominator approaches 1 (underflow region)
    print("\n" + "-" * 80)
    print("Checking for numerical underflow (denominator = 1.0)...")
    print("-" * 80)

    underflow = denominator == 1.0
    if np.any(underflow):
        first_underflow = np.where(underflow)[0][0]
        print(f"\nWARNING: Underflow begins at age {first_underflow + 1}")
        print(f"  Exponent at that age: {exponent[first_underflow]:.2f}")
        print(f"  exp(-{exponent[first_underflow]:.2f}) ≈ 0 (below machine precision)")
        print(f"  This causes denominator = 1 - 0 = 1, leading to LDF = 0")
    else:
        print("\nGOOD: No underflow detected in tested range")

    return tail_weibull, ldf_values, a, b


def test_different_parameters():
    """Test Weibull with different triangles to see parameter variation"""
    print("\n" + "=" * 80)
    print("Testing Weibull with Different Triangles")
    print("=" * 80)

    datasets = ['raa', 'genins', 'clrd']

    for dataset_name in datasets:
        try:
            print(f"\n--- Dataset: {dataset_name} ---")
            data = cl.load_sample(dataset_name)

            # Limit to single triangle if multi-dimensional
            if data.shape[0] > 1:
                data = data.iloc[0, 0]

            tail = cl.TailCurve(curve='weibull', fit_period=(12, None)).fit(data)

            a = np.exp(tail.intercept_.values[0, 0])
            b = tail.slope_.values[0, 0]

            print(f"  a (scale) = {a:.6f}")
            print(f"  b (shape) = {b:.6f}")
            print(f"  Tail factor: {tail.tail_.values[0, 0]:.6f}")

            # Check for issues
            ldf_values = tail.ldf_.values[0, 0, 0, :]
            if np.any(np.diff(ldf_values) > 0):
                print(f"  WARNING: Non-monotonic LDFs detected!")
            if np.any(ldf_values < 1.0):
                print(f"  ERROR: Invalid LDF < 1.0 detected!")

        except Exception as e:
            print(f"  Error: {e}")


def plot_weibull_comparison(tail_weibull, ldf_values, a, b, save_path='weibull_analysis.png'):
    """Plot Weibull tail behavior"""
    print("\n" + "=" * 80)
    print("Creating visualization...")
    print("=" * 80)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: LDF vs Age
    ages = np.arange(1, len(ldf_values) + 1)
    axes[0, 0].plot(ages, ldf_values, 'b-', linewidth=2)
    axes[0, 0].axhline(y=1.0, color='r', linestyle='--', label='LDF = 1.0')
    axes[0, 0].set_xlabel('Development Period')
    axes[0, 0].set_ylabel('LDF')
    axes[0, 0].set_title('Weibull Tail: LDF vs Age')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # Plot 2: LDF vs Age (log scale)
    axes[0, 1].semilogy(ages, ldf_values - 1, 'b-', linewidth=2)
    axes[0, 1].set_xlabel('Development Period')
    axes[0, 1].set_ylabel('LDF - 1 (log scale)')
    axes[0, 1].set_title('Weibull Tail: LDF - 1 vs Age (log scale)')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Exponent behavior
    t = np.arange(1, 201)
    exponent = a * t**b
    axes[1, 0].plot(t, exponent, 'g-', linewidth=2)
    axes[1, 0].axhline(y=700, color='r', linestyle='--', label='exp() overflow threshold ≈ 700')
    axes[1, 0].set_xlabel('Development Period')
    axes[1, 0].set_ylabel('a × t^b')
    axes[1, 0].set_title(f'Exponent: a={a:.3f}, b={b:.3f}')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # Plot 4: exp(-exponent) behavior
    exp_vals = np.exp(-exponent)
    axes[1, 1].semilogy(t, exp_vals, 'm-', linewidth=2)
    axes[1, 1].axhline(y=np.finfo(float).tiny, color='r', linestyle='--',
                       label=f'Machine epsilon ≈ {np.finfo(float).tiny:.2e}')
    axes[1, 1].set_xlabel('Development Period')
    axes[1, 1].set_ylabel('exp(-a × t^b)')
    axes[1, 1].set_title('Exponential Decay (log scale)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot saved to: {save_path}")


if __name__ == "__main__":
    tail_weibull, ldf_values, a, b = test_weibull_monotonicity()
    test_different_parameters()

    try:
        plot_weibull_comparison(tail_weibull, ldf_values, a, b)
    except Exception as e:
        print(f"\nNote: Could not create plot (likely no display): {e}")

    print("\n" + "=" * 80)
    print("Analysis complete")
    print("=" * 80)
