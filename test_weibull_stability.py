"""
Test script to investigate Weibull stability in TailCurve and ClarkLDF
"""
import numpy as np
import chainladder as cl
import warnings

def test_weibull_formula():
    """Verify the Weibull formula implementation"""
    print("=" * 80)
    print("Testing Weibull Formula Implementation")
    print("=" * 80)

    # Test the formula: LDF = 1/(1 - exp(-a * t^b)) - 1
    # This should match: t_t = l/(l - e^(-a * t^b))

    # Example parameters
    a = 0.5  # scale parameter (exp(intercept))
    b = 1.5  # shape parameter (slope)
    t = np.array([1, 2, 3, 5, 10, 20, 50, 100])

    # Formula 1: As stated by user (assuming l should be 1 for growth curve)
    # t_t = l/(l - e^(-a * t^b))
    l = 1.0
    formula1 = l / (l - np.exp(-a * t**b))

    # Formula 2: As implemented in code
    # LDF = 1/(1 - exp(-a * t^b)) - 1
    formula2 = 1 / (1 - np.exp(-a * t**b)) - 1

    print(f"\nParameters: a={a}, b={b}")
    print(f"\nDevelopment periods: {t}")
    print(f"\nFormula 1 (l/(l - e^(-a*t^b))): {formula1}")
    print(f"\nFormula 2 (1/(1 - e^(-a*t^b)) - 1): {formula2}")
    print(f"\nDifference: {formula1 - formula2}")
    print(f"\nNote: Formula 1 gives growth curve G(t), Formula 2 gives LDF = G(t) - 1")

    return formula1, formula2


def test_overflow_underflow():
    """Test for overflow/underflow issues at extreme values"""
    print("\n" + "=" * 80)
    print("Testing Numerical Stability: Overflow/Underflow")
    print("=" * 80)

    # Test with various parameter combinations
    test_cases = [
        {"a": 0.1, "b": 0.5, "desc": "Small a, small b"},
        {"a": 0.5, "b": 1.5, "desc": "Moderate a, moderate b"},
        {"a": 1.0, "b": 2.0, "desc": "Large a, large b"},
        {"a": 2.0, "b": 3.0, "desc": "Very large a and b"},
        {"a": 0.01, "b": 3.0, "desc": "Small a, large b"},
    ]

    t = np.array([1, 5, 10, 20, 50, 100, 200, 500, 1000])

    for case in test_cases:
        a, b = case["a"], case["b"]
        print(f"\n{case['desc']}: a={a}, b={b}")
        print("-" * 40)

        # Calculate the exponent
        exponent = a * t**b
        print(f"Exponent (-a*t^b): {-exponent}")

        # Check for overflow in exp()
        exp_vals = np.exp(-exponent)
        print(f"exp(-a*t^b): {exp_vals}")

        # Check denominator
        denom = 1 - exp_vals
        print(f"Denominator (1 - exp(...)): {denom}")

        # Calculate LDF
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ldf = 1 / denom - 1

        print(f"LDF: {ldf}")

        # Check for issues
        if np.any(denom == 1.0):
            print("WARNING: Denominator = 1.0 detected (underflow in exp)")
        if np.any(ldf <= 0):
            print("ERROR: Invalid LDF <= 0 detected!")
        if np.any(np.isnan(ldf)):
            print("ERROR: NaN detected in LDF!")
        if np.any(np.isinf(ldf)):
            print("WARNING: Inf detected in LDF!")


def test_with_real_data():
    """Test TailCurve Weibull with real triangle data"""
    print("\n" + "=" * 80)
    print("Testing TailCurve Weibull with Real Data")
    print("=" * 80)

    # Load sample data
    raa = cl.load_sample('raa')

    print("\nOriginal triangle shape:", raa.shape)

    # Fit Development
    dev = cl.Development().fit(raa)
    print("\nDevelopment LDFs:")
    print(dev.ldf_)

    # Try Weibull tail (pass the triangle, not the Development object)
    print("\n--- Testing Weibull Tail ---")
    try:
        tail_weibull = cl.TailCurve(curve='weibull', fit_period=(12, None)).fit(raa)
        print("Success! Weibull tail fitted.")
        print(f"\nSlope: {tail_weibull.slope_}")
        print(f"\nIntercept: {tail_weibull.intercept_}")
        print(f"\nTail factor: {tail_weibull.tail_}")
        print(f"\nLDF with tail (last 5 periods):")
        print(tail_weibull.ldf_.to_frame().iloc[:, -5:])

        # Check for invalid LDFs
        ldf_values = tail_weibull.ldf_.values
        if np.any(ldf_values <= 0):
            print("\nERROR: Found invalid LDF values <= 0!")
        if np.any(np.isnan(ldf_values)):
            print("\nERROR: Found NaN in LDF values!")
        if np.any(ldf_values == 0):
            print("\nERROR: Found LDF values = 0!")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

    # Try exponential tail for comparison
    print("\n--- Testing Exponential Tail (for comparison) ---")
    try:
        tail_exp = cl.TailCurve(curve='exponential', fit_period=(12, None)).fit(raa)
        print("Success! Exponential tail fitted.")
        print(f"\nSlope: {tail_exp.slope_}")
        print(f"\nIntercept: {tail_exp.intercept_}")
        print(f"\nTail factor: {tail_exp.tail_}")
    except Exception as e:
        print(f"ERROR: {e}")


def test_clark_ldf_weibull():
    """Test ClarkLDF with Weibull growth curve"""
    print("\n" + "=" * 80)
    print("Testing ClarkLDF Weibull")
    print("=" * 80)

    # Load sample data
    genins = cl.load_sample('genins')

    print("\nOriginal triangle shape:", genins.shape)

    # Try Weibull growth curve
    print("\n--- Testing ClarkLDF with Weibull ---")
    try:
        clark_weibull = cl.ClarkLDF(growth='weibull').fit(genins)
        print("Success! ClarkLDF Weibull fitted.")
        print(f"\nTheta: {clark_weibull.theta_}")
        print(f"\nOmega: {clark_weibull.omega_}")
        print(f"\nLDF:")
        print(clark_weibull.ldf_)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

    # Try loglogistic for comparison
    print("\n--- Testing ClarkLDF with Loglogistic (for comparison) ---")
    try:
        clark_ll = cl.ClarkLDF(growth='loglogistic').fit(genins)
        print("Success! ClarkLDF Loglogistic fitted.")
        print(f"\nTheta: {clark_ll.theta_}")
        print(f"\nOmega: {clark_ll.omega_}")
    except Exception as e:
        print(f"ERROR: {e}")


def test_transformation_stability():
    """Test the log-log transformation for stability"""
    print("\n" + "=" * 80)
    print("Testing Log-Log Transformation Stability")
    print("=" * 80)

    # Test LDFs near 1.0 (where transformation becomes unstable)
    ldf_values = np.array([1.0, 1.00001, 1.0001, 1.001, 1.01, 1.1, 1.5, 2.0, 5.0])

    print("\nTesting transformation: log(log(ldf / (ldf - 1)))")
    print("-" * 40)

    for ldf in ldf_values:
        try:
            # Step 1: ldf / (ldf - 1)
            step1 = ldf / (ldf - 1)

            # Step 2: log(step1)
            step2 = np.log(step1)

            # Step 3: log(step2)
            step3 = np.log(step2)

            print(f"LDF={ldf:8.5f}: step1={step1:10.3f}, step2={step2:10.5f}, step3={step3:10.5f}")
        except Exception as e:
            print(f"LDF={ldf:8.5f}: ERROR - {e}")


if __name__ == "__main__":
    # Run all tests
    test_weibull_formula()
    test_overflow_underflow()
    test_transformation_stability()
    test_with_real_data()
    test_clark_ldf_weibull()

    print("\n" + "=" * 80)
    print("All tests completed")
    print("=" * 80)
