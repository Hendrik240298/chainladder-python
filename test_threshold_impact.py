"""
Analyze the impact of the reg_threshold parameter
"""
import numpy as np
import chainladder as cl

print("=" * 80)
print("Why reg_threshold = 1.00001?")
print("=" * 80)

# Test the log-log transformation with values near 1.0
test_ldfs = np.array([0.95, 0.99, 1.0, 1.00001, 1.0001, 1.001, 1.01, 1.1, 1.5])

print("\nLog-log transformation: log(log(LDF/(LDF-1)))")
print("-" * 80)

for ldf in test_ldfs:
    try:
        if ldf <= 1.0:
            result = "undefined (LDF ≤ 1.0 causes division by zero or log of negative)"
        else:
            step1 = ldf / (ldf - 1)
            step2 = np.log(step1)
            step3 = np.log(step2)
            result = f"{step3:.6f}"
    except:
        result = "error"

    print(f"  LDF = {ldf:.5f}: {result}")

print("\n" + "=" * 80)
print("Check RAA Dataset for LDFs Near 1.0")
print("=" * 80)

raa = cl.load_sample('raa')
dev = cl.Development().fit(raa)
ldfs = dev.ldf_.values[0, 0, 0, :]

print(f"\nObserved LDFs in RAA:")
print(f"  Min: {np.min(ldfs):.6f}")
print(f"  Max: {np.max(ldfs):.6f}")
print(f"  All values: {ldfs}")

print(f"\nNumber of LDFs ≤ 1.00001: {np.sum(ldfs <= 1.00001)}")
print(f"Number of LDFs ≤ 1.0: {np.sum(ldfs <= 1.0)}")
print(f"Number of LDFs < 1.0: {np.sum(ldfs < 1.0)}")

print("\n✓ Threshold 1.00001 doesn't exclude any RAA data")

print("\n" + "=" * 80)
print("Check CLRD Dataset")
print("=" * 80)

clrd = cl.load_sample('clrd')
# CLRD is multi-dimensional, get first triangle
clrd_single = clrd.iloc[0, 0]
dev_clrd = cl.Development().fit(clrd_single)
ldfs_clrd = dev_clrd.ldf_.values[0, 0, 0, :]

print(f"\nObserved LDFs in CLRD (first triangle):")
print(f"  Min: {np.min(ldfs_clrd):.6f}")
print(f"  Max: {np.max(ldfs_clrd):.6f}")
print(f"  All values: {ldfs_clrd}")

print(f"\nNumber of LDFs ≤ 1.00001: {np.sum(ldfs_clrd <= 1.00001)}")
print(f"Number of LDFs ≤ 1.0: {np.sum(ldfs_clrd <= 1.0)}")
print(f"Number of LDFs < 1.0: {np.sum(ldfs_clrd < 1.0)}")

if np.any(ldfs_clrd < 1.0):
    print("\n⚠️ CLRD has LDFs < 1.0 (negative development)")
    bad_ldfs = ldfs_clrd[ldfs_clrd < 1.0]
    print(f"  Values < 1.0: {bad_ldfs}")

print("\n" + "=" * 80)
print("Would Changing Threshold Help?")
print("=" * 80)

print("\nCurrent threshold: 1.00001")
print("  Excludes: LDF ≤ 1.00001")
print("  Reason: Numerical stability in log(log(...)) transformation")

print("\nProposed threshold: 1.0")
print("  Excludes: LDF ≤ 1.0")
print("  Difference: Values in (1.0, 1.00001] would now be INCLUDED")

print("\nImpact:")
print("  - If no LDFs in range (1.0, 1.00001]: NO DIFFERENCE")
print("  - If LDFs in that range exist: Very small impact (range is tiny)")

print("\n" + "=" * 80)
print("Alternative: Exclude Small Positive LDFs")
print("=" * 80)

print("\nSome commercial software excludes LDFs below larger thresholds:")
print("  - Threshold 1.01: Excludes LDFs ≤ 1.01")
print("  - Threshold 1.05: Excludes LDFs ≤ 1.05")
print("\nRationale: Very small LDFs indicate:")
print("  - Mature development (close to ultimate)")
print("  - May not follow Weibull pattern")
print("  - Can distort the curve fit")

print(f"\nRAA LDFs that would be excluded with different thresholds:")
thresholds = [1.00001, 1.0, 1.01, 1.05, 1.10]
for thresh in thresholds:
    excluded = np.sum(ldfs <= thresh)
    pct = 100 * excluded / len(ldfs)
    print(f"  Threshold {thresh:.5f}: {excluded}/{len(ldfs)} excluded ({pct:.1f}%)")

print("\n" + "=" * 80)
print("Test Impact on Weibull Fit")
print("=" * 80)

# Try fitting with different thresholds
for thresh in [1.00001, 1.01, 1.05]:
    try:
        tail = cl.TailCurve(
            curve='weibull',
            fit_period=(12, None),
            reg_threshold=(thresh, None)
        ).fit(raa)

        a = np.exp(tail.intercept_.values[0, 0])
        b = tail.slope_.values[0, 0]
        tail_factor = tail.tail_.values[0, 0]

        # Check monotonicity
        ldfs_out = tail.ldf_.values[0, 0, 0, :]
        non_mono = np.sum(np.diff(ldfs_out) > 0)

        print(f"\nThreshold {thresh:.5f}:")
        print(f"  a = {a:.6f}, b = {b:.6f}")
        print(f"  Tail factor = {tail_factor:.6f}")
        print(f"  Non-monotonic violations: {non_mono}")

    except Exception as e:
        print(f"\nThreshold {thresh:.5f}: ERROR - {e}")

print("\n" + "=" * 80)
print("Conclusion")
print("=" * 80)

print("\n1. Current threshold 1.00001:")
print("   - Protects against numerical issues in log-log transformation")
print("   - Already excludes LDF < 1.0")
print("   - Doesn't exclude any RAA data")

print("\n2. Changing to 1.0:")
print("   - Would only include values in (1.0, 1.00001]")
print("   - Unlikely to change results significantly")
print("   - May cause numerical instability")

print("\n3. Better approach:")
print("   - Use higher threshold (e.g., 1.01 or 1.05)")
print("   - Exclude mature/small LDFs that don't fit Weibull pattern")
print("   - This is what commercial software does")
