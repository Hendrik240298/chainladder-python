"""
Find what conditions make Weibull unstable
Test problematic scenarios
"""
import numpy as np
import chainladder as cl
import warnings

print("=" * 80)
print("Finding Conditions That Break Weibull")
print("=" * 80)

print("\n1. SPARSE DATA (Few development periods)")
print("-" * 80)

# Simulate triangle with only 4-5 development periods
raa = cl.load_sample('raa')
raa_sparse = raa.iloc[:, :, :5, :5]  # Only first 5 periods

print(f"Sparse triangle shape: {raa_sparse.shape}")

try:
    tail_sparse = cl.TailCurve(curve='weibull', fit_period=(12, None)).fit(raa_sparse)

    a = np.exp(tail_sparse.intercept_.values[0, 0])
    b = tail_sparse.slope_.values[0, 0]
    tail_f = tail_sparse.tail_.values[0, 0]

    print(f"Result: a={a:.6f}, b={b:.6f}, tail={tail_f:.6f}")

    if b < 0:
        print(f"  ⚠️  NEGATIVE SHAPE PARAMETER")
    if tail_f > 2.0:
        print(f"  ⚠️  VERY LARGE TAIL FACTOR")

except Exception as e:
    print(f"ERROR: {e}")

print("\n2. VERY MATURE DATA (LDFs close to 1.0)")
print("-" * 80)

# Create synthetic mature triangle
mature_ldfs = np.array([1.001, 1.0005, 1.0003, 1.0002, 1.0001])
print(f"Mature LDFs: {mature_ldfs}")

try:
    # Can't easily create this, but can test with tail periods
    genins = cl.load_sample('genins')
    # Use only the tail periods which should be mature
    tail_mature = cl.TailCurve(curve='weibull', fit_period=(96, None)).fit(genins)

    a = np.exp(tail_mature.intercept_.values[0, 0])
    b = tail_mature.slope_.values[0, 0]
    tail_f = tail_mature.tail_.values[0, 0]

    print(f"Using late periods only:")
    print(f"  a={a:.6f}, b={b:.6f}, tail={tail_f:.6f}")

    if b < 0:
        print(f"  ⚠️  NEGATIVE SHAPE PARAMETER")

except Exception as e:
    print(f"ERROR: {e}")

print("\n3. NEGATIVE DEVELOPMENT (Some periods have LDF < 1.0)")
print("-" * 80)

# We know CLRD has this issue
clrd = cl.load_sample('clrd')
clrd_single = clrd.iloc[0, 0]  # Get single triangle

print("Testing CLRD (known to have LDF < 1.0):")

try:
    dev_clrd = cl.Development().fit(clrd_single)
    ldfs_clrd = dev_clrd.ldf_.values[0, 0, 0, :]

    print(f"CLRD LDFs: {ldfs_clrd}")
    print(f"Min LDF: {np.nanmin(ldfs_clrd):.6f}")

    if np.any(ldfs_clrd < 1.0):
        print(f"  ⚠️  Has LDFs < 1.0 (negative development)")

    tail_clrd = cl.TailCurve(curve='weibull', fit_period=(12, None)).fit(clrd_single)

    a = np.exp(tail_clrd.intercept_.values[0, 0])
    b = tail_clrd.slope_.values[0, 0]
    tail_f = tail_clrd.tail_.values[0, 0]

    print(f"Result: a={a:.6f}, b={b:.6f}, tail={tail_f:.6f}")

    if b < 0:
        print(f"  ⚠️  NEGATIVE SHAPE PARAMETER (b={b:.6f})")
    if np.isinf(tail_f):
        print(f"  ⚠️  INFINITE TAIL FACTOR")

except Exception as e:
    print(f"ERROR: {e}")

print("\n4. NON-MONOTONIC DEVELOPMENT PATTERN")
print("-" * 80)

# Check if GenIns has non-monotonic pattern
genins = cl.load_sample('genins')
dev_genins = cl.Development().fit(genins)
ldfs_genins = dev_genins.ldf_.values[0, 0, 0, :]

print(f"GenIns LDFs: {ldfs_genins}")

# Check if LDFs increase anywhere
diffs = np.diff(ldfs_genins)
if np.any(diffs > 0):
    print(f"  ⚠️  LDFs INCREASE at some periods (non-monotonic observed data)")
    inc_idx = np.where(diffs > 0)[0]
    for idx in inc_idx[:3]:
        print(f"    Position {idx} -> {idx+1}: {ldfs_genins[idx]:.6f} -> {ldfs_genins[idx+1]:.6f}")

try:
    tail_genins = cl.TailCurve(curve='weibull', fit_period=(12, None)).fit(genins)

    a = np.exp(tail_genins.intercept_.values[0, 0])
    b = tail_genins.slope_.values[0, 0]
    tail_f = tail_genins.tail_.values[0, 0]

    print(f"\nWeibull fit result:")
    print(f"  a={a:.6f}, b={b:.6f}, tail={tail_f:.6f}")

except Exception as e:
    print(f"ERROR: {e}")

print("\n5. COMPARISON: WEIBULL VS EXPONENTIAL ON PROBLEMATIC DATA")
print("-" * 80)

problem_datasets = [
    ('CLRD (negative dev)', clrd_single),
    ('GenIns (non-monotonic)', genins),
]

for name, data in problem_datasets:
    print(f"\n{name}:")

    try:
        tail_w = cl.TailCurve(curve='weibull', fit_period=(12, None)).fit(data)
        tail_e = cl.TailCurve(curve='exponential', fit_period=(12, None)).fit(data)

        w_tail = tail_w.tail_.values[0, 0]
        e_tail = tail_e.tail_.values[0, 0]

        print(f"  Weibull:     {w_tail:.6f}")
        print(f"  Exponential: {e_tail:.6f}")

        diff = abs(w_tail - e_tail)
        rel_diff = diff / e_tail

        print(f"  Difference:  {diff:.6f} ({rel_diff*100:.2f}%)")

        if rel_diff > 0.2:
            print(f"  ⚠️  LARGE DIFFERENCE (>20%)")

    except Exception as e:
        print(f"  ERROR: {e}")

print("\n6. DOES THRESHOLD HELP?")
print("-" * 80)

print("Testing CLRD with different thresholds:")

thresholds = [1.00001, 1.01, 1.05, 1.10]

for thresh in thresholds:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tail = cl.TailCurve(
                curve='weibull',
                fit_period=(12, None),
                reg_threshold=(thresh, None)
            ).fit(clrd_single)

            a = np.exp(tail.intercept_.values[0, 0])
            b = tail.slope_.values[0, 0]
            tail_f = tail.tail_.values[0, 0]

            status = "OK" if b > 0 and not np.isinf(tail_f) else "PROBLEM"
            print(f"  Threshold {thresh:.5f}: b={b:+.4f}, tail={tail_f:.6f} [{status}]")

    except Exception as e:
        print(f"  Threshold {thresh:.5f}: ERROR - {e}")

print("\n" + "=" * 80)
print("DIAGNOSIS")
print("=" * 80)

print("""
Weibull appears STABLE for well-behaved data (RAA), but breaks when:

1. Negative development (LDF < 1.0): Produces negative shape parameter
2. Non-monotonic observed LDFs: May produce unreliable fits
3. Very mature data (LDFs very close to 1.0): Numerical issues in log-log transformation

The "instability" is likely DATA-DEPENDENT, not a fundamental code bug.

However, Issues #2 (underflow) and #3 (negative params) are still real:
- Issue #2: Would occur with very long extrapolation or certain parameters
- Issue #3: Should be caught with parameter validation

The real fix needed:
- Add parameter validation (b > 0)
- Add better error messages for unsuitable data
- Document data requirements for Weibull fitting
""")
