"""
Investigate the REAL instability issue
What makes Weibull "unusable and unstable" in practice?
"""
import numpy as np
import chainladder as cl

print("=" * 80)
print("Investigating Real Instability in Weibull Tail Estimation")
print("=" * 80)

raa = cl.load_sample('raa')

print("\n1. SENSITIVITY TO FIT_PERIOD")
print("-" * 80)
print("Does changing fit_period dramatically change results?")

fit_periods = [
    (12, None),
    (24, None),
    (36, None),
    (48, None),
    (60, None),
]

results_by_period = {}

for fit_period in fit_periods:
    try:
        tail = cl.TailCurve(curve='weibull', fit_period=fit_period).fit(raa)

        a = np.exp(tail.intercept_.values[0, 0])
        b = tail.slope_.values[0, 0]
        tail_factor = tail.tail_.values[0, 0]

        results_by_period[fit_period] = {
            'a': a,
            'b': b,
            'tail_factor': tail_factor,
            'success': True
        }

        print(f"\nfit_period={fit_period}:")
        print(f"  a={a:.6f}, b={b:.6f}, tail={tail_factor:.6f}")

    except Exception as e:
        print(f"\nfit_period={fit_period}: ERROR - {e}")
        results_by_period[fit_period] = {'success': False, 'error': str(e)}

# Check variability
tail_factors = [r['tail_factor'] for r in results_by_period.values() if r['success']]
if len(tail_factors) > 1:
    tail_std = np.std(tail_factors)
    tail_mean = np.mean(tail_factors)
    tail_cv = tail_std / tail_mean

    print(f"\nVariability across fit_period:")
    print(f"  Mean tail factor: {tail_mean:.6f}")
    print(f"  Std dev: {tail_std:.6f}")
    print(f"  Coefficient of variation: {tail_cv:.4f} ({tail_cv*100:.2f}%)")

    if tail_cv > 0.1:
        print(f"  ⚠️  HIGH VARIABILITY (>10%)")

print("\n2. COMPARE WEIBULL VS OTHER METHODS")
print("-" * 80)
print("Are Weibull results different from other approaches?")

methods = {
    'Weibull': lambda: cl.TailCurve(curve='weibull', fit_period=(12, None)).fit(raa),
    'Exponential': lambda: cl.TailCurve(curve='exponential', fit_period=(12, None)).fit(raa),
    'Inverse Power': lambda: cl.TailCurve(curve='inverse_power', fit_period=(12, None)).fit(raa),
    'ClarkLDF Weibull': lambda: cl.ClarkLDF(growth='weibull').fit(raa),
    'ClarkLDF Loglogistic': lambda: cl.ClarkLDF(growth='loglogistic').fit(raa),
}

method_results = {}

for name, method in methods.items():
    try:
        result = method()

        if hasattr(result, 'tail_'):
            tail_factor = result.tail_.values[0, 0]
        else:
            # ClarkLDF doesn't have tail_, use CDF
            tail_factor = result.cdf_.values[0, 0, 0, -1]

        method_results[name] = tail_factor
        print(f"{name:25s}: tail factor = {tail_factor:.6f}")

    except Exception as e:
        print(f"{name:25s}: ERROR - {e}")

# Compare Weibull methods
if 'Weibull' in method_results and 'ClarkLDF Weibull' in method_results:
    diff = abs(method_results['Weibull'] - method_results['ClarkLDF Weibull'])
    rel_diff = diff / method_results['ClarkLDF Weibull']

    print(f"\nWeibull comparison:")
    print(f"  TailCurve Weibull:  {method_results['Weibull']:.6f}")
    print(f"  ClarkLDF Weibull:   {method_results['ClarkLDF Weibull']:.6f}")
    print(f"  Absolute diff:      {diff:.6f}")
    print(f"  Relative diff:      {rel_diff:.4f} ({rel_diff*100:.2f}%)")

    if rel_diff > 0.05:
        print(f"  ⚠️  SIGNIFICANT DIFFERENCE (>5%)")

print("\n3. EXTRAPOLATION BEHAVIOR")
print("-" * 80)
print("Does Weibull extrapolate reasonably far out?")

# Test with longer extrapolation
tail_short = cl.TailCurve(curve='weibull', fit_period=(12, None), extrap_periods=10).fit(raa)
tail_long = cl.TailCurve(curve='weibull', fit_period=(12, None), extrap_periods=100).fit(raa)

print(f"\nShort extrapolation (10 periods):")
print(f"  Tail factor: {tail_short.tail_.values[0, 0]:.6f}")
print(f"  Last LDF: {tail_short.ldf_.values[0, 0, 0, -1]:.6f}")

print(f"\nLong extrapolation (100 periods):")
print(f"  Tail factor: {tail_long.tail_.values[0, 0]:.6f}")
print(f"  Last LDF: {tail_long.ldf_.values[0, 0, 0, -1]:.6f}")

# Check for underflow
last_10_ldfs = tail_long.ldf_.values[0, 0, 0, -10:]
if np.any(last_10_ldfs == 0):
    print(f"  ⚠️  UNDERFLOW: Some LDFs = 0 in extrapolation")
if np.any(last_10_ldfs < 1.0):
    print(f"  ⚠️  INVALID: Some LDFs < 1.0 in extrapolation")

print("\n4. PARAMETER STABILITY")
print("-" * 80)
print("Are fitted parameters reasonable?")

a = np.exp(tail_short.intercept_.values[0, 0])
b = tail_short.slope_.values[0, 0]

print(f"\nFitted parameters:")
print(f"  a (scale) = {a:.6f}")
print(f"  b (shape) = {b:.6f}")

if b <= 0:
    print(f"  ⚠️  NEGATIVE/ZERO SHAPE PARAMETER")
if b > 5:
    print(f"  ⚠️  VERY LARGE SHAPE PARAMETER")
if a > 100:
    print(f"  ⚠️  VERY LARGE SCALE PARAMETER")

print("\n5. CHECK GENINS (Known to have issues)")
print("-" * 80)

genins = cl.load_sample('genins')

try:
    tail_genins = cl.TailCurve(curve='weibull', fit_period=(12, None)).fit(genins)

    a_g = np.exp(tail_genins.intercept_.values[0, 0])
    b_g = tail_genins.slope_.values[0, 0]
    tail_g = tail_genins.tail_.values[0, 0]

    print(f"GenIns Weibull:")
    print(f"  a={a_g:.6f}, b={b_g:.6f}, tail={tail_g:.6f}")

    # Check observed vs fitted
    dev_genins = cl.Development().fit(genins)
    obs_ldfs = dev_genins.ldf_.values[0, 0, 0, :]
    fitted_ldfs = tail_genins.ldf_.values[0, 0, 0, :len(obs_ldfs)]

    print(f"\nObserved vs Fitted (first few):")
    for i in range(min(5, len(obs_ldfs))):
        diff = fitted_ldfs[i] - obs_ldfs[i]
        print(f"  Age {dev_genins.ldf_.ddims[i]:3d}: obs={obs_ldfs[i]:.6f}, fitted={fitted_ldfs[i]:.6f}, diff={diff:+.6f}")

except Exception as e:
    print(f"GenIns ERROR: {e}")

print("\n6. ULTIMATE LOSSES - THE REAL TEST")
print("-" * 80)
print("How do different methods affect ultimate loss estimates?")

# Calculate ultimates with different methods
dev = cl.Development().fit(raa)
latest_val = raa.latest_diagonal.sum()
if hasattr(latest_val, 'values'):
    latest = latest_val.values[0, 0, 0, 0]
else:
    latest = float(latest_val)

methods_for_ultimate = {
    'No Tail': dev,
    'Weibull': cl.TailCurve(curve='weibull', fit_period=(12, None)).fit(raa),
    'Exponential': cl.TailCurve(curve='exponential', fit_period=(12, None)).fit(raa),
    'ClarkLDF Weibull': cl.ClarkLDF(growth='weibull').fit(raa),
}

print(f"\nLatest diagonal sum: {latest:,.0f}")
print(f"\nUltimate estimates:")

ultimates = {}
for name, tail_obj in methods_for_ultimate.items():
    cdf_to_ult = tail_obj.cdf_.values[0, 0, 0, -1]
    ultimate = latest * cdf_to_ult
    ultimates[name] = ultimate

    print(f"  {name:25s}: {ultimate:12,.0f} (CDF={cdf_to_ult:.6f})")

# Compare Weibull methods
if 'Weibull' in ultimates and 'ClarkLDF Weibull' in ultimates:
    diff_ult = abs(ultimates['Weibull'] - ultimates['ClarkLDF Weibull'])
    rel_diff_ult = diff_ult / ultimates['ClarkLDF Weibull']

    print(f"\nWeibull ultimate difference:")
    print(f"  Absolute: {diff_ult:,.0f}")
    print(f"  Relative: {rel_diff_ult:.4f} ({rel_diff_ult*100:.2f}%)")

print("\n" + "=" * 80)
print("SUMMARY: What is the real instability?")
print("=" * 80)

print("""
Key findings:
1. Sensitivity to fit_period: Does selection significantly change results?
2. Comparison with ClarkLDF: Are TailCurve and ClarkLDF Weibull consistent?
3. Extrapolation stability: Does long extrapolation cause underflow?
4. Parameter reasonableness: Are fitted parameters in valid ranges?
5. Ultimate loss impact: How much do method differences matter in practice?
""")
