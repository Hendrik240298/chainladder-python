"""
Test all TailCurve methods for monotonicity issues
Determine if the problem is Weibull-specific or affects all curves
"""
import numpy as np
import chainladder as cl

print("=" * 80)
print("Testing All Curve Types for Monotonicity")
print("=" * 80)

# Test datasets
datasets = {
    'raa': cl.load_sample('raa'),
    'genins': cl.load_sample('genins'),
}

# Curve types to test
curve_types = ['exponential', 'inverse_power', 'weibull']

results = {}

for dataset_name, data in datasets.items():
    print(f"\n{'=' * 80}")
    print(f"Dataset: {dataset_name.upper()}")
    print('=' * 80)

    results[dataset_name] = {}

    for curve_type in curve_types:
        print(f"\n--- Curve Type: {curve_type} ---")

        try:
            # Fit the tail
            tail = cl.TailCurve(
                curve=curve_type,
                fit_period=(12, None),
                extrap_periods=10
            ).fit(data)

            # Get parameters
            a = np.exp(tail.intercept_.values[0, 0])
            b = tail.slope_.values[0, 0]
            tail_factor = tail.tail_.values[0, 0]

            print(f"  Parameters: a={a:.6f}, b={b:.6f}")
            print(f"  Tail factor: {tail_factor:.6f}")

            # Get LDF values
            ldfs = tail.ldf_.values[0, 0, 0, :]

            print(f"  LDF shape: {ldfs.shape}")
            print(f"  Last 5 LDFs:")
            for i in range(max(0, len(ldfs) - 5), len(ldfs)):
                age = tail.ldf_.ddims[i]
                print(f"    Position {i} (Age {age}): {ldfs[i]:.6f}")

            # Check for monotonicity violations
            diffs = np.diff(ldfs)
            increases = diffs > 0
            num_violations = np.sum(increases)

            if num_violations > 0:
                print(f"  ⚠️  NON-MONOTONIC: {num_violations} violations detected")
                violation_indices = np.where(increases)[0]
                print(f"  First violation:")
                idx = violation_indices[0]
                print(f"    Position {idx} -> {idx+1}: {ldfs[idx]:.6f} -> {ldfs[idx+1]:.6f}")
            else:
                print(f"  ✓ MONOTONIC: All LDFs decrease")

            # Check for invalid values
            invalid_ldfs = ldfs < 1.0
            num_invalid = np.sum(invalid_ldfs)

            if num_invalid > 0:
                print(f"  ⚠️  INVALID: {num_invalid} LDFs < 1.0")
            else:
                print(f"  ✓ VALID: All LDFs >= 1.0")

            # Check for NaN/Inf
            if np.any(np.isnan(ldfs)):
                print(f"  ⚠️  Contains NaN values")
            if np.any(np.isinf(ldfs)):
                print(f"  ⚠️  Contains Inf values")

            # Store results
            results[dataset_name][curve_type] = {
                'success': True,
                'monotonic': num_violations == 0,
                'valid': num_invalid == 0,
                'violations': num_violations,
                'a': a,
                'b': b,
                'tail_factor': tail_factor
            }

        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            results[dataset_name][curve_type] = {
                'success': False,
                'error': str(e)
            }

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

for dataset_name in datasets.keys():
    print(f"\n{dataset_name.upper()}:")
    for curve_type in curve_types:
        result = results[dataset_name][curve_type]
        if result['success']:
            status = "✓ OK" if result['monotonic'] and result['valid'] else "❌ ISSUES"
            issues = []
            if not result['monotonic']:
                issues.append(f"{result['violations']} non-monotonic")
            if not result['valid']:
                issues.append("invalid LDFs")
            issue_str = f" ({', '.join(issues)})" if issues else ""
            print(f"  {curve_type:15s}: {status}{issue_str}")
        else:
            print(f"  {curve_type:15s}: ❌ FAILED - {result['error']}")

print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)

# Count issues by curve type
curve_issues = {ct: [] for ct in curve_types}
for dataset_name in datasets.keys():
    for curve_type in curve_types:
        result = results[dataset_name][curve_type]
        if result['success'] and not result['monotonic']:
            curve_issues[curve_type].append(dataset_name)

print("\nCurves with non-monotonic issues:")
for curve_type, affected_datasets in curve_issues.items():
    if affected_datasets:
        print(f"  {curve_type}: {', '.join(affected_datasets)}")
    else:
        print(f"  {curve_type}: None ✓")

# Determine if issue is Weibull-specific
weibull_only = (len(curve_issues['weibull']) > 0 and
                len(curve_issues['exponential']) == 0 and
                len(curve_issues['inverse_power']) == 0)

all_curves = (len(curve_issues['weibull']) > 0 and
              len(curve_issues['exponential']) > 0 and
              len(curve_issues['inverse_power']) > 0)

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

if weibull_only:
    print("\n✓ Issue is WEIBULL-SPECIFIC")
    print("  - Only Weibull exhibits non-monotonic behavior")
    print("  - Exponential and inverse_power work correctly")
    print("  - Fix should target Weibull only")
    print("  - Use flag: _produces_age_to_ultimate = True for Weibull only")
elif all_curves:
    print("\n⚠️  Issue affects ALL CURVES")
    print("  - All three curve types exhibit non-monotonic behavior")
    print("  - All curves may produce age-to-ultimate factors")
    print("  - Fix should apply to all curve types")
    print("  - Use flag: _produces_age_to_ultimate = True for all in TailCurve")
else:
    print("\n⚠️  Issue is PARTIAL")
    print("  - Some but not all curves affected")
    print(f"  - Affected: {[ct for ct, ds in curve_issues.items() if ds]}")
    print("  - Requires case-by-case analysis")

print("\n" + "=" * 80)
print("MATHEMATICAL ANALYSIS")
print("=" * 80)

# Analyze the formulas
print("\nFormula analysis:")

print("\n1. EXPONENTIAL: LDF = exp(b×t + a)")
print("   - This is NOT a growth curve form")
print("   - Direct exponential decay")
print("   - May produce period-to-period or age-to-ultimate depending on fitting")
print("   - Need to check actual behavior")

print("\n2. INVERSE POWER: LDF = exp(a) × t^b")
print("   - Power law decay")
print("   - Often used for age-to-ultimate")
print("   - Need to check actual behavior")

print("\n3. WEIBULL: LDF = 1/(1 - exp(-a×t^b)) - 1")
print("   - Growth curve form: G(t) = Ultimate/Cumulative(t)")
print("   - Definitively age-to-ultimate")
print("   - Confirmed non-monotonic when multiplied")
