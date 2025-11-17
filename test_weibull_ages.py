"""
Diagnostic script to understand the age/period issue in Weibull tail fitting
"""
import numpy as np
import chainladder as cl

# Load sample data
raa = cl.load_sample('raa')

print("=" * 80)
print("Investigating Age/Period Indexing in Weibull Tail")
print("=" * 80)

# Fit development
dev = cl.Development().fit(raa)

print("\nRAA Triangle Development Periods (in months):")
print(f"  ddims: {raa.ddims}")
print(f"  Number of periods: {len(raa.ddims)}")

print("\nDevelopment grain:", raa.development_grain)

# Convert ddims to period indices
print("\nPeriod indices (what the regression uses):")
for i, age in enumerate(raa.ddims, 1):
    print(f"  Index {i}: Age {age} months")

print("\n" + "=" * 80)
print("Manual Weibull Calculation")
print("=" * 80)

# Manually fit the Weibull to understand what's happening
ldf_values = dev.ldf_.values[0, 0, 0, :]
print(f"\nObserved LDFs:")
for i, (age, ldf) in enumerate(zip(raa.ddims, ldf_values), 1):
    print(f"  Period {i} (Age {age}): LDF = {ldf:.6f}")

# Transform for Weibull regression
# y_transformed = log(log(ldf / (ldf - 1)))
y_transformed = np.log(np.log(ldf_values / (ldf_values - 1)))

print(f"\nTransformed y values (log(log(ldf/(ldf-1)))):")
for i, (age, y_val) in enumerate(zip(raa.ddims, y_transformed), 1):
    print(f"  Period {i} (Age {age}): y = {y_val:.6f}")

# The x values used in regression are log(period_index)
x_values = np.log(np.arange(1, len(ldf_values) + 1))

print(f"\nX values for regression (log(period_index)):")
for i, (age, x_val) in enumerate(zip(raa.ddims, x_values), 1):
    print(f"  Period {i} (Age {age}): x = {x_val:.6f}")

# Fit linear regression: y = slope * x + intercept
slope, intercept = np.polyfit(x_values, y_transformed, 1)

print(f"\nRegression results:")
print(f"  Slope (b): {slope:.6f}")
print(f"  Intercept (log(a)): {intercept:.6f}")
print(f"  a = exp(intercept): {np.exp(intercept):.6f}")

# Now predict using the Weibull formula
# The issue: what 't' values should we use?

print("\n" + "=" * 80)
print("Comparing Different Age Interpretations")
print("=" * 80)

a = np.exp(intercept)
b = slope

# Interpretation 1: Use period indices [1, 2, 3, ..., 11, 12, 13, ...]
print("\nInterpretation 1: Using sequential period indices")
t1 = np.arange(1, 15)  # Extend beyond observed
ldf1 = 1 / (1 - np.exp(-a * t1**b)) - 1

for i, (t, ldf) in enumerate(zip(t1, ldf1), 1):
    marker = " <-- observed" if i <= len(ldf_values) else " <-- extrapolated"
    actual = f" (actual: {ldf_values[i-1]:.6f})" if i <= len(ldf_values) else ""
    print(f"  Period {i:2d} (t={t:3d}): LDF = {ldf:.6f}{actual}{marker}")

# Interpretation 2: Use actual ages in months [12, 24, 36, ..., 120, 132, 144, ...]
print("\nInterpretation 2: Using actual development ages (months)")
ages_extended = list(raa.ddims) + [raa.ddims[-1] + 12 * (i+1) for i in range(5)]
# But the formula was fit against log(period_index), not log(age_in_months)
# So this doesn't work directly

print("\n" + "=" * 80)
print("The Root Cause")
print("=" * 80)

print("""
The Weibull regression is fit using:
  - X values: log(period_index) = log([1, 2, 3, ..., 9])
  - Y values: log(log(LDF/(LDF-1)))

This creates a linear relationship:
  log(log(G/(G-1))) = b * log(t) + log(a)

Where 't' is the PERIOD INDEX (1, 2, 3, ...), NOT the age in months.

When predicting, the formula uses:
  LDF = 1/(1 - exp(-a * t^b)) - 1

The 't' here should be the period index.

But wait - let's check what index values correspond to what in the final output:
""")

# Fit TailCurve and examine
tail_weibull = cl.TailCurve(curve='weibull', fit_period=(12, None), extrap_periods=5).fit(raa)

print("\nTailCurve output:")
print(f"  Output shape: {tail_weibull.ldf_.shape}")
print(f"  Development dimensions: {tail_weibull.ldf_.ddims}")

final_ldfs = tail_weibull.ldf_.values[0, 0, 0, :]
print(f"\nFinal LDF values:")
for i, (age, ldf) in enumerate(zip(tail_weibull.ldf_.ddims, final_ldfs), 1):
    marker = " <-- ISSUE!" if i > 1 and ldf > final_ldfs[i-2] else ""
    print(f"  Period {i:2d} (Age {age:3d}): LDF = {ldf:.6f}{marker}")

print("\n" + "=" * 80)
print("Hypothesis")
print("=" * 80)

print("""
The issue appears to be in how the extrapolation indices are being mapped
to the actual development ages. The formula is being applied with period
indices that don't properly align with the final output structure.

Specifically, when concatenating observed LDFs with extrapolated tail LDFs,
there may be a mismatch in what 't' values are being used for the tail portion.
""")
