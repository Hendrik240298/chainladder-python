"""
Re-examine the interpretation of the Weibull formula
based on commercial software documentation stating:
"1/(1 - exp(-a × t^b)) are ratios of cumulative development from time t to t+1"
"""
import numpy as np
import chainladder as cl

print("=" * 80)
print("Testing Weibull Formula Interpretation")
print("=" * 80)

# Weibull parameters from RAA fit
a = 0.428995
b = 1.089102

# Test with sequential time periods
t = np.arange(1, 15)

# Calculate G(t) - the Weibull growth curve
# Standard interpretation: G(t) = proportion of ultimate emerged by time t
G_t = 1 - np.exp(-a * t**b)

print("\nStandard Weibull Growth Curve: G(t) = 1 - exp(-a × t^b)")
print("(Proportion of ultimate emerged by time t)")
print("-" * 80)
for i in range(min(10, len(t))):
    print(f"  t={t[i]:2d}: G(t) = {G_t[i]:.6f}")

# Calculate 1/(1 - exp(-a × t^b))
formula_values = 1 / (1 - np.exp(-a * t**b))

print("\nFormula: 1/(1 - exp(-a × t^b))")
print("-" * 80)
for i in range(min(10, len(t))):
    print(f"  t={t[i]:2d}: value = {formula_values[i]:.6f}")

# Interpretation 1: Age-to-ultimate factors
print("\n" + "=" * 80)
print("Interpretation 1: Age-to-Ultimate Factors")
print("=" * 80)
print("If G(t) = Cumulative(t)/Ultimate, then:")
print("  1/G(t) = Ultimate/Cumulative(t) = age-to-ultimate LDF")

age_to_ult = 1 / G_t
print("\nAge-to-ultimate LDF = 1/G(t):")
for i in range(min(10, len(t))):
    print(f"  t={t[i]:2d}: LDF = {age_to_ult[i]:.6f}")

print("\nCheck: These should match 1/(1 - exp(-a × t^b))?")
match = np.allclose(age_to_ult, formula_values)
print(f"  Match: {match}")
if match:
    print("  ✓ Formula represents age-to-ultimate factors")

# Interpretation 2: Period-to-period factors
print("\n" + "=" * 80)
print("Interpretation 2: Period-to-Period Factors")
print("=" * 80)
print("If these represent Cumulative(t+1)/Cumulative(t):")

# Calculate actual period-to-period from the growth curve
# Cumulative(t+1)/Cumulative(t) = G(t+1)/G(t)
G_t_plus_1 = np.roll(G_t, -1)
G_t_plus_1[-1] = 1.0  # Ultimate
period_to_period = G_t_plus_1 / G_t

print("\nPeriod-to-period LDF = G(t+1)/G(t):")
for i in range(min(10, len(t))):
    print(f"  t={t[i]:2d}→{t[i]+1}: LDF = {period_to_period[i]:.6f}")

print("\nCheck: These should match 1/(1 - exp(-a × t^b))?")
match2 = np.allclose(period_to_period, formula_values)
print(f"  Match: {match2}")

# Interpretation 3: Check what happens with products
print("\n" + "=" * 80)
print("Interpretation 3: Product Behavior")
print("=" * 80)

print("\nIf age-to-ultimate factors, product is meaningless:")
prod_age_to_ult = formula_values[9] * formula_values[10]
print(f"  LDF[10] × LDF[11] = {formula_values[9]:.6f} × {formula_values[10]:.6f} = {prod_age_to_ult:.6f}")
print(f"  This represents: (Ult/Cum(10)) × (Ult/Cum(11)) = Ult²/(Cum(10)×Cum(11)) - meaningless!")

print("\nIf period-to-period factors, product gives cumulative:")
prod_p2p = period_to_period[9] * period_to_period[10]
print(f"  LDF[10→11] × LDF[11→12] = {period_to_period[9]:.6f} × {period_to_period[10]:.6f} = {prod_p2p:.6f}")
print(f"  This represents: (Cum(11)/Cum(10)) × (Cum(12)/Cum(11)) = Cum(12)/Cum(10) - makes sense!")

# Test what the actual code produces
print("\n" + "=" * 80)
print("Actual TailCurve Output")
print("=" * 80)

raa = cl.load_sample('raa')
tail = cl.TailCurve(curve='weibull', fit_period=(12, None), extrap_periods=5).fit(raa)

print(f"\nFitted parameters:")
print(f"  a = {np.exp(tail.intercept_.values[0,0]):.6f}")
print(f"  b = {tail.slope_.values[0,0]:.6f}")

ldf_vals = tail.ldf_.values[0, 0, 0, :]
print(f"\nLDF values:")
for i, (age, ldf) in enumerate(zip(tail.ldf_.ddims, ldf_vals)):
    if i >= len(ldf_vals) - 3:  # Show last 3
        marker = ""
        if i > 0 and ldf > ldf_vals[i-1]:
            marker = " ← INCREASES (should decrease if tail factors)"
        print(f"  Position {i} (Age {age:3d}): LDF = {ldf:.6f}{marker}")

# Check: What should the last value be?
print("\n" + "=" * 80)
print("Expected vs Actual Final Values")
print("=" * 80)

# If these are age-to-ultimate, the last value should be the smallest individual factor
# If these are period-to-period, the last value should be their cumulative product

print("\nScenario: Last position should be tail factor from age 120 to ultimate")
print(f"\nIf using age-to-ultimate interpretation:")
print(f"  Individual LDF at position 10: {ldf_vals[9]:.6f}")
print(f"  This should be the tail factor")

print(f"\nActual last position value: {ldf_vals[10]:.6f}")

if ldf_vals[10] > ldf_vals[9]:
    print(f"  ✗ Last value is LARGER than second-to-last")
    print(f"  This suggests period-to-period factors were multiplied")
else:
    print(f"  ✓ Last value is smaller than second-to-last")

# Alternative: Maybe the formula needs different interpretation
print("\n" + "=" * 80)
print("Alternative: Check if formula produces period-to-period directly")
print("=" * 80)

# What if 1/(1-exp(-a×t^b)) at time t represents the factor from t to t+1?
# Then we need to verify this matches G(t+1)/G(t)

print("\nDoes 1/(1-exp(-a×t^b)) = G(t+1)/G(t)?")
for i in range(5):
    formula_val = formula_values[i]
    ratio_val = period_to_period[i]
    match = abs(formula_val - ratio_val) < 0.0001
    print(f"  t={t[i]:2d}: formula={formula_val:.6f}, G(t+1)/G(t)={ratio_val:.6f}, match={match}")
