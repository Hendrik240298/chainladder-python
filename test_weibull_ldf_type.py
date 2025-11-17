"""
Determine if Weibull formula produces period-to-period or age-to-ultimate LDFs
"""
import numpy as np

print("=" * 80)
print("Which Type of LDF Does the Weibull Formula Produce?")
print("=" * 80)

# Weibull parameters
a = 0.428995
b = 1.089102
t = np.arange(1, 11)

# Growth curve: proportion emerged by time t
G_t = 1 - np.exp(-a * t**b)

# Weibull formula output
formula_output = 1 / (1 - np.exp(-a * t**b))

# Type 1: Period-to-period LDFs
# These represent Cumulative(t+1) / Cumulative(t)
G_t_plus_1 = np.append(G_t[1:], 1.0)  # G(2), G(3), ..., G(ultimate=1)
period_to_period_LDF = G_t_plus_1 / G_t

# Type 2: Age-to-ultimate LDFs
# These represent Ultimate / Cumulative(t)
age_to_ultimate_LDF = 1.0 / G_t

print("\n1. Formula Output: 1/(1 - exp(-a × t^b))")
print("-" * 80)
for i in range(len(t)):
    print(f"  t={t[i]:2d}: {formula_output[i]:.6f}")

print("\n2. Period-to-Period LDFs: Cumulative(t+1) / Cumulative(t)")
print("-" * 80)
for i in range(len(t)):
    print(f"  t={t[i]:2d}→{t[i]+1}: {period_to_period_LDF[i]:.6f}")

print("\n3. Age-to-Ultimate LDFs: Ultimate / Cumulative(t)")
print("-" * 80)
for i in range(len(t)):
    print(f"  t={t[i]:2d}→Ult: {age_to_ultimate_LDF[i]:.6f}")

print("\n" + "=" * 80)
print("Comparison")
print("=" * 80)

match_p2p = np.allclose(formula_output, period_to_period_LDF, atol=1e-5)
match_a2u = np.allclose(formula_output, age_to_ultimate_LDF, atol=1e-5)

print(f"\nFormula matches Period-to-Period LDFs: {match_p2p}")
print(f"Formula matches Age-to-Ultimate LDFs: {match_a2u}")

if match_a2u:
    print("\n✓ The Weibull formula produces AGE-TO-ULTIMATE LDFs")
    print("  These represent: Ultimate / Cumulative(t)")
    print("  Example: t=5 gives factor from age 5 to ultimate")

if match_p2p:
    print("\n✓ The Weibull formula produces PERIOD-TO-PERIOD LDFs")
    print("  These represent: Cumulative(t+1) / Cumulative(t)")
    print("  Example: t=5 gives factor from age 5 to age 6")

print("\n" + "=" * 80)
print("Test: Can These Be Chained?")
print("=" * 80)

print("\nIf period-to-period, chaining should work:")
print("  LDF(1→2) × LDF(2→3) × LDF(3→4) should equal cumulative LDF(1→4)")

if match_p2p:
    chain_product = period_to_period_LDF[0] * period_to_period_LDF[1] * period_to_period_LDF[2]
    expected_cumulative = G_t[3] / G_t[0]  # Cumulative(4) / Cumulative(1)
    print(f"  Product: {chain_product:.6f}")
    print(f"  Expected: {expected_cumulative:.6f}")
    print(f"  Match: {np.isclose(chain_product, expected_cumulative)}")

print("\nIf age-to-ultimate, chaining is meaningless:")
print("  (Ult/Cum1) × (Ult/Cum2) × (Ult/Cum3) = Ult³/(Cum1×Cum2×Cum3) ← meaningless!")

if match_a2u:
    chain_product = age_to_ultimate_LDF[0] * age_to_ultimate_LDF[1] * age_to_ultimate_LDF[2]
    print(f"  Product: {chain_product:.6f} ← This doesn't represent anything meaningful")

print("\n" + "=" * 80)
print("Implication for _get_tail_prediction()")
print("=" * 80)

if match_a2u:
    print("\nSince Weibull produces age-to-ultimate LDFs:")
    print("  ✗ Taking products in _get_tail_prediction() is WRONG")
    print("  ✓ Should use the value directly as tail factor")
    print("  ")
    print("  Example:")
    print(f"    Value at position 10: {formula_output[9]:.6f}")
    print(f"    This already represents the tail factor from position 10 to ultimate")
    print(f"    No need to multiply anything!")

if match_p2p:
    print("\nSince Weibull produces period-to-period LDFs:")
    print("  ✓ Taking products in _get_tail_prediction() is CORRECT")
    print("  ✓ Product of remaining factors gives cumulative tail")
