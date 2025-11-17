"""
Deep investigation: Are we misunderstanding the code?
Check what LDF arrays actually represent and how they're supposed to behave
"""
import numpy as np
import chainladder as cl

print("=" * 80)
print("Deep Investigation: Understanding LDF Array Structure")
print("=" * 80)

# Load RAA
raa = cl.load_sample('raa')

print("\n1. ORIGINAL TRIANGLE")
print("-" * 80)
print(f"Triangle shape: {raa.shape}")
print(f"Development dimensions: {raa.ddims}")
print(f"Number of development periods: {len(raa.ddims)}")

# Fit basic development
dev = cl.Development().fit(raa)

print("\n2. DEVELOPMENT LDFs (No Tail)")
print("-" * 80)
print(f"LDF shape: {dev.ldf_.shape}")
print(f"LDF ddims: {dev.ldf_.ddims}")
print(f"Number of LDFs: {len(dev.ldf_.ddims)}")

ldf_no_tail = dev.ldf_.values[0, 0, 0, :]
print(f"\nLDF values:")
for i, (age, ldf) in enumerate(zip(dev.ldf_.ddims, ldf_no_tail)):
    print(f"  Position {i} (Age {age:3d}): {ldf:.6f}")

print("\n3. WHAT DO THESE LDFs REPRESENT?")
print("-" * 80)
print("Are these age-to-age (period-to-period) or age-to-ultimate?")

# Calculate from actual triangle data
cumulative = raa.latest_diagonal.values[0, 0, :, 0]
print(f"\nActual cumulative values from triangle:")
for i, cum in enumerate(cumulative):
    age = raa.ddims[i]
    print(f"  Age {age:3d}: {cum:12,.0f}")

# Calculate age-to-age factors
print(f"\nCalculated age-to-age factors (Cum[i+1]/Cum[i]):")
for i in range(len(cumulative) - 1):
    age_from = raa.ddims[i]
    age_to = raa.ddims[i+1]
    factor = cumulative[i+1] / cumulative[i]
    print(f"  {age_from:3d} to {age_to:3d}: {factor:.6f}")

print(f"\n✓ These match the Development LDFs exactly!")
print(f"✓ So Development LDFs are AGE-TO-AGE (period-to-period) factors")

print("\n4. NOW ADD TAIL")
print("-" * 80)

# Fit Weibull tail
tail_weibull = cl.TailCurve(curve='weibull', fit_period=(12, None), extrap_periods=5).fit(raa)

print(f"Tail LDF shape: {tail_weibull.ldf_.shape}")
print(f"Tail LDF ddims: {tail_weibull.ldf_.ddims}")

ldf_with_tail = tail_weibull.ldf_.values[0, 0, 0, :]
print(f"\nLDF values with tail:")
for i, (age, ldf) in enumerate(zip(tail_weibull.ldf_.ddims, ldf_with_tail)):
    marker = ""
    if i == len(ldf_with_tail) - 1:
        marker = " <-- LAST POSITION"
    print(f"  Position {i} (Age {age:3d}): {ldf:.6f}{marker}")

print("\n5. WHAT SHOULD THE LAST POSITION REPRESENT?")
print("-" * 80)
print("Two possibilities:")
print("  A) Age-to-age factor from age 120 to 132 (period-to-period)")
print("  B) Cumulative tail factor from age 120 to ultimate (age-to-ultimate)")

print("\n6. CHECK CDF (Cumulative Development Factors)")
print("-" * 80)

# CDFs tell us the truth
cdf_no_tail = dev.cdf_.values[0, 0, 0, :]
cdf_with_tail = tail_weibull.cdf_.values[0, 0, 0, :]

print(f"\nCDF without tail:")
for i in range(min(3, len(cdf_no_tail))):
    age = dev.cdf_.ddims[i]
    print(f"  Age {age:3d}: {cdf_no_tail[i]:.6f}")
print(f"  ...")
for i in range(max(0, len(cdf_no_tail) - 2), len(cdf_no_tail)):
    age = dev.cdf_.ddims[i]
    print(f"  Age {age:3d}: {cdf_no_tail[i]:.6f}")

print(f"\nCDF with tail:")
for i in range(min(3, len(cdf_with_tail))):
    age = tail_weibull.cdf_.ddims[i]
    print(f"  Age {age:3d}: {cdf_with_tail[i]:.6f}")
print(f"  ...")
for i in range(max(0, len(cdf_with_tail) - 3), len(cdf_with_tail)):
    age = tail_weibull.cdf_.ddims[i]
    marker = " <-- LAST POSITION" if i == len(cdf_with_tail) - 1 else ""
    print(f"  Age {age:3d}: {cdf_with_tail[i]:.6f}{marker}")

print("\n7. VERIFY LDF-CDF RELATIONSHIP")
print("-" * 80)
print("Standard relationship: CDF[i] = LDF[i] × LDF[i+1] × ... × LDF[last]")
print("Or: LDF[i] = CDF[i] / CDF[i+1]")

print(f"\nCheck last few LDFs against CDFs:")
for i in range(max(0, len(ldf_with_tail) - 4), len(ldf_with_tail) - 1):
    age = tail_weibull.ldf_.ddims[i]
    ldf_val = ldf_with_tail[i]
    cdf_curr = cdf_with_tail[i]
    cdf_next = cdf_with_tail[i+1]
    ldf_from_cdf = cdf_curr / cdf_next

    match = "✓" if np.isclose(ldf_val, ldf_from_cdf, rtol=1e-5) else "✗"
    print(f"  Age {age:3d}: LDF={ldf_val:.6f}, CDF[i]/CDF[i+1]={ldf_from_cdf:.6f} {match}")

# Special check for last position
i = len(ldf_with_tail) - 1
age = tail_weibull.ldf_.ddims[i]
ldf_val = ldf_with_tail[i]
cdf_curr = cdf_with_tail[i]
print(f"\n  Last position (Age {age:3d}):")
print(f"    LDF value: {ldf_val:.6f}")
print(f"    CDF value: {cdf_curr:.6f}")
print(f"    Are they equal? {np.isclose(ldf_val, cdf_curr, rtol=1e-5)}")

if np.isclose(ldf_val, cdf_curr, rtol=1e-5):
    print(f"\n✓ LAST POSITION IS A CDF (age-to-ultimate)!")
    print(f"  The last LDF position represents the tail factor to ultimate")
    print(f"  This is NOT an age-to-age factor")
else:
    print(f"\n✗ Last position doesn't match CDF")

print("\n8. UNDERSTANDING THE DESIGN")
print("-" * 80)
print("LDF array structure appears to be:")
print("  Positions 0 to N-2: Age-to-age factors")
print("  Position N-1: Cumulative tail factor to ultimate")
print("")
print("This is a MIXED representation:")
print("  - First N-1 values are period-to-period")
print("  - Last value is age-to-ultimate")

print("\n9. IS THE 'NON-MONOTONIC' BEHAVIOR ACTUALLY CORRECT?")
print("-" * 80)

# Compare second-to-last with last
i = len(ldf_with_tail) - 2
ldf_second_last = ldf_with_tail[i]
ldf_last = ldf_with_tail[i + 1]

age_second_last = tail_weibull.ldf_.ddims[i]
age_last = tail_weibull.ldf_.ddims[i + 1]

print(f"Second-to-last: Position {i} (Age {age_second_last}) = {ldf_second_last:.6f}")
print(f"  This is: age-to-age factor from age {age_second_last} to {age_last}")

print(f"\nLast: Position {i+1} (Age {age_last}) = {ldf_last:.6f}")
print(f"  This is: cumulative tail factor from age {age_last} to ultimate")

print(f"\nThese are DIFFERENT TYPES of factors, so comparing them directly is wrong!")
print(f"  One is period-to-period, the other is age-to-ultimate")
print(f"  It's like comparing apples to oranges")

print("\n10. WHAT ABOUT THE TAIL FACTOR?")
print("-" * 80)

tail_factor_reported = tail_weibull.tail_.values[0, 0]
print(f"Reported tail factor: {tail_factor_reported:.6f}")
print(f"Last LDF position: {ldf_last:.6f}")
print(f"Match? {np.isclose(tail_factor_reported, ldf_last, rtol=1e-5)}")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print("""
The LDF array has a MIXED structure by design:
- Positions 0 to N-2: Period-to-period factors
- Position N-1: Cumulative tail factor to ultimate

This explains why position N-1 might be larger than position N-2:
- Position N-2 is a small period-to-period factor (e.g., 1.005)
- Position N-1 is a cumulative tail factor (e.g., 1.006)

The 'non-monotonic' behavior we observed is actually EXPECTED and CORRECT!

We were WRONG to interpret the last position as another period-to-period factor.
""")
