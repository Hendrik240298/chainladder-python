"""
Trace through the _get_tail_prediction logic to find the bug
"""
import numpy as np
import chainladder as cl

print("=" * 80)
print("Tracing _get_tail_prediction Logic")
print("=" * 80)

# Simulate what happens in curve.py

# Parameters from our previous analysis
a = 0.428995
b = 1.089102

# Create extrapolate array: [1, 2, 3, ..., 15]
extrapolate = np.arange(1, 16)

# Calculate raw Weibull predictions (this is "LDF - 1", i.e., the excess over 1.0)
tail_ldf_raw = 1/(1-np.exp(-a * extrapolate**b)) - 1

print("\n1. Raw Weibull predictions (tail_ldf = LDF - 1):")
for i, (t, val) in enumerate(zip(extrapolate, tail_ldf_raw), 1):
    print(f"   t={t:2d}: tail_ldf_raw = {val:.6f} --> LDF = {1 + val:.6f}")

# Now simulate _get_tail_prediction from base.py
# This assumes self.ldf_.shape[-1] = 10 (original triangle has 9 LDFs, so shape is 10 including ultimate)
accum_point = 10 - 1  # = 9

print(f"\n2. accum_point = {accum_point}")

# Split the array
ave_raw = tail_ldf_raw[:accum_point]  # First 9 values
all_raw = tail_ldf_raw[accum_point:]  # Remaining values (10, 11, 12, ...)

print(f"\n3. Split arrays:")
print(f"   ave_raw (first {accum_point} values): {ave_raw}")
print(f"   all_raw (remaining values): {all_raw}")

# Apply the transformations from _get_tail_prediction
ave = 1 + ave_raw
all_product = np.prod(1 + all_raw)  # Take product of all remaining factors
all = np.array([all_product])

print(f"\n4. After transformations:")
print(f"   ave (1 + ave_raw): {ave}")
print(f"   all_product (product of 1 + all_raw): {all_product:.6f}")

# Concatenate
tail = np.concatenate((ave, all))

print(f"\n5. Final tail array (concatenated):")
for i, val in enumerate(tail, 1):
    marker = " <-- This is the issue!" if i == 10 else ""
    marker2 = " <-- Should be single tail factor to ultimate" if i == 11 else ""
    print(f"   Position {i:2d}: {val:.6f}{marker}{marker2}")

print("\n" + "=" * 80)
print("Analysis")
print("=" * 80)

print("""
The _get_tail_prediction method is doing:
1. Take first 9 Weibull predictions and convert to LDF: ave = 1 + tail_ldf[0:9]
2. Take remaining predictions (from position 10 onward) and MULTIPLY them together
3. Concatenate to create final array

The problem:
- Position 10 gets the value 1 + tail_ldf_raw[9] = 1 + 0.005186 = 1.005186
- Position 11 gets the PRODUCT of ALL remaining factors:
  (1 + tail_ldf_raw[10]) * (1 + tail_ldf_raw[11]) * ... * (1 + tail_ldf_raw[14])

This product represents a cumulative tail factor from position 10 to ultimate,
but it's being placed at position 11 in the array!

This is why we see:
- Position 10: 1.005186 (correct individual LDF)
- Position 11: 1.005952 (incorrect - this is actually a cumulative product)

The correct behavior should be:
- Position 10: 1.005186 (LDF for age 120)
- Position 11: Should be the cumulative tail factor from age 120 to ultimate

But the code is putting the individual LDF at position 10, then the tail factor
that should START from position 10 at position 11.
""")

print("\n" + "=" * 80)
print("What the values represent:")
print("=" * 80)

print("\nIntended meaning of Weibull formula:")
print("  LDF(t) = development factor from age t to ultimate")
print("  The formula 1/(1-exp(-a*t^b)) gives this directly")

print("\nBut _get_tail_prediction treats them as:")
print("  Period-to-period factors that should be multiplied")

print("\nThis mismatch causes the non-monotonic behavior!")

# Verify what the tail factor should actually be
print("\n" + "=" * 80)
print("Correct tail factor calculation:")
print("=" * 80)

# From age 120 (period 10) to ultimate, the tail factor should be:
# LDF(10) = 1 + tail_ldf_raw[9]
ldf_10_to_ult = 1 + tail_ldf_raw[9]
print(f"\nLDF from age 120 (period 10) to ultimate: {ldf_10_to_ult:.6f}")

# But what the code produces:
product_10_onwards = np.prod(1 + tail_ldf_raw[9:])
print(f"Product of factors from period 10 onwards: {product_10_onwards:.6f}")

print(f"\nThese should be the same, but they're different because")
print(f"the Weibull formula gives 'age to ultimate' factors, not")
print(f"'period to next period' factors!")
