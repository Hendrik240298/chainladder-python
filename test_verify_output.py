"""
Verify exactly what TailCurve outputs
"""
import numpy as np
import chainladder as cl

raa = cl.load_sample('raa')

print("=" * 80)
print("RAA Triangle Info")
print("=" * 80)
print(f"Shape: {raa.shape}")
print(f"Development dimensions (ddims): {raa.ddims}")
print(f"Number of development periods: {len(raa.ddims)}")

# Fit development
dev = cl.Development().fit(raa)
print(f"\nDevelopment LDF shape: {dev.ldf_.shape}")
print(f"Development LDF ddims: {dev.ldf_.ddims}")
print(f"Number of LDF values: {len(dev.ldf_.ddims)}")

print("\nDevelopment LDF values:")
for i, (age, ldf) in enumerate(zip(dev.ldf_.ddims, dev.ldf_.values[0, 0, 0, :])):
    print(f"  Position {i} (Age {age}): {ldf:.6f}")

# Fit Weibull tail
tail_weibull = cl.TailCurve(curve='weibull', fit_period=(12, None), extrap_periods=5).fit(raa)

print("\n" + "=" * 80)
print("Weibull Tail Output")
print("=" * 80)
print(f"Shape: {tail_weibull.ldf_.shape}")
print(f"Development dimensions (ddims): {tail_weibull.ldf_.ddims}")
print(f"Number of LDF values: {len(tail_weibull.ldf_.ddims)}")

print("\nWeibull LDF values:")
for i, (age, ldf) in enumerate(zip(tail_weibull.ldf_.ddims, tail_weibull.ldf_.values[0, 0, 0, :])):
    marker = ""
    if i > 0:
        prev_ldf = tail_weibull.ldf_.values[0, 0, 0, i-1]
        if ldf > prev_ldf:
            marker = " <-- INCREASES (BUG!)"
    print(f"  Position {i} (Age {age}): {ldf:.6f}{marker}")

print(f"\nTail factor: {tail_weibull.tail_.values[0, 0]:.6f}")

# Check attach_idx
print("\n" + "=" * 80)
print("Attachment Point Analysis")
print("=" * 80)
print(f"len(raa.ddims) = {len(raa.ddims)}")
print(f"attach_idx should be: len(X.ddims) - 1 = {len(raa.ddims) - 1}")
print(f"This means:")
print(f"  - Original LDFs kept: positions 0 to {len(raa.ddims) - 2} (inclusive)")
print(f"  - Tail predictions used: position {len(raa.ddims) - 1} onwards")
