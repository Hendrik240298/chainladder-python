"""
Debug the exact concatenation logic
"""
import numpy as np
import chainladder as cl

# Patch TailCurve to add debug prints
from chainladder.tails.curve import TailCurve
from chainladder.tails.base import TailBase

original_get_tail_prediction = TailBase._get_tail_prediction
original_predict_tail = TailCurve._predict_tail

def debug_get_tail_prediction(self, tail_ldf):
    xp = self.ldf_.get_array_module()
    print(f"\n_get_tail_prediction called:")
    print(f"  self.ldf_.shape = {self.ldf_.shape}")
    print(f"  tail_ldf.shape = {tail_ldf.shape}")

    accum_point = self.ldf_.shape[-1] - 1
    print(f"  accum_point = {accum_point}")

    ave = 1 + tail_ldf[..., :accum_point]
    print(f"  ave = 1 + tail_ldf[..., :accum_point]")
    print(f"    ave.shape = {ave.shape}")
    print(f"    ave values = {ave[0, 0, 0, :]}")

    all = xp.prod(1 + tail_ldf[..., accum_point:], -1)[..., None]
    print(f"  all = prod(1 + tail_ldf[..., accum_point:], -1)")
    print(f"    tail_ldf[..., accum_point:].shape = {tail_ldf[..., accum_point:].shape}")
    print(f"    tail_ldf[..., accum_point:] values = {tail_ldf[0, 0, 0, accum_point:]}")
    print(f"    all.shape = {all.shape}")
    print(f"    all value = {all[0, 0, 0, 0]}")

    tail = xp.concatenate((ave, all), -1)
    print(f"  tail = concatenate((ave, all), -1)")
    print(f"    tail.shape = {tail.shape}")
    print(f"    tail values = {tail[0, 0, 0, :]}")

    return tail

def debug_predict_tail(self, extrapolate):
    print(f"\n_predict_tail called:")
    print(f"  extrapolate.shape = {extrapolate.shape}")
    print(f"  extrapolate values = {extrapolate[0, 0, 0, :]}")

    result = original_predict_tail(self, extrapolate)

    print(f"  After _predict_tail:")
    print(f"    result.shape = {result.shape}")
    print(f"    result values = {result[0, 0, 0, :]}")

    return result

TailBase._get_tail_prediction = debug_get_tail_prediction
TailCurve._predict_tail = debug_predict_tail

# Now run the test
raa = cl.load_sample('raa')

print("=" * 80)
print("Running TailCurve with debug output")
print("=" * 80)

tail_weibull = cl.TailCurve(curve='weibull', fit_period=(12, None), extrap_periods=5).fit(raa)

print("\n" + "=" * 80)
print("Final Result")
print("=" * 80)
print(f"Final LDF shape: {tail_weibull.ldf_.shape}")
print(f"Final LDF ddims: {tail_weibull.ldf_.ddims}")
print(f"\nFinal LDF values:")
for i, (age, ldf) in enumerate(zip(tail_weibull.ldf_.ddims, tail_weibull.ldf_.values[0, 0, 0, :])):
    marker = ""
    if i > 0:
        prev_ldf = tail_weibull.ldf_.values[0, 0, 0, i-1]
        if ldf > prev_ldf:
            marker = " <-- INCREASES!"
    print(f"  Position {i} (Age {age}): {ldf:.6f}{marker}")
