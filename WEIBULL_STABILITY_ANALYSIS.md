# Weibull Stability Analysis Report

## Executive Summary

**YES, the Weibull implementation in chainladder-python is unstable and contains critical bugs.**

### Formula Verification
The implementation correctly uses the formula:
```
LDF(t) = 1 / (1 - exp(-a × t^b)) - 1
```

This matches your reference formula `t_t = l/(l - e^(-a t^b))` where `l=1` (growth curve), since `LDF = G(t) - 1`.

However, the implementation has **multiple critical issues** that make it produce incorrect results.

---

## Critical Issues Found

### Issue #1: Non-Monotonic LDFs (CRITICAL BUG)
**Severity:** CRITICAL
**Impact:** Violates fundamental actuarial principles

**Description:**
LDF values INCREASE instead of decrease at later development periods, violating the requirement that tail factors must decrease monotonically toward 1.0.

**Evidence:**
- RAA dataset: LDF at age 120 = 1.005186, then INCREASES to 1.005952 at age 132
- GenIns dataset: Shows non-monotonic behavior
- CLRD dataset: Shows non-monotonic behavior

**Example:**
```
Position  9 (Age 120): LDF = 1.005186
Position 10 (Age 132): LDF = 1.005952  <-- INCREASES (WRONG!)
```

**Root Cause:**
The `_get_tail_prediction()` method in [chainladder/tails/base.py:69-75](chainladder/tails/base.py#L69-L75) incorrectly treats Weibull predictions as period-to-period factors rather than age-to-ultimate factors.

```python
def _get_tail_prediction(self, tail_ldf):
    xp = self.ldf_.get_array_module()
    accum_point = self.ldf_.shape[-1] - 1
    ave = 1 + tail_ldf[..., :accum_point]
    all = xp.prod(1 + tail_ldf[..., accum_point:], -1)[..., None]  # BUG: Takes product
    tail = xp.concatenate((ave, all), -1)
    return tail
```

**What's happening:**
1. Weibull formula produces: `[LDF(t=1), LDF(t=2), ..., LDF(t=14)]` where each value represents "factor from age t to ultimate"
2. `_get_tail_prediction` incorrectly:
   - Takes first 10 values as individual period factors
   - Multiplies remaining 4 values together: `(1+LDF[11]) × (1+LDF[12]) × (1+LDF[13]) × (1+LDF[14])`
   - Places this product as position 11 in output
3. This product (1.00595158) is LARGER than the individual value at position 10 (1.00518638)
4. Result: LDFs increase instead of decrease

**Why this is wrong:**
The Weibull growth curve `G(t) = 1/(1 - exp(-a×t^b))` represents the cumulative development to ultimate from age t, NOT the development from age t to age t+1. Taking products of these values is mathematically incorrect.

---

### Issue #2: Numerical Underflow (CRITICAL)
**Severity:** CRITICAL
**Impact:** Produces invalid LDF values of 0

**Description:**
When the exponent `a × t^b` exceeds ~700, `exp(-700)` underflows to 0, causing:
```
denominator = 1 - 0 = 1
LDF = 1/1 - 1 = 0  (INVALID!)
```

**Evidence:**
```
Test case: a=0.5, b=1.5
Age  61: exponent = 37.74, exp(-37.74) ≈ 0, denominator = 1.0, LDF = 0.0
Age 100: exponent = 500, exp(-500) = 0, LDF = 0.0
```

**Location:** [chainladder/tails/curve.py:211-212](chainladder/tails/curve.py#L211-L212)

```python
tail_ldf = 1/(1-xp.exp(-xp.exp(self._intercept_)
              * extrapolate**self._slope_))-1
```

**Impact:**
All LDFs at long development periods collapse to 0, which is mathematically invalid (LDFs must be ≥ 1.0).

---

### Issue #3: Negative Shape Parameter (CRITICAL)
**Severity:** CRITICAL
**Impact:** Produces infinite tail factors and invalid LDFs < 1.0

**Description:**
When the fitted shape parameter `b` is negative, the Weibull curve grows instead of decays, producing nonsensical results.

**Evidence:**
```
CLRD dataset:
  a (scale) = 621.73
  b (shape) = -3.56  (NEGATIVE!)
  Tail factor: inf
  Invalid LDFs < 1.0 detected
```

**Root Cause:**
No validation that the shape parameter `b` must be positive. The log-linear regression used to fit the Weibull can produce negative slopes when the data doesn't follow a Weibull pattern.

**Location:** [chainladder/tails/curve.py:178-179](chainladder/tails/curve.py#L178-L179)

---

### Issue #4: Log-Log Transformation Instability
**Severity:** MEDIUM
**Impact:** Numerical instability near LDF = 1.0

**Description:**
The transformation `log(log(LDF/(LDF-1)))` becomes numerically unstable when LDF approaches 1.0.

**Evidence:**
```
LDF = 1.00000: transformation = inf
LDF = 1.00001: transformation = 2.44347
LDF = 1.00010: transformation = 2.22034
```

**Mitigation:**
The code has `reg_threshold = (1.00001, None)` to filter out LDFs too close to 1.0, which partially addresses this issue.

**Location:** [chainladder/tails/curve.py:170-171](chainladder/tails/curve.py#L170-L171)

---

## Comparison with ClarkLDF

The ClarkLDF class uses the same Weibull formula but:
1. Uses numerical optimization (scipy.minimize) instead of log-linear regression
2. Does NOT suffer from the non-monotonic issue because it calculates each LDF directly from the growth curve
3. Properly interprets the Weibull as a growth curve to ultimate

**ClarkLDF Weibull Implementation:** [chainladder/development/clark.py:67](chainladder/development/clark.py#L67)
```python
out = 1 / (1 - xp.exp(-((age / theta) ** omega)))
```

This is equivalent to TailCurve's formula but is used correctly as a growth curve.

---

## Test Results Summary

### Formula Verification Test
- ✅ Formula matches specification
- ✅ Mathematical relationship correct: `G(t) = LDF(t) + 1`

### Numerical Stability Test
- ❌ Underflow detected at age 61+ for typical parameters
- ❌ Invalid LDF = 0 produced at long development periods
- ❌ Negative shape parameters produce inf tail factors

### Monotonicity Test
- ❌ Non-monotonic behavior in ALL three test datasets (RAA, GenIns, CLRD)
- ❌ LDFs increase instead of decrease
- ❌ Violates actuarial principle

### Real Data Test
- ✅ RAA: Fits without error (but produces wrong results)
- ✅ GenIns: Fits without error (but produces wrong results)
- ❌ CLRD: Produces invalid LDFs < 1.0

---

## Affected Code Files

1. **[chainladder/tails/curve.py](chainladder/tails/curve.py)**
   - Lines 170-173: Weibull transformation
   - Lines 199-202: X-value inference
   - Lines 210-212: Weibull prediction formula
   - Lines 188-189: Concatenation logic

2. **[chainladder/tails/base.py](chainladder/tails/base.py)**
   - Lines 69-75: `_get_tail_prediction()` method (MAIN BUG)

3. **[chainladder/utils/weighted_regression.py](chainladder/utils/weighted_regression.py)**
   - Lines 21-27: `infer_x_w()` method (used for age indexing)

---

## Recommendations

### Immediate Actions (Priority 1)

1. **Fix `_get_tail_prediction()` for Weibull curves**
   - Weibull produces age-to-ultimate factors, not period-to-period factors
   - Should not take products of Weibull predictions
   - Need different logic for curve-based vs. geometric-based tail methods

2. **Add shape parameter validation**
   - Require `b > 0` for Weibull fits
   - Raise error or warning when negative shape parameter is fitted

3. **Add numerical stability checks**
   - Detect when `exp(-a×t^b)` underflows
   - Cap extrapolation at reasonable limits
   - Warn when tail factors approach 1.0 too quickly

### Secondary Actions (Priority 2)

4. **Add monotonicity validation**
   - Check that LDF[i] > LDF[i+1] for all i
   - Raise warning or error when non-monotonic LDFs detected

5. **Improve documentation**
   - Clarify that Weibull formula gives age-to-ultimate factors
   - Document the difference between curve-based and geometric-based tails

6. **Add comprehensive tests**
   - Test for monotonicity
   - Test for numerical stability at large ages
   - Test for invalid parameter values

---

## Technical Details

### Weibull Growth Curve Mathematics

The Weibull growth curve represents the proportion of ultimate losses that have emerged by age t:
```
G(t) = Ultimate / Cumulative(t) = 1 / (1 - exp(-a × t^b))
```

Where:
- `a` is the scale parameter (must be > 0)
- `b` is the shape parameter (must be > 0)
- `t` is the development age/period

**Key Properties:**
- G(t) increases monotonically: G(t₁) < G(t₂) for t₁ < t₂
- G(t) → 1 as t → ∞ (asymptotes to 1.0)
- LDF from age t to ultimate = G(t)
- LDF from age t to age t+1 = G(t) / G(t+1)

**Current Implementation Mistake:**
The code treats G(t) values as period-to-period factors and multiplies them, which is mathematically incorrect.

---

## Test Scripts Created

1. `test_weibull_stability.py` - Comprehensive stability testing
2. `test_weibull_monotonicity.py` - Detailed monotonicity analysis
3. `test_weibull_ages.py` - Age/period indexing investigation
4. `test_tail_prediction_logic.py` - Trace of `_get_tail_prediction()` logic
5. `test_verify_output.py` - Verification of actual output
6. `test_debug_concat.py` - Debug trace with instrumented code

All test scripts are in the repository root and can be run with:
```bash
python3 test_weibull_stability.py
python3 test_weibull_monotonicity.py
# etc.
```

---

## Conclusion

The Weibull implementation in `TailCurve` is **fundamentally broken** due to a conceptual error in how the Weibull growth curve values are being used. The `_get_tail_prediction()` method assumes all tail predictions are period-to-period factors that should be multiplied together, but Weibull predictions are age-to-ultimate factors that should be used directly.

This causes:
- Non-monotonic LDFs (violates actuarial principles)
- Incorrect tail factor calculations
- Unreliable results for practical use

The implementation should be either:
1. Fixed to properly handle curve-based tail factors, OR
2. Removed until a correct implementation can be developed

For reliable Weibull-based tail factors, users should use `ClarkLDF` with `growth='weibull'` instead, which correctly implements the Weibull growth curve.

---

**Report Generated:** 2025-11-17
**Analyst:** Claude Code
**Repository:** chainladder-python
**Branch:** feature/fixing_weibull_curve
