"""
Test what happens with invalid parameters and propose validation
"""
import numpy as np
import chainladder as cl
import warnings

print("=" * 80)
print("Parameter Validation Testing")
print("=" * 80)

# Test Case 1: What happens with negative shape parameter?
print("\n1. SIMULATING NEGATIVE SHAPE PARAMETER")
print("-" * 80)

print("Testing what happens when b < 0 in formula:")
print("  LDF(t) = 1/(1 - exp(-a × t^b)) - 1")

a = 621.73  # From CLRD fit
b = -3.56   # Negative shape parameter
t_values = np.array([1, 2, 3, 5, 10, 20, 50, 100])

print(f"\nWith a={a:.2f}, b={b:.2f}:")
for t in t_values:
    try:
        exponent = -a * (t ** b)
        exp_val = np.exp(exponent)
        denominator = 1 - exp_val
        ldf = 1/denominator - 1

        print(f"  t={t:3d}: exp(-a×t^b)={exp_val:10.6f}, 1-exp(...)={denominator:10.6f}, LDF={ldf:10.6f}")

        if ldf < 0:
            print(f"        ⚠️  NEGATIVE LDF!")
        if np.isinf(ldf):
            print(f"        ⚠️  INFINITE LDF!")

    except Exception as e:
        print(f"  t={t:3d}: ERROR - {e}")

# Test Case 2: When does positive b produce reasonable values?
print("\n\n2. COMPARISON WITH POSITIVE SHAPE PARAMETER")
print("-" * 80)

a_good = 0.38  # From GenIns fit
b_good = 1.06  # Positive shape parameter

print(f"\nWith a={a_good:.2f}, b={b_good:.2f}:")
for t in t_values:
    exponent = -a_good * (t ** b_good)
    exp_val = np.exp(exponent)
    denominator = 1 - exp_val
    ldf = 1/denominator - 1

    print(f"  t={t:3d}: LDF={ldf:.6f}")

# Test Case 3: Check current validation in chainladder
print("\n\n3. CURRENT VALIDATION IN CHAINLADDER")
print("-" * 80)

datasets_to_test = [
    ('RAA (good)', cl.load_sample('raa')),
    ('GenIns (non-monotonic obs)', cl.load_sample('genins')),
    ('CLRD single (negative dev)', cl.load_sample('clrd').iloc[0, 0]),
]

for name, data in datasets_to_test:
    print(f"\n{name}:")

    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            tail = cl.TailCurve(curve='weibull', fit_period=(12, None)).fit(data)

            a = np.exp(tail.intercept_.values[0, 0])
            b = tail.slope_.values[0, 0]
            tail_f = tail.tail_.values[0, 0]

            # Check for warnings
            if w:
                print(f"  Warnings raised: {len(w)}")
                for warning in w:
                    print(f"    - {warning.message}")
            else:
                print(f"  No warnings raised")

            # Report parameters
            print(f"  Parameters: a={a:.6f}, b={b:.6f}")
            print(f"  Tail factor: {tail_f:.6f}")

            # Check for problems
            problems = []
            if b <= 0:
                problems.append(f"NEGATIVE/ZERO SHAPE (b={b:.4f})")
            if np.isinf(tail_f):
                problems.append("INFINITE TAIL")
            if np.isnan(tail_f):
                problems.append("NAN TAIL")
            if tail_f < 1.0 and not np.isinf(tail_f):
                problems.append(f"INVALID TAIL (< 1.0)")

            if problems:
                print(f"  ⚠️  PROBLEMS: {', '.join(problems)}")
            else:
                print(f"  ✓ OK")

    except Exception as e:
        print(f"  ❌ ERROR: {e}")

# Test Case 4: What validation SHOULD exist?
print("\n\n4. PROPOSED VALIDATION CHECKS")
print("-" * 80)

print("""
Validation that should be added to TailCurve:

1. SHAPE PARAMETER CHECK (after fitting):
   if self._slope_ <= 0:
       raise ValueError(
           f"Weibull shape parameter must be positive, got {self._slope_:.4f}. "
           "This typically indicates data is unsuitable for Weibull fitting. "
           "Consider using curve='exponential' or check for negative development."
       )

2. INPUT DATA CHECK (before fitting):
   observed_ldfs = # calculate from triangle
   if np.any(observed_ldfs < 1.0):
       warnings.warn(
           "Triangle contains negative development (LDF < 1.0). "
           "Weibull growth curve assumes strictly positive development. "
           "Results may be unreliable."
       )

3. TAIL FACTOR VALIDATION (after prediction):
   if np.isinf(self.tail_) or np.isnan(self.tail_):
       raise ValueError(
           f"Invalid tail factor: {self.tail_}. "
           "This may indicate numerical instability or unsuitable data."
       )
   if self.tail_ < 1.0:
       raise ValueError(
           f"Tail factor must be >= 1.0, got {self.tail_:.6f}"
       )

4. NUMERICAL STABILITY CHECK (during prediction):
   # Check for underflow/overflow in exponential
   exponent = -a * t^b
   if exponent > 700:  # Will underflow to 0
       warnings.warn(
           f"Numerical underflow detected at t={t} with exponent={exponent:.1f}. "
           "Consider reducing extrapolation periods."
       )
""")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print("""
ROOT CAUSE: Negative shape parameter (b < 0) produces infinite tail factors.

TRIGGER: Data with negative development (LDF < 1.0) causes regression to fit
         negative shape parameter.

CURRENT STATE: No validation exists. Silently produces inf tail factors.

FIX PRIORITY:
1. Add shape parameter validation (b > 0) - CRITICAL
2. Add input data quality checks - HIGH
3. Add numerical stability checks - MEDIUM
4. Improve error messages - HIGH

The instability you experienced was likely caused by negative development
in your data. The fix is not in _get_tail_prediction() but in adding
proper validation and error handling.
""")
