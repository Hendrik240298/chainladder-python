#!/usr/bin/env python3
"""
Comprehensive Benchmark Suite for TailCurve Weibull Implementation

This module provides extensive testing of the Weibull curve fitting in chainladder's
TailCurve class, covering normal cases, edge cases, mathematical properties,
and performance characteristics.

Author: Bernard (AI Actuarial Assistant)
Date: 2025
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest
import numpy as np
import pandas as pd
import time
import warnings
from typing import Tuple, List, Optional

import chainladder as cl
from chainladder import TailCurve, Development


class WeibullBenchmarkSuite:
    """Comprehensive benchmark suite for TailCurve Weibull implementation"""
    
    def __init__(self):
        self.tolerance = 1e-6
        self.performance_results = []
        
    # ========== TEST DATA GENERATORS ==========
    
    def create_synthetic_triangle(self, 
                                size: int = 10, 
                                base_ultimate: float = 100000,
                                development_pattern: str = "standard",
                                noise_level: float = 0.05) -> cl.Triangle:
        """
        Create synthetic loss triangle with known development patterns
        
        Parameters:
        -----------
        size : int
            Triangle size (e.g., 10 for 10x10 triangle)
        base_ultimate : float
            Base ultimate loss amount
        development_pattern : str
            Type of development pattern ('standard', 'slow_settling', 'fast_settling')
        noise_level : float
            Amount of random noise to add (0.0 to 1.0)
        """
        # Define link ratios (LDFs) for each pattern
        if development_pattern == "standard":
            # Typical commercial auto pattern - these are the LDFs
            ldfs = [2.5, 1.8, 1.4, 1.2, 1.1, 1.05, 1.03, 1.02, 1.01]
        elif development_pattern == "slow_settling":
            # Long-tail liability pattern
            ldfs = [3.0, 2.2, 1.8, 1.5, 1.3, 1.2, 1.15, 1.1, 1.08, 1.06, 1.04, 1.03, 1.02, 1.01]
        elif development_pattern == "fast_settling":
            # Property or WC pattern
            ldfs = [1.8, 1.3, 1.1, 1.05, 1.02, 1.01]
        else:
            raise ValueError(f"Unknown development pattern: {development_pattern}")
            
        # Ensure we have enough factors
        while len(ldfs) < size - 1:
            ldfs.append(1.001)
        ldfs = ldfs[:size-1]
        
        # Convert LDFs to cumulative development factors (CDFs)
        # CDF[i] = LDF[i] * LDF[i+1] * ... * LDF[n]
        cdfs = []
        running_cdf = 1.0
        for ldf in reversed(ldfs):
            running_cdf *= ldf
            cdfs.append(running_cdf)
        cdfs = list(reversed(cdfs)) + [1.0]  # Add ultimate (CDF = 1.0)
        
        # Generate triangle data
        data = {}
        origins = pd.date_range('2010-01-01', periods=size, freq='YS')
        
        np.random.seed(42)  # For reproducible results
        for i, origin in enumerate(origins):
            # Each origin year has its own ultimate
            ultimate = base_ultimate * (0.9 + 0.2 * np.random.random())
            
            for j in range(size - i):
                if noise_level > 0:
                    # Add small amount of noise but ensure cumulative values increase
                    noise = 1 + noise_level * (np.random.random() - 0.5)
                    # Bound noise to prevent negative values
                    noise = max(0.8, min(1.2, noise))
                else:
                    noise = 1
                
                # Calculate cumulative value at this development period
                cumulative = ultimate / cdfs[j] * noise
                
                # Ensure monotonicity within each origin year
                if j > 0:
                    prev_key = (origin, origin + pd.DateOffset(years=j-1))
                    if prev_key in data:
                        cumulative = max(cumulative, data[prev_key] * 1.001)
                
                data[(origin, origin + pd.DateOffset(years=j))] = cumulative
                
        # Create DataFrame
        df = pd.DataFrame([{
            'AccidentYear': k[0],
            'DevelopmentYear': k[1], 
            'values': v
        } for k, v in data.items()])
        
        # Convert to Triangle
        triangle = cl.Triangle(
            df, 
            origin='AccidentYear',
            development='DevelopmentYear', 
            columns='values',
            cumulative=True
        )
        
        return triangle
    
    def create_extreme_triangle(self, case_type: str) -> cl.Triangle:
        """Create triangles with extreme characteristics for edge case testing"""
        
        if case_type == "minimal_ldfs":
            # Triangle with LDFs very close to 1.0
            data = {
                ('2020-01-01', '2020-01-01'): 95000,
                ('2020-01-01', '2021-01-01'): 98000,
                ('2020-01-01', '2022-01-01'): 99500,
                ('2020-01-01', '2023-01-01'): 100000,
                ('2021-01-01', '2021-01-01'): 94000,
                ('2021-01-01', '2022-01-01'): 97000,
                ('2021-01-01', '2023-01-01'): 99000,
                ('2022-01-01', '2022-01-01'): 96000,
                ('2022-01-01', '2023-01-01'): 98000,
                ('2023-01-01', '2023-01-01'): 95000,
            }
            
        elif case_type == "high_volatility":
            # Triangle with very volatile development
            data = {
                ('2020-01-01', '2020-01-01'): 50000,
                ('2020-01-01', '2021-01-01'): 150000,
                ('2020-01-01', '2022-01-01'): 120000,
                ('2020-01-01', '2023-01-01'): 180000,
                ('2021-01-01', '2021-01-01'): 60000,
                ('2021-01-01', '2022-01-01'): 200000,
                ('2021-01-01', '2023-01-01'): 250000,
                ('2022-01-01', '2022-01-01'): 40000,
                ('2022-01-01', '2023-01-01'): 160000,
                ('2023-01-01', '2023-01-01'): 75000,
            }
            
        elif case_type == "small_triangle":
            # Minimal 3x3 triangle
            data = {
                ('2021-01-01', '2021-01-01'): 80000,
                ('2021-01-01', '2022-01-01'): 95000,
                ('2021-01-01', '2023-01-01'): 100000,
                ('2022-01-01', '2022-01-01'): 85000,
                ('2022-01-01', '2023-01-01'): 98000,
                ('2023-01-01', '2023-01-01'): 90000,
            }
        else:
            raise ValueError(f"Unknown extreme case type: {case_type}")
        
        # Convert to DataFrame and Triangle
        df = pd.DataFrame([{
            'AccidentYear': pd.to_datetime(k[0]),
            'DevelopmentYear': pd.to_datetime(k[1]),
            'values': v
        } for k, v in data.items()])
        
        return cl.Triangle(
            df,
            origin='AccidentYear',
            development='DevelopmentYear',
            columns='values',
            cumulative=True
        )
    
    # ========== VALIDATION HELPERS ==========
    
    def validate_actuarial_properties(self, fitted_tail) -> dict:
        """Validate that fitted tail satisfies actuarial properties"""
        results = {}
        
        # Check all LDFs >= 1.0
        ldf_values = fitted_tail.ldf_.values.flatten()
        ldf_values = ldf_values[~np.isnan(ldf_values)]
        results['all_ldfs_valid'] = np.all(ldf_values >= 1.0)
        results['min_ldf'] = np.min(ldf_values)
        results['max_ldf'] = np.max(ldf_values)
        
        # Check tail factor validity
        tail_factor = fitted_tail.tail_.iloc[0, 0] if hasattr(fitted_tail, 'tail_') else None
        results['tail_factor'] = tail_factor
        results['tail_factor_valid'] = tail_factor >= 1.0 if tail_factor is not None else None
        
        # Check for reasonable tail behavior (should generally decrease)
        tail_ldfs = fitted_tail.ldf_.values[0, 0, 0, -5:]  # Last 5 LDFs
        tail_ldfs = tail_ldfs[~np.isnan(tail_ldfs)]
        if len(tail_ldfs) > 1:
            results['tail_decreasing'] = np.all(np.diff(tail_ldfs) <= 0.01)  # Allow small increases
        else:
            results['tail_decreasing'] = None
            
        return results
    
    def validate_mathematical_consistency(self, tail_curve, triangle) -> dict:
        """Validate mathematical consistency of fitted parameters"""
        results = {}
        
        try:
            # Get fitted parameters
            intercept = tail_curve._intercept_[0, 0, 0, 0]
            slope = tail_curve._slope_[0, 0, 0, 0]
            
            results['intercept'] = intercept
            results['slope'] = slope
            results['parameters_finite'] = np.isfinite([intercept, slope]).all()
            
            # Test prediction formula manually
            test_x = np.array([1.0, 2.0, 3.0])
            manual_prediction = 1 / (1 - np.exp(-np.exp(intercept) * test_x**slope))
            results['manual_prediction_valid'] = np.all(manual_prediction >= 1.0)
            results['manual_prediction_finite'] = np.all(np.isfinite(manual_prediction))
            
        except Exception as e:
            results['error'] = str(e)
            results['parameters_accessible'] = False
            
        return results
    
    # ========== BENCHMARK TESTS ==========
    
    def test_normal_case_standard_triangle(self):
        """Test normal case with standard development pattern"""
        print("\n=== Normal Case: Standard Triangle ===")
        
        triangle = self.create_synthetic_triangle(size=10, development_pattern="standard")
        dev = Development().fit_transform(triangle)
        
        # Test different parameter combinations
        test_cases = [
            {"fit_period": (24, None), "extrap_periods": 5},
            {"fit_period": (12, 60), "extrap_periods": 10},
            {"fit_period": (36, None), "extrap_periods": 3},
        ]
        
        results = []
        for i, params in enumerate(test_cases):
            print(f"\nTest case {i+1}: {params}")
            
            start_time = time.time()
            tail = TailCurve(curve='weibull', **params)
            fitted = tail.fit_transform(dev)
            elapsed = time.time() - start_time
            
            # Validate results
            actuarial_props = self.validate_actuarial_properties(fitted)
            math_consistency = self.validate_mathematical_consistency(tail, triangle)
            
            result = {
                'test_case': i + 1,
                'parameters': params,
                'elapsed_time': elapsed,
                'actuarial_valid': actuarial_props['all_ldfs_valid'],
                'tail_factor': actuarial_props['tail_factor'],
                'min_ldf': actuarial_props['min_ldf'],
                'max_ldf': actuarial_props['max_ldf'],
                'math_consistent': math_consistency.get('parameters_finite', False),
                'intercept': math_consistency.get('intercept', np.nan),
                'slope': math_consistency.get('slope', np.nan)
            }
            
            results.append(result)
            print(f"  ✓ Actuarially valid: {result['actuarial_valid']}")
            print(f"  ✓ Tail factor: {result['tail_factor']:.6f}")
            print(f"  ✓ LDF range: [{result['min_ldf']:.6f}, {result['max_ldf']:.6f}]")
            print(f"  ✓ Runtime: {result['elapsed_time']:.4f}s")
            
        return results
    
    def test_edge_cases(self):
        """Test edge cases with extreme triangles"""
        print("\n=== Edge Cases Testing ===")
        
        edge_cases = ['minimal_ldfs', 'high_volatility', 'small_triangle']
        results = []
        
        for case_type in edge_cases:
            print(f"\nTesting {case_type}...")
            
            try:
                triangle = self.create_extreme_triangle(case_type)
                dev = Development().fit_transform(triangle)
                
                # Use conservative parameters for edge cases
                tail = TailCurve(curve='weibull', 
                               fit_period=(12, None),
                               extrap_periods=5,
                               errors='ignore',  # Important for edge cases
                               reg_threshold=(1.001, None))
                
                start_time = time.time()
                fitted = tail.fit_transform(dev)
                elapsed = time.time() - start_time
                
                # Validate results
                actuarial_props = self.validate_actuarial_properties(fitted)
                math_consistency = self.validate_mathematical_consistency(tail, triangle)
                
                result = {
                    'case_type': case_type,
                    'success': True,
                    'elapsed_time': elapsed,
                    'actuarial_valid': actuarial_props['all_ldfs_valid'],
                    'tail_factor': actuarial_props.get('tail_factor'),
                    'min_ldf': actuarial_props.get('min_ldf'),
                    'parameters_finite': math_consistency.get('parameters_finite', False),
                    'error': None
                }
                
                print(f"  ✓ Success: {result['success']}")
                print(f"  ✓ Actuarially valid: {result['actuarial_valid']}")
                print(f"  ✓ Tail factor: {result['tail_factor']}")
                
            except Exception as e:
                result = {
                    'case_type': case_type,
                    'success': False,
                    'error': str(e)
                }
                print(f"  ✗ Failed: {str(e)}")
            
            results.append(result)
        
        return results
    
    def test_parameter_sensitivity(self):
        """Test sensitivity to different parameter settings"""
        print("\n=== Parameter Sensitivity Analysis ===")
        
        # Create base triangle
        triangle = self.create_synthetic_triangle(size=8, development_pattern="standard")
        dev = Development().fit_transform(triangle)
        
        # Test parameter variations
        parameter_tests = [
            # fit_period variations
            {"name": "fit_period_early", "params": {"fit_period": (12, 36)}},
            {"name": "fit_period_late", "params": {"fit_period": (48, None)}},
            {"name": "fit_period_middle", "params": {"fit_period": (24, 60)}},
            
            # extrap_periods variations
            {"name": "extrap_short", "params": {"extrap_periods": 3}},
            {"name": "extrap_long", "params": {"extrap_periods": 15}},
            
            # reg_threshold variations
            {"name": "tight_threshold", "params": {"reg_threshold": (1.001, 2.0)}},
            {"name": "loose_threshold", "params": {"reg_threshold": (1.1, None)}},
            
            # attachment_age variations
            {"name": "early_attachment", "params": {"attachment_age": 36}},
            {"name": "late_attachment", "params": {"attachment_age": 84}},
        ]
        
        results = []
        base_params = {"curve": "weibull", "extrap_periods": 5, "errors": "ignore"}
        
        for test in parameter_tests:
            print(f"\nTesting {test['name']}...")
            
            try:
                # Merge test params with base params
                full_params = {**base_params, **test['params']}
                
                tail = TailCurve(**full_params)
                start_time = time.time()
                fitted = tail.fit_transform(dev)
                elapsed = time.time() - start_time
                
                # Analyze results
                actuarial_props = self.validate_actuarial_properties(fitted)
                math_consistency = self.validate_mathematical_consistency(tail, triangle)
                
                result = {
                    'test_name': test['name'],
                    'parameters': test['params'],
                    'success': True,
                    'elapsed_time': elapsed,
                    'tail_factor': actuarial_props.get('tail_factor'),
                    'actuarial_valid': actuarial_props['all_ldfs_valid'],
                    'intercept': math_consistency.get('intercept'),
                    'slope': math_consistency.get('slope'),
                    'error': None
                }
                
                print(f"  ✓ Success: {result['success']}")
                print(f"  ✓ Tail factor: {result['tail_factor']:.6f}")
                print(f"  ✓ Parameters: intercept={result['intercept']:.4f}, slope={result['slope']:.4f}")
                
            except Exception as e:
                result = {
                    'test_name': test['name'],
                    'parameters': test['params'],
                    'success': False,
                    'error': str(e)
                }
                print(f"  ✗ Failed: {str(e)}")
            
            results.append(result)
        
        return results
    
    def test_performance_scaling(self):
        """Test performance characteristics with different triangle sizes"""
        print("\n=== Performance Scaling Analysis ===")
        
        triangle_sizes = [5, 10, 15, 20]
        results = []
        
        for size in triangle_sizes:
            print(f"\nTesting triangle size {size}x{size}...")
            
            # Create triangle
            triangle = self.create_synthetic_triangle(size=size, development_pattern="standard")
            dev = Development().fit_transform(triangle)
            
            # Run multiple iterations for stable timing
            times = []
            for i in range(3):  # 3 runs for average
                tail = TailCurve(curve='weibull', extrap_periods=5, errors='ignore')
                start_time = time.time()
                fitted = tail.fit_transform(dev)
                elapsed = time.time() - start_time
                times.append(elapsed)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            
            result = {
                'triangle_size': size,
                'avg_time': avg_time,
                'std_time': std_time,
                'times': times
            }
            
            results.append(result)
            print(f"  ✓ Average time: {avg_time:.4f}s (±{std_time:.4f}s)")
        
        return results
    
    def test_mathematical_properties(self):
        """Test adherence to mathematical properties"""
        print("\n=== Mathematical Properties Validation ===")
        
        triangle = self.create_synthetic_triangle(size=10, development_pattern="standard")
        dev = Development().fit_transform(triangle)
        
        tail = TailCurve(curve='weibull', fit_period=(24, None), extrap_periods=10)
        fitted = tail.fit_transform(dev)
        
        # Test various mathematical properties
        tests = {}
        
        # 1. All LDFs >= 1.0
        ldf_values = fitted.ldf_.values.flatten()
        ldf_values = ldf_values[~np.isnan(ldf_values)]
        tests['all_ldfs_geq_one'] = np.all(ldf_values >= 1.0)
        tests['min_ldf'] = np.min(ldf_values)
        
        # 2. Tail convergence (LDFs should approach 1.0)
        tail_ldfs = ldf_values[-5:]  # Last 5 LDFs
        tests['tail_convergence'] = tail_ldfs[-1] <= tail_ldfs[0]  # Should decrease or stay same
        tests['tail_approaching_one'] = tail_ldfs[-1] < 1.1  # Should be close to 1
        
        # 3. Parameter reasonableness
        intercept = tail._intercept_[0, 0, 0, 0]
        slope = tail._slope_[0, 0, 0, 0]
        tests['parameters_finite'] = np.isfinite([intercept, slope]).all()
        tests['slope_reasonable'] = 0.1 <= slope <= 5.0  # Reasonable range for Weibull shape
        
        # 4. Prediction consistency
        extrapolate_test = np.array([1.0, 2.0, 5.0, 10.0])
        pred_test = 1 / (1 - np.exp(-np.exp(intercept) * extrapolate_test**slope))
        tests['predictions_valid'] = np.all(pred_test >= 1.0)
        tests['predictions_decreasing'] = np.all(np.diff(pred_test) <= 0.01)  # Should generally decrease
        
        # Print results
        print("\nMathematical Property Tests:")
        for test_name, result in tests.items():
            status = "✓" if result else "✗"
            print(f"  {status} {test_name}: {result}")
        
        return tests
    
    def run_comprehensive_benchmark(self):
        """Run the complete benchmark suite"""
        print("=" * 60)
        print("COMPREHENSIVE WEIBULL TAILCURVE BENCHMARK")
        print("=" * 60)
        
        all_results = {}
        
        # Run all test suites
        try:
            all_results['normal_cases'] = self.test_normal_case_standard_triangle()
        except Exception as e:
            print(f"Normal cases failed: {e}")
            all_results['normal_cases'] = {'error': str(e)}
        
        try:
            all_results['edge_cases'] = self.test_edge_cases()
        except Exception as e:
            print(f"Edge cases failed: {e}")
            all_results['edge_cases'] = {'error': str(e)}
        
        try:
            all_results['parameter_sensitivity'] = self.test_parameter_sensitivity()
        except Exception as e:
            print(f"Parameter sensitivity failed: {e}")
            all_results['parameter_sensitivity'] = {'error': str(e)}
        
        try:
            all_results['performance_scaling'] = self.test_performance_scaling()
        except Exception as e:
            print(f"Performance scaling failed: {e}")
            all_results['performance_scaling'] = {'error': str(e)}
        
        try:
            all_results['mathematical_properties'] = self.test_mathematical_properties()
        except Exception as e:
            print(f"Mathematical properties failed: {e}")
            all_results['mathematical_properties'] = {'error': str(e)}
        
        # Generate summary report
        self.generate_summary_report(all_results)
        
        return all_results
    
    def generate_summary_report(self, results):
        """Generate a summary report of all benchmark results"""
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY REPORT")
        print("=" * 60)
        
        # Summary statistics
        total_tests = 0
        passed_tests = 0
        
        for suite_name, suite_results in results.items():
            if isinstance(suite_results, list):
                for test in suite_results:
                    total_tests += 1
                    if test.get('success', True) and test.get('actuarial_valid', True):
                        passed_tests += 1
            elif isinstance(suite_results, dict) and 'error' not in suite_results:
                total_tests += len(suite_results)
                passed_tests += sum(1 for v in suite_results.values() if v)
        
        print(f"\nOverall Results:")
        print(f"  Total tests: {total_tests}")
        print(f"  Passed tests: {passed_tests}")
        print(f"  Success rate: {passed_tests/total_tests*100:.1f}%" if total_tests > 0 else "  No valid tests")
        
        # Performance summary
        if 'performance_scaling' in results and isinstance(results['performance_scaling'], list):
            perf_data = results['performance_scaling']
            print(f"\nPerformance Summary:")
            for perf in perf_data:
                print(f"  {perf['triangle_size']}x{perf['triangle_size']}: {perf['avg_time']:.4f}s")
        
        print(f"\nRecommendation: Weibull TailCurve implementation shows {'EXCELLENT' if passed_tests/total_tests > 0.9 else 'GOOD' if passed_tests/total_tests > 0.7 else 'NEEDS IMPROVEMENT'} performance")


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')
    
    # Initialize and run benchmark
    benchmark = WeibullBenchmarkSuite()
    results = benchmark.run_comprehensive_benchmark()
    
    # Save results to file
    import json
    with open('weibull_benchmark_results.json', 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
            
        json.dump(convert_numpy(results), f, indent=2, default=str)
    
    print(f"\nDetailed results saved to: weibull_benchmark_results.json")