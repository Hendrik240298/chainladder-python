#!/usr/bin/env python3
"""
TailCurve vs TailBondy Comparison Example

This example demonstrates the differences between TailCurve and TailBondy methods
when dealing with challenging development data that includes:
- Negative development values
- Extensive NaN patterns
- Sparse early development
- LDFs < 1.0

Based on real-world data scenario where TailCurve fails but TailBondy succeeds.

Author: Bernard (AI Actuarial Assistant)
Date: 2025
"""

import numpy as np
import pandas as pd
import chainladder as cl
import warnings
warnings.filterwarnings('ignore')

def create_challenging_triangle():
    """
    Create a triangle with challenging characteristics similar to user's data:
    - Negative development values
    - Extensive NaN patterns
    - Sparse early development
    """
    # Based on the user's actual data pattern
    data = {
        'AccidentYear': [],
        'DevelopmentPeriod': [],  
        'values': []
    }
    
    # Accident years
    accident_years = [2020, 2021, 2022, 2023, 2024, 2025]
    development_periods = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60]
    
    # Sample data based on user's pattern (simplified for example)
    triangle_data = {
        2020: [np.nan, np.nan, np.nan, np.nan, 290630, 1403700, 3286000, 1390000, 1084900, 
               1038300, 20818, 567166, 109110, 243552, 23377, -142638, 190268, 183152, 42256, -68306],
        2021: [np.nan, np.nan, 45165, 406110, 786190, 2051200, 1399000, 1109300, 1071700, 
               440520, 106520, 147097, 143480, 260484, 146916, -34234, 359018, np.nan, -278715, np.nan],
        2022: [np.nan, 41593, 251491, 578580, 1379400, 4025800, 2561800, 2784300, 1770500, 
               965140, 124150, 716833, 1577600, np.nan, -297568, np.nan, np.nan, np.nan, np.nan, np.nan],
        2023: [np.nan, 6305, 235920, 236790, 1961600, 1796200, 1682700, 1000800, 1103200, 
               np.nan, 3346400, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        2024: [np.nan, 103055, 319860, 1747800, 790090, np.nan, 5273700, np.nan, np.nan, 
               np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        2025: [np.nan, np.nan, 363519, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 
               np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    }
    
    # Build DataFrame
    for ay in accident_years:
        for i, dp in enumerate(development_periods):
            if i < len(triangle_data[ay]):
                value = triangle_data[ay][i]
                if not pd.isna(value):
                    data['AccidentYear'].append(f"{ay}-01-01")
                    data['DevelopmentPeriod'].append(dp)
                    data['values'].append(value)
    
    df = pd.DataFrame(data)
    df['AccidentYear'] = pd.to_datetime(df['AccidentYear'])
    
    return cl.Triangle(df, 
                      origin='AccidentYear',
                      development='DevelopmentPeriod',
                      columns='values',
                      cumulative=True)

def analyze_triangle_characteristics(triangle):
    """Analyze the challenging characteristics of the triangle"""
    print("=== TRIANGLE CHARACTERISTICS ANALYSIS ===")
    print(f"Triangle shape: {triangle.shape}")
    print(f"Development grain: {triangle.development_grain}")
    print(f"Development periods: {triangle.ddims}")
    
    # Check for NaN patterns
    nan_count = np.sum(np.isnan(triangle.values))
    total_count = np.prod(triangle.values.shape)
    print(f"NaN percentage: {nan_count/total_count*100:.1f}%")
    
    # Check for negative values
    incremental = triangle.cum_to_incr()
    negative_count = np.sum(incremental.values < 0)
    print(f"Negative incremental values: {negative_count}")
    
    # Calculate LDFs and check validity
    dev = cl.Development().fit_transform(triangle)
    ldfs = dev.ldf_.values
    
    valid_ldfs = ldfs[~np.isnan(ldfs)]
    min_ldf = np.min(valid_ldfs) if len(valid_ldfs) > 0 else np.nan
    ldfs_below_one = np.sum(valid_ldfs < 1.0) if len(valid_ldfs) > 0 else 0
    
    print(f"Min LDF: {min_ldf:.6f}")
    print(f"LDFs < 1.0: {ldfs_below_one}")
    print(f"Valid LDFs for fitting: {len(valid_ldfs)}")
    
    return dev

def test_tailcurve(dev):
    """Test TailCurve method"""
    print("\n=== TAILCURVE TESTING ===")
    
    # Test different curve types and parameters
    curve_types = ['exponential', 'inverse_power', 'weibull']
    results = {}
    
    for curve_type in curve_types:
        print(f"\nTesting {curve_type} curve:")
        
        try:
            tail = cl.TailCurve(
                curve=curve_type,
                fit_period=(12, None),  # Fit from 12 months onward
                errors='ignore',        # Handle edge cases
                reg_threshold=(1.001, None)  # Filter extreme values
            )
            
            fitted = tail.fit_transform(dev.copy())
            
            # Check if parameters are valid
            if hasattr(tail, '_intercept_') and hasattr(tail, '_slope_'):
                intercept = tail._intercept_[0,0,0,0] if tail._intercept_.size > 0 else np.nan
                slope = tail._slope_[0,0,0,0] if tail._slope_.size > 0 else np.nan
                params_valid = np.isfinite([intercept, slope]).all()
            else:
                params_valid = False
                intercept = slope = np.nan
            
            tail_factor = fitted.tail_.iloc[0,0] if hasattr(fitted, 'tail_') else np.nan
            
            results[curve_type] = {
                'success': True,
                'tail_factor': tail_factor,
                'intercept': intercept,
                'slope': slope,
                'params_valid': params_valid,
                'error': None
            }
            
            print(f"  ✓ SUCCESS")
            print(f"    Tail factor: {tail_factor:.6f}")
            if params_valid:
                print(f"    Parameters: intercept={intercept:.4f}, slope={slope:.4f}")
            else:
                print(f"    Parameters: INVALID")
                
        except Exception as e:
            results[curve_type] = {
                'success': False,
                'error': str(e)
            }
            print(f"  ✗ FAILED: {str(e)}")
    
    return results

def test_tailbondy(dev):
    """Test TailBondy method"""
    print("\n=== TAILBONDY TESTING ===")
    
    # Test different earliest_age parameters
    earliest_ages = [None, 12, 18, 24, 30]
    results = {}
    
    for earliest_age in earliest_ages:
        age_desc = f"earliest_age={earliest_age}" if earliest_age else "earliest_age=None"
        print(f"\nTesting {age_desc}:")
        
        try:
            tail = cl.TailBondy(
                earliest_age=earliest_age,
                attachment_age=None  # Let it choose automatically
            )
            
            fitted = tail.fit_transform(dev.copy())
            
            tail_factor = fitted.tail_.iloc[0,0]
            bondy_exponent = fitted.b_.iloc[0,0]
            earliest_ldf = fitted.earliest_ldf_.iloc[0,0]
            
            results[earliest_age] = {
                'success': True,
                'tail_factor': tail_factor,
                'bondy_exponent': bondy_exponent,
                'earliest_ldf': earliest_ldf,
                'error': None
            }
            
            print(f"  ✓ SUCCESS")
            print(f"    Tail factor: {tail_factor:.6f}")
            print(f"    Bondy exponent (b): {bondy_exponent:.6f}")
            print(f"    Earliest LDF: {earliest_ldf:.6f}")
            
        except Exception as e:
            results[earliest_age] = {
                'success': False,
                'error': str(e)
            }
            print(f"  ✗ FAILED: {str(e)}")
    
    return results

def compare_results(tailcurve_results, tailbondy_results):
    """Compare and summarize results"""
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    # TailCurve summary
    tc_success_count = sum(1 for r in tailcurve_results.values() if r['success'])
    tc_total = len(tailcurve_results)
    
    print(f"\nTailCurve Results:")
    print(f"  Success rate: {tc_success_count}/{tc_total} ({tc_success_count/tc_total*100:.1f}%)")
    
    if tc_success_count > 0:
        valid_tc_results = [r for r in tailcurve_results.values() if r['success']]
        tc_tail_factors = [r['tail_factor'] for r in valid_tc_results if np.isfinite(r['tail_factor'])]
        if tc_tail_factors:
            print(f"  Tail factor range: {min(tc_tail_factors):.4f} - {max(tc_tail_factors):.4f}")
    
    # TailBondy summary  
    tb_success_count = sum(1 for r in tailbondy_results.values() if r['success'])
    tb_total = len(tailbondy_results)
    
    print(f"\nTailBondy Results:")
    print(f"  Success rate: {tb_success_count}/{tb_total} ({tb_success_count/tb_total*100:.1f}%)")
    
    if tb_success_count > 0:
        valid_tb_results = [r for r in tailbondy_results.values() if r['success']]
        tb_tail_factors = [r['tail_factor'] for r in valid_tb_results if np.isfinite(r['tail_factor'])]
        if tb_tail_factors:
            print(f"  Tail factor range: {min(tb_tail_factors):.4f} - {max(tb_tail_factors):.4f}")
    
    # Recommendations
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS")
    print("="*60)
    
    if tb_success_count > tc_success_count:
        print("\n✅ RECOMMENDATION: Use TailBondy")
        print("   Reasons:")
        print("   - Higher success rate with challenging data")
        print("   - More robust to negative development")
        print("   - Better handling of sparse/irregular patterns")
        print("   - Actuarial methodology designed for real-world data")
        
        if tb_success_count > 0:
            # Find best TailBondy result
            best_tb = min(valid_tb_results, key=lambda x: abs(x['tail_factor'] - 1.05))  # Closest to reasonable 5% tail
            print(f"\n   Recommended parameters:")
            for age, result in tailbondy_results.items():
                if result == best_tb:
                    print(f"   - earliest_age={age}")
                    print(f"   - Expected tail factor: {result['tail_factor']:.4f}")
                    break
    else:
        print("\n✅ RECOMMENDATION: Use TailCurve")
        print("   Note: This is unusual for challenging data patterns")
    
    print(f"\n{'='*60}")
    print("USAGE EXAMPLE")
    print("="*60)
    
    if tb_success_count > 0:
        print("\n# Recommended approach for challenging data:")
        print("import chainladder as cl")
        print("")
        print("# Load your triangle")
        print("triangle = your_triangle_data")
        print("dev = cl.Development().fit_transform(triangle)")
        print("")
        print("# Apply TailBondy (more robust)")
        print("tail = cl.TailBondy(")
        print("    earliest_age=24,        # Adjust based on your data")
        print("    attachment_age=None     # Auto-select attachment point")
        print(")")
        print("")
        print("fitted = tail.fit_transform(dev)")
        print("print(f'Tail factor: {fitted.tail_.iloc[0,0]:.4f}')")
        print("print(f'Bondy exponent: {fitted.b_.iloc[0,0]:.4f}')")

def main():
    """Main execution function"""
    print("TailCurve vs TailBondy Comparison")
    print("=" * 50)
    print("Testing with challenging development data:")
    print("- Negative development values")
    print("- Extensive NaN patterns")  
    print("- LDFs < 1.0")
    print("- Sparse early development")
    print("")
    
    # Create challenging triangle
    triangle = create_challenging_triangle()
    
    # Analyze triangle characteristics
    dev = analyze_triangle_characteristics(triangle)
    
    # Test both methods
    tailcurve_results = test_tailcurve(dev)
    tailbondy_results = test_tailbondy(dev)
    
    # Compare and provide recommendations
    compare_results(tailcurve_results, tailbondy_results)

if __name__ == "__main__":
    main()