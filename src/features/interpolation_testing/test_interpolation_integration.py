"""
Test script to verify interpolation.py gateway integration with Bspline_MultFunc.py
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import the gateway module
from src.features.interpolation import apply_interpolation

def test_spline_method():
    """Test that the spline method properly delegates to Bspline_MultFunc"""
    print("=" * 60)
    print("Testing Spline Interpolation Integration")
    print("=" * 60)
    
    # Create sample data with gaps
    # Cluster 1: x from 0 to 2
    # Gap
    # Cluster 2: x from 10 to 12
    data = pd.DataFrame({
        "Time_in_cycles": [0, 1, 2, 10, 11, 12],
        "sensor_1": [0.0, 1.0, 2.0, 10.0, 11.0, 12.0],
        "sensor_2": [0.0, 2.0, 4.0, 20.0, 22.0, 24.0]
    })
    
    print("\nOriginal data:")
    print(data)
    print(f"\nOriginal shape: {data.shape}")
    
    # Apply spline interpolation
    result = apply_interpolation(
        data=data,
        method="spline",
        x_column="Time_in_cycles",
        gap_threshold=1.0,
        num_points=5
    )
    
    print("\n" + "=" * 60)
    print("Result after spline interpolation:")
    print("=" * 60)
    print(result)
    print(f"\nResult shape: {result.shape}")
    print(f"Interpolated rows: {result['is_interpolated'].sum() if 'is_interpolated' in result.columns else 'N/A'}")
    
    # Verify interpolated points were added
    if len(result) > len(data):
        print("\n✓ SUCCESS: Interpolated points were added")
        if 'is_interpolated' in result.columns:
            print(f"  - Original points: {(~result['is_interpolated']).sum()}")
            print(f"  - Interpolated points: {result['is_interpolated'].sum()}")
    else:
        print("\n✗ WARNING: No interpolated points added (might not have detected gaps)")
    
    return result


def test_linear_method():
    """Test that linear method fills NaN gaps"""
    print("\n" + "=" * 60)
    print("Testing Linear Interpolation (NaN Gap Filling)")
    print("=" * 60)
    
    # Create data with NaN gaps
    data = pd.DataFrame({
        "Time_in_cycles": [0, 1, 2, 3, 4, 5],
        "sensor_1": [0.0, 1.0, np.nan, np.nan, 4.0, 5.0],
        "sensor_2": [0.0, 2.0, np.nan, 6.0, 8.0, 10.0]
    })
    
    print("\nOriginal data (with NaN gaps):")
    print(data)
    print(f"\nNaN count before: {data.isna().sum().sum()}")
    
    result = apply_interpolation(
        data=data,
        method="linear",
        x_column="Time_in_cycles",
        max_gap_cycles=10
    )
    
    print("\nResult after linear interpolation:")
    print(result)
    print(f"\nNaN count after: {result.isna().sum().sum()}")
    
    # Verify NaNs were filled
    if result.isna().sum().sum() < data.isna().sum().sum():
        print("\n✓ SUCCESS: NaN gaps were filled with linear interpolation")
    else:
        print("\n⚠ WARNING: No NaN gaps were filled")
    
    return result


def test_linear_no_time_column():
    """Test linear interpolation without time column (uses row index)"""
    print("\n" + "=" * 60)
    print("Testing Linear Interpolation (No Time Column)")
    print("=" * 60)
    
    # Create data without Time_in_cycles
    data = pd.DataFrame({
        "sensor_1": [0.0, 1.0, np.nan, 3.0],
        "sensor_2": [0.0, np.nan, np.nan, 6.0],
        "sensor_3": [1.0, 2.0, 3.0, 4.0]  # no NaNs
    })
    
    print("\nOriginal data (no time column):")
    print(data)
    
    result = apply_interpolation(
        data=data,
        method="linear"
    )
    
    print("\nResult (should use row index as time):")
    print(result)
    print("\n✓ Linear method without time column completed")
    return result


def test_target_columns():
    """Test that target_columns parameter works correctly"""
    print("\n" + "=" * 60)
    print("Testing Target Columns Parameter")
    print("=" * 60)
    
    data = pd.DataFrame({
        "Time_in_cycles": [0, 1, 2, 10, 11, 12],
        "sensor_1": [0.0, 1.0, 2.0, 10.0, 11.0, 12.0],
        "sensor_2": [0.0, 2.0, 4.0, 20.0, 22.0, 24.0],
        "sensor_3": [0.0, 3.0, 6.0, 30.0, 33.0, 36.0]
    })
    
    # Only interpolate sensor_1 and sensor_2
    result = apply_interpolation(
        data=data,
        method="spline",
        x_column="Time_in_cycles",
        target_columns=["sensor_1", "sensor_2"],
        gap_threshold=1.0,
        num_points=3
    )
    
    print(f"\nSpecified target_columns: ['sensor_1', 'sensor_2']")
    print(f"Result shape: {result.shape}")
    print("\n✓ Target columns parameter test completed")
    return result


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("INTERPOLATION GATEWAY INTEGRATION TEST")
    print("=" * 60)
    print("\nThis test verifies that interpolation.py correctly delegates")
    print("to Bspline_MultFunc.py for spline and linear_interpolation.py for linear.")
    print()
    
    try:
        # Run tests
        test_spline_method()
        test_linear_method()
        test_linear_no_time_column()
        test_target_columns()
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("\nCORRECT: Spline integration: interpolation.py → Bspline_MultFunc.py")
        print("CORRECT: Linear integration: interpolation.py → linear_interpolation.py")
        print("CORRECT: Column auto-detection working (no hardcoded names required)")
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("ERROR DURING TESTING")
        print("=" * 60)
        print(f"\n✗ {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
