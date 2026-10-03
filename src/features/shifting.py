"""
Shifting Feature
Adjusts sensor values by a user-specified amount.
Changes are reflected in real-time visualization.
"""
import pandas as pd
from typing import Dict, Optional
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def apply_shift(
    data: pd.DataFrame,
    shift_values: Dict[str, float],
    store_original: bool = True
) -> pd.DataFrame:
    """
    Shift sensor values by specified amounts.
    
    This function applies arithmetic shifts to specified columns in the data.
    No validation is performed - users can set unrealistic values if desired.
    
    Parameters
    ----------
    data : pd.DataFrame
        The data to shift. Can be real or synthetic data.
    shift_values : dict
        Dictionary mapping column names to shift amounts.
        Example: {"sensor_4": 50.0, "sensor_9": -20.0, "RPM": 200.0}
    store_original : bool, default=True
        Whether to store original values for reset capability.
        If True, original data should be cached elsewhere by the frontend.
    
    Returns
    -------
    pd.DataFrame
        New dataframe with shifted values applied.
        Formula: new_value = original_value + shift_amount
    
    Notes
    -----
    - Simple arithmetic operation: df[column] += shift_value
    - Works on any numeric column in the data
    - No validation on resulting values (intentional design)
    - Original data should be preserved by frontend for reset functionality
    - Shifts are cumulative: applying +30, then +20, then -10 equals +40 total
    
    Examples
    --------
    >>> data = pd.read_csv("synthetic_data.csv")
    >>> shifted_data = apply_shift(data, {"sensor_4": 50.0})
    >>> shifted_data = apply_shift(data, {"RPM": 200.0, "Temp": -15.5})
    """
    # work on a copy
    shifted_data = data.copy()

    # if nothing to shift, just return copy
    if not shift_values:
        return shifted_data

    # get numeric columns (assume this function works)
    numeric_columns = get_shiftable_columns(shifted_data)

    for column, shift_value in shift_values.items():

        # skip column if it doesn't exist
        if column not in shifted_data.columns:
            continue

        # skip if column is not numeric
        if column not in numeric_columns:
            continue

        # apply shift 
        shifted_data[column] = shifted_data[column] + float(shift_value)

    return shifted_data
    

def get_shiftable_columns(data: pd.DataFrame) -> list:
    """
    Get list of numeric columns that can be shifted.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data to analyze
    
    Returns
    -------
    list
        List of numeric column names
    
    Examples
    --------
    >>> data = pd.DataFrame({
    ...     "Engine_ID": [1, 2, 3],
    ...     "sensor_4": [10.0, 20.0, 30.0],
    ...     "name": ["a", "b", "c"]
    ... })
    >>> get_shiftable_columns(data)
    ['Engine_ID', 'sensor_4']
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"data must be a pandas DataFrame, got {type(data)}")
    
    if data.empty:
        return []
    
    return data.select_dtypes(include=[np.number]).columns.tolist()


def validate_shift_values(
    data: pd.DataFrame,
    shift_values: Dict[str, float]
) -> tuple:
    """
    Validate shift values before applying.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data to validate against
    shift_values : dict
        Shift values to validate
    
    Returns
    -------
    tuple
        (is_valid: bool, error_messages: list)
    
    Examples
    --------
    >>> data = pd.DataFrame({"sensor_4": [10, 20]})
    >>> is_valid, errors = validate_shift_values(data, {"sensor_4": 50})
    >>> print(is_valid)
    True
    """
    errors = []
    
    if not isinstance(data, pd.DataFrame):
        errors.append(f"Data must be DataFrame, got {type(data)}")
        return False, errors
    
    if not isinstance(shift_values, dict):
        errors.append(f"shift_values must be dict, got {type(shift_values)}")
        return False, errors
    
    for column, shift_value in shift_values.items():
        if column not in data.columns:
            errors.append(f"Column '{column}' not found in DataFrame")
            continue
        
        if not pd.api.types.is_numeric_dtype(data[column]):
            errors.append(f"Column '{column}' is not numeric")
            continue
        
        if not isinstance(shift_value, (int, float, np.integer, np.floating)):
            errors.append(f"Shift value for '{column}' must be numeric")
            continue
        
        if np.isnan(shift_value) or np.isinf(shift_value):
            errors.append(f"Shift value for '{column}' is NaN or infinity")
    
    return len(errors) == 0, errors


def get_recommended_shift_bounds(
    data: pd.DataFrame,
    column: str,
    multiplier: float = 2.0
) -> tuple:
    """
    Calculate recommended slider bounds for a column.
    Parameters
    ----------
    data : pd.DataFrame
        Data containing the column
    column : str
        Column name to calculate bounds for
    multiplier : float, default=2.0
        How much to extend beyond data range
    
    Returns
    -------
    tuple
        (min_bound, max_bound, step_size)
    
    Examples
    --------
    >>> data = pd.DataFrame({"sensor_4": [0, 100]})
    >>> min_val, max_val, step = get_recommended_shift_bounds(data, "sensor_4")
    >>> print(min_val, max_val, step)
    -200.0 200.0 1.0
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found")
    
    if not pd.api.types.is_numeric_dtype(data[column]):
        raise ValueError(f"Column '{column}' is not numeric")
    
    col_min = float(data[column].min())
    col_max = float(data[column].max())
    data_range = col_max - col_min
    
    if data_range == 0:
        data_range = abs(col_min) if col_min != 0 else 1.0
    
    shift_min = -data_range * multiplier
    shift_max = data_range * multiplier
    step_size = max(data_range / 100, 0.01)
    
    return (shift_min, shift_max, step_size)


def calculate_shift_statistics(
    original_data: pd.DataFrame,
    shifted_data: pd.DataFrame,
    column: str
) -> Dict[str, float]:
    """
    Calculate statistics showing impact of shift.
    
    Parameters
    ----------
    original_data : pd.DataFrame
        Data before shift
    shifted_data : pd.DataFrame
        Data after shift
    column : str
        Column to analyze
    
    Returns
    -------
    dict
        Statistics including mean, min, max differences
    
    Examples
    --------
    >>> original = pd.DataFrame({"sensor_4": [10, 20, 30]})
    >>> shifted = apply_shift(original, {"sensor_4": 50})
    >>> stats = calculate_shift_statistics(original, shifted, "sensor_4")
    >>> print(stats["mean_difference"])
    50.0
    """
    if column not in original_data.columns or column not in shifted_data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    return {
        'original_mean': float(original_data[column].mean()),
        'shifted_mean': float(shifted_data[column].mean()),
        'mean_difference': float(shifted_data[column].mean() - original_data[column].mean()),
        'original_std': float(original_data[column].std()),
        'shifted_std': float(shifted_data[column].std()),
        'original_min': float(original_data[column].min()),
        'shifted_min': float(shifted_data[column].min()),
        'min_difference': float(shifted_data[column].min() - original_data[column].min()),
        'original_max': float(original_data[column].max()),
        'shifted_max': float(shifted_data[column].max()),
        'max_difference': float(shifted_data[column].max() - original_data[column].max()),
    }


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    """Test the shifting module."""
    print("="*60)
    print("Shifting Feature - Test Run")
    print("="*60)
    
    # Create test data
    test_data = pd.DataFrame({
        'Engine_ID': [1, 1, 1, 2, 2, 2],
        'sensor_4': [10.0, 20.0, 30.0, 15.0, 25.0, 35.0],
        'sensor_9': [100.0, 110.0, 120.0, 105.0, 115.0, 125.0],
        'Altitude': [0.0, 1000.0, 2000.0, 500.0, 1500.0, 2500.0]
    })
    
    print("\nOriginal Data:")
    print(test_data)
    
    # Test 1: Get shiftable columns
    print("\n--- Test 1: Get Shiftable Columns ---")
    shiftable = get_shiftable_columns(test_data)
    print(f"Shiftable columns: {shiftable}")
    assert len(shiftable) == 4
    print("Test 1 PASSED")
    
    # Test 2: Single shift
    print("\n--- Test 2: Single Shift ---")
    shifted = apply_shift(test_data, {"sensor_4": 50.0})
    print("Shifted sensor_4 by +50:")
    print(shifted[['sensor_4']])
    assert shifted["sensor_4"].tolist() == [60.0, 70.0, 80.0, 65.0, 75.0, 85.0]
    print("Test 2 PASSED")
    
    # Test 3: Multiple shifts
    print("\n--- Test 3: Multiple Shifts ---")
    shifted = apply_shift(test_data, {"sensor_4": 50.0, "Altitude": -500.0})
    print("Shifted sensor_4 by +50 and Altitude by -500:")
    print(shifted[['sensor_4', 'Altitude']])
    assert shifted["sensor_4"].tolist() == [60.0, 70.0, 80.0, 65.0, 75.0, 85.0]
    assert shifted["Altitude"].tolist() == [-500.0, 500.0, 1500.0, 0.0, 1000.0, 2000.0]
    print("Test 3 PASSED")
    
    # Test 4: Original unchanged
    print("\n--- Test 4: Original Data Unchanged ---")
    assert test_data["sensor_4"].tolist() == [10.0, 20.0, 30.0, 15.0, 25.0, 35.0]
    print("Test 4 PASSED - Original data not modified")
    
    # Test 5: Statistics
    print("\n--- Test 5: Calculate Statistics ---")
    stats = calculate_shift_statistics(test_data, shifted, "sensor_4")
    print(f"Mean difference: {stats['mean_difference']:.2f}")
    assert stats['mean_difference'] == 50.0
    print("Test 5 PASSED")
    
    # Test 6: Validation
    print("\n--- Test 6: Validate Shift Values ---")
    is_valid, errors = validate_shift_values(test_data, {"sensor_4": 50.0})
    assert is_valid
    print("Valid shifts accepted")
    
    is_valid, errors = validate_shift_values(test_data, {"nonexistent": 10.0})
    assert not is_valid
    print(f"Invalid shifts rejected: {errors}")
    print("Test 6 PASSED")
    
    # Test 7: Recommended bounds
    print("\n--- Test 7: Recommended Slider Bounds ---")
    min_val, max_val, step = get_recommended_shift_bounds(test_data, "sensor_4")
    print(f"Bounds: [{min_val:.2f}, {max_val:.2f}], step={step:.2f}")
    assert min_val < 0
    assert max_val > 0
    print("Test 7 PASSED")
    
    # Test 8: Non-existent column (graceful handling)
    print("\n--- Test 8: Non-existent Column ---")
    shifted = apply_shift(test_data, {"sensor_999": 100.0})
    assert shifted.equals(test_data)
    print("Test 8 PASSED")
    
    # Test 9: Empty DataFrame
    print("\n--- Test 9: Empty DataFrame ---")
    empty_df = pd.DataFrame()
    result = apply_shift(empty_df, {"sensor_4": 50.0})
    assert result.empty
    print("Test 9 PASSED")
    
    # Test 10: No shift values
    print("\n--- Test 10: No Shift Values ---")
    result = apply_shift(test_data, {})
    assert result.equals(test_data)
    print("Test 10 PASSED")
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)
    print("\nShifting module is production ready!")