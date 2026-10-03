"""
Interpolation Feature Gateway
==============================
Acts as a central interface for applying various interpolation methods to
synthetic data. Routes requests to appropriate specialized implementations
(B-spline, linear, ML-based) while providing a consistent API.

This module serves as a connector between the frontend/application layer
and the underlying interpolation implementations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from pathlib import Path
from typing import Optional, Literal, Tuple
import sys

# Import B-spline implementation
from . import Bspline_MultFunc as bspline

# Import linear interpolation implementation
from .linear_interpolation import (
    find_nan_runs,
    interpolate_engine,
    detect_constant_columns,
    Config as LinearConfig,
)


# ── resolve linear_interpolation from src/features/ regardless of working dir
_FEATURES_DIR = Path(__file__).parent
if str(_FEATURES_DIR) not in sys.path:
    sys.path.insert(0, str(_FEATURES_DIR))


# ID columns that are never interpolated
_ID_COLS = ["Engine_ID", "Time_in_cycles"]


def apply_interpolation(
    data: pd.DataFrame,
    method: Literal["linear", "spline", "ml"] = "linear",
    target_columns: Optional[list] = None,
    x_column: Optional[str] = None,
    gap_threshold: float = 1.0,
    num_points: int = 10,
    max_gap_cycles: int = 10,
    engine_id_column: Optional[str] = None
) -> pd.DataFrame:
    """
    Apply interpolation to connect clusters/gaps in synthetic data.
    
    This function acts as a gateway, routing to the appropriate interpolation
    implementation based on the specified method. It identifies disconnected 
    clusters in the data and generates intermediate points to create smoother 
    transitions between clusters.
    
    Parameters
    ----------
    data : pd.DataFrame
        The synthetic data to interpolate. Must contain numeric columns.
        Expected to have Engine_ID and Time_in_cycles columns (CMAPSS schema).
    method : str, default="linear"
        Interpolation method to use:
        - "linear": Linear interpolation for filling NaN gaps (uses linear_interpolation)
        - "spline": B-spline interpolation for smoother curves (uses Bspline_MultFunc)
        - "ml": ML-based interpolation (advanced version, not yet implemented)
    target_columns : list, optional
        List of column names to apply interpolation to.
        If None, applies to all numeric columns (excluding ID columns).
    x_column : str, optional
        Name of the column to use as the independent variable (x-axis/time).
        If None, auto-detects "Time_in_cycles" or uses row index.
        Used for both linear and spline methods.
    gap_threshold : float, default=1.0
        Minimum spacing between consecutive x-values that constitutes a gap.
        Only used for "spline" method.
    num_points : int, default=10
        Number of interpolation points to generate within each detected gap.
        Only used for "spline" method.
    max_gap_cycles : int, default=10
        Maximum gap size (in cycles) to fill for linear interpolation.
        Gaps larger than this are left as NaN.
        Only used for "linear" method.
    engine_id_column : str, optional
        Name of the column containing engine IDs for per-engine interpolation.
        If None, auto-detects "Engine_ID" or treats entire dataset as one series.
        Only used for "linear" method.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with interpolated data.
        - For "linear" method: NaN gaps filled in-place, original structure preserved
        - For "spline" method: New rows added between clusters with "is_interpolated" column
        Original data is always preserved.
    
    Notes
    -----
    - Linear method: fills existing NaN gaps between valid data points
    - Spline method: adds new rows to connect separated data clusters
    - No hardcoded column names required - works with any DataFrame structure
    - Auto-detects common column names (Engine_ID, Time_in_cycles) if present
    - ID columns and constant columns are automatically excluded from interpolation
    
    Examples
    --------
    >>> synth_data = pd.read_csv("synthetic_data.csv")
    >>> 
    >>> # Linear interpolation (auto-detect time column)
    >>> interpolated_data = apply_interpolation(synth_data, method="linear")
    >>> 
    >>> # Linear interpolation with explicit parameters
    >>> interpolated_data = apply_interpolation(
    ...     synth_data,
    ...     method="linear",
    ...     x_column="Time_in_cycles",
    ...     engine_id_column="Engine_ID",
    ...     max_gap_cycles=20
    ... )
    >>> 
    >>> # Spline interpolation on all numeric columns
    >>> interpolated_data = apply_interpolation(
    ...     synth_data, 
    ...     method="spline",
    ...     x_column="Time_in_cycles"
    ... )
    >>> 
    >>> # Spline interpolation on specific columns
    >>> interpolated_data = apply_interpolation(
    ...     synth_data,
    ...     method="spline", 
    ...     target_columns=["sensor_4", "sensor_9"],
    ...     gap_threshold=2.0,
    ...     num_points=15
    ... )
    """
    if method == "linear":
        # Delegate to linear interpolation implementation
        return _apply_linear_interpolation(
            data=data,
            target_columns=target_columns,
            x_column=x_column,
            engine_id_column=engine_id_column,
            max_gap_cycles=max_gap_cycles
        )
    
    elif method == "spline":
        # Delegate to B-spline implementation
        # Auto-detect x_column if not provided
        if x_column is None:
            if "Time_in_cycles" in data.columns:
                x_column = "Time_in_cycles"
            else:
                # Use first numeric column as fallback
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    x_column = numeric_cols[0]
                    print(f"[interpolation] Auto-detected x_column: {x_column}")
                else:
                    raise ValueError(
                        "No numeric columns found for x_column. "
                        "Please specify x_column parameter."
                    )
        
        _, interpolated_df = bspline.apply_interpolation(
            data=data,
            x_column=x_column,
            method="spline",
            target_columns=target_columns,
            gap_threshold=gap_threshold,
            num_points=num_points
        )
        return interpolated_df
    
    elif method == "ml":
        # TODO: Implement ML-based interpolation
        print(f"[interpolation] ML-based interpolation not yet implemented.")
        print(f"Returning original data with shape {data.shape}")
        if target_columns:
            print(f"Target columns would be: {target_columns}")
        return data.copy()
    
    else:
        raise ValueError(
            f"Unsupported interpolation method: '{method}'. "
            f"Supported methods: 'linear', 'spline', 'ml'"
        )


def _apply_linear_interpolation(
    data: pd.DataFrame,
    target_columns: Optional[list] = None,
    x_column: Optional[str] = None,
    engine_id_column: Optional[str] = None,
    max_gap_cycles: int = 10
) -> pd.DataFrame:
    """
    Internal function to apply linear interpolation with column auto-detection.
    
    Handles three scenarios:
    1. Has both engine ID and time columns: per-engine interpolation
    2. Has only time column: single time-series interpolation
    3. No time column: use row index as synthetic time axis
    """
    # Make a copy to avoid mutating the original
    df = data.copy()
    df.columns = df.columns.str.strip()
    
    # Auto-detect engine ID column
    if engine_id_column is None:
        if "Engine_ID" in df.columns:
            engine_id_column = "Engine_ID"
    
    # Auto-detect time column
    if x_column is None:
        if "Time_in_cycles" in df.columns:
            x_column = "Time_in_cycles"
    
    has_engine_id = engine_id_column is not None and engine_id_column in df.columns
    has_time_col = x_column is not None and x_column in df.columns
    
    # Determine ID columns to exclude from interpolation
    id_cols = []
    if has_engine_id:
        id_cols.append(engine_id_column)
    if has_time_col:
        id_cols.append(x_column)
    
    # Determine which columns to interpolate
    all_sensor_cols = [
        c for c in df.columns 
        if c not in id_cols and pd.api.types.is_numeric_dtype(df[c])
    ]
    
    const_cols = detect_constant_columns(df, all_sensor_cols)
    default_active = [c for c in all_sensor_cols if c not in const_cols]
    
    if target_columns is not None:
        # Honor user selection but skip any non-numeric or constant columns
        active_cols = [
            c for c in target_columns
            if c in df.columns
            and pd.api.types.is_numeric_dtype(df[c])
            and c not in id_cols
        ]
    else:
        active_cols = default_active
    
    if not active_cols:
        print("[interpolation] No active columns to interpolate.")
        return df
    
    cfg = LinearConfig(max_gap_cycles=max_gap_cycles, simulate=False)
    
    # Apply interpolation based on available columns
    # ── per-engine interpolation ──────────────────────────────────────────────
    if has_engine_id and has_time_col:
        # Standard path: per-engine interpolation with time axis
        print(f"[interpolation] Linear interpolation per {engine_id_column} using {x_column}")
        for eid in df[engine_id_column].unique():
            sel = df[engine_id_column] == eid
            edf = df.loc[sel]
            cycles = edf[x_column].to_numpy(dtype=np.float64)
            matrix = edf[active_cols].to_numpy(dtype=np.float64)
            
            filled_mat, _, _ = interpolate_engine(cycles, matrix, active_cols, cfg)
            df.loc[sel, active_cols] = filled_mat
    
    elif has_time_col:
        # No engine grouping - treat whole dataset as one time series
        print(f"[interpolation] Linear interpolation using {x_column} as time axis")
        cycles = df[x_column].to_numpy(dtype=np.float64)
        matrix = df[active_cols].to_numpy(dtype=np.float64)
        
        filled_mat, _, _ = interpolate_engine(cycles, matrix, active_cols, cfg)
        df[active_cols] = filled_mat
    
    else:
        # No time axis - use row index as synthetic time
        print("[interpolation] Linear interpolation using row index as time axis")
        cycles = np.arange(len(df), dtype=np.float64)
        matrix = df[active_cols].to_numpy(dtype=np.float64)
        
        filled_mat, _, _ = interpolate_engine(cycles, matrix, active_cols, cfg)
        df[active_cols] = filled_mat
    
    return df