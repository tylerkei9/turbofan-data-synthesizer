# Interpolation Module

## Overview

This module provides a unified interface for applying interpolation
methods to time-series or tabular data.

It integrates: - `linear_interpolation.py` - `Bspline_MultFunc.py`

through a **gateway pattern**, where `interpolation.py` acts as the
central interface.

------------------------------------------------------------------------

## Architecture

    interpolation.py (Gateway)
        ↓
        ├─→ linear_interpolation.py (method="linear")
        ├─→ Bspline_MultFunc.py (method="spline")
        └─→ ML module (method="ml") [TODO]

-   Central routing handled in `interpolation.py`
-   Implementation logic remains modular and isolated
-   Easily extensible for future methods

------------------------------------------------------------------------

## API

``` python
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
```

------------------------------------------------------------------------

## Parameters

-   `data`: Input DataFrame
-   `method`: `"linear"`, `"spline"`, or `"ml"`
-   `target_columns`: Columns to interpolate (default = numeric columns)
-   `x_column`: Independent variable (auto-detected if not provided)
-   `engine_id_column`: Used for grouped interpolation (linear only)
-   `max_gap_cycles`: Max gap size for linear interpolation
-   `gap_threshold`: Gap threshold for spline interpolation
-   `num_points`: Number of interpolated points per gap (spline)

------------------------------------------------------------------------

## Behavior by Method

### Linear Interpolation (`method="linear"`)

-   Fills NaN gaps using linear interpolation
-   Excludes:
    -   ID columns (`Engine_ID`, `Time_in_cycles`)
    -   Constant columns (auto-detected)
-   Supports three modes:
    1.  Per-engine (Engine_ID + Time column)
    2.  Time-series only
    3.  Index-based fallback
-   Leaves gaps larger than `max_gap_cycles` unfilled

------------------------------------------------------------------------

### Spline Interpolation (`method="spline"`)

-   Detects clusters using `detect_clusters()`
-   Gaps defined where spacing between x-values exceeds `gap_threshold`
-   Inserts `num_points` new rows per gap
-   Adds column:
    -   `is_interpolated` (True for generated rows)

#### x_column Resolution Order

1.  User-provided value
2.  `"Time_in_cycles"` if present
3.  First numeric column
4.  Raises error if none found

#### Important Behavior

-   Only interpolates selected columns
-   Non-target columns are set to NaN in new rows
-   Internally returns `(original_df, interpolated_df)`\
    → Gateway returns only `interpolated_df`

------------------------------------------------------------------------

### ML Interpolation (`method="ml"`)

-   Not implemented
-   Returns a copy of the original DataFrame

------------------------------------------------------------------------

## Usage Examples

### Linear

``` python
result = apply_interpolation(data, method="linear")
```

``` python
result = apply_interpolation(
    data,
    method="linear",
    x_column="Time_in_cycles",
    engine_id_column="Engine_ID",
    max_gap_cycles=20
)
```

------------------------------------------------------------------------

### Spline

``` python
result = apply_interpolation(
    data,
    method="spline",
    x_column="Time_in_cycles"
)
```

``` python
result = apply_interpolation(
    data,
    method="spline",
    target_columns=["sensor_4", "sensor_9"],
    gap_threshold=2.0,
    num_points=15
)
```

------------------------------------------------------------------------

## ⚠️ Important Limitation

### Gap Detection Not Supported in Frontend

-   Gap detection is implemented in backend via `detect_clusters()`
-   Frontend does NOT support gap-aware logic

------------------------------------------------------------------------

## Benefits

### Separation of Concerns

-   Gateway handles routing only
-   Logic isolated in implementation modules

### Extensibility

-   Easy to add new methods (e.g., ML)

### Robust Defaults

-   Auto-detection of:
    -   Engine ID column
    -   Time column
    -   Numeric target columns

### Reusability

-   Works with arbitrary DataFrame schemas
-   No strict dependency on column names

### Clean Design

-   No duplicated logic
-   Consistent API across methods
-   Implementation details abstracted away

------------------------------------------------------------------------

## Method Comparison

| Feature        |  Linear                          |    Spline|
|----------------| -----------------------          |----------------------------|
| Purpose        |  Fill NaNs                       |   Connect clusters         |
| Row Count      |     Same                         |  Increases                 |
| Modification   |   In-place                       |Adds new rows               |
| Gap Handling   |   Limited     (`max_gap_cycles`) | Full cluster detection     |
| Marker Column  |   None                           | `is_interpolated`          |
  -----------------------------------------------------------------------


## Testing

``` bash
python test_interpolation_integration.py
```
This verifies:
- ✓ Linear method fills NaN gaps correctly
- ✓ Linear method works without time column (uses row index)
- ✓ Spline method delegates correctly to Bspline_MultFunc
- ✓ Spline method adds interpolated points between clusters
- ✓ Target columns parameter works for both methods
- ✓ Column auto-detection works (no hardcoded names required)
- ✓ ML method returns gracefully with TODO message
------------------------------------------------------------------------

## File Structure

    src/features/
    ├── interpolation.py
    ├── linear_interpolation.py
    ├── Bspline_MultFunc.py
    └── ml_interpolation.py   # future

------------------------------------------------------------------------

## Notes

-   Linear interpolation uses `interpolate_engine()` internally
-   Constant columns are automatically excluded
-   Spline interpolation operates on cluster boundaries only
-   Gateway ensures a consistent API regardless of method
-   No breaking changes to existing usage
-   Designed for backend-driven workflows
-   Frontend support is partial (no gap detection)

## Next Steps

### For Future Development:
1. **Add Method Selection UI**: Add dropdown/radio buttons in frontend to let users choose method
2. **Implement ML Interpolation**: Create ML-based module and update gateway
3. **Parameter Tuning UI**: Add controls for method-specific parameters
