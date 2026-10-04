# Interpolation

`interpolation.py` provides one function, `apply_interpolation`, and passes the work to a method
module:

| `method` | Module | Status |
|---|---|---|
| `"linear"` | `linear_interpolation.py` | Implemented |
| `"spline"` | `Bspline_MultFunc.py` | Implemented |
| `"ml"` | | Not implemented; returns a copy of the input |

## API

```python
apply_interpolation(
    data: pd.DataFrame,
    method: Literal["linear", "spline", "ml"] = "linear",
    target_columns: Optional[list] = None,   # default: all numeric columns
    x_column: Optional[str] = None,          # default: auto-detected
    gap_threshold: float = 1.0,              # spline: spacing that counts as a gap
    num_points: int = 10,                    # spline: rows inserted per gap
    max_gap_cycles: int = 10,                # linear: longest gap to fill
    engine_id_column: Optional[str] = None,  # linear: fill within each engine
) -> pd.DataFrame
```

## Methods

| | Linear | Spline |
|---|---|---|
| Purpose | Fill missing values | Add rows between clusters of points |
| Row count | Unchanged | Increases |
| Gap handling | Up to `max_gap_cycles` | Gap detection with `detect_clusters()` |
| Marker column | None | `is_interpolated` |

**Linear**
- Fills missing values by straight-line interpolation.
- Skips ID columns (`Engine_ID`, `Time_in_cycles`) and constant columns.
- Works per engine when engine and time columns exist, otherwise along time or row order.
- Leaves gaps longer than `max_gap_cycles` unfilled.

**Spline**
- Finds gaps where the spacing between x-values exceeds `gap_threshold`.
- Inserts `num_points` rows per gap. Columns not in `target_columns` are left empty in new rows.
- `x_column` is chosen in this order: the given value, `Time_in_cycles`, the first numeric column.
  An error is raised if none exists.

## Examples

```python
result = apply_interpolation(data, method="linear",
                             x_column="Time_in_cycles", engine_id_column="Engine_ID",
                             max_gap_cycles=20)

result = apply_interpolation(data, method="spline",
                             target_columns=["4", "11"], gap_threshold=2.0, num_points=15)
```

## Use from the dashboard

The dashboard lets the user choose linear or spline and calls `apply_interpolation()` through
`api/pipeline.py`. The bundled NASA data has no gaps, so the demo can remove a stretch of one
engine's readings first and then fill it.

## Tests

```bash
python src/features/interpolation_testing/test_interpolation_integration.py
```

Covers linear filling with and without a time column, spline gap filling, `target_columns`, column
auto-detection, and the `ml` placeholder.
