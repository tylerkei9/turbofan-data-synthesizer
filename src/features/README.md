# Features

Tools that edit synthetic data after it has been generated. Each tool takes a DataFrame and returns
a new one; the input is never modified, and no tool reads or writes files.

| Feature | Module | What it does |
|---|---|---|
| Shifting | `shifting.py` | Adds a fixed amount to one or more columns |
| Interpolation | `interpolation.py` | Fills gaps in a timeline (linear or spline) |
| Propagation | `propagation.py` | Runs edited data back through a trained model so related sensors respond to the edit |

Details: [Interpolation_README.md](Interpolation_README.md), [Propagation_README.md](Propagation_README.md).

## Files

```
src/features/
├── shifting.py                 # apply_shift and helpers
├── interpolation.py            # apply_interpolation: chooses the method
├── linear_interpolation.py     # linear method
├── Bspline_MultFunc.py         # spline method with gap detection
├── propagation.py              # apply_propagation
├── interpolation_testing/      # interpolation tests
└── propagation_testing/        # propagation tests and experiments
```

## Shifting

```python
apply_shift(data: pd.DataFrame, shift_values: dict[str, float], store_original: bool = True) -> pd.DataFrame
```

Returns a copy with `new_value = old_value + shift` for each listed column. Missing or non-numeric
columns are skipped. Values are not checked for realism.

Helpers:

| Function | Returns |
|---|---|
| `get_shiftable_columns(data)` | Numeric column names |
| `validate_shift_values(data, shift_values)` | `(is_valid, error_messages)` |
| `get_recommended_shift_bounds(data, column, multiplier=2.0)` | `(min, max, step)` for a slider |
| `calculate_shift_statistics(original, shifted, column)` | Mean, min and max differences |

```python
from src.features.shifting import apply_shift
result = apply_shift(data, {"11": 5.0, "4": -2.0})
```

Run `python src/features/shifting.py` for a short self-test.

## Interpolation

```python
apply_interpolation(data, method="linear", target_columns=None, x_column=None,
                    gap_threshold=1.0, num_points=10, max_gap_cycles=10,
                    engine_id_column=None) -> pd.DataFrame
```

- `linear` fills missing values within each engine, up to `max_gap_cycles` long.
- `spline` detects gaps between clusters of points and inserts `num_points` new rows per gap,
  marked with `is_interpolated = True`.
- `ml` is not implemented and returns a copy of the input.

## Propagation

```python
apply_propagation(modified_data, model_type, checkpoint_path, num_samples=None, device="cuda",
                  config_path=None, col_mins=None, ranges=None, observed_rows=None,
                  engine_id=None, immune_cols=None, reference_data=None) -> pd.DataFrame
```

Loads a trained Transformer or diffusion checkpoint and regenerates the data so that edits to one
sensor carry through to related sensors. The model is not retrained. Default checkpoints:

- Diffusion: `src/model_checkPoints/diffusion/diff_best_ckpt.pt`
- Transformer: `src/model_checkPoints/transformer/transformer_model.pt`

## Use from the dashboard

The dashboard sends edit requests to the local server, and `api/pipeline.py` calls the features:

| Dashboard action | `api/pipeline.py` | Feature |
|---|---|---|
| Shift a sensor | `apply_shift_to_session(column, amount)` | `apply_shift()` |
| Fill a gap | `apply_interpolation_to_session(columns, method, gap)` | `apply_interpolation()` |
| Propagate an edit | `apply_propagation_to_session(prop_cols, cutoff)` | `apply_propagation()` |

Each call passes the current synthetic data to the feature and stores the result, so edits stack in
the order they are made. Features do not call each other.

## Tests

```bash
python src/features/shifting.py
python src/features/interpolation_testing/test_interpolation_integration.py
python src/features/propagation_testing/test_propagation_transformer.py   # needs a trained checkpoint
python src/features/propagation_testing/test_propagation_diffusion.py     # needs a trained checkpoint
```
