# Propagation

`propagation.py` runs edited data through a trained model so that a change to one sensor carries
through to related sensors. For example, lowering sensor 11 (pressure) should change sensor 4
(temperature) if the model has learned how they relate. The model is not retrained.

## Model behavior

**Transformer.** The first `observed_rows` rows are kept as given. Later rows are generated one step
at a time, so an edit affects the generated rows that follow it.

**Diffusion** has two modes:

| Mode | When | Behavior |
|---|---|---|
| Conditional | `observed_rows` not set | Each row is generated independently from its operating conditions |
| Temporal (warm start) | `observed_rows` set | Each row starts from the previous row plus noise, then is denoised, so consecutive rows stay linked |

`warm_start_strength` (default 0.6) sets how much noise is added in temporal mode. Higher values
follow changes in conditions more closely; lower values give smoother transitions.

## API

```python
apply_propagation(
    modified_data: pd.DataFrame,
    model_type: Literal["diffusion", "transformer"],
    checkpoint_path: str,
    num_samples: Optional[int] = None,
    device: str = "cuda",
    config_path: Optional[str] = None,
    col_mins: Optional[pd.Series] = None,
    ranges: Optional[pd.Series] = None,
    observed_rows: Optional[int] = None,
    engine_id: Optional[int] = None,
    immune_cols: Optional[list[str]] = None,
    reference_data: Optional[pd.DataFrame] = None,
) -> pd.DataFrame
```

| Parameter | Meaning |
|---|---|
| `modified_data` | The edited data |
| `model_type` | `"transformer"` or `"diffusion"` |
| `checkpoint_path` | Trained model file |
| `num_samples` | Rows to generate; default is `len(modified_data)` |
| `device` | `"cuda"` or `"cpu"`; CPU is enough for inference |
| `config_path` | Transformer only: YAML file with normalization statistics, if the checkpoint lacks them |
| `col_mins`, `ranges` | Transformer only: normalization statistics that override the config |
| `observed_rows` | Rows treated as known history (see above) |
| `engine_id` | Transformer only: engine used for the engine embedding |
| `immune_cols` | Columns to keep unchanged; their values are copied from `reference_data` |
| `reference_data` | Unedited data; required with `immune_cols` |

Returns the regenerated DataFrame.

## Tests

In `src/features/propagation_testing/`:

| File | Purpose |
|---|---|
| `test_propagation_transformer.py`, `test_propagation_diffusion.py` | Basic checks of `apply_propagation` |
| `physics.py`, `diffusion_physics.py` | Edit experiment described below, with plots |
| `statistical_analysis.py` | Measures how much and how consistently edits change the output (distribution distances, confidence intervals) |

These need trained checkpoints. Plots are written to `propagation_testing/outputs/`.

## Edit experiment (NASA FD001, Transformer)

**Setup:** shift sensor 11 by -5.0 during cycles 15 to 30, then generate 80 cycles. Watch sensors 4
and 7.

**Steps:**
1. Generate a baseline from the unedited engine record.
2. Generate again from the edited record.
3. Generate again with `immune_cols=["4"]`.
4. Compare the three runs after the edit window.

**Expected result:**
- Sensor 11 shows the step change during the edit window.
- Sensors 4 and 7 diverge from the baseline once generation starts, and the difference persists.
- With sensor 4 immune, it stays at its reference values while sensor 7 still responds.

This shows that the change spreads through relationships the model learned, not by copying.

![Propagation experiment results](https://github.com/user-attachments/assets/2d1ce146-ff47-4693-aa7b-3ee4c693c26d)

The same experiment applies to the diffusion model in temporal mode.
