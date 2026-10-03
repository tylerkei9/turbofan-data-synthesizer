# Propagation

The propagation module applies a trained generative model to a modified time-series trajectory to observe how edits to one or more variables affect the model's generated output.

Supported model types:
- **Diffusion**
- **Transformer**
---

# Purpose

Propagation verifies that the generative model responds appropriately to changes in the input trajectory. For example, modifying **sensor 11 (pressure)** should result in a change in **sensor 4 (temperature)** if the model has learned the relationship between these variables.

---

## Model Behavior

### Transformer (Temporal)
- First `observed_rows` = known input (copied directly)
- Future rows are generated step-by-step
- Edits affect only the **generated region**
- Propagation occurs through time

---

### Diffusion (Conditional & Temporal)

The diffusion model supports two distinct modes of propagation:

#### **Mode A: Conditional (Standard)**
- **Behavior**: Generates outputs independently for every cycle based only on the operating conditions.
- **Independence**: Does not use past sequence history.
- **Use Case**: Best for creating high-fidelity "snapshots" of sensors for a given set of controls.

#### **Mode B: Temporal (Advanced)**
- **Behavior**: Uses **Warm-Started Diffusion** to create coupling between consecutive steps.
- **Workflow**: When `observed_rows` is provided, the model uses the previous cycle's sensors as a "seed," adds noise up to a specific threshold, and then denoises to find the new state.
- **Use Case**: Best for generating smooth, continuous trajectories where the next step must stay statistically tied to the previous one.

# Public API

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
    reference_data: Optional[pd.DataFrame] = None
) -> pd.DataFrame
```

# Input Parameters

**modified_data**
Input dataframe containing the edited trajectory or samples to propagate through the model.

**model_type**
Model architecture to use: `"transformer"` or `"diffusion"`.

**checkpoint_path**
Path to the pretrained model checkpoint.

**num_samples**
Number of output rows to generate. Defaults to `len(modified_data)` if not specified.

**device**
Compute device (`"cuda"` or `"cpu"`). CPU is typically sufficient since the script performs inference only.

**config_path**
Optional YAML configuration path. For the transformer model, this may provide normalization statistics if they are not stored in the checkpoint.

**col_mins, ranges**
Optional normalization statistics for the transformer model. If provided, these override statistics loaded from the config.

**observed_rows**
Number of rows treated as "known" history. 
- **Transformer**: Future generation begins after this row.
- **Diffusion**: Enables the **Warm-Start** temporal coupling starting from this row.

**engine_id**
Transformer only. Engine identifier used for engine embeddings.

**immune_cols**
Optional list of column names to protect from regeneration.
- **Requirement**: Requires `reference_data` to provide the unedited ground-truth values.
- **Effect**: The model will generate all sensors EXCEPT those listed in `immune_cols`, which are copied from the reference.

**reference_data**
Optional DataFrame of unedited data. Required when using `immune_cols`.

# Returns

Returns a `pd.DataFrame` containing the propagated or generated trajectory.

---

# Propagation Testing
# Testing and Verifying Propagation

Propagation validation scripts are located in:

```
src/features/propagation_testing
```
`test_propagation_diffusion.py` and `test_propagation_transformer.py` provide *basic* tests to verify the functionality of the propagation module.

# NASA CMAPSS DATASET - Propagation Validation Example (Transformer)

To validate that propagation behaves correctly, a controlled intervention experiment is performed using `physics.py`.
=======
To validate that propagation behaves correctly, a controlled intervention experiment is performed using `physics.py`. Visualizations and plots are found in `src/features/propagation_testing/outputs'

### Experiment Setup

* **Source feature:** Sensor 11 (pressure)
* **Target feature:** Sensor 4 (temperature)
* **Secondary target:** Sensor 7 
* **Edit window:** cycles 15–30
* **Edit magnitude:** −5.0 applied to source
* **Prediction horizon:** 80 generated cycles

### Procedure

1. Load a real engine trajectory from CMAPSS.
2. Generate a baseline trajectory using the original data.
3. Create an edited trajectory by applying a step shift to the source feature.
4. Run both trajectories through `apply_propagation(...)`.
5. Compare the generated outputs after the edit window.

6. Run a second propagation with immune columns enabled: (specify immune_cols=[target_col], in this case, immune_cols=[4]
7. Compare three cases:
    * baseline
    * edited (default propagation)
    * edited with immune column
Analyze:
    * whether targets change after the edit window
    * whether immune targets remain fixed
    * whether non-immune targets still respond

### Expected Output

If propagation functions correctly:

* The edited source variable remains clamped during the intervention window.
* Downstream variables respond after the edit window once autoregressive generation begins.
* Differences between baseline and edited trajectories appear primarily in the generated region.

### Expected Result
* Source variable shows a clear step shift during the edit window
* Target variables begin diverging after generation starts
* Differences between baseline and edited trajectories persist over time
* Immune targets remain stable when specified

This indicates:
* Propagation occurs through learned relationships, not direct copying
* Effects emerge only when the model begins autoregressive generation
* The model respects structural constraints (e.g., immune columns)

<img width="2398" height="2378" alt="image" src="https://github.com/user-attachments/assets/2d1ce146-ff47-4693-aa7b-3ee4c693c26d" />

### Notes

* This validation currently applies to both the **Transformer** and **Diffusion** models.
* For **Diffusion**, temporal coupling is enabled via the `warm_start_strength` parameter, allowing for relationship-based propagation similar to the Transformer.

---

## ⚙️ Tuning & Advanced Parameters (Diffusion)

When using Diffusion for propagation, the following advanced settings can be tuned via the configuration or the internal API:

### Warm-Start Strength
Parameter: `warm_start_strength` (Default: `0.6`)
Controls the balance between consistency and responsiveness during temporal generation ($t >$ `observed_rows`):

- **High Strength (e.g., 0.8+)**: Adds more noise to the seed data. The model becomes more stochastic and follows the new operating conditions more closely, but may lose some temporal continuity.
- **Low Strength (e.g., 0.3)**: Adds very little noise. This creates high "inertia," where the generated data stays extremely close to the seed values, ensuring very smooth transitions but potentially slow reaction to sharp condition changes.

### Immune Sensors
Parameter: `immune_cols`
Specific sensors can be "protected" (immune) from the generative process. This is useful for:
1. **Preserving specific manual edits** across the entire trajectory.
2. **Locking known-good sensors** while letting the model fill in ambiguous or unknown sensors.
