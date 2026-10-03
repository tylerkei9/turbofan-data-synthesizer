# Features Documentation

## Overview

This directory contains three independent features for synthetic data manipulation:
1. **Interpolation** - Connects disconnected data clusters by generating intermediate points
2. **Shifting** - Adjusts sensor values by user-specified amounts with real-time visualization
3. **Propagation** - Regenerates synthetic data using trained models to reflect cascading relationships

Each feature is self-contained with a single public function that the frontend calls.
These features are designed to work **after** synthetic data has been generated, providing post-processing and exploration capabilities through an interactive Streamlit interface.

---
## 📁 Module Structure

```
src/features/
├── __init__.py           # Package exports (apply_interpolation, ShiftingManager, apply_propagation)
├── interpolation.py      # Cluster detection and gap-filling interpolation
├── shifting.py           # Sensor value adjustment with cumulative tracking
├── propagation.py        # Model-based data regeneration
└── README.md            # This file
```

---
## General Principles

### Division of Responsibilities

**Feature Teams ARE Responsible For:**
- Implementing all logic within their feature function
- Processing input data and returning modified data
- Error handling within the feature
- Internal helper functions
- Documentation of their feature

**Feature Teams are NOT Responsible For:**
- UI implementation (handled by frontend)
- Calling other features (handled by frontend orchestration)
- File I/O operations (handled by frontend)
- Data visualization (handled by frontend)
- Session state management (handled by frontend)

### Data Flow

```
Frontend → Feature Function → Backend Logic → Return DataFrame → Frontend
```

**Important:** Feature functions should:
- Accept pandas DataFrames as input
- Return pandas DataFrames as output
- NOT modify the original DataFrame (always work on copies)
- NOT save/load files directly
- NOT depend on other features

---

## 🔗 Feature 1: Interpolation

### Purpose
Identifies clusters/gaps in synthetic data and adds intermediate data points to create smoother, more continuous datasets.

### Team Responsibilities

**Interpolation Team IS Responsible For:**
- Implementing cluster detection logic (DBSCAN or distance-based)
- Calculating interpolation paths between clusters
- Generating new data points along paths
- Supporting multiple interpolation methods (linear, spline, ML)
- Testing with various synthetic datasets

**Interpolation Team is NOT Responsible For:**
- Creating UI controls for method selection
- Deciding which columns to interpolate (frontend provides this)
- Training generative models
- Visualizing results

### Function Signature

```python
def apply_interpolation(
    data: pd.DataFrame,
    method: Literal["linear", "spline", "ml"] = "linear",
    target_columns: Optional[list] = None
) -> pd.DataFrame:
```

### Parameters
- `data`: Synthetic data to interpolate
- `method`: Interpolation method ("linear", "spline", "ml")
- `target_columns`: Specific columns to interpolate (None = all numeric columns)

### Returns
- New DataFrame with interpolated points added between clusters
- `pd.DataFrame`: Original data + interpolated points

- Original data preserved, new rows appended (clarify if this is all data or only newly interpolated points)


### Implementation Notes

**Basic Version (Week 8-9):**
- Use simple distance threshold to detect gaps between points
- Implement linear interpolation between gap boundaries
- Add spline interpolation for smoother curves
- Add fixed number of points (e.g., 10) between gaps

**Advanced Version (Week 11-14):**
- Implement DBSCAN clustering for robust gap detection
- Implement ML-based interpolation (e.g., using KNN or neural networks)
- Adaptive number of interpolation points based on gap size

### Example Usage

```python
# Frontend calls:
synth_data = pd.read_csv("synthetic_data.csv")

# Basic linear interpolation
result = apply_interpolation(synth_data, method="linear")

# Spline interpolation on specific columns
result = apply_interpolation(
    synth_data, 
    method="spline",
    target_columns=["sensor_4", "sensor_9"]
)
```

### Testing

```python
# Test case: Should add points between gaps
def test_interpolation_adds_points():
    data = pd.DataFrame({
        "x": [0, 10],
        "y": [0, 10]
    })
    result = apply_interpolation(data, method="linear")
    assert len(result) > len(data)

# Test case: Original data preserved
def test_original_data_preserved():
    data = pd.DataFrame({"x": [1, 2, 3]})
    result = apply_interpolation(data)
    assert data["x"].iloc[0] == 1  # Original unchanged
```

---

## 📊Feature 2: Shifting

### Purpose
Adjusts sensor values by user-specified amounts with batch visualization updates.

### Team Responsibilities

**Shifting Team IS Responsible For:**
- Implementing arithmetic shift operations on DataFrame columns
- Handling multiple column shifts simultaneously
- Validating that columns exist
- Ensuring only numeric columns are shifted
- Providing utility functions (get_shiftable_columns)

**Shifting Team is NOT Responsible For:**
- Creating UI sliders
- Storing shift history (frontend handles this)
- Deciding shift amounts (user input via frontend)
- Visualizing shifted data
- Validating "realistic" values (intentionally no validation)
- Tracking shift history (frontend handles this)


### Function Signature

```python
def apply_shift(
    data: pd.DataFrame,
    shift_values: Dict[str, float],
    store_original: bool = True
) -> pd.DataFrame:
```

### Parameters
- `data` (pd.DataFrame): Data to shift (real or synthetic)
- `shift_values` (dict): Dictionary mapping column names to shift amounts
  - Example: `{"sensor_4": 50.0, "RPM": -20.0}`
- `store_original` (bool, default=True): Whether to preserve original data (frontend handles storage)

### Returns
- `pd.DataFrame`: New DataFrame with shifts applied
- Formula: `new_value = original_value + shift_amount`
- All data returned (not just shifted columns)
- Original DataFrame is never modified

### Implementation Notes


**Current Implementation (Complete):**
- Works on a copy of input DataFrame to preserve original
- Identifies numeric columns using `get_shiftable_columns()`
- Skips non-existent columns gracefully (logs warning, continues)
- Skips non-numeric columns automatically
- Applies arithmetic shift: `shifted_data[column] = data[column] + float(shift_value)`
- Returns complete DataFrame with shifts applied

**Helper Functions:**
```python
def get_shiftable_columns(data: pd.DataFrame) -> list:
    """Returns list of numeric column names that can be shifted."""

def validate_shift_values(data: pd.DataFrame, shift_values: Dict[str, float]) -> tuple:
    """Returns (is_valid: bool, error_messages: list)."""

def get_recommended_shift_bounds(data: pd.DataFrame, column: str, multiplier: float = 2.0) -> tuple:
    """Returns (min_bound, max_bound, step_size) for UI sliders."""

def calculate_shift_statistics(original_data: pd.DataFrame, shifted_data: pd.DataFrame, column: str) -> Dict[str, float]:
    """Returns statistics showing impact of shift (mean, min, max differences)."""
```

### Example Usage
```python
# Frontend calls:
from src.features.shifting import apply_shift, get_shiftable_columns

# Load data
data = pd.read_csv("synthetic_data.csv")

# Get shiftable columns for UI
shiftable_columns = get_shiftable_columns(data)
# Returns: ["Engine_ID", "sensor_4", "sensor_9", "RPM", ...]

# Single column shift
result = apply_shift(data, {"sensor_4": 50.0})

# Multiple column shifts
result = apply_shift(data, {
    "sensor_4": 50.0,
    "sensor_9": -20.0,
    "RPM": 200.0
})

# Validate before applying
is_valid, errors = validate_shift_values(data, {"sensor_4": 50.0})
if is_valid:
    result = apply_shift(data, {"sensor_4": 50.0})

# Get slider bounds
min_val, max_val, step = get_recommended_shift_bounds(data, "sensor_4")

# Calculate impact statistics
stats = calculate_shift_statistics(original_data, shifted_data, "sensor_4")
print(f"Mean difference: {stats['mean_difference']}")
```

### Testing

**Implemented Tests (10 test cases):**

The shifting module includes 10 automated tests verifying all functionality. All tests are located in `src/features/shifting/test_shifting.py`.

**Test Coverage:**

1. **Shift applied correctly** - Verifies arithmetic shift operation works as expected
2. **Original data unchanged** - Confirms original DataFrame is never modified
3. **Non-existent column handled gracefully** - Ensures missing columns don't cause errors
4. **Multiple shifts work simultaneously** - Validates shifting multiple columns at once
5. **Empty DataFrame handled** - Tests behavior with empty input data
6. **No shift values provided** - Verifies graceful handling when no shifts requested
7. **Get shiftable columns** - Tests column detection excluding non-numeric types
8. **Validation works** - Confirms validation function correctly identifies valid/invalid inputs
9. **Slider bounds calculation** - Tests automatic slider range generation
10. **Statistics calculation** - Verifies impact statistics are computed correctly

**Running Tests:**
```bash
# Run all shifting tests
python src/features/shifting.py

# Expected output: All 10 tests pass
```

### Status
**Complete and Ready**
- All core functions implemented
- 10 tests passing
- Error handling and validation complete
- Documentation complete

### Module Location
`src/features/shifting.py`

---

## 🔄 Feature 3: Propagation

### Purpose
Regenerates synthetic data by feeding modified (shifted/interpolated) data back through existing trained generative models (Diffusion or Transformer). This creates relationship-based cascading effects where changes naturally propagate through learned correlations.

### Team Responsibilities

**Propagation Team IS Responsible For:**
- Loading trained models from checkpoints (Diffusion and Transformer)
- Preparing modified data for model input
- Running model inference to generate new samples
- Converting model output back to DataFrame format
- Supporting both diffusion and transformer architectures

**Propagation Team is NOT Responsible For:**
- Training models (use existing trained models)
- Deciding when to apply propagation (frontend toggle)
- Applying shifts/interpolation (those are separate features)
- Managing model checkpoints (frontend provides paths)
- UI for model selection

### Function Signature

```python
def apply_propagation(
    modified_data: pd.DataFrame,
    model_type: Literal["diffusion", "transformer"],
    checkpoint_path: str,
    num_samples: Optional[int] = None,
    device: str = "cuda",
    config_path: str
) -> pd.DataFrame:
```

### Parameters
- `modified_data`: Synthetic data that has been shifted/interpolated
- `model_type`: "diffusion" or "transformer"
- `checkpoint_path`: Path to trained model checkpoint
- `num_samples`: Number of samples to generate (None = match input size)
- `device`: "cuda" or "cpu"
- `config_path` : str, optional
    Path to model configuration file. If None, uses default configs:
    - Diffusion: `src/config/diffusion_default.yaml`
    - Transformer: `src/config/transformer_default.yaml`
### Returns
- `pd.DataFrame`: Newly generated synthetic data based on modified input
- Newly generated synthetic data based on modified input
- Replaces old synthetic data in the application

### Default Model Paths
- Diffusion: `src/model_checkPoints/diffusion/checkpoint_best.pt`
- Transformer: `src/model_checkPoints/transformer/checkpoint_best.pt`

### Implementation Notes
- **Reuses existing trained models** from `src/model_checkPoints/`
- Does NOT retrain - only runs inference
- Cascading relationships (A→B, B→C, therefore A→C) happen through model
- All model loading and inference logic contained within feature

**Basic Version (Week 8-9):**
- Load model checkpoint
- Run simple inference with modified data as conditioning
- Return generated samples as DataFrame
- Handle both model types with basic configuration

**Advanced Version (Week 11-14):**
- Optimize inference speed
- Add batch processing for large datasets
- Implement progress tracking
- Fine-tune conditioning strategy
- Add model caching to avoid reloading



### Workflow

```
1. User shifts sensor_4 by +50
2. Frontend calls apply_shift() → modified_data
3. If propagation enabled:
   Frontend calls apply_propagation(modified_data, ...)
4. Propagation team:
   - Load trained model
   - Use modified_data as conditioning input
   - Generate new synthetic data via model inference
   - Return new DataFrame
5. Frontend replaces old synth_data with new data
```

### Example Usage

```python
# Frontend calls:
# Step 1: User shifts data
shifted_data = apply_shift(synth_data, {"sensor_4": 50.0})

# Step 2: User enables propagation
if propagate_enabled:
    new_synth_data = apply_propagation(
        modified_data=shifted_data,
        model_type="diffusion",
        checkpoint_path="outputs/diffusion/models/model_conditional.pth",
        device="cuda"
    )
    # new_synth_data now reflects shifted relationships

# Example with transformer
interpolated_data = apply_interpolation(synth_data, method="spline")
if propagate_enabled:
    new_synth_data = apply_propagation(
        modified_data=interpolated_data,
        model_type="transformer",
        checkpoint_path="src/model_checkPoints/transformer/checkpoint_best.pt"
    )
```

### Testing

```python
# Test case: Model loads successfully
def test_model_loading():
    model = load_model("path/to/checkpoint.pt", "diffusion")
    assert model is not None

# Test case: Generated data has correct shape
def test_output_shape():
    data = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    result = apply_propagation(data, "diffusion", "checkpoint.pt")
    assert result.shape[1] == data.shape[1]  # Same columns

# Test case: Both model types work
def test_both_models():
    data = pd.DataFrame({"x": [1, 2, 3]})
    diffusion_result = apply_propagation(data, "diffusion", "diff_ckpt.pt")
    transformer_result = apply_propagation(data, "transformer", "trans_ckpt.pt")
    assert diffusion_result is not None
    assert transformer_result is not None
```

---

## Integration with Frontend

### How Features Are Called

The frontend orchestrates all feature calls in the `apply_all_changes()` function:

```python
def apply_all_changes():
    modified_data = st.session_state.synth_data.copy()
    
    # Step 1: Apply shifts
    if st.session_state.shift_values:
        modified_data = apply_shift(modified_data, st.session_state.shift_values)
    
    # Step 2: Apply interpolation
    interpolate_cols = [col for col, enabled in st.session_state.interpolate_enabled.items() if enabled]
    if interpolate_cols:
        modified_data = apply_interpolation(modified_data, target_columns=interpolate_cols)
    
    # Step 3: Apply propagation
    if any(st.session_state.propagate_enabled.values()):
        modified_data = apply_propagation(
            modified_data,
            model_type=selected_model,
            checkpoint_path=checkpoint_path
        )
    
    # Update session state
    st.session_state.synth_data = modified_data
```

### Key Points

1. **Frontend decides order** - Features are called in sequence
2. **Features don't call each other** - Each feature is independent
3. **Frontend passes parameters** - UI state converted to function parameters
4. **Features return results** - Frontend handles result storage and visualization

---

## Development Workflow

### Week 6-7: Planning & Skeleton
- [ ] Define function signatures
- [ ] Write README documentation
- [ ] Create skeleton code with placeholders
- [ ] Agree on data formats and interfaces
- [ ] Study feature requirements and create documentation
- [ ] Build skeleton code with clear function signatures
- [ ] Establish integration points with frontend


### Week 8-9: Basic Implementation
- [ ] Implement basic version of feature
- [ ] Test with sample data
- [ ] Integrate with frontend
- [ ] Verify end-to-end flow works

### Week 11-14: Advanced Implementation
- [ ] Add advanced algorithms
    - **Interpolation**: ML-based methods (Gaussian Processes, neural networks)
- [ ] Optimize performance
- [ ] Handle edge cases
- [ ] Comprehensive testing
- [ ] Final integration testing

### Weeks 15-16: Testing & Handoff
- End-to-end testing with real synthetic data
- Documentation finalization
- Symposium preparation and project handoff

---

## Common Utilities

### Shared Helper Functions (Optional)

If multiple features need common functionality, create:

```python
# src/features/utils.py

def validate_dataframe(df: pd.DataFrame) -> bool:
    """Check if DataFrame is valid for processing."""
    pass

def get_numeric_columns(df: pd.DataFrame) -> list:
    """Get list of numeric columns."""
    return df.select_dtypes(include=[np.number]).columns.tolist()
```

---

## Notes

- Each feature directory can have additional helper files as needed
- Keep the main function (`apply_*`) as the only public API
- Document any assumptions about data format
- Handle errors gracefully and return meaningful error messages
- Always work on copies of DataFrames, never modify in-place


## 📚 Additional Resources

- [NASA C-MAPSS Dataset Documentation](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/)
- [Project Main README](../../README_spring.md)
- [Diffusion Model Documentation](../training/Diffusion/)
- [Transformer Model Documentation](../training/Transformer/)
- [Frontend Documentation](../../frontend/README.md)

---

## 📝 Notes

- Features work on **synthetic data only**, not real turbofan data
- All features are **optional** (can be toggled ON/OFF)
- Features can be used **independently or together**
- Original data is **never modified** (functions return new DataFrames)
- Features are **post-processing** operations, not part of model training

---

**Developed by The Data Mine @ Purdue University**  
*Spring 2026 - Rolls-Royce Data Synthesizer Project*