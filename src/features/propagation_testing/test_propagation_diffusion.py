#!/usr/bin/env python3
"""
Small, incremental tests for the propagation feature.
Each test is independent and verifies ONE specific thing.

Usage:
    python src/features/test_propagation.py              # Run all tests
    python src/features/test_propagation.py --test 1     # Run only test 1
    python src/features/test_propagation.py --test 1,2,3 # Run tests 1-3
"""

import sys
import traceback
from pathlib import Path

# Setup path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Checkpoint path
DIFFUSION_CKPT = PROJECT_ROOT / "src" / "model_checkPoints" / "diffusion" / "diff_best_ckpt.pt"

passed = []
failed = []


def run_test(name, func):
    """Run a single test and print result."""
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")
    try:
        func()
        print(f"  ✅ PASS: {name}")
        passed.append(name)
    except Exception as e:
        print(f"  ❌ FAIL: {name}")
        print(f"  Error: {e}")
        traceback.print_exc()
        failed.append(name)


# ============================================================
# TEST 1: Imports work
# ============================================================
def test_1_imports():
    """Can we import the propagation module at all?"""
    from src.features.propagation import (
        load_model,
        prepare_data_for_inference,
        generate_with_model,
        apply_propagation,
    )
    assert callable(load_model), "load_model should be callable"
    assert callable(prepare_data_for_inference), "prepare_data_for_inference should be callable"
    assert callable(generate_with_model), "generate_with_model should be callable"
    assert callable(apply_propagation), "apply_propagation should be callable"
    print("  All 4 public functions imported successfully")


# ============================================================
# TEST 2: Path resolution and constants
# ============================================================
def test_2_paths_and_constants():
    """Are the paths and column constants set up correctly?"""
    from src.features.propagation import (
        PROJECT_ROOT as PR,
        SCRIPT_DIR as SD,
        ENGINE_ID_COL, TIME_COL, ALTITUDE_COL, MACH_COL, TRA_COL, SENSOR_START_COL,
        _resolve_path,
    )
    # Check constants match FD001 layout
    assert ENGINE_ID_COL == 0, f"ENGINE_ID_COL should be 0, got {ENGINE_ID_COL}"
    assert TIME_COL == 1
    assert ALTITUDE_COL == 2
    assert MACH_COL == 3
    assert TRA_COL == 4
    assert SENSOR_START_COL == 5
    print(f"  Column constants OK")

    # Check path resolution
    resolved = _resolve_path("src/features/propagation.py")
    assert resolved.exists(), f"Resolved path should exist: {resolved}"
    print(f"  _resolve_path works: {resolved}")

    # Check PROJECT_ROOT looks right
    assert (PR / "src").is_dir(), f"PROJECT_ROOT/src should be a directory"
    print(f"  PROJECT_ROOT OK: {PR}")


# ============================================================
# TEST 3: Checkpoint file exists
# ============================================================
def test_3_checkpoint_exists():
    """Does the diffusion checkpoint file exist on disk?"""
    assert DIFFUSION_CKPT.exists(), (
        f"Diffusion checkpoint not found at: {DIFFUSION_CKPT}\n"
        f"Make sure you've trained the model first."
    )
    size_mb = DIFFUSION_CKPT.stat().st_size / (1024 * 1024)
    print(f"  Checkpoint found: {DIFFUSION_CKPT}")
    print(f"  Size: {size_mb:.1f} MB")


# ============================================================
# TEST 4: Checkpoint loads and has expected keys
# ============================================================
def test_4_checkpoint_contents():
    """Can we load the .pt file and does it have the right keys?"""
    import torch
    ckpt = torch.load(DIFFUSION_CKPT, map_location="cpu", weights_only=False)

    expected_keys = ["model_state", "schedule", "condition_stats", "sensor_stats"]
    for key in expected_keys:
        assert key in ckpt, f"Missing key in checkpoint: '{key}'"
        print(f"  ✓ Found key: '{key}'")

    # Check schedule has T and betas
    sched = ckpt["schedule"]
    assert "T" in sched, "schedule should have 'T'"
    assert "betas" in sched, "schedule should have 'betas'"
    print(f"  Schedule T={sched['T']}, betas length={len(sched['betas'])}")

    # Check stats have mean and std
    for stat_name in ["condition_stats", "sensor_stats"]:
        stats = ckpt[stat_name]
        assert "mean" in stats, f"{stat_name} should have 'mean'"
        assert "std" in stats, f"{stat_name} should have 'std'"
        print(f"  {stat_name}: mean has {len(stats['mean'])} values, std has {len(stats['std'])} values")


# ============================================================
# TEST 5: load_model() returns correct bundle
# ============================================================
def test_5_load_model():
    """Does load_model() successfully reconstruct the model?"""
    from src.features.propagation import load_model

    bundle = load_model(str(DIFFUSION_CKPT), "diffusion", device="cpu")

    # Check bundle has all expected keys
    expected = ["model", "schedule", "condition_stats", "sensor_stats", "device"]
    for key in expected:
        assert key in bundle, f"Bundle missing key: '{key}'"
        print(f"  ✓ Bundle has '{key}'")

    # Check model is in eval mode
    model = bundle["model"]
    assert not model.training, "Model should be in eval mode"
    print(f"  Model is in eval mode ✓")

    # Check model type
    from src.training.Diffusion.diffusion_model5 import ConditionalDenoiser
    assert isinstance(model, ConditionalDenoiser), "Model should be ConditionalDenoiser"
    print(f"  Model type: ConditionalDenoiser ✓")


# ============================================================
# TEST 6: Create fake data and prepare it for diffusion
# ============================================================
def test_6_prepare_data():
    """Can we prepare a small fake DataFrame for diffusion inference?"""
    import pandas as pd
    import numpy as np
    from src.features.propagation import load_model, prepare_data_for_inference

    # Create a small DataFrame matching FD001 format
    n_rows = 5
    data = pd.DataFrame({
        "Engine_ID": [1] * n_rows,
        "Time_in_cycles": list(range(1, n_rows + 1)),
        "Altitude": np.random.uniform(-0.005, 0.005, n_rows),
        "Mach_number": np.random.uniform(-0.0005, 0.0005, n_rows),
        "TRA": [100.0] * n_rows,
    })
    # Add 21 sensor columns
    for i in range(1, 22):
        data[str(i)] = np.random.uniform(500, 600, n_rows)

    print(f"  Created fake data: {data.shape}")

    bundle = load_model(str(DIFFUSION_CKPT), "diffusion", device="cpu")
    prepared = prepare_data_for_inference(data, "diffusion", bundle)

    # Check output structure
    assert "conditions_norm" in prepared, "Should have 'conditions_norm'"
    assert "cond_stats" in prepared, "Should have 'cond_stats'"
    assert "sensor_stats" in prepared, "Should have 'sensor_stats'"

    cond = prepared["conditions_norm"]
    assert cond.shape == (n_rows, 3), f"conditions_norm shape should be ({n_rows}, 3), got {cond.shape}"
    print(f"  conditions_norm shape: {cond.shape} ✓")
    print(f"  conditions_norm sample: {cond[0]}")


# ============================================================
# TEST 7: Generate a very small number of samples
# ============================================================
def test_7_generate_small():
    """Can we generate just 3 samples via diffusion?"""
    import pandas as pd
    import numpy as np
    from src.features.propagation import load_model, prepare_data_for_inference, generate_with_model

    # Create minimal input data
    n_rows = 3
    data = pd.DataFrame({
        "Engine_ID": [1] * n_rows,
        "Time_in_cycles": list(range(1, n_rows + 1)),
        "Altitude": [0.0] * n_rows,
        "Mach_number": [0.0] * n_rows,
        "TRA": [100.0] * n_rows,
    })
    for i in range(1, 22):
        data[str(i)] = [500.0] * n_rows

    print(f"  Input data shape: {data.shape}")

    bundle = load_model(str(DIFFUSION_CKPT), "diffusion", device="cpu")
    prepared = prepare_data_for_inference(data, "diffusion", bundle)

    num_samples = 3
    result = generate_with_model(bundle, prepared, num_samples, "diffusion")

    assert isinstance(result, pd.DataFrame), f"Result should be DataFrame, got {type(result)}"
    assert len(result) == num_samples, f"Should have {num_samples} rows, got {len(result)}"
    print(f"  Generated DataFrame shape: {result.shape}")
    print(f"  Columns: {list(result.columns)}")
    print(f"  First row sample:")
    print(f"    {result.iloc[0].to_dict()}")

    # Basic sanity: values should be finite
    assert result.select_dtypes(include=[np.number]).notna().all().all(), "All numeric values should be finite"
    print(f"  All values are finite ✓")


# ============================================================
# TEST 8: Full apply_propagation() end-to-end
# ============================================================
def test_8_apply_propagation_e2e():
    """Does apply_propagation() work end-to-end with 3 samples?"""
    import pandas as pd
    import numpy as np
    from src.features.propagation import apply_propagation

    # Create small input matching FD001 format
    n_rows = 5
    data = pd.DataFrame({
        "Engine_ID": [1] * n_rows,
        "Time_in_cycles": list(range(1, n_rows + 1)),
        "Altitude": [0.001, -0.002, 0.0, 0.003, -0.001],
        "Mach_number": [0.0003, -0.0001, 0.0, 0.0002, 0.0001],
        "TRA": [100.0] * n_rows,
    })
    for i in range(1, 22):
        data[str(i)] = np.random.uniform(500, 600, n_rows)

    print(f"  Input: {data.shape}")

    # Run full pipeline
    result = apply_propagation(
        modified_data=data,
        model_type="diffusion",
        checkpoint_path=str(DIFFUSION_CKPT),
        num_samples=3,
        device="cpu",
    )

    assert isinstance(result, pd.DataFrame), f"Should return DataFrame, got {type(result)}"
    assert len(result) == 3, f"Should have 3 rows, got {len(result)}"
    print(f"  Output: {result.shape}")
    print(f"  Columns ({len(result.columns)}): {list(result.columns)}")
    print(f"  ✓ Full end-to-end pipeline works!")


# ============================================================
# Main
# ============================================================
ALL_TESTS = [
    ("1. Imports work", test_1_imports),
    ("2. Paths & constants", test_2_paths_and_constants),
    ("3. Checkpoint exists", test_3_checkpoint_exists),
    ("4. Checkpoint contents", test_4_checkpoint_contents),
    ("5. load_model()", test_5_load_model),
    ("6. prepare_data()", test_6_prepare_data),
    ("7. generate (3 samples)", test_7_generate_small),
    ("8. apply_propagation() e2e", test_8_apply_propagation_e2e),
]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Propagation micro-tests")
    parser.add_argument("--test", type=str, default=None,
                        help="Comma-separated test numbers to run, e.g. '1,2,3'")
    args = parser.parse_args()

    if args.test:
        indices = [int(x.strip()) - 1 for x in args.test.split(",")]
        tests_to_run = [ALL_TESTS[i] for i in indices if 0 <= i < len(ALL_TESTS)]
    else:
        tests_to_run = ALL_TESTS

    print(f"\n Running {len(tests_to_run)} propagation tests...\n")

    for name, func in tests_to_run:
        run_test(name, func)

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {len(passed)} passed, {len(failed)} failed")
    print(f"{'='*60}")
    if passed:
        for p in passed:
            print(f"  ✅ {p}")
    if failed:
        for f in failed:
            print(f"  ❌ {f}")
        sys.exit(1)
    else:
        print("\n🎉 All tests passed!")
