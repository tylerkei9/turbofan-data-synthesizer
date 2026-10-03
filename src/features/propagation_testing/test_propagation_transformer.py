#!/usr/bin/env python3
"""
Small, incremental tests for TRANSFORMER propagation.
"""

import sys
import traceback
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TRANSFORMER_CKPT = PROJECT_ROOT / "src" /  "model_checkPoints" / "transformer" / "checkpoint_best.pt" 

passed, failed = [], []


def run_test(name, func):
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


def test_1_imports():
    from src.features.propagation import load_model, prepare_data_for_inference, generate_with_model, apply_propagation
    assert callable(load_model)
    assert callable(prepare_data_for_inference)
    assert callable(generate_with_model)
    assert callable(apply_propagation)
    print("  Imported public functions ✓")


def test_2_checkpoint_exists():
    assert TRANSFORMER_CKPT.exists(), f"Transformer checkpoint not found: {TRANSFORMER_CKPT}"
    size_mb = TRANSFORMER_CKPT.stat().st_size / (1024 * 1024)
    print(f"  Checkpoint found: {TRANSFORMER_CKPT}")
    print(f"  Size: {size_mb:.1f} MB")


def test_3_checkpoint_contents():
    import torch
    ckpt = torch.load(TRANSFORMER_CKPT, map_location="cpu", weights_only=False)
    assert isinstance(ckpt, dict), f"Expected dict checkpoint, got {type(ckpt)}"
    keys = set(ckpt.keys())
    print(f"  Top-level keys: {sorted(list(keys))}")

    assert "feature_cols" in keys or "features" in keys
    assert "model_state_dict" in keys or "state_dict" in keys
    assert "config" in keys
    print("  Required transformer keys present ✓")
    print("  Note: col_mins/ranges are missing (OK: runtime fallback normalization will be used).")


def test_4_load_model():
    from src.features.propagation import load_model
    bundle = load_model(str(TRANSFORMER_CKPT), "transformer", device="cpu")
    for k in ["model", "feature_cols", "eng_ids", "eng_to_idx", "input_window", "device"]:
        assert k in bundle, f"Bundle missing: {k}"
    assert not bundle["model"].training
    print("  load_model(transformer) works ✓")


def test_5_prepare_data():
    import pandas as pd
    import numpy as np
    from src.features.propagation import load_model, prepare_data_for_inference

    bundle = load_model(str(TRANSFORMER_CKPT), "transformer", device="cpu")
    feature_cols = bundle["feature_cols"]

    n = 20
    df = pd.DataFrame({c: np.random.uniform(0, 1, n).astype(np.float32) for c in feature_cols})
    df.insert(0, "Engine_ID", 1)

    prepared = prepare_data_for_inference(df, "transformer", bundle)

    assert "normalized" in prepared
    assert prepared["normalized"].shape == (n, len(prepared["feature_cols"]))
    assert np.isfinite(prepared["normalized"]).all()
    assert prepared.get("norm_source") in ("checkpoint", "runtime_data")
    print(f"  prepare_data ok; norm_source={prepared.get('norm_source')} ✓")


def test_6_generate_small():
    import pandas as pd
    import numpy as np
    from src.features.propagation import load_model, prepare_data_for_inference, generate_with_model

    bundle = load_model(str(TRANSFORMER_CKPT), "transformer", device="cpu")
    feature_cols = bundle["feature_cols"]

    n = max(10, int(bundle["input_window"]) // 4)
    df = pd.DataFrame({c: np.random.uniform(0, 1, n).astype(np.float32) for c in feature_cols})
    df.insert(0, "Engine_ID", 1)

    prepared = prepare_data_for_inference(df, "transformer", bundle)
    out = generate_with_model(bundle, prepared, num_samples=3, model_type="transformer")

    assert isinstance(out, pd.DataFrame)
    assert len(out) == 3
    assert "Engine_ID" in out.columns and "cycle" in out.columns
    assert "norm_source" in out.columns
    assert out.select_dtypes(include=[np.number]).notna().all().all()
    print(f"  generate ok; output columns={len(out.columns)} ✓")
    print(f"  First row: {out.iloc[0].to_dict()}")


def test_7_apply_propagation_e2e():
    import pandas as pd
    import numpy as np
    from src.features.propagation import load_model, apply_propagation

    bundle = load_model(str(TRANSFORMER_CKPT), "transformer", device="cpu")
    feature_cols = bundle["feature_cols"]

    n = 20
    df = pd.DataFrame({c: np.random.uniform(0, 1, n).astype(np.float32) for c in feature_cols})
    df.insert(0, "Engine_ID", 1)

    out = apply_propagation(
        modified_data=df,
        model_type="transformer",
        checkpoint_path=str(TRANSFORMER_CKPT),
        num_samples=3,
        device="cpu",
    )
    assert isinstance(out, pd.DataFrame)
    assert len(out) == 3
    print("  apply_propagation(transformer) e2e ✓")


ALL_TESTS = [
    ("1. Imports work", test_1_imports),
    ("2. Transformer checkpoint exists", test_2_checkpoint_exists),
    ("3. Checkpoint contents", test_3_checkpoint_contents),
    ("4. load_model(transformer)", test_4_load_model),
    ("5. prepare_data(transformer)", test_5_prepare_data),
    ("6. generate(transformer, 3 samples)", test_6_generate_small),
    ("7. apply_propagation(transformer) e2e", test_7_apply_propagation_e2e),
]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Transformer propagation micro-tests")
    parser.add_argument("--test", type=str, default=None, help="Comma-separated test numbers, e.g. '1,2,3'")
    args = parser.parse_args()

    if args.test:
        indices = [int(x.strip()) - 1 for x in args.test.split(",")]
        tests_to_run = [ALL_TESTS[i] for i in indices if 0 <= i < len(ALL_TESTS)]
    else:
        tests_to_run = ALL_TESTS

    print(f"\n Running {len(tests_to_run)} transformer propagation tests...\n")

    for name, func in tests_to_run:
        run_test(name, func)

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
        print("\n🎉 All transformer tests passed!")