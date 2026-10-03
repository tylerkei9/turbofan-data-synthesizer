#!/usr/bin/env python3
"""
Tests for multi-file CSV input support in diffusion_model5.py.

Each test targets ONE aspect of the multi-file loader:
  1. Backward compatibility: a single string path still works exactly as before.
  2. Concatenation: a list of files with matching headers yields concatenated rows.
  3. Header-order independence: files may list columns in different orders as
     long as the column *names* match; data is aligned to the first file.
  4. Mismatched column sets raise a clear error.
  5. The downstream pipeline (auto_resolve_columns + prepare_conditional_data)
     works transparently on data loaded from multiple files.

Usage:
    python src/training/Diffusion/test_diffusion_multifile.py            # Run all
    python src/training/Diffusion/test_diffusion_multifile.py --test 1   # Run one
    python src/training/Diffusion/test_diffusion_multifile.py --test 1,3 # Run several
"""

import sys
import traceback
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.Diffusion.diffusion_model5 import (
    load_engine_data,
    auto_resolve_columns,
    prepare_conditional_data,
)

passed = []
failed = []


def run_test(name, func):
    print(f"\n{'=' * 60}")
    print(f"TEST: {name}")
    print(f"{'=' * 60}")
    try:
        func()
        print(f"  PASS: {name}")
        passed.append(name)
    except Exception as e:
        print(f"  FAIL: {name}")
        print(f"  Error: {e}")
        traceback.print_exc()
        failed.append(name)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic(n_rows: int, col_names: list, seed: int = 0) -> np.ndarray:
    """Deterministic synthetic data so tests are reproducible."""
    rng = np.random.default_rng(seed)
    cols = []
    for i, _ in enumerate(col_names):
        if i % 3 == 0:
            cols.append(rng.normal(loc=i * 10, scale=i + 1, size=n_rows))
        elif i % 3 == 1:
            cols.append(rng.uniform(low=-i, high=i + 5, size=n_rows))
        else:
            cols.append(np.linspace(0, i + 1, n_rows) + rng.normal(0, 0.1, n_rows))
    return np.column_stack(cols).astype(np.float32)


def _write_csv(path: Path, columns: list, data: np.ndarray) -> None:
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(columns)
        w.writerows(data)


def _tmp(name: str) -> Path:
    return SCRIPT_DIR / f"_test_mf_{name}.csv"


# ===========================================================================
# TEST 1: Backward compatibility — single path string still works
# ===========================================================================
def test_1_single_path_backward_compat():
    """Passing a single string path must behave exactly as before."""
    columns = ["a", "b", "c", "d"]
    data = _make_synthetic(50, columns, seed=1)
    p = _tmp("single")
    _write_csv(p, columns, data)

    try:
        loaded, loaded_cols = load_engine_data(str(p))
        assert loaded_cols == columns, f"header mismatch: {loaded_cols}"
        assert loaded.shape == (50, 4), f"shape mismatch: {loaded.shape}"
        assert np.allclose(loaded, data, atol=1e-3), "row-values drifted"
        print(f"  Single-path load: shape={loaded.shape}, cols={loaded_cols}")
    finally:
        p.unlink(missing_ok=True)


# ===========================================================================
# TEST 2: Two files with identical headers concatenate row-wise
# ===========================================================================
def test_2_concat_matching_headers():
    """A list of two CSVs with the same header yields concatenated rows."""
    columns = ["x", "y", "z"]
    data1 = _make_synthetic(30, columns, seed=2)
    data2 = _make_synthetic(45, columns, seed=3)
    p1 = _tmp("concat1")
    p2 = _tmp("concat2")
    _write_csv(p1, columns, data1)
    _write_csv(p2, columns, data2)

    try:
        loaded, loaded_cols = load_engine_data([str(p1), str(p2)])
        assert loaded_cols == columns, f"header mismatch: {loaded_cols}"
        assert loaded.shape == (75, 3), f"shape mismatch: {loaded.shape}"
        # First 30 rows must be from file 1; next 45 must be from file 2.
        assert np.allclose(loaded[:30], data1, atol=1e-3), "file 1 block drifted"
        assert np.allclose(loaded[30:], data2, atol=1e-3), "file 2 block drifted"
        print(f"  Concatenated two files: shape={loaded.shape}")
    finally:
        p1.unlink(missing_ok=True)
        p2.unlink(missing_ok=True)


# ===========================================================================
# TEST 3: Files with reordered columns still align correctly
# ===========================================================================
def test_3_reordered_columns_align():
    """Second file's columns are in a different order; loader must realign."""
    columns = ["alpha", "beta", "gamma", "delta"]
    data1 = _make_synthetic(20, columns, seed=4)

    # Same logical rows, but the CSV lists columns in a different order
    perm = [2, 0, 3, 1]
    perm_columns = [columns[i] for i in perm]
    data2_perm = data1[:, perm]  # same data, permuted columns

    p1 = _tmp("order1")
    p2 = _tmp("order2")
    _write_csv(p1, columns, data1)
    _write_csv(p2, perm_columns, data2_perm)

    try:
        loaded, loaded_cols = load_engine_data([str(p1), str(p2)])
        assert loaded_cols == columns, (
            f"expected header from first file, got {loaded_cols}"
        )
        assert loaded.shape == (40, 4), f"shape mismatch: {loaded.shape}"
        # Both halves should equal data1 once realigned
        assert np.allclose(loaded[:20], data1, atol=1e-3), "file 1 drifted"
        assert np.allclose(loaded[20:], data1, atol=1e-3), (
            "file 2 was not realigned to the first file's column order"
        )
        print(f"  Column reorder handled: shape={loaded.shape}")
    finally:
        p1.unlink(missing_ok=True)
        p2.unlink(missing_ok=True)


# ===========================================================================
# TEST 4: Mismatched column sets raise a clear error
# ===========================================================================
def test_4_mismatched_headers_errors():
    """If a secondary file's column set differs, the loader must error out."""
    cols_a = ["a", "b", "c"]
    cols_b = ["a", "b", "d"]  # 'c' missing, 'd' extra
    data_a = _make_synthetic(10, cols_a, seed=5)
    data_b = _make_synthetic(10, cols_b, seed=6)
    pa = _tmp("bad_a")
    pb = _tmp("bad_b")
    _write_csv(pa, cols_a, data_a)
    _write_csv(pb, cols_b, data_b)

    try:
        raised = False
        try:
            load_engine_data([str(pa), str(pb)])
        except ValueError as e:
            raised = True
            msg = str(e)
            assert "missing" in msg and "extra" in msg, (
                f"error should name missing/extra columns, got: {msg}"
            )
            print(f"  Got expected ValueError: {msg}")
        assert raised, "expected ValueError for mismatched headers"
    finally:
        pa.unlink(missing_ok=True)
        pb.unlink(missing_ok=True)


# ===========================================================================
# TEST 5: Downstream pipeline consumes the combined data transparently
# ===========================================================================
def test_5_pipeline_works_on_combined():
    """auto_resolve_columns + prepare_conditional_data must see the combined
    dataset as one contiguous frame — no trace of the file boundary.
    """
    columns = ["engine_id", "timestep", "p0", "p1", "p2", "p3"]
    data1 = _make_synthetic(80, columns, seed=7)
    data2 = _make_synthetic(120, columns, seed=8)
    p1 = _tmp("pipe1")
    p2 = _tmp("pipe2")
    _write_csv(p1, columns, data1)
    _write_csv(p2, columns, data2)

    try:
        loaded, loaded_cols = load_engine_data([str(p1), str(p2)])
        assert loaded.shape == (200, 6)

        mapping = auto_resolve_columns(loaded, loaded_cols, n_conditions=2)
        # Metadata should pick up engine_id + timestep by name
        assert 0 in mapping["metadata_idx"] and 1 in mapping["metadata_idx"], (
            f"expected metadata from names, got {mapping['metadata_idx']}"
        )
        assert len(mapping["condition_idx"]) == 2
        # Every column is accounted for exactly once
        total = (len(mapping["metadata_idx"]) + len(mapping["condition_idx"])
                 + len(mapping["sensor_idx"]))
        assert total == len(loaded_cols)

        meta, cond, sens, cstats, sstats = prepare_conditional_data(loaded, mapping)
        assert meta.shape[0] == 200
        assert cond.shape == (200, len(mapping["condition_idx"]))
        assert sens.shape == (200, len(mapping["sensor_idx"]))
        print(f"  Pipeline on combined data: meta={meta.shape}, "
              f"cond={cond.shape}, sens={sens.shape}")
    finally:
        p1.unlink(missing_ok=True)
        p2.unlink(missing_ok=True)


# ===========================================================================
# TEST 6: Single-element list behaves like a single path
# ===========================================================================
def test_6_single_element_list():
    """A list containing one path should give the same result as the raw path."""
    columns = ["m", "n", "o"]
    data = _make_synthetic(25, columns, seed=9)
    p = _tmp("single_list")
    _write_csv(p, columns, data)

    try:
        a_data, a_cols = load_engine_data(str(p))
        b_data, b_cols = load_engine_data([str(p)])
        assert a_cols == b_cols
        assert np.array_equal(a_data, b_data), (
            "list-of-one must match string-path behaviour"
        )
        print(f"  [path] and 'path' agree: shape={b_data.shape}")
    finally:
        p.unlink(missing_ok=True)


# ===========================================================================
# Runner
# ===========================================================================

ALL_TESTS = [
    ("1: single-path backward compat",       test_1_single_path_backward_compat),
    ("2: concat files with matching headers", test_2_concat_matching_headers),
    ("3: reordered columns align",            test_3_reordered_columns_align),
    ("4: mismatched headers raise error",     test_4_mismatched_headers_errors),
    ("5: downstream pipeline on combined",    test_5_pipeline_works_on_combined),
    ("6: single-element list equals path",    test_6_single_element_list),
]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, default=None,
                        help="Comma-separated test numbers (e.g. '1,3')")
    args = parser.parse_args()

    selected = ALL_TESTS
    if args.test:
        wanted = {int(x) for x in args.test.split(",")}
        selected = [t for i, t in enumerate(ALL_TESTS, start=1) if i in wanted]

    for name, fn in selected:
        run_test(name, fn)

    print(f"\n{'=' * 60}")
    print(f"RESULTS: {len(passed)} passed, {len(failed)} failed")
    print(f"{'=' * 60}")
    if failed:
        for f in failed:
            print(f"  FAILED: {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
