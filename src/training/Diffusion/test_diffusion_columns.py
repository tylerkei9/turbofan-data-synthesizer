#!/usr/bin/env python3
"""
Tests for the column-agnostic behaviour of diffusion_model5.py.

Each test isolates ONE of the four requirements:
  1. No hardcoded column names anywhere in the resolution path.
  2. No hardcoded column count: model adapts to whatever the data has.
  3. No dependence on column order: shuffling columns produces equivalent results.
  4. User does not need to know which columns to condition on: the model
     auto-selects them via the unsupervised variance + correlation heuristic.

Usage:
    python src/training/Diffusion/test_diffusion_columns.py            # Run all
    python src/training/Diffusion/test_diffusion_columns.py --test 1   # Run one
    python src/training/Diffusion/test_diffusion_columns.py --test 1,3 # Run several
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
    _detect_metadata_columns,
    auto_resolve_columns,
    select_conditioning_columns,
    prepare_conditional_data,
    save_engine_data,
    load_engine_data,
    ConditionalDenoiser,
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
    """Build deterministic synthetic data so tests are reproducible.

    Each column is a different deterministic signal so tests can verify
    column-by-column equivalence after reordering.
    """
    rng = np.random.default_rng(seed)
    cols = []
    for i, _ in enumerate(col_names):
        # Mix of distributions so variance ranking is meaningful
        if i % 4 == 0:
            cols.append(rng.normal(loc=i * 10, scale=i + 1, size=n_rows))
        elif i % 4 == 1:
            cols.append(rng.uniform(low=-i, high=i + 5, size=n_rows))
        elif i % 4 == 2:
            cols.append(np.linspace(0, i + 1, n_rows) + rng.normal(0, 0.1, n_rows))
        else:
            cols.append(rng.exponential(scale=i + 1, size=n_rows))
    return np.column_stack(cols).astype(np.float32)


def _write_csv(path: Path, columns: list, data: np.ndarray) -> None:
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(columns)
        w.writerows(data)


# ===========================================================================
# TEST 1: No hardcoded column names, arbitrary names work
# ===========================================================================
def test_1_no_hardcoded_names():
    """auto_resolve_columns must not depend on any specific column name.

    We give it columns named with random gibberish (no 'Engine_ID', no
    'Altitude', etc.) and it must still produce a sensible mapping.
    """
    columns = ["zxqv", "kpwo", "asdf", "mlkj", "qwer", "yuio"]
    data = _make_synthetic(200, columns, seed=1)

    mapping = auto_resolve_columns(data, columns, n_conditions=2)

    # No column names matched the time/engine_id hints, so metadata is empty
    assert mapping["metadata_idx"] == [], (
        f"expected no metadata for gibberish names, got {mapping['metadata_idx']}"
    )
    # Still picked exactly n_conditions condition columns
    assert len(mapping["condition_idx"]) == 2, (
        f"expected 2 conditions, got {len(mapping['condition_idx'])}"
    )
    # Every column is accounted for
    total = (len(mapping["metadata_idx"]) + len(mapping["condition_idx"])
             + len(mapping["sensor_idx"]))
    assert total == len(columns), f"column accounting mismatch: {mapping}"
    # Condition columns and sensor columns are disjoint
    assert set(mapping["condition_idx"]).isdisjoint(set(mapping["sensor_idx"]))
    print(f"  Resolved gibberish columns: conds={mapping['condition_idx']}, "
          f"targets={mapping['sensor_idx']}")


# ===========================================================================
# TEST 2: No hardcoded column count, model adapts to width
# ===========================================================================
def test_2_no_hardcoded_count():
    """The full pipeline must work for varied total column counts."""
    for n_total in (4, 9, 20):
        columns = [f"col_{i}" for i in range(n_total)]
        data = _make_synthetic(150, columns, seed=n_total)

        mapping = auto_resolve_columns(data, columns, n_conditions=3)
        meta, cond, sens, cstats, sstats = prepare_conditional_data(data, mapping)

        # Shapes flow through correctly
        assert sens.shape == (150, len(mapping["sensor_idx"]))
        assert cond.shape == (150, len(mapping["condition_idx"]))
        assert sstats["mean"].shape == (len(mapping["sensor_idx"]),)

        # Model can be built with the derived dims (no hardcoded 21/3 inside)
        model = ConditionalDenoiser(
            sensor_dim=sens.shape[1], condition_dim=cond.shape[1], hidden_dim=32,
            time_emb_dim=16, n_layers=2,
        )
        assert model.sensor_dim == sens.shape[1]
        assert model.condition_dim == cond.shape[1]
        print(f"  n_total={n_total}: sensor_dim={sens.shape[1]}, "
              f"condition_dim={cond.shape[1]} OK")


# ===========================================================================
# TEST 3: No dependence on column order
# ===========================================================================
def test_3_no_order_dependence():
    """Reordering CSV columns must not change which *columns* are picked.

    We build the same logical data twice, once in canonical order and once
    permuted, then verify:
      - The set of column NAMES selected as conditions is identical.
      - prepare_conditional_data produces the same per-column statistics
        for the targets, regardless of source order.
      - save_engine_data writes columns back in the original CSV order.
    """
    columns = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
    data = _make_synthetic(300, columns, seed=42)

    # Permutation: reverse, with a swap, so no column stays in place
    perm = [5, 0, 3, 1, 4, 2]
    perm_columns = [columns[i] for i in perm]
    perm_data = data[:, perm]

    map_a = auto_resolve_columns(data, columns, n_conditions=2)
    map_b = auto_resolve_columns(perm_data, perm_columns, n_conditions=2)

    names_a = {columns[i] for i in map_a["condition_idx"]}
    names_b = {perm_columns[i] for i in map_b["condition_idx"]}
    assert names_a == names_b, (
        f"conditioning selection depends on column order: {names_a} vs {names_b}"
    )

    # Per-column target statistics must agree once we identify columns by name
    _, _, sens_a, _, sstats_a = prepare_conditional_data(data, map_a)
    _, _, sens_b, _, sstats_b = prepare_conditional_data(perm_data, map_b)

    target_means_a = {columns[map_a["sensor_idx"][i]]: sstats_a["mean"][i]
                      for i in range(len(map_a["sensor_idx"]))}
    target_means_b = {perm_columns[map_b["sensor_idx"][i]]: sstats_b["mean"][i]
                      for i in range(len(map_b["sensor_idx"]))}
    assert set(target_means_a.keys()) == set(target_means_b.keys())
    for k in target_means_a:
        assert np.isclose(target_means_a[k], target_means_b[k], atol=1e-5), (
            f"target mean for {k} differs: {target_means_a[k]} vs {target_means_b[k]}"
        )

    # save_engine_data must place each role back at its original index
    out = SCRIPT_DIR / "_test_roundtrip.csv"
    try:
        meta_b, cond_b_std, sens_b_std, cstats_b, sstats_b2 = prepare_conditional_data(
            perm_data, map_b)
        # De-standardise to recover original values
        cond_b = (cond_b_std * cstats_b["std"] + cstats_b["mean"]
                  if cond_b_std.shape[1] > 0 else cond_b_std)
        sens_b_raw = sens_b_std * sstats_b2["std"] + sstats_b2["mean"]
        save_engine_data(meta_b, cond_b, sens_b_raw,
                         str(out), perm_columns, map_b)

        loaded, loaded_cols = load_engine_data(str(out))
        assert loaded_cols == perm_columns, (
            f"saved column header order changed: {loaded_cols} vs {perm_columns}"
        )
        # Each column ends up in the slot the header says it should
        for i, name in enumerate(perm_columns):
            original_col_idx_in_perm = i
            assert np.allclose(loaded[:, i], perm_data[:, original_col_idx_in_perm],
                               atol=1e-3), f"column {name} round-trip mismatch"
        print(f"  Order-independent: same conditions picked, round-trip OK")
    finally:
        if out.exists():
            out.unlink()


# ===========================================================================
# TEST 4: User does not need to know which columns to condition on
# ===========================================================================
def test_4_auto_selects_conditions():
    """select_conditioning_columns must pick informative columns automatically.

    We construct data where:
      - col 0 is constant       -> must ALWAYS be REJECTED (no signal)
      - col 1 is high-variance, independent
      - col 2 = col 1 * 2 + eps -> highly correlated with col 1
      - col 3 is high-variance, independent
      - col 4 is low-variance noise

    The heuristic guarantees:
      * the constant column is never picked
      * at most one of {col 1, col 2} is picked (they are near-duplicates)
      * an independent diverse second column is picked
    The user supplies no hints whatsoever.
    """
    n = 500
    rng = np.random.default_rng(7)
    col0 = np.full(n, 3.14, dtype=np.float64)
    col1 = rng.normal(0, 50, n)
    col2 = col1 * 2.0 + rng.normal(0, 0.01, n)
    col3 = rng.normal(0, 40, n)
    col4 = rng.normal(0, 0.5, n)
    data = np.column_stack([col0, col1, col2, col3, col4]).astype(np.float32)

    candidate_idx = [0, 1, 2, 3, 4]
    picked = select_conditioning_columns(data, candidate_idx, n_conditions=2,
                                         corr_threshold=0.9)

    assert 0 not in picked, f"constant column was selected: {picked}"
    assert len(picked) == 2, f"expected 2 conditions, got {picked}"
    # At most one of the correlated pair {1, 2} survives
    assert not ({1, 2}.issubset(set(picked))), (
        f"both members of correlated pair were picked: {picked}"
    )
    # The independent high-variance column must be picked alongside it
    assert 3 in picked, f"independent high-variance col 3 not picked: {picked}"
    print(f"  Auto-picked conditions: {picked} (rejected constant + correlated)")


# ===========================================================================
# TEST 5: Metadata auto-detection, gracefully handles "no engine_id"
# ===========================================================================
def test_5_metadata_optional():
    """Per spec, only time and engine_id MAY be present. Code must work
    when both are present, only one is present, and when neither exists.
    """
    n = 100
    rng = np.random.default_rng(0)

    # Case A: both metadata columns present (mixed casing / underscores)
    cols_a = ["EngineId", "TimeStep", "x", "y", "z"]
    data_a = _make_synthetic(n, cols_a, seed=10)
    meta_a = _detect_metadata_columns(cols_a)
    assert 0 in meta_a and 1 in meta_a, f"failed to detect metadata: {meta_a}"

    # Case B: only time present
    cols_b = ["timestep", "v1", "v2", "v3"]
    data_b = _make_synthetic(n, cols_b, seed=11)
    meta_b = _detect_metadata_columns(cols_b)
    assert meta_b == [0], f"expected only time col, got {meta_b}"

    # Case C: neither present; pipeline still resolves and trains a model
    cols_c = ["foo", "bar", "baz", "qux"]
    data_c = _make_synthetic(n, cols_c, seed=12)
    meta_c = _detect_metadata_columns(cols_c)
    assert meta_c == [], f"expected no metadata, got {meta_c}"

    mapping = auto_resolve_columns(data_c, cols_c, n_conditions=1)
    assert mapping["metadata_idx"] == []
    assert len(mapping["condition_idx"]) == 1
    assert len(mapping["sensor_idx"]) == 3
    print("  Metadata detection handles all 3 presence cases")


# ===========================================================================
# Runner
# ===========================================================================

ALL_TESTS = [
    ("1: no hardcoded column names",          test_1_no_hardcoded_names),
    ("2: no hardcoded column count",          test_2_no_hardcoded_count),
    ("3: no dependence on column order",      test_3_no_order_dependence),
    ("4: auto-selects conditioning columns",  test_4_auto_selects_conditions),
    ("5: metadata detection is optional",     test_5_metadata_optional),
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
