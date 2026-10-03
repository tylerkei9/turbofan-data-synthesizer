#!/usr/bin/env python3
"""
Tests for the advanced diffusion features added in the project-context rewrite:
  - learned column embeddings + attention-based self-conditioning (transformer)
  - inpainting sampler for subset-column seed data
  - LoRA adapters + seed-data fine-tuning + adapter composition
  - snapshot vs continuous format detection + single-engine fallback
  - MMD / UMAP / domain-classifier evaluation
  - unit-insensitive variance ranking

Each test isolates ONE feature and uses small in-memory data so it stays fast.

Usage:
    python src/training/Diffusion/test_diffusion_advanced.py
    python src/training/Diffusion/test_diffusion_advanced.py --test 1,3
"""

import sys
import traceback
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.Diffusion.diffusion_model5 import (
    # Existing pieces still in use
    DiffusionSchedule,
    auto_resolve_columns,
    select_conditioning_columns,
    load_engine_data,
    # New pieces
    TabularTransformerDenoiser,
    compute_column_stats,
    COLUMN_STATS_DIM,
    sample_random_mask,
    masked_p_losses,
    train_transformer,
    sample_inpaint,
    LoRALinear,
    apply_lora_to_model,
    extract_lora_state,
    load_lora_state,
    fine_tune_with_seed,
    detect_data_format,
    group_rows_by_engine,
    mmd_rbf,
    domain_classifier_accuracy,
    evaluate_synthetic,
)

from torch.utils.data import DataLoader, TensorDataset

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

def _make_two_modes(n: int, seed: int = 0) -> np.ndarray:
    """Two-cluster Gaussian mixture in 4 dims, easy for the transformer to learn."""
    rng = np.random.default_rng(seed)
    half = n // 2
    a = rng.normal(loc=[0, 0, 5, 5], scale=0.5, size=(half, 4))
    b = rng.normal(loc=[5, 5, 0, 0], scale=0.5, size=(n - half, 4))
    return np.vstack([a, b]).astype(np.float32)


def _train_tiny_transformer(x_std: np.ndarray, epochs: int = 30):
    n_cols = x_std.shape[1]
    model = TabularTransformerDenoiser(n_cols=n_cols, hidden_dim=32,
                                       n_heads=2, n_layers=2)
    model.set_column_stats(compute_column_stats(x_std))
    schedule = DiffusionSchedule(T=20)
    loader = DataLoader(TensorDataset(torch.from_numpy(x_std).float()),
                        batch_size=64, shuffle=True)
    train_transformer(model, schedule, loader, epochs=epochs, lr=2e-3,
                      device=torch.device("cpu"), verbose=False)
    return model, schedule


# ===========================================================================
# TEST 1: Variance heuristic is unit-insensitive
# ===========================================================================
def test_1_unit_insensitive_heuristic():
    """A column rescaled by 1e6 must NOT dominate selection.

    Two columns with identical *shape* but different units (one in Pa,
    one in K) should both be eligible — the heuristic must score them
    equivalently.
    """
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, 400)

    # Same shape, vastly different scales
    col_pa = base * 1_000_000  # "pressure"
    col_k = base * 1.0          # "temperature"
    # Diverse third column so the picker has a meaningful choice
    col_other = rng.normal(0, 1, 400)

    data = np.column_stack([col_pa, col_k, col_other]).astype(np.float32)
    picked = select_conditioning_columns(data, [0, 1, 2], n_conditions=2)

    # The huge-magnitude column must NOT crowd out the small-magnitude one
    # purely on unit grounds. Both should be candidates of equal standing.
    # In particular, picking exactly one of {0, 1} (since they're correlated)
    # plus column 2 is correct.
    assert 2 in picked, f"diverse independent column 2 not picked: {picked}"
    # The pa/k pair is perfectly correlated; only one should survive
    pa_or_k = (0 in picked) + (1 in picked)
    assert pa_or_k == 1, f"unit-pair handling wrong: {picked}"
    print(f"  Unit-blind selection picked: {picked}")


# ===========================================================================
# TEST 2: Column statistics produce stable, unit-free features
# ===========================================================================
def test_2_column_stats_unit_free():
    """Same shape, different scales => same column-stats vector."""
    rng = np.random.default_rng(1)
    raw = rng.normal(0, 1, 500)
    a = raw * 1.0
    b = raw * 1e6
    data = np.column_stack([a, b]).astype(np.float32)

    stats = compute_column_stats(data)
    assert stats.shape == (2, COLUMN_STATS_DIM)
    assert np.allclose(stats[0], stats[1], atol=1e-5), (
        f"unit-free stats differ across scales: {stats}"
    )
    print(f"  Stats invariant to scale: {stats[0]}")


# ===========================================================================
# TEST 3: Transformer denoiser is column-order invariant
# ===========================================================================
def test_3_transformer_order_invariant():
    """Permuting input columns must give a permuted output prediction.

    The transformer's column identity comes from per-column statistics,
    not from positional embeddings, so this is the critical invariance.
    """
    n_cols = 5
    model = TabularTransformerDenoiser(n_cols=n_cols, hidden_dim=16,
                                       n_heads=2, n_layers=2)
    rng = np.random.default_rng(2)
    fake_data = rng.normal(0, 1, (200, n_cols)).astype(np.float32)
    stats = compute_column_stats(fake_data)
    model.set_column_stats(stats)
    model.eval()

    x = torch.tensor(fake_data[:4], dtype=torch.float32)
    t = torch.tensor([5, 5, 5, 5], dtype=torch.long)
    mask = torch.zeros_like(x)

    with torch.no_grad():
        y = model(x, t, mask)

    # Permute columns AND the matching column-stats. Build a fresh model
    # with the permuted stats and check it gives the permuted output.
    perm = [4, 2, 0, 3, 1]
    x_perm = x[:, perm]
    model2 = TabularTransformerDenoiser(n_cols=n_cols, hidden_dim=16,
                                        n_heads=2, n_layers=2)
    model2.load_state_dict(model.state_dict())
    model2.set_column_stats(stats[perm])
    model2.eval()
    with torch.no_grad():
        y_perm = model2(x_perm, t, mask)

    # The denoised value at logical column j should match between the two
    # models, regardless of which physical position it lives at.
    for original_pos, new_pos in enumerate(perm):
        assert torch.allclose(y[:, new_pos], y_perm[:, original_pos], atol=1e-4), (
            f"column {new_pos}->{original_pos} mismatch: "
            f"{y[:, new_pos]} vs {y_perm[:, original_pos]}"
        )
    print("  Transformer output follows column permutation correctly")


# ===========================================================================
# TEST 4: Masked diffusion training reduces loss
# ===========================================================================
def test_4_masked_training_learns():
    """The transformer should drive masked-diffusion loss down on a tiny
    two-mode dataset."""
    x = _make_two_modes(400, seed=3)
    model = TabularTransformerDenoiser(n_cols=4, hidden_dim=32,
                                       n_heads=2, n_layers=2)
    model.set_column_stats(compute_column_stats(x))
    schedule = DiffusionSchedule(T=20)
    loader = DataLoader(TensorDataset(torch.from_numpy(x).float()),
                        batch_size=64, shuffle=True)

    # Snapshot loss before
    model.eval()
    batch = next(iter(loader))[0]
    t = torch.zeros(batch.shape[0], dtype=torch.long)
    mask = sample_random_mask(batch.shape[0], 4, batch.device)
    loss_before = masked_p_losses(model, schedule, batch, t, mask).item()

    train_transformer(model, schedule, loader, epochs=20, lr=2e-3,
                      device=torch.device("cpu"), verbose=False)

    model.eval()
    loss_after = masked_p_losses(model, schedule, batch, t, mask).item()
    assert loss_after < loss_before * 0.7, (
        f"loss did not drop enough: {loss_before:.4f} -> {loss_after:.4f}"
    )
    print(f"  Masked diffusion loss: {loss_before:.4f} -> {loss_after:.4f}")


# ===========================================================================
# TEST 5: Inpainting sampler honours observed columns
# ===========================================================================
def test_5_inpaint_respects_observed():
    """Columns marked as observed must come back unchanged in the sample."""
    x = _make_two_modes(400, seed=4)
    model, schedule = _train_tiny_transformer(x, epochs=15)

    n = 8
    obs_vals = torch.tensor(x[:n], dtype=torch.float32)
    obs_mask = torch.zeros(n, 4)
    obs_mask[:, [0, 2]] = 1.0  # observe cols 0 and 2

    samples = sample_inpaint(model, schedule, n_samples=n,
                             observed_values=obs_vals,
                             observed_mask=obs_mask,
                             device=torch.device("cpu"), verbose=False)

    # Observed columns must be (very nearly) preserved
    for j in (0, 2):
        diff = np.max(np.abs(samples[:, j] - x[:n, j]))
        assert diff < 1e-3, f"observed column {j} drifted by {diff}"
    print("  Inpainted observed columns recovered exactly")


# ===========================================================================
# TEST 6: LoRA wraps linears and lets the model still run
# ===========================================================================
def test_6_lora_wrap_and_run():
    n_cols = 4
    model = TabularTransformerDenoiser(n_cols=n_cols, hidden_dim=16,
                                       n_heads=2, n_layers=2)
    rng = np.random.default_rng(5)
    model.set_column_stats(compute_column_stats(
        rng.normal(0, 1, (100, n_cols)).astype(np.float32)))

    lora_params = apply_lora_to_model(model, rank=2, alpha=4.0)
    n_lora = sum(1 for m in model.modules() if isinstance(m, LoRALinear))

    # MultiheadAttention's internal projections are intentionally skipped, so
    # we just verify that *some* layers were wrapped and that the parameter
    # count matches (one A and one B matrix per wrap).
    assert n_lora > 0, "no linears were wrapped"
    assert len(lora_params) == 2 * n_lora

    # Forward still works
    model.eval()
    x = torch.zeros(2, n_cols)
    t = torch.zeros(2, dtype=torch.long)
    mask = torch.zeros_like(x)
    with torch.no_grad():
        y = model(x, t, mask)
    assert y.shape == (2, n_cols)

    # Only LoRA params should require grad
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert len(trainable) == len(lora_params), (
        f"expected {len(lora_params)} trainable params, got {len(trainable)}")

    # Save / restore round-trip
    state = extract_lora_state(model)
    assert all(k.endswith("lora_A") or k.endswith("lora_B") for k in state.keys())

    # Bump A so it's nonzero, save, zero it, restore — should match.
    for p in lora_params:
        p.data.add_(0.1)
    state = extract_lora_state(model)
    for p in lora_params:
        p.data.zero_()
    load_lora_state(model, state)
    for k, v in state.items():
        own = dict(model.named_parameters())[k]
        assert torch.allclose(own, v.to(own.device))
    print(f"  LoRA wrapped {n_lora} linears, save/restore OK")


# ===========================================================================
# TEST 7: Seed-data fine-tune produces a smaller adapter than the base ckpt
# ===========================================================================
def test_7_seed_finetune_subset_columns():
    """The fine-tune workflow must accept a seed CSV that has only a SUBSET
    of the base model's columns."""
    import csv
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    try:
        # Train a tiny base model on 4 columns
        x = _make_two_modes(300, seed=6)
        model, schedule = _train_tiny_transformer(x, epochs=10)

        # Save base ckpt with the metadata fine_tune_with_seed expects
        cols = ["a", "b", "c", "d"]
        col_indices = {"metadata_idx": [], "condition_idx": [], "sensor_idx": [0, 1, 2, 3]}
        sensor_stats = {
            "mean": np.zeros(4, dtype=np.float32),
            "std": np.ones(4, dtype=np.float32),
        }
        ckpt = {
            "arch": "transformer",
            "model_state": model.state_dict(),
            "schedule": {"T": schedule.T, "betas": schedule.betas.cpu().numpy()},
            "columns": cols,
            "col_indices": col_indices,
            "sensor_stats": sensor_stats,
            "column_stats": compute_column_stats(x),
            "model_cfg": {"hidden_dim": 32, "n_heads": 2, "n_layers": 2},
        }
        base_path = tmp / "base.pt"
        torch.save(ckpt, base_path)

        # Seed CSV containing ONLY columns "a" and "c" (subset)
        seed_path = tmp / "seed.csv"
        with open(seed_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["a", "c"])
            seed_data = _make_two_modes(80, seed=7)[:, [0, 2]]
            w.writerows(seed_data)

        adapter_path = tmp / "adapter.pt"
        fine_tune_with_seed(
            base_ckpt_path=str(base_path),
            seed_data_path=str(seed_path),
            adapter_out_path=str(adapter_path),
            epochs=2, lr=1e-4, rank=2, alpha=4.0,
            batch_size=16, verbose=False,
        )

        adapter = torch.load(adapter_path, map_location="cpu", weights_only=False)
        assert "lora_state" in adapter
        assert adapter["rank"] == 2
        assert adapter["seed_columns"] == ["a", "c"]

        # Adapter file should be much smaller than the base checkpoint
        base_size = base_path.stat().st_size
        adapter_size = adapter_path.stat().st_size
        assert adapter_size < base_size, (
            f"adapter ({adapter_size}B) not smaller than base ({base_size}B)")
        print(f"  Adapter trained on subset cols, "
              f"size {adapter_size}B vs base {base_size}B")
    finally:
        import shutil
        shutil.rmtree(tmp)


# ===========================================================================
# TEST 8: Snapshot vs continuous format detection
# ===========================================================================
def test_8_format_detection():
    # Continuous: 5 engines, 20 timesteps each
    cont = []
    for eid in range(5):
        for t in range(20):
            cont.append([eid, t, np.random.randn()])
    cont = np.array(cont, dtype=np.float32)
    cols_cont = ["engine_id", "time", "value"]
    ci_cont = auto_resolve_columns(cont, cols_cont, n_conditions=1)
    fmt = detect_data_format(cont, ci_cont)
    assert fmt == "continuous", f"expected continuous, got {fmt}"

    # Snapshot: 50 unique engines, 1 row each
    snap = np.column_stack([np.arange(50), np.random.randn(50)]).astype(np.float32)
    cols_snap = ["engine_id", "value"]
    ci_snap = auto_resolve_columns(snap, cols_snap, n_conditions=1)
    fmt2 = detect_data_format(snap, ci_snap)
    assert fmt2 == "snapshot", f"expected snapshot, got {fmt2}"

    # No engine_id => single-engine fallback, snapshot format
    plain = np.random.randn(30, 3).astype(np.float32)
    cols_plain = ["x", "y", "z"]
    ci_plain = auto_resolve_columns(plain, cols_plain, n_conditions=1)
    groups = group_rows_by_engine(plain, ci_plain)
    assert len(groups) == 1, f"no-engine-id should be one group, got {len(groups)}"
    assert len(groups[0][1]) == 30
    print("  Format detection: continuous, snapshot, single-engine fallback")


# ===========================================================================
# TEST 9: MMD and domain classifier behave on identical vs different data
# ===========================================================================
def test_9_evaluation_metrics():
    rng = np.random.default_rng(8)
    # Identical-distribution case: small MMD, ~0.5 classifier accuracy
    a = rng.normal(0, 1, (400, 4))
    b = rng.normal(0, 1, (400, 4))
    mmd_close = mmd_rbf(a, b)
    acc_close = domain_classifier_accuracy(a, b)
    assert mmd_close < 0.05, f"identical-dist MMD too large: {mmd_close}"
    assert 0.4 < acc_close < 0.6, f"identical-dist acc not near 0.5: {acc_close}"

    # Wildly different distribution: large MMD, near-perfect classifier
    c = rng.normal(8, 1, (400, 4))
    mmd_far = mmd_rbf(a, c)
    acc_far = domain_classifier_accuracy(a, c)
    assert mmd_far > mmd_close * 5, (
        f"shifted-dist MMD ({mmd_far}) not >> identical ({mmd_close})")
    assert acc_far > 0.9, f"shifted-dist classifier acc too low: {acc_far}"
    print(f"  MMD identical={mmd_close:.4f}, shifted={mmd_far:.4f}; "
          f"acc identical={acc_close:.3f}, shifted={acc_far:.3f}")


# ===========================================================================
# TEST 10: evaluate_synthetic returns the full report dict
# ===========================================================================
def test_10_evaluate_synthetic_full_report():
    rng = np.random.default_rng(9)
    real = rng.normal(0, 1, (200, 3))
    synth = rng.normal(0.05, 1, (200, 3))
    report = evaluate_synthetic(real, synth, umap_path=None, verbose=False)
    for key in ("ks_mean", "corr_dist", "mmd", "domain_acc", "umap_path"):
        assert key in report, f"missing report key: {key}"
    assert report["umap_path"] is None  # asked for no UMAP
    print(f"  Report: {report}")


# ===========================================================================
# Runner
# ===========================================================================

ALL_TESTS = [
    ("1: variance heuristic is unit-insensitive",   test_1_unit_insensitive_heuristic),
    ("2: column stats are unit-free",                test_2_column_stats_unit_free),
    ("3: transformer is column-order invariant",     test_3_transformer_order_invariant),
    ("4: masked diffusion training reduces loss",    test_4_masked_training_learns),
    ("5: inpainting respects observed columns",      test_5_inpaint_respects_observed),
    ("6: LoRA wrap + save/restore",                  test_6_lora_wrap_and_run),
    ("7: seed fine-tune handles subset columns",     test_7_seed_finetune_subset_columns),
    ("8: snapshot vs continuous format detection",   test_8_format_detection),
    ("9: MMD + domain classifier metrics",           test_9_evaluation_metrics),
    ("10: evaluate_synthetic full report",           test_10_evaluate_synthetic_full_report),
]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, default=None)
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
