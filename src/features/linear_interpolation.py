"""
linear_interpolation.py
──────────────────────────────────────────────────────────────────────────────
Linear interpolation for synthetic CMAPSS outputs produced by the
Transformer or Diffusion models.

Location : src/features/linear_interpolation.py

Directory contract
──────────────────
  Input  : outputs/transformer/synthetic_transformer.csv
           outputs/diffusion/synthetic_diffusion.csv
  Output : outputs/interpolated_data/
              <stem>_interpolated.csv   – filled data
              <stem>_interp_mask.csv   – 1 where a value was interpolated

How to run (from src/features/)
────────────────────────────────
  cd src/features

  # interpolate both model outputs
  python linear_interpolation.py

  # one model only
  python linear_interpolation.py --source transformer
  python linear_interpolation.py --source diffusion

  # fill only real NaNs in the file (skip gap simulation)
  python linear_interpolation.py --no-simulation

  # tune the gap limit (default 10 cycles)
  python linear_interpolation.py --max-gap 20
"""

import argparse
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Project-relative paths
# ─────────────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent.parent  # src/features/ -> src/ -> project root

SOURCE_FILES = {
    "transformer": ROOT / "outputs" / "transformer" / "synthetic_transformer.csv",
    "diffusion":   ROOT / "outputs" / "diffusion"   / "synthetic_diffusion.csv",
}

OUTPUT_DIR = ROOT / "outputs" / "interpolated_data"

# Column roles – identical to the CMAPSS training data schema
ID_COLS   = ["Engine_ID", "Time_in_cycles"]   # never interpolated
#TODO: CANNOT NOT DEPEND ON NAMES OF COLUMNS
SKIP_COLS: set[str] = set()                   # constant columns detected automatically


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    max_gap_cycles: int   = 10     # gaps wider than this are left as NaN
    simulate:       bool  = True   # inject artificial gaps to demo reconstruction
    drop_fraction:  float = 0.04   # fraction of rows to drop in simulation mode
    random_seed:    int   = 42


# ─────────────────────────────────────────────────────────────────────────────
# Gap detection
# ─────────────────────────────────────────────────────────────────────────────

def find_nan_runs(arr: np.ndarray) -> list[tuple[int, int]]:
    """Return (start, end) inclusive index pairs for each NaN run."""
    nan_mask = np.isnan(arr)
    if not nan_mask.any():
        return []
    padded = np.concatenate(([False], nan_mask, [False]))
    diff   = np.diff(padded.astype(np.int8))
    starts = np.where(diff ==  1)[0]
    ends   = np.where(diff == -1)[0] - 1
    return list(zip(starts.tolist(), ends.tolist()))


# ─────────────────────────────────────────────────────────────────────────────
# Per-engine linear interpolation
# ─────────────────────────────────────────────────────────────────────────────

def interpolate_engine(
    cycles:       np.ndarray,     # shape (N,)
    matrix:       np.ndarray,     # shape (N, C)
    sensor_names: list[str],
    cfg:          Config,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Fill NaN gaps in matrix via linear interpolation on the cycle axis.
    Gaps wider than cfg.max_gap_cycles are left as NaN.

    Returns filled matrix, bool mask (True = interpolated), stats dict.
    """
    N, C   = matrix.shape
    filled = matrix.copy()
    mask   = np.zeros((N, C), dtype=bool)
    stats  = {n: {"gaps": 0, "filled": 0, "skipped": 0} for n in sensor_names}

    for ci, name in enumerate(sensor_names):
        col  = filled[:, ci]
        gaps = find_nan_runs(col)
        stats[name]["gaps"] = len(gaps)

        for s, e in gaps:
            gap_span = int(cycles[e] - cycles[s])
            left_ok  = s > 0     and not np.isnan(col[s - 1])
            right_ok = e < N - 1 and not np.isnan(col[e + 1])
            bounded  = left_ok and right_ok

            if not bounded or gap_span > cfg.max_gap_cycles:
                stats[name]["skipped"] += (e - s + 1)
                continue

            t0, v0 = cycles[s - 1], col[s - 1]
            t1, v1 = cycles[e + 1], col[e + 1]
            alpha  = (cycles[s:e+1] - t0) / (t1 - t0)
            col[s:e+1]      = v0 + alpha * (v1 - v0)
            mask[s:e+1, ci] = True
            stats[name]["filled"] += (e - s + 1)

    return filled, mask, stats


# ─────────────────────────────────────────────────────────────────────────────
# Dataset-level processing
# ─────────────────────────────────────────────────────────────────────────────

def detect_constant_columns(df: pd.DataFrame, cols: list[str]) -> set[str]:
    return {c for c in cols if df[c].nunique() <= 1}


def process_file(csv_path: Path, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Load a CSV, interpolate all active sensor columns per engine, and return
    (filled_df, mask_df, active_cols).
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # ── validate schema ───────────────────────────────────────────────────────
    missing_id = [c for c in ID_COLS if c not in df.columns]
    if missing_id:
        raise ValueError(
            f"{csv_path.name}: missing expected columns {missing_id}.\n"
            f"  Found: {df.columns.tolist()}\n"
            f"  Check ID_COLS at the top of interpolate.py."
        )

    sensor_cols  = [c for c in df.columns if c not in ID_COLS]
    const_cols   = detect_constant_columns(df, sensor_cols) | SKIP_COLS
    active_cols  = [c for c in sensor_cols if c not in const_cols]

    print(f"\n  File       : {csv_path.name}")
    print(f"  Rows       : {len(df):,}    Engines: {df['Engine_ID'].nunique()}")
    print(f"  Active cols: {len(active_cols)}  ->  {active_cols}")
    if const_cols & set(sensor_cols):
        print(f"  Const/skip : {sorted(const_cols & set(sensor_cols))}")

    # ── optional gap simulation ───────────────────────────────────────────────
    original_df = df.copy()
    if cfg.simulate:
        rng      = np.random.default_rng(cfg.random_seed)
        drop_idx = rng.choice(len(df), size=int(len(df) * cfg.drop_fraction),
                              replace=False)
        df.loc[df.index[drop_idx], active_cols] = np.nan
        injected = df[active_cols].isna().sum().sum()
        print(f"  NaNs injected (simulation): {injected:,} "
              f"({injected / max(len(df) * len(active_cols), 1):.2%})")
    else:
        existing = df[active_cols].isna().sum().sum()
        print(f"  Existing NaNs to fill: {existing:,}")

    # ── per-engine interpolation ───────────────────────────────────────────────
    engines   = df["Engine_ID"].unique()
    filled_df = df.copy()
    mask_df   = pd.DataFrame(False, index=df.index, columns=df.columns)

    agg = {"gaps": 0, "filled": 0, "skipped": 0}
    t0  = time.perf_counter()

    for eid in engines:
        #TODO: REMOVE HARDCODING
        sel    = df["Engine_ID"] == eid
        edf    = df.loc[sel]
        cycles = edf["Time_in_cycles"].to_numpy(dtype=np.float64)
        matrix = edf[active_cols].to_numpy(dtype=np.float64)

        filled_mat, interp_mask, stats = interpolate_engine(
            cycles, matrix, active_cols, cfg
        )

        filled_df.loc[sel, active_cols] = filled_mat
        mask_df.loc[sel, active_cols]   = interp_mask

        for v in stats.values():
            agg["gaps"]    += v["gaps"]
            agg["filled"]  += v["filled"]
            agg["skipped"] += v["skipped"]

    elapsed = time.perf_counter() - t0

    remaining = filled_df[active_cols].isna().sum().sum()
    print(f"\n  Interpolation done in {elapsed*1000:.1f} ms")
    print(f"  Gaps detected : {agg['gaps']:,}")
    print(f"  Samples filled: {agg['filled']:,}")
    print(f"  Samples skipped (gap > {cfg.max_gap_cycles} cycles): {agg['skipped']:,}")
    print(f"  Remaining NaNs: {remaining:,}")

    # ── accuracy report (simulation mode only) ────────────────────────────────
    if cfg.simulate and mask_df[active_cols].any(axis=None):
        print(f"\n  {'Column':<20} {'MAE':>10} {'RMSE':>10} {'MaxErr':>10} {'N':>8}")
        print(f"  {'─'*52}")
        for col in active_cols:
            col_mask = mask_df[col]
            if not col_mask.any():
                continue
            orig  = original_df.loc[col_mask, col].to_numpy(dtype=np.float64)
            pred  = filled_df.loc[col_mask, col].to_numpy(dtype=np.float64)
            valid = ~(np.isnan(orig) | np.isnan(pred))
            if not valid.any():
                continue
            err  = np.abs(orig[valid] - pred[valid])
            rmse = np.sqrt((err ** 2).mean())
            print(f"  {col:<20} {err.mean():>10.4f} {rmse:>10.4f} "
                  f"{err.max():>10.4f} {valid.sum():>8,}")

    return filled_df, mask_df, active_cols


# ─────────────────────────────────────────────────────────────────────────────
# Save outputs
# ─────────────────────────────────────────────────────────────────────────────

def save_outputs(
    stem:        str,
    filled_df:   pd.DataFrame,
    mask_df:     pd.DataFrame,
    active_cols: list[str],
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    out_filled = OUTPUT_DIR / f"{stem}_interpolated.csv"
    out_mask   = OUTPUT_DIR / f"{stem}_interp_mask.csv"

    filled_df.to_csv(out_filled, index=False)

    mask_out = mask_df[active_cols].astype(int).copy()
    #TODO: NO HARDCODING
    mask_out.insert(0, "Time_in_cycles", filled_df["Time_in_cycles"])
    mask_out.insert(0, "Engine_ID",      filled_df["Engine_ID"])
    mask_out.to_csv(out_mask, index=False)

    print(f"\n  -> {out_filled.relative_to(ROOT)}")
    print(f"  -> {out_mask.relative_to(ROOT)}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def collect_files(source: Optional[str]) -> list[Path]:
    """Return the list of source files to process based on --source argument."""
    keys = [source] if source else list(SOURCE_FILES.keys())
    files = []
    for key in keys:
        path = SOURCE_FILES[key]
        if path.exists():
            files.append(path)
        else:
            print(f"  [skip] not found: {path.relative_to(ROOT)}")
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Interpolate synthetic CMAPSS outputs.")
    parser.add_argument("--source",        choices=["transformer", "diffusion"],
                        help="Process only one model's output.")
    parser.add_argument("--no-simulation", dest="simulate", action="store_false",
                        help="Fill real NaNs only; skip gap simulation.")
    parser.add_argument("--max-gap",       type=int, default=10,
                        help="Max gap size (cycles) to interpolate (default: 10).")
    parser.set_defaults(simulate=True)
    args = parser.parse_args()

    cfg = Config(
        max_gap_cycles = args.max_gap,
        simulate       = args.simulate,
    )

    csv_files = collect_files(args.source)

    if not csv_files:
        print("No source files found. Expected:")
        for k, p in SOURCE_FILES.items():
            print(f"  {p.relative_to(ROOT)}")
        return

    print(f"\nProcessing {len(csv_files)} file(s).")

    for csv_path in csv_files:
        print(f"\n{'='*60}")
        try:
            filled_df, mask_df, active_cols = process_file(csv_path, cfg)
            save_outputs(csv_path.stem, filled_df, mask_df, active_cols)
        except Exception as exc:
            print(f"  [ERROR] {csv_path.name}: {exc}")

    print(f"\n{'='*60}")
    print(f"All outputs written to: {OUTPUT_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()