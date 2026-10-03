import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 3. Standard Imports
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
from .. import propagation

def _make_baseline_input(feature_cols: list[str], n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        c: rng.uniform(0, 1, n).astype(np.float32)
        for c in feature_cols
    })
    df.insert(0, "Engine_ID", 1)
    return df


def _apply_shift_segment(df: pd.DataFrame, col: str, delta: float, start: int, end: int) -> pd.DataFrame:
    df2 = df.copy()
    df2.loc[start:end, col] = (df2.loc[start:end, col].astype(np.float32) + np.float32(delta))
    return df2


def prepare_transformer_input_frozen(data: pd.DataFrame, bundle: dict, frozen_mins: pd.Series, frozen_ranges: pd.Series):
    feature_cols = bundle["feature_cols"]
    available = [c for c in feature_cols if c in data.columns]
    col_mins = frozen_mins.reindex(available)
    ranges = frozen_ranges.reindex(available).replace(0.0, 1.0)
    raw_values = data[available].values.astype(np.float32)
    mins = col_mins.values.astype(np.float32)
    rngs = ranges.values.astype(np.float32)
    normalized = 2.0 * ((raw_values - mins) / rngs) - 1.0
    return {
        "normalized": normalized,
        "feature_cols": available,
        "col_mins": col_mins,
        "ranges": ranges,
        "norm_source": "frozen_baseline",
    }


def _compare_df_numeric(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    common_cols = [c for c in a.columns if c in b.columns]
    rows = []
    for c in common_cols:
        if not (pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c])):
            continue
        a_vals = a[c].to_numpy(np.float64)
        b_vals = b[c].to_numpy(np.float64)
        m = min(len(a_vals), len(b_vals))
        if m == 0:
            continue
        diff = b_vals[:m] - a_vals[:m]
        rows.append({
            "col": c,
            "mae": float(np.mean(np.abs(diff))),
            "rmse": float(np.sqrt(np.mean(diff * diff))),
            "max_abs": float(np.max(np.abs(diff))),
            "mean_diff": float(np.mean(diff)),
        })
    return pd.DataFrame(rows).sort_values("rmse", ascending=False).reset_index(drop=True)

def _col_array(df: pd.DataFrame, col: str) -> np.ndarray:
        return df[col].to_numpy(dtype=np.float64, copy=False)

def _wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    # 1D Earth mover distance approximation (exact in 1D via sorted CDFs)
    a = np.sort(a)
    b = np.sort(b)
    m = min(len(a), len(b))
    if m == 0:
        return float("nan")
    a = a[:m]
    b = b[:m]
    return float(np.mean(np.abs(a - b)))

def _js_divergence_hist(a: np.ndarray, b: np.ndarray, bins: int = 30) -> float:
    # Jensen-Shannon divergence on histogram probabilities (stable, bounded)
    if len(a) == 0 or len(b) == 0:
        return float("nan")

    lo = float(min(np.min(a), np.min(b)))
    hi = float(max(np.max(a), np.max(b)))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return 0.0
    ha, edges = np.histogram(a, bins=bins, range=(lo, hi), density=False)
    hb, _ = np.histogram(b, bins=bins, range=(lo, hi), density=False)
    pa = ha.astype(np.float64)
    pb = hb.astype(np.float64)
    pa = pa / max(pa.sum(), 1.0)
    pb = pb / max(pb.sum(), 1.0)
    m = 0.5 * (pa + pb)
    # KL with safe handling
    def kl(p, q):
        mask = p > 0
        return float(np.sum(p[mask] * np.log(p[mask] / np.maximum(q[mask], 1e-300))))
    js = 0.5 * kl(pa, m) + 0.5 * kl(pb, m)
    return float(js)

def distribution_report(out0: pd.DataFrame, out1: pd.DataFrame, cols: list[str], bins: int = 30) -> pd.DataFrame:
    rows = []
    for c in cols:
        if c not in out0.columns or c not in out1.columns:
            continue
        if not (pd.api.types.is_numeric_dtype(out0[c]) and pd.api.types.is_numeric_dtype(out1[c])):
            continue

        a = _col_array(out0, c)
        b = _col_array(out1, c)

        # align length only for simple comparability
        m = min(len(a), len(b))
        a = a[:m]
        b = b[:m]

        q = [0.05, 0.25, 0.50, 0.75, 0.95]
        qa = np.quantile(a, q) if m else np.full(len(q), np.nan)
        qb = np.quantile(b, q) if m else np.full(len(q), np.nan)

        rows.append({
            "col": c,
            "mean_A": float(np.mean(a)) if m else np.nan,
            "mean_B": float(np.mean(b)) if m else np.nan,
            "mean_shift(B-A)": float(np.mean(b - a)) if m else np.nan,
            "std_A": float(np.std(a)) if m else np.nan,
            "std_B": float(np.std(b)) if m else np.nan,
            "wasserstein_1d": _wasserstein_1d(a, b),
            "js_div_hist": _js_divergence_hist(a, b, bins=bins),
            "q50_A": float(qa[2]) if m else np.nan,
            "q50_B": float(qb[2]) if m else np.nan,
            "q95_A": float(qa[4]) if m else np.nan,
            "q95_B": float(qb[4]) if m else np.nan,
        })
    df = pd.DataFrame(rows)
    # sort by a robust distance metric first, then by mean shift magnitude
    if not df.empty:
        df = df.sort_values(["wasserstein_1d", "js_div_hist"], ascending=False).reset_index(drop=True)
    return df

def _ci95_halfwidth(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64)
        x = x[np.isfinite(x)]
        if x.size <= 1:
            return 0.0
        return 1.96 * float(np.std(x, ddof=1)) / float(np.sqrt(x.size))

def _linear_fit_r2(x: np.ndarray, y: np.ndarray):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2 or np.allclose(x, x[0]):
        return float("nan"), float("nan"), float("nan")
    a, b = np.polyfit(x, y, 1)
    yhat = a * x + b
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return float(a), float(b), float(r2)

def save_full_ab(out0, out1, out_dir: Path, prefix: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    out0.to_csv(out_dir / f"{prefix}_A.csv", index=False)
    out1.to_csv(out_dir / f"{prefix}_B.csv", index=False)

    m = min(len(out0), len(out1))
    A = out0.iloc[:m].reset_index(drop=True)
    B = out1.iloc[:m].reset_index(drop=True)

    diff = A.copy()
    for c in diff.columns:
        if c in B.columns and pd.api.types.is_numeric_dtype(A[c]) and pd.api.types.is_numeric_dtype(B[c]):
            diff[c] = B[c].to_numpy(np.float64) - A[c].to_numpy(np.float64)

    diff.to_csv(out_dir / f"{prefix}_difference.csv", index=False)

if __name__ == "__main__":
    TRANSFORMER_CKPT = str(PROJECT_ROOT / "outputs" / "transformer" / "transformer_model.pt")
    device = "cpu"

    n_input_rows = 120
    n_out = 120

    # Delta sweep (edit a fixed segment)
    edit_start, edit_end = 10, 25
    deltas = [-0.5, -0.2, 0.0, 0.2, 0.5]

    # Length sweep (edit varying-length segments centered at seg_center)
    length_delta = 0.20
    seg_lengths = [1, 8, 32, 64]
    seg_center = 60

    # Seeds
    seeds = [0, 1, 2, 3, 4]

    # Negative control segment (outside transformer input_window=64 => outside 0..63)
    # If propagation works, this should produce ~0 change vs baseline (within noise).
    neg_start, neg_end = 80, 87

    # ----------------------------
    # Load model ONCE to get feature columns
    # ----------------------------
    bundle = propagation.load_model(TRANSFORMER_CKPT, "transformer", device=device)
    feature_cols = bundle["feature_cols"]
    target_col = "Time_in_cycles" if "Time_in_cycles" in feature_cols else feature_cols[0]

    # ============================================================
    # A) Delta sweep (mean shift in target)
    # ============================================================
    delta_rows = []
    for seed in seeds:
        baseline = _make_baseline_input(feature_cols, n_input_rows, seed=seed)

        # Freeze normalization based on baseline (critical for A/B comparability)
        frozen_mins, frozen_ranges = propagation._compute_runtime_min_range(baseline, feature_cols)

        out0 = propagation.apply_propagation(
            modified_data=baseline,
            model_type="transformer",
            checkpoint_path=TRANSFORMER_CKPT,
            num_samples=n_out,
            device=device,
            config_path=None,
            col_mins=frozen_mins,
            ranges=frozen_ranges,
        )

        for d in deltas:
            edited = _apply_shift_segment(baseline, target_col, delta=d, start=edit_start, end=edit_end)

            out1 = propagation.apply_propagation(
                modified_data=edited,
                model_type="transformer",
                checkpoint_path=TRANSFORMER_CKPT,
                num_samples=n_out,
                device=device,
                config_path=None,
                col_mins=frozen_mins,
                ranges=frozen_ranges,
            )

            mean_shift = float("nan")
            if target_col in out0.columns and target_col in out1.columns:
                mean_shift = float(out1[target_col].mean() - out0[target_col].mean())

            delta_rows.append({"seed": seed, "delta": float(d), "mean_shift": mean_shift})

    delta_df = pd.DataFrame(delta_rows)
    delta_agg = (
        delta_df.groupby("delta", as_index=False)
        .agg(mean_shift_mean=("mean_shift", "mean"), mean_shift_std=("mean_shift", "std"))
        .sort_values("delta")
        .reset_index(drop=True)
    )
    delta_agg["mean_shift_ci95"] = [
        _ci95_halfwidth(delta_df.loc[delta_df["delta"] == d, "mean_shift"].to_numpy())
        for d in delta_agg["delta"].to_numpy()
    ]

    slope, intercept, r2 = _linear_fit_r2(
        delta_agg["delta"].to_numpy(),
        delta_agg["mean_shift_mean"].to_numpy(),
    )

    fig = plt.figure()
    plt.errorbar(
        delta_agg["delta"], delta_agg["mean_shift_mean"],
        yerr=delta_agg["mean_shift_ci95"],
        fmt="o-", capsize=3
    )
    plt.xlabel("delta (applied to seed segment)")
    plt.ylabel(f"mean shift in generated {target_col} (B - A)")
    plt.title(
        f"Delta sweep (segment=[{edit_start},{edit_end}], seeds={len(seeds)})\n"
        f"slope={slope:.4g}, intercept={intercept:.4g}, R²={r2:.4g}"
    )
    plt.grid(True)
    fig.savefig(OUT_DIR / "delta_sweep.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ============================================================
    # B) Segment-length sweep (top RMSE across numeric columns)
    # Includes a negative control segment outside the seed window.
    # ============================================================
    len_rows = []
    neg_rows = []

    for seed in seeds:
        baseline = _make_baseline_input(feature_cols, n_input_rows, seed=seed)

        frozen_mins, frozen_ranges = propagation._compute_runtime_min_range(baseline, feature_cols)

        out0 = propagation.apply_propagation(
            modified_data=baseline,
            model_type="transformer",
            checkpoint_path=TRANSFORMER_CKPT,
            num_samples=n_out,
            device=device,
            config_path=None,
            col_mins=frozen_mins,
            ranges=frozen_ranges,
        )

        # ---- Negative control: edit outside the input window
        edited_out = _apply_shift_segment(
            baseline, target_col, delta=length_delta, start=neg_start, end=neg_end
        )
        out_out = propagation.apply_propagation(
            modified_data=edited_out,
            model_type="transformer",
            checkpoint_path=TRANSFORMER_CKPT,
            num_samples=n_out,
            device=device,
            config_path=None,
            col_mins=frozen_mins,
            ranges=frozen_ranges,
        )
        summary_out = _compare_df_numeric(out0, out_out)
        top_rmse_out = float(summary_out["rmse"].iloc[0]) if not summary_out.empty else 0.0
        neg_rows.append({"seed": seed, "segment_len": int(neg_end - neg_start + 1), "top_rmse": top_rmse_out})

        # ---- Length sweep: edit around seg_center
        for L in seg_lengths:
            L = int(L)
            start = int(max(0, seg_center - (L // 2)))
            end = int(min(n_input_rows - 1, start + L - 1))

            edited = _apply_shift_segment(baseline, target_col, delta=length_delta, start=start, end=end)

            out1 = propagation.apply_propagation(
                modified_data=edited,
                model_type="transformer",
                checkpoint_path=TRANSFORMER_CKPT,
                num_samples=n_out,
                device=device,
                config_path=None,
                col_mins=frozen_mins,
                ranges=frozen_ranges,
            )

            summary = _compare_df_numeric(out0, out1)
            top_rmse = float(summary["rmse"].iloc[0]) if not summary.empty else 0.0
            len_rows.append({"seed": seed, "segment_len": int(end - start + 1), "top_rmse": top_rmse})

    len_df = pd.DataFrame(len_rows)
    len_agg = (
        len_df.groupby("segment_len", as_index=False)
        .agg(top_rmse_mean=("top_rmse", "mean"), top_rmse_std=("top_rmse", "std"))
        .sort_values("segment_len")
        .reset_index(drop=True)
    )
    len_agg["top_rmse_ci95"] = [
        _ci95_halfwidth(len_df.loc[len_df["segment_len"] == L, "top_rmse"].to_numpy())
        for L in len_agg["segment_len"].to_numpy()
    ]

    neg_df = pd.DataFrame(neg_rows)
    neg_agg = (
        neg_df.groupby("segment_len", as_index=False)
        .agg(top_rmse_mean=("top_rmse", "mean"), top_rmse_std=("top_rmse", "std"))
        .sort_values("segment_len")
        .reset_index(drop=True)
    )
    neg_agg["top_rmse_ci95"] = [
        _ci95_halfwidth(neg_df.loc[neg_df["segment_len"] == L, "top_rmse"].to_numpy())
        for L in neg_agg["segment_len"].to_numpy()
    ]

    # Plot length sweep
    fig = plt.figure()
    plt.errorbar(
        len_agg["segment_len"], len_agg["top_rmse_mean"],
        yerr=len_agg["top_rmse_ci95"],
        fmt="o-", capsize=3,
        label="Edited segment (centered)"
    )
    # Plot negative control as a single point/line (same x each time = segment_len of neg control)
    plt.errorbar(
        neg_agg["segment_len"], neg_agg["top_rmse_mean"],
        yerr=neg_agg["top_rmse_ci95"],
        fmt="o--", capsize=3,
        label="Negative control (outside window)"
    )
    plt.xlabel("edited segment length (rows)")
    plt.ylabel("top RMSE across numeric columns (B - A)")
    plt.title(f"Segment-length scaling (delta={length_delta}, seeds={len(seeds)})")
    plt.grid(True)
    plt.legend()
    fig.savefig(OUT_DIR / "length_sweep.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ============================================================
    # Save summary CSV
    # ============================================================
    summary_rows = []

    for _, r in delta_agg.iterrows():
        summary_rows.append({
            "kind": "delta_sweep_mean_shift",
            "x": float(r["delta"]),
            "y_mean": float(r["mean_shift_mean"]),
            "y_ci95": float(r["mean_shift_ci95"]),
            "notes": f"target={target_col}, segment=[{edit_start},{edit_end}]",
        })

    for _, r in len_agg.iterrows():
        summary_rows.append({
            "kind": "length_sweep_top_rmse",
            "x": float(r["segment_len"]),
            "y_mean": float(r["top_rmse_mean"]),
            "y_ci95": float(r["top_rmse_ci95"]),
            "notes": f"target={target_col}, delta={length_delta}, centered_at={seg_center}",
        })

    # Negative control summary
    for _, r in neg_agg.iterrows():
        summary_rows.append({
            "kind": "negative_control_outside_window_top_rmse",
            "x": float(r["segment_len"]),
            "y_mean": float(r["top_rmse_mean"]),
            "y_ci95": float(r["top_rmse_ci95"]),
            "notes": f"target={target_col}, delta={length_delta}, segment=[{neg_start},{neg_end}]",
        })

    summary_rows.append({
        "kind": "delta_fit",
        "x": float("nan"),
        "y_mean": float("nan"),
        "y_ci95": float("nan"),
        "notes": f"slope={slope:.6g}, intercept={intercept:.6g}, R2={r2:.6g}",
    })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "propagation_summary.csv", index=False)

    print("Saved:")
    print(f"  {OUT_DIR / 'propagation_summary.csv'}")
    print(f"  {OUT_DIR / 'delta_sweep.png'}")
    print(f"  {OUT_DIR / 'length_sweep.png'}")
