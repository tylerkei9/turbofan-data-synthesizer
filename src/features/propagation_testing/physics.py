import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------
# Paths
# -------------------------------------------------
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[3]
OUT_DIR = SCRIPT_PATH.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import propagation

# -------------------------------------------------
# Simple helpers
# -------------------------------------------------
def load_engine_slice(csv_path: Path, engine_id: int, n_rows: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    out = df[df.iloc[:, 0] == engine_id].copy().reset_index(drop=True)
    out.rename(columns={out.columns[0]: "Engine_ID"}, inplace=True)
    if len(out) < n_rows:
        print(f"[WARNING] Engine {engine_id} only has {len(out)} rows. Using all available rows.")
        n_rows = len(out)
    return out.iloc[:n_rows].copy().reset_index(drop=True)

def apply_step_edit(df: pd.DataFrame, col: str, delta: float, start: int, end: int) -> pd.DataFrame:
    out = df.copy()
    out.loc[start:end, col] = out.loc[start:end, col].astype(float) + float(delta)
    return out

def summarize_response(
    baseline: pd.DataFrame,
    edited: pd.DataFrame,
    source_col: str,
    target_col: str,
    edit_start: int,
    edit_end: int,
    post_horizon: int = 20,
):
    # During edit
    source_shift_during = (
        edited.loc[edit_start:edit_end, source_col].mean()
        - baseline.loc[edit_start:edit_end, source_col].mean()
    )
    target_shift_during = (
        edited.loc[edit_start:edit_end, target_col].mean()
        - baseline.loc[edit_start:edit_end, target_col].mean()
    )

    # After edit
    post_start = edit_end + 1
    post_end = min(len(baseline) - 1, edit_end + post_horizon)
    target_shift_after = (
        edited.loc[post_start:post_end, target_col].mean()
        - baseline.loc[post_start:post_end, target_col].mean()
    )

    print("\n" + "=" * 60)
    print("PROPAGATION SUMMARY")
    print("=" * 60)
    print(f"Source feature:   {source_col}")
    print(f"Target feature:   {target_col}")
    print(f"Edit window:      {edit_start} to {edit_end}")
    print(f"Source shift:     {source_shift_during:.4f}  (during edit)")
    print(f"Target shift:     {target_shift_during:.4f}  (during edit)")
    print(f"Target shift:     {target_shift_after:.4f}  (after edit)")
    print()

    if abs(source_shift_during) < 1e-6:
        print("WARNING: source did not change during the edit window.")
    elif abs(target_shift_after) > 1.0:
        print("PASS: target remains shifted after the edit window -> propagation visible.")
    else:
        print("WEAK RESULT: target does not remain strongly shifted after the edit window.")

def plot_result(
    baseline: pd.DataFrame,
    edited: pd.DataFrame,
    source_col: str,
    target_col: str,
    edit_start: int,
    edit_end: int,
    history_len: int,
    save_path: Path,
):
    x = np.arange(len(baseline))
    diff = edited[target_col].values - baseline[target_col].values

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    for ax in axes:
        # edit window
        ax.axvspan(edit_start, edit_end, alpha=0.20, label="Edit window")

        # history region
        ax.axvspan(0, history_len, alpha=0.05, color="green", label="Input history")

        # prediction start
        ax.axvline(history_len, linestyle="--", linewidth=1.5, color="black")

        ax.grid(True, alpha=0.3)

    # Source feature
    axes[0].plot(x, baseline[source_col], label="Baseline", linewidth=2)
    axes[0].plot(x, edited[source_col], "--", label="Edited", linewidth=2)
    axes[0].set_ylabel(f"Source {source_col}")
    axes[0].set_title("Source Feature (Edited Variable)")
    axes[0].legend(loc="upper left")

    # Target feature
    axes[1].plot(x, baseline[target_col], label="Baseline", linewidth=2)
    axes[1].plot(x, edited[target_col], "--", label="Edited", linewidth=2)
    axes[1].set_ylabel(f"Target {target_col}")
    axes[1].set_title("Target Feature Response")
    axes[1].legend(loc="upper left")

    # Difference
    axes[2].plot(x, diff, "--", linewidth=2, label="Edited − Baseline")
    axes[2].axhline(0.0, linestyle=":", linewidth=1.5)
    axes[2].set_ylabel("Difference")
    axes[2].set_xlabel("Cycle")
    axes[2].set_title("Propagation Effect on Target")
    axes[2].legend(loc="upper left")

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def summarize_multi_response(
    baseline: pd.DataFrame,
    edited: pd.DataFrame,
    source_col: str,
    target_cols: list,
    edit_start: int,
    edit_end: int,
    post_horizon: int = 20,
    label: str = "MULTI-TARGET PROPAGATION SUMMARY",
):
    print("\n" + "=" * 60)
    print(label)
    print("=" * 60)
    print(f"Source feature:   {source_col}")
    print(f"Edit window:      {edit_start} to {edit_end}")

    post_start = edit_end + 1
    post_end = min(len(baseline) - 1, edit_end + post_horizon)

    for target_col in target_cols:
        target_shift_during = (
            edited.loc[edit_start:edit_end, target_col].mean()
            - baseline.loc[edit_start:edit_end, target_col].mean()
        )
        target_shift_after = (
            edited.loc[post_start:post_end, target_col].mean()
            - baseline.loc[post_start:post_end, target_col].mean()
        )
        print(f"Target {target_col}: during = {target_shift_during:.4f}, after = {target_shift_after:.4f}")
    print()
def plot_multi_target_result(
    baseline: pd.DataFrame,
    edited_default: pd.DataFrame,
    edited_one_immune: pd.DataFrame,
    source_col: str,
    target_a: str,
    target_b: str,
    immune_target: str,
    edit_start: int,
    edit_end: int,
    history_len: int,
    save_path: Path,
):
    x = np.arange(len(baseline))
    diff_a_default = edited_default[target_a].values - baseline[target_a].values
    diff_b_default = edited_default[target_b].values - baseline[target_b].values
    diff_a_immune = edited_one_immune[target_a].values - baseline[target_a].values
    diff_b_immune = edited_one_immune[target_b].values - baseline[target_b].values

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    for ax in axes:
        ax.axvspan(edit_start, edit_end, alpha=0.20, label="Edit window")
        ax.axvspan(0, history_len, alpha=0.05, color="green", label="Input history")
        ax.axvline(history_len, linestyle="--", linewidth=1.5, color="black")
        ax.grid(True, alpha=0.3)

    axes[0].plot(x, baseline[source_col], label="Baseline", linewidth=2)
    axes[0].plot(x, edited_default[source_col], "--", label="Edited", linewidth=2)
    axes[0].set_ylabel(f"Source {source_col}")
    axes[0].set_title("Source Feature (Edited Variable)")
    axes[0].legend(loc="upper left")

    axes[1].plot(x, baseline[target_a], label="Baseline", linewidth=2)
    axes[1].plot(x, edited_default[target_a], "--", label="Default edited", linewidth=2)
    axes[1].plot(x, edited_one_immune[target_a], ":", label=f"Edited with {immune_target} immune", linewidth=2)
    axes[1].set_ylabel(f"Target {target_a}")
    axes[1].set_title(f"Primary target response ({target_a})")
    axes[1].legend(loc="upper left")

    axes[2].plot(x, baseline[target_b], label="Baseline", linewidth=2)
    axes[2].plot(x, edited_default[target_b], "--", label="Default edited", linewidth=2)
    axes[2].plot(x, edited_one_immune[target_b], ":", label=f"Edited with {immune_target} immune", linewidth=2)
    axes[2].set_ylabel(f"Target {target_b}")
    axes[2].set_title(f"Secondary target response ({target_b})")
    axes[2].legend(loc="upper left")

    axes[3].plot(x, diff_a_default, "--", linewidth=2, label=f"{target_a}: default")
    axes[3].plot(x, diff_b_default, "--", linewidth=2, label=f"{target_b}: default")
    axes[3].plot(x, diff_a_immune, ":", linewidth=2, label=f"{target_a}: {immune_target} immune")
    axes[3].plot(x, diff_b_immune, ":", linewidth=2, label=f"{target_b}: {immune_target} immune")
    axes[3].axhline(0.0, linestyle=":", linewidth=1.5)
    axes[3].set_ylabel("Difference")
    axes[3].set_xlabel("Cycle")
    axes[3].set_title("Propagation comparison")
    axes[3].legend(loc="upper left")

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    CKPT = PROJECT_ROOT / "src" / "model_checkPoints" / "transformer" / "transformer_model.pt"
    DATA = PROJECT_ROOT / "data" / "FD001.csv"
    DEVICE = "cpu"

    # Tuning
    SOURCE_COL = "11" # source feature to edit
    TARGET_COL = "3" # target feature to observe for propagation (should be different from source)
    ENGINE_ID = 1
    N_ROWS = 30 # determines rows fed into the model before it starts generating predictions
    N_OUT = 80  # number of future rows model generates after n_rows
    EDIT_START = 15 
    EDIT_END = 30
    SHIFT_DELTA = -5.0

    print("Loading model...")
    bundle = propagation.load_model(str(CKPT), "transformer", device=DEVICE)
    feature_cols = bundle["feature_cols"]

    print("Loading real data...")
    baseline_input = load_engine_slice(DATA, engine_id=ENGINE_ID, n_rows=N_ROWS)

    missing = [c for c in [SOURCE_COL, TARGET_COL] if c not in baseline_input.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Use the SAME normalization stats for both runs
    col_mins, ranges = propagation._compute_runtime_min_range(baseline_input, feature_cols)
    print("Creating edited input...")
    edited_input = apply_step_edit(
        baseline_input,
        col=SOURCE_COL,
        delta=SHIFT_DELTA,
        start=EDIT_START,
        end=EDIT_END,
    )

    print("Running baseline propagation...")
    out_baseline = propagation.apply_propagation(
        modified_data=baseline_input,
        model_type="transformer",
        checkpoint_path=str(CKPT),
        num_samples=N_OUT,
        device=DEVICE,
        col_mins=col_mins,
        ranges=ranges,
    )

    print("Running edited propagation...")
    out_edited = propagation.apply_propagation(
        modified_data=edited_input,
        model_type="transformer",
        checkpoint_path=str(CKPT),
        num_samples=N_OUT,
        device=DEVICE,
        col_mins=col_mins,
        ranges=ranges,
    )

    summarize_response(
        out_baseline,
        out_edited,
        source_col=SOURCE_COL,
        target_col=TARGET_COL,
        edit_start=EDIT_START,
        edit_end=EDIT_END,
        post_horizon=20,
    )

    plot_path = OUT_DIR / f"propagation_{SOURCE_COL}_to_{TARGET_COL}.png"
    plot_result(
        out_baseline,
        out_edited,
        source_col=SOURCE_COL,
        target_col=TARGET_COL,
        edit_start=EDIT_START,
        edit_end=EDIT_END,
        history_len=N_ROWS,
        save_path=plot_path,
    )
    print(f"\nSaved plot to: {plot_path}")

    additional_target_col = '7'

    print(f"Running multi-target default propagation check on targets: {TARGET_COL}, {additional_target_col} ...")
    summarize_multi_response(
        out_baseline,
        out_edited,
        source_col=SOURCE_COL,
        target_cols=[TARGET_COL, additional_target_col],
        edit_start=EDIT_START,
        edit_end=EDIT_END,
        post_horizon=20,
        label="MULTI-TARGET PROPAGATION SUMMARY (DEFAULT)",
    )

    immune_target_col = TARGET_COL
    print(f"Running multi-target propagation with immune target column: {immune_target_col} ...")
    out_edited_one_immune = propagation.apply_propagation(
        modified_data=edited_input,
        model_type="transformer",
        checkpoint_path=str(CKPT),
        num_samples=N_OUT,
        device=DEVICE,
        col_mins=col_mins,
        ranges=ranges,
        immune_cols=[immune_target_col],
        reference_data=baseline_input,
    )

    summarize_multi_response(
        out_baseline,
        out_edited_one_immune,
        source_col=SOURCE_COL,
        target_cols=[TARGET_COL, additional_target_col],
        edit_start=EDIT_START,
        edit_end=EDIT_END,
        post_horizon=20,
        label="MULTI-TARGET PROPAGATION SUMMARY (ONE TARGET IMMUNE)",
    )

    multi_plot_path = OUT_DIR / f"propagation_multi_{SOURCE_COL}_to_{TARGET_COL}_{additional_target_col}.png"
    plot_multi_target_result(
        out_baseline,
        out_edited,
        out_edited_one_immune,
        source_col=SOURCE_COL,
        target_a=TARGET_COL,
        target_b=additional_target_col,
        immune_target=immune_target_col,
        edit_start=EDIT_START,
        edit_end=EDIT_END,
        history_len=N_ROWS,
        save_path=multi_plot_path,
    )
    print(f"Saved multi-target plot to: {multi_plot_path}")

