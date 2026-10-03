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
        print(f"[WARNING] Engine {engine_id} only has {len(out)} rows.")
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
    source_shift_during = (edited.loc[edit_start:edit_end, source_col].mean() - baseline.loc[edit_start:edit_end, source_col].mean())
    target_shift_during = (edited.loc[edit_start:edit_end, target_col].mean() - baseline.loc[edit_start:edit_end, target_col].mean())
    post_start = edit_end + 1
    post_end = min(len(baseline) - 1, edit_end + post_horizon)
    target_shift_after = (edited.loc[post_start:post_end, target_col].mean() - baseline.loc[post_start:post_end, target_col].mean())

    print("\n" + "=" * 60)
    print("DIFFUSION PROPAGATION SUMMARY")
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
    elif abs(target_shift_after) > 0.05:
        print("PASS: target remains shifted after the edit window -> propagation visible.")
    else:
        print("WEAK RESULT: target does not remain strongly shifted after the edit window.")

def plot_result(baseline, edited, source_col, target_col, edit_start, edit_end, history_len, save_path):
    x = np.arange(len(baseline))
    diff = edited[target_col].values - baseline[target_col].values

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for ax in axes:
        ax.axvspan(edit_start, edit_end, alpha=0.20, label="Edit window")
        ax.axvspan(0, history_len, alpha=0.05, color="green", label="Input history")
        ax.axvline(history_len, linestyle="--", linewidth=1.5, color="black")
        ax.grid(True, alpha=0.3)

    axes[0].plot(x, baseline[source_col], label="Baseline", linewidth=2)
    axes[0].plot(x, edited[source_col], "--", label="Edited", linewidth=2)
    axes[0].set_ylabel(f"Source {source_col}")
    axes[0].set_title("Source Feature (Edited Variable)")
    axes[0].legend(loc="upper left")

    axes[1].plot(x, baseline[target_col], label="Baseline", linewidth=2)
    axes[1].plot(x, edited[target_col], "--", label="Edited", linewidth=2)
    axes[1].set_ylabel(f"Target {target_col}")
    axes[1].set_title("Target Feature Response")
    axes[1].legend(loc="upper left")

    axes[2].plot(x, diff, "--", linewidth=2, label="Edited − Baseline")
    axes[2].axhline(0.0, linestyle=":", linewidth=1.5)
    axes[2].set_ylabel("Difference")
    axes[2].set_xlabel("Cycle")
    axes[2].set_title("Diffusion Propagation Effect on Target")
    axes[2].legend(loc="upper left")

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    CKPT = PROJECT_ROOT / "src" / "model_checkPoints" / "diffusion" / "diff_best_ckpt.pt"
    DATA = PROJECT_ROOT / "data" / "FD001.csv"
    DEVICE = "cpu"

    # Tuning
    SOURCE_COL = "11"   # source feature to edit
    TARGET_COL = "4"    # target feature to observe for propagation
    ENGINE_ID = 1
    N_ROWS = 30         # rows of real data fed as input (observed region)
    N_OUT = 40          # total output rows (observed + generated) (REDUCED FOR SPEED)
    EDIT_START = 15
    EDIT_END = 30
    SHIFT_DELTA = -5.0

    print("Loading model...")
    # Add warm_start_strength equivalent
    
    print("Loading real data...")
    baseline_input = load_engine_slice(DATA, engine_id=ENGINE_ID, n_rows=N_ROWS)

    print("Creating edited input...")
    edited_input = apply_step_edit(
        baseline_input,
        col=SOURCE_COL,
        delta=SHIFT_DELTA,
        start=EDIT_START,
        end=EDIT_END,
    )

    print("Running baseline propagation (diffusion)...")
    out_baseline = propagation.apply_propagation(
        modified_data=baseline_input,
        model_type="diffusion",
        checkpoint_path=str(CKPT),
        num_samples=N_OUT,
        device=DEVICE,
        observed_rows=N_ROWS,
    )

    print("Running edited propagation (diffusion)...")
    out_edited = propagation.apply_propagation(
        modified_data=edited_input,
        model_type="diffusion",
        checkpoint_path=str(CKPT),
        num_samples=N_OUT,
        device=DEVICE,
        observed_rows=N_ROWS,
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

    plot_path = OUT_DIR / f"diffusion_propagation_{SOURCE_COL}_to_{TARGET_COL}.png"
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
    print(f"Saved plot to: {plot_path}")
