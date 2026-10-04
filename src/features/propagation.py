import sys
import math
import pandas as pd
import numpy as np
import torch
from typing import Literal, Optional
from pathlib import Path
import yaml

# -------------------------
# Paths / Defaults
# -------------------------
SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent                # .../src/features
PROJECT_ROOT = SCRIPT_DIR.parents[1]           # repo root

# Add project root to sys.path so training modules can be imported
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# -------------------------
# Column layout (matches FD001.csv)
# -------------------------
ENGINE_ID_COL = 0
TIME_COL = 1
ALTITUDE_COL = 2
MACH_COL = 3
TRA_COL = 4
SENSOR_START_COL = 5  # Sensors 1-21 start at column 5

# Default checkpoint locations
DEFAULT_DIFFUSION_CKPT = PROJECT_ROOT / "src" / "model_checkPoints" / "diffusion" / "diff_best_ckpt.pt"
DEFAULT_TRANSFORMER_CKPT = PROJECT_ROOT / "src" / "model_checkPoints" / "transformer" / "transformer_model.pt"

def _resolve_path(p):
    """Resolve a path relative to PROJECT_ROOT if not absolute."""
    p = Path(p)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def load_config(cfg_path):
    """Load a YAML config file."""
    cfg_path = _resolve_path(cfg_path)
    if not cfg_path.exists():
        raise FileNotFoundError("Config not found: {}".format(cfg_path))
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _to_series_or_none(x) -> Optional[pd.Series]:
    """Convert dict/list/Series to Series; return None if x is None."""
    if x is None:
        return None
    if isinstance(x, pd.Series):
        return x.copy()
    if isinstance(x, dict):
        return pd.Series(x)
    if isinstance(x, (list, tuple, np.ndarray)):
        return pd.Series(list(x))
    raise TypeError(f"Unsupported stats type: {type(x)}")


# =============================================================================
# Model loading
# =============================================================================

def load_model(checkpoint_path: str, model_type: str, device: str = "cpu"):
    """
    Load a trained model from checkpoint.

    Returns a dictionary containing:
      - "model": the loaded torch.nn.Module in eval mode
      - Plus model-type-specific metadata needed for inference
    """
    ckpt_path = _resolve_path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device_obj = torch.device(device)
    ckpt = torch.load(ckpt_path, map_location=device_obj, weights_only=False)

    if model_type == "diffusion":
        return _load_diffusion_model(ckpt, device_obj)
    elif model_type == "transformer":
        return _load_transformer_model(ckpt, device_obj)
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}. Use 'diffusion' or 'transformer'.")



def _load_diffusion_model(ckpt: dict, device):
    """Load a diffusion (ConditionalDenoiser) model from checkpoint."""
    from src.training.Diffusion.diffusion_model5 import (
        ConditionalDenoiser,
        DiffusionSchedule,
    )

    sched_data = ckpt["schedule"]
    T = int(sched_data["T"])
    schedule = DiffusionSchedule(T=T, device=device)
    saved_betas = torch.tensor(sched_data["betas"], dtype=torch.float32, device=device)
    schedule.betas = saved_betas
    schedule.alphas = 1.0 - saved_betas
    schedule.alpha_bar = torch.cumprod(schedule.alphas, dim=0)
    schedule.sqrt_alpha_bar = torch.sqrt(schedule.alpha_bar)
    schedule.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - schedule.alpha_bar)

    hidden_dim = 256
    if "model_state" in ckpt and "net.0.weight" in ckpt["model_state"]:
        hidden_dim = ckpt["model_state"]["net.0.weight"].shape[0]

    model = ConditionalDenoiser(
        sensor_dim=21,
        condition_dim=3,
        hidden_dim=hidden_dim,
        time_emb_dim=128,
        n_layers=4,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    return {
        "model": model,
        "schedule": schedule,
        "condition_stats": ckpt["condition_stats"],
        "sensor_stats": ckpt["sensor_stats"],
        "columns": ckpt.get("columns", None),
        "device": device,
    }



def _load_transformer_model(ckpt: dict, device):
    """
    Load a transformer (StrongTimeTransformer) model from checkpoint.

    Supports V7 checkpoints where model input is [values | masks], i.e. src_dim = 2 * F.
    """
    from src.training.Transformer.Transformer_TrainingV7 import StrongTimeTransformer

    feature_cols = ckpt.get("feature_cols") or ckpt.get("features")
    if feature_cols is None:
        raise KeyError("Transformer checkpoint missing 'feature_cols' (or 'features').")

    state_dict = ckpt.get("model_state_dict") or ckpt.get("state_dict")
    if state_dict is None:
        raise KeyError("Transformer checkpoint missing 'model_state_dict' (or 'state_dict').")

    cfg = ckpt.get("config", {}) or {}
    runtime = cfg.get("runtime", {}) if isinstance(cfg, dict) else {}

    eng_ids = ckpt.get("eng_ids", None)
    eng_to_idx = ckpt.get("eng_to_idx", None)

    if eng_ids is None:
        emb_w = state_dict["eng_emb.weight"]
        n_engines_vocab = int(emb_w.shape[0])
        eng_ids = list(range(n_engines_vocab))
        eng_to_idx = {e: i for i, e in enumerate(eng_ids)}
    else:
        if eng_to_idx is None:
            eng_to_idx = {e: i for i, e in enumerate(eng_ids)}
        n_engines_vocab = len(eng_ids)

    d_model = int(runtime.get("d_model", 256))
    nhead = int(runtime.get("nhead", 8))
    num_layers = int(runtime.get("num_layers", 4))
    d_ff = int(runtime.get("d_ff", 512))
    d_eng = int(runtime.get("d_eng", 4))
    dropout = float(runtime.get("dropout", 0.15))

    input_window = int(ckpt.get("INPUT_WINDOW", runtime.get("input_window", 64)))
    F = int(ckpt.get("out_dim", len(feature_cols)))
    src_dim = int(ckpt.get("src_dim", 2 * F))

    model = StrongTimeTransformer(
        src_dim=src_dim,
        out_dim=F,
        n_engines=n_engines_vocab,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        d_ff=d_ff,
        d_eng=d_eng,
        dropout=dropout,
    ).to(device)

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    col_mins = _to_series_or_none(ckpt.get("col_mins"))
    ranges = _to_series_or_none(ckpt.get("ranges"))

    return {
        "model": model,
        "feature_cols": feature_cols,
        "col_mins": col_mins,
        "ranges": ranges,
        "eng_ids": eng_ids,
        "eng_to_idx": eng_to_idx,
        "input_window": input_window,
        "src_dim": src_dim,
        "out_dim": F,
        "device": device,
    }

# =============================================================================
# Data preparation
# =============================================================================
def prepare_data_for_inference(data: pd.DataFrame, model_type: str, model_bundle: dict):
    if model_type == "diffusion":
        return _prepare_diffusion_input(data, model_bundle)
    elif model_type == "transformer":
        return _prepare_transformer_input(data, model_bundle)
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")



def _prepare_diffusion_input(data: pd.DataFrame, bundle: dict):
    cond_stats = bundle["condition_stats"]
    sensor_stats = bundle["sensor_stats"]

    all_cols = data.columns.tolist()

    cond_name_candidates = {"altitude": None, "mach": None, "tra": None}
    for col in all_cols:
        col_lower = col.strip().lower()
        if "altitude" in col_lower:
            cond_name_candidates["altitude"] = col
        elif "mach" in col_lower:
            cond_name_candidates["mach"] = col
        elif col_lower == "tra":
            cond_name_candidates["tra"] = col

    if all(v is not None for v in cond_name_candidates.values()):
        cond_cols = [
            cond_name_candidates["altitude"],
            cond_name_candidates["mach"],
            cond_name_candidates["tra"],
        ]
    else:
        cond_cols = [all_cols[ALTITUDE_COL], all_cols[MACH_COL], all_cols[TRA_COL]]

    conditions_raw = data[cond_cols].values.astype(np.float32)

    cond_mean = np.array(cond_stats["mean"], dtype=np.float32)
    cond_std = np.array(cond_stats["std"], dtype=np.float32)
    conditions_norm = (conditions_raw - cond_mean) / cond_std

    # Also extract and normalize sensor columns for warm-start propagation
    sensor_mean = np.array(sensor_stats["mean"], dtype=np.float32)
    sensor_std = np.array(sensor_stats["std"], dtype=np.float32)

    columns_stored = bundle.get("columns", None)
    if columns_stored and len(columns_stored) >= SENSOR_START_COL + 1:
        sensor_col_names = [c.strip() for c in columns_stored[SENSOR_START_COL:]]
    else:
        sensor_col_names = [str(i) for i in range(1, 22)]

    available_sensor_cols = [c for c in sensor_col_names if c in data.columns]
    if len(available_sensor_cols) == len(sensor_col_names):
        sensors_raw = data[available_sensor_cols].values.astype(np.float32)
        sensors_norm = (sensors_raw - sensor_mean) / sensor_std
    else:
        sensors_raw = None
        sensors_norm = None

    return {
        "conditions_norm": conditions_norm,  # (N, 3)
        "cond_stats": cond_stats,
        "sensor_stats": sensor_stats,
        "sensors_norm": sensors_norm,          # (N, 21) or None, for propagation
        "sensor_col_names": sensor_col_names,
        "conditions_raw": conditions_raw,      # (N, 3) un-normalized
        "input_data": data,
    }



def _compute_runtime_min_range(df: pd.DataFrame, cols: list[str]) -> tuple[pd.Series, pd.Series]:
    """
    Compute mins/ranges from runtime data for the specified columns.
    Range uses (max-min); zeros replaced with 1 to avoid division by zero.
    """
    mins = df[cols].min(numeric_only=True)
    maxs = df[cols].max(numeric_only=True)
    rngs = (maxs - mins).replace(0.0, 1.0)
    return mins, rngs



def _prepare_transformer_input(data: pd.DataFrame, bundle: dict):
    """
    For transformer, normalize feature columns to [-1, 1] and build observation masks.

    V7 training uses concatenated [values | masks] inputs, but predicts values only.
    """
    feature_cols = bundle["feature_cols"]

    available = [c for c in feature_cols if c in data.columns]
    if not available:
        raise ValueError(
            f"No matching feature columns found.\n"
            f"Model expects (first 10): {feature_cols[:10]}\n"
            f"Data has (first 10): {list(data.columns)[:10]}"
        )

    col_mins = bundle.get("col_mins", None)
    ranges = bundle.get("ranges", None)

    if col_mins is None or ranges is None:
        col_mins_rt, ranges_rt = _compute_runtime_min_range(data, available)
        col_mins = col_mins_rt
        ranges = ranges_rt
        norm_source = "runtime_data"
    else:
        col_mins = col_mins.reindex(available)
        ranges = ranges.reindex(available).replace(0.0, 1.0)
        norm_source = "checkpoint"

    raw_df = data[available].apply(pd.to_numeric, errors="coerce")
    masks = raw_df.notna().astype(np.float32).values.astype(np.float32)

    raw_values = raw_df.fillna(0.0).values.astype(np.float32)
    mins = col_mins.values.astype(np.float32)
    rngs = ranges.values.astype(np.float32)

    normalized = 2.0 * ((raw_values - mins) / rngs) - 1.0

    return {
        "normalized": normalized,       # (N, F) normalized values
        "masks": masks,                 # (N, F) observation masks
        "feature_cols": available,
        "col_mins": col_mins,
        "ranges": ranges,
        "norm_source": norm_source,
    }

# =============================================================================
# Generation
# =============================================================================
@torch.no_grad()
def generate_with_model(
    model_bundle: dict,
    prepared_data: dict,
    num_samples: int,
    model_type: str,
    observed_rows: Optional[int] = None,
    engine_id: Optional[int] = None,
    immune_cols: Optional[list[str]] = None,
    prepared_reference_data: Optional[dict] = None,
):
    if model_type == "diffusion":
        return _generate_diffusion(
            model_bundle, prepared_data, num_samples,
            observed_rows=observed_rows,
            immune_cols=immune_cols,
            prepared_reference_data=prepared_reference_data,
        )
    elif model_type == "transformer":
        return _generate_transformer(
            model_bundle,
            prepared_data,
            num_samples,
            observed_rows=observed_rows,
            engine_id=engine_id,
            immune_cols=immune_cols,
            prepared_reference_data=prepared_reference_data,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")



def _generate_diffusion(
    bundle: dict,
    prepared: dict,
    num_samples: int,
    observed_rows: Optional[int] = None,
    immune_cols: Optional[list[str]] = None,
    prepared_reference_data: Optional[dict] = None,
    warm_start_strength: float = 0.6,
):
    """
    Diffusion generation.

    When observed_rows is provided (propagation mode):
      - Rows [0 : observed_rows-1] are copied from the input data.
      - Rows [observed_rows : num_samples-1] are generated one-by-one
        using warm-started diffusion, creating temporal coupling.

    When observed_rows is None (legacy mode):
      - All rows are generated independently from conditions only.
    """
    from src.training.Diffusion.diffusion_model5 import (
        sample_conditional,
        sample_conditional_warm_start,
    )

    model = bundle["model"]
    schedule = bundle["schedule"]
    device = bundle.get("device", "cpu")
    columns = bundle.get("columns", None)
    sensor_stats = prepared["sensor_stats"]
    cond_stats = prepared["cond_stats"]
    conditions_norm = prepared["conditions_norm"]  # (N, 3)
    sensors_norm = prepared.get("sensors_norm", None)  # (N, 21) or None
    sensor_col_names = prepared.get("sensor_col_names", None)
    conditions_raw = prepared.get("conditions_raw", None)

    sensor_mean = np.array(sensor_stats["mean"], dtype=np.float32)
    sensor_std = np.array(sensor_stats["std"], dtype=np.float32)
    cond_mean = np.array(cond_stats["mean"], dtype=np.float32)
    cond_std = np.array(cond_stats["std"], dtype=np.float32)

    # Resolve column names
    if columns and len(columns) >= SENSOR_START_COL + 1:
        cond_col_names = [c.strip() for c in columns[ALTITUDE_COL: ALTITUDE_COL + 3]]
        if sensor_col_names is None:
            sensor_col_names = [c.strip() for c in columns[SENSOR_START_COL:]]
    else:
        cond_col_names = ["Altitude", "Mach_number", "TRA"]
        if sensor_col_names is None:
            sensor_col_names = [str(i) for i in range(1, 22)]

    n_available = conditions_norm.shape[0]

    # ----------------------------------------------------------------
    # Legacy mode (no observed_rows): original non-autoregressive path
    # ----------------------------------------------------------------
    if observed_rows is None:
        if num_samples <= n_available:
            cond_subset = conditions_norm[:num_samples]
        else:
            indices = np.random.choice(n_available, size=num_samples, replace=True)
            cond_subset = conditions_norm[indices]

        cond_tensor = torch.from_numpy(cond_subset).float()
        gen_sensors_norm = sample_conditional(
            model, schedule, cond_tensor, device=device, verbose=False
        )
        sensors = gen_sensors_norm * sensor_std + sensor_mean
        conditions_denorm = cond_subset * cond_std + cond_mean

        engine_ids = np.arange(1, num_samples + 1, dtype=np.float32)
        time_cycles = np.ones(num_samples, dtype=np.float32)

        result = pd.DataFrame()
        result["Engine_ID"] = engine_ids
        result["Time_in_cycles"] = time_cycles
        for i, name in enumerate(cond_col_names):
            result[name] = conditions_denorm[:, i]
        for i, name in enumerate(sensor_col_names):
            result[name] = sensors[:, i]
        return result

    # ----------------------------------------------------------------
    # Propagation mode: autoregressive with warm-start
    # ----------------------------------------------------------------
    observed_rows = min(observed_rows, n_available)

    if sensors_norm is None:
        raise ValueError(
            "Propagation mode requires sensor data in the input. "
            "Make sure the input DataFrame has sensor columns."
        )

    # Resolve immune columns
    immune_cols = immune_cols or []
    immune_idx = []
    if immune_cols:
        missing = [c for c in immune_cols if c not in sensor_col_names]
        if missing:
            raise ValueError(f"Immune columns not found in sensor columns: {missing}")
        if prepared_reference_data is None:
            raise ValueError(
                "immune_cols requires reference_data so immune columns "
                "have an unedited reference stream."
            )
        immune_idx = [sensor_col_names.index(c) for c in immune_cols]

    rows = []

    # 1. Copy observed region from input data
    for t in range(observed_rows):
        row = {
            "Engine_ID": float(
                prepared["input_data"].iloc[t].get("Engine_ID", t + 1)
            ),
            "cycle": int(t + 1),
            "region": "observed",
        }
        for i, name in enumerate(cond_col_names):
            row[name] = float(conditions_raw[t, i]) if conditions_raw is not None else 0.0
        for i, name in enumerate(sensor_col_names):
            row[name] = float(sensors_norm[t, i] * sensor_std[i] + sensor_mean[i])
        rows.append(row)

    # 2. Generate future rows autoregressively
    prev_sensor_norm = sensors_norm[observed_rows - 1 : observed_rows]  # (1, 21)

    ref_prev_sensor_norm = None
    if immune_idx and prepared_reference_data is not None:
        ref_sensors_norm = prepared_reference_data.get("sensors_norm", None)
        if ref_sensors_norm is not None:
            ref_obs = min(observed_rows, ref_sensors_norm.shape[0])
            ref_prev_sensor_norm = ref_sensors_norm[ref_obs - 1 : ref_obs]

    for t in range(observed_rows, num_samples):
        # Pick condition for this row
        if t < n_available:
            cond_row = conditions_norm[t : t + 1]
            cond_raw_row = conditions_raw[t] if conditions_raw is not None else None
        else:
            idx = min(t, n_available - 1)
            cond_row = conditions_norm[idx : idx + 1]
            cond_raw_row = conditions_raw[idx] if conditions_raw is not None else None

        cond_tensor = torch.from_numpy(cond_row).float()
        prev_tensor = torch.from_numpy(prev_sensor_norm).float()

        gen_norm = sample_conditional_warm_start(
            model, schedule, cond_tensor, prev_tensor,
            t_start_frac=warm_start_strength, device=device, verbose=False,
        )  # (1, 21)

        # Handle immune columns via reference stream
        if immune_idx and ref_prev_sensor_norm is not None:
            ref_cond_norm = prepared_reference_data["conditions_norm"]
            if t < ref_cond_norm.shape[0]:
                ref_cond_row = ref_cond_norm[t : t + 1]
            else:
                ref_cond_row = ref_cond_norm[-1:]
            ref_cond_tensor = torch.from_numpy(ref_cond_row).float()
            ref_prev_tensor = torch.from_numpy(ref_prev_sensor_norm).float()

            ref_gen_norm = sample_conditional_warm_start(
                model, schedule, ref_cond_tensor, ref_prev_tensor,
                t_start_frac=warm_start_strength, device=device, verbose=False,
            )
            gen_norm[0, immune_idx] = ref_gen_norm[0, immune_idx]
            ref_prev_sensor_norm = ref_gen_norm

        gen_denorm = gen_norm[0] * sensor_std + sensor_mean

        row = {
            "Engine_ID": float(
                prepared["input_data"].iloc[0].get("Engine_ID", 1)
            ),
            "cycle": int(t + 1),
            "region": "generated",
        }
        if cond_raw_row is not None:
            for i, name in enumerate(cond_col_names):
                row[name] = float(cond_raw_row[i])
        for i, name in enumerate(sensor_col_names):
            row[name] = float(gen_denorm[i])
        rows.append(row)

        prev_sensor_norm = gen_norm  # advance seed

    return pd.DataFrame(rows).reset_index(drop=True)



def _build_initial_window(
    normalized: np.ndarray,
    masks: np.ndarray,
    observed_rows: int,
    input_window: int,
    F: int,
) -> torch.Tensor:
    history_vals = normalized[:observed_rows]
    history_masks = masks[:observed_rows]

    if observed_rows >= input_window:
        vals_np = history_vals[-input_window:].copy()
        masks_np = history_masks[-input_window:].copy()
    else:
        pad_len = input_window - observed_rows
        vals_pad = np.zeros((pad_len, F), dtype=np.float32)
        masks_pad = np.zeros((pad_len, F), dtype=np.float32)
        vals_np = np.concatenate([vals_pad, history_vals], axis=0)
        masks_np = np.concatenate([masks_pad, history_masks], axis=0)

    window_np = np.concatenate([vals_np, masks_np], axis=1)
    return torch.tensor(window_np, dtype=torch.float32)


def _generate_transformer(
    bundle: dict,
    prepared: dict,
    num_samples: int,
    observed_rows: Optional[int] = None,
    engine_id: Optional[int] = None,
    immune_cols: Optional[list[str]] = None,
    prepared_reference_data: Optional[dict] = None,
):
    """
    - Rows [0 : observed_rows-1] are copied from the provided input data.
    - Rows [observed_rows : num_samples-1] are generated autoregressively.
    - The model only propagates edits into the FUTURE region.

    If immune_cols and prepared_reference_data are provided, the listed columns are
    generated from the reference stream instead of the edited stream. This makes
    those columns immune to propagation from the edited inputs while leaving the
    default behavior unchanged for all existing callers.
    """

    model = bundle["model"]
    device = bundle["device"]
    eng_ids = bundle["eng_ids"]
    eng_to_idx = bundle["eng_to_idx"]
    input_window = bundle["input_window"]

    feature_cols = prepared["feature_cols"]
    normalized = prepared["normalized"]  # shape: (N_obs, F)
    masks = prepared["masks"]            # shape: (N_obs, F)
    F = len(feature_cols)

    if observed_rows is None:
        observed_rows = normalized.shape[0]
    observed_rows = min(observed_rows, normalized.shape[0])

    if num_samples < observed_rows:
        raise ValueError(
            f"num_samples={num_samples} must be >= observed_rows={observed_rows}"
        )

    # Use one engine only for propagation testing
    if engine_id is None:
        engine_id = eng_ids[0]

    if engine_id not in eng_to_idx:
        raise ValueError(f"engine_id {engine_id} not found in transformer engine vocabulary.")

    eng_idx = torch.tensor([eng_to_idx[engine_id]], dtype=torch.long, device=device)

    col_mins = prepared["col_mins"].reindex(feature_cols).values.astype(np.float32)
    ranges = prepared["ranges"].reindex(feature_cols).replace(0.0, 1.0).values.astype(np.float32)

    def denorm_row(x_norm: np.ndarray) -> np.ndarray:
        return ((x_norm + 1.0) * 0.5) * ranges + col_mins

    immune_cols = immune_cols or []
    immune_idx = []
    if immune_cols:
        missing = [c for c in immune_cols if c not in feature_cols]
        if missing:
            raise ValueError(f"Immune columns not found in transformer feature columns: {missing}")
        if prepared_reference_data is None:
            raise ValueError("immune_cols requires reference_data so immune columns have an unedited reference stream.")

        ref_feature_cols = prepared_reference_data["feature_cols"]
        if list(ref_feature_cols) != list(feature_cols):
            raise ValueError("reference_data feature columns do not match modified_data feature columns.")

        immune_idx = [feature_cols.index(c) for c in immune_cols]

    rows = []

    # -------------------------------------------------
    # 1. Copy observed region directly from input
    # -------------------------------------------------
    for t in range(observed_rows):
        denorm = denorm_row(normalized[t])
        row = {
            "Engine_ID": engine_id,
            "cycle": int(t + 1),
            "region": "observed",
            "norm_source": prepared.get("norm_source", "unknown"),
        }
        row.update({c: float(v) for c, v in zip(feature_cols, denorm)})
        rows.append(row)

    # -------------------------------------------------
    # 2. Build initial rolling windows
    # -------------------------------------------------
    window = _build_initial_window(normalized, masks, observed_rows, input_window, F).to(device).unsqueeze(0)

    reference_window = None
    if immune_idx:
        reference_normalized = prepared_reference_data["normalized"]
        reference_masks = prepared_reference_data["masks"]
        ref_observed_rows = min(observed_rows, reference_normalized.shape[0])
        reference_window = _build_initial_window(reference_normalized, reference_masks, ref_observed_rows, input_window, F).to(device).unsqueeze(0)

    # -------------------------------------------------
    # 3. Generate future autoregressively
    # -------------------------------------------------
    for t in range(observed_rows, num_samples):
        pred = model(window, eng_idx)  # shape: (1, F)
        pred_np = pred.squeeze(0).detach().cpu().numpy().astype(np.float32)

        if immune_idx:
            ref_pred = model(reference_window, eng_idx)
            ref_pred_np = ref_pred.squeeze(0).detach().cpu().numpy().astype(np.float32)
            pred_np[immune_idx] = ref_pred_np[immune_idx]

            ref_next_vals = torch.tensor(ref_pred_np, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(1)
            ref_next_masks = torch.ones_like(ref_next_vals)
            ref_next_step = torch.cat([ref_next_vals, ref_next_masks], dim=-1)
            reference_window = torch.cat([reference_window[:, 1:, :], ref_next_step], dim=1)

        denorm = denorm_row(pred_np)
        row = {
            "Engine_ID": engine_id,
            "cycle": int(t + 1),
            "region": "generated",
            "norm_source": prepared.get("norm_source", "unknown"),
        }
        row.update({c: float(v) for c, v in zip(feature_cols, denorm)})
        rows.append(row)

        next_vals = torch.tensor(pred_np, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(1)
        next_masks = torch.ones_like(next_vals)
        next_step = torch.cat([next_vals, next_masks], dim=-1)
        window = torch.cat([window[:, 1:, :], next_step], dim=1)

    return pd.DataFrame(rows).reset_index(drop=True)



def _load_transformer_norm_from_config(config_path: Optional[str]) -> tuple[Optional[pd.Series], Optional[pd.Series]]:
    if not config_path:
        return None, None

    cfg = load_config(config_path)
    if not isinstance(cfg, dict):
        return None, None

    norm_cfg = cfg.get("transformer_norm", {})
    if not isinstance(norm_cfg, dict) or not norm_cfg:
        return None, None

    # Inline
    if "col_mins" in norm_cfg and "ranges" in norm_cfg:
        return _to_series_or_none(norm_cfg.get("col_mins")), _to_series_or_none(norm_cfg.get("ranges"))

    # CSV file form
    def _read_series_csv(p: str) -> pd.Series:
        p = _resolve_path(p)
        df = pd.read_csv(p)
        if "feature" in df.columns and "value" in df.columns:
            return pd.Series(df["value"].values, index=df["feature"].values)
        return pd.Series(df.iloc[:, 1].values, index=df.iloc[:, 0].values)

    col_mins = _read_series_csv(norm_cfg["col_mins_path"]) if "col_mins_path" in norm_cfg else None
    ranges = _read_series_csv(norm_cfg["ranges_path"]) if "ranges_path" in norm_cfg else None
    return col_mins, ranges


# =============================================================================
# Main public API
# =============================================================================
def apply_propagation(
    modified_data: pd.DataFrame,
    model_type: Literal["diffusion", "transformer"],
    checkpoint_path: str,
    num_samples: Optional[int] = None,
    device: str = "cuda",
    config_path: Optional[str] = None,
    col_mins: Optional[pd.Series] = None,
    ranges: Optional[pd.Series] = None,
    observed_rows: Optional[int] = None,
    engine_id: Optional[int] = None,
    immune_cols: Optional[list[str]] = None,
    reference_data: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:

    if checkpoint_path is None:
        if model_type == "transformer":
            checkpoint_path = DEFAULT_TRANSFORMER_CKPT
        else:
            checkpoint_path = DEFAULT_DIFFUSION_CKPT
    if num_samples is None:
        num_samples = len(modified_data)

    model_bundle = load_model(checkpoint_path, model_type, device)

    if model_type == "transformer":
        if col_mins is not None and ranges is not None:
            model_bundle["col_mins"] = col_mins
            model_bundle["ranges"] = ranges
        else:
            cfg_mins, cfg_ranges = _load_transformer_norm_from_config(config_path)
            if cfg_mins is not None and cfg_ranges is not None:
                model_bundle["col_mins"] = cfg_mins
                model_bundle["ranges"] = cfg_ranges

    prepared = prepare_data_for_inference(modified_data, model_type, model_bundle)

    prepared_reference = None
    if reference_data is not None:
        prepared_reference = prepare_data_for_inference(reference_data, model_type, model_bundle)

    return generate_with_model(
        model_bundle,
        prepared,
        num_samples,
        model_type,
        observed_rows=observed_rows,
        engine_id=engine_id,
        immune_cols=immune_cols,
        prepared_reference_data=prepared_reference,
    )
