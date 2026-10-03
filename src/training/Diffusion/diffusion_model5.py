#!/usr/bin/env python3
"""
conditional_tabular_diffusion.py

Conditional diffusion model for tabular aerospace engine sensor data.

Column-agnostic design: the only columns the model ever assumes *might* be
present are ``time`` and ``engine_id`` (auto-detected by name pattern, with
graceful fallback when absent).  Every other column is data; the model itself
chooses which of those columns to condition on via an unsupervised
variance + correlation heuristic, so engineers do not have to know which
parameters drive operational distributions.

Usage:
  python diffusion_model5.py --mode train --data data.csv --epochs 50
  python diffusion_model5.py --mode generate --model model.pt --n-engines 5 --cycles-per-engine 100
"""

import argparse
import logging
import os
import math
import time
from pathlib import Path
from datetime import datetime
import sys
import json

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None
from typing import Tuple, Optional

import numpy as np

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


DEFAULT_CONFIG_REL_PATH = Path("src") / "config" / "diffusion_config.yaml"


def _resolve_config_path(path: Optional[str]) -> Path:
    root = _repo_root()
    if not path:
        return root / DEFAULT_CONFIG_REL_PATH
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = root / config_path
    return config_path


def _load_config(path: Optional[str]) -> dict:
    config_path = _resolve_config_path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    suffix = config_path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        if yaml is None:
            raise RuntimeError("PyYAML is required to load .yaml/.yml configs")
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    if suffix == ".json":
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    raise ValueError(f"Unsupported config type: {config_path}")


def _select_checkpoint_path(args, config: dict) -> Path:
    if args.checkpoint:
        model_path = Path(args.checkpoint)
        if model_path.suffix.lower() != ".pt":
            raise ValueError(f"Checkpoint must be a .pt file: {model_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {model_path}")
        return model_path
    
    paths_cfg = config.get("paths", {}) if isinstance(config.get("paths"), dict) else {}
    checkpoints_dir = paths_cfg.get("checkpoints_dir")
    
    if not checkpoints_dir:
        # Fallback to the default directory used during training
        checkpoints_dir = Path("src") / "model_checkPoints" / args.model_name
        
    ckpt_base = Path(checkpoints_dir)
    if not ckpt_base.is_absolute():
        ckpt_base = _repo_root() / ckpt_base
        
    model_name = args.model if args.model.endswith(".pt") else args.model + "pt"
    model_path = ckpt_base / model_name
    
    if model_path.exists():
        return model_path
        
    candidates = sorted(ckpt_base.glob("*.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoints found in {ckpt_base}. Train first or pass --checkpoint."
        )
    return candidates[-1]


def _maybe_print_checkpoint_only() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mode", choices=["train", "generate", "fine-tune"], default="train")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--model", type=str, default="diff_best_ckpt.pt")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--print-checkpoint-only", action="store_true", default=False)
    args, _ = parser.parse_known_args()
    if args.print_checkpoint_only and args.mode == "generate":
        config = _load_config(args.config)
        model_path = _select_checkpoint_path(args, config)
        print(str(model_path))
        raise SystemExit(0)


if __name__ == "__main__":
    _maybe_print_checkpoint_only()

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from scipy import stats

# ---------------------------
# Configuration
# ---------------------------

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

# Name fragments used to *opportunistically* locate the only two columns the
# spec says we may assume might exist: time and engine_id.  Matching is
# case-insensitive substring; if nothing matches we fall back to "no metadata"
# and the entire file is treated as a single engine (per project spec).
_TIME_NAME_HINTS = ("time", "cycle", "timestep", "tstep")
_ENGINE_ID_NAME_HINTS = ("engine_id", "engineid", "unit", "engine")


def _detect_metadata_columns(columns: list) -> list:
    """Best-effort name-based detection of time and engine_id columns.

    Returns a list of column indices (possibly empty).  Any failure to match
    is silently absorbed — downstream code handles zero metadata columns.
    """
    lowered = [c.strip().lower() for c in columns]
    found = []

    def _find(hints):
        # Prefer exact match, then substring match, then fail.
        for h in hints:
            if h in lowered:
                return lowered.index(h)
        for h in hints:
            for i, name in enumerate(lowered):
                if h in name:
                    return i
        return None

    eid = _find(_ENGINE_ID_NAME_HINTS)
    tcol = _find(_TIME_NAME_HINTS)
    # Order matters for downstream synthetic-id generation: engine_id first.
    if eid is not None:
        found.append(eid)
    if tcol is not None and tcol != eid:
        found.append(tcol)
    return found


def _normalised_histogram_entropy(values: np.ndarray, n_bins: int = 32) -> float:
    """Shannon entropy of a histogram, normalised to [0, 1].

    Computed on already-standardised values, so the result is unit-free:
    a column in Pa and a column in K with the same shape get the same score.
    A constant column scores 0; a uniform-over-bins column scores 1.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0 or np.ptp(finite) < 1e-12:
        return 0.0
    hist, _ = np.histogram(finite, bins=n_bins)
    p = hist.astype(np.float64)
    p = p[p > 0]
    p = p / p.sum()
    h = -np.sum(p * np.log(p))
    h_max = math.log(n_bins)
    return float(h / h_max) if h_max > 0 else 0.0


def select_conditioning_columns(data: np.ndarray, candidate_idx: list,
                                n_conditions: int = 3,
                                corr_threshold: float = 0.9) -> list:
    """Pick informative columns to condition on, *without unit bias*.

    Strategy (unsupervised — no engineer input required):
      1. Restrict to ``candidate_idx`` (everything except metadata).
      2. Standardise each column to z-scores, then score it by the normalised
         Shannon entropy of its histogram. This is unit-free: a column
         measured in Pa and one in K with the same shape get the same score.
         A constant column scores 0; a uniformly-spread one scores ~1.
      3. Greedily accept top-ranked columns, rejecting any whose absolute
         correlation with an already-accepted column exceeds ``corr_threshold``
         (so the conditioning set stays diverse).
      4. Stop once we have ``n_conditions`` columns or exhaust candidates.

    Returns a list of selected column indices (in the original frame), in the
    order they were chosen.  May return fewer than ``n_conditions`` if not
    enough diverse candidates exist; may return an empty list, in which case
    the model becomes unconditional.
    """
    if not candidate_idx or n_conditions <= 0:
        return []

    sub = data[:, candidate_idx].astype(np.float64, copy=False)
    col_std = np.std(sub, axis=0)
    valid_mask = col_std > 1e-9
    if not valid_mask.any():
        return []

    # Standardise so the entropy score and the correlation check are both
    # invariant to unit / scale differences across columns.
    standardised = np.zeros_like(sub)
    standardised[:, valid_mask] = (
        (sub[:, valid_mask] - np.mean(sub[:, valid_mask], axis=0)) / col_std[valid_mask]
    )

    # Unit-free information score per column.
    scores = np.full(sub.shape[1], -np.inf)
    for i in range(sub.shape[1]):
        if valid_mask[i]:
            scores[i] = _normalised_histogram_entropy(standardised[:, i])

    order_local = np.argsort(-scores)  # descending

    selected_local = []
    for cand in order_local:
        if scores[cand] == -np.inf:
            break
        if not selected_local:
            selected_local.append(int(cand))
        else:
            already = standardised[:, selected_local]
            this = standardised[:, cand:cand + 1]
            corrs = np.abs(np.mean(already * this, axis=0))
            if np.max(corrs) < corr_threshold:
                selected_local.append(int(cand))
        if len(selected_local) >= n_conditions:
            break

    return [candidate_idx[i] for i in selected_local]


def auto_resolve_columns(data: np.ndarray, columns: list,
                         n_conditions: int = 3) -> dict:
    """Decide metadata / condition / target columns with no human input.

    - Metadata: time and engine_id columns, located by name pattern only as a
      best-effort hint.  May be empty.
    - Conditions: chosen by :func:`select_conditioning_columns` from the
      remaining columns.
    - Targets: every column not classified above.

    The mapping is stored in the checkpoint so generation can reproduce the
    exact column layout without re-running the heuristic.
    """
    metadata_idx = _detect_metadata_columns(columns)
    metadata_set = set(metadata_idx)
    candidate_idx = [i for i in range(len(columns)) if i not in metadata_set]

    condition_idx = select_conditioning_columns(data, candidate_idx,
                                                n_conditions=n_conditions)
    cond_set = set(condition_idx)

    sensor_idx = [i for i in candidate_idx if i not in cond_set]
    return {
        "metadata_idx": metadata_idx,
        "condition_idx": condition_idx,
        "sensor_idx": sensor_idx,
    }


# ---------------------------------------------------------------------------
# Snapshot vs continuous data format detection + per-engine grouping
# ---------------------------------------------------------------------------

def detect_data_format(data: np.ndarray, col_indices: dict) -> str:
    """Best-effort guess: ``"snapshot"`` (one row = one flight) or ``"continuous"``.

    Heuristic:
      - If we have an engine_id column AND the engine_id values repeat
        (more rows than unique engines), the file is continuous time-series.
      - If engine_id values are all unique (or there is no engine_id at all
        but we have many rows), assume snapshot.
      - Edge case: very small files default to snapshot, since the model
        does not need temporal modelling for those.

    The format string is stashed in the checkpoint and used to drive the
    generation strategy (rolling time index for continuous, single row per
    "engine" for snapshot).
    """
    meta_idx = col_indices.get("metadata_idx", [])
    n_rows = data.shape[0]
    if not meta_idx:
        return "snapshot"
    eid_col = data[:, meta_idx[0]]
    unique = np.unique(eid_col)
    if unique.size == 0:
        return "snapshot"
    avg_rows_per_engine = n_rows / unique.size
    return "continuous" if avg_rows_per_engine > 1.5 else "snapshot"


def group_rows_by_engine(data: np.ndarray, col_indices: dict) -> list:
    """Return a list of (engine_id, row_indices) groups.

    If there is no engine_id metadata column, the entire file is treated
    as a single engine — per the project context's "no engine_id ⇒ one
    engine" rule.
    """
    meta_idx = col_indices.get("metadata_idx", [])
    n_rows = data.shape[0]
    if not meta_idx:
        return [(0, np.arange(n_rows))]
    eid_col = data[:, meta_idx[0]]
    groups = []
    for eid in np.unique(eid_col):
        idx = np.where(eid_col == eid)[0]
        groups.append((int(eid), idx))
    return groups

def verbose_print(verbose: bool, *args, **kwargs):
    """Print only when verbosity enabled."""
    if verbose:
        print(*args, **kwargs)


# ---------------------------
# Data loading with conditioning
# ---------------------------

def _load_single_csv(path: str) -> Tuple[np.ndarray, list]:
    import csv
    with open(path, "r", newline='', encoding='utf-8-sig') as f:
        # Read header
        first = f.readline().strip()
        cols = [c.strip() for c in csv.reader([first]).__next__()]

        # Read data
        f.seek(0)
        data = np.genfromtxt(f, delimiter=",", skip_header=1)

    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    return data, cols


def load_engine_data(path) -> Tuple[np.ndarray, list]:
    """
    Load engine sensor data from one or more CSV files.

    ``path`` may be a single path (str / Path) or a sequence of paths. When
    multiple files are given, their headers must name the same columns
    (order may differ — each file is reordered to match the first).  Rows
    from all files are concatenated in the order given.

    Returns:
      - data: (N, D) float32 numpy array with all columns
      - columns: list of column names (from the first file)
    """
    if isinstance(path, (list, tuple)):
        paths = [str(p) for p in path]
    else:
        paths = [str(path)]

    if not paths:
        raise ValueError("load_engine_data requires at least one CSV path")

    if len(paths) == 1:
        return _load_single_csv(paths[0])

    first_data, first_cols = _load_single_csv(paths[0])
    first_stripped = [c.strip() for c in first_cols]
    first_set = set(first_stripped)

    arrays = [first_data]
    for p in paths[1:]:
        data, cols = _load_single_csv(p)
        stripped = [c.strip() for c in cols]
        cols_set = set(stripped)
        if cols_set != first_set:
            missing = sorted(first_set - cols_set)
            extra = sorted(cols_set - first_set)
            raise ValueError(
                f"Column mismatch in {p} vs {paths[0]}: "
                f"missing={missing}, extra={extra}"
            )
        # Reorder this file's columns to match the first file's order.
        index_map = [stripped.index(name) for name in first_stripped]
        arrays.append(data[:, index_map])

    combined = np.concatenate(arrays, axis=0).astype(np.float32)
    return combined, first_cols


def prepare_conditional_data(data: np.ndarray, col_indices: dict):
    """
    Split data into role-based groups using ``col_indices``.

    - metadata columns are excluded from training (passed through for output).
    - condition columns are standardised and used as conditioning input.
    - target/sensor columns are standardised; the model learns to generate these.

    If a role has no columns the corresponding array will have zero width
    (shape ``(N, 0)``).

    Returns: metadata, conditions_std, sensors_std, condition_stats, sensor_stats
    """
    n = data.shape[0]

    # Extract columns using resolved indices
    meta_idx = col_indices["metadata_idx"]
    cond_idx = col_indices["condition_idx"]
    sens_idx = col_indices["sensor_idx"]

    metadata = data[:, meta_idx] if meta_idx else np.empty((n, 0), dtype=np.float32)
    conditions = data[:, cond_idx] if cond_idx else np.empty((n, 0), dtype=np.float32)
    sensors = data[:, sens_idx] if sens_idx else np.empty((n, 0), dtype=np.float32)

    # Standardize conditions (for conditioning)
    if conditions.shape[1] > 0:
        cond_mean = np.mean(conditions, axis=0)
        cond_std = np.std(conditions, axis=0)
        cond_std = np.where(cond_std < 1e-6, 1.0, cond_std)
        conditions_std = (conditions - cond_mean) / cond_std
    else:
        cond_mean = np.array([], dtype=np.float32)
        cond_std = np.array([], dtype=np.float32)
        conditions_std = conditions

    # Standardize sensors (what we generate)
    sensor_mean = np.mean(sensors, axis=0)
    sensor_std = np.std(sensors, axis=0)
    sensor_std = np.where(sensor_std < 1e-6, 1.0, sensor_std)
    sensors_std = (sensors - sensor_mean) / sensor_std

    condition_stats = {'mean': cond_mean, 'std': cond_std}
    sensor_stats = {'mean': sensor_mean, 'std': sensor_std}

    return (metadata.astype(np.float32),
            conditions_std.astype(np.float32),
            sensors_std.astype(np.float32),
            condition_stats,
            sensor_stats)


def save_engine_data(metadata: np.ndarray, conditions: np.ndarray, sensors: np.ndarray,
                     path: str, columns: list, col_indices: dict):
    """
    Save complete engine data, placing each column back in its original position.

    Args:
        metadata:    (N, M) metadata values
        conditions:  (N, C) condition values
        sensors:     (N, S) sensor/target values
        path:        output CSV path
        columns:     column names (original order)
        col_indices: mapping of role -> list of original column indices
    """
    n_rows = metadata.shape[0]
    n_cols = len(columns)
    full_data = np.zeros((n_rows, n_cols), dtype=np.float32)

    for i, idx in enumerate(col_indices["metadata_idx"]):
        full_data[:, idx] = metadata[:, i]
    for i, idx in enumerate(col_indices["condition_idx"]):
        full_data[:, idx] = conditions[:, i]
    for i, idx in enumerate(col_indices["sensor_idx"]):
        full_data[:, idx] = sensors[:, i]

    import csv
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(full_data)


def validate_checkpoint(ckpt: dict, sensor_dim: int, condition_dim: int) -> None:
    if not isinstance(ckpt, dict):
        raise ValueError("Checkpoint must be a dict")

    required_keys = {"model_state", "schedule", "condition_stats", "sensor_stats", "columns"}
    missing = required_keys - set(ckpt.keys())
    if missing:
        raise ValueError(f"Checkpoint missing keys: {sorted(missing)}")

    schedule = ckpt["schedule"]
    if not isinstance(schedule, dict):
        raise ValueError("schedule must be a dict")
    if "T" not in schedule or "betas" not in schedule:
        raise ValueError("schedule must contain 'T' and 'betas'")

    betas = schedule["betas"]
    if hasattr(betas, "shape"):
        betas_len = betas.shape[0]
    else:
        betas_len = len(betas)
    if betas_len != int(schedule["T"]):
        raise ValueError("schedule['betas'] length must equal schedule['T']")

    for name, stats, dim in (
        ("condition_stats", ckpt["condition_stats"], condition_dim),
        ("sensor_stats", ckpt["sensor_stats"], sensor_dim),
    ):
        if not isinstance(stats, dict) or "mean" not in stats or "std" not in stats:
            raise ValueError(f"{name} must have 'mean' and 'std'")
        if len(stats["mean"]) != dim or len(stats["std"]) != dim:
            raise ValueError(f"{name} mean/std must be length {dim}")

    columns = ckpt["columns"]
    col_indices = ckpt.get("col_indices", {})
    n_metadata = len(col_indices.get("metadata_idx", []))
    min_columns = n_metadata + condition_dim + sensor_dim
    if not isinstance(columns, (list, tuple)) or len(columns) < min_columns:
        raise ValueError("columns must cover metadata, conditions, and sensors")


# ---------------------------
# Diffusion utilities
# ---------------------------

def linear_beta_schedule(T: int, beta_start=1e-4, beta_end=2e-2) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, T)


class DiffusionSchedule:
    def __init__(self, T: int = 50, beta_start=1e-4, beta_end=2e-2, device=None):
        if device is None:
            device = DEVICE
        self.T = T
        betas = linear_beta_schedule(T, beta_start, beta_end)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.betas = betas.to(device)
        self.alphas = alphas.to(device)
        self.alpha_bar = alpha_bar.to(device)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        if noise is None:
            noise = torch.randn_like(x0)
        a_bar = self.sqrt_alpha_bar[t].unsqueeze(-1)
        b = self.sqrt_one_minus_alpha_bar[t].unsqueeze(-1)
        return a_bar * x0 + b * noise


# ---------------------------
# Conditional denoiser model
# ---------------------------

class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half, dtype=torch.float32, device=t.device) / float(half)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class ConditionalDenoiser(nn.Module):
    """
    Denoiser that takes:
    - x_t: noisy sensor readings (sensor_dim dims)
    - t: timestep
    - c: conditions (condition_dim dims)

    Predicts noise in sensor readings.
    """
    def __init__(self, sensor_dim: int, condition_dim: int,
                 hidden_dim: int = 256, time_emb_dim: int = 128, n_layers: int = 4):
        super().__init__()
        self.sensor_dim = sensor_dim
        self.condition_dim = condition_dim

        # Time embedding
        self.time_mlp = nn.Sequential(
            TimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
        )

        # Condition embedding (only created when there are condition columns)
        cond_out_dim = 0
        if condition_dim > 0:
            cond_out_dim = hidden_dim // 2
            self.condition_mlp = nn.Sequential(
                nn.Linear(condition_dim, cond_out_dim),
                nn.SiLU(),
                nn.Linear(cond_out_dim, cond_out_dim),
                nn.SiLU(),
            )
        else:
            self.condition_mlp = None

        # Main network: combines noisy sensors + time + (optional) conditions
        layers = []
        in_dim = sensor_dim + time_emb_dim + cond_out_dim

        for i in range(n_layers):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(0.1))

        layers.append(nn.Linear(hidden_dim, sensor_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor = None):
        """
        x: (B, sensor_dim) noisy sensor readings
        t: (B,) timesteps
        c: (B, condition_dim) conditions — may be None if unconditional

        Returns: (B, sensor_dim) predicted noise
        """
        # Embed time
        temb = self.time_mlp(t)

        # Concatenate all inputs
        parts = [x, temb]
        if self.condition_mlp is not None and c is not None:
            parts.append(self.condition_mlp(c))
        h = torch.cat(parts, dim=-1)

        # Predict noise
        return self.net(h)


# ===========================================================================
# LoRA (Low-Rank Adaptation) — for distribution shift via seed-data fine-tune
# ===========================================================================

class LoRALinear(nn.Module):
    """Wraps an existing nn.Linear with a low-rank trainable update.

    The base layer's weights stay frozen; only ``lora_A`` and ``lora_B`` are
    learned, which keeps the per-adapter parameter count tiny (~rank * (in+out)
    instead of in*out).  The forward pass is::

        y = base(x) + (x @ A) @ B * (alpha / rank)

    Adapters can be saved/loaded independently of the base checkpoint, so the
    LoRA / seed-data fine-tuning workflow can produce many lightweight files
    that compose with one shared base model.
    """
    def __init__(self, base: nn.Linear, rank: int = 4, alpha: float = 8.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.lora_A = nn.Parameter(torch.zeros(base.in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, base.out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B starts at zero so the adapter is a no-op until trained.

    # PyTorch's transformer fast paths introspect Linear submodules via the
    # raw ``.weight`` / ``.bias`` attributes. Proxy them straight through to
    # the frozen base layer so any Linear can be wrapped without breakage.
    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.lora_A) @ self.lora_B * self.scale


def apply_lora_to_model(model: nn.Module, rank: int = 4, alpha: float = 8.0) -> list:
    """Walk a module tree and wrap *safe-to-wrap* nn.Linear modules with LoRA.

    PyTorch's ``nn.MultiheadAttention`` reaches into the underlying ``Linear``
    layers' raw ``.weight`` / ``.bias`` tensors via attribute access (so it
    can fuse the q/k/v projection), so wrapping those would crash the forward
    pass.  We skip every Linear that lives anywhere inside a MultiheadAttention
    subtree.

    Returns a list of the parameters that should be trained (i.e., the
    LoRA matrices), so callers can pass them straight to an optimizer.
    """
    # Find every module that is or sits inside a MultiheadAttention so we can
    # filter wrapping decisions against it.
    forbidden_ids = set()
    for m in model.modules():
        if isinstance(m, nn.MultiheadAttention):
            for sub in m.modules():
                forbidden_ids.add(id(sub))

    lora_params = []
    for parent in list(model.modules()):
        if id(parent) in forbidden_ids:
            continue
        for child_name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear) and id(child) not in forbidden_ids:
                wrapped = LoRALinear(child, rank=rank, alpha=alpha)
                setattr(parent, child_name, wrapped)
                lora_params.append(wrapped.lora_A)
                lora_params.append(wrapped.lora_B)
    # Freeze every non-lora parameter for safety
    for name, p in model.named_parameters():
        if not name.endswith("lora_A") and not name.endswith("lora_B"):
            p.requires_grad = False
    return lora_params


def extract_lora_state(model: nn.Module) -> dict:
    """Snapshot only the LoRA parameters from a model into a state dict."""
    out = {}
    for name, p in model.named_parameters():
        if name.endswith("lora_A") or name.endswith("lora_B"):
            out[name] = p.detach().cpu().clone()
    return out


def load_lora_state(model: nn.Module, lora_state: dict) -> None:
    """Load a LoRA state dict back onto a (LoRA-wrapped) model."""
    own = dict(model.named_parameters())
    for k, v in lora_state.items():
        if k in own:
            own[k].data.copy_(v.to(own[k].device))


# ===========================================================================
# Per-column statistics — used to seed learned column embeddings
# ===========================================================================

# Number of unit-free distribution features per column.  Order matters because
# checkpoints encode the dimension; if you change this constant, retrain.
COLUMN_STATS_DIM = 7


def compute_column_stats(data: np.ndarray) -> np.ndarray:
    """Per-column distribution shape features, computed on standardised data.

    Each column is independently z-scored, then summarised with seven
    unit-free shape features::

        [skew, excess_kurtosis, q10, q25, q50, q75, q90]

    Constant columns yield an all-zero feature vector.  Returned shape is
    ``(n_cols, COLUMN_STATS_DIM)``.
    """
    n_rows, n_cols = data.shape
    out = np.zeros((n_cols, COLUMN_STATS_DIM), dtype=np.float32)
    for j in range(n_cols):
        col = data[:, j].astype(np.float64)
        col = col[np.isfinite(col)]
        if col.size < 2:
            continue
        std = col.std()
        if std < 1e-9:
            continue
        z = (col - col.mean()) / std
        # skew, excess kurt
        m3 = np.mean(z ** 3)
        m4 = np.mean(z ** 4)
        skew = m3
        excess_kurt = m4 - 3.0
        q10, q25, q50, q75, q90 = np.quantile(z, [0.1, 0.25, 0.5, 0.75, 0.9])
        out[j] = [skew, excess_kurt, q10, q25, q50, q75, q90]
    return out


# ===========================================================================
# Transformer-based masked-diffusion denoiser
# ===========================================================================

class TabularTransformerDenoiser(nn.Module):
    """Transformer denoiser that lets the model learn what to condition on.

    Each column of the row becomes a token whose embedding is the sum of:
      - a content-derived column identity (a learned projection of the
        column's distribution-shape statistics — see ``compute_column_stats``),
      - a value projection (the noisy or observed value at this position),
      - a mask projection (1 if the column is being held fixed as a
        conditioning input at this denoising step, 0 if it is being denoised),
      - a broadcast time embedding.

    Self-attention then lets every column attend to every other, so the
    model itself learns which columns inform which — there is no fixed
    "condition" / "target" split.  Order independence is real because the
    column identity is derived purely from each column's standardised
    statistics, never from its position in the CSV.
    """

    def __init__(self, n_cols: int, hidden_dim: int = 128, n_heads: int = 4,
                 n_layers: int = 4, time_emb_dim: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.n_cols = n_cols
        self.hidden_dim = hidden_dim

        # Content-derived column identity.  ``column_stats`` is a buffer that
        # callers populate via ``set_column_stats`` before training.
        self.col_stats_proj = nn.Linear(COLUMN_STATS_DIM, hidden_dim)
        self.register_buffer("column_stats", torch.zeros(n_cols, COLUMN_STATS_DIM))

        # Token projections
        self.value_proj = nn.Linear(1, hidden_dim)
        self.mask_proj = nn.Linear(1, hidden_dim)

        # Time embedding broadcast across all column tokens
        self.time_mlp = nn.Sequential(
            TimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, hidden_dim),
            nn.SiLU(),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout,
            batch_first=True, norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(hidden_dim, 1)

    def set_column_stats(self, stats: np.ndarray) -> None:
        """Populate the content-derived per-column identity features."""
        if stats.shape != (self.n_cols, COLUMN_STATS_DIM):
            raise ValueError(
                f"column_stats must be ({self.n_cols}, {COLUMN_STATS_DIM}), "
                f"got {stats.shape}"
            )
        self.column_stats.copy_(torch.from_numpy(stats.astype(np.float32)))

    def _column_embeddings(self) -> torch.Tensor:
        # (n_cols, hidden_dim)
        return self.col_stats_proj(self.column_stats)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x:    (B, n_cols) noisy / partially-observed values
        t:    (B,)        diffusion timesteps
        mask: (B, n_cols) 1 where the column is held fixed as a known
              conditioning value, 0 where the model is denoising it.

        Returns: (B, n_cols) predicted noise (only meaningful where mask==0).
        """
        B, D = x.shape
        if mask is None:
            mask = torch.zeros_like(x)

        col_emb = self._column_embeddings()              # (D, H)
        value_tok = self.value_proj(x.unsqueeze(-1))      # (B, D, H)
        mask_tok = self.mask_proj(mask.unsqueeze(-1))     # (B, D, H)
        temb = self.time_mlp(t).unsqueeze(1)              # (B, 1, H)

        tokens = col_emb.unsqueeze(0) + value_tok + mask_tok + temb  # (B, D, H)
        out = self.encoder(tokens)                        # (B, D, H)
        return self.head(out).squeeze(-1)                 # (B, D)


# ---------------------------
# Training
# ---------------------------

def p_losses(model: nn.Module, schedule: DiffusionSchedule, 
             x0: torch.Tensor, c: torch.Tensor, t: torch.Tensor):
    """
    Conditional diffusion loss.
    
    x0: (B, sensor_dim) clean sensors
    c: (B, condition_dim) conditions
    t: (B,) timesteps
    """
    noise = torch.randn_like(x0)
    xt = schedule.q_sample(x0, t, noise=noise)
    eps_hat = model(xt, t, c)
    return nn.functional.mse_loss(eps_hat, noise)


def train(model: nn.Module, schedule: DiffusionSchedule, dataloader: DataLoader,
          epochs: int = 10, lr: float = 1e-3, device=None, verbose: bool = True,
          save_dir: Optional[str] = None, cond_stats: dict = None, sensor_stats: dict = None,
          cols: list = None, col_indices: dict = None):
    if device is None:
        device = DEVICE
    
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    T = schedule.T

    best_loss = float('inf')
    last_ckpt = None

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        num_batches = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}", unit="batch")
        
        for batch in pbar:
            conditions, sensors = batch
            conditions = conditions.to(device)
            sensors = sensors.to(device)
            batch_size = sensors.shape[0]

            # Pass None when unconditional (zero-width conditions)
            c = conditions if conditions.shape[1] > 0 else None
            t = torch.randint(0, T, (batch_size,), device=device, dtype=torch.long)
            loss = p_losses(model, schedule, sensors, c, t)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.6f}")

        avg_loss = epoch_loss / max(1, num_batches)
        verbose_print(verbose, f"[Epoch {epoch}] avg loss: {avg_loss:.6f}")

        if save_dir is not None:
            ckpt = {
                "model_state": model.state_dict(),
                "opt_state": optimizer.state_dict(),
                "epoch": epoch,
                "schedule": {
                    "T": schedule.T,
                    "betas": schedule.betas.cpu().numpy(),
                },
                "condition_stats": cond_stats,
                "sensor_stats": sensor_stats,
                "columns": cols,
                "col_indices": col_indices,
            }
            last_ckpt = ckpt
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = Path(save_dir) / "diff_best_ckpt.pt"
                torch.save(ckpt, best_path)
                verbose_print(verbose, f"  [Best Model] New best loss: {best_loss:.6f} (saved to {best_path.name})")

    if save_dir is not None and last_ckpt is not None:
        save_dir_path = Path(save_dir)
        existing_runs = list(save_dir_path.glob("checkpoint_run_*.pt"))
        run_nums = []
        for p in existing_runs:
            try:
                num = int(p.stem.split("_")[-1])
                run_nums.append(num)
            except ValueError:
                pass
        next_run = max(run_nums) + 1 if run_nums else 1
        run_path = save_dir_path / f"checkpoint_run_{next_run:03d}.pt"
        torch.save(last_ckpt, run_path)
        verbose_print(verbose, f"Saved run checkpoint to {run_path}")

    return model


# ===========================================================================
# Masked-diffusion training (transformer denoiser)
# ===========================================================================

def sample_random_mask(batch_size: int, n_cols: int, device,
                       min_observed: float = 0.0,
                       max_observed: float = 0.8) -> torch.Tensor:
    """Sample a random observed-column mask per row.

    Each row gets an independent fraction ``p`` of columns held fixed.
    Training across the full mask range teaches the model to handle every
    subset-column case the seed-data inpainting workflow can throw at it
    (zero conditioning, all-but-one observed, anywhere in between).
    """
    p = torch.rand(batch_size, 1, device=device) * (max_observed - min_observed) + min_observed
    rand = torch.rand(batch_size, n_cols, device=device)
    return (rand < p).float()


def masked_p_losses(model: TabularTransformerDenoiser,
                    schedule: DiffusionSchedule,
                    x0: torch.Tensor,
                    t: torch.Tensor,
                    mask: torch.Tensor) -> torch.Tensor:
    """Conditional diffusion loss restricted to the unmasked positions.

    Observed columns (mask==1) are passed through unchanged on the input
    side, and the loss only counts the model's noise predictions for the
    columns it is supposed to denoise (mask==0).
    """
    noise = torch.randn_like(x0)
    xt = schedule.q_sample(x0, t, noise=noise)
    # Where mask==1 we keep the clean value as the "observed" input.
    x_in = mask * x0 + (1.0 - mask) * xt
    eps_hat = model(x_in, t, mask)

    # Only score noise prediction for the denoised positions.
    target = (1.0 - mask) * noise
    pred = (1.0 - mask) * eps_hat
    denom = (1.0 - mask).sum().clamp_min(1.0)
    return ((pred - target) ** 2).sum() / denom


def train_transformer(model: TabularTransformerDenoiser,
                      schedule: DiffusionSchedule,
                      dataloader: DataLoader,
                      epochs: int = 10,
                      lr: float = 1e-3,
                      device=None,
                      verbose: bool = True,
                      save_dir: Optional[str] = None,
                      ckpt_meta: Optional[dict] = None,
                      trainable_params: Optional[list] = None) -> nn.Module:
    """Train the masked-diffusion transformer denoiser.

    ``ckpt_meta`` is merged into every saved checkpoint and is the place to
    stash columns / col_indices / sensor_stats so generation can reproduce
    the exact training-time layout.

    ``trainable_params`` lets callers train only a subset of parameters
    (e.g. just the LoRA matrices for seed-data fine-tuning).
    """
    if device is None:
        device = DEVICE
    model.to(device)
    model.train()
    params = trainable_params if trainable_params is not None else list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    T = schedule.T
    n_cols = model.n_cols

    best_loss = float('inf')

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        num_batches = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}", unit="batch")
        for batch in pbar:
            x0 = batch[0].to(device)
            B = x0.shape[0]
            t = torch.randint(0, T, (B,), device=device, dtype=torch.long)
            mask = sample_random_mask(B, n_cols, device)
            loss = masked_p_losses(model, schedule, x0, t, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.6f}")

        avg = epoch_loss / max(1, num_batches)
        verbose_print(verbose, f"[Epoch {epoch}] avg loss: {avg:.6f}")

        if save_dir is not None:
            ckpt = {
                "arch": "transformer",
                "model_state": model.state_dict(),
                "epoch": epoch,
                "schedule": {"T": schedule.T, "betas": schedule.betas.cpu().numpy()},
                **(ckpt_meta or {}),
            }
            if avg < best_loss:
                best_loss = avg
                best_path = Path(save_dir) / "diff_best_ckpt.pt"
                torch.save(ckpt, best_path)
                verbose_print(verbose, f"  [Best] {best_loss:.6f} -> {best_path.name}")

    return model


# ===========================================================================
# Inpainting sampler — produces full rows from any subset of observed cols
# ===========================================================================

@torch.no_grad()
def sample_inpaint(model: TabularTransformerDenoiser,
                   schedule: DiffusionSchedule,
                   n_samples: int,
                   observed_values: Optional[torch.Tensor] = None,
                   observed_mask: Optional[torch.Tensor] = None,
                   device=None,
                   verbose: bool = True) -> np.ndarray:
    """Reverse-diffuse a full row, holding ``observed`` columns fixed.

    ``observed_values`` and ``observed_mask`` are both ``(n_samples, n_cols)``
    in standardised space.  Pass ``None``/``None`` for fully unconditional
    sampling.

    Implementation: at every reverse step we re-noise the *observed* part of
    the row to the current timestep's noise level (so it lives on the same
    manifold as the iterate) and then overwrite the model's prediction for
    those positions.  This is the standard RePaint-style inpainting trick.
    """
    if device is None:
        device = DEVICE
    model.to(device)
    model.eval()
    T = schedule.T
    n_cols = model.n_cols

    if observed_values is None:
        observed_values = torch.zeros(n_samples, n_cols, device=device)
    else:
        observed_values = observed_values.to(device)
    if observed_mask is None:
        observed_mask = torch.zeros(n_samples, n_cols, device=device)
    else:
        observed_mask = observed_mask.to(device)

    x = torch.randn(n_samples, n_cols, device=device)
    # Initialise the observed positions to a noised version of the truth.
    if observed_mask.any():
        t_init = torch.full((n_samples,), T - 1, device=device, dtype=torch.long)
        noised_obs = schedule.q_sample(observed_values, t_init)
        x = observed_mask * noised_obs + (1.0 - observed_mask) * x

    for t_idx in tqdm(range(T - 1, -1, -1), desc="Sampling", disable=not verbose):
        t = torch.full((n_samples,), t_idx, device=device, dtype=torch.long)
        eps_hat = model(x, t, observed_mask)

        beta_t = schedule.betas[t_idx]
        alpha_t = schedule.alphas[t_idx]
        coef1 = 1.0 / torch.sqrt(alpha_t)
        coef2 = beta_t / torch.sqrt(1.0 - schedule.alpha_bar[t_idx])
        mean = coef1 * (x - coef2 * eps_hat)

        if t_idx > 0:
            noise = torch.randn_like(x)
            sigma = torch.sqrt(beta_t)
            x_new = mean + sigma * noise
            # Re-noise the observed values to the next timestep so the
            # observed and unobserved parts of the row live at the same
            # noise level.
            t_prev = torch.full((n_samples,), t_idx - 1, device=device, dtype=torch.long)
            noised_obs = schedule.q_sample(observed_values, t_prev)
        else:
            x_new = mean
            noised_obs = observed_values

        x = observed_mask * noised_obs + (1.0 - observed_mask) * x_new

    return x.cpu().numpy()


# ===========================================================================
# Seed-data fine-tuning (LoRA-based distribution shift)
# ===========================================================================

def fine_tune_with_seed(base_ckpt_path: str,
                        seed_data_path,
                        adapter_out_path: str,
                        epochs: int = 8,
                        lr: float = 1e-5,
                        rank: int = 4,
                        alpha: float = 8.0,
                        batch_size: int = 64,
                        verbose: bool = True) -> str:
    """Train a LoRA adapter from one or more seed CSVs.

    ``seed_data_path`` accepts a single path or a list of paths (delegated to
    :func:`load_engine_data`, so column names must match across files).

    Seed files may contain only a *subset* of the base model's columns —
    missing columns are simply masked out of the loss, so the adapter learns
    to shift the distribution of whatever channels the seeds actually have.
    """
    ckpt = torch.load(base_ckpt_path, map_location='cpu', weights_only=False)
    if ckpt.get("arch") != "transformer":
        raise RuntimeError("Seed fine-tuning requires a transformer checkpoint.")

    cols = ckpt["columns"]
    col_indices = ckpt["col_indices"]
    sensor_stats = ckpt["sensor_stats"]
    column_stats = ckpt["column_stats"]
    n_cols = len(col_indices["sensor_idx"]) + len(col_indices["condition_idx"])

    # Rebuild the transformer
    model_cfg = ckpt.get("model_cfg", {})
    model = TabularTransformerDenoiser(
        n_cols=n_cols,
        hidden_dim=model_cfg.get("hidden_dim", 128),
        n_heads=model_cfg.get("n_heads", 4),
        n_layers=model_cfg.get("n_layers", 4),
    )
    model.set_column_stats(np.asarray(column_stats, dtype=np.float32))
    model.load_state_dict(ckpt["model_state"])

    # Wrap with LoRA
    lora_params = apply_lora_to_model(model, rank=rank, alpha=alpha)
    verbose_print(verbose, f"LoRA: {sum(p.numel() for p in lora_params):,} trainable params "
                           f"(rank={rank}, alpha={alpha})")

    # Load seed CSV and align columns by NAME (subset is OK)
    seed_raw, seed_cols = load_engine_data(seed_data_path)
    seed_raw = seed_raw[~np.isnan(seed_raw).any(axis=1)]
    seed_cols_l = [c.strip() for c in seed_cols]
    base_cols_l = [cols[i].strip() for i in col_indices["sensor_idx"] + col_indices["condition_idx"]]
    base_order = col_indices["sensor_idx"] + col_indices["condition_idx"]

    # Build a (n_seed_rows, n_cols) tensor where missing-in-seed positions
    # remain at the base mean (zero in standardised space) and are masked
    # out so they don't contribute to the loss.
    n_seed = seed_raw.shape[0]
    aligned = np.zeros((n_seed, n_cols), dtype=np.float32)
    seed_mask_np = np.zeros((n_seed, n_cols), dtype=np.float32)
    for col_pos, base_col_idx in enumerate(base_order):
        base_name = cols[base_col_idx].strip()
        if base_name in seed_cols_l:
            j = seed_cols_l.index(base_name)
            mu = sensor_stats["mean"][col_pos] if col_pos < len(sensor_stats["mean"]) else 0.0
            sd = sensor_stats["std"][col_pos] if col_pos < len(sensor_stats["std"]) else 1.0
            aligned[:, col_pos] = (seed_raw[:, j] - mu) / sd
            seed_mask_np[:, col_pos] = 1.0
    if seed_mask_np.sum() == 0:
        raise RuntimeError("No seed columns matched the base model's columns.")

    aligned_t = torch.from_numpy(aligned).float()
    mask_t = torch.from_numpy(seed_mask_np).float()
    dataset = TensorDataset(aligned_t, mask_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    schedule = DiffusionSchedule(T=int(ckpt["schedule"]["T"]))
    schedule.betas = torch.tensor(ckpt["schedule"]["betas"], device=DEVICE)
    schedule.alphas = 1.0 - schedule.betas
    schedule.alpha_bar = torch.cumprod(schedule.alphas, dim=0)
    schedule.sqrt_alpha_bar = torch.sqrt(schedule.alpha_bar)
    schedule.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - schedule.alpha_bar)

    model.to(DEVICE)
    model.train()
    optimizer = torch.optim.Adam(lora_params, lr=lr)
    T = schedule.T
    for epoch in range(1, epochs + 1):
        total = 0.0
        n_batches = 0
        for x0, m_seed in loader:
            x0 = x0.to(DEVICE)
            m_seed = m_seed.to(DEVICE)
            B = x0.shape[0]
            t = torch.randint(0, T, (B,), device=DEVICE, dtype=torch.long)

            # Use the seed-presence mask as the masked-diffusion mask.  We
            # are *training* the LoRA to denoise unobserved positions
            # conditioned on the observed seed columns; rows where everything
            # is observed contribute nothing (loss denom is 0), so flip half
            # the seed columns to "denoise" each step.
            denoise_mask = (torch.rand_like(m_seed) > 0.5).float() * m_seed
            input_mask = m_seed - denoise_mask  # observed = seed and not picked
            noise = torch.randn_like(x0)
            xt = schedule.q_sample(x0, t, noise=noise)
            x_in = input_mask * x0 + (1.0 - input_mask) * xt
            eps_hat = model(x_in, t, input_mask)

            target = denoise_mask * noise
            pred = denoise_mask * eps_hat
            denom = denoise_mask.sum().clamp_min(1.0)
            loss = ((pred - target) ** 2).sum() / denom

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            n_batches += 1
        verbose_print(verbose, f"[LoRA epoch {epoch}] avg loss: {total / max(1, n_batches):.6f}")

    adapter_state = extract_lora_state(model)
    if isinstance(seed_data_path, (list, tuple)):
        seed_record = [str(p) for p in seed_data_path]
    else:
        seed_record = str(seed_data_path)
    out = {
        "lora_state": adapter_state,
        "rank": rank,
        "alpha": alpha,
        "base_ckpt": str(base_ckpt_path),
        "seed_data": seed_record,
        "seed_columns": seed_cols_l,
    }
    Path(adapter_out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, adapter_out_path)
    verbose_print(verbose, f"Saved LoRA adapter to {adapter_out_path}")
    return adapter_out_path


def compose_adapters(model: TabularTransformerDenoiser,
                     adapter_paths: list,
                     verbose: bool = True) -> nn.Module:
    """Wrap ``model`` with LoRA layers and load multiple adapters additively.

    The composition rule is the standard LoRA-merge: each adapter writes its
    own ``A``/``B`` into the wrapped layer, and we sum them.  In practice we
    iterate adapters in order; the last one wins for any layer it touches,
    which matches the project context's "compose one condition adapter and
    one engine-variant adapter" intent.
    """
    if not adapter_paths:
        return model
    apply_lora_to_model(model, rank=4)  # placeholder rank, overwritten below
    for ap in adapter_paths:
        ad = torch.load(ap, map_location='cpu', weights_only=False)
        load_lora_state(model, ad["lora_state"])
        verbose_print(verbose, f"  applied adapter: {ap}")
    return model


# ---------------------------
# Sampling
# ---------------------------

@torch.no_grad()
def sample_conditional(model: nn.Module, schedule: DiffusionSchedule,
                      conditions: torch.Tensor = None, n_samples: int = None,
                      device=None, verbose: bool = True):
    """
    Sample sensor readings, optionally conditioned on operating conditions.

    conditions: (N, condition_dim) tensor — pass ``None`` for unconditional.
    n_samples:  required when ``conditions`` is None.

    Returns: (N, sensor_dim) numpy array of generated sensor readings.
    """
    if device is None:
        device = DEVICE

    model.to(device)
    model.eval()
    T = schedule.T

    if conditions is not None:
        n_samples = conditions.shape[0]
        conditions = conditions.to(device)
    elif n_samples is None:
        raise ValueError("Either conditions or n_samples must be provided")

    sensor_dim = model.sensor_dim

    # Start from pure noise
    x = torch.randn(n_samples, sensor_dim, device=device)

    # Reverse diffusion
    for t_idx in tqdm(range(T - 1, -1, -1), desc="Sampling", disable=not verbose):
        t = torch.full((n_samples,), t_idx, device=device, dtype=torch.long)

        # Predict noise (conditioned on operating conditions when available)
        eps_hat = model(x, t, conditions)
        
        # Compute posterior mean
        beta_t = schedule.betas[t_idx]
        alpha_t = schedule.alphas[t_idx]
        
        coef1 = 1.0 / torch.sqrt(alpha_t)
        coef2 = beta_t / torch.sqrt(1.0 - schedule.alpha_bar[t_idx])
        mean = coef1 * (x - coef2 * eps_hat)
        
        if t_idx > 0:
            noise = torch.randn_like(x)
            sigma = torch.sqrt(beta_t)
            x = mean + sigma * noise
        else:
            x = mean
    
    return x.cpu().numpy()


@torch.no_grad()
def sample_conditional_warm_start(
    model: nn.Module,
    schedule: DiffusionSchedule,
    conditions: torch.Tensor,
    x_init: torch.Tensor,
    t_start_frac: float = 0.6,
    device=None,
    verbose: bool = False,
):
    """
    Warm-started conditional sampling for propagation.

    Instead of starting from pure noise (t=T), we:
      1. Take x_init (the previous row's normalized sensor values)
      2. Run the forward diffusion process on it up to t_start
      3. Reverse-denoise from t_start back to t=0

    This anchors the generated sample near x_init while allowing the
    model to produce realistic variation conditioned on the operating
    conditions. Edits in x_init will propagate into the output.

    Args:
        model: trained ConditionalDenoiser
        schedule: DiffusionSchedule with precomputed noise tables
        conditions: (N, condition_dim) normalized operating conditions
        x_init: (N, sensor_dim) normalized sensor values from previous row
        t_start_frac: fraction of T to start from (0.0 = copy, 1.0 = pure noise)
        device: torch device
        verbose: print progress

    Returns:
        (N, sensor_dim) generated sensor readings (normalized)
    """
    if device is None:
        device = DEVICE

    model.to(device)
    model.eval()
    T = schedule.T

    t_start = max(1, int(T * t_start_frac))

    n_samples = conditions.shape[0]
    conditions = conditions.to(device)
    x_init = x_init.to(device)

    # Forward-diffuse x_init to timestep t_start
    noise = torch.randn_like(x_init)
    a_bar = schedule.sqrt_alpha_bar[t_start - 1]
    b = schedule.sqrt_one_minus_alpha_bar[t_start - 1]
    x = a_bar * x_init + b * noise

    # Reverse diffusion from t_start down to 0
    from tqdm import tqdm
    for t_idx in tqdm(range(t_start - 1, -1, -1), desc="Warm-start sampling", disable=not verbose):
        t = torch.full((n_samples,), t_idx, device=device, dtype=torch.long)

        eps_hat = model(x, t, conditions)

        beta_t = schedule.betas[t_idx]
        alpha_t = schedule.alphas[t_idx]

        coef1 = 1.0 / torch.sqrt(alpha_t)
        coef2 = beta_t / torch.sqrt(1.0 - schedule.alpha_bar[t_idx])
        mean = coef1 * (x - coef2 * eps_hat)

        if t_idx > 0:
            noise = torch.randn_like(x)
            sigma = torch.sqrt(beta_t)
            x = mean + sigma * noise
        else:
            x = mean

    return x.cpu().numpy()


def generate_engines_with_cycles(model: nn.Module, schedule: DiffusionSchedule,
                                 n_engines: int, cycles_per_engine: int,
                                 cond_stats: dict, sensor_stats: dict,
                                 n_metadata_cols: int = 0,
                                 conditions_ref: np.ndarray = None,
                                 device=None, verbose: bool = True):
    """
    Generate synthetic data for multiple engines, each with multiple cycles.

    Args:
        model: trained conditional denoiser
        schedule: diffusion schedule
        n_engines: number of engines to generate
        cycles_per_engine: number of time cycles per engine
        cond_stats: condition normalization statistics
        sensor_stats: sensor normalization statistics
        n_metadata_cols: how many metadata columns to synthesise
        conditions_ref: (optional) reference conditions to sample from
        device: torch device
        verbose: print progress

    Returns:
        metadata:   (N, n_metadata_cols) array of synthetic IDs
        conditions: (N, condition_dim) array of condition values
        sensors:    (N, sensor_dim) array of sensor readings
    """
    total_samples = n_engines * cycles_per_engine
    condition_dim = len(cond_stats["mean"]) if len(cond_stats["mean"]) > 0 else 0

    # Generate metadata columns (group-id, within-group-seq, extras get zeros)
    if n_metadata_cols > 0:
        engine_ids = []
        time_cycles = []
        for engine_id in range(1, n_engines + 1):
            engine_ids.extend([engine_id] * cycles_per_engine)
            time_cycles.extend(range(1, cycles_per_engine + 1))
        meta_arrays = [np.array(engine_ids, dtype=np.float32)]
        if n_metadata_cols >= 2:
            meta_arrays.append(np.array(time_cycles, dtype=np.float32))
        for _ in range(n_metadata_cols - 2):
            meta_arrays.append(np.zeros(total_samples, dtype=np.float32))
        metadata = np.column_stack(meta_arrays)
    else:
        metadata = np.empty((total_samples, 0), dtype=np.float32)

    # Generate or sample operating conditions
    if condition_dim == 0:
        conditions_std = np.empty((total_samples, 0), dtype=np.float32)
    elif conditions_ref is not None:
        idx = np.random.choice(len(conditions_ref), size=total_samples, replace=True)
        conditions_std = conditions_ref[idx]
    else:
        conditions_std = np.random.randn(total_samples, condition_dim).astype(np.float32)

    # Generate sensors conditioned on operating conditions
    verbose_print(verbose, f"Generating {total_samples} samples ({n_engines} engines × {cycles_per_engine} cycles)...")
    cond_torch = torch.from_numpy(conditions_std).float() if condition_dim > 0 else None
    sensors_std = sample_conditional(model, schedule, cond_torch, n_samples=total_samples,
                                     device=device, verbose=verbose)

    # Denormalize
    if condition_dim > 0:
        conditions = conditions_std * cond_stats['std'] + cond_stats['mean']
    else:
        conditions = conditions_std
    sensors = sensors_std * sensor_stats['std'] + sensor_stats['mean']

    return metadata, conditions, sensors


# ---------------------------
# Diagnostics
# ---------------------------

def marginal_ks_statistic(real: np.ndarray, synth: np.ndarray) -> np.ndarray:
    D = []
    for i in range(real.shape[1]):
        stat, _ = stats.ks_2samp(real[:, i], synth[:, i])
        D.append(stat)
    return np.array(D)


def correlation_distance(real: np.ndarray, synth: np.ndarray) -> float:
    real_corr = np.corrcoef(real.T)
    synth_corr = np.corrcoef(synth.T)
    real_corr = np.nan_to_num(real_corr, nan=0.0)
    synth_corr = np.nan_to_num(synth_corr, nan=0.0)
    return np.linalg.norm(real_corr - synth_corr, ord="fro")


def mmd_rbf(real: np.ndarray, synth: np.ndarray,
            sigma: Optional[float] = None,
            max_samples: int = 1000) -> float:
    """Maximum Mean Discrepancy with an RBF kernel.

    A standard MMD^2 estimator for evaluating distribution shift.  Lower
    values mean the two samples are closer in distribution.  ``sigma`` is
    set to the median pairwise distance if not given.
    """
    rng = np.random.default_rng(0)
    if real.shape[0] > max_samples:
        real = real[rng.choice(real.shape[0], max_samples, replace=False)]
    if synth.shape[0] > max_samples:
        synth = synth[rng.choice(synth.shape[0], max_samples, replace=False)]

    def _sqdist(a, b):
        a2 = np.sum(a * a, axis=1, keepdims=True)
        b2 = np.sum(b * b, axis=1, keepdims=True)
        return a2 + b2.T - 2 * a @ b.T

    xx = _sqdist(real, real)
    yy = _sqdist(synth, synth)
    xy = _sqdist(real, synth)

    if sigma is None:
        all_d = np.concatenate([xx.ravel(), yy.ravel(), xy.ravel()])
        med = np.median(all_d[all_d > 0]) if np.any(all_d > 0) else 1.0
        sigma = float(np.sqrt(med / 2.0)) or 1.0
    gamma = 1.0 / (2.0 * sigma * sigma)

    kxx = np.exp(-gamma * xx)
    kyy = np.exp(-gamma * yy)
    kxy = np.exp(-gamma * xy)
    return float(kxx.mean() + kyy.mean() - 2 * kxy.mean())


def domain_classifier_accuracy(real: np.ndarray, synth: np.ndarray,
                               n_iters: int = 200, lr: float = 0.05) -> float:
    """Train a tiny logistic-regression classifier to tell real from synthetic.

    Accuracy near 0.5 means the synthetic distribution is indistinguishable
    from the real one (the project context's evaluation goal).  Accuracy
    near 1.0 means trivial separation — bad.

    We use NumPy only so this works in CPU-only test environments.
    """
    n_real = real.shape[0]
    n_synth = synth.shape[0]
    n = min(n_real, n_synth, 2000)
    rng = np.random.default_rng(0)
    r_idx = rng.choice(n_real, n, replace=False)
    s_idx = rng.choice(n_synth, n, replace=False)
    X = np.vstack([real[r_idx], synth[s_idx]]).astype(np.float64)
    y = np.concatenate([np.ones(n), np.zeros(n)])

    # Standardise features for stable training
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-8
    X = (X - mu) / sd
    X = np.hstack([X, np.ones((X.shape[0], 1))])  # bias

    # Shuffle and split
    perm = rng.permutation(X.shape[0])
    X, y = X[perm], y[perm]
    split = int(0.8 * X.shape[0])
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    w = np.zeros(X.shape[1])
    for _ in range(n_iters):
        z = X_tr @ w
        p = 1.0 / (1.0 + np.exp(-z))
        grad = X_tr.T @ (p - y_tr) / X_tr.shape[0]
        w -= lr * grad

    pred = (1.0 / (1.0 + np.exp(-(X_te @ w)))) > 0.5
    return float((pred == y_te).mean())


def umap_latent_visualisation(real: np.ndarray, synth: np.ndarray,
                              out_path: str) -> Optional[str]:
    """Save a UMAP scatter of real vs synthetic to ``out_path``.

    Quietly returns ``None`` if umap-learn or matplotlib are unavailable —
    the rest of the diagnostics still run.  This is the project context's
    "UMAP visualization of latent trajectories" requirement.
    """
    try:
        import umap  # type: ignore
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    n = min(real.shape[0], synth.shape[0], 1500)
    rng = np.random.default_rng(0)
    r = real[rng.choice(real.shape[0], n, replace=False)]
    s = synth[rng.choice(synth.shape[0], n, replace=False)]
    X = np.vstack([r, s])
    labels = np.array([0] * n + [1] * n)

    reducer = umap.UMAP(n_components=2, random_state=0)
    emb = reducer.fit_transform(X)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(emb[labels == 0, 0], emb[labels == 0, 1], s=8, alpha=0.5, label="real")
    ax.scatter(emb[labels == 1, 0], emb[labels == 1, 1], s=8, alpha=0.5, label="synthetic")
    ax.legend()
    ax.set_title("UMAP: real vs synthetic")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def evaluate_synthetic(real: np.ndarray, synth: np.ndarray,
                       umap_path: Optional[str] = None,
                       verbose: bool = True) -> dict:
    """Run the full evaluation suite from the project context.

    Returns a dict with keys: ks_mean, corr_dist, mmd, domain_acc, umap_path.
    """
    ncomp = min(real.shape[0], synth.shape[0])
    real_c = real[:ncomp]
    synth_c = synth[:ncomp]

    ks = marginal_ks_statistic(real_c, synth_c)
    cd = correlation_distance(real_c, synth_c)
    mmd = mmd_rbf(real_c, synth_c)
    dom = domain_classifier_accuracy(real_c, synth_c)
    up = umap_latent_visualisation(real_c, synth_c, umap_path) if umap_path else None

    if verbose:
        print("\n" + "=" * 60)
        print("EVALUATION (real vs synthetic)")
        print("=" * 60)
        print(f"  Mean KS:              {ks.mean():.4f}  (lower is better)")
        print(f"  Correlation distance: {cd:.4f}        (lower is better)")
        print(f"  MMD (RBF):            {mmd:.6f}     (lower is better)")
        print(f"  Domain classifier acc:{dom:.4f}        (closer to 0.5 is better)")
        if up:
            print(f"  UMAP plot: {up}")
        elif umap_path:
            print(f"  UMAP skipped (umap-learn or matplotlib not installed)")
        print("=" * 60)

    return {
        "ks_mean": float(ks.mean()),
        "corr_dist": float(cd),
        "mmd": float(mmd),
        "domain_acc": float(dom),
        "umap_path": up,
    }


def print_diagnostics(real: np.ndarray, synth: np.ndarray, verbose: bool = True):
    verbose_print(verbose, "\n" + "=" * 80)
    verbose_print(verbose, "DIAGNOSTICS: Real vs Synthetic Sensor Data")
    verbose_print(verbose, "=" * 80)
    
    ncomp = min(real.shape[0], synth.shape[0])
    real_comp = real[:ncomp]
    synth_comp = synth[:ncomp]
    
    ks = marginal_ks_statistic(real_comp, synth_comp)
    verbose_print(verbose, "\nKS Statistics per Sensor (lower is better, <0.1 is good):")
    for i, k in enumerate(ks):
        status = "✓" if k < 0.1 else "⚠" if k < 0.2 else "✗"
        verbose_print(verbose, f"  {status} Sensor {i+1}: KS={k:.4f}")
    
    avg_ks = np.mean(ks)
    verbose_print(verbose, f"\n  Average KS: {avg_ks:.4f}")
    
    corr_dist = correlation_distance(real_comp, synth_comp)
    verbose_print(verbose, f"\nCorrelation Distance: {corr_dist:.4f}")
    verbose_print(verbose, "=" * 80 + "\n")


def _flatten_config(cfg: dict) -> dict:
    if not isinstance(cfg, dict):
        return {}
    flat = {}
    for key in ("mode", "data", "model_name", "output_dir"):
        if key in cfg:
            flat[key] = cfg[key]
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model"), dict) else {}
    if "hidden_dim" in model_cfg:
        flat["hidden_dim"] = model_cfg["hidden_dim"]
    training = cfg.get("training", {}) if isinstance(cfg.get("training"), dict) else {}
    for key in ("epochs", "batch_size", "lr"):
        if key in training:
            flat[key] = training[key]
    diffusion = cfg.get("diffusion", {}) if isinstance(cfg.get("diffusion"), dict) else {}
    for key in ("T", "beta_start"):
        if key in diffusion:
            flat[key] = diffusion[key]
    generation = cfg.get("generation", {}) if isinstance(cfg.get("generation"), dict) else {}
    for key in ("n_engines", "cycles_per_engine"):
        if key in generation:
            flat[key] = generation[key]
    conditioning = cfg.get("conditioning", {}) if isinstance(cfg.get("conditioning"), dict) else {}
    if "n_conditions" in conditioning:
        flat["n_conditions"] = conditioning["n_conditions"]
    return flat


# ===========================================================================
# Transformer pipeline handlers (used when --arch transformer)
# ===========================================================================

def _build_transformer_data(data_path: str, n_conditions: int, verbose: bool):
    """Load CSV, drop NaNs, auto-resolve columns, standardise, return everything
    the transformer trainer needs.

    Returns: (raw_data, cols, col_indices, x_std, col_stats, sensor_stats,
              data_format, groups)
    """
    raw_data, cols = load_engine_data(data_path)
    verbose_print(verbose, f"Loaded {raw_data.shape[0]} samples from {data_path}")

    nan_mask = np.isnan(raw_data).any(axis=1)
    if nan_mask.any():
        verbose_print(verbose, f"Removing {nan_mask.sum()} NaN rows")
        raw_data = raw_data[~nan_mask]

    col_indices = auto_resolve_columns(raw_data, cols, n_conditions=n_conditions)
    data_format = detect_data_format(raw_data, col_indices)
    groups = group_rows_by_engine(raw_data, col_indices)

    verbose_print(verbose, f"  Format detected: {data_format}")
    verbose_print(verbose, f"  Engine groups:   {len(groups)}")
    verbose_print(verbose, f"  Metadata:        {[cols[i] for i in col_indices['metadata_idx']]}")
    verbose_print(verbose, f"  Targets:         {len(col_indices['sensor_idx']) + len(col_indices['condition_idx'])} columns")

    # Transformer treats sensors + conditions as one unified token sequence —
    # the model figures out itself which columns inform which.
    learn_idx = col_indices["sensor_idx"] + col_indices["condition_idx"]
    learn_data = raw_data[:, learn_idx]
    mean = learn_data.mean(axis=0)
    std = learn_data.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    x_std = ((learn_data - mean) / std).astype(np.float32)
    sensor_stats = {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}

    col_stats = compute_column_stats(x_std)
    return raw_data, cols, col_indices, x_std, col_stats, sensor_stats, data_format, groups


def _run_transformer_train(args, models_dir: Path, samples_dir: Path,
                           timestamp: str, model_hidden, logger):
    (raw_data, cols, col_indices, x_std, col_stats,
     sensor_stats, data_format, groups) = _build_transformer_data(
        args.data, args.n_conditions, args.verbose)

    n_cols = x_std.shape[1]
    hidden = args.hidden_dim if args.hidden_dim is not None else 128
    model = TabularTransformerDenoiser(
        n_cols=n_cols,
        hidden_dim=hidden,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
    )
    model.set_column_stats(col_stats)

    if args.checkpoint and Path(args.checkpoint).exists():
        verbose_print(args.verbose, f"Resuming from {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"])

    verbose_print(args.verbose, f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    schedule = DiffusionSchedule(T=args.T, beta_start=args.beta_start)
    dataset = TensorDataset(torch.from_numpy(x_std).float())
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    ckpt_meta = {
        "columns": cols,
        "col_indices": col_indices,
        "sensor_stats": sensor_stats,
        "column_stats": col_stats,
        "data_format": data_format,
        "model_cfg": {"hidden_dim": hidden, "n_heads": args.n_heads,
                      "n_layers": args.n_layers},
        # Kept for validate_checkpoint() backwards compat:
        "condition_stats": {"mean": np.array([], dtype=np.float32),
                            "std": np.array([], dtype=np.float32)},
    }

    start = time.time()
    train_transformer(model, schedule, loader,
                      epochs=args.epochs, lr=args.lr,
                      device=DEVICE, verbose=args.verbose,
                      save_dir=str(models_dir), ckpt_meta=ckpt_meta)
    verbose_print(args.verbose, f"Training finished in {time.time() - start:.1f}s")

    # Auto-sample after training
    if not args.no_auto_sample:
        verbose_print(args.verbose, "\nAUTO-GENERATING SAMPLES")
        n_samples = args.n_engines * args.cycles_per_engine
        synth_std = sample_inpaint(model, schedule, n_samples=n_samples,
                                   device=DEVICE, verbose=args.verbose)
        synth = synth_std * sensor_stats["std"] + sensor_stats["mean"]

        # Reconstruct full row in original column order
        out_path = samples_dir / (args.out or f"synthetic_{timestamp}.csv")
        _write_synth_csv(out_path, synth, cols, col_indices, args, data_format)
        verbose_print(args.verbose, f"Saved {n_samples} rows to {out_path}")
        logger.info("output_csv_path=%s", out_path)

        # Evaluation
        learn_idx = col_indices["sensor_idx"] + col_indices["condition_idx"]
        real = raw_data[:, learn_idx]
        evaluate_synthetic(real, synth, umap_path=str(samples_dir / f"umap_{timestamp}.png"),
                           verbose=args.verbose)


def _run_transformer_generate(args, config: dict, samples_dir: Path,
                              timestamp: str, model_hidden, logger):
    model_path = _select_checkpoint_path(args, config)
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    if ckpt.get("arch") != "transformer":
        raise RuntimeError(f"{model_path} is not a transformer checkpoint. "
                           f"Use --arch mlp to load it.")

    cols = ckpt["columns"]
    col_indices = ckpt["col_indices"]
    sensor_stats = ckpt["sensor_stats"]
    col_stats = np.asarray(ckpt["column_stats"], dtype=np.float32)
    data_format = ckpt.get("data_format", "snapshot")
    model_cfg = ckpt.get("model_cfg", {})

    n_cols = len(col_indices["sensor_idx"]) + len(col_indices["condition_idx"])
    model = TabularTransformerDenoiser(
        n_cols=n_cols,
        hidden_dim=model_cfg.get("hidden_dim", 128),
        n_heads=model_cfg.get("n_heads", 4),
        n_layers=model_cfg.get("n_layers", 4),
    )
    model.set_column_stats(col_stats)
    model.load_state_dict(ckpt["model_state"])

    # Optional adapter composition (LoRA)
    if args.adapters:
        verbose_print(args.verbose, "Composing LoRA adapters:")
        compose_adapters(model, args.adapters, verbose=args.verbose)

    schedule = DiffusionSchedule(T=int(ckpt["schedule"]["T"]))
    schedule.betas = torch.tensor(ckpt["schedule"]["betas"], device=DEVICE)
    schedule.alphas = 1.0 - schedule.betas
    schedule.alpha_bar = torch.cumprod(schedule.alphas, dim=0)
    schedule.sqrt_alpha_bar = torch.sqrt(schedule.alpha_bar)
    schedule.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - schedule.alpha_bar)

    # Optional inpainting: --observe picks which columns are copied from rows of --data and held fixed
    # while the model samples the rest. Default "none" samples every column. (Observing every column
    # would just return resampled real rows.)
    observed_values = None
    observed_mask = None
    n_samples = args.n_engines * args.cycles_per_engine
    learn_idx = col_indices["sensor_idx"] + col_indices["condition_idx"]
    observe = (getattr(args, "observe", None) or "none").strip()
    if observe == "none":
        observe_names = set()
    elif observe == "conditions":
        observe_names = {cols[i].strip() for i in col_indices["condition_idx"]}
    else:
        observe_names = {c.strip() for c in observe.split(",") if c.strip()}
        unknown = observe_names - {cols[i].strip() for i in learn_idx}
        if unknown:
            raise ValueError(f"--observe: unknown column(s) {sorted(unknown)}")
    if observe_names and not args.data:
        raise ValueError("--observe needs --data to take the observed values from")
    if args.data and observe_names:
        seed_raw, seed_cols = load_engine_data(args.data)
        seed_raw = seed_raw[~np.isnan(seed_raw).any(axis=1)]
        seed_cols_l = [c.strip() for c in seed_cols]
        ov = np.zeros((n_samples, n_cols), dtype=np.float32)
        om = np.zeros((n_samples, n_cols), dtype=np.float32)
        rng = np.random.default_rng(0)
        choose = rng.choice(seed_raw.shape[0], size=n_samples, replace=True)
        for col_pos, base_idx in enumerate(learn_idx):
            base_name = cols[base_idx].strip()
            if base_name in observe_names and base_name in seed_cols_l:
                j = seed_cols_l.index(base_name)
                mu = sensor_stats["mean"][col_pos]
                sd = sensor_stats["std"][col_pos]
                ov[:, col_pos] = (seed_raw[choose, j] - mu) / sd
                om[:, col_pos] = 1.0
        observed_values = torch.from_numpy(ov)
        observed_mask = torch.from_numpy(om)
        verbose_print(args.verbose, f"Inpainting from {int(om.sum() / n_samples)} observed columns: {sorted(observe_names)}")
    else:
        verbose_print(args.verbose, f"Sampling all {n_cols} columns (no observed columns)")

    synth_std = sample_inpaint(model, schedule, n_samples=n_samples,
                               observed_values=observed_values,
                               observed_mask=observed_mask,
                               device=DEVICE, verbose=args.verbose)
    synth = synth_std * sensor_stats["std"] + sensor_stats["mean"]

    out_path = samples_dir / (args.out or f"synthetic_{timestamp}.csv")
    _write_synth_csv(out_path, synth, cols, col_indices, args, data_format)
    verbose_print(args.verbose, f"Saved {n_samples} rows to {out_path}")
    logger.info("output_csv_path=%s", out_path)

    if args.data:
        seed_raw, seed_cols = load_engine_data(args.data)
        seed_raw = seed_raw[~np.isnan(seed_raw).any(axis=1)]
        seed_cols_l = [c.strip() for c in seed_cols]
        real = seed_raw[:, [seed_cols_l.index(cols[i].strip())
                            for i in learn_idx
                            if cols[i].strip() in seed_cols_l]]
        if real.shape[1] == synth.shape[1]:
            evaluate_synthetic(real, synth,
                               umap_path=str(samples_dir / f"umap_{timestamp}.png"),
                               verbose=args.verbose)


def _write_synth_csv(out_path: Path, synth: np.ndarray, cols: list,
                     col_indices: dict, args, data_format: str) -> None:
    """Write synthetic targets back into the original CSV column layout.

    Synthesises metadata (engine_id and time) appropriate to the detected
    format: continuous data gets per-engine cycle indices; snapshot data
    gets one row per engine.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_rows = synth.shape[0]
    full = np.zeros((n_rows, len(cols)), dtype=np.float32)

    learn_idx = col_indices["sensor_idx"] + col_indices["condition_idx"]
    for i, base_idx in enumerate(learn_idx):
        full[:, base_idx] = synth[:, i]

    meta_idx = col_indices["metadata_idx"]
    if meta_idx:
        if data_format == "continuous":
            engine_ids = np.repeat(np.arange(1, args.n_engines + 1),
                                   args.cycles_per_engine)
            time_vals = np.tile(np.arange(1, args.cycles_per_engine + 1),
                                args.n_engines)
        else:
            engine_ids = np.arange(1, n_rows + 1)
            time_vals = np.zeros(n_rows)
        full[:, meta_idx[0]] = engine_ids[:n_rows]
        if len(meta_idx) >= 2:
            full[:, meta_idx[1]] = time_vals[:n_rows]

    import csv
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(full)


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Conditional diffusion for tabular data")
    parser.add_argument("--mode", choices=["train", "generate", "fine-tune"], default="train")
    parser.add_argument("--arch", choices=["transformer", "mlp"], default="transformer",
                        help="Denoiser architecture. The default 'transformer' uses learned "
                             "column embeddings + masked self-attention, so the model itself "
                             "decides what to condition on.")
    parser.add_argument("--data", type=str, nargs="+", default=None,
                        help="Path(s) to one or more CSV data files. When multiple are "
                             "given, column headers must match (order may differ) and "
                             "rows are concatenated in the order supplied.")
    parser.add_argument("--seed-data", type=str, nargs="+", default=None,
                        help="One or more seed CSVs for fine-tune mode (subset of "
                             "training columns is OK). Multiple files are concatenated "
                             "by load_engine_data() — column names must match across "
                             "files (order may differ).")
    parser.add_argument("--adapter-out", type=str, default=None,
                        help="Where to save the LoRA adapter in fine-tune mode")
    parser.add_argument("--adapters", type=str, nargs="*", default=None,
                        help="LoRA adapters to compose at generation time")
    parser.add_argument("--observe", type=str, default="none",
                        help="Generate mode (transformer denoiser): columns to hold fixed at values from rows of "
                             "--data while the model samples the rest. 'none' (default) samples every column; "
                             "'conditions' observes the auto-selected condition columns; or a comma-separated "
                             "list of column names.")
    parser.add_argument("--model", type=str, default="diff_best_ckpt.pt", help="Model checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to load")
    parser.add_argument("--out", type=str, default=None, help="Output CSV file")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--model-name", type=str, default="diffusion")
    parser.add_argument("--config", type=str, default=None, help="Path to config file (.json/.yaml/.yml)")
    parser.add_argument("--n-conditions", type=int, default=3,
                        help="(Legacy MLP arch only) how many columns to pre-select for conditioning")
    parser.add_argument("--hidden-dim", type=int, default=None, help="Hidden dimension size")
    parser.add_argument("--n-heads", type=int, default=4, help="(Transformer) attention heads")
    parser.add_argument("--n-layers", type=int, default=4, help="(Transformer) encoder layers")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--lora-rank", type=int, default=4, help="LoRA rank for fine-tune mode")
    parser.add_argument("--lora-alpha", type=float, default=8.0, help="LoRA alpha")
    parser.add_argument("--n-engines", type=int, default=5, help="Number of groups to generate")
    parser.add_argument("--cycles-per-engine", type=int, default=100, help="Number of rows per group")
    parser.add_argument("--T", type=int, default=50, help="Diffusion timesteps")
    parser.add_argument("--beta-start", type=float, default=1e-4, help="Diffusion beta schedule start")
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--no-auto-sample", action="store_true", default=False)
    parser.add_argument("--print-checkpoint-only", action="store_true", default=False)
    args = parser.parse_args()

    config = _load_config(args.config)
    defaults = {a.dest: a.default for a in parser._actions if a.dest != "help"}
    config_data = _flatten_config(config)
    merged = {**defaults, **config_data}
    for action in parser._actions:
        if action.dest == "help":
            continue
        if any(opt in sys.argv for opt in action.option_strings):
            merged[action.dest] = getattr(args, action.dest)
    for key, value in merged.items():
        setattr(args, key, value)

    model_cfg = config.get("model", {}) if isinstance(config.get("model"), dict) else {}
    model_hidden = args.hidden_dim if args.hidden_dim is not None else model_cfg.get("hidden_dim", 256)

    verbose_print(args.verbose, "=" * 80)
    verbose_print(args.verbose, "Conditional Diffusion for Tabular Data")
    verbose_print(args.verbose, f"Device: {DEVICE}")
    verbose_print(args.verbose, "=" * 80)

    base_dir = Path(args.output_dir) / "diffusion" / args.model_name
    models_dir = Path("src") / "model_checkPoints" / "diffusion" / args.model_name
    logs_dir = base_dir / "logs"
    samples_dir = base_dir / "synthetic_data"
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"{args.mode}_{timestamp}.log"
    logger = logging.getLogger("diffusion")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    effective_config = {
        "mode": args.mode,
        "data": args.data,
        "model_name": args.model_name,
        "n_conditions": args.n_conditions,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "T": args.T,
        "beta_start": args.beta_start,
        "hidden_dim": model_hidden,
        "n_engines": args.n_engines,
        "cycles_per_engine": args.cycles_per_engine,
    }
    logger.info("effective_config=%s", effective_config)

    # ----- New transformer pipeline (default) -----
    if args.arch == "transformer":
        if args.mode == "fine-tune":
            assert args.checkpoint, "fine-tune requires --checkpoint (base model)"
            assert args.seed_data, "fine-tune requires --seed-data"
            if args.adapter_out:
                out = args.adapter_out
            else:
                adapters_dir = Path("src") / "model_checkPoints" / "diffusion" / "adapters"
                adapters_dir.mkdir(parents=True, exist_ok=True)
                out = str(adapters_dir / f"adapter_{timestamp}.pt")
            fine_tune_with_seed(
                base_ckpt_path=args.checkpoint,
                seed_data_path=args.seed_data,
                adapter_out_path=out,
                epochs=args.epochs,
                lr=args.lr if args.lr != 1e-3 else 1e-5,  # default to small LR
                rank=args.lora_rank,
                alpha=args.lora_alpha,
                batch_size=args.batch_size,
                verbose=args.verbose,
            )
            return

        if args.mode == "train":
            _run_transformer_train(args, models_dir, samples_dir, timestamp,
                                   model_hidden, logger)
            return
        if args.mode == "generate":
            _run_transformer_generate(args, config, samples_dir, timestamp,
                                      model_hidden, logger)
            return

    # ----- Legacy MLP pipeline (kept for old tests / checkpoints) -----
    if args.mode == "train":
        assert args.data is not None, "Training requires --data"

        # Load and prepare data
        raw_data, cols = load_engine_data(args.data)
        verbose_print(args.verbose, f"Loaded {raw_data.shape[0]} samples from {args.data}")

        # Remove NaNs before column auto-resolution (variance/correlation
        # would otherwise be polluted by missing values).
        nan_mask = np.isnan(raw_data).any(axis=1)
        if nan_mask.any():
            verbose_print(args.verbose, f"Removing {nan_mask.sum()} rows with NaNs")
            raw_data = raw_data[~nan_mask]

        # Auto-detect metadata + auto-select conditioning columns from the data
        col_indices = auto_resolve_columns(raw_data, cols, n_conditions=args.n_conditions)
        verbose_print(args.verbose, f"  Metadata columns:  {[cols[i] for i in col_indices['metadata_idx']]}")
        verbose_print(args.verbose, f"  Condition columns: {[cols[i] for i in col_indices['condition_idx']]} (auto-selected)")
        verbose_print(args.verbose, f"  Target columns:    {len(col_indices['sensor_idx'])} columns")

        # Prepare conditional data
        metadata, conditions_std, sensors_std, cond_stats, sensor_stats = prepare_conditional_data(raw_data, col_indices)
        sensor_dim = sensors_std.shape[1]
        condition_dim = conditions_std.shape[1]
        verbose_print(args.verbose, f"Conditions shape: {conditions_std.shape}")
        verbose_print(args.verbose, f"Sensors shape: {sensors_std.shape}")

        # Create dataset
        if condition_dim > 0:
            cond_tensor = torch.from_numpy(conditions_std).float()
            sensor_tensor = torch.from_numpy(sensors_std).float()
            dataset = TensorDataset(cond_tensor, sensor_tensor)
        else:
            sensor_tensor = torch.from_numpy(sensors_std).float()
            # Unconditional: use a dummy zero-width condition tensor so the
            # dataloader still yields (conditions, sensors) tuples.
            dummy_cond = torch.zeros(sensor_tensor.shape[0], 0)
            dataset = TensorDataset(dummy_cond, sensor_tensor)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        # Build model
        schedule = DiffusionSchedule(T=args.T, beta_start=args.beta_start)
        model = ConditionalDenoiser(sensor_dim=sensor_dim, condition_dim=condition_dim,
                                    hidden_dim=model_hidden, time_emb_dim=128, n_layers=4)

        if args.checkpoint:
            ckpt_path = Path(args.checkpoint)
            if ckpt_path.exists():
                verbose_print(args.verbose, f"Loading checkpoint from {ckpt_path} for transfer training...")
                ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                model.load_state_dict(ckpt["model_state"])
            else:
                verbose_print(args.verbose, f"Warning: Checkpoint {ckpt_path} not found. Starting from scratch.")

        verbose_print(args.verbose, f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Train
        start = time.time()
        model_path = models_dir / args.model
        output_path = None
        logger.info("mode=%s", args.mode)
        logger.info("data_path=%s", args.data)
        logger.info("base_dir=%s", base_dir)
        logger.info("checkpoint_path=%s", model_path)
        trained = train(model, schedule, loader, epochs=args.epochs, lr=args.lr,
                   device=DEVICE, verbose=args.verbose, save_dir=str(models_dir),
                   cond_stats=cond_stats, sensor_stats=sensor_stats, cols=cols,
                   col_indices=col_indices)
        elapsed = time.time() - start
        verbose_print(args.verbose, f"Training finished in {elapsed:.1f}s")

        # Save checkpoint with stats and column layout
        ckpt = {
            "model_state": trained.state_dict(),
            "schedule": {"T": schedule.T, "betas": schedule.betas.cpu().numpy()},
            "condition_stats": cond_stats,
            "sensor_stats": sensor_stats,
            "columns": cols,
            "col_indices": col_indices,
        }
        torch.save(ckpt, model_path)
        verbose_print(args.verbose, f"Saved checkpoint to {model_path}")

        # Auto-sample
        if not args.no_auto_sample:
            verbose_print(args.verbose, "\n" + "=" * 80)
            verbose_print(args.verbose, "AUTO-GENERATING SAMPLES")
            verbose_print(args.verbose, "=" * 80)

            n_meta = len(col_indices["metadata_idx"])
            metadata_gen, conditions_gen, sensors_gen = generate_engines_with_cycles(
                trained, schedule, args.n_engines, args.cycles_per_engine,
                cond_stats, sensor_stats, n_metadata_cols=n_meta,
                conditions_ref=conditions_std,
                device=DEVICE, verbose=args.verbose
            )

            # Save
            output_name = args.out if args.out else f"synthetic_engine_data_{timestamp}.csv"
            output_path = samples_dir / output_name
            save_engine_data(metadata_gen, conditions_gen, sensors_gen,
                             str(output_path), cols, col_indices)
            verbose_print(args.verbose, f"Saved {len(metadata_gen)} samples to {output_path}")
            verbose_print(args.verbose, f"  ({args.n_engines} groups × {args.cycles_per_engine} rows)")
            logger.info("output_csv_path=%s", output_path)

            # Diagnostics
            real_sensors = raw_data[:len(sensors_gen)][:, col_indices["sensor_idx"]]
            print_diagnostics(real_sensors, sensors_gen, verbose=args.verbose)

    elif args.mode == "generate":
        model_path = _select_checkpoint_path(args, config)
        logger.info("checkpoint_path=%s", model_path)
        if args.print_checkpoint_only:
            return
        output_path = None
        logger.info("mode=%s", args.mode)
        logger.info("data_path=%s", args.data)
        logger.info("base_dir=%s", base_dir)
        logger.info("checkpoint_path=%s", model_path)
        try:
            ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
        except Exception as exc:
            raise RuntimeError(f"Failed to load checkpoint: {model_path}") from exc
        cond_stats = ckpt["condition_stats"]
        sensor_stats = ckpt["sensor_stats"]
        cols = ckpt["columns"]
        col_indices = ckpt.get("col_indices")
        sensor_dim = len(sensor_stats["mean"])
        condition_dim = len(cond_stats["mean"])

        # Old checkpoints (pre auto-conditioning) cannot be safely loaded — the
        # original column-role mapping is unknown and would need to be guessed.
        if col_indices is None:
            raise RuntimeError(
                f"Checkpoint {model_path} was trained before auto-conditioning "
                "and does not contain a 'col_indices' mapping. Please retrain."
            )

        validate_checkpoint(ckpt, sensor_dim=sensor_dim, condition_dim=condition_dim)
        logger.info("checkpoint_validation=passed")

        # Build schedule and model
        T = ckpt["schedule"]["T"]
        schedule = DiffusionSchedule(T=T)
        schedule.betas = torch.tensor(ckpt["schedule"]["betas"], device=DEVICE)
        schedule.alphas = 1.0 - schedule.betas
        schedule.alpha_bar = torch.cumprod(schedule.alphas, dim=0)
        schedule.sqrt_alpha_bar = torch.sqrt(schedule.alpha_bar)
        schedule.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - schedule.alpha_bar)

        model = ConditionalDenoiser(sensor_dim=sensor_dim, condition_dim=condition_dim,
                                    hidden_dim=model_hidden)
        model.load_state_dict(ckpt["model_state"])

        verbose_print(args.verbose, f"Loaded model from {model_path}")

        # Get reference conditions if data provided
        conditions_ref = None
        if args.data and condition_dim > 0:
            raw_data, _ = load_engine_data(args.data)
            raw_data = raw_data[~np.isnan(raw_data).any(axis=1)]
            _, conditions_ref, _, _, _ = prepare_conditional_data(raw_data, col_indices)

        # Generate
        n_meta = len(col_indices["metadata_idx"])
        metadata_gen, conditions_gen, sensors_gen = generate_engines_with_cycles(
            model, schedule, args.n_engines, args.cycles_per_engine,
            cond_stats, sensor_stats, n_metadata_cols=n_meta,
            conditions_ref=conditions_ref,
            device=DEVICE, verbose=args.verbose
        )

        # Save
        output_name = args.out if args.out else f"synthetic_engine_data_{timestamp}.csv"
        output_path = samples_dir / output_name
        save_engine_data(metadata_gen, conditions_gen, sensors_gen,
                         str(output_path), cols, col_indices)
        verbose_print(args.verbose, f"Saved {len(metadata_gen)} samples to {output_path}")
        verbose_print(args.verbose, f"  ({args.n_engines} groups × {args.cycles_per_engine} rows)")
        logger.info("output_csv_path=%s", output_path)

        # Diagnostics if real data provided
        if args.data:
            raw_data_diag, _ = load_engine_data(args.data)
            raw_data_diag = raw_data_diag[~np.isnan(raw_data_diag).any(axis=1)]
            real_sensors = raw_data_diag[:len(sensors_gen)][:, col_indices["sensor_idx"]]
            print_diagnostics(real_sensors, sensors_gen, verbose=args.verbose)

    elif args.mode == "inspect":
        assert args.data is not None, "Inspect requires --data"
        raw_data, cols = load_engine_data(args.data)
        raw_data = raw_data[~np.isnan(raw_data).any(axis=1)]
        col_indices = auto_resolve_columns(raw_data, cols, n_conditions=args.n_conditions)

        verbose_print(args.verbose, f"Data shape: {raw_data.shape}")
        verbose_print(args.verbose, f"Columns: {cols}")

        col_stripped = [c.strip() for c in cols]

        # Metadata summary
        if col_indices["metadata_idx"]:
            first_meta = col_indices["metadata_idx"][0]
            meta_vals = raw_data[:, first_meta].astype(int)
            unique_vals = np.unique(meta_vals)
            verbose_print(args.verbose, f"\nUnique values in '{col_stripped[first_meta]}': {len(unique_vals)}")
            for val in unique_vals[:10]:
                count = np.sum(meta_vals == val)
                verbose_print(args.verbose, f"  {val}: {count} rows")
            if len(unique_vals) > 10:
                verbose_print(args.verbose, f"  ... and {len(unique_vals) - 10} more")

        # Condition ranges
        if col_indices["condition_idx"]:
            conditions = raw_data[:, col_indices["condition_idx"]]
            verbose_print(args.verbose, "\nCondition ranges:")
            for i, ci in enumerate(col_indices["condition_idx"]):
                name = col_stripped[ci]
                verbose_print(args.verbose, f"  {name}: [{conditions[:,i].min():.4f}, {conditions[:,i].max():.4f}]")

        # Sensor/target statistics
        sensors = raw_data[:, col_indices["sensor_idx"]]
        verbose_print(args.verbose, f"\nTarget statistics ({sensors.shape[1]} columns):")
        for i in range(min(5, sensors.shape[1])):
            col_name = col_stripped[col_indices["sensor_idx"][i]]
            verbose_print(args.verbose, f"  {col_name}: mean={sensors[:,i].mean():.2f}, std={sensors[:,i].std():.2f}")


if __name__ == "__main__":
    main()