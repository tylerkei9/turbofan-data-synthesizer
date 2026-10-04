# TRAINING + GENERATION: Generic Sequence Transformer Synthesizer
# File: src/training/Transformer/Transformer_TrainingV7.py
import os
import math
import time
import random
import logging
import csv
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, TYPE_CHECKING
import torch.optim as optim
import copy 

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import yaml
import argparse

# Type checking imports (for Pylance)
if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

# -------------------------
# Paths / Defaults
# -------------------------
# TensorBoard imports (optional dependency)
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


# -------------------------
# Paths / Defaults
# -------------------------
SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent               # .../src/training/Transformer
PROJECT_ROOT = SCRIPT_DIR.parents[2]          # repo root

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "src" / "config" / "transformer_config.yaml"
CHECKPOINT_DIR = PROJECT_ROOT / "src" / "model_checkPoints" / "transformer"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def _resolve_path(p):
    p = Path(p)
    return p if p.is_absolute() else (PROJECT_ROOT / p)

def load_config(cfg_path):
    cfg_path = _resolve_path(cfg_path)
    if not cfg_path.exists():
        raise FileNotFoundError("Config not found: {}".format(cfg_path))
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

# -------------------------
# Logging
# -------------------------
def setup_logging(log_cfg: dict, output_dir: Path) -> logging.Logger:
    output_dir = _resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("TransformerTraining")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    # Console
    console_level = getattr(logging, str(log_cfg.get("console_level", "INFO")).upper(), logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S")
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    # File
    file_level = getattr(logging, str(log_cfg.get("file_level", "DEBUG")).upper(), logging.DEBUG)
    log_file = output_dir/"logs" / "training_{}.log".format(datetime.now().strftime("%Y%m%d_%H%M%S"))
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(file_level)
    file_fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)

    logger.info("Log file: {}".format(log_file))
    return logger
    

# -------------------------
# Model
# -------------------------
class PositionalEncoding(nn.Module):
    pe: torch.Tensor  # Type annotation for registered buffer
    
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)  # [max_len, d_model]

    def forward(self, x):
        # x: [S, B, d_model]
        S = x.size(0)
        return x + self.pe[:S].unsqueeze(1)


class StrongTimeTransformer(nn.Module):
    def __init__(
        self,
        src_dim,
        out_dim,
        n_engines,
        d_model=256,
        nhead=8,
        num_layers=4,
        d_ff=512,
        d_eng=4,
        dropout=0.15,
    ):
        super().__init__()

        self.src_dim = src_dim
        self.out_dim = out_dim

        self.eng_emb = nn.Embedding(n_engines, d_eng)
        self.eng_dropout = nn.Dropout(0.5)

        self.input = nn.Linear(src_dim + d_eng, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=False,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.pos = PositionalEncoding(d_model)
        self.norm = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, out_dim),
        )

    def forward(self, src, eng_idx):
        """
        src: [B, S, src_dim]
        eng_idx: [B] long
        returns: [B, out_dim]
        """
        B, S, F = src.shape
        e = self.eng_dropout(self.eng_emb(eng_idx))
        e = e.unsqueeze(1).expand(B, S, e.size(-1))
        x = torch.cat([src, e], dim=-1)
        x = self.input(x).transpose(0, 1)
        x = self.pos(x)
        x = self.encoder(x)
        x = self.norm(x[-1])
        return self.head(x)



# -------------------------
# Data / Training utilities
# -------------------------
def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _prepare_single_dataframe(df, file_idx, source_name, logger=None):
    """
      - single long recordings: treated as generic single-sequence handling (does NOT depend on column names, fixed number of columns, fixed feature order)
      - NASA-style files: multi-sequence handling (at the very least, has to assume first column is engine id)
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # normalize column names for matching
    col_map = {c.lower(): c for c in df.columns}

    has_engine_id = "engine_id" in col_map

    if has_engine_id:
        print("Multisequence file detected (engine_id column found). Using multi-sequence preparation.")
        original_engine_col = col_map["engine_id"]
        
        # FIND AND EXCLUDE THE TIME COLUMN TOO
        original_time_col = None
        for candidate in ["time_in_cycles", "cycle", "time", "timestamp"]:
            if candidate in col_map:
                original_time_col = col_map[candidate]
                break
        
        eng_col = "__sequence_id__"
        time_col = "__row_idx__"

        raw_engine = pd.to_numeric(df[original_engine_col], errors="coerce")

        if raw_engine.notna().sum() == 0:
            raise RuntimeError("engine_id column exists but could not be parsed as numeric.")

        df[eng_col] = raw_engine.apply(
            lambda x: "{}::eng_{}".format(source_name, int(x)) if pd.notna(x) else np.nan
        )

        df[time_col] = df.groupby(eng_col, sort=False).cumcount().astype(np.int64)

        feature_cols = []
        for c in df.columns:
            # EXCLUDE BOTH ENGINE_ID AND TIME_IN_CYCLES
            if c in {"Source_File", original_engine_col, original_time_col, eng_col, time_col}:
                continue
            numeric = pd.to_numeric(df[c], errors="coerce")
            if numeric.notna().sum() > 0:
                df[c] = numeric
                feature_cols.append(c)

        mode_msg = "engine_id_column_detected"

    else:
        # SINGLE SEQUENCE MODE
        print("No engine_id column detected. Treating entire file as one sequence with synthetic engine_id={}.".format(file_idx))
        eng_col = "__sequence_id__"
        time_col = "__row_idx__"

        df[eng_col] = file_idx
        df[time_col] = np.arange(len(df), dtype=np.int64)

        # Detect and exclude any explicit time/index column so it is not
        # treated as a feature.  We use the same candidate list as the
        # multi-sequence path so behaviour is consistent across formats.
        original_time_col = None
        for candidate in ["time_in_cycles", "cycle", "time", "timestamp"]:
            if candidate in col_map:
                original_time_col = col_map[candidate]
                break

        feature_cols = []
        for c in df.columns:
            if c in {"Source_File", eng_col, time_col, original_time_col}:
                continue
            numeric = pd.to_numeric(df[c], errors="coerce")
            if numeric.notna().sum() > 0:
                df[c] = numeric
                feature_cols.append(c)

        mode_msg = "single_sequence_no_engine_id"

    if not feature_cols:
        raise RuntimeError("No numeric feature columns found in input file.")
    if logger:
        logger.info(
            "Prepared file {} | source={} | mode={} | sequences_col={} | time_col={} | numeric_features={}".format(
                file_idx, source_name, mode_msg, eng_col, time_col, len(feature_cols)
            )
        )
    return df, eng_col, time_col, feature_cols

def load_and_prepare_dataframe(
    cfg,
    logger=None
) -> Tuple[pd.DataFrame, str, str, list, pd.Series, pd.Series]:
    """
    Notes:
        - Makes no assumptions about semantic column roles.
        - Each input file is treated as one generic sequence.
        - Row order defines sequence order.
        - All numeric columns are treated as candidate features.
        - Supports missing feature values caused by asynchronous sampling.
        - Column order from first file is preserved in output.
    """
    io = cfg.get("io", {})

    data_paths = [_resolve_path(p) for p in io.get("data_paths", [])]
    if not data_paths:
        raise ValueError("io.data_paths is empty in the YAML config.")

    prepared = []
    feature_cols_ordered = []  # Preserve order from first file
    all_feature_names = set()
    missing = []

    for file_idx, p in enumerate(data_paths):
        if not p.exists():
            missing.append(p)
            continue

        df = pd.read_csv(p)
        df["Source_File"] = p.name
        if logger:
            logger.info("Loaded {} {}".format(p.name, df.shape))
        else:
            print("Loaded {} {}".format(p, df.shape))

        df_prep, eng_col_local, time_col_local, local_features = _prepare_single_dataframe(
            df=df,
            file_idx=file_idx,
            source_name=p.name,
            logger=logger,
        )
        prepared.append((df_prep, eng_col_local, time_col_local, local_features))
        all_feature_names.update(local_features)
        
        # Preserve column order from first file
        if file_idx == 0:
            feature_cols_ordered = list(local_features)

    if missing:
        raise FileNotFoundError("Missing dataset files:\n" + "\n".join(str(p) for p in missing))
    if not prepared:
        raise RuntimeError("No datasets were loaded.")

    # Add any additional columns from subsequent files (preserving their order)
    for df_prep, eng_col_local, time_col_local, local_features in prepared:
        for feat in local_features:
            if feat not in feature_cols_ordered:
                feature_cols_ordered.append(feat)

    eng_col = "__sequence_id__"
    time_col = "__row_idx__"
    feature_cols = feature_cols_ordered  # Use preserved order, NOT sorted

    aligned = []
    for df_prep, eng_col_local, time_col_local, _ in prepared:
        base_df = pd.DataFrame({
            eng_col: df_prep[eng_col_local],
            time_col: pd.to_numeric(df_prep[time_col_local], errors="coerce"),
        }, index=df_prep.index)

        feature_df = df_prep.reindex(columns=feature_cols)
        feature_df = feature_df.apply(pd.to_numeric, errors="coerce")

        out = pd.concat([base_df, feature_df], axis=1).copy()
        aligned.append(out)

    df_all = pd.concat(aligned, ignore_index=True)

    if df_all[time_col].isna().any():
        df_all[time_col] = np.arange(len(df_all), dtype=np.float64)

    df_all = df_all.sort_values([eng_col, time_col]).reset_index(drop=True)

    if not feature_cols:
        raise RuntimeError("No numeric feature columns found.")

    msg = "Using {} features: {}{}".format(
        len(feature_cols),
        feature_cols[:12],
        " ..." if len(feature_cols) > 12 else ""
    )
    if logger:
        logger.info(msg)
    else:
        print(msg)

    work = df_all[[eng_col, time_col] + feature_cols].copy()

    mask_cols = [c + "_mask" for c in feature_cols]
    mask_df = work[feature_cols].notna().astype(np.float32)
    mask_df.columns = mask_cols

    col_mins = work[feature_cols].min(skipna=True)
    col_maxs = work[feature_cols].max(skipna=True)

    for c in feature_cols:
        if pd.isna(col_mins[c]) or pd.isna(col_maxs[c]):
            col_mins[c] = 0.0
            col_maxs[c] = 0.0

    ranges = (col_maxs - col_mins).replace(0, 1.0)

    work_scaled = work.copy()
    work_scaled[feature_cols] = 2.0 * ((work_scaled[feature_cols] - col_mins) / ranges) - 1.0
    work_scaled[feature_cols] = (
        work_scaled
        .groupby(eng_col, sort=False)[feature_cols]
        .transform(lambda s: s.ffill().bfill())
    )
    work_scaled[feature_cols] = work_scaled[feature_cols].fillna(-1.0)

    work_scaled = pd.concat([work_scaled, mask_df], axis=1)

    return work_scaled, eng_col, time_col, feature_cols, col_mins, ranges

def _impute_group_features(g, feature_cols):
    feat = g[feature_cols].copy()
    feat = feat.apply(pd.to_numeric, errors="coerce")
    mask = feat.notna().astype(np.float32)

    feat = feat.ffill().bfill()
    feat = feat.fillna(0.0)

    g_out = g.copy()
    for c in feature_cols:
        g_out[c] = feat[c].astype(np.float32)
        g_out["{}_mask".format(c)] = mask[c].astype(np.float32)
    return g_out


def build_sequence_groups(work_scaled, eng_col, time_col, input_window, pred_window):
    groups = []
    for seq_id, g in work_scaled.groupby(eng_col, sort=False):
        g = g.sort_values(time_col).reset_index(drop=True)
        g = _impute_group_features(
            g,
            [
                c for c in g.columns
                if c not in {eng_col, time_col} and not c.endswith("_mask")
            ],
        )
        if len(g) < (input_window + pred_window):
            continue

        mask_cols = [c for c in g.columns if c.endswith("_mask")]
        valid_target_steps = g[mask_cols].sum(axis=1)
        if valid_target_steps.sum() > 0:
            groups.append((seq_id, g))
    return groups


def split_groups(groups, seed, train_frac=0.9):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(groups))
    cut = int(train_frac * len(groups))
    train_groups = [groups[i] for i in perm[:cut]]
    val_groups = [groups[i] for i in perm[cut:]]
    return train_groups, val_groups

class EngineSeqDataset(Dataset):
    def __init__(self, groups, feature_cols, eng_to_idx, in_len, out_len=1):
        self.feature_cols = feature_cols
        self.mask_cols = ["{}_mask".format(c) for c in feature_cols]
        self.eng_to_idx = eng_to_idx
        self.in_len = in_len
        self.out_len = out_len

        self.groups = []
        self.rows = []

        for eng, g in groups:
            g = g.reset_index(drop=True).copy()
            val_arr = g[self.feature_cols].to_numpy(dtype=np.float32)
            mask_arr = g[self.mask_cols].to_numpy(dtype=np.float32)

            gi = len(self.groups)
            self.groups.append((eng, val_arr, mask_arr))

            L = len(g)
            max_i = L - (in_len + out_len) + 1
            for i in range(max_i):
                tgt_mask = mask_arr[i + in_len]
                if float(tgt_mask.sum()) > 0.0:
                    self.rows.append((gi, i))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        gi, i0 = self.rows[idx]
        eng, val_arr, mask_arr = self.groups[gi]

        src_vals = val_arr[i0:i0 + self.in_len]
        src_masks = mask_arr[i0:i0 + self.in_len]
        src = np.concatenate([src_vals, src_masks], axis=1)

        tgt = val_arr[i0 + self.in_len]
        tgt_mask = mask_arr[i0 + self.in_len]

        eng_idx = self.eng_to_idx[eng]
        return (
            torch.from_numpy(src),
            torch.from_numpy(tgt),
            torch.from_numpy(tgt_mask),
            torch.tensor(eng_idx, dtype=torch.long),
        )


def build_dataloaders(
    train_groups,
    val_groups,
    feature_cols,
    eng_to_idx,
    input_window,
    pred_window,
    batch_size,
    num_workers,
    device,
):
    train_ds = EngineSeqDataset(train_groups, feature_cols, eng_to_idx, input_window, pred_window)
    val_ds = EngineSeqDataset(val_groups, feature_cols, eng_to_idx, input_window, pred_window)

    pin = (device.type == "cuda")
    persistent = (num_workers > 0)

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=pin,
        num_workers=num_workers,
        persistent_workers=persistent,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=pin,
        num_workers=num_workers,
        persistent_workers=persistent,
    )
    return train_ds, val_ds, train_dl, val_dl


def build_model_from_cfg(cfg, src_dim, out_dim, n_engines):
    runtime = cfg.get("runtime", {})
    d_model = int(runtime.get("d_model", 256))
    nhead = int(runtime.get("nhead", 8))
    num_layers = int(runtime.get("num_layers", 4))
    d_ff = int(runtime.get("d_ff", 512))
    d_eng = int(runtime.get("d_eng", 4))
    dropout = float(runtime.get("dropout", 0.15))

    return StrongTimeTransformer(
        src_dim=src_dim,
        out_dim=out_dim,
        n_engines=n_engines,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        d_ff=d_ff,
        d_eng=d_eng,
        dropout=dropout,
    )





def run_epoch(model, dl, opt, scaler, device, lambda_var, train=True):
    """
    Returns dict with mean losses:
      - base:  masked L1 loss
      - var:   variance matching penalty (std of predictions vs std of targets)
      - total: base + lambda_var * var
    """
    model.train(train)
    base_sum, var_sum, total_sum, n = 0.0, 0.0, 0.0, 0

    for src, tgt, tgt_mask, eng_idx in dl:
        src = src.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        tgt_mask = tgt_mask.to(device, non_blocking=True)
        eng_idx = eng_idx.to(device, non_blocking=True)

        if train:
            opt.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
            pred = model(src, eng_idx)

            elem_loss = torch.abs(pred - tgt)
            masked_loss = elem_loss * tgt_mask
            denom = torch.clamp(tgt_mask.sum(), min=1.0)
            base_loss = masked_loss.sum() / denom

            # Variance penalty: encourage predictions to match the spread of
            # targets across the batch, controlled by lambda_var in config.
            if src.size(0) > 1:
                valid_cols = tgt_mask.sum(dim=0) > 0
                if bool(valid_cols.any()):
                    pred_std = pred[:, valid_cols].float().std(dim=0, unbiased=False)
                    tgt_std  = tgt[:, valid_cols].float().std(dim=0, unbiased=False)
                    var_loss = torch.mean(torch.abs(pred_std - tgt_std))
                else:
                    var_loss = torch.tensor(0.0, device=device, dtype=base_loss.dtype)
            else:
                var_loss = torch.tensor(0.0, device=device, dtype=base_loss.dtype)

            total_loss = base_loss + lambda_var * var_loss

        if train:
            scaler.scale(total_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

        bs = int(src.size(0))
        base_sum  += float(base_loss.item())  * bs
        var_sum   += float(var_loss.item())   * bs
        total_sum += float(total_loss.item()) * bs
        n += bs

    denom = max(1, n)
    return {
        "base":  base_sum  / denom,
        "var":   var_sum   / denom,
        "total": total_sum / denom,
    }


def save_training_curve(outdir, train_hist, val_hist):
    outdir = _resolve_path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.plot(train_hist, label="train")
    plt.plot(val_hist, label="val")
    plt.xlabel("epoch")
    plt.ylabel("Total loss")
    plt.title("Training")
    plt.legend()
    plt.tight_layout()

    curve_path = outdir / "training_curve.png"
    plt.savefig(curve_path, dpi=220)
    plt.close()
    return curve_path


def save_packed_model_checkpoint(
    path,
    model,
    feature_cols,
    eng_ids,
    eng_to_idx,
    col_mins,
    ranges,
    eng_col,
    time_col,
    input_window,
    seed,
    cfg,
    config_path,
    work_scaled=None,
):
    seed_windows = {}
    sequence_lengths = {}
    full_sequences = {}
    if work_scaled is not None:
        feat_only = list(feature_cols)
        for eng_id in work_scaled[eng_col].unique():
            grp = (
                work_scaled[work_scaled[eng_col] == eng_id]
                .sort_values(time_col)
                .reset_index(drop=True)
            )
            vals = grp[feat_only].values  # already in [-1, 1]
            sequence_lengths[str(eng_id)] = int(len(vals))
            if len(vals) >= input_window:
                seed_windows[str(eng_id)] = vals[:input_window].astype(np.float32)
            # Store the full normalised sequence so generation can sample
            # random re-seed windows to prevent autoregressive collapse.
            if len(vals) > input_window:
                full_sequences[str(eng_id)] = vals.astype(np.float32)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_cols": feature_cols,
            "eng_ids": eng_ids,
            "eng_to_idx": eng_to_idx,
            "col_mins": col_mins,
            "ranges": ranges,
            "eng_col": eng_col,
            "time_col": time_col,
            "INPUT_WINDOW": input_window,
            "SEED": seed,
            "src_dim": len(feature_cols) * 2,
            "out_dim": len(feature_cols),
            "config_path": str(_resolve_path(config_path)),
            "config": cfg,
            "seed_windows": seed_windows,
            "sequence_lengths": sequence_lengths,
            "full_sequences": full_sequences,
        },
        path,
    )
    return path


def generate_from_seed_file(
    base_checkpoint_path,
    seed_csv_path,
    config_path,
    num_samples=500,
    adaptation_epochs=5,
    adapt_lr=1e-4,
    reseed_interval=0,
    device=None,
    verbose=True,
):
    """
    Args:
        base_checkpoint_path : path to the trained transformer_model.pt
        seed_csv_path        : path to the seed CSV (any supported format)
        config_path          : path to transformer_config.yaml
        num_samples          : number of rows to generate
        adaptation_epochs    : fine-tuning epochs on seed data (5-10 recommended;
                               more risks catastrophic forgetting of base physics)
        adapt_lr             : learning rate for fine-tuning (keep well below
                               original training lr to avoid destroying base weights)
        reseed_interval      : re-inject a real seed window every N generation steps
                               to prevent autoregressive collapse (0 = disabled).
                               Recommended for long single-sequence seed files.
        device               : torch device (auto-detected if None)
        verbose              : print progress
 
    Returns:
        pd.DataFrame with the same column structure as the seed file.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    if verbose:
        print("=" * 60)
        print("Seeded Generation")
        print(f"  Base model : {Path(base_checkpoint_path).name}")
        print(f"  Seed file  : {Path(seed_csv_path).name}")
        print(f"  Adapt epochs: {adaptation_epochs}  lr: {adapt_lr}")
        print(f"  Generate   : {num_samples} steps")
        print("=" * 60)
 
    # load checkpoint
    ckpt = torch.load(_resolve_path(base_checkpoint_path), map_location=device, weights_only=False)
    cfg  = load_config(config_path)
 
    feature_cols  = ckpt["feature_cols"]
    F             = len(feature_cols)
    src_dim       = ckpt.get("src_dim", 2 * F)
    input_window  = ckpt.get("INPUT_WINDOW", cfg.get("runtime", {}).get("input_window", 32))
    pred_window   = cfg.get("runtime", {}).get("pred_window", 1)
    lambda_var    = float(cfg.get("runtime", {}).get("lambda_var", 0.01))
 
    col_mins = ckpt["col_mins"]
    ranges   = ckpt["ranges"]
 
    emb_w          = ckpt["model_state_dict"]["eng_emb.weight"]
    n_engines_vocab = int(emb_w.shape[0])
 
    # Work on a deep copy so the base checkpoint is never modified.
    model = build_model_from_cfg(cfg, src_dim=src_dim, out_dim=F, n_engines=n_engines_vocab).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model = copy.deepcopy(model)
 
    # Prepare seed data
    seed_path_resolved = _resolve_path(seed_csv_path)
    df_seed_raw = pd.read_csv(seed_path_resolved)
    df_seed_raw["Source_File"] = seed_path_resolved.name
 
    df_seed_prep, eng_col, time_col, _ = _prepare_single_dataframe(
        df=df_seed_raw,
        file_idx=0,
        source_name=seed_path_resolved.name,
    )
 
    work_seed = df_seed_prep.reindex(columns=[eng_col, time_col] + feature_cols).copy()
    work_seed[feature_cols] = work_seed[feature_cols].apply(pd.to_numeric, errors="coerce")
 
    mask_df_seed = work_seed[feature_cols].notna().astype(np.float32)
    mask_df_seed.columns = [c + "_mask" for c in feature_cols]
 
    # Normalize using base model stats so inputs are in the model's native space.
    work_seed[feature_cols] = 2.0 * ((work_seed[feature_cols] - col_mins) / ranges) - 1.0
 
    # Forward-fill within each sequence (handles asynchronous sampling) then
    # fall back to -1.0 for columns with no data at all.
    work_seed[feature_cols] = (
        work_seed
        .groupby(eng_col, sort=False)[feature_cols]
        .transform(lambda s: s.ffill().bfill())
    )
    work_seed[feature_cols] = work_seed[feature_cols].fillna(-1.0)
    work_seed = pd.concat([work_seed, mask_df_seed], axis=1)
 
    groups = build_sequence_groups(work_seed, eng_col, time_col, input_window, pred_window)
    if not groups:
        raise RuntimeError(
            "Seed file '{}' does not have enough rows to form a sequence "
            "(need at least {} rows).".format(seed_path_resolved.name, input_window + pred_window)
        )
 
    # Map seed engines into the existing embedding vocabulary via modulo so we
    # never exceed the trained embedding table size.
    seed_eng_to_idx = {e: (i % n_engines_vocab) for i, (e, _) in enumerate(groups)}
 
    if verbose:
        print(f"  Seed sequences found: {len(groups)}")
 
    # Fine tune on seeded data 
    adapt_ds = EngineSeqDataset(groups, feature_cols, seed_eng_to_idx, input_window, pred_window)
    adapt_dl = DataLoader(adapt_ds, batch_size=min(64, len(adapt_ds)), shuffle=True, drop_last=False)
 
    opt    = optim.AdamW(model.parameters(), lr=adapt_lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    model.train()
 
    if verbose:
        print(f"  Fine-tuning for {adaptation_epochs} epoch(s)...")
 
    for epoch in range(adaptation_epochs):
        epoch_loss = 0.0
        n_batches  = 0
        for src_b, tgt_b, tgt_mask_b, eng_idx_b in adapt_dl:
            src_b, tgt_b, tgt_mask_b, eng_idx_b = (
                src_b.to(device), tgt_b.to(device),
                tgt_mask_b.to(device), eng_idx_b.to(device),
            )
            opt.zero_grad(set_to_none=True)
 
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                pred_b    = model(src_b, eng_idx_b)
                elem_loss = torch.abs(pred_b - tgt_b)
                denom     = torch.clamp(tgt_mask_b.sum(), min=1.0)
                base_loss = (elem_loss * tgt_mask_b).sum() / denom
 
                # Variance penalty, same as training, keeps spread realistic.
                if src_b.size(0) > 1:
                    valid_cols = tgt_mask_b.sum(dim=0) > 0
                    if bool(valid_cols.any()):
                        pred_std = pred_b[:, valid_cols].float().std(dim=0, unbiased=False)
                        tgt_std  = tgt_b[:, valid_cols].float().std(dim=0, unbiased=False)
                        var_loss = torch.mean(torch.abs(pred_std - tgt_std))
                    else:
                        var_loss = torch.tensor(0.0, device=device)
                else:
                    var_loss = torch.tensor(0.0, device=device)
 
                total_loss = base_loss + lambda_var * var_loss
 
            scaler.scale(total_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            epoch_loss += float(total_loss.item())
            n_batches  += 1
 
        if verbose:
            print("    Epoch {}/{}, loss: {:.6f}".format(
                epoch + 1, adaptation_epochs, epoch_loss / max(1, n_batches)
            ))
 
    # Generate from seed sequences
    model.eval()
 
    # Determine output format from the seed file (multi-sequence vs single-sequence).
    first_id     = str(groups[0][0])
    is_multi_out = "::" in first_id and "eng_" in first_id
 
    # Build a flat array of all normalised frames for re-seeding during generation.
    # For multi-sequence seed files, concatenate all groups.
    all_seed_frames = np.concatenate(
        [g[feature_cols].values.astype(np.float32) for _, g in groups], axis=0
    )
    rng = np.random.default_rng(int(ckpt.get("SEED", 42)))
 
    rows      = []
    steps_per = max(1, num_samples // len(groups)) if len(groups) > 1 else num_samples
 
    for grp_idx, (engine_id, grp_df) in enumerate(groups):
        eng_idx_tensor = torch.tensor(
            [seed_eng_to_idx[engine_id]], dtype=torch.long, device=device
        )
        grp_frames = grp_df[feature_cols].values.astype(np.float32)
 
        # Seed the context window from the first input_window rows of this group.
        actual_window = min(input_window, len(grp_frames))
        seed_vals  = grp_frames[:actual_window]
        if actual_window < input_window:
            pad       = np.zeros((input_window - actual_window, F), dtype=np.float32)
            seed_vals = np.concatenate([pad, seed_vals], axis=0)
 
        seed_t = torch.tensor(seed_vals, dtype=torch.float32, device=device).unsqueeze(0)
        window = torch.cat([seed_t, torch.ones_like(seed_t)], dim=-1)
 
        # For the last group, generate any remaining rows to hit num_samples exactly.
        steps = steps_per if grp_idx < len(groups) - 1 else num_samples - len(rows)
        steps = max(1, steps)
 
        numeric_eng_id = int(first_id.split("eng_")[-1]) if is_multi_out else None
 
        with torch.no_grad():
            for t in range(steps):
                # Periodic re-seed to prevent autoregressive variance collapse.
                if reseed_interval > 0 and t > 0 and t % reseed_interval == 0:
                    max_start = max(0, len(all_seed_frames) - input_window)
                    start     = int(rng.integers(0, max_start + 1))
                    real_slice = all_seed_frames[start:start + input_window]
                    if len(real_slice) < input_window:
                        pad        = np.zeros((input_window - len(real_slice), F), dtype=np.float32)
                        real_slice = np.concatenate([pad, real_slice], axis=0)
                    rs_t   = torch.tensor(real_slice, dtype=torch.float32, device=device).unsqueeze(0)
                    window = torch.cat([rs_t, torch.ones_like(rs_t)], dim=-1)
 
                pred = model(window, eng_idx_tensor)
 
                next_vals  = pred.unsqueeze(1)
                next_masks = torch.ones_like(next_vals)
                next_step  = torch.cat([next_vals, next_masks], dim=-1)
                window     = torch.cat([window[:, 1:, :], next_step], dim=1)
 
                pred_np = pred.squeeze(0).float().cpu().numpy()
                denorm  = ((pred_np + 1.0) * 0.5) * ranges.values + col_mins.values
 
                if is_multi_out:
                    row = [("Engine_ID", numeric_eng_id), ("Time_in_cycles", t + 1)]
                else:
                    row = [("Time", t + 1)]
 
                for i, col_name in enumerate(feature_cols):
                    row.append((col_name, float(denorm[i])))
                rows.append(dict(row))
 
        if verbose:
            print("  Generated {} steps for engine {} ({} total rows so far)".format(
                steps, engine_id, len(rows)
            ))
 
    df = pd.DataFrame(rows).iloc[:num_samples].reset_index(drop=True)
 
    if is_multi_out:
        ordered_cols = ["Engine_ID", "Time_in_cycles"] + list(feature_cols)
    else:
        ordered_cols = ["Time"] + list(feature_cols)
    df = df[ordered_cols]
 
    if verbose:
        print("=" * 60)
        print(f"Seeded generation complete. Rows: {len(df)}")
        print("=" * 60)
 
    return df
_CYCLES_NOT_SET = object()  # sentinel: caller did not explicitly pass cycles_per_engine

@torch.no_grad()
def generate_transformer_data(
    checkpoint_path,
    num_samples=None,
    n_engines=None,
    cycles_per_engine=_CYCLES_NOT_SET,
    strategy="by_engines",
    device=None,
    verbose=False
):
    """
    Generate synthetic engine time-series data from trained transformer.
    
    Args:
        checkpoint_path: Path to model checkpoint
        num_samples: Total samples (used when strategy="by_samples")
        n_engines: Number of engines to generate (None = all engines)
        cycles_per_engine: Cycles per engine (used when strategy="by_engines")
        strategy: "by_engines" or "by_samples"
        device: torch device
        verbose: Print progress
    
    Returns:
        DataFrame matching the original data format exactly.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = _resolve_path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError("Checkpoint not found: {}".format(checkpoint_path))
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    feature_cols = ckpt["feature_cols"]
    col_mins = ckpt["col_mins"]
    ranges = ckpt["ranges"]
    cfg = ckpt.get("config", {}) or {}

    # Load generation config from checkpoint if not provided
    gen_cfg = cfg.get("generation", {}) if isinstance(cfg, dict) else {}
    if strategy is None:
        strategy = gen_cfg.get("strategy", "by_engines")
    if n_engines is None:
        n_engines = gen_cfg.get("n_engines", None)

    # See if user explicitly passed cycles_per_engine or if we should load it from config (default 200)
    cycles_per_engine_explicit = cycles_per_engine is not _CYCLES_NOT_SET
    if not cycles_per_engine_explicit:
        cycles_per_engine = gen_cfg.get("cycles_per_engine", 200)
    if num_samples is None:
        num_samples = gen_cfg.get("num_samples", 500)

    eng_ids = ckpt.get("eng_ids", None)
    eng_to_idx = ckpt.get("eng_to_idx", None)

    if eng_ids is None:
        emb_w = ckpt["model_state_dict"]["eng_emb.weight"]
        n_engines_vocab = int(emb_w.shape[0])
        eng_ids = list(range(n_engines_vocab))
        eng_to_idx = {e: i for i, e in enumerate(eng_ids)}
    else:
        if eng_to_idx is None:
            eng_to_idx = {e: i for i, e in enumerate(eng_ids)}
        n_engines_vocab = len(eng_ids)

    # Sort engine IDs by their numeric value
    def extract_numeric_id(eid):
        eid_str = str(eid)
        if "::" in eid_str and "eng_" in eid_str:
            return int(eid_str.split("eng_")[-1])
        try:
            return int(eid)
        except (ValueError, TypeError):
            return 0

    # Use ALL engines recorded at save time (sequence_lengths covers every engine
    # in the raw data, including those too short to train on).  Fall back to the
    # trained eng_ids list when the checkpoint pre-dates this change.
    all_recorded_ids = ckpt.get("sequence_lengths", {})
    if all_recorded_ids:
        # Reconstruct engine-id objects in the same format as eng_ids
        # (e.g. "FD001.csv::eng_1" strings for multisequence data)
        format_ref = eng_ids[0] if eng_ids else None
        def _reformat(key_str, ref):
            """Convert a sequence_lengths key back to the eng_ids format."""
            if ref is None:
                return key_str
            ref_str = str(ref)
            if "::" in ref_str and "eng_" in ref_str:
                # key_str is already "file::eng_N" format
                return key_str
            # Single-sequence: key is just the file_idx as string
            return key_str
        generation_ids = sorted(
            [_reformat(k, format_ref) for k in all_recorded_ids.keys()],
            key=extract_numeric_id
        )
    else:
        generation_ids = sorted(eng_ids, key=extract_numeric_id)

    # Limit to n_engines if specified
    if n_engines is not None and n_engines < len(generation_ids):
        generation_ids = generation_ids[:n_engines]

    eng_ids_sorted = generation_ids  # used throughout the rest of the function

    runtime = cfg.get("runtime", {}) if isinstance(cfg, dict) else {}
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
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    # Load seed windows and sequence lengths written at training time.
    # seed_windows:     {str(eng_id) -> np.array [input_window, F], normalised [-1,1]}
    # sequence_lengths: {str(eng_id) -> int}: actual lifecycle length in training data
    seed_windows = ckpt.get("seed_windows", {})
    sequence_lengths = ckpt.get("sequence_lengths", {})

    # reseed_interval: re-inject a randomly sampled real window every N steps.
    # Prevents autoregressive variance collapse on long single-sequence rollouts.
    # Only active when > 0 and seed_windows is available. Disabled (None/0) by
    # default; has no effect on multi-sequence (NASA-style) generation.
    _gen_cfg = cfg.get("generation", {}) if isinstance(cfg, dict) else {}
    reseed_interval = _gen_cfg.get("reseed_interval", None)
    reseed_interval = int(reseed_interval) if reseed_interval else 0

    # Determine if this is multi-sequence data
    is_multisequence = False
    if eng_ids_sorted and len(eng_ids_sorted) > 0:
        first_id = str(eng_ids_sorted[0])
        is_multisequence = "::" in first_id and "eng_" in first_id

    # Determine steps per engine based on strategy
    if strategy == "by_engines":
        if verbose:
            print("Strategy: by_engines | Engines: {} | Cycles per engine: from actual lengths (fallback: {})".format(
                len(eng_ids_sorted), cycles_per_engine
            ))
    else:  # "by_samples"
        num_engines = len(eng_ids_sorted)
        steps_per_engine_default = int(math.ceil(float(num_samples) / float(num_engines)))
        if verbose:
            print("Strategy: by_samples | Target: {} | Engines: {} | Steps per engine: {}".format(
                num_samples, num_engines, steps_per_engine_default
            ))

    rows = []
    for engine_id in eng_ids_sorted:
        seed_key = str(engine_id)

        # --- Resolve seed window and step count for this engine ---
        # For engines too short to train on (no seed_window entry), find the
        # closest-length trained peer and borrow its seed window instead.
        resolved_seed_key = seed_key
        if seed_key not in seed_windows and seed_windows:
            this_len = sequence_lengths.get(seed_key, 0)
            # Pick the trained engine whose length is closest to this one
            resolved_seed_key = min(
                seed_windows.keys(),
                key=lambda k: abs(sequence_lengths.get(k, 0) - this_len)
            )
            if verbose:
                print("  Engine {} (len={}) too short for seed window; "
                      "using peer {} (len={}) as seed.".format(
                    seed_key, this_len,
                    resolved_seed_key, sequence_lengths.get(resolved_seed_key, '?')
                ))

        # --- Resolve embedding index ---
        # Untrained engines (too short for training) won't be in eng_to_idx.
        # Use the same resolved peer's embedding so the model gets a valid input.
        if engine_id in eng_to_idx:
            eng_idx = torch.tensor([eng_to_idx[engine_id]], dtype=torch.long, device=device)
        else:
            # resolved_seed_key is a str key; eng_to_idx keys may be original objects.
            # Find the matching eng_to_idx entry by string comparison.
            peer_idx = next(
                (v for k, v in eng_to_idx.items() if str(k) == resolved_seed_key),
                0  # absolute fallback: use embedding 0
            )
            eng_idx = torch.tensor([peer_idx], dtype=torch.long, device=device)

        # --- Seed the context window from real early-life data ---
        if resolved_seed_key in seed_windows:
            seed_vals = torch.tensor(
                seed_windows[resolved_seed_key], dtype=torch.float32, device=device
            ).unsqueeze(0)  # [1, input_window, F]
            window = torch.cat([seed_vals, torch.ones_like(seed_vals)], dim=-1)
        else:
            # Fallback for old checkpoints with no seed_windows at all.
            value_window = torch.zeros((1, input_window, F), dtype=torch.float32, device=device)
            col_ranges_arr = np.array([
                float(ranges[c]) if hasattr(ranges, "__getitem__") else float(ranges.iloc[i])
                for i, c in enumerate(feature_cols)
            ])
            zero_var_mask = torch.tensor(col_ranges_arr == 1.0, dtype=torch.bool, device=device)
            value_window[:, :, zero_var_mask] = -1.0
            window = torch.cat([value_window, torch.ones((1, input_window, F), dtype=torch.float32, device=device)], dim=-1)

        # --- Determine how many steps to generate for this engine ---
        if strategy == "by_engines":
            if cycles_per_engine_explicit:
                # User explicitly passed cycles_per_engine; honour it unconditionally.
                steps = int(cycles_per_engine)
            else:
                # Use the engine's actual training length so the synthetic lifecycle
                # distribution mirrors the real data.
                actual_len = sequence_lengths.get(seed_key, None)
                if actual_len is not None and actual_len > 0:
                    if actual_len <= input_window:
                        # Engine shorter than context window: generate its full length.
                        steps = actual_len
                    else:
                        # Normal case: seed covers first input_window rows, generate rest.
                        steps = actual_len - input_window
                else:
                    steps = cycles_per_engine  # old-checkpoint fallback
        else:
            steps = steps_per_engine_default

        numeric_eng_id = extract_numeric_id(engine_id) if is_multisequence else None

        # Build a flat array of all normalised frames for this engine so we can
        # sample random re-seed windows during generation.
        # Only used when reseed_interval > 0 and seed data is available.
        reseed_frames = None
        if reseed_interval > 0 and resolved_seed_key in seed_windows:
            full_key = resolved_seed_key
            full_len = sequence_lengths.get(full_key, 0)
            full_sequences = ckpt.get("full_sequences", {})
            if full_key in full_sequences:
                reseed_frames = full_sequences[full_key]  # np.array [T, F]
            else:
                reseed_frames = seed_windows[full_key]    # fallback: just the seed window

        rng = np.random.default_rng(int(ckpt.get('SEED', 42)))

        for t in range(steps):
            # Periodic re-seed: replace the context window with a randomly
            # sampled real window to prevent autoregressive variance collapse.
            if (reseed_interval > 0
                    and t > 0
                    and t % reseed_interval == 0
                    and reseed_frames is not None):
                max_start = max(0, len(reseed_frames) - input_window)
                start = int(rng.integers(0, max_start + 1))
                real_slice = reseed_frames[start:start + input_window]
                # Pad with zeros at the front if the slice is shorter than input_window
                if len(real_slice) < input_window:
                    pad = np.zeros((input_window - len(real_slice), F), dtype=np.float32)
                    real_slice = np.concatenate([pad, real_slice], axis=0)
                seed_t = torch.tensor(real_slice, dtype=torch.float32, device=device).unsqueeze(0)
                window = torch.cat([seed_t, torch.ones_like(seed_t)], dim=-1)

            pred = model(window, eng_idx)

            next_vals = pred.unsqueeze(1)
            next_masks = torch.ones_like(next_vals)
            next_step = torch.cat([next_vals, next_masks], dim=-1)
            window = torch.cat([window[:, 1:, :], next_step], dim=1)

            pred_np = pred.squeeze(0).float().cpu().numpy()
            denorm = ((pred_np + 1.0) * 0.5) * ranges.values + col_mins.values

            # Build row - use list to preserve order
            if is_multisequence:
                row = [
                    ("Engine_ID", numeric_eng_id),
                    ("Time_in_cycles", t + 1)
                ]
            else:
                # Single-sequence: emit a synthetic time index so the output
                # has the same structure as the input (e.g. a Time column).
                row = [("Time", t + 1)]

            # Add feature columns in exact order
            for i, col_name in enumerate(feature_cols):
                row.append((col_name, float(denorm[i])))

            rows.append(dict(row))

            # Only stop early if using by_samples strategy
            if strategy == "by_samples" and len(rows) >= num_samples:
                break

        if verbose:
            print("Generated engine {} ({} steps, rows so far: {})".format(
                numeric_eng_id or engine_id, steps, len(rows)
            ))

        # Only stop early if using by_samples strategy
        if strategy == "by_samples" and len(rows) >= num_samples:
            break

    df = pd.DataFrame(rows)
    
    # Trim to exact num_samples only if using by_samples strategy
    if strategy == "by_samples":
        df = df.iloc[:num_samples].reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    
    # Force the exact column order
    if is_multisequence:
        ordered_cols = ["Engine_ID", "Time_in_cycles"] + list(feature_cols)
    else:
        ordered_cols = ["Time"] + list(feature_cols)
    
    df = df[ordered_cols]
    
    return df
# -------------------------
# Training entrypoint (orchestrator)
# -------------------------
def train_transformer(args, device):
    cfg = load_config(args.config)
    runtime = cfg.get("runtime", {})
    io_cfg = cfg.get("io", {}) or {}
    log_cfg = cfg.get("logging", {}) or {}

    lambda_var = float(runtime.get("lambda_var", 0.01))
    seed = int(runtime.get("seed", 42))
    input_window = int(runtime.get("input_window", 64))
    pred_window = int(runtime.get("pred_window", 1))
    batch_size = int(runtime.get("batch_size", 256))
    epochs = int(runtime.get("epochs", 60))
    lr = float(runtime.get("lr", 5e-4))
    weight_decay = float(runtime.get("weight_decay", 1e-4))
    num_workers = int(0)

    model_name = cfg.get("model", {}).get("name")
    if not model_name:
        raise ValueError("model.name is required in transformer_config.yaml")
    model_path = CHECKPOINT_DIR / "{}.pt".format(model_name)

    outdir = _resolve_path(io_cfg.get("outputs_dir"))

    # Setup logging first
    logger = setup_logging(log_cfg, outdir)
    # Setup TensorBoard
    if TENSORBOARD_AVAILABLE:
        tb_writer: Optional[SummaryWriter] = None
        if bool(log_cfg.get("use_tensorboard", True)) and TENSORBOARD_AVAILABLE:
            tb_log_dir = outdir / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
            tb_writer = SummaryWriter(log_dir=tb_log_dir)
            logger.info("TensorBoard logging enabled: {}".format(tb_log_dir))
            logger.info("  View with: tensorboard --logdir {}".format(outdir / "runs"))
        elif bool(log_cfg.get("use_tensorboard", True)) and not TENSORBOARD_AVAILABLE:
            logger.warning("TensorBoard requested but not available. Install with: pip install tensorboard")
    else:
        tb_writer = None

    # Setup CSV logging
    csv_file = None
    csv_writer_obj = None
    if bool(log_cfg.get("use_csv", True)):
        csv_path = outdir / "metrics.csv"
        csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        csv_writer_obj = csv.writer(csv_file)
        csv_writer_obj.writerow([
            "timestamp",
            "epoch",
            "train_base",
            "train_var",
            "train_total",
            "val_base",
            "val_var",
            "val_total",
            "learning_rate",
            "best_val_total",
        ])
        logger.info("CSV logging enabled: {}".format(csv_path))

    plt.style.use("seaborn-v0_8-darkgrid")
    set_seeds(seed)

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    logger.info("=" * 60)
    logger.info("Transformer Training - Generic Sequence Synthesizer")
    logger.info("=" * 60)
    logger.info("Device: {}".format(device))
    logger.info("Config: {}".format(_resolve_path(args.config)))
    logger.info("Output directory: {}".format(outdir))
    logger.info("Checkpoints will be saved to: {}".format(CHECKPOINT_DIR))

    work_scaled, eng_col, time_col, feature_cols, col_mins, ranges = load_and_prepare_dataframe(cfg, logger=logger)

    groups = build_sequence_groups(work_scaled, eng_col, time_col, input_window, pred_window)
    logger.info("Sequences with sufficient length: {} of {}".format(len(groups), work_scaled[eng_col].nunique()))

    if len(groups) == 0:
        raise RuntimeError("No sequences with sufficient length for training.")

    eng_ids = [e for e, _ in groups]
    eng_to_idx = {e: i for i, e in enumerate(eng_ids)}

    if len(groups) >= 2:
        train_groups, val_groups = split_groups(groups, seed, train_frac=0.9)
        logger.info("Train sequences: {} | Val sequences: {}".format(len(train_groups), len(val_groups)))
    else:
        logger.warning("Only one valid engine/sequence found. Falling back to time-based split.")

        eng_id, sub = groups[0]
        sub = sub.sort_values(time_col).reset_index(drop=True).copy()

        n = len(sub)
        min_needed = input_window + pred_window
        split_idx = int(0.9 * n)

        if split_idx < min_needed:
            split_idx = min_needed
        if (n - split_idx) < min_needed:
            split_idx = n - min_needed

        if split_idx <= 0 or split_idx >= n:
            raise RuntimeError("Not enough sequence length for time-based train/val split.")

        train_sub = sub.iloc[:split_idx].copy()
        val_sub = sub.iloc[split_idx:].copy()

        train_groups = [(eng_id, train_sub)]
        val_groups = [(eng_id, val_sub)]

        logger.info(
            "Time-based split for single sequence: train rows {} | val rows {}".format(
                len(train_sub), len(val_sub)
            )
        )

    train_ds, val_ds, train_dl, val_dl = build_dataloaders(
        train_groups=train_groups,
        val_groups=val_groups,
        feature_cols=feature_cols,
        eng_to_idx=eng_to_idx,
        input_window=input_window,
        pred_window=pred_window,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    logger.info("Train windows: {} | Val windows: {}".format(len(train_ds), len(val_ds)))

    F = len(feature_cols)
    src_dim = 2 * F
    out_dim = F
    model = build_model_from_cfg(cfg, src_dim=src_dim, out_dim=out_dim, n_engines=len(eng_ids)).to(device)

    # Log model + params
    logger.info("=" * 60)
    logger.info("Model Architecture:\n{}".format(str(model)))
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Total parameters: {:,}".format(total_params))
    logger.info("Trainable parameters: {:,}".format(trainable_params))
    logger.info("=" * 60)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    # Log hyperparams
    logger.info("=" * 60)
    logger.info("Hyperparameters:")
    logger.info("  Input window: {}".format(input_window))
    logger.info("  Prediction window: {}".format(pred_window))
    logger.info("  Batch size: {}".format(batch_size))
    logger.info("  Epochs: {}".format(epochs))
    logger.info("  Learning rate: {}".format(lr))
    logger.info("  Weight decay: {}".format(weight_decay))
    logger.info("  Lambda variance: {}".format(lambda_var))
    logger.info("  Num workers: {}".format(num_workers))
    logger.info("=" * 60)

    start_epoch = 1
    best_val_total = float("inf")

    train_hist = []
    val_hist = []

    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60)

    for ep in range(start_epoch, epochs + 1):
        epoch_start = time.time()

        tr = run_epoch(model, train_dl, opt, scaler, device, lambda_var, train=True)
        va = run_epoch(model, val_dl, opt, scaler, device, lambda_var, train=False)

        epoch_time = time.time() - epoch_start
        current_lr = opt.param_groups[0]["lr"]

        # Console/file log
        logger.info(
            "Epoch {:02d}/{}  "
            "train base {:.6f} var {:.6f} total {:.6f} | "
            "val base {:.6f} var {:.6f} total {:.6f} | "
            "lr {:.2e} | time {:.1f}s".format(
                ep, epochs,
                tr["base"], tr["var"], tr["total"],
                va["base"], va["var"], va["total"],
                current_lr, epoch_time
            )
        )

        # TensorBoard scalars
        if tb_writer:
            tb_writer.add_scalar("Loss/train_base", tr["base"], ep)
            tb_writer.add_scalar("Loss/train_var", tr["var"], ep)
            tb_writer.add_scalar("Loss/train_total", tr["total"], ep)
            tb_writer.add_scalar("Loss/val_base", va["base"], ep)
            tb_writer.add_scalar("Loss/val_var", va["var"], ep)
            tb_writer.add_scalar("Loss/val_total", va["total"], ep)
            tb_writer.add_scalar("Learning_Rate", current_lr, ep)
            tb_writer.add_scalar("Time/epoch", epoch_time, ep)

        # CSV row
        if csv_writer_obj and csv_file:
            csv_writer_obj.writerow([
                datetime.now().isoformat(),
                ep,
                "{:.8f}".format(tr["base"]),
                "{:.8f}".format(tr["var"]),
                "{:.8f}".format(tr["total"]),
                "{:.8f}".format(va["base"]),
                "{:.8f}".format(va["var"]),
                "{:.8f}".format(va["total"]),
                "{:.6e}".format(current_lr),
                "{:.8f}".format(best_val_total),
            ])
            csv_file.flush()

        # Save best model (by val_total), generation-ready packed format
        if va["total"] < best_val_total:
            best_val_total = va["total"]
            save_packed_model_checkpoint(
                path=model_path,
                model=model,
                feature_cols=feature_cols,
                eng_ids=eng_ids,
                eng_to_idx=eng_to_idx,
                col_mins=col_mins,
                ranges=ranges,
                eng_col=eng_col,
                time_col=time_col,
                input_window=input_window,
                seed=seed,
                cfg=cfg,
                config_path=args.config,
                work_scaled=work_scaled,
            )
            logger.info("  [Best Model] New best val total loss: {:.6f} (saved to {})".format(best_val_total, model_path.name))

        sched.step()
        train_hist.append(tr["total"])
        val_hist.append(va["total"])

    if csv_file:
        csv_file.close()
        logger.info("CSV metrics saved to: {}".format(outdir / "metrics.csv"))

    curve_path = None
    if bool(log_cfg.get("save_plots", True)):
        curve_path = save_training_curve(outdir, train_hist, val_hist)
        logger.info("Training curve saved to: {}".format(curve_path))

        if tb_writer:
            # Add final curve figure
            fig = plt.figure(figsize=(8, 4))
            plt.plot(train_hist, label="train_total")
            plt.plot(val_hist, label="val_total")
            plt.xlabel("epoch")
            plt.ylabel("Total loss")
            plt.title("Training")
            plt.legend()
            plt.tight_layout()
            tb_writer.add_figure("Training Curves", fig)
            plt.close(fig)

    # Final TensorBoard: hparams (single call, at end)
    if tb_writer is not None:
        tb_writer.add_hparams(
            {
                "input_window": input_window,
                "pred_window": pred_window,
                "batch_size": batch_size,
                "epochs": epochs,
                "lr": lr,
                "weight_decay": weight_decay,
                "lambda_var": lambda_var,
                "num_features": F,
                "num_engines": len(eng_ids),
                "total_params": total_params,
                "trainable_params": trainable_params,
            },
            {
                "hparam/final_train_total": train_hist[-1] if train_hist else 0.0,
                "hparam/final_val_total": val_hist[-1] if val_hist else 0.0,
                "hparam/best_val_total": best_val_total,
            }
        )
        if tb_writer is not None:
            tb_writer.close()
            logger.info("TensorBoard writer closed")

    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)
    logger.info("Model + metadata saved to: {}".format(model_path))
    logger.info("Best validation total loss: {:.6f}".format(best_val_total))
    if train_hist:
        logger.info("Final train total loss: {:.6f}".format(train_hist[-1]))
    if val_hist:
        logger.info("Final val total loss: {:.6f}".format(val_hist[-1]))
    logger.info("All outputs saved to: {}".format(outdir))
    logger.info("=" * 60)

    return model_path, curve_path


# -------------------------
# CLI
# -------------------------
def build_parser():
    parser = argparse.ArgumentParser(description="Transformer training and generation")

    parser.add_argument(
        "--mode",
        choices=["train", "generate"],
        default="train",
        help="Mode: train or generate (default: train)",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to transformer_config.yaml",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to checkpoint (used for generation)",
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to seed CSV for generating data from realistic context. If provided, seeding generation is used.",
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=500,
        help="Number of samples to generate (default: 500)",
    )

    parser.add_argument(
        "--run_name",
        type=str,
        default="",
        help="Optional run name for organizing output directories",
    )

    return parser


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = build_parser().parse_args()

    if args.mode == "train":
        train_transformer(args, device)
        return

    # generate mode
    if args.checkpoint:
        checkpoint = _resolve_path(args.checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError("Checkpoint not found: {}".format(checkpoint))
        ckpt_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cfg = ckpt_data.get("config", {})
    else:
        cfg = load_config(args.config)
        model_name = cfg.get("model", {}).get("name")
        if not model_name:
            raise ValueError("model.name is required in transformer_config.yaml")
        checkpoint = _resolve_path(CHECKPOINT_DIR / "{}.pt".format(model_name))
        if not checkpoint.exists():
            raise FileNotFoundError("Checkpoint not found: {}".format(checkpoint))

    gen_cfg = cfg.get("generation", {})

    # If a config file was explicitly provided, let it override the generation
    # section from the checkpoint's embedded config.
    if args.config and Path(args.config).exists():
        try:
            override_cfg = load_config(args.config)
            if "generation" in override_cfg:
                gen_cfg = override_cfg["generation"]
        except Exception:
            pass

    model_name = cfg.get("model", {}).get("name", "transformer_model")

    # Only pass cycles_per_engine if the user explicitly set it in the config.
    # Passing _CYCLES_NOT_SET lets generate_transformer_data use each engine's
    # actual training length, which reproduces the real distribution.
    # Passing an explicit integer overrides that and generates that many cycles
    # for every engine uniformly.
    cycles_per_engine_arg = (
        gen_cfg["cycles_per_engine"]
        if "cycles_per_engine" in gen_cfg
        else _CYCLES_NOT_SET
    )

    if args.data:
        # Seeding generation
        df = generate_from_seed_file(
            base_checkpoint_path=checkpoint,
            seed_csv_path=args.data,
            config_path=args.config,
            num_samples=args.num_samples if args.num_samples != 500 else gen_cfg.get("num_samples", 500),
            adaptation_epochs=5,
            adapt_lr=1e-4,
            reseed_interval=gen_cfg.get("reseed_interval", 0),
            device=device,
            verbose=True
        )
    else:
        # Standard generation from random noise/sequences
        df = generate_transformer_data(
            checkpoint_path=checkpoint,
            num_samples=args.num_samples if args.num_samples != 500 else gen_cfg.get("num_samples", 500),
            n_engines=gen_cfg.get("n_engines", None),
            cycles_per_engine=cycles_per_engine_arg,
            strategy=gen_cfg.get("strategy", "by_engines"),
            device=device,
            verbose=True
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # If run_name provided by CLI, use it as a subfolder
    if args.run_name:
        out = _resolve_path(
            Path("outputs") / "Transformer" / "synthetic_data" / args.run_name /
            "{}_synthetic_data_{}.csv".format(model_name, timestamp)
        )
    else:
        out = _resolve_path(
            Path("outputs") / "Transformer" / "synthetic_data" /
            "{}_synthetic_data_{}.csv".format(model_name, timestamp)
        )
        
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print("Saved {} rows ({} unique engines) to: {}".format(
        len(df),
        df['Engine_ID'].nunique() if 'Engine_ID' in df.columns else 1,
        out
    ))

if __name__ == "__main__":
    main()