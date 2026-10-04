# Tabular Diffusion Training & Generation

## Overview

This module implements a **Conditional Diffusion Probabilistic Model** designed for generating synthetic tabular data. Unlike temporal models (which learn step-by-step sequences), this model focuses on learning and reproducing the complex **conditional relationships** between variables at any given point in time.

At a high level, it:
- Learns the joint distribution of sensor measurements and operating conditions.
- Uses an unsupervised heuristic (variance + correlation) to decide which variables to condition on.
- Progressively denoises random Gaussian noise into realistic data samples.
- Operates in a **column-agnostic** manner, adapting automatically to new datasets.

---

## What Problem This Solves

Traditional generative models often struggle with:
- **Complex Dependencies**: Non-linear relationships between sensors that are hard to capture with simple rules.
- **Data Scarcity**: Creating high-fidelity synthetic data to augment small training sets.
- **Fixed Schemas**: Most models require fixed input/output dimensions; this implementation adapts to whatever numeric columns are provided.

Diffusion is particularly effective when you need **high-fidelity snapshots** of data that respect operating conditions without the compounding error of long-term temporal recursions.

---

## How It Works (Simple View)

### 1. Forward Phase (Training)
The system takes real data and gradually adds Gaussian noise over many steps (e.g., 1000 steps) until the data is completely unrecognizable.

### 2. Reverse Phase (Generation)
A neural network is trained to "predict the noise" at each step. By knowing how much noise was added, the model can subtract it, effectively "denoising" random noise back into realistic sensor data.

### 3. Conditioning
During the denoising process, the model is "told" what the operating conditions are (e.g., Altitude, Temperature). It uses this information to guide the denoising process so the final sensor values match the environment.

---

## Key Technical Concepts

### 1. Dynamic Column Resolution
All hardcoded column indices and column-name assumptions have been removed. On every run, `auto_resolve_columns` inspects the CSV header and data to decide which columns are **metadata** (e.g. `engine_id`, `time`), which are **conditions** (the informative anchors), and which are **targets** (sensors the model generates). The resolved mapping is stored in the checkpoint so generation reproduces the exact layout.

### 2. Unsupervised Conditioning Heuristic
The model does not require you to label which columns are "inputs" and which are "outputs." It automatically analyzes the dataset's variance and correlations (a unit-free heuristic) to identify the most informative features to use as conditioning anchors.

### 3. Column-Agnostic Architecture
The denoiser dynamically resizes its input/output layers based on the detected numeric features in your CSV. This allows it to work on NASA, Rolls-Royce, or any other tabular dataset without code changes.

### 4. Transformer-Based Masked-Diffusion Denoiser
The default architecture (`--arch transformer`) is a [TabularTransformerDenoiser](diffusion_model5.py#L802) built from `nn.TransformerEncoderLayer` blocks with learned per-column embeddings. Instead of hand-splitting columns into "inputs" vs "outputs," the model receives every column as a token and uses masked self-attention to learn, from the data, which columns should inform which. This makes the pipeline flexible for heterogeneous tabular schemas.

### 5. Inpainting-Style Sampling
Generation uses a RePaint-style [sample_inpaint](diffusion_model5.py#L1104) routine: any columns you supply as "observed" are clamped to their real values at every denoising step, while the remaining columns are generated from noise. This means you can hand the model a partial row (e.g. only operating conditions) and it will fill in the missing sensors consistently with them — no separate "conditional" vs "unconditional" code paths.

### 6. LoRA Adapters for Lightweight Fine-Tuning
When the deployment distribution shifts (new engine family, new operating envelope, seed data from a different regime), a full retrain is overkill. The module ships [LoRALinear](diffusion_model5.py#L660) wrappers and a [fine-tune mode](diffusion_model5.py#L1186) that freezes the base transformer and trains only small low-rank `A`/`B` matrices on a seed CSV. Adapters are saved as tiny `.pt` files and can be composed additively at generation time via `--adapters a.pt b.pt`.

### 7. Format Detection & Single-Engine Fallback
[detect_data_format](diffusion_model5.py#L294) classifies incoming CSVs as `"snapshot"` (one row per flight) or `"continuous"` (multi-row time series per engine) by checking whether `engine_id` values repeat. If no `engine_id` column is present, the whole file is treated as a single engine ([group_rows_by_engine](diffusion_model5.py#L321)) so datasets without an engine identifier still work end-to-end.

### 8. Flexible Denoising Scheduler
Supports multiple noise schedules (linear, cosine) and sampling methods to balance generation speed versus quality.

---

## Inputs & Outputs

### Inputs
- **Training Data**: One or more CSV/TXT files containing numeric sensor data and operating conditions. Multiple files are supported natively — see [Multi-File CSV Input](#multi-file-csv-input).
- **Reference Data (Generation only)**: A CSV of operating conditions to guide the synthetic generation. Only a subset of the training columns is required; the model inpaints the rest.
- **Seed Data (Fine-tune only)**: A small CSV (possibly with a subset of columns) representing a new distribution, used to train a LoRA adapter.
- **Configuration**: A YAML file defining model depth, noise steps, and learning hyperparameters.

### Outputs
- **Model Checkpoints**: `.pt` files containing weights and metadata (normalization stats, feature names).
- **Synthetic Data**: A CSV file containing generated sensor readings that match your reference conditions.
- **Training Logs**: TensorBoard events and CSV metrics.

---

## How to Run

### 1. Train the Model

Ensure your data is in the `data/` directory and referenced in your config.

```bash
python src/training/Diffusion/diffusion_model5.py \
    --mode train \
    --config src/config/diffusion_config.yaml
```

### 2. Generate Synthetic Data

To generate new data, you must provide a trained checkpoint and a reference data file for conditioning.

```bash
python src/training/Diffusion/diffusion_model5.py \
    --mode generate \
    --checkpoint src/model_checkPoints/diffusion/<model_name>/diff_best_ckpt.pt \
    --data data/reference_trajectory.csv \
    --n-engines 10 \
    --cycles-per-engine 50
```

The reference file may contain **only a subset** of the training columns — inpainting sampling will clamp whatever columns are present and generate the rest.

### 3. Fine-Tune with LoRA (Distribution Shift)

```bash
python src/training/Diffusion/diffusion_model5.py \
    --mode fine-tune \
    --checkpoint src/model_checkPoints/diffusion/<model_name>/diff_best_ckpt.pt \
    --seed-data data/new_regime.csv \
    --adapter-out src/model_checkPoints/diffusion/adapters/<adapter_name>.pt \
    --lora-rank 4 --lora-alpha 8.0
```

At generation time, compose one or more adapters without touching the base weights:

```bash
python src/training/Diffusion/diffusion_model5.py \
    --mode generate \
    --checkpoint src/model_checkPoints/diffusion/<model_name>/diff_best_ckpt.pt \
    --adapters src/model_checkPoints/diffusion/adapters/<adapter_name>.pt \
    --data data/reference_trajectory.csv
```

---

## Multi-File CSV Input

`--data` accepts **one or many** CSV paths (`nargs="+"`). When multiple files are supplied, [load_engine_data](diffusion_model5.py#L367) validates and concatenates them before the rest of the pipeline ever sees the data, so every downstream stage (column resolution, format detection, training, evaluation) operates on one contiguous frame.

### Behavior

- **Single path**: behaves exactly as before — `--data file.csv`.
- **Multiple paths, identical headers**: rows are concatenated in the order given.
- **Multiple paths, reordered headers**: columns are reordered to match the **first** file's header before concatenation, so each file may list its columns in any order as long as the *names* match.
- **Mismatched column sets**: raises a `ValueError` listing the `missing` and `extra` column names so the problem file is easy to identify.
- **List of one path**: equivalent to passing that path as a string.

### Examples

```bash
# Concatenate two training files
python src/training/Diffusion/diffusion_model5.py \
    --mode train \
    --data data/fleet_A.csv data/fleet_B.csv \
    --config src/config/diffusion_config.yaml

# Glob multiple files (shell expansion)
python src/training/Diffusion/diffusion_model5.py \
    --mode train \
    --data data/engine_*.csv
```

Programmatic use:

```python
from src.training.Diffusion.diffusion_model5 import load_engine_data
data, columns = load_engine_data(["data/fleet_A.csv", "data/fleet_B.csv"])
```

### Multi-File Seed Data (Fine-tune)

`--seed-data` accepts the same `nargs="+"` contract as `--data`. The fine-tune
pipeline delegates to the same [`load_engine_data`](diffusion_model5.py#L367)
loader, so seed CSVs are concatenated with matching-header validation before
being aligned against the base checkpoint's column schema. Each seed file may
contain only a subset of the base columns — missing columns are masked out of
the LoRA loss.

```bash
python src/training/Diffusion/diffusion_model5.py \
    --mode fine-tune \
    --checkpoint src/model_checkPoints/diffusion/<model_name>/diff_best_ckpt.pt \
    --seed-data data/regime_A.csv data/regime_B.csv \
    --adapter-out src/model_checkPoints/diffusion/adapters/<adapter_name>.pt \
    --lora-rank 4 --lora-alpha 8.0
```

The saved adapter records its provenance: `seed_data` in the adapter `.pt`
file is a list when multiple seed files were used, a string otherwise.

---

## Running from the Dashboard

The dashboard does not import this file. `api/pipeline.py` runs `diffusion_model5.py` as a
subprocess (`DIFFUSION_SCRIPT`), builds the command-line flags from the dashboard's settings
(`--mode train`, `--mode fine-tune`, `--mode generate`, plus data paths, epochs and LoRA options),
and streams the script's printed log back to the page. `api/events.py` turns those log lines into
progress events (epoch, loss, saved files) for the progress bars and charts.

Because the dashboard only uses the command-line interface, anything it does can be reproduced
from a terminal with the commands in [How to Run](#how-to-run).

---

## YAML Configuration Guide

The `diffusion_config.yaml` file controls the model's behavior:

```yaml
model_params:
  num_layers: 4           # Number of residual blocks
  hidden_dim: 256         # Width of each layer
  noise_steps: 1000       # Total diffusion steps
  beta_schedule: "cosine" # "linear" or "cosine" noise growth

training_params:
  batch_size: 128
  epochs: 100
  learning_rate: 0.0002
  weight_decay: 0.00001
```

---

## Checkpoints and Metadata

Diffusion checkpoints (`.pt`) are self-contained. When loaded, they automatically restore:
- **Feature Normalization**: Min/Max values learned during training.
- **Feature Schema**: Exactly which columns the model knows how to generate.
- **Conditioning Masks**: Information on which columns are used as anchors.

---

## Integration with Propagation

The Diffusion model can be used for **Conditional Propagation**. In this mode, you "clamp" certain sensor values (operating conditions) and let the Diffusion model fill in the rest of the sensors.

For more details, see [src/features/Propagation_README.md](../../features/Propagation_README.md).

---

## Evaluation Suite

After training (and on demand), synthetic samples are scored against the real distribution by [evaluate_synthetic](diffusion_model5.py#L1670):

- **MMD (RBF kernel)** — [mmd_rbf](diffusion_model5.py#L1553): a kernel two-sample statistic quantifying distributional gap between real and synthetic data. **Lower is better.**
- **Domain-Classifier Accuracy** — [domain_classifier_accuracy](diffusion_model5.py#L1589): trains a small classifier to tell real from synthetic. **Closer to 0.5 is better** (synthetic is indistinguishable).
- **KS-mean / correlation distance**: marginal and pairwise sanity checks.
- **UMAP visualization** (optional) — [umap_latent_visualisation](diffusion_model5.py#L1632): 2D scatter of real vs. synthetic. Silently skipped if `umap-learn` / `matplotlib` are not installed, so it is safe in minimal environments.

Reports are printed to the console and a UMAP PNG is written to `outputs/<timestamp>_samples/umap_<timestamp>.png` when available.

---

## Testing

The module ships **15 tests** covering robustness, invariance, fine-tuning, format handling, and evaluation — all passing. Plus a dedicated multi-file loader suite. Each file is runnable standalone and supports `--test N[,M,...]` to run a subset:

| File | Tests | What it validates |
|------|-------|-------------------|
| [test_diffusion_columns.py](test_diffusion_columns.py) | 5 | No hardcoded column names or counts; heuristic picks conditions correctly; metadata is optional; column-order invariance. |
| [test_diffusion_advanced.py](test_diffusion_advanced.py) | 10 | Unit-insensitive heuristics, transformer order-invariance, masked-diffusion learning, inpainting clamps observed columns, LoRA wrap/run, seed-data fine-tuning with column subsets, snapshot/continuous format detection, MMD + domain-classifier + full evaluation report. |
| [test_diffusion_multifile.py](test_diffusion_multifile.py) | 6 | Single-path backward compatibility, concatenation of matching headers, reordered-header realignment, clear errors for mismatched column sets, downstream pipeline on combined data, list-of-one ≡ bare path. |

Run everything:

```bash
python src/training/Diffusion/test_diffusion_columns.py
python src/training/Diffusion/test_diffusion_advanced.py
python src/training/Diffusion/test_diffusion_multifile.py
```

---

## Common Issues

- **High GPU Memory Usage**: Diffusion models can be memory-intensive. If you encounter OOM errors, reduce the `batch_size` or `hidden_dim`.
- **Mode Collapse**: If the generated data looks too similar regardless of conditions, try increasing the `epochs` or adjusting the `noise_steps`.
- **Normalization Errors**: Ensure your input data is numeric. The model will automatically skip non-numeric columns.

---

## Changelog — Multi-file Data and LoRA Support (April 2026)

This pass made multi-file data, model naming, and LoRA adapters controllable
from the command line without editing configs by hand.

### Backend — `diffusion_model5.py`

- **`--seed-data` now accepts `nargs="+"`.** Multi-file seed fine-tuning is
  supported end-to-end; seed CSVs are concatenated by `load_engine_data`
  before column alignment against the base checkpoint.
  ([diffusion_model5.py#L2006](diffusion_model5.py#L2006))
- **`fine_tune_with_seed(seed_data_path, ...)`** now accepts either a single
  path or a list of paths.
  ([diffusion_model5.py#L1177](diffusion_model5.py#L1177))
- **Adapter metadata preserves multi-file provenance.** The saved
  `adapter_*.pt` stores `seed_data` as a list when multiple seeds were used,
  so downstream tooling can re-derive the training regime.
  ([diffusion_model5.py#L1294](diffusion_model5.py#L1294))
- **Bug fix — pre-parser `--mode` choices.** The early-exit pre-parser that
  handles `--print-checkpoint-only` had `--mode` restricted to
  `["train", "generate"]`, which caused `--mode fine-tune` to error out
  before `main()` ran. Expanded to `["train", "generate", "fine-tune"]`.
  ([diffusion_model5.py#L106](diffusion_model5.py#L106))

### Verified subprocess contracts

The following commands were smoke-tested:

```bash
# Multi-file training with naming
python src/training/Diffusion/diffusion_model5.py \
    --mode train --data data/FD001.csv data/FD002.csv \
    --model-name my_run --out my_synth.csv \
    --epochs 1 --batch-size 256 --T 10

# Multi-file LoRA fine-tune
python src/training/Diffusion/diffusion_model5.py \
    --mode fine-tune \
    --checkpoint src/model_checkPoints/my_run/diff_best_ckpt.pt \
    --seed-data data/FD001.csv data/FD002.csv \
    --adapter-out src/model_checkPoints/my_run/adapter_combined.pt \
    --lora-rank 4 --lora-alpha 8.0 --epochs 1 --no-auto-sample

# Generate with adapter composition + multi-file seed
python src/training/Diffusion/diffusion_model5.py \
    --mode generate \
    --checkpoint src/model_checkPoints/my_run/diff_best_ckpt.pt \
    --adapters src/model_checkPoints/my_run/adapter_combined.pt \
    --data data/FD001.csv data/FD002.csv \
    --out gen_with_adapter.csv
```

All three commands completed, produced the expected artifacts at the
expected paths, and evaluated synthetic samples against real data.

---

**Developed by The Data Mine @ Purdue University**  
*Spring 2026 - Rolls-Royce Data Synthesizer Project*
