# Diffusion Model

`diffusion_model5.py` trains a diffusion model on tabular sensor data, fine-tunes it with LoRA
adapters, and generates synthetic rows. Each row is generated independently; the model does not
produce cycle-by-cycle timelines.

## How it works

1. **Training.** Real rows are mixed with increasing amounts of Gaussian noise over `T` steps. A
   neural network learns to predict the noise that was added.
2. **Generation.** Starting from pure noise, the network removes the predicted noise one step at a
   time until realistic rows remain.
3. **Conditioning.** Columns can be fixed ("observed") during generation. The model fills in the
   remaining columns so they are consistent with the fixed ones.

## Design

| Component | Function | Purpose |
|---|---|---|
| Column resolution | `auto_resolve_columns` | Splits columns into metadata (engine ID, time), conditions and targets from the data alone. The mapping is saved in the checkpoint. |
| Condition selection | `select_conditioning_columns` | Picks the `n_conditions` most informative columns using a variance and correlation heuristic. No manual labels are needed. |
| Denoiser | `TabularTransformerDenoiser` | Treats every column as a token with a learned embedding. Self-attention learns which columns inform which. `--arch mlp` selects a simpler alternative. |
| Sampler | `sample_inpaint` | Clamps observed columns to their real values at every step and generates the rest. |
| LoRA | `LoRALinear`, `fine_tune_with_seed` | Freezes the base model and trains small low-rank matrices on seed data. Adapters are saved as small `.pt` files and can be stacked at generation time. |
| Format detection | `detect_data_format`, `group_rows_by_engine` | Classifies data as snapshot (one row per engine) or continuous. Files without an engine ID are treated as one engine. |
| Data loading | `load_engine_data` | Loads one or more CSV files and aligns their columns. |

The model adapts to any number of numeric columns and does not depend on column names or order.

## Usage

Run from the repository root.

**Train**

```bash
python src/training/Diffusion/diffusion_model5.py --mode train \
    --config src/config/diffusion_config.yaml
```

**Generate**

```bash
python src/training/Diffusion/diffusion_model5.py --mode generate \
    --checkpoint src/model_checkPoints/diffusion/<model_name>/diff_best_ckpt.pt \
    --data data/FD001.csv --n-engines 10 --cycles-per-engine 50
```

`--observe none` (default) samples every column. `--observe conditions` copies the conditioning
columns from rows of `--data` and samples the rest. The `--data` file may contain a subset of the
training columns.

**Fine-tune with LoRA**

```bash
python src/training/Diffusion/diffusion_model5.py --mode fine-tune \
    --checkpoint src/model_checkPoints/diffusion/<model_name>/diff_best_ckpt.pt \
    --seed-data data/FD001.csv \
    --adapter-out src/model_checkPoints/diffusion/adapters/<adapter_name>.pt \
    --lora-rank 4 --lora-alpha 8.0
```

Apply adapters at generation time with `--adapters a.pt b.pt`. The base weights are not changed.

### Main options

| Flag | Default | Meaning |
|---|---|---|
| `--mode` | `train` | `train`, `generate` or `fine-tune` |
| `--arch` | `transformer` | Denoiser type: `transformer` or `mlp` |
| `--data` | | One or more training or reference CSV files |
| `--seed-data` | | One or more CSV files for LoRA fine-tuning |
| `--config` | | YAML or JSON file with any of the settings below |
| `--model-name` | `diffusion` | Name of the checkpoint folder |
| `--epochs`, `--batch-size`, `--lr` | 20, 256, 0.001 | Training settings |
| `--T`, `--beta-start` | 50, 0.0001 | Number of noise steps and the start of the linear noise schedule |
| `--n-conditions` | 3 | Number of conditioning columns to select |
| `--n-engines`, `--cycles-per-engine` | 5, 100 | Amount of data to generate |
| `--no-auto-sample` | off | Skip the sample generation that follows training |

## Multiple input files

`--data` and `--seed-data` accept several paths. Files are concatenated in the given order.

- Columns are reordered to match the first file, so column order may differ between files.
- If the column names differ, a `ValueError` lists the missing and extra columns.
- Seed files for fine-tuning may contain a subset of the base model's columns. Missing columns are
  excluded from the loss.
- The adapter file records which seed files were used.

```python
from src.training.Diffusion.diffusion_model5 import load_engine_data
data, columns = load_engine_data(["data/fleet_A.csv", "data/fleet_B.csv"])
```

## Configuration

`src/config/diffusion_config.yaml`:

```yaml
mode: train
data: data/FD001.csv
model_name: diffusion
model:
  hidden_dim: 256
conditioning:
  n_conditions: 3     # how many columns to condition on; the columns are chosen automatically
training:
  epochs: 100
  batch_size: 16
  lr: 0.0002
diffusion:
  T: 1000
  beta_start: 0.0001
generation:
  n_engines: 10
  cycles_per_engine: 20
```

## Checkpoints

A checkpoint (`diff_best_ckpt.pt`) contains the weights plus everything needed to generate without
the training data: normalization statistics, column names, and the condition and target mapping.

## Evaluation

After sampling, `evaluate_synthetic` compares synthetic and real data:

| Metric | Function | Better when |
|---|---|---|
| MMD (RBF kernel) | `mmd_rbf` | Lower |
| Domain classifier accuracy | `domain_classifier_accuracy` | Closer to 0.5 |
| Mean KS statistic, correlation distance | `evaluate_synthetic` | Lower |
| UMAP plot (optional) | `umap_latent_visualisation` | Point clouds overlap |

The UMAP plot is skipped if `umap-learn` is not installed.

## Use from the dashboard

`api/pipeline.py` runs this script as a subprocess with the flags above and streams its output to
the dashboard. `api/events.py` converts the output into progress events. Anything the dashboard does
can be reproduced with the commands in [Usage](#usage).

Propagation uses this model to regenerate edited data; see
[Propagation_README.md](../../features/Propagation_README.md).

## Tests

```bash
python src/training/Diffusion/test_diffusion_columns.py     # 5 tests
python src/training/Diffusion/test_diffusion_advanced.py    # 10 tests
python src/training/Diffusion/test_diffusion_multifile.py   # 6 tests
```

Each file accepts `--test N[,M,...]` to run selected tests.

| File | Covers |
|---|---|
| `test_diffusion_columns.py` | Arbitrary column names and counts, condition selection, optional metadata, column order |
| `test_diffusion_advanced.py` | Unit-independent heuristics, masked training, inpainting, LoRA, seed fine-tuning, format detection, evaluation metrics |
| `test_diffusion_multifile.py` | Single and multiple paths, header alignment, mismatch errors, training on combined data |

## Troubleshooting

- **Out of memory:** lower `--batch-size` or `--hidden-dim`.
- **Output does not vary with conditions:** train for more epochs or increase `--T`.
- **Loading errors:** every column must be numeric. Remove text columns before training.
