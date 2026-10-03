# Transformer Training & Generation (V7)

## STATUS 
- Basic Generation [COMPLETED]
- Seeded Generation [COMPLETED]
- Training [COMPLETED]
- Support for multiple files  [COMPLETED]
- Frontend generation settings control  [COMPLETED]
  
---

## Overview

A Transformer-based system for generating realistic synthetic sequences. Designed to work across different dataset formats without hardcoded assumptions.

---

## Supported Dataset Formats

**Multi-sequence** (e.g. NASA CMAPSS): multiple engines, each with its own timeline. Detected automatically when an `engine_id`-style column is present.

**Single-sequence** (e.g. one long sensor log): one continuous recording. The entire file is treated as a single sequence. Asynchronous sampling (sensors firing at different rates) is handled automatically via forward-fill.

Both formats produce the same output structure and use the same training pipeline.

---

## How to Run

Note: by default it already knows where the default config location is, so no need to put the path every time, unless you intend to use a different config file.

### Train
```
python Transformer_TrainingV7.py --mode train --config src/config/transformer_config.yaml
```

### Generate
```
python Transformer_TrainingV7.py --mode generate --config src/config/transformer_config.yaml
```

---

## Config Reference

```yaml
runtime:
  lambda_var: 0.01      # variance matching penalty weight
  seed: 42
  input_window: 32      # context window fed to the model each step
  pred_window: 1
  batch_size: 256       # reduce for short single-sequence datasets (~64)
  epochs: 60
  lr: 5.0e-4
  weight_decay: 1.0e-4

generation:
  strategy: "by_engine"     # "by_engine" or "by_samples"
  n_engines: null           # null = all engines; integer = first N by ID
  cycles_per_engine: 200    # omit this key to use each engine's actual length
  num_samples: 500          # used only when strategy = "by_samples"
  reseed_interval: 0        # re-inject a real window every N steps during
                            # generation; prevents collapse on long single-
                            # sequence rollouts. 0 = disabled. Try 64-100
                            # for RR-style data.

io:
  data_paths:
    - "data/FD001.csv"
  outputs_dir: "outputs/transformer"
  checkpoint_dir: "src/model_checkPoints/transformer"
```

**`cycles_per_engine` behaviour:**
- Key **present** → generate exactly that many steps per engine (uniform override)
- Key **absent** → generate each engine's actual training length, reproducing the real lifecycle distribution

---

## Generation Details

### Seed windows
At the end of training, the first `input_window` rows per engine are saved into the checkpoint. Generation starts from these real early-life windows rather than zeros, which prevents the systematic right-shift in distribution that a cold-start would cause.

### Autoregressive rollout
Each predicted step is fed back as input for the next step. For long rollouts this can cause variance collapse (predictions converge to a fixed point). Use `reseed_interval` to periodically reset the context window to a randomly sampled real window from training data.

### Short engines
Engines with fewer rows than `input_window + 1` are excluded from training but still appear in generation output. Their seed window is borrowed from the nearest-length trained engine.

---

## Seeded Generation

Adapts a trained model to a new operating condition without retraining from scratch.

```python
from Transformer_TrainingV7 import generate_from_seed_file, _resolve_path

df = generate_from_seed_file(
    base_checkpoint_path=_resolve_path("src/model_checkPoints/transformer/transformer_model.pt"),
    seed_csv_path=_resolve_path("data/wet_conditions.csv"),
    config_path=_resolve_path("src/config/transformer_config.yaml"),
    num_samples=2000,
    adaptation_epochs=5,    # keep low (5-10) to avoid catastrophic forgetting
    adapt_lr=1e-4,
    reseed_interval=0,      # set > 0 for long single-sequence seed files
)
```

The base model weights are **never modified** - a deep copy is fine-tuned and discarded after generation. The adaptation shifts the model's operating point (mean, variance, inter-sensor correlations, temporal dynamics) toward the seed distribution while preserving the base physics learned during training.

Best suited for: same failure mechanism, different operating conditions (e.g. train on dry conditions, seed with wet conditions). Not suited for learning entirely new failure modes in 5 epochs.

---
 

## Validation Scripts

**`validate_generate.py`** — compares synthetic vs actual distributions. Set `DATASET = "NASA"` or `"RR"` at the top of the file.

**`validate_seeding.py`** — three-way comparison: actual seed data vs non-seeded synthetic (control) vs seeded synthetic. Quantifies how much the seeded generation closed the gap to the target distribution.

To run them, train the model first, then run: 
```
python validate_generate.py
```
```
python validate_seeding.py
```
---

Example output from validate_generate.py: 
<img width="1000" alt="image" src="https://github.com/user-attachments/assets/1d0895ec-9c98-4be4-9753-04fd610799ad" />

Image shows how synthetically generated data with n_engines=50, cycles_per_engine=20, compares to actual data.

Example output from validate_seeding.py: 
<img width="1000" alt="image" src="https://github.com/user-attachments/assets/ebb151e3-b38f-4908-bad1-a24d0ec865a6" />

Image shows model trained on FD001.csv, seeded with FD003.csv. The seeded generation allows the distribution of synthetic FD001.csv to approach closer to the actual distribution of FD003.csv. Generation settings were the same as above: n_engines=50, cycles_per_engine=20.



