# Transformer Model

`Transformer_TrainingV7.py` trains a Transformer that predicts an engine's next reading from its
recent history, then generates synthetic engine timelines by repeating that prediction.

## Supported data

- **Multiple sequences** (for example NASA C-MAPSS): one timeline per engine. Detected when an
  engine ID column is present.
- **Single sequence:** one long recording, treated as one sequence. Sensors recorded at different
  rates are aligned by forward-filling.

Both use the same training code and produce the same output format.

## Usage

Run from the repository root. `--config` defaults to `src/config/transformer_config.yaml`.

```bash
python src/training/Transformer/Transformer_TrainingV7.py --mode train
python src/training/Transformer/Transformer_TrainingV7.py --mode generate \
    --checkpoint src/model_checkPoints/transformer/<model name>.pt --run_name my_run
```

| Flag | Meaning |
|---|---|
| `--mode` | `train` or `generate` |
| `--config` | Configuration file |
| `--checkpoint` | Model to use for generation |
| `--data` | Seed CSV; when given, generation starts from these records |
| `--num_samples` | Number of samples to generate (default 500) |
| `--run_name` | Name for the output folder |

## Configuration

Main settings in `src/config/transformer_config.yaml`:

```yaml
runtime:
  lambda_var: 0.01       # weight of the penalty that keeps output variance realistic
  seed: 42
  input_window: 31       # past cycles the model reads
  pred_window: 1
  batch_size: 256        # use about 64 for short single-sequence data
  epochs: 50
  lr: 5.0e-4
  weight_decay: 1.0e-4

generation:
  strategy: "by_engines"  # "by_engines" or "by_samples"
  n_engines: 50           # null for all engines
  cycles_per_engine: 30   # remove to use each engine's real length
  num_samples: 2000       # used only with "by_samples"
  reseed_interval: 64     # single sequence only: insert a real window every N steps; 0 disables

io:
  data_paths: ["data/FD001.csv"]
  outputs_dir: "outputs/transformer"
  checkpoint_dir: "src/model_checkPoints/transformer"
```

## Generation details

- **Starting point.** The first `input_window` rows of each engine are saved in the checkpoint.
  Generation starts from these real rows instead of zeros, which avoids a shift in the output.
- **Rollout.** Each prediction is fed back as input for the next step. Long rollouts can lose
  variance; `reseed_interval` counters this by inserting a real window at intervals.
- **Short engines.** Engines with fewer than `input_window + 1` rows are left out of training. They
  still appear in the output, using the start window of the trained engine with the closest length.

## Seeded generation

Adapts a trained model to new data, such as a different operating condition, without full
retraining:

```python
from Transformer_TrainingV7 import generate_from_seed_file, _resolve_path

df = generate_from_seed_file(
    base_checkpoint_path=_resolve_path("src/model_checkPoints/transformer/transformer_model.pt"),
    seed_csv_path=_resolve_path("data/new_condition.csv"),
    config_path=_resolve_path("src/config/transformer_config.yaml"),
    num_samples=2000,
    adaptation_epochs=5,   # 5 to 10; more epochs risk losing what the base model learned
    adapt_lr=1e-4,
    reseed_interval=0,     # set above 0 for long single-sequence seed files
)
```

A copy of the model is fine-tuned and discarded afterward; the saved model is not changed. This
shifts the output's means, spreads and correlations toward the seed data. It suits the same type of
wear under different conditions, not new types of wear.

Example: a model trained on FD001 and seeded with FD003 produces output closer to FD003.

![Seeded generation compared with FD003](https://github.com/user-attachments/assets/ebb151e3-b38f-4908-bad1-a24d0ec865a6)
