# Running the code

This guide is for anyone who wants to run the models themselves. To just see the project, open
`dashboard/index.html` instead; it needs none of this.

## 1. Set up (one time)

You need Python 3.10 or newer. From the repository folder:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

PyTorch installs its CPU version by default, which is enough for everything here. For an NVIDIA GPU,
first install PyTorch with the command from https://pytorch.org/get-started/locally/ and then run the
`pip install` line.

## 2. Open the dashboard through the local server

```bash
./run.sh                 # starts api/server.py on http://127.0.0.1:8765/ and opens the browser
```

`python api/server.py 8765` does the same without opening a browser. The server also provides a small
API (`/api/health`, `/api/checkpoints/meta`, `/api/train`, `/api/generate`, `/api/fine-tune`,
`/api/validate`, `/api/jobs/<id>/events`); see [api/README.md](../api/README.md).

## 3. Train, generate, fine-tune and validate from the command line

All commands are run from the repository folder. Trained models are saved under
`src/model_checkPoints/` and generated data under `outputs/` (both are created automatically).

**Transformer** (settings in `src/config/transformer_config.yaml`; set `runtime.epochs` there):

```bash
python src/training/Transformer/Transformer_TrainingV7.py --mode train --config src/config/transformer_config.yaml
python src/training/Transformer/Transformer_TrainingV7.py --mode generate --config src/config/transformer_config.yaml \
    --checkpoint src/model_checkPoints/transformer/<model name>.pt --run_name my_run
```

**Diffusion** (settings in `src/config/diffusion_config.yaml`; 100 epochs takes about 30 minutes on a laptop CPU):

```bash
python src/training/Diffusion/diffusion_model5.py --mode train --arch transformer --model-name my_diffusion \
    --data data/FD001.csv --output-dir outputs --no-auto-sample

python src/training/Diffusion/diffusion_model5.py --mode fine-tune --model-name my_diffusion \
    --checkpoint src/model_checkPoints/diffusion/my_diffusion/diff_best_ckpt.pt --seed-data data/FD001.csv \
    --adapter-out src/model_checkPoints/diffusion/adapters/my_lora.pt \
    --lora-rank 4 --lora-alpha 8 --epochs 30 --lr 0.0005 --batch-size 128 --no-auto-sample

python src/training/Diffusion/diffusion_model5.py --mode generate --arch transformer --model-name my_diffusion \
    --checkpoint src/model_checkPoints/diffusion/my_diffusion/diff_best_ckpt.pt \
    --adapters src/model_checkPoints/diffusion/adapters/my_lora.pt \
    --n-engines 2 --cycles-per-engine 137 --output-dir outputs --out samples.csv
```

When generating, `--observe` controls which columns are copied from real data: `none` (the default)
samples every column; `conditions` copies the columns the model chose to condition on during training
(picked automatically from the data) from rows of `--data`, and samples the rest.

**Validate** any synthetic CSV against real data (writes tables and charts to `--out`):

```bash
python src/validate/validate.py --real data/FD001.csv --synthetic outputs/<path to>/samples.csv --out validation_outputs
```

## 4. Run the tests

```bash
python src/training/Diffusion/test_diffusion_advanced.py
python src/training/Diffusion/test_diffusion_columns.py
python src/training/Diffusion/test_diffusion_multifile.py
python src/features/interpolation_testing/test_interpolation_integration.py
```

The two propagation tests (`src/features/propagation_testing/test_propagation_*.py`) need trained
checkpoints at the paths named inside them, so train a model first. Their final "generate" and
"end-to-end" steps currently fail even with checkpoints present; this is a known open issue in the
propagation feature.

## 5. Rebuild the dashboard's data

The dashboard shows recorded runs. To re-record them on your machine:

```bash
# Transformer runs, the bundled sample, and its one-step predictions (about 2 minutes)
python api/record_replays.py train_t gen_t sample predict

# Diffusion: 100-epoch training, 30-epoch LoRA fine-tune, and generation (about 45 minutes)
python api/record_replays.py train_d_full finetune_full_ft gen_d_full gen_d_full_ft

# Package the results for the dashboard, then embed them in the page
python api/record_replays.py dsample drev compare ckpt_meta replays
python dashboard/embed_data.py
```

Each step writes a timestamped capture to `outputs/demo_recordings/`, and the packaging steps write
`dashboard/demo_data.js`, `dashboard/replays.json` and `dashboard/ckpt_meta.json`. `compare` scores
every model it finds on disk and skips any that have not been run.

Training uses fixed random seeds, but results can still differ slightly between machines and library
versions, so a rebuilt dashboard may show slightly different numbers from the published one.
