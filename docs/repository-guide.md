# Repository guide

What every folder and file is for.

## Top level

| Path | Purpose |
|---|---|
| `README.md` | Start here: what the project is, the result, and how to see it. |
| `requirements.txt` | The Python packages the code needs. |
| `run.sh` | Starts the local server and opens the dashboard. |
| `.gitignore` | Keeps trained models, generated outputs and local environments out of the repository. |

## `dashboard/`: the interactive demo

| Path | Purpose |
|---|---|
| `index.html` | The whole dashboard in one file. Open it in a browser. |
| `demo_data.js` | The data the dashboard shows: real and artificial readings, the model's predictions, and the model comparison. |
| `replays.json` | Recordings of the real training and generation runs that the demo replays. |
| `ckpt_meta.json` | Facts about the trained model files (size, number of parameters). |
| `embed_data.py` | Copies the three data files above into `index.html` after they are regenerated. |
| `preview.png`, `preview-result.png` | Screenshots for sharing. |

## `data/`: the data

| Path | Purpose |
|---|---|
| `FD001.csv` | Real NASA engine records (100 engines). See [data/README.md](../data/README.md). |
| `synthetic_demo.csv` | A sample artificial dataset used by the server's "load bundled synthetic data" option. |

## `src/`: the core code

| Path | Purpose |
|---|---|
| `config/transformer_config.yaml` | Settings for the Transformer: data file, training length, model size. |
| `config/diffusion_config.yaml` | Settings for the diffusion model. |
| `config/test_diffusion_config.yaml` | Small settings used by the diffusion tests. |
| `training/Transformer/Transformer_TrainingV7.py` | The Transformer: trains on real records and generates artificial engine timelines. |
| `training/Diffusion/diffusion_model5.py` | The diffusion model: trains, fine-tunes with LoRA, and generates artificial readings. |
| `training/Diffusion/test_diffusion_*.py` | Automated checks for the diffusion model. |
| `features/shifting.py` | Edit tool: shifts one sensor's values by a set amount. |
| `features/interpolation.py`, `linear_interpolation.py`, `Bspline_MultFunc.py` | Edit tool: fills gaps in an engine's timeline (straight-line or smooth-curve). |
| `features/propagation.py` | Edit tool: keeps an engine's early cycles and lets a model generate the rest. |
| `features/interpolation_testing/`, `features/propagation_testing/` | Checks and experiments for the edit tools. |
| `validate/validate.py` | Runs the full set of statistical tests on a real and an artificial file. |
| `validate/Descriptive_Stats_Validation.py` | Compares averages, spreads and ranges. |
| `validate/Covariance_and_Graphs_Validation.py` | KS tests, correlations, PCA and distribution charts. |
| `validate/Conditional_Validation_Script.py` | The same tests restricted to chosen operating conditions. |
| `validate/validate_components.py`, `sample_validation_data.csv` | Helpers and a small example for the validation code. |

The `README.md` files inside `src/features/` and `src/training/` give technical detail for developers.

## `api/`: the local server and recorder

| Path | Purpose |
|---|---|
| `server.py` | A small web server: serves the dashboard and lets it start training, generation and validation jobs. |
| `pipeline.py` | The logic behind each server request: loading data, running jobs, editing and validating. |
| `events.py` | Turns the training programs' text output into progress updates (used live and in recordings). |
| `ckpt_info.py` | Reads facts (size, parameter count) from trained model files. |
| `record_replays.py` | Records real runs and packages everything the dashboard shows. |
| `README.md` | Technical notes on the API. |

## `docs/`: guides

| Path | Purpose |
|---|---|
| `how-it-works.md` | The process, step by step, in plain language. |
| `results.md` | What was tested, the numbers, and how the test was kept fair. |
| `dashboard-guide.md` | What each screen and button of the dashboard does. |
| `glossary.md` | Definitions of every technical term. |
| `running-the-code.md` | Setup, commands, tests and rebuilding the dashboard's data. |
| `repository-guide.md` | This file. |
