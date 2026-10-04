# API

A local web server that connects the dashboard to the project code. It contains no model or
statistics logic; each endpoint calls functions in `src/`.

## Run

From the repository root:

```bash
./run.sh                          # starts the server and opens the dashboard
python api/server.py 8765         # starts the server only
```

The dashboard is served at http://127.0.0.1:8765/, on the same address as the API.

## Files

| File | Purpose |
|---|---|
| `server.py` | HTTP routing, JSON and serving `dashboard/index.html`. Uses only the Python standard library. |
| `pipeline.py` | Session data, checkpoint listing, editing, validation, and background training and generation jobs |
| `events.py` | Converts the training scripts' printed output into progress events |
| `ckpt_info.py` | Reads size and parameter counts from model files |
| `record_replays.py` | Records real runs and builds the data the dashboard replays |

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Server check |
| GET | `/api/checkpoints`, `/api/checkpoints/meta` | Available models and their details |
| GET | `/api/data/rows` | Rows of the loaded data |
| POST | `/api/data/load` | Load bundled or uploaded data |
| POST | `/api/shift`, `/api/interpolate`, `/api/propagate` | Edit the synthetic data |
| POST | `/api/validate` | Run the statistical tests |
| POST | `/api/train`, `/api/generate`, `/api/fine-tune` | Start a job |
| GET | `/api/jobs/<id>`, `/api/jobs/<id>/events` | Job status and progress |
| POST | `/api/jobs/<id>/cancel` | Stop a job |
| GET | `/api/export/synth.csv` | Download the synthetic data |

No packages beyond `requirements.txt` are needed.
