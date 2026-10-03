# Data Synthesizer API

A thin HTTP adapter over the real project code (`src/features/*`, `src/training/*`,
`src/validate/*`). No model or statistics logic lives here — every endpoint
imports and calls the existing functions directly.

**Note on framework:** this environment has no PyPI access, so FastAPI/Flask
could not be installed. `api/server.py` is built on Python's standard-library
`http.server` instead (`ThreadingHTTPServer` + `BaseHTTPRequestHandler`) —
same JSON-over-HTTP contract, same CORS behavior, no third-party dependency.
Swap it for FastAPI later with no change to `api/pipeline.py` or the contract.

## Run

```
./run.sh            # from the repo root — starts the server and opens the dashboard
# or directly:
.venv/bin/python api/server.py 8765
```

Then open **http://127.0.0.1:8765/** — the dashboard is served from the same
origin as the API (avoids any cross-origin/mixed-content issue with a
browser-hosted page fetching `http://127.0.0.1`).

## Files

- `pipeline.py` — pure business logic (session state, checkpoint listing,
  shift/interpolate/propagate orchestration, the real validation suite,
  background job runners for train/generate/fine-tune). No HTTP here.
- `server.py` — the HTTP layer: routing, JSON, CORS, and serving the
  demo page `dashboard/index.html` at `/`.

## Dependencies

Everything `pipeline.py` imports (`pandas`, `numpy`, `torch`, `matplotlib`,
`seaborn`, `scipy`, `scikit-learn`, `PyYAML`) is already in the project's own
`requirements.txt` / `.venv` — no new Python packages were added.
