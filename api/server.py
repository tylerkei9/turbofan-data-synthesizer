"""
Data Synthesizer API: a standard-library HTTP server that wraps api/pipeline.py.

Run:
    .venv/bin/python api/server.py [port]

Every route is namespaced under /api. CORS is wide open (Access-Control-
Allow-Origin: *) since this is a local dev tool meant to be called from a
browser-hosted dashboard on a different origin.
"""

from __future__ import annotations

import json
import sys
import threading
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pipeline as pl


def _json_default(o):
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


class Handler(BaseHTTPRequestHandler):
    server_version = "DataSynthAPI/1"

    def log_message(self, fmt, *args):  # quieter default logging
        sys.stderr.write("[api] " + (fmt % args) + "\n")

    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, default=_json_default).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, text: str) -> None:
        body = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_csv(self, text: str, filename: str) -> None:
        body = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/csv")
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0) or 0)
        if not length:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw) if raw else {}

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path, qs = parsed.path, parse_qs(parsed.query)
        try:
            if path in ("/", "/dashboard"):
                dash = Path(__file__).resolve().parent.parent / "dashboard" / "index.html"
                self._send_html(dash.read_text(encoding="utf-8") if dash.exists() else "<h1>dashboard/index.html not found</h1>")
            elif path == "/api/health":
                self._send(200, {"ok": True})
            elif path == "/api/checkpoints":
                self._send(200, pl.list_checkpoints())
            elif path == "/api/data/rows":
                kind = qs.get("kind", ["synth"])[0]
                sensors = qs.get("sensors", [""])[0].split(",") if qs.get("sensors") else []
                engines = [int(e) for e in qs.get("engines", [""])[0].split(",") if e] if qs.get("engines") else None
                df = pl.SESSION.real_data if kind == "real" else pl.SESSION.synth_data
                if df is None:
                    self._send(200, {"rows": []})
                else:
                    sens = sensors or pl.sensor_columns(df)[:3]
                    self._send(200, {"rows": pl.compact_rows(df, sens, engines), "sensors": sens})
            elif path == "/api/checkpoints/meta":
                from ckpt_info import adapter_paths, checkpoint_files
                self._send(200, {"files": checkpoint_files(), "adapters": [str(p.relative_to(pl.PROJECT_ROOT)) for p in adapter_paths()]})
            elif path.startswith("/api/jobs/") and path.endswith("/events"):
                job = pl.SESSION.jobs.get(path.split("/")[3])
                if not job:
                    self._send(404, {"error": "unknown job id"})
                else:
                    since = int(qs.get("since", ["0"])[0])
                    events = job["events"][since:]
                    self._send(200, {"status": job["status"], "events": events, "next": since + len(events), "result": job.get("result")})
            elif path.startswith("/api/jobs/"):
                job_id = path.rsplit("/", 1)[-1]
                job = pl.SESSION.jobs.get(job_id)
                if not job:
                    self._send(404, {"error": "unknown job id"})
                else:
                    self._send(200, {"status": job["status"], "log": job.get("log", []), "result": job.get("result")})
            elif path == "/api/export/synth.csv":
                self._send_csv(pl.export_synth_csv(), "synthetic_data.csv")
            else:
                self._send(404, {"error": f"no route {path}"})
        except Exception as e:
            traceback.print_exc()
            self._send(500, {"error": str(e)})

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            body = self._body()
            if path == "/api/data/load":
                kind, source = body["kind"], body.get("source", "bundled")
                if source == "bundled":
                    df = pl.load_bundled(kind)
                    name = ("FD001.csv" if kind == "real" else "synthetic_demo.csv") + " (bundled)"
                else:
                    df = pl.load_csv_text(body["text"], body.get("filename", "upload.csv"))
                    name = body.get("filename", "upload.csv")
                self._send(200, pl.set_data(kind, df, name))
            elif path == "/api/shift":
                stats = pl.apply_shift_to_session(body["column"], float(body["amount"]))
                self._send(200, {"stats": stats})
            elif path == "/api/interpolate":
                info = pl.apply_interpolation_to_session(body["columns"], body.get("method", "linear"), body.get("gap"))
                self._send(200, info)
            elif path == "/api/propagate":
                info = pl.apply_propagation_to_session(body["columns"], body.get("cutoff"))
                self._send(200, info)
            elif path == "/api/validate":
                self._send(200, pl.run_validation_suite(body.get("conditional")))
            elif path.startswith("/api/jobs/") and path.endswith("/cancel"):
                job = pl.SESSION.jobs.get(path.split("/")[3])
                self._send(404 if not job else 200, {"error": "unknown job id"} if not job else {"cancelled": pl.cancel_job(job)})
            elif path == "/api/train":
                job_id = str(uuid.uuid4())[:8]
                job = pl.new_job(job_id)
                pl.SESSION.jobs[job_id] = job
                threading.Thread(target=self._run(pl.job_train, job, body["model"], body.get("model_name", "api_model"),
                                                  body.get("opts"), bool(body.get("generate_after", True))), daemon=True).start()
                self._send(200, {"job_id": job_id})
            elif path == "/api/generate":
                job_id = str(uuid.uuid4())[:8]
                job = pl.new_job(job_id)
                pl.SESSION.jobs[job_id] = job
                threading.Thread(
                    target=self._run(
                        pl.job_generate, job, body["model"], body["checkpoint"], bool(body.get("seed", False)),
                        body.get("n_engines", 5), body.get("cycles_per_engine", 30), body.get("arch", "transformer"), body.get("opts"),
                    ), daemon=True,
                ).start()
                self._send(200, {"job_id": job_id})
            elif path == "/api/fine-tune":
                job_id = str(uuid.uuid4())[:8]
                job = pl.new_job(job_id)
                pl.SESSION.jobs[job_id] = job
                threading.Thread(
                    target=self._run(
                        pl.job_finetune, job, body["base_checkpoint"], body["seed_paths"], body.get("adapter_name", "adapter_api.pt"),
                        int(body.get("lora_rank", 4)), float(body.get("lora_alpha", 8.0)),
                        int(body.get("epochs", 8)), float(body.get("lr", 1e-5)), int(body.get("batch_size", 64)),
                    ), daemon=True,
                ).start()
                self._send(200, {"job_id": job_id})
            else:
                self._send(404, {"error": f"no route {path}"})
        except Exception as e:
            traceback.print_exc()
            self._send(500, {"error": str(e)})

    @staticmethod
    def _run(fn, job, *args):
        def wrapped():
            try:
                fn(job, *args)
            except Exception as e:
                traceback.print_exc()
                job["status"] = "error"
                job["result"] = {"error": str(e)}
                pl._emit(job, {"type": "error", "message": str(e)})
        return wrapped


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Data Synthesizer API listening on http://127.0.0.1:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
