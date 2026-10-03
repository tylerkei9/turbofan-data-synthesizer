"""
Pure business logic for the Data Synthesizer API — no HTTP here.

Wraps the real project code (src/features/*, src/training/*, src/validate/*)
behind plain Python functions that server.py calls. Keeping this separate
from server.py means the pipeline logic is testable without spinning up an
HTTP server.
"""

from __future__ import annotations

import contextlib
import io
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")  # must happen before src.validate.* imports pyplot — the
                        # HTTP server runs each request on a worker thread, and
                        # macOS's default GUI backend can only draw on the main one.
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.shifting import apply_shift, calculate_shift_statistics, get_shiftable_columns
from src.features.interpolation import apply_interpolation
from src.validate.Descriptive_Stats_Validation import run_descriptive_stats
from src.validate.Covariance_and_Graphs_Validation import run_covariance_and_graphs
from src.validate.Conditional_Validation_Script import run_conditional_validation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from events import LogParser  # noqa: E402

try:
    from src.features.propagation import apply_propagation
    PROPAGATION_AVAILABLE = True
    PROPAGATION_ERROR = ""
except ImportError as exc:  # torch not installed
    PROPAGATION_AVAILABLE = False
    PROPAGATION_ERROR = str(exc)

TRANSFORMER_SCRIPT = PROJECT_ROOT / "src" / "training" / "Transformer" / "Transformer_TrainingV7.py"
DIFFUSION_SCRIPT = PROJECT_ROOT / "src" / "training" / "Diffusion" / "diffusion_model5.py"
TRANSFORMER_CKPT_DIR = PROJECT_ROOT / "src" / "model_checkPoints" / "transformer"
DIFFUSION_CKPT_ROOT = PROJECT_ROOT / "src" / "model_checkPoints"
DIFFUSION_ADAPTERS_DIR = DIFFUSION_CKPT_ROOT / "adapters"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

_ID_COLS = {"Engine_ID", "Time_in_cycles", "Altitude", "Mach_number", "TRA"}


# ── session (single in-memory session — this is a local dev tool, not a
#    multi-tenant service) ───────────────────────────────────────────────

class Session:
    def __init__(self) -> None:
        self.real_data: pd.DataFrame | None = None
        self.synth_data: pd.DataFrame | None = None
        self.synth_data_original: pd.DataFrame | None = None
        self.real_name: str = ""
        self.synth_name: str = ""
        self.last_checkpoint_path: str | None = None
        self.last_model_type: str | None = None
        self.last_config_path: str | None = None
        self.jobs: dict[str, dict[str, Any]] = {}


SESSION = Session()


# ── data loading ─────────────────────────────────────────────────────────

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def load_csv_text(text: str, filename: str = "upload.csv") -> pd.DataFrame:
    """Parse CSV or whitespace-delimited TXT (raw NASA format), like the app did."""
    buf = io.StringIO(text)
    try:
        df = pd.read_csv(buf)
        if df.shape[1] < 2:
            raise ValueError("too few columns for comma CSV")
    except Exception:
        buf.seek(0)
        df = pd.read_csv(buf, sep=r"\s+", header=None)
    return _normalize_columns(df)


def load_bundled(kind: str) -> pd.DataFrame:
    path = DATA_DIR / ("FD001.csv" if kind == "real" else "synthetic_demo.csv")
    return _normalize_columns(pd.read_csv(path))


def sensor_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _ID_COLS and str(c).isdigit()]


def set_data(kind: str, df: pd.DataFrame, name: str) -> dict:
    if kind == "real":
        SESSION.real_data = df
        SESSION.real_name = name
    else:
        SESSION.synth_data = df
        SESSION.synth_data_original = df.copy()
        SESSION.synth_name = name
    return summarize(df, name)


def summarize(df: pd.DataFrame, name: str) -> dict:
    return {
        "name": name,
        "rows": int(len(df)),
        "engines": int(df["Engine_ID"].nunique()) if "Engine_ID" in df.columns else 0,
        "sensors": sensor_columns(df),
    }


def compact_rows(df: pd.DataFrame, sensors: list[str], engines: list[int] | None = None) -> list[dict]:
    """Down-convert a full real DataFrame to the frontend's {e,c,s<N>...} shape."""
    if df is None:
        return []
    f = df
    if engines:
        f = f[f["Engine_ID"].isin(engines)]
    cols = {"Engine_ID": "e", "Time_in_cycles": "c"}
    out = []
    for _, r in f.iterrows():
        row = {"e": int(r["Engine_ID"]), "c": int(r["Time_in_cycles"])}
        for s in sensors:
            if s in f.columns:
                v = r[s]
                row["s" + s] = None if pd.isna(v) else float(v)
        if "region" in f.columns:
            row["region"] = r["region"]
        out.append(row)
    return out


# ── checkpoints ──────────────────────────────────────────────────────────

def list_checkpoints() -> dict:
    transformer = sorted(str(p.relative_to(PROJECT_ROOT)) for p in TRANSFORMER_CKPT_DIR.glob("*.pt"))
    diffusion = sorted(
        str(p.relative_to(PROJECT_ROOT))
        for p in DIFFUSION_CKPT_ROOT.rglob("*.pt")
        if "adapters" not in p.parts and "transformer" not in p.parts
    )
    adapters = sorted(str(p.relative_to(PROJECT_ROOT)) for p in DIFFUSION_ADAPTERS_DIR.glob("*.pt")) if DIFFUSION_ADAPTERS_DIR.exists() else []
    return {"transformer": transformer, "diffusion": diffusion, "adapters": adapters}


def resolve_propagation_assets() -> tuple[str | None, str | None, str | None]:
    """Same preference order the Streamlit app used: last-used -> newest transformer -> newest diffusion."""
    if SESSION.last_model_type in {"diffusion", "transformer"} and SESSION.last_checkpoint_path and Path(SESSION.last_checkpoint_path).exists():
        return SESSION.last_model_type, SESSION.last_checkpoint_path, SESSION.last_config_path

    transformer_ckpts = sorted(TRANSFORMER_CKPT_DIR.glob("*.pt"))
    if transformer_ckpts:
        cfg = PROJECT_ROOT / "src" / "config" / "transformer_config.yaml"
        return "transformer", str(transformer_ckpts[-1]), str(cfg if cfg.exists() else "")

    diffusion_ckpts = sorted(DIFFUSION_CKPT_ROOT.rglob("diff_best_ckpt.pt"))
    if diffusion_ckpts:
        return "diffusion", str(diffusion_ckpts[-1]), None

    return None, None, None


def _standardize_propagation_output(output_df: pd.DataFrame, original_df: pd.DataFrame) -> pd.DataFrame:
    standardized = _normalize_columns(output_df)
    original_cols = [str(c) for c in original_df.columns]
    if "cycle" in standardized.columns and "Time_in_cycles" not in standardized.columns:
        standardized = standardized.rename(columns={"cycle": "Time_in_cycles"})
    # keep 'region' (unlike the old app, the API wants it for the chart) but
    # still drop other debug columns the model bundle may add
    for debug_col in ("norm_source",):
        if debug_col in standardized.columns and debug_col not in original_cols:
            standardized = standardized.drop(columns=debug_col)
    ordered = [c for c in original_cols if c in standardized.columns]
    extra = [c for c in standardized.columns if c not in ordered]
    return standardized[ordered + extra]


def apply_propagation_to_session(prop_cols: list[str], cutoff: int | None = None) -> dict:
    if not PROPAGATION_AVAILABLE:
        raise RuntimeError(f"Propagation unavailable: {PROPAGATION_ERROR}")
    if SESSION.synth_data is None:
        raise RuntimeError("No synthetic data loaded.")
    modified = SESSION.synth_data.copy()
    model_type, checkpoint_path, config_path = resolve_propagation_assets()
    if not model_type or not checkpoint_path:
        raise RuntimeError("No propagation checkpoint is available. Train or generate a model first.")

    if model_type == "transformer":
        if len(modified) < 2:
            raise ValueError("Transformer propagation needs at least 2 rows of synthetic data.")
        protected = {"Engine_ID", "Time_in_cycles", "__sequence_id__", "__row_idx__"}
        feature_cols = [c for c in modified.columns if c not in protected]
        immune_cols = [c for c in feature_cols if c not in prop_cols]
        if cutoff is not None and "Time_in_cycles" in modified.columns:
            observed_rows = max(1, min(len(modified) - 1, int((modified["Time_in_cycles"] <= cutoff).sum())))
        else:
            observed_rows = max(1, min(len(modified) - 1, round(len(modified) * 0.7)))
        engine_id = None
        if "Engine_ID" in modified.columns and not modified["Engine_ID"].dropna().empty:
            engine_id = modified["Engine_ID"].dropna().iloc[0]
        raw = apply_propagation(
            modified_data=modified, model_type="transformer", checkpoint_path=checkpoint_path,
            device="cpu", config_path=config_path or None, observed_rows=observed_rows,
            engine_id=engine_id, immune_cols=immune_cols, reference_data=modified,
        )
    else:
        raw = apply_propagation(
            modified_data=modified, model_type="diffusion", checkpoint_path=checkpoint_path,
            num_samples=len(modified), device="cpu",
        )

    standardized = _standardize_propagation_output(raw, modified)
    SESSION.synth_data = standardized
    return {"model_type": model_type, "checkpoint": checkpoint_path}


# ── shift / interpolate ─────────────────────────────────────────────────

def apply_shift_to_session(column: str, amount: float) -> dict:
    if SESSION.synth_data is None:
        raise RuntimeError("No synthetic data loaded.")
    before = SESSION.synth_data
    after = apply_shift(before, {column: amount})
    stats = calculate_shift_statistics(before, after, column) if column in before.columns else {}
    SESSION.synth_data = after
    return stats


def apply_interpolation_to_session(columns: list[str], method: str, gap: dict | None = None) -> dict:
    if SESSION.synth_data is None:
        raise RuntimeError("No synthetic data loaded.")
    before = SESSION.synth_data.copy()
    if gap and "Engine_ID" in before.columns and "Time_in_cycles" in before.columns:
        # Demo aid: the real bundled data has no natural gaps, so the caller
        # can ask for one to be punched in before the real fill runs.
        lo = gap["start"]
        hi = gap["start"] + gap["length"]
        mask = (before["Engine_ID"] == gap["engine"]) & (before["Time_in_cycles"] >= lo) & (before["Time_in_cycles"] < hi)
        for c in columns:
            if c in before.columns:
                before.loc[mask, c] = float("nan")
        SESSION.synth_data = before
    pre_mask = {c: before[c].isna().copy() for c in columns if c in before.columns}
    after = apply_interpolation(
        before, method=method, target_columns=columns,
        x_column="Time_in_cycles" if "Time_in_cycles" in before.columns else None,
        engine_id_column="Engine_ID" if "Engine_ID" in before.columns else None,
    )
    SESSION.synth_data = after
    filled = {}
    if "is_interpolated" in after.columns:
        filled = {c: after["is_interpolated"].astype(bool).tolist() for c in columns}
    else:
        for c in columns:
            if c in pre_mask and len(pre_mask[c]) == len(after):
                filled[c] = (pre_mask[c] & ~after[c].isna()).tolist()
    return {"filled_mask": filled}


# ── validate ─────────────────────────────────────────────────────────────

def run_validation_suite(conditional: dict | None = None) -> dict:
    if SESSION.real_data is None or SESSION.synth_data is None:
        raise RuntimeError("Load both real and synthetic data first.")
    tmp = Path(tempfile.mkdtemp())
    result: dict[str, Any] = {"tables": {}, "images": {}, "log": "", "errors": []}
    try:
        real_path = tmp / "real.csv"
        synth_path = tmp / "synth.csv"
        SESSION.real_data.to_csv(real_path, index=False)
        SESSION.synth_data.to_csv(synth_path, index=False)

        out_desc, out_cov, out_cond = tmp / "desc", tmp / "cov", tmp / "cond"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            try:
                run_descriptive_stats(str(real_path), str(synth_path), str(out_desc))
            except Exception as e:
                result["errors"].append(f"Descriptive stats failed: {e}")
            try:
                run_covariance_and_graphs(str(real_path), str(synth_path), str(out_cov))
            except Exception as e:
                result["errors"].append(f"Covariance/graphs failed: {e}")
            try:
                answers = [""]
                if conditional and conditional.get("column"):
                    answers = [conditional["column"], str(conditional.get("center", 0)), str(conditional.get("tolerance", 0))]
                with patch("builtins.input", side_effect=answers):
                    run_conditional_validation(str(real_path), str(synth_path), str(out_cond))
            except Exception as e:
                result["errors"].append(f"Conditional validation failed: {e}")
        result["log"] = buf.getvalue()

        def load_csv(p: Path, index_col=None):
            if p.exists():
                result["tables"][p.name] = pd.read_csv(p, index_col=index_col).fillna("null").to_dict(orient="index" if index_col is not None else "records")

        def load_png(p: Path):
            if p.exists():
                import base64
                result["images"][p.name] = base64.b64encode(p.read_bytes()).decode("ascii")

        load_csv(out_desc / "descriptive_stats.csv", index_col=0)
        load_csv(out_cov / "summary_comparison.csv", index_col=0)
        load_csv(out_cov / "statistical_tests.csv")
        load_png(out_cov / "correlation_diff.png")
        load_png(out_cov / "pca_comparison.png")
        for png in sorted(out_cov.glob("dist_*.png")):
            load_png(png)

        cond_dirs = list(out_cond.glob("*")) if out_cond.exists() else []
        if cond_dirs:
            cdir = cond_dirs[0]
            load_csv(cdir / "summary_filtered.csv", index_col=0)
            load_csv(cdir / "statistical_tests_filtered.csv")
            load_png(cdir / "correlation_diff_filtered.png")

        return result
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── export ────────────────────────────────────────────────────────────────

def export_synth_csv() -> str:
    if SESSION.synth_data is None:
        raise RuntimeError("No synthetic data loaded.")
    return SESSION.synth_data.to_csv(index=False)


# ── background jobs (train / generate / fine-tune via the real scripts) ──
#
# Each job dict carries "events": [[ms since job start, event], ...] in the same
# typed format the dashboard replays (api/events.py parses the script output),
# so the dashboard polls /api/jobs/<id>/events?since=n and renders a live run
# exactly like a recorded one.

def new_job(job_id: str) -> dict:
    return {"id": job_id, "status": "running", "log": [], "events": [], "t0": time.monotonic(), "result": None}


def _emit(job: dict, event: dict) -> None:
    job["events"].append([int((time.monotonic() - job["t0"]) * 1000), event])


def _run_subprocess(cmd: list[str], job: dict, parser: LogParser | None = None) -> int:
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}  # stream lines as they are printed, not in one burst at exit
    proc = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            cwd=str(PROJECT_ROOT), bufsize=1, env=env)
    job["_proc"] = proc
    for raw in proc.stdout:
        line = raw.rstrip()
        job["log"].append(line)
        if parser is not None:
            for _, event in parser.feed(time.monotonic() - job["t0"], line):
                _emit(job, event)
    proc.wait()
    job.pop("_proc", None)
    return proc.returncode


def cancel_job(job: dict) -> bool:
    proc = job.get("_proc")
    if job.get("status") != "running":
        return False
    job["status"] = "cancelled"
    if proc is not None:
        proc.terminate()
    _emit(job, {"type": "log", "line": "KeyboardInterrupt: run cancelled by user"})
    return True


def _finish(job: dict, ok: bool, parser: LogParser | None = None, error: str = "see log") -> None:
    if job["status"] == "cancelled":
        return
    if parser is not None and ok:
        for _, event in parser.finish(time.monotonic() - job["t0"]):
            _emit(job, event)
    if ok:
        _emit(job, {"type": "done"})
        job["status"] = "done"
    else:
        tail = next((l for l in reversed(job["log"]) if l.strip()), error)
        _emit(job, {"type": "error", "message": tail})
        job["status"] = "error"


def _size(path: Path) -> str:
    return f"{path.stat().st_size / 1e6:.1f} MB" if path.exists() else "–"


def _temp_yaml(cfg: dict, name: str) -> Path:
    import yaml
    path = Path(tempfile.mkdtemp()) / name
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    return path


def _transformer_config(model_name: str, opts: dict) -> Path:
    """The repo's transformer config with the run name and any dashboard settings applied."""
    import yaml
    with open(PROJECT_ROOT / "src" / "config" / "transformer_config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("model", {})["name"] = model_name  # the script reads the run name from here, not from a CLI flag
    rt = cfg.setdefault("runtime", {})
    for key in ("epochs", "input_window", "lr", "weight_decay", "lambda_var", "batch_size"):
        if opts.get(key) is not None:
            rt[key] = opts[key]
    gen = cfg.setdefault("generation", {})
    for key in ("strategy", "n_engines", "num_samples", "reseed_interval"):
        if key in opts:
            gen[key] = opts[key]
    if "cycles_per_engine" in opts:
        if opts["cycles_per_engine"] is None:
            gen.pop("cycles_per_engine", None)  # no key -> each engine at its real length
        else:
            gen["cycles_per_engine"] = opts["cycles_per_engine"]
    return _temp_yaml(cfg, "run_config.yaml")


def _diffusion_config(opts: dict, keep_data: bool = True) -> Path:
    """The repo's diffusion config with dashboard settings applied (T is only settable here)."""
    import yaml
    with open(PROJECT_ROOT / "src" / "config" / "diffusion_config.yaml") as f:
        cfg = yaml.safe_load(f)
    if not keep_data:
        cfg.pop("data", None)
    if opts.get("hidden_dim") is not None:
        cfg.setdefault("model", {})["hidden_dim"] = opts["hidden_dim"]
    for key in ("epochs", "batch_size", "lr"):
        if opts.get(key) is not None:
            cfg.setdefault("training", {})[key] = opts[key]
    if opts.get("T") is not None:
        cfg.setdefault("diffusion", {})["T"] = opts["T"]
    return _temp_yaml(cfg, "diffusion_run.yaml")


def job_train(job: dict, model: str, model_name: str, opts: dict | None = None, generate_after: bool = True) -> None:
    opts = opts or {}
    if model == "transformer":
        parser = LogParser("train_t", epochs=opts.get("epochs"))
        for e in parser.start():
            _emit(job, e[1])
        cfg_path = _transformer_config(model_name, opts)
        rc = _run_subprocess([sys.executable, TRANSFORMER_SCRIPT, "--mode", "train", "--config", cfg_path], job, parser)
        ckpt = TRANSFORMER_CKPT_DIR / f"{model_name}.pt"
        csv = None
        if rc == 0 and generate_after and job["status"] == "running":
            rc = _run_subprocess([sys.executable, TRANSFORMER_SCRIPT, "--mode", "generate", "--config", cfg_path,
                                  "--num_samples", "2000", "--checkpoint", ckpt, "--run_name", model_name], job)
            out_dir = OUTPUTS_DIR / "Transformer" / "synthetic_data" / model_name
            csvs = sorted(out_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime) if out_dir.exists() else []
            csv = str(csvs[-1]) if csvs else None
        job["result"] = {"checkpoint": str(ckpt), "csv": csv, "model_type": "transformer"}
        meta = {"type": "transformer", "arch": "transformer", "epochs": parser.n_metrics, "best": parser.best,
                "params": parser.params, "size": _size(ckpt), "input_window": opts.get("input_window"), "src": "live backend"}
    else:
        parser = LogParser("train_d", epochs=opts.get("epochs"))
        for e in parser.start():
            _emit(job, e[1])
        cmd = [sys.executable, DIFFUSION_SCRIPT, "--mode", "train", "--model-name", model_name, "--config", _diffusion_config(opts),
               "--data", "data/FD001.csv", "--output-dir", OUTPUTS_DIR, "--arch", opts.get("arch", "transformer"), "--no-auto-sample"]
        for flag, key in (("--n-heads", "n_heads"), ("--n-layers", "n_layers")):
            if opts.get(key) is not None:
                cmd += [flag, opts[key]]
        rc = _run_subprocess(cmd, job, parser)
        ckpt = DIFFUSION_CKPT_ROOT / "diffusion" / model_name / "diff_best_ckpt.pt"
        if not ckpt.exists():
            ckpt = DIFFUSION_CKPT_ROOT / model_name / "diff_best_ckpt.pt"
        job["result"] = {"checkpoint": str(ckpt), "csv": None, "model_type": "diffusion"}
        eff = parser.effective
        meta = {"type": "diffusion", "arch": opts.get("arch", "transformer"), "epochs": parser.n_metrics, "best": parser.best,
                "params": parser.params, "size": _size(ckpt), "T": int(float(eff["T"])) if "T" in eff else None,
                "lr": float(eff["lr"]) if "lr" in eff else None, "bs": int(float(eff["batch_size"])) if "batch_size" in eff else None,
                "hid": int(float(eff["hidden_dim"])) if "hidden_dim" in eff else None, "src": "live backend"}
    ok = rc == 0 and ckpt.exists()
    if ok and job["status"] == "running":
        _emit(job, {"type": "artifact", "kind": "checkpoint", "path": str(ckpt.relative_to(PROJECT_ROOT)), "meta": meta})
        SESSION.last_checkpoint_path = job["result"]["checkpoint"]
        SESSION.last_model_type = job["result"]["model_type"]
    _finish(job, ok, parser)


def job_generate(job: dict, model: str, checkpoint: str, seed: bool, n_engines: int | None, cycles_per_engine: int | None,
                 arch: str = "transformer", opts: dict | None = None) -> None:
    """Generate synthetic rows and make them the session's synthetic data.

    Transformer: every engine is seeded with its first input_window real rows; n_engines None means all
    engines and cycles_per_engine None means each engine's real length.
    Diffusion: seed=True holds the auto-selected condition columns at values from real rows
    (--observe conditions) and samples the rest; seed=False samples every column.
    """
    opts = opts or {}
    out_name = f"api_gen_{job['id']}.csv"
    if model == "transformer":
        parser = LogParser("gen_t")
        for e in parser.start():
            _emit(job, e[1])
        out_dir = OUTPUTS_DIR / "Transformer" / "synthetic_data" / "api"
        out_dir.mkdir(parents=True, exist_ok=True)
        before = set(out_dir.glob("*.csv"))
        cfg_path = _transformer_config("api", {**opts, "strategy": opts.get("strategy", "by_engines"),
                                               "n_engines": n_engines or 100, "cycles_per_engine": cycles_per_engine})
        rc = _run_subprocess([sys.executable, TRANSFORMER_SCRIPT, "--mode", "generate", "--config", cfg_path,
                              "--checkpoint", checkpoint, "--run_name", "api"], job, parser)
        new = sorted(set(out_dir.glob("*.csv")) - before, key=lambda p: p.stat().st_mtime)
        csv_path = new[-1] if new else None
    else:
        parser = LogParser("gen_d")
        for e in parser.start():
            _emit(job, e[1])
        out_dir = OUTPUTS_DIR / "diffusion" / "api" / "synthetic_data"
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, DIFFUSION_SCRIPT, "--mode", "generate", "--model-name", "api", "--config", _diffusion_config({}, keep_data=False),
               "--arch", arch, "--output-dir", OUTPUTS_DIR, "--checkpoint", checkpoint,
               "--n-engines", n_engines or 4, "--cycles-per-engine", cycles_per_engine or 90, "--out", out_name]
        if seed:
            cmd += ["--data", "data/FD001.csv", "--observe", "conditions"]
        if opts.get("adapters"):
            cmd += ["--adapters", *opts["adapters"]]
        rc = _run_subprocess(cmd, job, parser)
        csv_path = out_dir / out_name
        csv_path = csv_path if csv_path.exists() else None
    ok = rc == 0 and csv_path is not None
    if ok and job["status"] == "running":
        df = _normalize_columns(pd.read_csv(csv_path))
        offset = 0
        if model == "transformer":
            # generated row t continues real cycle t + input_window (the seed rows are not written)
            import torch
            ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
            offset = int(ck.get("INPUT_WINDOW") or ck.get("config", {}).get("runtime", {}).get("input_window", 0) or 0)
            df["Time_in_cycles"] = df["Time_in_cycles"] + offset
        SESSION.synth_data = df
        SESSION.synth_data_original = df.copy()
        SESSION.synth_name = csv_path.name
        SESSION.last_checkpoint_path = checkpoint
        SESSION.last_model_type = model
        job["result"] = {**summarize(df, csv_path.name), "cycle_offset": offset}
        _emit(job, {"type": "artifact", "kind": "synthetic", "path": str(csv_path.relative_to(PROJECT_ROOT)),
                    "meta": {"rows": int(len(df)), "engines": job["result"]["engines"], "ckpt": Path(checkpoint).name,
                             "adapters": [Path(a).name for a in opts.get("adapters") or []], "model": model,
                             "cycle_offset": offset, "src": "live backend", "ops": []}})
    elif not ok:
        job["result"] = {"error": "generation failed, see log"}
    _finish(job, ok, parser)


def job_finetune(job: dict, base_checkpoint: str, seed_paths: list[str], adapter_name: str,
                  lora_rank: int, lora_alpha: float, epochs: int, lr: float, batch_size: int) -> None:
    parser = LogParser("finetune", epochs=epochs)
    for e in parser.start():
        _emit(job, e[1])
    DIFFUSION_ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)
    adapter_out = DIFFUSION_ADAPTERS_DIR / adapter_name
    cmd = [
        sys.executable, DIFFUSION_SCRIPT, "--mode", "fine-tune", "--model-name", "diffusion",
        "--checkpoint", base_checkpoint, "--seed-data", *seed_paths, "--adapter-out", adapter_out,
        "--lora-rank", lora_rank, "--lora-alpha", f"{lora_alpha:.4f}",
        "--epochs", epochs, "--lr", f"{lr:.8f}", "--batch-size", batch_size,
        "--no-auto-sample",
    ]
    rc = _run_subprocess(cmd, job, parser)
    ok = rc == 0 and adapter_out.exists()
    job["result"] = {"adapter": str(adapter_out)} if ok else {"error": "fine-tune failed, see log"}
    if ok and job["status"] == "running":
        _emit(job, {"type": "artifact", "kind": "adapter", "path": str(adapter_out.relative_to(PROJECT_ROOT)),
                    "meta": {"rank": lora_rank, "alpha": lora_alpha, "epochs": parser.n_metrics, "loss": parser.last,
                             "trainable": parser.trainable, "base": Path(base_checkpoint).name, "src": "live backend"}})
    _finish(job, ok, parser)
