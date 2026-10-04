"""Record real Quick-preset runs and rebuild the dashboard's bundled demo sample.

Runs the project's own training / generation / fine-tune scripts, captures every
stdout line with its timestamp, and converts the capture into the typed event
stream the dashboard replays (phase / progress / metric / best / log / rows /
artifact / done). Nothing is synthesized: every number in an event is parsed
from a line the real script printed.

Usage (from the repo root):
    .venv/bin/python api/record_replays.py all
    .venv/bin/python api/record_replays.py train_t gen_t sample      # individual steps

Outputs:
    outputs/demo_recordings/<step>.json   raw timestamped captures (gitignored)
    dashboard/replays.json                event streams embedded by the dashboard build
    dashboard/demo_data.js                bundled real + synthetic sample (window.CM)
    dashboard/ckpt_meta.json              saved-checkpoint metadata and real parameter counts
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ckpt_info import checkpoint_files  # noqa: E402
from events import LogParser  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
T_SCRIPT = ROOT / "src" / "training" / "Transformer" / "Transformer_TrainingV7.py"
D_SCRIPT = ROOT / "src" / "training" / "Diffusion" / "diffusion_model5.py"
REC = ROOT / "outputs" / "demo_recordings"
REPLAYS = ROOT / "dashboard" / "replays.json"
DEMO_DATA = ROOT / "dashboard" / "demo_data.js"

QUICK = {"t_epochs": 8, "d_epochs": 6, "ft_epochs": 5, "lora_rank": 4, "lora_alpha": 8.0}
T_NAME, D_NAME, ADAPTER = "demo_transformer_quick", "demo_diffusion_quick", "demo_lora_r4.pt"
DEMO_ENGINES, DEMO_SENSORS, CYCLE_CAP = [9, 79, 97, 57], ["11", "4", "7"], 121


def capture(step: str, cmd: list[str]) -> dict:
    """Run cmd, keep every output line with seconds since start."""
    print(f"[{step}] {' '.join(str(c) for c in cmd)}", flush=True)
    t0 = time.monotonic()
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}  # otherwise print() output arrives in one burst at exit
    proc = subprocess.Popen([str(c) for c in cmd], cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
    lines = []
    for line in proc.stdout:
        lines.append([round(time.monotonic() - t0, 3), line.rstrip()])
    rc = proc.wait()
    rec = {"step": step, "cmd": [str(c) for c in cmd], "rc": rc, "seconds": round(time.monotonic() - t0, 2),
           "recorded": dt.datetime.now().isoformat(timespec="seconds"), "lines": lines}
    REC.mkdir(parents=True, exist_ok=True)
    (REC / f"{step}.json").write_text(json.dumps(rec, indent=1))
    print(f"[{step}] rc={rc} · {rec['seconds']} s · {len(lines)} lines", flush=True)
    if rc != 0:
        tail = "\n".join(l for _, l in lines[-15:])
        raise SystemExit(f"{step} failed (rc={rc}):\n{tail}")
    return rec


def load(step: str) -> dict:
    return json.loads((REC / f"{step}.json").read_text())


# ───────────────────────────── real runs ─────────────────────────────
def quick_transformer_config() -> Path:
    cfg = yaml.safe_load((ROOT / "src" / "config" / "transformer_config.yaml").read_text())
    cfg.setdefault("model", {})["name"] = T_NAME
    cfg.setdefault("runtime", {})["epochs"] = QUICK["t_epochs"]
    # Generate every engine at its real length: a cycles_per_engine key would be treated as an explicit
    # override for all engines, and n_engines would keep only the first N engine ids.
    gen = cfg.setdefault("generation", {})
    gen["n_engines"] = 100  # all FD001 engines; null would fall back to the value saved inside the checkpoint
    gen.pop("cycles_per_engine", None)
    path = Path(tempfile.mkdtemp()) / "quick_transformer.yaml"
    path.write_text(yaml.dump(cfg, sort_keys=False))
    return path


def train_t():
    capture("train_t", [PY, T_SCRIPT, "--mode", "train", "--config", quick_transformer_config()])


def gen_t():
    ckpt = ROOT / "src" / "model_checkPoints" / "transformer" / f"{T_NAME}.pt"
    capture("gen_t", [PY, T_SCRIPT, "--mode", "generate", "--config", quick_transformer_config(),
                      "--checkpoint", ckpt, "--run_name", "demo_quick"])


def train_d():
    capture("train_d", [PY, D_SCRIPT, "--mode", "train", "--arch", "transformer", "--model-name", D_NAME,
                        "--data", "data/FD001.csv", "--output-dir", "outputs", "--epochs", QUICK["d_epochs"],
                        "--no-auto-sample"])


def diffusion_ckpt() -> Path:
    p = ROOT / "src" / "model_checkPoints" / "diffusion" / D_NAME / "diff_best_ckpt.pt"
    if not p.exists():
        raise SystemExit(f"diffusion checkpoint not found at {p}; run train_d first")
    return p


def finetune():
    out = ROOT / "src" / "model_checkPoints" / "diffusion" / "adapters" / ADAPTER
    out.parent.mkdir(parents=True, exist_ok=True)
    capture("finetune", [PY, D_SCRIPT, "--mode", "fine-tune", "--model-name", D_NAME, "--checkpoint", diffusion_ckpt(),
                         "--seed-data", "data/FD001.csv", "--adapter-out", out, "--lora-rank", QUICK["lora_rank"],
                         "--lora-alpha", QUICK["lora_alpha"], "--epochs", QUICK["ft_epochs"], "--no-auto-sample"])


def unconditional_diffusion_config() -> Path:
    """The repo's diffusion config without `data`.

    With `data` set, generate marks every learnable column as observed and inpaints it from resampled real
    rows, so the output would be real data rather than model samples (diffusion_model5.py, generate mode).
    """
    cfg = yaml.safe_load((ROOT / "src" / "config" / "diffusion_config.yaml").read_text())
    cfg.pop("data", None)
    out = Path(tempfile.mkdtemp()) / "unconditional_diffusion.yaml"
    out.write_text(yaml.safe_dump(cfg))
    return out


def gen_d():
    capture("gen_d", [PY, D_SCRIPT, "--mode", "generate", "--arch", "transformer", "--model-name", D_NAME,
                      "--config", unconditional_diffusion_config(), "--checkpoint", diffusion_ckpt(),
                      "--output-dir", "outputs", "--n-engines", 4, "--cycles-per-engine", 90, "--out", "demo_quick_diffusion.csv"])


def gen_d_lora():
    adapter = ROOT / "src" / "model_checkPoints" / "diffusion" / "adapters" / ADAPTER
    capture("gen_d_lora", [PY, D_SCRIPT, "--mode", "generate", "--arch", "transformer", "--model-name", D_NAME,
                           "--config", unconditional_diffusion_config(), "--checkpoint", diffusion_ckpt(), "--output-dir", "outputs", "--n-engines", 4,
                           "--cycles-per-engine", 90, "--adapters", adapter, "--out", "demo_quick_diffusion_lora.csv"])


D_FULL = "demo_diffusion_full"


def train_d_full():
    """Full preset: the repo's diffusion_config.yaml as is (epochs, hidden_dim, T, lr, batch size, seed)."""
    capture("train_d_full", [PY, D_SCRIPT, "--mode", "train", "--arch", "transformer", "--model-name", D_FULL,
                             "--data", "data/FD001.csv", "--output-dir", "outputs", "--no-auto-sample"])


def gen_d_full():
    ck = ROOT / "src" / "model_checkPoints" / "diffusion" / D_FULL / "diff_best_ckpt.pt"
    capture("gen_d_full", [PY, D_SCRIPT, "--mode", "generate", "--arch", "transformer", "--model-name", D_FULL,
                           "--config", unconditional_diffusion_config(), "--checkpoint", ck, "--output-dir", "outputs",
                           "--n-engines", 2, "--cycles-per-engine", 137, "--out", "demo_full_diffusion.csv"])  # 274 rows, the demo comparison size


ADAPTER_FULL = "demo_full_lora_r4.pt"


def finetune_full():
    """The demo's recorded Fine-tune settings (rank 4, alpha 8, 5 epochs, FD001 seed) applied to the Full model."""
    ck = ROOT / "src" / "model_checkPoints" / "diffusion" / D_FULL / "diff_best_ckpt.pt"
    out = ROOT / "src" / "model_checkPoints" / "diffusion" / "adapters" / ADAPTER_FULL
    capture("finetune_full", [PY, D_SCRIPT, "--mode", "fine-tune", "--model-name", D_FULL, "--checkpoint", ck,
                              "--seed-data", "data/FD001.csv", "--adapter-out", out, "--lora-rank", QUICK["lora_rank"],
                              "--lora-alpha", QUICK["lora_alpha"], "--epochs", QUICK["ft_epochs"], "--no-auto-sample"])


def gen_d_full_lora():
    ck = ROOT / "src" / "model_checkPoints" / "diffusion" / D_FULL / "diff_best_ckpt.pt"
    ad = ROOT / "src" / "model_checkPoints" / "diffusion" / "adapters" / ADAPTER_FULL
    capture("gen_d_full_lora", [PY, D_SCRIPT, "--mode", "generate", "--arch", "transformer", "--model-name", D_FULL,
                                "--config", unconditional_diffusion_config(), "--checkpoint", ck, "--output-dir", "outputs",
                                "--n-engines", 2, "--cycles-per-engine", 137, "--adapters", ad, "--out", "demo_full_diffusion_lora.csv"])


ADAPTER_FULL_FT = "demo_full_lora_r4_fullft.pt"
FT_FULL = {"epochs": 30, "lr": 0.0005, "batch_size": 128}  # the page's Fine-tune "Full run" preset (cfgF)


def finetune_full_ft():
    ck = ROOT / "src" / "model_checkPoints" / "diffusion" / D_FULL / "diff_best_ckpt.pt"
    out = ROOT / "src" / "model_checkPoints" / "diffusion" / "adapters" / ADAPTER_FULL_FT
    capture("finetune_full_ft", [PY, D_SCRIPT, "--mode", "fine-tune", "--model-name", D_FULL, "--checkpoint", ck,
                                 "--seed-data", "data/FD001.csv", "--adapter-out", out, "--lora-rank", QUICK["lora_rank"],
                                 "--lora-alpha", QUICK["lora_alpha"], "--epochs", FT_FULL["epochs"], "--lr", FT_FULL["lr"],
                                 "--batch-size", FT_FULL["batch_size"], "--no-auto-sample"])


def gen_d_full_ft():
    ck = ROOT / "src" / "model_checkPoints" / "diffusion" / D_FULL / "diff_best_ckpt.pt"
    ad = ROOT / "src" / "model_checkPoints" / "diffusion" / "adapters" / ADAPTER_FULL_FT
    capture("gen_d_full_ft", [PY, D_SCRIPT, "--mode", "generate", "--arch", "transformer", "--model-name", D_FULL,
                              "--config", unconditional_diffusion_config(), "--checkpoint", ck, "--output-dir", "outputs",
                              "--n-engines", 2, "--cycles-per-engine", 137, "--adapters", ad, "--out", "demo_full_diffusion_fullft.csv"])


REV_STEPS = [999, 800, 600, 450, 300, 200, 140, 90, 50, 25, 10, 0]


def drev():
    """Reverse-process snapshots of the passing diffusion run (gen_d_full_ft), for the Generate chart.

    Re-runs the script's own main() with the recorded arguments in-process, with sample_inpaint replaced
    by a line-for-line copy that also keeps the iterate at the timesteps in REV_STEPS. The run must
    reproduce the recorded CSV exactly; the snapshots are then mapped to sensor units with the same
    per-column affine transform the script applied to the final sample.
    """
    import importlib.util
    import numpy as np
    import torch
    spec = importlib.util.spec_from_file_location("dm5", D_SCRIPT)
    dm5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(dm5)  # seeds torch exactly as the CLI does
    snaps = {}

    @torch.no_grad()
    def sample_inpaint(model, schedule, n_samples, observed_values=None, observed_mask=None, device=None, verbose=True):
        if device is None:
            device = dm5.DEVICE
        model.to(device); model.eval()
        T = schedule.T; n_cols = model.n_cols
        observed_values = torch.zeros(n_samples, n_cols, device=device) if observed_values is None else observed_values.to(device)
        observed_mask = torch.zeros(n_samples, n_cols, device=device) if observed_mask is None else observed_mask.to(device)
        x = torch.randn(n_samples, n_cols, device=device)
        if observed_mask.any():
            t_init = torch.full((n_samples,), T - 1, device=device, dtype=torch.long)
            x = observed_mask * schedule.q_sample(observed_values, t_init) + (1.0 - observed_mask) * x
        snaps[T - 1] = x.clone()
        for t_idx in range(T - 1, -1, -1):
            t = torch.full((n_samples,), t_idx, device=device, dtype=torch.long)
            eps_hat = model(x, t, observed_mask)
            beta_t = schedule.betas[t_idx]; alpha_t = schedule.alphas[t_idx]
            mean = (1.0 / torch.sqrt(alpha_t)) * (x - beta_t / torch.sqrt(1.0 - schedule.alpha_bar[t_idx]) * eps_hat)
            if t_idx > 0:
                x_new = mean + torch.sqrt(beta_t) * torch.randn_like(x)
                t_prev = torch.full((n_samples,), t_idx - 1, device=device, dtype=torch.long)
                noised_obs = schedule.q_sample(observed_values, t_prev)
            else:
                x_new = mean; noised_obs = observed_values
            x = observed_mask * noised_obs + (1.0 - observed_mask) * x_new
            if t_idx in REV_STEPS:
                snaps[t_idx] = x.clone()
        return x.cpu().numpy()

    dm5.sample_inpaint = sample_inpaint
    cmd = load("gen_d_full_ft")["cmd"]
    tmp = Path(tempfile.mkdtemp())
    args = cmd[2:]
    args[args.index("--output-dir") + 1] = str(tmp)
    args[args.index("--config") + 1] = str(unconditional_diffusion_config())
    sys.argv = ["diffusion_model5.py", *args]
    dm5.main()
    out = next(tmp.rglob("demo_full_diffusion_fullft.csv"))
    rec, new = pd.read_csv(saved_csv("gen_d_full_ft")), pd.read_csv(out)
    assert np.allclose(rec.values, new.values, atol=1e-4), "re-run does not reproduce the recorded samples"
    rec.columns = [c.strip() for c in rec.columns]
    final = snaps[0].cpu().numpy()
    rows = {}
    for c in DEMO_SENSORS:
        y = rec[c].values
        j = int(np.argmax([abs(np.corrcoef(final[:, k], y)[0, 1]) for k in range(final.shape[1])]))
        A = np.vstack([final[:, j], np.ones(len(y))]).T
        (a, b), res, *_ = np.linalg.lstsq(A, y, rcond=None)
        assert np.max(np.abs(A @ [a, b] - y)) < 1e-3, f"column {c}: not an affine map of the sample"
        rows[c] = (j, a, b)
    out_snaps = []
    for t in sorted(snaps, reverse=True):
        x = snaps[t].cpu().numpy()
        out_snaps.append({"t": int(t), **{"s" + c: [round(float(v), 3) for v in x[:, j] * a + b] for c, (j, a, b) in rows.items()}})
    text = DEMO_DATA.read_text()
    cm = json.loads(text[len("window.CM="):].rstrip().rstrip(";"))
    cm["drev"] = {"T": int(max(snaps) + 1), "snapshots": out_snaps}
    DEMO_DATA.write_text("window.CM=" + json.dumps(cm, separators=(",", ":")) + ";\n")
    print(f"[drev] re-run reproduced the recorded {len(rec)} samples exactly; {len(out_snaps)} snapshots at t = {[s['t'] for s in out_snaps]}")


def dsample():
    """The passing diffusion run and its fair real reference, for the demo's Validate step.

    dsynth: the 274 rows of gen_d_full_ft (Full diffusion + Full fine-tune adapter), with the script's own
    group/index labels (diffusion rows are independent samples, not engine timelines).
    ref: 274 rows drawn at random from all of FD001 with random_state=42 (fixed before any result).
    ft: base (no adapter) vs adapted vs the same reference, for the Fine-tune chart.
    """
    syn = pd.read_csv(saved_csv("gen_d_full_ft")); syn.columns = [c.strip() for c in syn.columns]
    base = pd.read_csv(saved_csv("gen_d_full")); base.columns = [c.strip() for c in base.columns]
    real = pd.read_csv(ROOT / "data" / "FD001.csv"); real.columns = [c.strip() for c in real.columns]
    ref = real.sample(n=len(syn), random_state=42)
    rows = lambda d: [[int(x.Engine_ID), int(x.Time_in_cycles)] + [round(float(getattr(x, f"_{i + 3}")), 4) for i in range(len(DEMO_SENSORS))]
                      for x in d[["Engine_ID", "Time_in_cycles"] + DEMO_SENSORS].itertuples()]
    text = DEMO_DATA.read_text()
    cm = json.loads(text[len("window.CM="):].rstrip().rstrip(";"))
    cm["dsynth"], cm["ref"] = rows(syn), rows(ref)
    pack = lambda d: {"s" + c: [round(float(v), 4) for v in d[c]] for c in DEMO_SENSORS}
    cm["meta"]["ft"] = {"base": pack(base), "adapted": pack(syn), "target": pack(ref), "base_ckpt": f"{D_FULL}/diff_best_ckpt.pt",
                        "adapter": ADAPTER_FULL_FT, "rows": len(syn)}
    cm["meta"]["dsynth"] = {"checkpoint": f"{D_FULL}/diff_best_ckpt.pt", "adapter": ADAPTER_FULL_FT, "file": "demo_full_diffusion_fullft.csv",
                            "ref": f"{len(ref)} rows drawn at random from all {real.Engine_ID.nunique()} FD001 engines (random_state=42)"}
    DEMO_DATA.write_text("window.CM=" + json.dumps(cm, separators=(",", ":")) + ";\n")
    from scipy.stats import ks_2samp
    print("[dsample] KS (repo call) passing run vs reference:",
          {c: f"{ks_2samp(ref[c], syn[c]).statistic:.4f} p {ks_2samp(ref[c], syn[c]).pvalue:.4f}" for c in DEMO_SENSORS})


def page_ks(a, b):
    """Line-for-line port of the dashboard's ks() (asymptotic p-value with Stephens' correction), so the
    comparison table shows exactly what the page's own validation would compute."""
    import math
    a, b = sorted(a), sorted(b); i = j = 0; d = 0.0
    while i < len(a) and j < len(b):
        x = min(a[i], b[j])
        while i < len(a) and a[i] <= x: i += 1
        while j < len(b) and b[j] <= x: j += 1
        d = max(d, abs(i / len(a) - j / len(b)))
    en = math.sqrt(len(a) * len(b) / (len(a) + len(b))); lam = (en + 0.12 + 0.11 / en) * d
    p = sum(2 * (1 if k % 2 else -1) * math.exp(-2 * k * k * lam * lam) for k in range(1, 101))
    return d, max(0.0, min(1.0, p))


def compare():
    """Transformer vs diffusion: every model's real output against the same real reference.

    Reference: 274 FD001 rows, random_state=42 (the demo's Validate reference). Synthetic: 274 rows per
    model; where a run wrote more rows, a random_state=42 draw of 274. Same three sensors, same KS.
    """
    real = pd.read_csv(ROOT / "data" / "FD001.csv"); real.columns = [c.strip() for c in real.columns]
    ref = real.sample(n=274, random_state=42)
    def find(get):
        try:
            return get()
        except (SystemExit, FileNotFoundError, IndexError):
            return None
    runs = [("Transformer", "Quick, 8 epochs", find(generated_csv)),
            ("Transformer", "Full, 50 epochs", find(lambda: sorted((ROOT / "outputs" / "Transformer" / "synthetic_data" / "pf_full").glob("*.csv"))[-1])),
            ("Diffusion", "Quick, 6 epochs", find(lambda: saved_csv("gen_d"))), ("Diffusion", "Full, 100 epochs", find(lambda: saved_csv("gen_d_full"))),
            ("Diffusion", "Full + LoRA, 30 epochs", find(lambda: saved_csv("gen_d_full_ft")))]
    rows = []
    for model, train, path in runs:
        if path is None:
            print(f"[compare] skipped {model} ({train}): its generated data is not on disk; run that model first")
            continue
        s = pd.read_csv(path); s.columns = [c.strip() for c in s.columns]
        if len(s) > 274:
            s = s.sample(n=274, random_state=42)
        ks = {}
        for c in DEMO_SENSORS:
            d, p = page_ks(ref[c].tolist(), s[c].tolist())
            ks["s" + c] = [round(d, 4), round(p, 4)]
        npass = sum(v[1] >= 0.05 for v in ks.values())
        rows.append({"model": model, "train": train, "file": Path(path).relative_to(ROOT).as_posix(), "rows": len(s), "ks": ks,
                     "pass": npass, "sim": round(sum(1 - v[0] for v in ks.values()) / len(ks) * 100, 1),
                     "spread": {"s" + c: round(float(s[c].std() / ref[c].std() * 100)) for c in DEMO_SENSORS}})
        print(f"[compare] {model:11s} {train:24s} pass {npass}/3  sim {rows[-1]['sim']:5.1f}%  " + "  ".join(f"{k} p={v[1]:.4f}" for k, v in ks.items()))
    text = DEMO_DATA.read_text()
    cm = json.loads(text[len("window.CM="):].rstrip().rstrip(";"))
    cm["compare"] = {"ref": "274 FD001 rows drawn at random from all 100 engines (random_state=42)", "rows": rows}
    DEMO_DATA.write_text("window.CM=" + json.dumps(cm, separators=(",", ":")) + ";\n")


def saved_csv(step: str) -> Path:
    for _, line in reversed(load(step)["lines"]):
        m = re.search(r"Saved \d+ (?:rows|samples) to:? (\S+\.csv)", line)
        if m:
            p = Path(m.group(1))
            return p if p.is_absolute() else ROOT / p
    raise SystemExit(f"no CSV path in the {step} capture")


def ftsample():
    """Real base vs adapter-composed diffusion samples, plus the seed target, for the Fine-tune chart."""
    base = pd.read_csv(saved_csv("gen_d")); lora = pd.read_csv(saved_csv("gen_d_lora"))
    tgt = pd.read_csv(ROOT / "data" / "FD001.csv")
    for d in (base, lora, tgt):
        d.columns = [c.strip() for c in d.columns]
    n = min(len(base), len(lora))
    tgt = tgt.sample(n=n, random_state=42)
    pack = lambda d: {"s" + c: [round(float(v), 4) for v in d[c].head(n)] for c in DEMO_SENSORS}
    text = DEMO_DATA.read_text()
    cm = json.loads(text[len("window.CM="):].rstrip().rstrip(";"))
    cm["meta"]["ft"] = {"base": pack(base), "adapted": pack(lora), "target": pack(tgt),
                        "base_ckpt": f"{D_NAME}/diff_best_ckpt.pt", "adapter": ADAPTER, "rows": n}
    DEMO_DATA.write_text("window.CM=" + json.dumps(cm, separators=(",", ":")) + ";\n")
    print(f"[ftsample] {n} rows each: base, adapted ({ADAPTER}), seed target")


# ───────────────────────────── events ─────────────────────────────
def to_events(step: str) -> dict:
    """Replay a capture through the same parser live jobs use, then add the artifact the run wrote."""
    rec = load(step)
    total = rec["seconds"]
    epochs = {"train_t": QUICK["t_epochs"], "train_d": QUICK["d_epochs"], "finetune": QUICK["ft_epochs"],
              "finetune_full_ft": FT_FULL["epochs"]}.get(step)  # train_d_full: the parser reads epochs from effective_config
    kind = {"train_d_full": "train_d", "finetune_full_ft": "finetune", "gen_d_full_ft": "gen_d"}.get(step, step)
    p = LogParser(kind, epochs=epochs, engines=DEMO_ENGINES if step == "gen_t" else None)
    pairs = p.start()
    for t, line in rec["lines"]:
        pairs += p.feed(t, line.replace(f"{ROOT}/", ""))  # show repo-relative paths, not the recording machine's
    pairs += p.finish(total)
    if step == "train_t":
        ck = ROOT / "src" / "model_checkPoints" / "transformer" / f"{T_NAME}.pt"
        pairs.append((total, {"type": "artifact", "kind": "checkpoint", "path": str(ck.relative_to(ROOT)),
                              "meta": {"type": "transformer", "arch": "transformer", "epochs": QUICK["t_epochs"], "best": p.best,
                                       "params": p.params, "size": f"{ck.stat().st_size / 1e6:.1f} MB", "src": "recorded real run"}}))
    elif step == "train_d":
        ck, eff = diffusion_ckpt(), p.effective
        pairs.append((total, {"type": "artifact", "kind": "checkpoint", "path": str(ck.relative_to(ROOT)),
                              "meta": {"type": "diffusion", "arch": "transformer", "epochs": QUICK["d_epochs"], "best": p.best, "params": p.params,
                                       "size": f"{ck.stat().st_size / 1e6:.1f} MB", "T": int(float(eff.get("T", 0))) or None,
                                       "lr": float(eff["lr"]) if "lr" in eff else None, "bs": int(float(eff["batch_size"])) if "batch_size" in eff else None,
                                       "hid": int(float(eff["hidden_dim"])) if "hidden_dim" in eff else None, "src": "recorded real run"}}))
    elif step == "finetune":
        pairs.append((total, {"type": "artifact", "kind": "adapter", "path": f"src/model_checkPoints/diffusion/adapters/{ADAPTER}",
                              "meta": {"rank": QUICK["lora_rank"], "alpha": QUICK["lora_alpha"], "epochs": QUICK["ft_epochs"],
                                       "loss": p.last, "trainable": p.trainable, "src": "recorded real run"}}))
    elif step == "train_d_full":
        ck, eff = ROOT / "src" / "model_checkPoints" / "diffusion" / D_FULL / "diff_best_ckpt.pt", p.effective
        pairs.append((total, {"type": "artifact", "kind": "checkpoint", "path": str(ck.relative_to(ROOT)),
                              "meta": {"type": "diffusion", "arch": "transformer", "epochs": p.n_metrics, "best": p.best, "params": p.params,
                                       "size": f"{ck.stat().st_size / 1e6:.1f} MB", "T": int(float(eff.get("T", 0))) or None,
                                       "lr": float(eff["lr"]) if "lr" in eff else None, "bs": int(float(eff["batch_size"])) if "batch_size" in eff else None,
                                       "hid": int(float(eff["hidden_dim"])) if "hidden_dim" in eff else None, "src": "recorded real run"}}))
    elif step == "finetune_full_ft":
        pairs.append((total, {"type": "artifact", "kind": "adapter", "path": f"src/model_checkPoints/diffusion/adapters/{ADAPTER_FULL_FT}",
                              "meta": {"rank": QUICK["lora_rank"], "alpha": QUICK["lora_alpha"], "epochs": FT_FULL["epochs"], "lr": FT_FULL["lr"],
                                       "bs": FT_FULL["batch_size"], "loss": p.last, "trainable": p.trainable, "src": "recorded real run"}}))
    pairs.append((total, {"type": "done"}))
    ev = sorted(([int(t * 1000), e] for t, e in pairs), key=lambda x: x[0])
    return {"step": step, "recorded": rec["recorded"], "seconds": total, "events": ev}


# ───────────────────────────── demo sample ─────────────────────────────
def generated_csv() -> Path:
    for _, line in reversed(load("gen_t")["lines"]):
        m = re.search(r"Saved \d+ rows .* to: (\S+\.csv)", line)
        if m:
            p = Path(m.group(1))
            return p if p.is_absolute() else ROOT / p
    raise SystemExit("could not find the CSV path in the gen_t capture")


def sample():
    """Real and synthetic rows for the same engines over the same cycle range.

    The Transformer seeds each engine with its first INPUT_WINDOW real rows and
    numbers the generated rows from 1, so generated row t is real cycle
    t + INPUT_WINDOW. Seed rows are never copied into the synthetic side.
    """
    real = pd.read_csv(ROOT / "data" / "FD001.csv"); real.columns = [c.strip() for c in real.columns]
    syn = pd.read_csv(generated_csv()); syn.columns = [c.strip() for c in syn.columns]
    win = yaml.safe_load((ROOT / "src" / "config" / "transformer_config.yaml").read_text())["runtime"]["input_window"]
    syn = syn.assign(Time_in_cycles=syn["Time_in_cycles"] + win)
    pick = lambda df: df[df["Engine_ID"].isin(DEMO_ENGINES) & df["Time_in_cycles"].between(win + 1, CYCLE_CAP)]
    r, s = pick(real), pick(syn)
    key = ["Engine_ID", "Time_in_cycles"]
    common = r[key].merge(s[key], on=key)
    r, s = (d.merge(common, on=key).sort_values(key) for d in (r, s))
    order = {e: i for i, e in enumerate(DEMO_ENGINES)}
    rows = lambda d: sorted(([int(x.Engine_ID), int(x.Time_in_cycles)] + [round(float(getattr(x, f"_{i + 3}")), 4) for i in range(len(DEMO_SENSORS))]
                             for x in d[key + DEMO_SENSORS].itertuples()), key=lambda q: (order[q[0]], q[1]))
    cm = {"real": rows(r), "synth": rows(s),
          "meta": {"engines": DEMO_ENGINES, "sensors": ["s" + c for c in DEMO_SENSORS], "cycles": [win + 1, CYCLE_CAP],
                   "synthetic_source": generated_csv().relative_to(ROOT).as_posix(), "checkpoint": f"{T_NAME}.pt",
                   "note": f"Synthetic rows are real {T_NAME} output; generated row t is shifted to cycle t+{win} (the script numbers from 1 after the {win}-row seed window). Seed rows are not included."}}
    DEMO_DATA.write_text("window.CM=" + json.dumps(cm, separators=(",", ":")) + ";\n")
    print(f"[sample] {len(cm['real'])} real rows / {len(cm['synth'])} synthetic rows · engines {DEMO_ENGINES} · cycles {win + 1}–{CYCLE_CAP}")


def predict():
    """Real one-step predictions and attention from the demo checkpoint, for the Train sliding-window chart.

    For each demo engine, every window of INPUT_WINDOW real frames (the checkpoint's own normalized
    full_sequences) is fed to the model, which predicts the next cycle. The attention row recorded is
    what the last time step attends to in the last encoder layer, averaged over heads; the model's
    output head reads only that last position. Also adds each engine's real lifespan to meta.life,
    because the bundled sample stops at cycle CYCLE_CAP.
    """
    import numpy as np
    import torch
    sys.path.insert(0, str(T_SCRIPT.parent))
    from Transformer_TrainingV7 import StrongTimeTransformer

    ck_path = ROOT / "src" / "model_checkPoints" / "transformer" / f"{T_NAME}.pt"
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    rt = ck.get("config", {}).get("runtime", {})
    feats, mins, ranges = ck["feature_cols"], ck["col_mins"], ck["ranges"]
    win, out_dim = int(ck["INPUT_WINDOW"]), int(ck["out_dim"])
    model = StrongTimeTransformer(src_dim=int(ck["src_dim"]), out_dim=out_dim, n_engines=len(ck["eng_ids"]),
                                  d_model=int(rt.get("d_model", 256)), nhead=int(rt.get("nhead", 8)),
                                  num_layers=int(rt.get("num_layers", 4)), d_ff=int(rt.get("d_ff", 512)),
                                  d_eng=int(rt.get("d_eng", 4)), dropout=float(rt.get("dropout", 0.15)))
    model.load_state_dict(ck["model_state_dict"], strict=True)
    model.eval()

    attn = {}
    last = model.encoder.layers[-1].self_attn
    plain_forward = last.forward
    def forward_with_weights(*a, **k):  # the encoder layer asks for need_weights=False; ask for the head-averaged weights
        k["need_weights"], k["average_attn_weights"] = True, True
        out, w = plain_forward(*a, **k)
        attn["w"] = w
        return out, w
    last.forward = forward_with_weights

    denorm = lambda v, c: (v + 1) / 2 * float(ranges[c]) + float(mins[c])
    col = {c: feats.index(c) for c in DEMO_SENSORS}
    real = pd.read_csv(ROOT / "data" / "FD001.csv"); real.columns = [c.strip() for c in real.columns]
    engines, life = {}, {}
    with torch.no_grad():
        for e in DEMO_ENGINES:
            key = f"FD001.csv::eng_{e}"
            frames = torch.tensor(np.asarray(ck["full_sequences"][key]), dtype=torch.float32)
            n = len(frames); life[str(e)] = n
            idx = list(range(win, min(n, CYCLE_CAP)))  # frame t is cycle t+1
            src = torch.stack([torch.cat([frames[t - win:t], torch.ones(win, out_dim)], dim=-1) for t in idx])
            eng = torch.full((len(idx),), ck["eng_to_idx"][key], dtype=torch.long)
            pred = model(src, eng).numpy()
            w = attn["w"][:, -1, :].numpy()  # [windows, win]: last query over the window's keys
            engines[str(e)] = {"c": [t + 1 for t in idx],
                               **{"s" + c: [round(float(denorm(pred[i, col[c]], c)), 4) for i in range(len(idx))] for c in DEMO_SENSORS},
                               "att": [[round(float(x), 4) for x in row] for row in w]}
            r = real[real.Engine_ID == e].set_index("Time_in_cycles")
            err = {c: round(float(np.mean([abs(engines[str(e)]["s" + c][i] - r.loc[cyc, c]) for i, cyc in enumerate(engines[str(e)]["c"])])), 4) for c in DEMO_SENSORS}
            print(f"[predict] engine {e}: {len(idx)} windows, cycles {idx[0] + 1}-{idx[-1] + 1}, mean |actual - predicted| {err}")

    # the first one-step prediction starts from the same real seed window as generation, so it must equal generated row 1
    gen = pd.read_csv(generated_csv()); gen.columns = [c.strip() for c in gen.columns]
    for e in DEMO_ENGINES:
        g1 = gen[(gen.Engine_ID == e) & (gen.Time_in_cycles == 1)]
        diff = max(abs(float(g1[c].iloc[0]) - engines[str(e)]["s" + c][0]) for c in DEMO_SENSORS)
        assert diff < 1e-2, f"engine {e}: first prediction differs from generated row 1 by {diff}"
    print("[predict] first predictions match the checkpoint's own generate output (row 1 of each engine)")

    text = DEMO_DATA.read_text()
    cm = json.loads(text[len("window.CM="):].rstrip().rstrip(";"))
    cm["meta"]["life"] = life
    cm["pred"] = {"checkpoint": f"{T_NAME}.pt", "window": win, "attention": "last encoder layer, mean of heads, last time step",
                  "heads": int(rt.get("nhead", 8)), "layers": int(rt.get("num_layers", 4)), "engines": engines}
    DEMO_DATA.write_text("window.CM=" + json.dumps(cm, separators=(",", ":")) + ";\n")
    print(f"[predict] wrote pred for engines {DEMO_ENGINES}; real lifespans {life}")


def ckpt_meta():
    """Metadata for the dashboard's saved-checkpoint table, read from the real .pt files."""
    sys.path.insert(0, str(ROOT))
    from src.training.Diffusion.diffusion_model5 import TabularTransformerDenoiser, apply_lora_to_model
    files = checkpoint_files()
    grid, lora = {}, {}
    for H in (64, 128, 256):
        for L in range(1, 9):
            grid[f"{H}_{L}"] = sum(q.numel() for q in TabularTransformerDenoiser(n_cols=26, hidden_dim=H, n_heads=4, n_layers=L).parameters())
            for r in range(1, 17):
                lora[f"{H}_{L}_{r}"] = sum(q.numel() for q in apply_lora_to_model(TabularTransformerDenoiser(n_cols=26, hidden_dim=H, n_heads=4, n_layers=L), rank=r))
    meta = {"read_on": dt.date.today().isoformat(), "total_on_disk": len(list(ROOT.glob("src/model_checkPoints/**/*.pt"))),
            "files": files, "diffusion_transformer_params": grid, "lora_params": lora}
    (ROOT / "dashboard" / "ckpt_meta.json").write_text(json.dumps(meta, separators=(",", ":")))
    print(f"[ckpt_meta] {len(files)} files · e.g. " + ", ".join(f"{Path(f['path']).name}={f['params']:,}" for f in files[:3]))


def replays():
    out = {}
    for step in ("train_t", "gen_t", "train_d", "finetune", "gen_d", "train_d_full", "finetune_full_ft", "gen_d_full_ft"):
        if (REC / f"{step}.json").exists():
            out[step] = to_events(step)
    REPLAYS.write_text(json.dumps(out, separators=(",", ":")))
    print("[replays]", {k: f"{v['seconds']} s · {len(v['events'])} events" for k, v in out.items()})


STEPS = {"train_t": train_t, "gen_t": gen_t, "train_d": train_d, "finetune": finetune, "gen_d": gen_d,
         "gen_d_lora": gen_d_lora, "sample": sample, "ftsample": ftsample, "predict": predict, "train_d_full": train_d_full, "gen_d_full": gen_d_full, "finetune_full": finetune_full, "gen_d_full_lora": gen_d_full_lora, "finetune_full_ft": finetune_full_ft, "gen_d_full_ft": gen_d_full_ft, "dsample": dsample, "drev": drev, "compare": compare, "ckpt_meta": ckpt_meta, "replays": replays}

if __name__ == "__main__":
    todo = sys.argv[1:] or ["all"]
    if todo == ["all"]:
        todo = ["train_t", "gen_t", "train_d", "finetune", "gen_d", "gen_d_lora", "sample", "ftsample", "predict", "ckpt_meta", "replays"]
    for s in todo:
        STEPS[s]()
