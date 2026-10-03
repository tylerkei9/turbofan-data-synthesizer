"""Turn the training scripts' stdout into the dashboard's typed event stream.

Shared by live backend jobs (api/pipeline.py, fed line by line as the script runs)
and the replay recorder (api/record_replays.py, fed from a timestamped capture),
so a live run and a replay of a recorded run produce the same events.

Event types: phase, progress, metric, best, log, rows (plus artifact / done /
error, which the caller adds because only it knows where outputs were written).
Every number in an event is parsed from a line the script printed.
"""
from __future__ import annotations

import re

F = r"([-\d.eE+]+)"
TQDM = re.compile(r"\|[█▏▎▍▌▋▊▉ ]*\|.*(?:it/s|batch/s|s/it)")

ORDERS = {
    "train_t": ["load_data", "build_windows", "split", "build_model", "epochs", "save_best"],
    "train_d": ["load_data", "normalize", "build_model", "epochs", "save_best"],
    "finetune": ["load_base", "load_seeds", "inject_lora", "epochs", "save_adapter"],
    "gen_t": ["load_checkpoint", "generate", "save_csv"],
    "gen_d": ["load_checkpoint", "sample", "save_csv"],
}


class LogParser:
    """Incremental parser for one run. Times are seconds since the run started."""

    def __init__(self, kind: str, epochs: int | None = None, engines: list | None = None):
        self.kind, self.order, self.epochs = kind, ORDERS[kind], epochs
        self.engines = engines            # gen_t: only emit rows events for these engine ids (None = all)
        self.seen: set[str] = set()
        self.params = self.trainable = self.best = self.last = None
        self.effective: dict[str, str] = {}
        self.n_metrics = self.n_rows = 0
        self.n_engines_total = None

    def _phase(self, t, name, out):
        if name not in self.seen:
            self.seen.add(name)
            out.append((t, {"type": "phase", "name": name, "index": self.order.index(name), "of": len(self.order)}))

    def start(self) -> list:
        out: list = []
        self._phase(0.0, self.order[0], out)
        return out

    def feed(self, t: float, line: str) -> list:
        """Events for one output line; progress-bar redraws are dropped."""
        if TQDM.search(line):
            return []
        out: list = []
        k = self.kind
        if k == "train_t":
            if re.search(r"Using \d+ features|Sequences with sufficient length", line): self._phase(t, "build_windows", out)
            if "Train sequences:" in line: self._phase(t, "split", out)
            if "Model Architecture" in line: self._phase(t, "build_model", out)
            if "Starting training" in line: self._phase(t, "epochs", out)
            m = re.search(r"Total parameters: ([\d,]+)", line)
            if m: self.params = int(m.group(1).replace(",", ""))
            m = re.search(rf"Epoch (\d+)/(\d+)\s+train base {F} var {F} total {F} \| val base {F} var {F} total {F} \| lr {F} \| time {F}s", line)
            if m:
                self._phase(t, "epochs", out)
                e, n = int(m.group(1)), int(m.group(2))
                met = {"type": "metric", "epoch": e, "train_base": float(m.group(3)), "train_var": float(m.group(4)),
                       "train_total": float(m.group(5)), "val_base": float(m.group(6)), "val_var": float(m.group(7)),
                       "val_total": float(m.group(8)), "lr": float(m.group(9)), "sec": float(m.group(10))}
                out += [(t, met), (t, {"type": "progress", "done": e, "total": n, "unit": "epoch"})]
                self.n_metrics += 1
                if self.best is None or met["val_total"] < self.best:
                    self.best = met["val_total"]
                    out.append((t, {"type": "best", "epoch": e, "value": self.best}))
        elif k == "train_d":
            if "effective_config" in line:  # the settings actually used (config file merged with CLI flags)
                self.effective = dict(re.findall(r"'(T|lr|batch_size|hidden_dim|beta_start|epochs)': ([\d.eE+-]+)", line))
                if self.epochs is None and "epochs" in self.effective:
                    self.epochs = int(float(self.effective["epochs"]))
            if re.search(r"[Nn]ormali", line): self._phase(t, "normalize", out)
            m = re.search(r"Model parameters: ([\d,]+)", line)
            if m:
                self.params = int(m.group(1).replace(",", ""))
                self._phase(t, "normalize", out); self._phase(t, "build_model", out)
            m = re.search(rf"\[Epoch (\d+)\] avg loss: {F}", line)
            if m:
                self._phase(t, "epochs", out)
                e, self.last = int(m.group(1)), float(m.group(2))
                self.n_metrics += 1
                out += [(t, {"type": "metric", "epoch": e, "loss": self.last}),
                        (t, {"type": "progress", "done": e, "total": self.epochs or e, "unit": "epoch"})]
            m = re.search(rf"\[Best(?: Model)?\][^\d-]*{F}", line)
            if m:
                self.best = float(m.group(1))
                out.append((t, {"type": "best", "epoch": self.n_metrics, "value": self.best}))
        elif k == "finetune":
            if re.search(r"[Ll]oaded \d+ samples|seed", line): self._phase(t, "load_seeds", out)
            m = re.search(r"LoRA: ([\d,]+) trainable params", line)
            if m:
                self.trainable = int(m.group(1).replace(",", ""))
                self._phase(t, "load_seeds", out); self._phase(t, "inject_lora", out)
            m = re.search(rf"\[LoRA epoch (\d+)\] avg loss: {F}", line)
            if m:
                self._phase(t, "epochs", out)
                e, self.last = int(m.group(1)), float(m.group(2))
                met = {"type": "metric", "epoch": e, "loss": self.last}
                if e == 1 and self.trainable:
                    met["trainable_params"] = self.trainable
                self.n_metrics += 1
                out += [(t, met), (t, {"type": "progress", "done": e, "total": self.epochs or e, "unit": "epoch"})]
        elif k in ("gen_t", "gen_d"):
            m = re.search(r"Engines: (\d+)", line)
            if m: self.n_engines_total = int(m.group(1))
            m = re.search(r"Generated engine (\S+) \((\d+) steps, rows so far: (\d+)\)", line)
            if m:
                self._phase(t, self.order[1], out)
                raw = m.group(1)
                eid = int(float(raw)) if re.fullmatch(r"[\d.]+", raw) else raw
                if self.engines is None or eid in self.engines:
                    self.n_rows += 1
                    out.append((t, {"type": "rows", "engine": self.n_rows, "engine_id": eid, "steps": int(m.group(2)),
                                    "rows_so_far": int(m.group(3))}))
                    if self.engines is None and self.n_engines_total:
                        out.append((t, {"type": "progress", "done": self.n_rows, "total": self.n_engines_total, "unit": "engine"}))
            if re.search(r"Generating \d+ samples|Sampling all|Inpainting from", line): self._phase(t, self.order[1], out)
            if re.search(r"Saved \d+ (?:rows|samples)", line): self._phase(t, "save_csv", out)
        if line.strip():
            out.append((t, {"type": "log", "line": line}))
        return out

    def finish(self, t: float) -> list:
        out: list = []
        self._phase(t, self.order[-1], out)
        return out
