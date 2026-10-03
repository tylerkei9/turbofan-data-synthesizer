"""Metadata read from the saved .pt files, for the dashboard's checkpoint table and model cards."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CKPT_ROOT = ROOT / "src" / "model_checkPoints"
BUFFERS = (".pe", "column_stats")  # registered buffers stored in state_dicts; not parameters
_cache: dict[tuple, dict] = {}


def checkpoint_paths() -> list[Path]:
    paths = sorted(CKPT_ROOT.glob("transformer/*.pt")) + sorted(CKPT_ROOT.glob("diffusion/*/diff_best_ckpt.pt"))
    paths += [p for p in (CKPT_ROOT / "demo_model/diff_best_ckpt.pt", CKPT_ROOT / "diffusion2/demo_model_best_ckpt.pt") if p.exists()]
    return paths


def adapter_paths() -> list[Path]:
    return sorted(CKPT_ROOT.glob("adapters/*.pt")) + sorted(CKPT_ROOT.glob("diffusion/adapters/*.pt"))


def describe(p: Path) -> dict:
    """Type, architecture, parameter count (buffers excluded), size and stored loss of one checkpoint."""
    key = (str(p), p.stat().st_mtime, p.stat().st_size)
    if key in _cache:
        return _cache[key]
    import torch
    ck = torch.load(p, map_location="cpu", weights_only=False)
    sd = next((ck[k] for k in ("model_state", "model_state_dict", "model") if k in ck and isinstance(ck[k], dict)), {})
    n = sum(v.numel() for k, v in sd.items() if hasattr(v, "numel") and not k.endswith(BUFFERS))
    e = {"path": p.relative_to(ROOT).as_posix(), "params": n, "size_mb": round(p.stat().st_size / 1e6, 1),
         "modified": dt.datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d"), "best": ck.get("val_loss"),
         "best_epoch": ck.get("epoch") if ck.get("val_loss") is not None else None}
    if "transformer" in p.relative_to(CKPT_ROOT).parts[:1]:
        cfg = ck.get("config", {}).get("runtime", {})
        e.update(type="transformer", arch="transformer", input_window=ck.get("INPUT_WINDOW", cfg.get("input_window")),
                 d_model=int(cfg.get("d_model", 256)), layers=int(cfg.get("num_layers", 4)))
    else:
        sch = ck.get("schedule") or ck.get("sched") or {}
        T = (sch.get("T") if isinstance(sch, dict) else None) or (len(sch["betas"]) if isinstance(sch, dict) and "betas" in sch else None)
        e.update(type="diffusion", arch="transformer" if any("encoder.layers" in k for k in sd) else "mlp",
                 T=int(T) if T is not None else None, epoch=ck.get("epoch"), legacy="model" in ck and "model_state" not in ck)
    _cache[key] = e
    return e


def checkpoint_files() -> list[dict]:
    return [describe(p) for p in checkpoint_paths()]
