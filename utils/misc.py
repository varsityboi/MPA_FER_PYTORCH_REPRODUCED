"""
Small helpers used by train.py, eval.py and the tools/ scripts.

    config       load YAML + command line overrides (--set train.epochs=3)
    seed         make runs repeatable
    logging      print + save to log.txt, per-epoch numbers to metrics.jsonl
    meters       running averages of losses
    params       count trainable parameters
    checkpoints  save / resume only the trainable prompts (safe for Google Drive)
"""
import json
import os
import random
import time

import numpy as np
import torch
import yaml


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
def load_config(path, overrides=None):
    """Read the YAML file, then apply any --set overrides."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if overrides:
        apply_overrides(cfg, overrides)
    return cfg


def _find_key(node, key, full_key):
    """Find a key in a dict. YAML numbers like 1: become int keys, so try int too."""
    if isinstance(node, dict):
        if key in node:
            return key
        if key.isdigit() and int(key) in node:
            return int(key)
    raise KeyError(f"Override key '{full_key}' not found in config (problem at '{key}')")


def apply_overrides(cfg, overrides):
    """
    overrides: list like ["train.epochs=3", "model.visual_prompts.enabled=false"]
    Values are read as YAML: 3 -> int, false -> bool, null -> None, [0.8,1.0] -> list.
    Unknown keys raise an error, so a typo can't silently do nothing.
    """
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override '{item}' must look like key.subkey=value")
        full_key, value = item.split("=", 1)
        parts = full_key.strip().split(".")
        node = cfg
        for part in parts[:-1]:
            node = node[_find_key(node, part, full_key)]
        last = _find_key(node, parts[-1], full_key)
        node[last] = yaml.safe_load(value)
    return cfg


def save_config(cfg, out_dir):
    """Save the exact settings used for this run, next to the checkpoints."""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config_used.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


# ------------------------------------------------------------------
# Seed and device
# ------------------------------------------------------------------
def set_seed(seed, deterministic=False):
    """
    Fix all random number generators.
    deterministic=False keeps cuDNN's fast mode, so runs match closely but not bit-for-bit.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def seed_worker(worker_id):
    """Give each DataLoader worker its own fixed seed (pass as worker_init_fn)."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
class Logger:
    """Prints messages and appends them to files, so logs survive a Colab disconnect."""

    def __init__(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        self.log_path = os.path.join(out_dir, "log.txt")
        self.metrics_path = os.path.join(out_dir, "metrics.jsonl")

    def log(self, msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def log_metrics(self, metrics):
        """One JSON line per epoch, e.g. {"epoch": 1, "loss_vt": 1.2, "test_acc": 71.3}."""
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")


def format_time(seconds):
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h {m:02d}m {s:02d}s" if h else f"{m:d}m {s:02d}s"


# ------------------------------------------------------------------
# Running averages
# ------------------------------------------------------------------
class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value, n=1):
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(self.count, 1)


class MeterDict:
    """Keeps an AverageMeter for every key in the loss logs dict."""

    def __init__(self):
        self.meters = {}

    def update(self, logs, n=1):
        for key, value in logs.items():
            self.meters.setdefault(key, AverageMeter()).update(value, n)

    def averages(self):
        return {key: meter.avg for key, meter in self.meters.items()}

    def __str__(self):
        return "  ".join(f"{key} {meter.avg:.4f}" for key, meter in self.meters.items())


# ------------------------------------------------------------------
# Parameter count
# ------------------------------------------------------------------
def param_report(model):
    """Total vs trainable parameters, with every trainable tensor listed."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lines = [
        f"Total params:     {total:,}",
        f"Trainable params: {trainable:,} ({100 * trainable / max(total, 1):.4f}%)",
    ]
    for name, p in model.named_parameters():
        if p.requires_grad:
            lines.append(f"  {name}: {tuple(p.shape)} = {p.numel():,}")
    return "\n".join(lines)


# ------------------------------------------------------------------
# Checkpoints
# ------------------------------------------------------------------
def save_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None,
                    epoch=0, best_acc=0.0, cfg=None):
    """
    Saves only the prompts (not all of CLIP) + training state.
    Writes to a .tmp file first, then renames. If Colab dies mid-save,
    the previous checkpoint is still safe.
    """
    state = {
        "epoch": epoch,
        "best_acc": best_acc,
        "model": model.trainable_state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "config": cfg,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = path + ".tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None):
    """Loads prompts (and training state if given). Returns the saved dict (epoch, best_acc, ...)."""
    # weights_only=False is fine: it's our own file, and it also stores config + optimizer state
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_trainable_state_dict(state["model"])
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])
    return state


def find_resume_checkpoint(out_dir):
    """Path to last.pt if a previous run left one, else None."""
    path = os.path.join(out_dir, "last.pt")
    return path if os.path.isfile(path) else None