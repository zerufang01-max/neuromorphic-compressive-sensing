"""Paths, JSON helpers and checkpoint loading shared by all scripts."""

import json
from pathlib import Path

import torch

from config import Protocol
from data import measurement_matrix
from models import build_model

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT_DIR / "outputs"
PAPER_CONFIG = PROJECT_DIR / "configs" / "paper.json"


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def checkpoint_path(output, seed, condition_id, model_id):
    """Layout written by train.py."""
    return Path(output) / f"seed{seed}" / "checkpoints" / f"{condition_id}__{model_id}.pth"


def load_model(path, device):
    """Rebuild a trained model and its sensing matrix from a checkpoint."""
    payload = torch.load(path, map_location=device, weights_only=False)
    # Ignore unknown keys.
    known = Protocol.__dataclass_fields__
    job = payload["job"]
    cfg = Protocol(**{k: v for k, v in payload["protocol"].items() if k in known})
    condition = job["condition"]
    matrix = measurement_matrix(cfg.n, condition["m"], cfg.max_m, cfg.matrix_seed, device)
    model = build_model(job["method"], matrix, job["depth"], job["parameters"]).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return model, matrix, job, cfg
