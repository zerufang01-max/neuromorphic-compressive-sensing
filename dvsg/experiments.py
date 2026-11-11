"""Experiment registry, run specs and baseline weight resolution.

Two kinds of entries:
- source='train': trained by trainer.py (all S-LISTA runs, and the Vanilla
  bit-budget runs, which use the Vanilla settings unchanged and only change C);
- source='baseline': baselines trained by baselines/train.py, whose weights are
  loaded and evaluated (ANN LISTA, ANN ConvLSTM, Vanilla SNN dense, Vanilla SNN
  at its AER point).

A run spec is a plain JSON dict; a worker process executes exactly one spec.
"""

import copy
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import config as C


@dataclass(frozen=True)
class Experiment:
    key: str
    method: str                 # slista | vanilla | ann_lista | ann_lstm
    display: str
    transport: str = "dense"    # dense | aer   (ANN: 8-bit dense)
    channels: int = 4           # measurement channels C (32 x 32 grid each)
    semantic_weight: float = 0.01
    rate_weight: float = 0.0
    tau_mode: str = "anneal"    # anneal | fixed (transmitter temperature)
    source: str = "train"       # train | baseline
    baseline_key: str = ""       # key in baselines/train.py (source='baseline')
    classifier: bool = True     # train / load a downstream classifier

    @property
    def mode(self):
        return "ann" if self.method.startswith("ann") else "spiking"


def _registry():
    t = C.SLISTA_TRAINING
    sem = t["semantic_weight"]
    entries = [
        # ---------------------------------------------------------- S-LISTA (trained)
        Experiment("slista_dense", "slista", "S-LISTA", semantic_weight=sem),
        Experiment("slista_dense_no_semantic", "slista", "S-LISTA", semantic_weight=0.0),
        Experiment("slista_aer", "slista", "S-LISTA", transport="aer", rate_weight=t["rate_weight"],
                   semantic_weight=sem),
        Experiment("slista_aer_no_rate", "slista", "S-LISTA", transport="aer", rate_weight=0.0,
                   semantic_weight=sem),
        Experiment("slista_aer_no_semantic", "slista", "S-LISTA", transport="aer",
                   rate_weight=t["rate_weight"], semantic_weight=0.0),
        Experiment("slista_aer_fixed_tau", "slista", "S-LISTA", transport="aer",
                   rate_weight=t["rate_weight"], tau_mode="fixed", semantic_weight=sem),
        # ------------------------------------------------------- baselines (baselines/train.py)
        Experiment("vanilla_dense", "vanilla", "Vanilla SNN", source="baseline", baseline_key="scnn_dense"),
        Experiment("vanilla_aer", "vanilla", "Vanilla SNN", transport="aer", rate_weight=0.002,
                   source="baseline", baseline_key="scnn_rate_0p002"),
        Experiment("ann_lista", "ann_lista", "ANN LISTA", source="baseline", baseline_key="ann_lista_q8"),
        Experiment("ann_lstm", "ann_lstm", "ANN LSTM", source="baseline", baseline_key="ann_lstm_q8"),
    ]
    # ----------------------------------------------------------------- bit budget
    for c in C.BUDGET_CHANNELS:
        if c != 4:
            entries.append(Experiment(f"slista_dense_c{c}", "slista", "S-LISTA", channels=c,
                                      semantic_weight=sem))
        # All Vanilla budget points (including C = 4) are trained by trainer.py with the
        # Vanilla settings of the baseline, so the Vanilla curve comes from one training code.
        entries.append(Experiment(f"vanilla_dense_c{c}", "vanilla", "Vanilla SNN", channels=c,
                                  semantic_weight=0.01))       # Vanilla value
    # ---------------------------------------------- task-supervision weight study
    for w in C.SEMANTIC_SWEEP:
        entries.append(Experiment(semweight_key(w), "slista", "S-LISTA", semantic_weight=w))
    return {e.key: e for e in entries}


def semweight_key(weight):
    return f"slista_dense_sem{weight:g}"


REGISTRY = _registry()

GROUPS = {
    "slista": ["slista_dense", "slista_dense_no_semantic", "slista_aer", "slista_aer_no_rate",
               "slista_aer_no_semantic"],
    "budget": [f"slista_dense_c{c}" for c in C.BUDGET_CHANNELS if c != 4]
              + [f"vanilla_dense_c{c}" for c in C.BUDGET_CHANNELS],
    "baselines": ["vanilla_dense", "vanilla_aer", "ann_lista", "ann_lstm"],
    "semweight": [semweight_key(w) for w in C.SEMANTIC_SWEEP],
}

# Experiments that make up each paper item.
MAIN_KEYS = ["slista_dense", "vanilla_dense", "ann_lista", "ann_lstm"]           # Table 1, Fig. 4
SEMANTIC_KEYS = ["slista_dense", "slista_dense_no_semantic"]                     # Fig. 5e,f
ABLATION_KEYS = ["slista_aer", "slista_aer_no_rate", "slista_aer_no_semantic"]  # Table 2 (no temperature row)
ENERGY_KEYS = ["slista_aer", "vanilla_aer", "ann_lista", "ann_lstm"]             # Table 3 (AER models)
AER_BIT_KEYS = ["slista_aer", "vanilla_aer"]                                     # Fig. 5b


def budget_key(method, channels):
    if method == "slista":
        return "slista_dense" if channels == 4 else f"slista_dense_c{channels}"
    return f"vanilla_dense_c{channels}"


# ----------------------------------------------------------------- settings
def load_slista_config(path=C.SLISTA_CONFIG_FILE):
    """S-LISTA defaults overlaid with configs/slista.json (the selected settings)."""
    settings = dict(C.SLISTA_DEFAULTS)
    path = Path(path)
    if path.is_file():
        selected = json.loads(path.read_text())
        unknown = set(selected) - set(settings)
        if unknown:
            raise ValueError(f"Unknown S-LISTA settings in {path}: {sorted(unknown)}")
        settings.update(selected)
    return settings


def training_settings(experiment):
    if experiment.method == "slista":
        out = dict(C.SLISTA_TRAINING)
    elif experiment.method == "vanilla":
        out = dict(C.Config.VANILLA)
        out["tau_init"] = C.SLISTA_TRAINING["tau_init"]
        out["tau_final"] = C.SLISTA_TRAINING["tau_final"]
    else:
        return {}
    out["semantic_weight"] = experiment.semantic_weight
    out["rate_weight"] = experiment.rate_weight
    out["tau_mode"] = experiment.tau_mode
    return out


def classifier_settings(experiment):
    """Downstream classifier protocol of one method (same augmentation for every method)."""
    return {"lr": C.CLASSIFIER_LR[experiment.method], **C.CLASSIFIER_AUGMENT}


def make_spec(key, seed, run_dir, slista=None, recon_only=False, with_classifier=None, tag=None,
              epochs=None):
    experiment = REGISTRY[key]
    training = training_settings(experiment)
    if epochs is not None and training:
        training["epochs"] = int(epochs)
    spec = {
        "experiment": asdict(experiment),
        "seed": int(seed),
        "run_dir": str(run_dir),
        "slista": dict(slista if slista is not None else load_slista_config())
        if experiment.method == "slista" else {},
        "training": training,
        "classifier_settings": classifier_settings(experiment),
        "recon_only": bool(recon_only),
        "with_classifier": experiment.classifier if with_classifier is None else bool(with_classifier),
        "tag": tag or key,
        "code_version": C.CODE_VERSION,
    }
    return spec


def formal_run_dir(key, seed):
    return C.OUTPUTS / "runs" / key / f"seed{int(seed)}"


def recon_fingerprint(spec):
    """Everything that defines the reconstruction weights (used to decide retraining)."""
    keep = copy.deepcopy({k: spec[k] for k in ("experiment", "seed", "slista", "training")})
    for name, default in C.SLISTA_FINGERPRINT_DEFAULTS.items():
        if keep["slista"].get(name) == default:
            keep["slista"].pop(name)
    keep["training"].pop("classifier_lr", None)
    keep["experiment"].pop("classifier", None)
    keep["experiment"].pop("key", None)
    keep["experiment"].pop("display", None)
    return keep


def experiment_from_spec(spec):
    return Experiment(**spec["experiment"])


# ------------------------------------------------------------ baseline weight paths
def _read_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def baseline_nominal_row(baseline_key, seed):
    path = C.BASELINE_RESULTS / f"formal_point_{baseline_key}_seed{int(seed)}.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [r for r in _read_rows(path) if abs(float(r["Eval SNR (dB)"]) - 10.0) < 1e-9]
    if len(rows) != 1:
        raise ValueError(f"Expected one AWGN 10 dB row in {path}")
    return rows[0], path


def _relocate(recorded, folder):
    p = Path(recorded)
    if p.is_file():
        return p
    p = folder / p.name
    if p.is_file():
        return p
    aliased = Path(re.sub(r"_th\d+\.\d+\.pth$", "_th0.10.pth", str(p)))   # the file name carries the initial threshold
    if aliased.is_file():
        return aliased
    raise FileNotFoundError(recorded)


def baseline_weights(baseline_key, seed):
    """(recon checkpoint, classifier checkpoint, nominal CSV row) of a baseline."""
    row, source = baseline_nominal_row(baseline_key, seed)
    recon = _relocate(row["Recon Checkpoint"], source.parent)
    classifier = _relocate(row["Classifier Checkpoint"], source.parent)
    return recon.resolve(), classifier.resolve(), row
