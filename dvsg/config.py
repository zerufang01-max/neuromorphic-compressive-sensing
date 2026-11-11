"""Constants and defaults for the DVS-Gesture experiments.

Importing this module has no side effects: it does not touch CUDA, read
parameter files or create directories.  Everything that changes between
experiments is passed explicitly through an experiment spec (experiments.py);
``Config`` below only carries the fixed dataset geometry and the settings
that the baseline classes (baseline_models.py) read at construction time.
"""

import math
import os
from pathlib import Path

CODE_VERSION = "dvsg-slista"

# ----------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get(
    "DVSG_DATA_DIR",
    str(ROOT / "data" / "DVSGesture" / "frames_number_16_split_by_number"),
))
# Results folder of the baseline training code (baselines/train.py).
BASELINE_RESULTS = Path(os.environ.get(
    "DVSG_BASELINE_RESULTS",
    str(ROOT / "baselines" / "results_dvsg_awgn_formal_T16"),
))

OUTPUTS = ROOT / "outputs"               # formal runs, baselines, tables
CONFIG_DIR = ROOT / "configs"
SLISTA_CONFIG_FILE = CONFIG_DIR / "slista.json"   # selected S-LISTA settings

# --------------------------------------------------------------- protocol
SEEDS = [42, 43, 44, 45, 46]
TRAIN_SNR_DB = 10.0
SNR_GRID = [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0,
            7.5, 10.0, 15.0, 20.0]
EVAL_CHANNEL_SEED = 2026          # evaluation noise seed
CALIBRATION_SEED = 2025           # S-LISTA threshold calibration
CALIBRATION_SAMPLES = 256
BATCH_SIZE = 64                   # reconstruction, classifier and evaluation
NUM_WORKERS = 8
DEFAULT_GPUS = [0]
ENERGY_MAC_PJ = 4.6
ENERGY_AC_PJ = 0.9

# Robustness summary, identical to SHD: AWGN -5..5 dB plus Rayleigh -5..20 dB.
ROBUST_AWGN_MAX_DB = 5.0

# Bit-budget experiment: measurement channels C at a fixed 32 x 32 grid.
BUDGET_CHANNELS = [1, 2, 3, 4, 6, 8]

# Number of AER blocks (16 coordinates each for C = 4).
NUM_BLOCKS = 256


def latent_dim(channels):
    return int(channels) * (Config.IMG_H // Config.MEAS_STRIDE) * (Config.IMG_W // Config.MEAS_STRIDE)


def awgn_sigma(snr_db):
    """Noise std for 0/1 bits with hard threshold 0.5 (BER = Q(sqrt(2 Eb/N0)))."""
    ebn0 = 10.0 ** (float(snr_db) / 10.0)
    return 0.5 / math.sqrt(2.0 * ebn0)


# --------------------------------------------------------- S-LISTA defaults
# Defaults.  The selected values are in configs/slista.json; experiments.load_slista_config()
# merges that file over these defaults explicitly (never at import time).
SLISTA_DEFAULTS = {
    "kappa": 2.5,             # theta_1 = kappa * sigma
    "beta": 2.0,              # theta_{L-1} = beta * sigma (geometric in between)
    "alpha": 0.3,             # boxcar surrogate half-width = alpha * theta
    "rho": 0.0,               # membrane memory of the spiking layers across frames
    "lambda_s": 0.01,         # lambda_s * mean_l ||z^l||_1 / n
    "tx_threshold": 0.25,     # output LIF threshold theta_tx of Eq. (1)
    "p_weight_decay": 0.0,    # Adam L2 on P only
    "lr_scale": 1.0,          # multiplies reconstruction, sensor and dictionary rates
    "theta_lr_ratio": 0.3,
    "g_lr_ratio": 0.5,
    "threshold_floor": 0.01,
    # Random translation (fraction of the 32 x 32 code map, one shift per sample shared by all
    # frames) of the sparse codes fed to the task head during reconstruction training.  0.1 is
    # the value used for the baselines; 0 removes it.
    "semantic_shift": 0.1,
}
# A value equal to its entry here is left out of the run tag and of the reconstruction
# fingerprint.
SLISTA_FINGERPRINT_DEFAULTS = {"semantic_shift": 0.1}

# The two internal LIF layers of the S-LISTA transmitter use this fixed value; only the
# output LIF (theta_tx, Eq. (1)) is tuned, as in SHD.  (Vanilla uses one value, 0.2, for all three.)
SLISTA_INTERNAL_LIF_THRESHOLD = 0.25

# Fixed training settings shared by every S-LISTA run.
SLISTA_TRAINING = {
    "epochs": 150,
    "lr_recon": 4e-4,
    "lr_sensor": 3e-4,
    "lr_dict": 5e-4,
    "lr_semantic": 3e-4,
    "lr_min": 1e-7,
    "semantic_weight": 0.01,  # task-supervision weight of every supervised S-LISTA run
    "rate_weight": 6e-4,      # AER runs only
    "tau_init": 1.0,
    "tau_final": 0.1,
    "classifier_lr": 5e-4,
}

CLASSIFIER_TRAINING = {"epochs": 100, "lr_min": 1e-6, "weight_decay": 1e-4}

# Downstream classifier (all methods, including the baselines whose reconstruction weights
# come from baselines/): random translation of the reconstructed video by up to shift_ratio of the
# frame (integer pixels, zero padding, one shift per sample shared by all 16 frames) and
# label smoothing.  No flips: DVS-G has left/right-hand and (counter)clockwise classes.
CLASSIFIER_AUGMENT = {"shift_ratio": 0.1, "label_smoothing": 0.1}
# Downstream classifier learning rates per method (baselines/selected_awgn_parameters.json).
CLASSIFIER_LR = {"slista": 5e-4, "vanilla": 8e-4, "ann_lista": 8e-4, "ann_lstm": 5e-4}


class Config:
    """Fixed geometry plus the values read by the baseline classes.

    The attribute names are those of baselines/config.py, so that
    baseline_models.py is the same file as baselines/snn_models.py.
    ``use_vanilla_settings`` sets the remaining Vanilla-specific values right
    before a Vanilla model is constructed.
    """

    IMG_H, IMG_W = 128, 128
    INPUT_CHANNELS = 2
    NUM_CLASSES = 11
    TIME_STEPS = 16
    MEAS_STRIDE = 4
    MEAS_CHANNELS = 4
    LATENT_DIM = 4 * 32 * 32
    FEATURE_DIM = 64
    SPARSE_DIM = 128
    NUM_LAYERS = 8

    # Receiver settings used by the baseline classes.
    INIT_THRESHOLD = 0.10
    THETA_LISTA_ANN = 0.10
    THETA_LISTA_SNN = 0.10
    DECODER_TAU = 0.5            # Vanilla leak (persistent buffer, also loaded)
    DECODER_THRESHOLD = 0.2      # Vanilla neuron threshold (NON-persistent buffer)
    DECODER_SURROGATE_TAU = 0.3
    DECODER_SURROGATE_MODE = "sigmoid"
    SURROGATE_WIDTH = 0.5
    SEMANTIC_DROPOUT = 0.50
    SEMANTIC_WEIGHT_DECAY = 1e-5
    SEMANTIC_SHIFT_RATIO = 0.10
    SEMANTIC_SHIFT_PROB = 1.0

    # Vanilla training settings (used for the Vanilla bit-budget runs).
    VANILLA = {
        "lr_spiking_cnn": 4e-4, "lr_dict": 5e-4, "lr_semantic": 3e-4,
        "lr_sensor": 3e-4, "lr_min": 1e-7, "tx_threshold": 0.2,
        "neuron_threshold": 0.2, "leak": 0.5, "surrogate_tau": 0.3,
        "alpha_l1": 1e-7, "semantic_weight": 0.01, "epochs": 150,
        "classifier_lr": 8e-4,
    }
    TX_RHO = 0.1

    @staticmethod
    def set_measurement_channels(channels):
        Config.MEAS_CHANNELS = int(channels)
        Config.LATENT_DIM = latent_dim(channels)

    @staticmethod
    def use_vanilla_settings():
        v = Config.VANILLA
        Config.DECODER_THRESHOLD = float(v["neuron_threshold"])
        Config.DECODER_TAU = float(v["leak"])
        Config.DECODER_SURROGATE_TAU = float(v["surrogate_tau"])
        Config.DECODER_SURROGATE_MODE = "sigmoid"


def setup_torch(device_index=None):
    """Backend flags (the same as in baselines/config.py)."""
    import torch
    if not torch.cuda.is_available():
        raise SystemError("CUDA is required.")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return torch.device(f"cuda:{int(device_index or 0)}")
