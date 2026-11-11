"""ECG model and training defaults."""
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT_DIR / "outputs"
import torch

class Config:
    # Bump this tag whenever an algorithm definition changes so that stale
    # checkpoints cannot be loaded silently.
    CODE_VERSION = "ecg-slista"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("high")

    N = 256
    TOTAL_SIGNAL_LEN = N
    # Single-frame ECG beats: one time step, so the membrane leak rho of the
    # paper's Eq. (6) never acts and is not a parameter of this code.

    TOTAL_MEAS_TARGET = 78
    M = max(1, round(TOTAL_MEAS_TARGET * N / 256))
    EFFECTIVE_TOTAL_MEAS = M

    DICT_SIZE = N
    # Unfolded depth L.  S-LISTA: L-1 = 7 spiking layers + 1 non-spiking
    # readout layer, i.e. one m x n and L-1 n x n matrices, the same matrix
    # count as ANN LISTA with K = L layers.
    NUM_LAYERS = 8

    # S-LISTA threshold initialisation (dimensionless, see SpikingLISTA):
    # theta_1 = kappa * sigma, theta_{L-1} = beta * sigma, geometric in
    # between; eta = 1 / theta_{L-1}.  sigma = standard deviation of the
    # initial current P y over all coordinates of one training batch.
    SLISTA_KAPPA = "universal"          # sqrt(2 ln n)
    SLISTA_BETA = 2.0
    SLISTA_ALPHA = 0.4                  # surrogate half-width / theta_l
    SLISTA_THRESHOLD_FLOOR = 1e-2       # theta = floor + softplus(r)
    SLISTA_CALIBRATION_SEED = 2027      # offset added to SEED: calibration batch and channel draw
    # S-LISTA learning rates of the sensing matrix and the dictionary relative
    # to lr_model (P, G, thresholds, eta).  configs/paper.json may override.
    SLISTA_PHI_LR_RATIO = 0.5
    SLISTA_DICTIONARY_LR_RATIO = 1.0
    # Learning rate of the firing thresholds relative to lr_model: controls how
    # fast the thresholds move (decrease) during training.
    SLISTA_THETA_LR_RATIO = 1.0
    # Learning rate of the residual operators G relative to lr_model (P, eta).
    SLISTA_G_LR_RATIO = 1.0
    VAL_EVERY = 10                      # validation / checkpoint selection interval (epochs)

    USE_WIRELESS = True
    TRAIN_SNR_DB = 5.0
    SNR_DB = TRAIN_SNR_DB
    EVAL_SNR_DB_GRID = [-5.0, -2.0, 0.0, 2.0, 5.0, 7.5, 10.0, 12.5, 15.0, 20.0]
    
    QUANT_BITS = 8
    QUANT_CLIP_VALUE = 3.0
    BITS_PER_SAMPLE = M * QUANT_BITS
    
    TRAIN_CHANNEL_MODE = "bit_awgn_hard"
    EVAL_CHANNEL_MODE = "bit_awgn_hard"

    BATCH_SIZE = 256
    NUM_WORKERS = 2
    SEED = 42
    EVAL_CHANNEL_SEED = 2026

    ALPHA_MSE = 1.0
    ALPHA_L1_ANN = 1e-4
    ALPHA_L1_SNN = 1e-4
    DICTIONARY_INIT = "random"
    CS_MODE = "learnable_dictionary"
    DICTIONARY_LR = 1e-3
    MEASUREMENT_INIT = "gaussian"
    MEASUREMENT_NORMALIZATION = "none"
    WAVELET_NAME = "sym4"
    LR_SCALE = 1.0

    THETA_LAMP = 0.20
    THETA_LISTA_ANN = 0.20
    # Standard fan-in Kaiming-uniform initialization of P and G.
    SLISTA_INIT = "random"

    THETA_FISTA = 0.20
    THETA_ALISTA = 0.20
    FISTA_ITERATIONS = NUM_LAYERS

    E_MAC_UJ = 4.6e-6
    E_AC_UJ = 0.9e-6

    FORMAL_TRAIN_EPOCHS = 200

    LAMP_ANN_CONFIG = {
        "epochs": FORMAL_TRAIN_EPOCHS, "lr_model": 3e-3,
        "lr_theta": 3e-3,
        "lr_phi": 1.5e-3,
        "lr_dictionary": 3e-3,
        "weight_decay": 0.0,
    }
    LISTA_ANN_CONFIG = {
        "epochs": FORMAL_TRAIN_EPOCHS, "lr_model": 7e-4,
        "lr_theta": 7e-4,
        "lr_phi": 2e-3,
        "lr_dictionary": 1e-3,
        "weight_decay": 0.0,
    }
    # Base learning rates of every method.  Each job multiplies all rates of
    # its method by one scale s (configs/paper.json "lr_scale", selected on
    # the validation split), so only the ratios between the
    # parameter groups below are fixed.  S-LISTA: P, G, thresholds and eta use
    # lr_model; the sensing matrix 0.5 lr_model; the dictionary 1.0 lr_model.
    LISTA_SNN_CONFIG = {
        "epochs": FORMAL_TRAIN_EPOCHS, "lr_model": 1e-3,
        "lr_phi": 5e-4,
        "lr_dictionary": 1e-3,
        "p_weight_decay": 0.0,
        "weight_decay": 0.0,
    }
    
    
    ALISTA_CONFIG = {
        "epochs": FORMAL_TRAIN_EPOCHS, "lr_model": 2e-2,
        "lr_theta": 2e-2,
        "lr_phi": 1e-3,
        "lr_dictionary": 1e-3,
        "weight_decay": 0.0,
    }
    FISTA_CONFIG = {
        "epochs": 0, "lr_model": 0.0, "lr_phi": 0.0,
        "weight_decay": 0.0,
    }

    RESULTS_DIR = str(DEFAULT_OUTPUT)
    WARM_START_CHECKPOINT = None
    DATA_DIR = str(PROJECT_DIR / "data" / "mitdb_pt")
    TRAIN_FILE = "train_data.pt"
    VAL_FILE = "val_data.pt"
    TEST_FILE = "test_data.pt"
    TEST_CHANNEL_SEED = 20260919



    @staticmethod
    def setup(args):
        if getattr(args, "gpu", None) is not None and torch.cuda.is_available():
            Config.DEVICE = torch.device(f"cuda:{args.gpu}")
        if getattr(args, "data_dir", None):
            Config.DATA_DIR = args.data_dir
        if getattr(args, "m", None) is not None:
            Config.M = args.m
            Config.TOTAL_MEAS_TARGET = args.m
            Config.EFFECTIVE_TOTAL_MEAS = args.m
            Config.BITS_PER_SAMPLE = args.m * Config.QUANT_BITS
        if getattr(args, "train_snr_db", None) is not None:
            Config.TRAIN_SNR_DB = float(args.train_snr_db)
            Config.SNR_DB = float(args.train_snr_db)
        if getattr(args, "quant_bits", None) is not None:
            Config.QUANT_BITS = int(args.quant_bits)
            Config.BITS_PER_SAMPLE = Config.M * Config.QUANT_BITS
        if getattr(args, "cs_mode", None) is not None:
            Config.CS_MODE = args.cs_mode

    @staticmethod
    def make_dir():
        os.makedirs(Config.RESULTS_DIR, exist_ok=True)

    @staticmethod
    def print_config():
        print(f"Device: {Config.DEVICE}")
        print(f"Wireless enabled: {Config.USE_WIRELESS}")
        print(f"AWGN SNR: {Config.SNR_DB:g} dB")
        print(f"Quantization bits: {Config.QUANT_BITS}-bit")
        print(f"Bits per sample: {Config.BITS_PER_SAMPLE}")
        print(f"Train channel mode: {Config.TRAIN_CHANNEL_MODE}")
        print(f"Eval channel mode: {Config.EVAL_CHANNEL_MODE}")
        print(f"Compression M/N: {Config.M}/{Config.N}")
        print(f"Results dir: {Config.RESULTS_DIR}")

