"""Settings of the DVS-Gesture baseline experiments (Vanilla SNN, ANN LISTA, ANN ConvLSTM).

Importing this module requires CUDA, reads selected_awgn_parameters.json (the
selected learning rates and Vanilla SNN settings) and creates the results
folder.  train.py sets the experiment-specific values before a model is built.
"""
import os
import torch
import math
import json

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class Config:
    """Global settings read by the sensor, the channel, the models and the trainer."""
    CODE_VERSION = "dvsg-awgn-formal"
    if not torch.cuda.is_available():
        raise SystemError("CRITICAL ERROR: CUDA is not available!")
    
    DEVICE = torch.device("cuda")
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    IMG_H, IMG_W = 128, 128
    INPUT_CHANNELS = 2  
    NUM_CLASSES = 11 

    MEAS_STRIDE = 4 
    MEAS_CHANNELS = 4
    LATENT_DIM = (IMG_H // MEAS_STRIDE) * (IMG_W // MEAS_STRIDE) * MEAS_CHANNELS
    
    FEATURE_DIM = 64
    SPARSE_DIM = 128 
    NUM_LAYERS = 8 
    
    USE_WIRELESS = True
    TRAIN_SNR_DB = 10.0
    EVAL_SNR_DB_LIST = [
        -5.0, -4.0, -3.0, -2.0, -1.0,
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0,
        7.5, 10.0, 15.0, 20.0,
    ]
    CHANNEL_BIT_ZERO = 0.0
    CHANNEL_BIT_ONE = 1.0
    CHANNEL_HARD_THRESHOLD = 0.5
    ANN_RX_MODE = "hard"

    DT = 1e-3                  
    TIME_STEPS = 16      
    SURROGATE_WIDTH = 0.5  
     
    TX_RHO = 0.1
    DECODER_TAU = 0.7
    DECODER_SURROGATE_TAU = 0.5
    DECODER_SURROGATE_MODE = "sigmoid"
    
    ENCODER_THRESHOLD = 0.20
    DECODER_THRESHOLD = 0.20
    INIT_THRESHOLD = 0.10
    THETA_LISTA_ANN = INIT_THRESHOLD
    THETA_LISTA_SNN = INIT_THRESHOLD
    SNN_ETA_SCALE = 1.0

    BATCH_SIZE_RECON = 64
    BATCH_SIZE_CLS = 32  
    NUM_WORKERS = 8             
    SEED = 42
    DEFAULT_SEEDS = [42, 43, 44, 45, 46]
    EVAL_CHANNEL_SEED = 2026
    
    ALPHA_MSE = 1.0
    ALPHA_L1_LISTA_SNN = 1e-7
    ALPHA_L1_LISTA_ANN = 1e-7
    ALPHA_L1_LSTM = 1e-7
    ALPHA_L1_SCNN = 1e-7
    ALPHA_RATE_LISTA = 6e-4
    ALPHA_RATE_SCNN = 6e-4
    # CE is divided by log(NUM_CLASSES), so ALPHA_SEMANTIC has a comparable
    # interpretation across DVS-Gesture and SHD.
    ALPHA_SEMANTIC = 0.01
    SEMANTIC_DROPOUT = 0.50
    SEMANTIC_WEIGHT_DECAY = 1e-5
    # Reproduce the original frozen Phase-4 latent augmentation.  A single
    # zero-padded H/W translation is shared by all frames and channels of
    # each sample.  It is used only by the semantic branch during training.
    SEMANTIC_SHIFT_RATIO = 0.10
    SEMANTIC_SHIFT_PROB = 1.0
    SEMANTIC_HEAD_TAG = "latent_3dconv_c64_128_256"
    SEMANTIC_EVAL_INTERVAL = 5
    TSNE_COMPARISON_MAX_PER_REGIME = 1200
    TSNE_PERPLEXITY = 30.0

    TAU_INIT = 1.0
    TAU_FINAL = 0.1
    TAU_FIXED = 0.1
    TAU_ANNEAL_TYPE = "exp"

    # Annealing diagnostics.  These do not alter the forward path.
    LOG_ANNEAL_STATS = True
    ANNEAL_DENSITY_MAX_SAMPLES = 100000
    ANNEAL_DENSITY_BINS = 200
    
    TX_ABLATION_MODE = 'none'

    USE_RELAXED_AER_AWGN = True
    AER_RELAXATION_TYPE = "gaussian_expected_routing"

    # The 4096-dimensional DVS-Gesture measurement map is divided into
    # exactly 256 blocks.  BLOCK_SIZE is therefore 16.
    NUM_BLOCKS = 256
    if LATENT_DIM % NUM_BLOCKS != 0:
        raise ValueError("LATENT_DIM must be divisible by NUM_BLOCKS.")
    BLOCK_SIZE = LATENT_DIM // NUM_BLOCKS
    # Dense binary transmission is used by the main and robustness tests.
    # Block-AER remains available for the performance--bit experiment.
    SNN_TRANSPORT = "dense_binary_awgn_hard"
    ANN_QUANT_BITS = 8
    
    
    TX_RESET_MODE = "soft"
    LOG_TX_STATE_STATS = True

    RATE_BITS_PER_EVENT = max(1, math.ceil(math.log2(MEAS_CHANNELS))) + \
                          max(1, math.ceil(math.log2(IMG_H // MEAS_STRIDE))) + \
                          max(1, math.ceil(math.log2(IMG_W // MEAS_STRIDE)))

    PHASE3_CONFIG = {
        'epochs': 150,
        'lr_lista_snn': 3e-4,   
        'lr_lista_ann': 7e-4,   
        'lr_lstm': 5e-4,        
        'lr_spiking_cnn': 1e-3,
        'lr_theta_ann': 5e-4,
        'lr_semantic': 3e-4,
        'lr_sensor':     3e-4,
        'lr_dict': 5e-4,
        'lr_min':     1e-7,
        'weight_decay': 0.0
    }

    PHASE4_CONFIG = {
        'epochs': 100,
        'lr': 5e-4,
        'lr_min': 1e-6,
        'weight_decay': 1e-4,
    }
    
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.environ.get("DVSG_DATA_DIR", os.path.join(
        os.path.dirname(PROJECT_ROOT), "data", "DVSGesture",
        "frames_number_16_split_by_number",
    ))
    RESULTS_DIR = os.path.join(
        PROJECT_ROOT, f"results_dvsg_awgn_formal_T{TIME_STEPS}"
    )
    SELECTED_PARAMETER_FILE = os.path.join(
        PROJECT_ROOT, "selected_awgn_parameters.json"
    )
    SELECTED_SNN_PARAMS = {}
    SELECTED_CLASSIFIER_LR = {}

    @staticmethod
    def load_selected_parameters(path=None):
        path = path or Config.SELECTED_PARAMETER_FILE
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as handle:
            selected = json.load(handle)
        ann = selected.get("ann", {})
        snn = selected.get("snn", {})
        Config.SELECTED_SNN_PARAMS = snn
        Config.SELECTED_CLASSIFIER_LR = selected.get("classifier", {})
        if "ann_lista_lr" in ann:
            Config.PHASE3_CONFIG["lr_lista_ann"] = float(ann["ann_lista_lr"])
        if "ann_lstm_lr" in ann:
            Config.PHASE3_CONFIG["lr_lstm"] = float(ann["ann_lstm_lr"])
        if "lr" in snn.get("slista", {}):
            Config.PHASE3_CONFIG["lr_lista_snn"] = float(snn["slista"]["lr"])
        if "lr" in snn.get("scnn", {}):
            Config.PHASE3_CONFIG["lr_spiking_cnn"] = float(snn["scnn"]["lr"])

    @staticmethod
    def awgn_sigma(snr_db=None):
        value = Config.TRAIN_SNR_DB if snr_db is None else float(snr_db)
        ebn0 = 10.0 ** (value / 10.0)
        amplitude = 0.5 * (
            Config.CHANNEL_BIT_ONE - Config.CHANNEL_BIT_ZERO
        )
        return amplitude / math.sqrt(2.0 * ebn0)

    @staticmethod
    def awgn_hard_ber(snr_db=None):
        sigma = Config.awgn_sigma(snr_db)
        return 0.5 * math.erfc(
            0.5 / (math.sqrt(2.0) * sigma)
        )

    @staticmethod
    def transport_tag(mode):
        if mode == "ann":
            return (
                f"dense_q{int(Config.ANN_QUANT_BITS)}_bitawgn_"
                f"{Config.ANN_RX_MODE}"
            )
        if Config.SNN_TRANSPORT == "block_aer_awgn_hard":
            return f"blockaer_awgn_hard_n{int(Config.NUM_BLOCKS)}"
        return str(Config.SNN_TRANSPORT)

    @staticmethod
    def semantic_config_tag():
        hp = Config.PHASE3_CONFIG
        alpha = f"{Config.ALPHA_SEMANTIC:g}".replace(".", "p")
        lr = f"{float(hp['lr_semantic']):g}".replace(".", "p")
        return f"sem{alpha}_semlr{lr}_{Config.SEMANTIC_HEAD_TAG}"

    @staticmethod
    def get_recon_ckpt_name(
        mode, model_name, threshold=None, alpha_rate=None
    ):
        th_suffix = (
            f"_th{float(threshold):.2f}"
            if threshold is not None else ""
        )
        transport = Config.transport_tag(mode)
        sem_tag = f"{Config.ALPHA_SEMANTIC:.6g}".replace(".", "p")
        alpha = (
            float(alpha_rate)
            if alpha_rate is not None else float(Config.ALPHA_RATE_LISTA)
        )
        return (
            f"dynamic_recon_{mode}_{transport}_{model_name.lower()}"
            f"_T{Config.TIME_STEPS}_trainsnr{Config.TRAIN_SNR_DB:g}"
            f"_rho{Config.TX_RHO}_alpha{alpha}_sem{sem_tag}"
            f"_{Config.SEMANTIC_HEAD_TAG}"
            f"_{Config.semantic_config_tag()}{th_suffix}.pth"
        )
    
    @staticmethod
    def setup(args):
        if args.gpu is not None:
            Config.DEVICE = torch.device(f"cuda:{args.gpu}")
            print(f" -> Configured to use GPU: {args.gpu} ({Config.DEVICE})")
            
        if hasattr(args, "threshold") and args.threshold is not None:
            Config.INIT_THRESHOLD = args.threshold
            Config.THETA_LISTA_ANN = args.threshold
            Config.THETA_LISTA_SNN = args.threshold
            print(f" -> Configured Initial Threshold: {Config.INIT_THRESHOLD}")
            
        if hasattr(args, 'tx_rho') and args.tx_rho is not None:
            Config.TX_RHO = args.tx_rho
            print(f" -> Configured Transmitter Rho (State Retention): {Config.TX_RHO}")

        if hasattr(args, "train_snr_db") and args.train_snr_db is not None:
            Config.TRAIN_SNR_DB = float(args.train_snr_db)
        if hasattr(args, "eval_snr_db") and args.eval_snr_db is not None:
            Config.EVAL_SNR_DB_LIST = [
                float(value) for value in args.eval_snr_db
            ]

        if (
            hasattr(args, "semantic_weight")
            and args.semantic_weight is not None
        ):
            if args.semantic_weight < 0:
                raise ValueError("--semantic_weight must be non-negative.")
            Config.ALPHA_SEMANTIC = float(args.semantic_weight)

    @staticmethod
    def get_tau(epoch, total_epochs):
        if Config.TAU_ANNEAL_TYPE == "exp":
            return Config.TAU_INIT * (Config.TAU_FINAL / Config.TAU_INIT) ** (epoch / max(1, total_epochs - 1))
        if Config.TAU_ANNEAL_TYPE == "fixed":
            return Config.TAU_FIXED
        return Config.TAU_INIT

    @staticmethod
    def make_dir():
        try:
            os.makedirs(Config.RESULTS_DIR, exist_ok=True)
        except OSError as e:
            print(f"ERROR: Could not create results directory at {Config.RESULTS_DIR}")
            raise

    @staticmethod
    def print_config():
        print(f"Device: {Config.DEVICE}")
        print(f"Batch Size (Recon): {Config.BATCH_SIZE_RECON}")
        print(f"Batch Size (Cls): {Config.BATCH_SIZE_CLS}")
        print(f"Latent Resolution: {Config.IMG_H // Config.MEAS_STRIDE}x{Config.IMG_W // Config.MEAS_STRIDE}")
        print(f"Training Eb/N0: {Config.TRAIN_SNR_DB:g} dB")
        print(f"Evaluation Eb/N0: {Config.EVAL_SNR_DB_LIST}")
        print(f"SNN Transport: {Config.SNN_TRANSPORT}")
        print(
            f"Block-AER: {Config.NUM_BLOCKS} blocks x "
            f"{Config.BLOCK_SIZE} measurements"
        )
        print(f"ANN Quantization: Q{Config.ANN_QUANT_BITS}")
        print(f"Semantic Loss Weight: {Config.ALPHA_SEMANTIC}")
        print(
            "Semantic Latent Shift: "
            f"ratio={Config.SEMANTIC_SHIFT_RATIO}, "
            f"prob={Config.SEMANTIC_SHIFT_PROB}"
        )
        print(f"Initial Threshold: {Config.INIT_THRESHOLD}")
        print(f"Transmitter Rho (TX_RHO): {Config.TX_RHO}")
        print(f"Saving Results to: {Config.RESULTS_DIR}")
        
Config.load_selected_parameters()
Config.make_dir()
