import os
from pathlib import Path
import torch
import math

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class Config:
    CODE_VERSION = "shd-awgn-formal-5seed-v8-robust-slista"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    PROJECT_ROOT = str(Path(__file__).resolve().parent)
    DATASET = "SHD"
    DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
    RAW_DIR = os.path.join(DATA_ROOT, "raw")
    PROCESSED_DIR = os.environ.get("SHD_DATA_DIR", os.path.join(DATA_ROOT, "processed"))
    
    INPUT_DIM = 700
    N = 700
    NUM_CLASSES = 20
    
    TIME_STEPS = 25
    T_STOP_MS = 1000.0
    BIN_DT_MS = T_STOP_MS / TIME_STEPS
    BIN_DT_SEC = BIN_DT_MS / 1000.0

    # Dataset Formatting Configurations
    DATA_FORMAT = "raw_count"
    CLIP_SPIKE_COUNT = False
    NORMALIZE_SPIKE_COUNT = False
    COUNT_CLIP_VALUE = None
    DECODER_OUTPUT = "linear"

    # Processed file names
    TRAIN_FILE = f"shd_train_T{TIME_STEPS}_D{INPUT_DIM}_rawcount.pt"
    TEST_FILE = f"shd_test_T{TIME_STEPS}_D{INPUT_DIM}_rawcount.pt"

    SEED = 42
    DEFAULT_SEEDS = [42, 43, 44, 45, 46]
    # Every stochastic channel evaluation is restarted from the same seed so
    # all models and SNR points see reproducible AWGN realizations.
    EVAL_CHANNEL_SEED = 2026

    MEAS_DIM = 256
    M = MEAS_DIM
    SPARSE_DIM = 700
    NUM_LAYERS = 8
    
    USE_WIRELESS = True
    # Bit-input AWGN operating point.  Binary channel symbols are physically
    # represented by amplitudes {0, 1}; the useful centered constellation is
    # {-0.5, +0.5}.  TRAIN_SNR_DB is therefore interpreted as Eb/N0 for that
    # centered binary constellation.  Formal training uses a fixed 10 dB SNR;
    # evaluation covers the low-to-high SNR regime in 5 dB increments.
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
    TX_RHO = 0.1
    DECODER_TAU = 0.3

    # Formal SNN operating points selected by independent pilots.
    SENSING_NORMALIZATION = "energy"
    # Selected by the sequential S-LISTA robustness pilot.  The pilot keeps
    # ACC@10 dB unchanged while improving mean ACC over -5--5 dB.
    SLISTA_TX_THRESHOLD = 0.75
    SCNN_TX_THRESHOLD = 1.5
    SCNN_NEURON_THRESHOLD = 0.8
    SCNN_LEAK = 0.06
    SCNN_SURROGATE_TAU = 0.30
    SLISTA_DECODER_TAU = 0.3
    SLISTA_SURROGATE_TAU = 0.7
    SCNN_DECODER_TAU = 0.06
    GENERIC_SNN_TX_THRESHOLD = 2.0
    GENERIC_SNN_NEURON_THRESHOLD = 0.5
    GENERIC_SNN_LEAK = 0.1
    GENERIC_SNN_SURROGATE_TAU = 0.3
    GENERIC_SNN_HIDDEN_DIM = 700
    GENERIC_SNN_BLOCKS = 3
    

    ENCODER_THRESHOLD = 1.0
    THETA_LISTA_ANN = 0.05
    THETA_LISTA_SNN = 0.65
    # Formal S-LISTA readout.  Comparison jobs may temporarily select
    # "spike_only" for the dedicated amplitude-readout ablation.
    SLISTA_MEMBRANE_READOUT = "support_gated"


    BATCH_SIZE_RECON = 512
    # Downstream reconstruction-input classifiers use batch size 256.
    BATCH_SIZE_CLS = 256
    NUM_WORKERS = int(os.environ.get("SHD_NUM_WORKERS", "8"))             
    
    ALPHA_MSE = 1.0
    # Sparse-code regularization weights.
    ALPHA_L1_ANN = 1e-3
    ALPHA_L1_SNN = 1e-3   
    
    ALPHA_RATE_SLISTA = 1e-1
    # Rate regularization for the SCNN AER operating point.
    ALPHA_RATE_SCNN = 1.2e-1
    # CE is divided by log(NUM_CLASSES), so a random semantic classifier
    # contributes approximately ALPHA_SEMANTIC to the joint objective.
    ALPHA_SEMANTIC = 3.0
    # Diagnostic ablations can train the same semantic head on detached Z.
    # This isolates semantic feedback into the representation while avoiding
    # the invalid comparison against an entirely untrained classifier.
    SEMANTIC_DETACH_BACKBONE = False
    # Keep the diagnostic head's CE scale equal to the proposed setting when
    # semantic feedback into Z is disabled.
    DETACHED_HEAD_LOSS_WEIGHT = 3.0
    SEMANTIC_TCN_DIM = 384
    SEMANTIC_NORM_GROUPS = 32
    SEMANTIC_DROPOUT = 0.20
    SEMANTIC_WEIGHT_DECAY = 1e-4
    SEMANTIC_HEAD_TAG = "tcn_f256_d124"
    SEMANTIC_EVAL_INTERVAL = 5

    # Final task accuracy is measured by a separate classifier trained on
    # reconstructed x after joint reconstruction/semantic training finishes.
    DOWNSTREAM_TCN_DIM = 384
    DOWNSTREAM_NORM_GROUPS = 32
    DOWNSTREAM_DROPOUT = 0.20
    DOWNSTREAM_LABEL_SMOOTHING = 0.0
    DOWNSTREAM_INIT_SEED_OFFSET = 10000
    PHASE4_CONFIG = {'epochs': 150, 'lr': 1e-3, 'lr_min': 1e-7, 'weight_decay': 1e-4}

    # Z is reported only through a qualitative class-separability plot.
    TSNE_MAX_SAMPLES = 2500
    TSNE_PERPLEXITY = 30.0
    TSNE_COMPARISON_MAX_PER_REGIME = 1200

    TAU_INIT = 1.0
    TAU_FINAL = 0.1
    TAU_FIXED = 0.1
    TAU_ANNEAL_TYPE = "exp"

    # Theorem-1 diagnostics.  These statistics are detached and do not change
    # the forward or backward computation.
    LOG_ANNEAL_STATS = True
    ANNEAL_DENSITY_MAX_SAMPLES = 100000
    ANNEAL_DENSITY_BINS = 200
    
    TX_ABLATION_MODE = "none"
    
    USE_RELAXED_AER_AWGN = True
    AER_RELAXATION_TYPE = "gaussian_expected_routing"

    # Hierarchical AER: each non-empty block sends its high (block) address
    # once, followed only by the local addresses of actual spikes.  Zeros
    # inside the block are not serialized.  Link-layer packet framing is
    # excluded consistently from both ordinary and block-AER bit accounting.
    BLOCK_SIZE = 16
    # Formal ANN measurements use sigmoid-domain 8-bit uniform quantization
    # and a dense bitstream.
    # Robustness experiment: transmit the complete positive/negative spike
    # bitmap.  Block-AER remains implemented separately for rate/Pareto work.
    SNN_TRANSPORT = "dense_spike_awgn_hard"
    ANN_QUANT_BITS = 8
    
 
    
    TX_RESET_MODE = "soft"
    LOG_TX_STATE_STATS = True
    
    RATE_BITS_PER_EVENT = max(1, math.ceil(math.log2(MEAS_DIM)))

    @staticmethod
    def awgn_sigma(snr_db=None):
        """Noise std for {0,1} binary input at the requested centered Eb/N0."""
        value = Config.TRAIN_SNR_DB if snr_db is None else float(snr_db)
        ebn0 = 10.0 ** (value / 10.0)
        amplitude = 0.5 * (
            Config.CHANNEL_BIT_ONE - Config.CHANNEL_BIT_ZERO
        )
        return amplitude / math.sqrt(2.0 * ebn0)

    @staticmethod
    def transport_tag(mode):
        if mode == "ann":
            return (
                f"dense_q{int(Config.ANN_QUANT_BITS)}_bitawgn_"
                f"{Config.ANN_RX_MODE}"
            )
        if Config.SNN_TRANSPORT == "block_aer_awgn_hard":
            return f"blockaer_awgn_hard_b{int(Config.BLOCK_SIZE)}"
        if Config.SNN_TRANSPORT == "dense_spike_awgn_hard":
            return "dense_spike_awgn_hard"
        return str(Config.SNN_TRANSPORT)

    @staticmethod
    def semantic_config_tag(hp=None):
        hp = Config.PHASE3_CONFIG if hp is None else hp
        alpha = f"{Config.ALPHA_SEMANTIC:g}".replace(".", "p")
        lr = f"{float(hp['lr_semantic']):g}".replace(".", "p")
        return f"sem{alpha}_semlr{lr}_{Config.SEMANTIC_HEAD_TAG}"
    
    FORCE_RETRAIN_RECON = False
    RECON_LR_OVERRIDE = None
    ARTIFACT_TAG = ""
    DOWNSTREAM_LR_OVERRIDE = None
    CLASSIFIER_ARTIFACT_TAG = ""
    SLISTA_P_WEIGHT_DECAY = 0.0


    PHASE3_CONFIG = {
        'epochs': 100,
        'lr_lista_snn': 5e-4,   
        'lr_lista_ann': 7e-4,
        'lr_theta_ann': 5e-4,
        'lr_semantic': 3e-4,
        'lr_lstm': 7e-4,
        'lr_dict': 5e-4,        # Unified synthesis-dictionary LR across LISTA/SCNN
        'lr_sensor': 3e-4,      
        'lr_min': 1e-7,
        'weight_decay': 0.0
    }

    # Model-specific joint-training optimizer settings.
    SLISTA_PHASE3_CONFIG = dict(PHASE3_CONFIG)
    SLISTA_PHASE3_CONFIG.update({
        'epochs': 100,
        'lr_lista_snn': 7.5e-4,
        'lr_dict': 1e-3,
        'lr_sensor': 3e-4,
        'lr_semantic': 8e-4,
    })
    SCNN_PHASE3_CONFIG = dict(PHASE3_CONFIG)
    SCNN_PHASE3_CONFIG.update({
        'epochs': 100,
        'lr_spiking_dense': 6e-4,
        'lr_dict': 5e-4,
        'lr_sensor': 3e-4,
        'lr_semantic': 3e-4,
    })
    GENERIC_SNN_PHASE3_CONFIG = dict(PHASE3_CONFIG)
    GENERIC_SNN_PHASE3_CONFIG.update({
        'epochs': 100,
        'lr_generic_snn': 6e-4,
        'lr_dict': 5e-4,
        'lr_sensor': 3e-4,
        'lr_semantic': 3e-4,
    })
    
    RESULTS_DIR = os.environ.get("SHD_OUTPUT_DIR", os.path.join(PROJECT_ROOT, "outputs_binary"))
    
    @staticmethod
    def get_recon_ckpt_name(
        mode, model_name, threshold=None, alpha_rate=None, hp=None
    ):
        th_suffix = f"_th{threshold:.2f}" if threshold is not None else ""
        transport = Config.transport_tag(mode)
        sem_tag = f"{Config.ALPHA_SEMANTIC:.6g}".replace(".", "p")
        semantic_tag = Config.semantic_config_tag(hp)
        
        # Fallback to SLISTA value if none provided
        alpha_val = alpha_rate if alpha_rate is not None else getattr(Config, 'ALPHA_RATE_SLISTA', 1e-2)
        
        return (
            f"dynamic_recon_{mode}_{transport}_{model_name.lower()}_"
            f"{Config.DATA_FORMAT}_{Config.DECODER_OUTPUT}_"
            f"T{Config.TIME_STEPS}_trainsnr{Config.TRAIN_SNR_DB:g}_"
            f"rho{Config.TX_RHO}_alpha{alpha_val}_sem{sem_tag}_"
            f"{semantic_tag}"
            f"{th_suffix}.pth"
        )

    @staticmethod
    def setup(args):
        if args.gpu is not None:
            Config.DEVICE = torch.device(f"cuda:{args.gpu}")
            print(f" -> Configured to use GPU: {args.gpu} ({Config.DEVICE})")
            
        if hasattr(args, 'threshold') and args.threshold is not None:
            Config.THETA_LISTA_ANN = args.threshold
            Config.THETA_LISTA_SNN = args.threshold
            
        if hasattr(args, 'train_snr_db') and args.train_snr_db is not None:
            Config.TRAIN_SNR_DB = float(args.train_snr_db)

        if hasattr(args, 'eval_snr_db') and args.eval_snr_db:
            Config.EVAL_SNR_DB_LIST = [float(v) for v in args.eval_snr_db]

        if hasattr(args, 'ann_rx_mode') and args.ann_rx_mode is not None:
            Config.ANN_RX_MODE = str(args.ann_rx_mode)
            
        if hasattr(args, 'decoder_tau') and args.decoder_tau is not None:
            Config.DECODER_TAU = args.decoder_tau
            
        if hasattr(args, 'force_retrain_recon'):
            Config.FORCE_RETRAIN_RECON = args.force_retrain_recon
            
        if hasattr(args, 'tx_rho') and args.tx_rho is not None:
            Config.TX_RHO = args.tx_rho

        if hasattr(args, 'semantic_weight') and args.semantic_weight is not None:
            if args.semantic_weight < 0:
                raise ValueError("--semantic_weight must be non-negative.")
            Config.ALPHA_SEMANTIC = float(args.semantic_weight)

        if hasattr(args, 'recon_lr') and args.recon_lr is not None:
            if args.recon_lr <= 0:
                raise ValueError("--recon_lr must be positive.")
            Config.RECON_LR_OVERRIDE = float(args.recon_lr)

        if hasattr(args, 'artifact_tag') and args.artifact_tag is not None:
            safe = str(args.artifact_tag).strip()
            if safe and not all(c.isalnum() or c in "-_" for c in safe):
                raise ValueError("--artifact_tag must be alphanumeric, '-' or '_'.")
            Config.ARTIFACT_TAG = safe

        if hasattr(args, 'classifier_lr') and args.classifier_lr is not None:
            if args.classifier_lr <= 0:
                raise ValueError("--classifier_lr must be positive.")
            Config.DOWNSTREAM_LR_OVERRIDE = float(args.classifier_lr)

        if hasattr(args, 'classifier_tag') and args.classifier_tag is not None:
            safe = str(args.classifier_tag).strip()
            if safe and not all(c.isalnum() or c in "-_" for c in safe):
                raise ValueError(
                    "--classifier_tag must be alphanumeric, '-' or '_'."
                )
            Config.CLASSIFIER_ARTIFACT_TAG = safe

        if (
            hasattr(args, 'slista_p_weight_decay')
            and args.slista_p_weight_decay is not None
        ):
            if args.slista_p_weight_decay < 0:
                raise ValueError(
                    "--slista_p_weight_decay must be non-negative."
                )
            Config.SLISTA_P_WEIGHT_DECAY = float(
                args.slista_p_weight_decay
            )

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
        print(f"\n{'-'*50}")
        print(">>> CONFIGURATION")
        print(f"{'-'*50}")
        print(f"Dataset: {Config.DATASET}")
        print(f"Code Version: {Config.CODE_VERSION}")
        print(f"Device: {Config.DEVICE}")
        print(f"Input Dimension: {Config.INPUT_DIM}")
        print(f"Measurement Dimension: {Config.MEAS_DIM}")
        print(f"Time Steps: {Config.TIME_STEPS}")
        print(f"Decoder Tau: {Config.DECODER_TAU}")
        print(f"Transmitter Rho (TX_RHO): {Config.TX_RHO}")
        print(f"Train Eb/N0: {Config.TRAIN_SNR_DB:g} dB")
        print(f"Evaluation Eb/N0 grid: {Config.EVAL_SNR_DB_LIST}")
        print(f"SNN Transport: {Config.SNN_TRANSPORT}")
        print(f"Block Size: {Config.BLOCK_SIZE}")
        print(
            f"ANN Quantization: sigmoid-domain {Config.ANN_QUANT_BITS}-bit "
            f"with {Config.ANN_RX_MODE} AWGN reception"
        )
        print(f"Semantic Loss Weight: {Config.ALPHA_SEMANTIC}")
        print(
            f"Semantic Head: TCN, "
            f"frame_dim={Config.SEMANTIC_TCN_DIM}, "
            f"dilations=(1,2,4), "
            f"dropout={Config.SEMANTIC_DROPOUT}, "
            f"lr={Config.PHASE3_CONFIG['lr_semantic']}, "
            f"weight_decay={Config.SEMANTIC_WEIGHT_DECAY}"
        )
        print(f"Encoder Threshold (Fixed): {Config.ENCODER_THRESHOLD}")
        print(f"Saving Results to: {Config.RESULTS_DIR}")
        print(f"{'-'*50}\n")
        
Config.make_dir()
