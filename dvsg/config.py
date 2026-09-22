import os
import torch
import math

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class Config:
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
    BSC_CROSSOVER_PROB = 0.01 

    DT = 1e-3                  
    TIME_STEPS = 16      
    SURROGATE_WIDTH = 0.5  
     
    TX_RHO = 0.1
    DECODER_TAU = 0.7
    
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
    
    ALPHA_MSE = 1.0
    ALPHA_L1_LISTA_SNN = 1e-7
    ALPHA_L1_LISTA_ANN = 1e-7
    ALPHA_L1_LSTM = 5e-6
    ALPHA_L1_SCNN = 1e-7
    ALPHA_RATE_LISTA = 6.5e-4
    ALPHA_RATE_SCNN = 6.5e-4

    TAU_INIT = 1.0
    TAU_FINAL = 0.1
    TAU_ANNEAL_TYPE = "exp"
    
    TX_ABLATION_MODE = 'none'
    
    USE_RELAXED_AER_BSC = True
    AER_RELAXATION_TYPE = "expected_routing"
    
    
    TX_RESET_MODE = "soft"
    LOG_TX_STATE_STATS = True

    RATE_BITS_PER_EVENT = max(1, math.ceil(math.log2(MEAS_CHANNELS))) + \
                          max(1, math.ceil(math.log2(IMG_H // MEAS_STRIDE))) + \
                          max(1, math.ceil(math.log2(IMG_W // MEAS_STRIDE)))

    PHASE3_CONFIG = {
        'epochs': 150,
        'lr_lista_snn': 3e-4,   
        'lr_lista_ann': 7e-4,   
        'lr_lstm': 7e-4,        
        'lr_spiking_cnn': 1e-3,
        'lr_sensor':     3e-4,
        'lr_dict': 5e-4,
        'lr_min':     1e-7,
        'weight_decay': 0.0
    }
    
    PHASE4_CONFIG = {
        'epochs': 200,
        'lr': 3e-4,
        'lr_min': 1e-7,
        'weight_decay': 3e-3
    }
    
    RESULTS_DIR = "/workspace0/zf925/spiking_lista_dvsg/results_dvsg_recon_final"
    
    @staticmethod
    def setup(args):
        if args.gpu is not None:
            Config.DEVICE = torch.device(f"cuda:{args.gpu}")
            print(f" -> Configured to use GPU: {args.gpu} ({Config.DEVICE})")
            
        if args.threshold is not None:
            Config.INIT_THRESHOLD = args.threshold
            Config.THETA_LISTA_ANN = args.threshold
            Config.THETA_LISTA_SNN = args.threshold
            print(f" -> Configured Initial Threshold: {Config.INIT_THRESHOLD}")
            
        if hasattr(args, 'tx_rho') and args.tx_rho is not None:
            Config.TX_RHO = args.tx_rho
            print(f" -> Configured Transmitter Rho (State Retention): {Config.TX_RHO}")

    @staticmethod
    def get_tau(epoch, total_epochs):
        if Config.TAU_ANNEAL_TYPE == "exp":
            return Config.TAU_INIT * (Config.TAU_FINAL / Config.TAU_INIT) ** (epoch / max(1, total_epochs - 1))
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
        print(f"BSC Crossover Probability: {Config.BSC_CROSSOVER_PROB}")
        print(f"Initial Threshold: {Config.INIT_THRESHOLD}")
        print(f"Transmitter Rho (TX_RHO): {Config.TX_RHO}")
        print(f"Saving Results to: {Config.RESULTS_DIR}")
        
Config.make_dir()