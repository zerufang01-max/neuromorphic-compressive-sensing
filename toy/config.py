"""Experimental protocol for the synthetic sparse-recovery experiments.

Every number used by the paper's synthetic figures is defined here:
  Fig. 2a,b   NMSE / energy vs depth (s = 28)   DEPTH_SWEEP_*
  Fig. 2c     NMSE vs sparsity (L = 10)         SPARSITY_SWEEP_DEPTH, CONDITIONS
  Fig. 2d     AWGN robustness (s = 28, L = 10)  SNR_*
  Fig. 2e,f   coefficient panels (s = 14, L = 6) EXAMPLE_*
  Supp. Fig. 4  S-LCA comparison  SLCA_*             (L = 10, s = 28)
experiment_grid() lists every model that has to be trained (34 in total).
"""

from dataclasses import dataclass

# Numerical constants shared by every method.
NUMERICAL_EPS = 1e-12
POSITIVE_FLOOR = 1e-6
SPECTRAL_FLOOR = 1e-8
ALISTA_RIDGE_RELATIVE = 1e-4
ALISTA_STEP_INIT = 1.0
ANN_THRESHOLD_INIT = 0.05


@dataclass(frozen=True)
class Protocol:
    # Signal model: x in R^n with exactly s nonzeros, |x_i| ~ U[0, 4), random sign.
    n: int = 256
    max_m: int = 218            # parent Gaussian matrix; the first m rows are used
    matrix_seed: int = 40042
    amplitude_min: float = 0.0
    amplitude_max: float = 4.0
    # Optimisation (online training data, two stages, cosine annealing).
    batch_size: int = 1024
    stage1_steps: int = 5000
    stage2_steps: int = 15000
    cosine_eta_min: float = 1e-7
    stage2_lr_factor: float = 0.25
    validation_interval: int = 250
    # Held-out sets: checkpoint selection / hyper-parameter selection / test.
    validation_samples: int = 1536
    selection_samples: int = 6000
    test_samples: int = 10000
    eval_batch_size: int = 1000
    # Seeds.  Matrix, validation, selection and test sets are shared by all
    # training seeds; model initialisation and training data change per seed.
    model_seed: int = 1616000
    train_seed: int = 20260801
    stage2_seed: int = 20260811
    calibration_seed: int = 20260815
    validation_seed: int = 20260822
    selection_seed: int = 20260823
    test_seed: int = 20260827
    noise_seed: int = 20260828
    # Analytical energy per operation (uJ).
    e_mac_uj: float = 4.6e-6
    e_ac_uj: float = 0.9e-6


TRAINING_SEEDS = (42, 43, 44, 45, 46)

# ---------------------------------------------------------------- conditions
# Fixed M = 141 (55.1 %) Gaussian measurements; s nonzeros out of n = 256.
CONDITIONS = (
    {"id": "m141_s14", "m": 141, "s": 14},
    {"id": "m141_s28", "m": 141, "s": 28},
    {"id": "m141_s42", "m": 141, "s": 42},
    {"id": "m141_s56", "m": 141, "s": 56},
)
ALL_CONDITIONS = CONDITIONS
METHODS = ("lista", "alista", "lamp", "slista")

# Fig. 2a,b: every method at the same unfolded depths, s = 28.
DEPTH_SWEEP_CONDITION = "m141_s28"
DEPTH_SWEEP_DEPTHS = (2, 4, 6, 8, 10)
# Fig. 2c: every method at L = 10 for every sparsity.
SPARSITY_SWEEP_DEPTH = 10
# Fig. 2e,f: S-LISTA and ANN LISTA at s = 14, L = 6 (five spiking layers).
EXAMPLE_CONDITION_ID = "m141_s14"
EXAMPLE_DEPTH = 6

# ------------------------------------------------------------- S-LISTA setup
# S-LISTA is invariant to scaling (P, G, theta) by c and eta by 1/c, so every
# setting is a dimensionless ratio, identical for all depths and sparsities.
#   sigma: std of the initial current (P y)_i off the true support (one batch)
#   theta_1 = kappa * sigma, theta_{L-1} = beta * sigma, geometric in between
#   eta (readout gain) initialised to 1 / theta_{L-1}
#   surrogate half-width of layer l = alpha * theta_l
# Thresholds and eta are trained.
THRESHOLD_KAPPA = "universal"     # sqrt(2 ln n); or a positive float
THRESHOLD_END_RATIO = 2.0         # beta
SURROGATE_RELATIVE_WIDTH = 0.4    # alpha
THRESHOLD_FLOOR = 1e-2
SPIKE_PENALTY = 0.0               # lambda_s

# -------------------------------------------------------------- optimisation
GRAD_CLIP = 1.0

# ------------------------------------------------------------ figure settings
SNR_CONDITION_ID = "m141_s28"                                   # Fig. 2d
SNR_MODEL_IDS = ("lista_l10", "alista_l10", "lamp_l10", "slista_l10")
SNR_DB_VALUES = (0, 5, 10, 15, 20, 30, 40, None)                # None = noise-free

EXAMPLE_SLISTA_ID = f"slista_l{EXAMPLE_DEPTH}"                 # Fig. 2e, f
EXAMPLE_ANN_ID = f"lista_l{EXAMPLE_DEPTH}"
EXAMPLE_SELECTION_SEED = 20260829
EXAMPLE_CANDIDATES = 6000
EXAMPLE_RULE = "median"                # "median" or "best" final NMSE
DISTRIBUTION_SEED = 20260830
DISTRIBUTION_SAMPLES = 2000

SLCA_CONDITION_ID = "m141_s28"                                      # Supp. Fig. 4
SLCA_SLISTA_ID = "slista_l10"
SLCA_SIGNAL_SEEDS = (42, 43, 44, 45, 46)
SLCA_PENALTY = 0.1
SLCA_DT = 0.01
SLCA_HORIZONS = (100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000)


def architecture(method, depth):
    return {"id": f"{method}_l{depth}", "method": method, "depth": depth}


def experiment_grid():
    """All (condition, architecture) pairs needed by the paper's figures."""
    pairs = {}
    sweep = condition_by_id(DEPTH_SWEEP_CONDITION)
    for method in METHODS:
        for depth in DEPTH_SWEEP_DEPTHS:                      # Fig. 2a, b, d
            pairs[(sweep["id"], f"{method}_l{depth}")] = (sweep, architecture(method, depth))
        for condition in CONDITIONS:                          # Fig. 2c, Supp. Figs.
            arch = architecture(method, SPARSITY_SWEEP_DEPTH)
            pairs[(condition["id"], arch["id"])] = (condition, arch)
    example = condition_by_id(EXAMPLE_CONDITION_ID)
    for method in ("slista", "lista"):                        # Fig. 2e, f
        arch = architecture(method, EXAMPLE_DEPTH)
        pairs[(example["id"], arch["id"])] = (example, arch)
    return list(pairs.values())


def model_parameters(architecture):
    if architecture["method"] == "slista":
        return {"threshold_kappa": THRESHOLD_KAPPA,
                "threshold_end_ratio": THRESHOLD_END_RATIO,
                "surrogate_relative_width": SURROGATE_RELATIVE_WIDTH,
                "threshold_floor": THRESHOLD_FLOOR,
                "spike_penalty": SPIKE_PENALTY}
    return {"threshold_scale": 1.0}


def condition_by_id(condition_id):
    for row in ALL_CONDITIONS:
        if row["id"] == condition_id:
            return row
    raise KeyError(condition_id)
