"""Public protocol for the fixed-measurement noiseless CS experiment."""

from dataclasses import dataclass


# Named numerical and initialization constants used by every method.
NUMERICAL_EPS = 1e-12
POSITIVE_FLOOR = 1e-6
SPECTRAL_FLOOR = 1e-8
ALISTA_RIDGE_RELATIVE = 1e-4
ALISTA_STEP_INIT = 1.0
ANN_THRESHOLD_INIT = 0.05


@dataclass(frozen=True)
class Protocol:
    n: int = 256
    max_m: int = 218
    matrix_seed: int = 40042
    amplitude_min: float = 0.0
    amplitude_max: float = 4.0
    batch_size: int = 1024
    stage1_steps: int = 5000
    stage2_steps: int = 15000
    validation_interval: int = 250
    validation_samples: int = 1536
    selection_samples: int = 6000
    test_samples: int = 10000
    eval_batch_size: int = 1000
    stage2_sources: int = 1
    stage2_lr_factors: tuple = (0.25,)
    cosine_eta_min: float = 1e-7
    max_boundary_expansions: int = 3
    model_seed: int = 1616000
    train_seed: int = 20260801
    stage2_seed: int = 20260811
    validation_seed: int = 20260822
    selection_seed: int = 20260823
    test_seed: int = 20260827
    noise_seed: int = 20260828
    e_mac_uj: float = 4.6e-6
    e_ac_uj: float = 0.9e-6


# Fixed measurement dimension with s/M approximately 0.2, 0.3 and 0.4.
# max_m specifies the parent Gaussian matrix before row selection.
CONDITIONS = (
    {"id": "m141_s28",  "m": 141, "s": 28},
    {"id": "m141_s42",  "m": 141, "s": 42},
    {"id": "m141_s14",  "m": 141, "s": 14},
)

SNR_DB_VALUES = (0, 5, 10, 15, 20, 30, 40, None)
SNR_CONDITION_ID = "m141_s28"
SNR_MODEL_IDS = (
    "lista_l5", "alista_l5", "lamp_l5",
    "slista_l20_t1",
)

ANN_DEPTHS = (2, 3, 4, 5)
SNN_DEPTHS = (5, 10, 15, 20)
SNN_TIME_STEPS = (1,)

LEARNING_RATE_LADDER = (
    1e-6, 2e-6, 5e-6,
    1e-5, 2e-5, 5e-5,
    1e-4, 2e-4, 3e-4, 5e-4,
    1e-3, 2e-3, 3e-3, 5e-3,
    1e-2,
)
INITIAL_LEARNING_RATES = (
    1e-4, 2e-4, 3e-4, 5e-4,
    1e-3, 2e-3, 3e-3, 5e-3,
)

# These neuron settings are fixed across all three CS conditions. T=1 ignores
# tau because there is no temporal transition.
SNN_PARAMETERS = {
    5:  {"theta_snn": 1.83, "decoder_tau": 0.80,
         "surrogate_width": 0.31},
    10: {"theta_snn": 1.20, "decoder_tau": 0.95,
         "surrogate_width": 0.20},
    15: {"theta_snn": 1.35, "decoder_tau": 1.00,
         "surrogate_width": 0.19},
    20: {"theta_snn": 1.28, "decoder_tau": 0.98,
         "surrogate_width": 0.35},
    25: {"theta_snn": 1.28, "decoder_tau": 0.98,
         "surrogate_width": 0.35},
    30: {"theta_snn": 1.28, "decoder_tau": 0.98,
         "surrogate_width": 0.35},
}


def architectures():
    rows = []
    for method in ("lista", "alista", "lamp"):
        for depth in ANN_DEPTHS:
            rows.append({"id": f"{method}_l{depth}", "method": method,
                         "depth": depth, "time_steps": 1})
    for depth in SNN_DEPTHS:
        for time_steps in SNN_TIME_STEPS:
            rows.append({"id": f"slista_l{depth}_t{time_steps}",
                         "method": "slista", "depth": depth,
                         "time_steps": time_steps})
    return rows


def model_parameters(architecture, learning_rate):
    common = {"learning_rate": float(learning_rate),
              "aux_weight": 0.0, "grad_clip": 1.0}
    if architecture["method"] == "slista":
        return {**SNN_PARAMETERS[architecture["depth"]], **common}
    return {"threshold_scale": 1.0, **common}


VISUAL_CONDITION = {"id": "m141_s14", "m": 141, "s": 14}
ALL_CONDITIONS = CONDITIONS
