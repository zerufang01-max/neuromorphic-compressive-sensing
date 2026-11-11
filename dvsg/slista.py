"""S-LISTA for DVS-Gesture (convolutional, multi-frame).

Per frame t (paper Eqs. (5)-(7)):

    I^(1) = P * y_t,            z^(0) = 0
    for l = 1 .. L-1   (spiking layers, ternary spikes, soft reset)
        u_bar  = rho * u^(l)_{t-1} + I^(l)
        xi     = H(u_bar - theta_l) - H(-u_bar - theta_l)          in {-1, 0, 1}
        u^(l)_t = u_bar - theta_l * xi        (xi not differentiated in the reset)
        z^(l)  = z^(l-1) + xi
        I^(l+1) = I^(1) - G^(l) * z^(l)
    readout layer L (non-spiking, no memory across frames)
        m = 1[z^(L-1) != 0]         (event gating, per frame)
        z = z^(L-1) + eta * m * I^(L)
    x_hat_t = tail(z)               (learned dictionary: TConv128 -> TConv64 -> Conv2)

P is a 3x3 convolution C -> 128 and G^(l) are 3x3 convolutions 128 -> 128 on
the 32 x 32 grid.  theta_l = floor + softplus(r_l) (one scalar per layer) and
eta = softplus(r_eta) are trained; the boxcar surrogate has half-width
alpha * theta_l and also passes gradient to theta_l.  Initial thresholds are
kappa*sigma .. beta*sigma (geometric), eta = 1 / theta_{L-1}, where sigma is
the standard deviation of P * y over all coordinates of a calibration batch
(frames with at least one received spike).  ``forward`` is the only
implementation used for training, evaluation and operation counting.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C
from baseline_models import LatentVideoClassifier


def inverse_softplus(value):
    value = float(value)
    return value + math.log(-math.expm1(-value))


class BoxcarSpike(torch.autograd.Function):
    """H(x - theta) with a boxcar surrogate of half-width ``width``; d/dtheta = -d/dx."""

    @staticmethod
    def forward(ctx, x, theta, width):
        ctx.save_for_backward(x, theta, width)
        return (x >= theta).to(x.dtype)

    @staticmethod
    def backward(ctx, grad):
        x, theta, width = ctx.saved_tensors
        surrogate = grad * ((x - theta).abs() < width).to(grad.dtype)
        return surrogate, -surrogate.sum().reshape(theta.shape), None


class SpikingLISTA(nn.Module):
    def __init__(self, meas_channels=4, code_channels=C.Config.SPARSE_DIM, depth=C.Config.NUM_LAYERS,
                 kappa=2.5, beta=2.0, alpha=0.3, rho=0.0, lambda_s=0.01, threshold_floor=0.01,
                 grid=None, **_):
        super().__init__()
        if depth < 2:
            raise ValueError("S-LISTA needs at least one spiking layer and the readout layer")
        self.meas_channels = int(meas_channels)
        self.code_channels = int(code_channels)
        self.grid = int(grid or C.Config.IMG_H // C.Config.MEAS_STRIDE)
        self.depth = int(depth)
        self.spiking_layers = self.depth - 1
        self.kappa, self.beta, self.alpha = float(kappa), float(beta), float(alpha)
        self.rho = float(rho)
        self.lambda_s = float(lambda_s)
        self.code_size = self.code_channels * self.grid * self.grid
        self.register_buffer("threshold_floor", torch.tensor(float(threshold_floor)))
        self.register_buffer("calibrated", torch.tensor(False))

        self.P = nn.Conv2d(self.meas_channels, self.code_channels, 3, 1, 1, bias=False)
        self.G = nn.ModuleList(nn.Conv2d(self.code_channels, self.code_channels, 3, 1, 1, bias=False)
                               for _ in range(self.spiking_layers))
        self.raw_level = nn.Parameter(torch.zeros(self.spiking_layers))   # set by calibrate()
        self.raw_eta = nn.Parameter(torch.zeros(()))
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.P.weight, a=math.sqrt(5))
            for layer in self.G:
                nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))

        # Dictionary readout and task head as in ANN LISTA (ConvolutionalLISTA).
        self.tail = nn.Sequential(
            nn.ConvTranspose2d(self.code_channels, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, C.Config.INPUT_CHANNELS, 3, 1, 1),
        )
        nn.init.orthogonal_(self.tail[-1].weight, gain=0.5)
        nn.init.constant_(self.tail[-1].bias, 0)
        self.semantic_head = LatentVideoClassifier(self.code_channels, C.Config.NUM_CLASSES,
                                                   C.Config.TIME_STEPS)

    # ----------------------------------------------------------- parameters
    @property
    def theta(self):
        return self.threshold_floor + F.softplus(self.raw_level)

    @property
    def eta(self):
        return F.softplus(self.raw_eta)

    @torch.no_grad()
    def set_initial_thresholds(self, sigma):
        first, last = self.kappa * sigma, self.beta * sigma
        if self.spiking_layers == 1:
            values = [first]
        else:
            ratio = (last / first) ** (1.0 / (self.spiking_layers - 1))
            values = [first * ratio ** k for k in range(self.spiking_layers)]
        floor = float(self.threshold_floor)
        if min(values) <= floor:
            raise ValueError(f"initial thresholds {values} are below the floor {floor}")
        self.raw_level.copy_(torch.tensor([inverse_softplus(v - floor) for v in values]))
        self.raw_eta.copy_(torch.tensor(inverse_softplus(1.0 / values[-1])))
        self.calibrated.fill_(True)
        return {"sigma": float(sigma), "kappa": self.kappa, "beta": self.beta,
                "theta_init": values, "eta_init": 1.0 / values[-1]}

    def parameter_groups(self):
        """Names -> parameters for the optimizer (see trainer.slista_optimizer)."""
        groups = {"P": [self.P.weight], "eta": [self.raw_eta], "G": [], "theta": [self.raw_level],
                  "dict": list(self.tail.parameters()), "semantic": list(self.semantic_head.parameters())}
        for layer in self.G:
            groups["G"].append(layer.weight)
        return groups

    # -------------------------------------------------------------- forward
    def forward(self, y_flat, states=None):
        """One frame.  Returns (x_hat, z, next_states, trace).

        states: list of the L-1 membrane potentials of the previous frame (rho > 0),
        or None.  trace holds the spikes, integer codes, mask and inputs of this frame.
        """
        batch = y_flat.shape[0]
        y = y_flat.view(batch, self.meas_channels, self.grid, self.grid)
        theta = self.theta
        width = self.alpha * theta.detach()
        initial = self.P(y)
        current = initial
        code = torch.zeros_like(initial)
        codes, spikes, next_states = [], [], []
        for layer in range(self.spiking_layers):
            if self.rho != 0.0 and states is not None:
                u_bar = self.rho * states[layer] + current
            else:
                u_bar = current
            spike = (BoxcarSpike.apply(u_bar, theta[layer], width[layer])
                     - BoxcarSpike.apply(-u_bar, theta[layer], width[layer]))
            if self.rho != 0.0:
                next_states.append(u_bar - theta[layer] * spike.detach())
            code = code + spike
            spikes.append(spike)
            codes.append(code)
            current = initial - self.G[layer](code)
        mask = (code != 0).to(code.dtype)
        z = code + self.eta * mask * current
        trace = {"input": y, "spikes": spikes, "codes": codes, "mask": mask, "spike_code": code}
        return self.tail(z), z, (next_states if self.rho != 0.0 else None), trace

    # ------------------------------------------------- loss and op counting
    def sparsity_penalty(self, trace):
        """lambda_s * mean_l ||z^(l)||_1 / n, averaged over the batch (one frame)."""
        if self.lambda_s == 0.0:
            return trace["spike_code"].new_zeros(())
        l1 = torch.stack([code.abs().sum(dim=(1, 2, 3)) for code in trace["codes"]]).mean(0)
        return self.lambda_s * (l1 / self.code_size).mean()

    @torch.no_grad()
    def frame_counts(self, trace):
        """Batch-summed counts for one frame (tensors; no host synchronisation).

        AC : P * y (binary received bits) and G^(l) * z^(l) for l = 1..L-1.  A 3x3
             convolution with K output channels costs 9 K additions per unit of
             |input| (an integer magnitude n costs n additions per connection).
        MAC: readout gain eta * I^(L) on the gated support (||m||_0).
        """
        fan_out = 9 * self.code_channels
        ac = fan_out * (trace["input"].abs().sum() + sum(code.abs().sum() for code in trace["codes"]))
        layer_spikes = torch.stack([s.abs().sum() for s in trace["spikes"]])
        return {"ac": ac, "mac": trace["mask"].sum(), "layer_spikes": layer_spikes,
                "support": trace["mask"].sum()}

    def firing_denominator(self, frames_times_batch):
        """Firing rate: signed spikes on two channels, L-1 spiking layers."""
        return 2.0 * frames_times_batch * self.code_size * self.spiking_layers

    def settings(self):
        return {"meas_channels": self.meas_channels, "code_channels": self.code_channels,
                "depth": self.depth, "kappa": self.kappa, "beta": self.beta, "alpha": self.alpha,
                "rho": self.rho, "lambda_s": self.lambda_s,
                "threshold_floor": float(self.threshold_floor)}
