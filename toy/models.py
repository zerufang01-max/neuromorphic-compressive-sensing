"""S-LISTA and the ANN unfolded baselines (LISTA, ALISTA, LAMP).

S-LISTA (single frame, as in the synthetic experiments; paper Eqs. (5)-(7)):

    I^(1) = P y,                    z^(0) = 0
    for l = 1 .. L-1  (spiking layers)
        u_bar^(l) = I^(l)                                   (rho is inactive for T = 1)
        xi^(l)    = H(u_bar - theta_l) - H(-u_bar - theta_l)  in {-1, 0, 1}
        z^(l)     = z^(l-1) + xi^(l)
        I^(l+1)   = I^(1) - G^(l) z^(l)
    readout layer L (non-spiking, memoryless)
        v         = I^(L)
        m         = 1[z^(L-1) != 0]                         (event gating)
        z         = z^(L-1) + eta * m * v

theta_l > 0 and eta > 0 are trained.  The surrogate gradient of layer l uses a
boxcar of half-width alpha * theta_l.  ``SpikingLISTA.trace`` is the single
implementation used for training, evaluation, figures and theory audits.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    ALISTA_RIDGE_RELATIVE, ALISTA_STEP_INIT, ANN_THRESHOLD_INIT,
    NUMERICAL_EPS, POSITIVE_FLOOR, SPECTRAL_FLOOR,
)


def soft_threshold(x, threshold):
    return torch.sign(x) * F.relu(torch.abs(x) - threshold)


def inverse_softplus(value):
    value = float(value)
    return value + math.log(-math.expm1(-value))


class SurrogateSpike(torch.autograd.Function):
    """Heaviside H(x - theta) with a boxcar surrogate of half-width w.

    The surrogate derivative is passed to both x and theta (d/dtheta = -d/dx),
    so the thresholds are trainable."""

    @staticmethod
    def forward(ctx, x, theta, width):
        ctx.save_for_backward(x, theta, width)
        return (x >= theta).to(x.dtype)

    @staticmethod
    def backward(ctx, grad):
        x, theta, width = ctx.saved_tensors
        surrogate = grad * ((x - theta).abs() < width).to(x.dtype)
        return surrogate, -surrogate.sum().reshape(theta.shape), None


class SpikingLISTA(nn.Module):
    def __init__(self, matrix, depth, threshold_kappa="universal", threshold_end_ratio=0.5,
                 surrogate_relative_width=0.2, threshold_floor=1e-2, spike_penalty=0.0, **_):
        super().__init__()
        if depth < 2:
            raise ValueError("S-LISTA needs at least one spiking layer and the readout layer")
        self.m, self.n = matrix.shape
        self.depth = int(depth)
        self.spiking_layers = self.depth - 1
        self.kappa = (math.sqrt(2.0 * math.log(self.n))
                      if threshold_kappa == "universal" else float(threshold_kappa))
        self.beta = float(threshold_end_ratio)
        self.alpha = float(surrogate_relative_width)
        self.spike_penalty_weight = float(spike_penalty)
        self.register_buffer("threshold_floor", matrix.new_tensor(float(threshold_floor)))
        self.P = nn.Linear(self.m, self.n, bias=False)
        self.G = nn.ModuleList(nn.Linear(self.n, self.n, bias=False)
                               for _ in range(self.depth - 1))
        # Placeholders; calibrate() sets the actual initial values.
        self.raw_theta = nn.Parameter(torch.zeros(self.spiking_layers))
        self.raw_eta = nn.Parameter(torch.zeros(()))
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.P.weight, a=math.sqrt(5))
            for layer in self.G:
                nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))

    # ------------------------------------------------------------ parameters
    @property
    def theta(self):
        return self.threshold_floor + F.softplus(self.raw_theta)

    @property
    def eta(self):
        return F.softplus(self.raw_eta)

    @torch.no_grad()
    def calibrate(self, target, measurements):
        """Initial thresholds theta_1 = kappa * sigma and theta_{L-1} = beta * sigma,
        geometric in between (a single spiking layer uses kappa * sigma); eta is
        initialised to 1 / theta_{L-1}.  sigma is the standard deviation of
        (P y)_i over coordinates outside the true support (one training batch)."""
        current = self.P(measurements)
        sigma = float(current[target == 0].std())
        first, last = self.kappa * sigma, self.beta * sigma
        if self.spiking_layers == 1:
            values = [first]
        else:
            ratio = (last / first) ** (1.0 / (self.spiking_layers - 1))
            values = [first * ratio ** k for k in range(self.spiking_layers)]
        floor = float(self.threshold_floor)
        self.raw_theta.copy_(torch.tensor([inverse_softplus(v - floor) for v in values]))
        self.raw_eta.copy_(torch.tensor(inverse_softplus(1.0 / values[-1])))
        return {"sigma": sigma, "kappa": self.kappa, "beta": self.beta,
                "theta_init": values, "eta_init": 1.0 / values[-1]}

    # --------------------------------------------------------------- forward
    def trace(self, measurements):
        """Run S-LISTA and return every intermediate quantity.

        spikes[l] = xi^(l+1), codes[l] = z^(l+1) for the L-1 spiking layers;
        currents[l] = I^(l+1) for all L layers; v = I^(L); mask; output."""
        theta = self.theta
        width = self.alpha * theta.detach()
        initial = self.P(measurements)
        code = torch.zeros_like(initial)
        spikes, codes, currents = [], [], []
        current = initial
        for layer in range(self.spiking_layers):
            currents.append(current)
            positive = SurrogateSpike.apply(current, theta[layer], width[layer])
            negative = SurrogateSpike.apply(-current, theta[layer], width[layer])
            spike = positive - negative
            code = code + spike
            spikes.append(spike)
            codes.append(code)
            current = initial - self.G[layer](code)
        currents.append(current)
        mask = (code != 0).to(code.dtype)
        output = code + self.eta * mask * current
        return {"spikes": spikes, "codes": codes, "currents": currents,
                "v": current, "mask": mask, "spike_code": code,
                "correction": output - code, "output": output}

    def forward(self, measurements):
        trace = self.trace(measurements)
        self.last_trace = trace
        return trace["output"], trace["codes"]

    # -------------------------------------------------- loss and op counting
    def spike_penalty(self, trace):
        """lambda_s / (L-1) * sum_l ||z^(l)||_1 / n, averaged over the batch.
        sum_l ||z^(l)||_1 * n is exactly the synaptic AC count, so the penalty
        is proportional to the estimated reconstruction energy."""
        if self.spike_penalty_weight == 0.0:
            return trace["output"].new_zeros(())
        # mean over the L-1 layers of ||z^(l)||_1, then / n, then batch mean
        l1 = torch.stack([code.abs().sum(1) for code in trace["codes"]]).mean(0)
        return self.spike_penalty_weight * (l1 / self.n).mean()

    @staticmethod
    @torch.no_grad()
    def operation_counts(trace, n, m):
        """Per-sample operation counts (summed over the batch).

        MAC: P y (m n, dense real input) + readout gain eta * v on the gated
             support (||m||_0).
        AC : G^(l) z^(l) for l = 1 .. L-1; an integer magnitude k costs k
             additions per connection, i.e. n * ||z^(l)||_1.
        The dictionary readout (identity here) is excluded, as for the ANNs."""
        batch = trace["output"].shape[0]
        ac = sum(float(code.abs().sum()) for code in trace["codes"]) * n
        mac = batch * m * n + float(trace["mask"].sum())
        spikes = sum(float(spike.abs().sum()) for spike in trace["spikes"])
        layers = len(trace["spikes"])
        return {"mac": mac, "ac": ac, "spikes": spikes,
                "support": float(trace["mask"].sum()),
                # Signed spikes counted on two channels.
                "firing_rate_numerator": spikes,
                "firing_rate_denominator": 2.0 * batch * n * layers}


class LISTA(nn.Module):
    def __init__(self, matrix, depth, threshold_scale, **_):
        super().__init__()
        self.m, self.n = matrix.shape
        self.depth = depth
        lipschitz = torch.linalg.svdvals(matrix)[0].square().clamp_min(SPECTRAL_FLOOR)
        step = 1.0 / lipschitz
        threshold = threshold_scale * ANN_THRESHOLD_INIT / float(lipschitz)
        self.W = nn.Linear(self.m, self.n, bias=False)
        self.S = nn.ModuleList(nn.Linear(self.n, self.n, bias=False)
                               for _ in range(max(0, depth - 1)))
        self.raw_theta = nn.ParameterList(
            nn.Parameter(torch.tensor(inverse_softplus(threshold)))
            for _ in range(depth))
        with torch.no_grad():
            self.W.weight.copy_(step * matrix.t())
            recurrent = torch.eye(self.n, device=matrix.device) - step * matrix.t() @ matrix
            for layer in self.S:
                layer.weight.copy_(recurrent)

    def forward(self, y):
        encoded = self.W(y)
        code, outputs = y.new_zeros(y.shape[0], self.n), []
        for index in range(self.depth):
            pre = encoded if index == 0 else encoded + self.S[index - 1](code)
            code = soft_threshold(pre, F.softplus(self.raw_theta[index]) + POSITIVE_FLOOR)
            outputs.append(code)
        return code, outputs

    def macs_per_sample(self):
        return self.m * self.n + max(0, self.depth - 1) * self.n * self.n


class ALISTA(nn.Module):
    def __init__(self, matrix, depth, threshold_scale, **_):
        super().__init__()
        self.m, self.n = matrix.shape
        self.depth = depth
        gram = matrix @ matrix.t()
        ridge = ALISTA_RIDGE_RELATIVE * torch.trace(gram) / self.m
        solved = torch.linalg.solve(
            gram + ridge * torch.eye(self.m, device=matrix.device, dtype=matrix.dtype),
            matrix)
        diagonal = (matrix * solved).sum(dim=0).clamp_min(POSITIVE_FLOOR)
        self.register_buffer("A", matrix.detach().clone())
        self.register_buffer("W", (solved / diagonal.unsqueeze(0)).t())
        self.raw_gamma = nn.ParameterList(
            nn.Parameter(torch.tensor(inverse_softplus(ALISTA_STEP_INIT)))
            for _ in range(depth))
        self.raw_theta = nn.ParameterList(
            nn.Parameter(torch.tensor(inverse_softplus(ANN_THRESHOLD_INIT * threshold_scale)))
            for _ in range(depth))

    def forward(self, y):
        code, outputs = y.new_zeros(y.shape[0], self.n), []
        for index in range(self.depth):
            residual = y - code @ self.A.t()
            code = soft_threshold(
                code + F.softplus(self.raw_gamma[index]) * (residual @ self.W.t()),
                F.softplus(self.raw_theta[index]) + POSITIVE_FLOOR)
            outputs.append(code)
        return code, outputs

    def macs_per_sample(self):
        return (2 * self.depth - 1) * self.m * self.n


class LAMP(nn.Module):
    def __init__(self, matrix, depth, threshold_scale, **_):
        super().__init__()
        self.m, self.n = matrix.shape
        self.depth = depth
        self.register_buffer("A", matrix.detach().clone())
        self.B = nn.ModuleList(nn.Linear(self.m, self.n, bias=False) for _ in range(depth))
        self.raw_alpha = nn.ParameterList(
            nn.Parameter(torch.tensor(inverse_softplus(ANN_THRESHOLD_INIT * threshold_scale)))
            for _ in range(depth))
        with torch.no_grad():
            for layer in self.B:
                layer.weight.copy_(matrix.t())

    def forward(self, y):
        code, residual, outputs = y.new_zeros(y.shape[0], self.n), y, []
        for index in range(self.depth):
            pre = code + self.B[index](residual)
            noise = residual.norm(dim=1, keepdim=True) / math.sqrt(self.m)
            threshold = (F.softplus(self.raw_alpha[index]) + POSITIVE_FLOOR) * noise
            code = soft_threshold(pre, threshold)
            outputs.append(code)
            if index + 1 < self.depth:
                active = (pre.abs() > threshold).float().sum(dim=1, keepdim=True)
                residual = y - code @ self.A.t() + (active / self.m) * residual
        return code, outputs

    def macs_per_sample(self):
        return (2 * self.depth - 1) * self.m * self.n


def build_model(method, matrix, depth, parameters):
    classes = {"slista": SpikingLISTA, "lista": LISTA, "alista": ALISTA, "lamp": LAMP}
    return classes[method](matrix, depth, **parameters)


def nmse_loss(estimate, target):
    """Training loss: mean over samples of ||x_hat - x||^2 / ||x||^2."""
    return ((estimate - target).square().sum(1)
            / target.square().sum(1).clamp_min(NUMERICAL_EPS)).mean()
