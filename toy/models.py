"""S-LISTA and ANN unfolded sparse-recovery baselines."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    ALISTA_RIDGE_RELATIVE, ALISTA_STEP_INIT, ANN_THRESHOLD_INIT,
    POSITIVE_FLOOR, SPECTRAL_FLOOR,
)


def soft_threshold(x, threshold):
    return torch.sign(x) * F.relu(torch.abs(x) - threshold)


def inverse_softplus(value):
    return math.log(math.expm1(float(value)))


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold, width):
        ctx.save_for_backward(x, threshold, width)
        return (x >= threshold).to(x.dtype)

    @staticmethod
    def backward(ctx, grad):
        x, threshold, width = ctx.saved_tensors
        return grad * ((x - threshold).abs() < width).to(x.dtype), None, None


class SpikingLISTA(nn.Module):
    def __init__(self, matrix, depth, time_steps, theta_snn, decoder_tau,
                 surrogate_width, **_):
        super().__init__()
        self.m, self.n = matrix.shape
        self.depth, self.time_steps = int(depth), int(time_steps)
        self.collect_stats = True
        self.P = nn.Linear(self.m, self.n, bias=False)
        self.G = nn.ModuleList(nn.Linear(self.n, self.n, bias=False)
                               for _ in range(max(0, self.depth - 1)))
        self.register_buffer("theta_snn", matrix.new_tensor(theta_snn))
        self.register_buffer("decoder_tau", matrix.new_tensor(decoder_tau))
        self.register_buffer("surrogate_width", matrix.new_tensor(surrogate_width))
        with torch.no_grad():
            # Fan-in uniform initialization for the learned operators.
            nn.init.kaiming_uniform_(self.P.weight, a=math.sqrt(5))
            for layer in self.G:
                nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))

    def forward(self, measurements):
        batch = measurements.shape[0]
        initial = self.P(measurements)
        states = [torch.zeros_like(initial) for _ in range(self.depth)]
        cumulative = torch.zeros_like(initial)
        outputs, total_ac = [], initial.new_zeros(())
        total_spikes = initial.new_zeros(())
        for time_index in range(self.time_steps):
            code = torch.zeros_like(initial)
            next_states = []
            for layer_index in range(self.depth):
                if layer_index == 0:
                    drive = initial
                else:
                    if self.collect_stats:
                        # Integer magnitude n requires n additions per connection.
                        total_ac += code.detach().abs().sum() * self.n
                    drive = initial - self.G[layer_index - 1](code)
                membrane = self.decoder_tau * states[layer_index] + drive
                positive = SurrogateSpike.apply(
                    membrane, self.theta_snn, self.surrogate_width)
                negative = SurrogateSpike.apply(
                    -membrane, self.theta_snn, self.surrogate_width)
                spike = positive - negative
                code = code + spike
                next_states.append(membrane - self.theta_snn * spike)
                if self.collect_stats:
                    total_spikes += positive.detach().sum() + negative.detach().sum()
            states = next_states
            cumulative += code
            elapsed = time_index + 1
            # Signed spikes alone select the support.  The last residual
            # membrane is a fractional threshold crossing and may refine only
            # an already selected coefficient.  Division by elapsed preserves
            # the rate-coded interpretation for T > 1; at T = 1 this
            # reduces exactly to z + mask * u/theta.
            support = (cumulative != 0).to(cumulative.dtype)
            outputs.append(
                (
                    cumulative
                    + support
                    * (states[-1] / self.theta_snn.clamp_min(POSITIVE_FLOOR))
                )
                / elapsed
            )
        self.last_spike_count = total_spikes
        self.last_ac_count = total_ac
        self.last_firing_rate = total_spikes / (
            2 * batch * self.n * self.depth * self.time_steps
        )
        return outputs[-1], outputs


class LISTA(nn.Module):
    def __init__(self, matrix, depth, threshold_scale, **_):
        super().__init__(); self.m, self.n = matrix.shape; self.depth = depth
        lipschitz = torch.linalg.svdvals(matrix)[0].square().clamp_min(
            SPECTRAL_FLOOR)
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
            for layer in self.S: layer.weight.copy_(recurrent)

    def forward(self, y):
        encoded, code, outputs = self.W(y), y.new_zeros(y.shape[0], self.n), []
        for index in range(self.depth):
            pre = encoded if index == 0 else encoded + self.S[index - 1](code)
            code = soft_threshold(
                pre, F.softplus(self.raw_theta[index]) + POSITIVE_FLOOR)
            outputs.append(code)
        return code, outputs

    def macs_per_sample(self):
        return self.m * self.n + max(0, self.depth - 1) * self.n * self.n


class ALISTA(nn.Module):
    def __init__(self, matrix, depth, threshold_scale, **_):
        super().__init__(); self.m, self.n = matrix.shape; self.depth = depth
        gram = matrix @ matrix.t()
        mean_eigenvalue = torch.trace(gram) / self.m
        ridge = ALISTA_RIDGE_RELATIVE * mean_eigenvalue
        solved = torch.linalg.solve(
            gram + ridge * torch.eye(
                self.m, device=matrix.device, dtype=matrix.dtype),
            matrix)
        diagonal = (matrix * solved).sum(dim=0).clamp_min(POSITIVE_FLOOR)
        self.register_buffer("A", matrix.detach().clone())
        self.register_buffer("W", (solved / diagonal.unsqueeze(0)).t())
        self.raw_gamma = nn.ParameterList(
            nn.Parameter(torch.tensor(inverse_softplus(ALISTA_STEP_INIT)))
            for _ in range(depth))
        self.raw_theta = nn.ParameterList(
            nn.Parameter(torch.tensor(inverse_softplus(
                ANN_THRESHOLD_INIT * threshold_scale)))
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
        super().__init__(); self.m, self.n = matrix.shape; self.depth = depth
        self.register_buffer("A", matrix.detach().clone())
        self.B = nn.ModuleList(nn.Linear(self.m, self.n, bias=False)
                               for _ in range(depth))
        self.raw_alpha = nn.ParameterList(
            nn.Parameter(torch.tensor(inverse_softplus(
                ANN_THRESHOLD_INIT * threshold_scale)))
            for _ in range(depth))
        with torch.no_grad():
            for layer in self.B: layer.weight.copy_(matrix.t())

    def forward(self, y):
        code, residual, outputs = y.new_zeros(y.shape[0], self.n), y, []
        for index in range(self.depth):
            pre = code + self.B[index](residual)
            noise = residual.norm(dim=1, keepdim=True) / math.sqrt(self.m)
            threshold = (F.softplus(self.raw_alpha[index])
                         + POSITIVE_FLOOR) * noise
            code = soft_threshold(pre, threshold); outputs.append(code)
            if index + 1 < self.depth:
                active = (pre.abs() > threshold).float().sum(dim=1, keepdim=True)
                residual = y - code @ self.A.t() + (active / self.m) * residual
        return code, outputs

    def macs_per_sample(self):
        return (2 * self.depth - 1) * self.m * self.n


def build_model(method, matrix, depth, time_steps, parameters):
    if method == "slista":
        return SpikingLISTA(matrix, depth, time_steps, **parameters)
    return {"lista": LISTA, "alista": ALISTA, "lamp": LAMP}[method](
        matrix, depth, **parameters)
